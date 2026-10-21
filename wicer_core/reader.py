"""llama-server client -- the reader under test.

Every condition (full-context, RAG, compiled wiki) goes through this one class
so the chat template is byte-identical across conditions.  That matters: the
paper's comparison is only meaningful if the sole difference between arms is
what goes into the system prompt.

Timing follows paper Section 3.5: TTFT, total response time and generation
throughput are measured from the streaming SSE response.  We record both our
client-side measurement and llama.cpp's own ``timings`` block when the server
sends one, because they answer slightly different questions (wall clock the
user feels vs. server-side compute).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Iterator

import requests

from wicer_core.config import ReaderConfig


@dataclass
class Generation:
    """One reader response with its timing record."""

    text: str
    ttft_s: float
    total_s: float
    n_generated_tokens: int
    throughput_tok_s: float
    n_prompt_tokens: int | None = None
    cache_n: int | None = None           # tokens llama.cpp reused from the KV cache
    server_timings: dict[str, Any] | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReaderError(RuntimeError):
    pass


class ContextOverflow(ReaderError):
    """The context does not fit the server's window -- paper's 2 excluded topics."""

    def __init__(self, n_tokens: int, limit: int):
        self.n_tokens = n_tokens
        self.limit = limit
        super().__init__(f"context is {n_tokens} tokens, limit is {limit}")


class ReaderClient:
    def __init__(self, cfg: ReaderConfig, session: requests.Session | None = None):
        self.cfg = cfg
        self.session = session or requests.Session()

    # -- server introspection -------------------------------------------------

    def health(self, timeout: float = 5.0) -> bool:
        try:
            r = self.session.get(f"{self.cfg.base_url}/health", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def wait_for_health(self, timeout_s: float = 300.0, interval_s: float = 1.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.health():
                return
            time.sleep(interval_s)
        raise ReaderError(f"llama-server at {self.cfg.base_url} not healthy after {timeout_s}s")

    def props(self) -> dict[str, Any]:
        r = self.session.get(f"{self.cfg.base_url}/props", timeout=30)
        r.raise_for_status()
        return r.json()

    def server_n_ctx(self) -> int | None:
        """The context window the server was actually launched with."""
        try:
            props = self.props()
        except requests.RequestException:
            return None
        for key in ("n_ctx", "default_generation_settings"):
            v = props.get(key)
            if isinstance(v, int):
                return v
            if isinstance(v, dict) and isinstance(v.get("n_ctx"), int):
                return v["n_ctx"]
        return None

    # -- tokenisation ---------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        r = self.session.post(
            f"{self.cfg.base_url}/tokenize",
            json={"content": text},
            timeout=self.cfg.request_timeout_s,
        )
        r.raise_for_status()
        return r.json().get("tokens", [])

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def context_limit(self) -> int:
        """Usable prompt budget: the window minus generation and template slack.

        Takes the smaller of the configured and the server-reported window.
        A server launched with a larger window than the profile asks for would
        otherwise let topics through that the configured experiment excludes --
        and the result would silently not be the 96K run it claims to be.
        """
        server = self.server_n_ctx()
        n_ctx = min(server, self.cfg.n_ctx) if server else self.cfg.n_ctx
        return n_ctx - self.cfg.max_tokens - self.cfg.ctx_margin_tokens

    def check_fits(self, context: str) -> int:
        """Return the context's token count, or raise ContextOverflow.

        This is how ``regional_cuisine`` and ``regional_folklore`` drop out of
        the full-context arm (paper Section 5, footnote 2) -- by measurement,
        not by a hardcoded exclusion list.
        """
        n = self.count_tokens(context)
        limit = self.context_limit()
        if n > limit:
            raise ContextOverflow(n, limit)
        return n

    # -- generation -----------------------------------------------------------

    def _payload(self, system: str, user: str, *, max_tokens: int | None) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "top_k": self.cfg.top_k,
            "seed": self.cfg.seed,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            # llama.cpp extensions: reuse the KV prefix across requests, and
            # report per-token server timings inside the stream.
            "cache_prompt": True,
            "timings_per_token": True,
        }

    def chat(self, system: str, user: str, *, max_tokens: int | None = None) -> Generation:
        """Stream one completion and measure TTFT / total / throughput."""
        payload = self._payload(system, user, max_tokens=max_tokens)
        started = time.perf_counter()
        ttft: float | None = None
        pieces: list[str] = []
        usage: dict[str, Any] = {}
        timings: dict[str, Any] | None = None
        finish_reason: str | None = None

        with self.session.post(
            f"{self.cfg.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=self.cfg.request_timeout_s,
        ) as resp:
            if resp.status_code != 200:
                raise ReaderError(f"llama-server returned {resp.status_code}: {resp.text[:500]}")
            for event in _iter_sse(resp):
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        pieces.append(piece)
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("timings"):
                    timings = event["timings"]

        total = time.perf_counter() - started
        if ttft is None:
            # Server produced no content at all; treat the whole call as TTFT so
            # the record stays interpretable rather than silently zero.
            ttft = total

        text = "".join(pieces)
        n_gen = int(usage.get("completion_tokens") or 0)
        if not n_gen and timings:
            n_gen = int(timings.get("predicted_n") or 0)
        if not n_gen:
            n_gen = max(1, len(text.split()))

        decode_s = max(total - ttft, 1e-9)
        throughput = (n_gen - 1) / decode_s if n_gen > 1 else 0.0

        return Generation(
            text=text,
            ttft_s=ttft,
            total_s=total,
            n_generated_tokens=n_gen,
            throughput_tok_s=throughput,
            n_prompt_tokens=int(usage["prompt_tokens"]) if usage.get("prompt_tokens") else None,
            cache_n=int(timings["cache_n"]) if timings and "cache_n" in timings else None,
            server_timings=timings,
            finish_reason=finish_reason,
        )

    def prime(self, system: str) -> Generation:
        """Preflight request that builds the KV cache for a corpus.

        Paper Section 3.4: "A preflight request primes the KV cache; subsequent
        queries reuse cached states and process only the question suffix."
        Every later query must reuse this exact system string verbatim, and
        queries must run sequentially -- concurrent requests land in different
        server slots and defeat prefix reuse.
        """
        return self.chat(system, "Reply with the single word: ready.", max_tokens=4)


ANSWER_INSTRUCTION = (
    "Answer the question using only the information above. "
    "Be concise and factual."
)


def build_user_prompt(question: str) -> str:
    """The per-query suffix.  ~30-50 tokens, per paper Section 3.4."""
    return f"{ANSWER_INSTRUCTION}\n\nQuestion: {question}\nAnswer:"


def _iter_sse(resp: requests.Response) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from an OpenAI-style SSE stream."""
    # SSE is UTF-8 by definition, but llama-server sends a bare
    # `text/event-stream` with no charset, and requests decodes any such text/*
    # response as ISO-8859-1 -- "García" became "GarcÃ­a" in 1-2% of answers.
    resp.encoding = "utf-8"
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue
