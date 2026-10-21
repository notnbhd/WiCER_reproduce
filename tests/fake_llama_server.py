#!/usr/bin/env python3
"""A stand-in for llama-server, so the whole pipeline runs with no GPU.

Implements the endpoints ``wicer_core.reader`` uses -- ``/health``, ``/props``,
``/tokenize`` and a streaming ``/v1/chat/completions`` -- with the same SSE
shape llama.cpp emits, including the ``timings`` extension.

The answering model is extractive: it returns the sentence in its context that
best overlaps the question. That is crude, but it makes the offline pipeline
behave the way the real one does in the respect that matters -- an answer is
only available if the relevant text is actually in the context. So a
full-context run answers well, an over-compressed wiki drops answers, and a
wiki with the right facts pinned back in recovers them.

    python tests/fake_llama_server.py --port 8299 --n-ctx 8192
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "was", "were", "and", "or", "for",
    "on", "at", "by", "with", "as", "that", "this", "it", "its", "what", "which",
    "who", "when", "where", "how", "why", "did", "does", "do", "answer", "question",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def content_words(text: str) -> set[str]:
    return {w.lower() for w in TOKEN_RE.findall(text) if w.lower() not in STOPWORDS and len(w) > 2}


def answer_from_context(context: str, question: str, max_sentences: int = 2) -> str:
    """Pick the best-overlapping sentences from the context."""
    q = content_words(question)
    if not q:
        return "I don't know."
    scored: list[tuple[float, int, str]] = []
    for i, sentence in enumerate(SENTENCE_RE.split(context)):
        s = sentence.strip()
        if len(s) < 12 or s.startswith("==="):
            continue
        overlap = len(q & content_words(s))
        if overlap:
            scored.append((overlap / len(q), i, s))
    if not scored:
        return "The provided material does not contain that information."
    scored.sort(key=lambda t: (-t[0], t[1]))
    return " ".join(s for _, _, s in scored[:max_sentences])


class FakeLlamaHandler(BaseHTTPRequestHandler):
    server_version = "fake-llama-server/0"
    n_ctx = 8192
    token_delay_s = 0.001
    prefill_delay_per_1k_s = 0.004
    _cache_lock = threading.Lock()
    _last_prefix: str = ""

    def log_message(self, *args: Any) -> None:  # keep the test output clean
        pass

    # -- plumbing -------------------------------------------------------------

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/props":
            self._json(200, {
                "n_ctx": self.n_ctx,
                "model_path": "<fake>",
                "default_generation_settings": {"n_ctx": self.n_ctx},
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/tokenize":
            body = self._read_json()
            toks = tokenize(str(body.get("content", "")))
            self._json(200, {"tokens": list(range(len(toks)))})
        elif self.path in ("/v1/chat/completions", "/chat/completions"):
            self._chat(self._read_json())
        else:
            self._json(404, {"error": "not found"})

    # -- generation -----------------------------------------------------------

    def _chat(self, body: dict) -> None:
        messages = body.get("messages") or []
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        max_tokens = int(body.get("max_tokens") or 128)

        n_prompt = len(tokenize(system)) + len(tokenize(user))
        if n_prompt > self.n_ctx:
            self._json(400, {"error": {"message": f"context {n_prompt} exceeds n_ctx {self.n_ctx}"}})
            return

        # Model the KV cache: a repeated system prefix prefills almost for free,
        # which is what gives the full-context arm its warm TTFT.
        with FakeLlamaHandler._cache_lock:
            reused = system == FakeLlamaHandler._last_prefix
            FakeLlamaHandler._last_prefix = system
        cache_n = len(tokenize(system)) if reused else 0
        prefill_tokens = n_prompt - cache_n

        question = user.split("Question:")[-1].split("Answer:")[0].strip() or user
        text = answer_from_context(system, question)
        pieces = re.findall(r"\S+\s*", text)[:max_tokens] or ["(empty)"]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        started = time.perf_counter()
        time.sleep(prefill_tokens / 1000.0 * self.prefill_delay_per_1k_s)
        prompt_ms = (time.perf_counter() - started) * 1000.0

        def send(obj: dict) -> None:
            # Raw UTF-8, as llama-server streams it, not json.dumps' default
            # \u escapes -- those are plain ASCII and would hide a client that
            # decodes the stream with the wrong charset.
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        try:
            for piece in pieces:
                time.sleep(self.token_delay_s)
                send({
                    "id": "fake", "object": "chat.completion.chunk", "model": "fake-llama",
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                })
            predicted_ms = (time.perf_counter() - started) * 1000.0 - prompt_ms
            send({
                "id": "fake", "object": "chat.completion.chunk", "model": "fake-llama",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": n_prompt,
                    "completion_tokens": len(pieces),
                    "total_tokens": n_prompt + len(pieces),
                },
                "timings": {
                    "prompt_n": prefill_tokens,
                    "prompt_ms": prompt_ms,
                    "predicted_n": len(pieces),
                    "predicted_ms": predicted_ms,
                    "cache_n": cache_n,
                },
            })
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass


class FakeLlamaServer:
    """Context manager wrapper, for use from tests and scripts/smoke.sh."""

    def __init__(self, port: int = 8299, n_ctx: int = 8192, token_delay_s: float = 0.001):
        FakeLlamaHandler.n_ctx = n_ctx
        FakeLlamaHandler.token_delay_s = token_delay_s
        FakeLlamaHandler._last_prefix = ""
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), FakeLlamaHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "FakeLlamaServer":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8299)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--token-delay", type=float, default=0.001)
    args = ap.parse_args()

    FakeLlamaHandler.n_ctx = args.n_ctx
    FakeLlamaHandler.token_delay_s = args.token_delay
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), FakeLlamaHandler)
    print(f"fake llama-server on http://127.0.0.1:{args.port}  n_ctx={args.n_ctx}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
