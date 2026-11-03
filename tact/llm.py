"""A JSON-replying model call, shared by the extractor and the question writer.

Same backends as wicer's compiler (``openrouter``, ``anthropic``, and an
offline ``fake`` that subclasses implement).  A reply cut off at max_tokens is
an error, never parsed: a truncated JSON list would silently drop its tail.
"""

from __future__ import annotations

import json
import os
from typing import Any

_TRUNCATED = frozenset({"length", "max_tokens"})


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("reply is not a JSON object")
    return value


class JsonLLM:
    """``env`` names the knobs: ``{env}_EFFORT`` and ``{env}_MAX_TOKENS``."""

    system = "You reply with JSON only."

    def __init__(self, model: str, *, env: str, effort: str, max_tokens: int,
                 backend: str | None = None, openrouter_client: Any = None):
        self.model = model
        self.env = env
        self.backend = backend or os.environ.get("WICER_COMPILER_BACKEND", "anthropic")
        self.effort = os.environ.get(f"{env}_EFFORT", effort)
        self.max_tokens = int(os.environ.get(f"{env}_MAX_TOKENS", max_tokens))
        self._or_client = openrouter_client
        self._client = None
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.measured_cost_usd = 0.0

    @property
    def openrouter(self):
        if self._or_client is None:
            from wicer_core.openrouter import OpenRouterClient

            self._or_client = OpenRouterClient()
        return self._or_client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    @property
    def provenance(self) -> str:
        return "fake" if self.backend == "fake" else f"{self.model} effort={self.effort}"

    def ask(self, prompt: str) -> dict[str, Any]:
        """One prompt, one JSON object; a reply that does not parse is retried once."""
        last: Exception | None = None
        for _ in range(2):
            try:
                return parse_json(self._call(prompt))
            except ValueError as e:
                last = e
        raise LLMError(f"{self.model} returned no parseable JSON: {last}")

    def _call(self, prompt: str) -> str:
        self.n_calls += 1
        if self.backend == "openrouter":
            from wicer_core.config import resolve_openrouter_model

            reply = self.openrouter.chat(
                model=resolve_openrouter_model(self.model), system=self.system,
                user=prompt, max_tokens=self.max_tokens, effort=self.effort,
            )
            self.input_tokens += reply.input_tokens
            self.output_tokens += reply.output_tokens
            self.measured_cost_usd += reply.cost_usd
            text, finish = reply.text, reply.finish_reason
        else:
            with self.client.messages.stream(
                model=self.model, max_tokens=self.max_tokens, system=self.system,
                thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                msg = stream.get_final_message()
            self.input_tokens += msg.usage.input_tokens
            self.output_tokens += msg.usage.output_tokens
            text = "".join(b.text for b in msg.content if b.type == "text")
            finish = msg.stop_reason or ""
        if finish in _TRUNCATED:
            raise LLMError(
                f"{self.model} hit max_tokens={self.max_tokens}; raise "
                f"{self.env}_MAX_TOKENS or lower {self.env}_EFFORT ({self.effort!r})"
            )
        return text

    def usage_summary(self) -> dict[str, Any]:
        return {"model": self.model, "backend": self.backend, "effort": self.effort,
                "calls": self.n_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "measured_cost_usd": round(self.measured_cost_usd, 6)}
