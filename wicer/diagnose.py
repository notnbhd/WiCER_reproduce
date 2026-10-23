#!/usr/bin/env python3
"""Failure diagnosis: what exactly did the compiler drop?

Paper Section 7.2: WiCER focuses on catastrophic failures (score 1/5) and
extracts ~50-100 words per failure, against ~700 words per source document.
That ratio is the whole result -- the ablation (Section 7.5, Table 5) shows
pinning a random passage of equivalent length buys +0.16 while pinning the
diagnosed facts buys +0.95.

Diagnosis sees one source document, the probe question, the gold answer and
the wrong answer.  It never sees the wiki: the question is "what did this
answer need", not "what is missing from the compilation".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import WicerConfig
from wicer_core.corpus import Document, QAPair
from wicer.compile import PROMPT_DIR, PreservedFact

DIAGNOSIS_SYSTEM = (
    "You extract the specific facts a wrong answer needed from its source "
    "document. You are terse and you quote specifics exactly."
)


class Diagnoser:
    def __init__(
        self,
        model: str,
        cfg: WicerConfig,
        backend: str | None = None,
        price_in: float = 3.0,
        price_out: float = 15.0,
        openrouter_client: Any = None,
    ):
        import os

        self.model = model
        self.cfg = cfg
        self.backend = backend or os.environ.get("WICER_COMPILER_BACKEND", "anthropic")
        self.price_in = price_in
        self.price_out = price_out
        self._client = None
        self._or_client = openrouter_client
        self.measured_cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_calls = 0

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    @property
    def openrouter(self):
        if self._or_client is None:
            from wicer_core.openrouter import OpenRouterClient

            self._or_client = OpenRouterClient()
        return self._or_client

    def diagnose(
        self, probe: QAPair, document: Document, generated: str, *, iteration: int = 0
    ) -> PreservedFact:
        prompt = (PROMPT_DIR / "diagnose.md").read_text().format(
            question=probe.question,
            gold=probe.answer,
            generated=generated.strip() or "(no answer)",
            document_id=document.document_id,
            document=document.text,
            min_words=self.cfg.diagnosis_min_words,
            max_words=self.cfg.diagnosis_max_words,
        )
        if self.backend == "fake":
            text = _fake_diagnose(probe, document, self.cfg.diagnosis_max_words)
        elif self.backend == "openrouter":
            text = self._diagnose_openrouter(prompt)
        else:
            text = self._diagnose_anthropic(prompt)
        self.n_calls += 1
        return PreservedFact(
            document_id=document.document_id,
            question_id=probe.question_id,
            text=_clip_words(text, self.cfg.diagnosis_max_words),
            iteration=iteration,
        )

    def _diagnose_openrouter(self, prompt: str) -> str:
        """Diagnose via OpenRouter.

        No cache breakpoint: every diagnosis carries a different single
        document, so there is no shared prefix worth paying to write.
        """
        from wicer_core.config import resolve_openrouter_model

        reply = self.openrouter.chat(
            model=resolve_openrouter_model(self.model),
            system=DIAGNOSIS_SYSTEM,
            user=prompt,
            # ~100 words of output, but Haiku's reasoning bills against this cap
            # too: 600 once came back with all 600 spent thinking and no text.
            max_tokens=2_000,
            effort="low",
            cache_system=False,
        )
        self.input_tokens += reply.input_tokens
        self.output_tokens += reply.output_tokens
        self.measured_cost_usd += reply.cost_usd
        return reply.text

    def _diagnose_anthropic(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            # ~100 words of output; the cap is generous, not a target.
            max_tokens=600,
            system=DIAGNOSIS_SYSTEM,
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def usage_summary(self) -> dict[str, Any]:
        return {
            "diagnosis_model": self.model,
            "backend": self.backend,
            "calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(
                self.input_tokens / 1e6 * self.price_in
                + self.output_tokens / 1e6 * self.price_out,
                4,
            ),
            "measured_cost_usd": round(self.measured_cost_usd, 6),
            "cost_source": "openrouter" if self.backend == "openrouter" else "estimate",
        }


def _clip_words(text: str, max_words: int) -> str:
    """Enforce the word budget.  Over-long diagnoses would quietly turn the
    experiment into 'pin the whole document', which is the thing being tested
    against."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words])


def _fake_diagnose(probe: QAPair, document: Document, max_words: int) -> str:
    """Offline stand-in: the window of the document around the gold answer.

    Deterministic, and genuinely targeted -- it recovers the passage the answer
    came from -- so the offline loop exercises real recovery behaviour rather
    than a no-op.
    """
    words = document.text.split()
    gold_tokens = {w.strip(".,;:()").lower() for w in probe.answer.split() if len(w) > 3}
    best_i, best_hits = 0, -1
    window = max_words
    for i in range(0, max(1, len(words) - window + 1), 5):
        hits = sum(1 for w in words[i : i + window] if w.strip(".,;:()").lower() in gold_tokens)
        if hits > best_hits:
            best_i, best_hits = i, hits
    return " ".join(words[best_i : best_i + window])


def diagnose_failures(
    diagnoser: Diagnoser,
    failures: Sequence[tuple[QAPair, Document, str]],
    *,
    iteration: int = 0,
) -> list[PreservedFact]:
    return [
        diagnoser.diagnose(probe, doc, generated, iteration=iteration)
        for probe, doc, generated in failures
    ]
