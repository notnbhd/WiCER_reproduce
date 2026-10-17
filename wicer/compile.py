#!/usr/bin/env python3
"""Knowledge compiler: documents -> a single compiled wiki.

Paper Section 6.1: an LLM acts as a knowledge engineer, blind to the
evaluation questions, targeting light (~75%), moderate (~50%) or aggressive
(~25%) compression.

Two things this module deliberately does *not* do:

* It does not enforce the compression target.  Paper Section 6.2 reports that
  targets of 75/50/25% produce actual ratios of 35.4/12.2/8.2%, and that
  over-compression is the mechanism behind the compilation gap.  Forcing
  compliance would change the experiment (spec.md Section 5).  We measure the
  actual ratio in both words and reader tokens and report it.
* It does not see the questions.  Blindness is what the whole WiCER loop is
  built to compensate for.

The source documents go in a cached system block: the same collection is
recompiled at three levels and again on every WiCER iteration, so prompt
caching cuts the compiler bill substantially.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import (
    TOPICS,
    CompilerConfig,
    Profile,
    add_profile_arg,
    get_profile,
)
from wicer_core.corpus import Document, format_documents, load_topic
from wicer_core.metrics import compression_ratio
from wicer_core.records import wiki_paths

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PREFIX = (
    "You are compiling the following source document collection. "
    "The collection is fixed; the compilation instructions follow separately.\n\n"
)


@dataclass
class PreservedFact:
    """One diagnosed fact block, pinned into subsequent compilations."""

    document_id: str
    question_id: str
    text: str
    iteration: int = 0

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise(text: str) -> str:
    """Collapse whitespace so a reflowed fact still counts as preserved."""
    return " ".join(text.split())


def count_preserved(wiki: str, preserve: Sequence[PreservedFact]) -> int:
    """How many pinned facts appear verbatim in `wiki`, ignoring whitespace."""
    haystack = _normalise(wiki)
    return sum(1 for f in preserve if _normalise(f.text) in haystack)


@dataclass
class CompileResult:
    text: str
    topic: str
    target_ratio: float
    target_words: int
    source_words: int
    actual_words: int
    actual_ratio_words: float
    source_tokens: int | None = None
    actual_tokens: int | None = None
    actual_ratio_tokens: float | None = None
    n_preserved_facts: int = 0
    preserved_words: int = 0
    # How many pinned facts actually came back in the wiki.  WiCER's whole
    # mechanism is the pinning, so a compiler that ignores the instruction
    # yields a null result indistinguishable from "the method does not work".
    # None (not 0.0) when nothing was pinned -- blind compilation.
    n_preserved_found: int = 0
    preservation_rate: float | None = None
    model: str = ""
    seed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float = 0.0
    preserved_facts: list[dict[str, Any]] = field(default_factory=list)

    def meta(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text")
        return d


class Compiler:
    def __init__(self, cfg: CompilerConfig, backend: str | None = None,
                 price_in: float = 3.0, price_out: float = 15.0,
                 openrouter_client: Any = None):
        self.cfg = cfg
        import os

        self.backend = backend or os.environ.get("WICER_COMPILER_BACKEND", "anthropic")
        self._or_client = openrouter_client
        # OpenRouter bills per call and tells us what it charged; the price
        # table below is only a fallback for backends that report nothing.
        self.measured_cost_usd = 0.0
        self.price_in = price_in
        self.price_out = price_out
        self._client = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
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

    # -- prompt assembly ------------------------------------------------------

    @staticmethod
    def _load(name: str) -> str:
        return (PROMPT_DIR / name).read_text()

    def build_instruction(
        self, source_words: int, target_ratio: float, preserve: Sequence[PreservedFact]
    ) -> str:
        target_words = max(1, int(round(source_words * target_ratio)))
        instruction = self._load("compile.md").format(
            target_words=f"{target_words:,}",
            target_pct=target_ratio * 100,
            source_words=f"{source_words:,}",
        )
        if preserve:
            facts = "\n\n".join(
                f"[fact {i + 1} | source document {f.document_id}]\n{f.text.strip()}"
                for i, f in enumerate(preserve)
            )
            instruction += "\n\n" + self._load("compile_preserve.md").format(
                n_facts=len(preserve), facts=facts
            )
        return instruction

    # -- compilation ----------------------------------------------------------

    def compile(
        self,
        documents: Sequence[Document],
        target_ratio: float,
        *,
        topic: str = "",
        preserve: Sequence[PreservedFact] = (),
        seed: int | None = None,
        count_tokens: Callable[[str], int] | None = None,
    ) -> CompileResult:
        seed = self.cfg.seed if seed is None else seed
        source_text = format_documents(list(documents))
        source_words = sum(len(d.text.split()) for d in documents)
        target_words = max(1, int(round(source_words * target_ratio)))
        instruction = self.build_instruction(source_words, target_ratio, preserve)

        if self.backend == "fake":
            wiki, usage = _fake_compile(documents, target_words, preserve, seed), {}
        elif self.backend == "openrouter":
            wiki, usage = self._compile_openrouter(source_text, instruction)
        else:
            wiki, usage = self._compile_anthropic(source_text, instruction)

        actual_words = len(wiki.split())
        n_found = count_preserved(wiki, preserve)
        source_tokens = count_tokens(source_text) if count_tokens else None
        actual_tokens = count_tokens(wiki) if count_tokens else None

        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        cr = int(usage.get("cache_read_input_tokens", 0))
        cw = int(usage.get("cache_creation_input_tokens", 0))
        self.n_calls += 1
        self.input_tokens += in_tok + cw
        self.cache_read_tokens += cr
        self.output_tokens += out_tok

        return CompileResult(
            text=wiki,
            topic=topic,
            target_ratio=target_ratio,
            target_words=target_words,
            source_words=source_words,
            actual_words=actual_words,
            actual_ratio_words=compression_ratio(actual_words, source_words),
            source_tokens=source_tokens,
            actual_tokens=actual_tokens,
            actual_ratio_tokens=(
                compression_ratio(actual_tokens, source_tokens)
                if actual_tokens is not None and source_tokens
                else None
            ),
            n_preserved_facts=len(preserve),
            preserved_words=sum(f.n_words for f in preserve),
            n_preserved_found=n_found,
            preservation_rate=(n_found / len(preserve)) if preserve else None,
            model=usage.get("model", "fake"),
            seed=seed,
            input_tokens=in_tok + cw,
            output_tokens=out_tok,
            cache_read_tokens=cr,
            cache_write_tokens=cw,
            estimated_cost_usd=(
                float(usage["cost_usd"])
                if "cost_usd" in usage
                else (in_tok + cw * 1.25 + cr * 0.1) / 1e6 * self.price_in
                + out_tok / 1e6 * self.price_out
            ),
            preserved_facts=[f.to_dict() for f in preserve],
        )

    def _compile_openrouter(
        self, source_text: str, instruction: str
    ) -> tuple[str, dict[str, Any]]:
        """Compile via OpenRouter.

        The source blob goes in a cached system block with the same 1-hour TTL
        the Anthropic path asks for -- OpenRouter passes it through.  The same
        collection is recompiled at three levels, once per loop iteration and
        once per random-pin iteration, and the loop's reader and judge work
        between compiles far exceeds the default 5-minute window.
        """
        from wicer_core.config import resolve_openrouter_model

        reply = self.openrouter.chat(
            model=resolve_openrouter_model(self.cfg.model),
            system=SYSTEM_PREFIX + source_text,
            user=instruction,
            max_tokens=self.cfg.max_tokens,
            effort=self.cfg.effort,
            cache_system=True,
            cache_ttl="1h",
        )
        self.measured_cost_usd += reply.cost_usd
        return reply.text, {
            "model": reply.served_model,
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "cache_read_input_tokens": reply.cache_read_tokens,
            "cache_creation_input_tokens": 0,
            "cost_usd": reply.cost_usd,
        }

    def _compile_anthropic(self, source_text: str, instruction: str) -> tuple[str, dict[str, Any]]:
        # Streaming: a compiled wiki can run to tens of thousands of tokens and
        # a non-streaming request of that size risks an HTTP timeout.
        with self.client.messages.stream(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PREFIX + source_text,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            thinking={"type": "adaptive"},
            output_config={"effort": self.cfg.effort},
            messages=[{"role": "user", "content": instruction}],
        ) as stream:
            message = stream.get_final_message()

        text = "".join(b.text for b in message.content if b.type == "text").strip()
        u = message.usage
        return text, {
            "model": message.model,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }

    def estimate_cost_usd(self, source_words: int, n_calls: int = 1) -> float:
        """Rough pre-flight estimate: ~1.35 tokens/word in, cfg.max_tokens/4 out."""
        in_tok = source_words * 1.35
        out_tok = self.cfg.max_tokens / 4
        return n_calls * (in_tok / 1e6 * self.price_in + out_tok / 1e6 * self.price_out)

    def usage_summary(self) -> dict[str, Any]:
        return {
            "compiler_model": self.cfg.model,
            "backend": self.backend,
            "calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(
                self.input_tokens / 1e6 * self.price_in
                + self.cache_read_tokens / 1e6 * self.price_in * 0.1
                + self.output_tokens / 1e6 * self.price_out,
                4,
            ),
            "measured_cost_usd": round(self.measured_cost_usd, 6),
            "cost_source": "openrouter" if self.backend == "openrouter" else "estimate",
        }


def _fake_compile(
    documents: Sequence[Document],
    target_words: int,
    preserve: Sequence[PreservedFact],
    seed: int,
) -> str:
    """Deterministic extractive stand-in used by the offline harness.

    It reproduces the behaviour that matters for testing the loop: a lossy,
    budget-limited compression that drops detail, and an absolute priority for
    pinned facts.  It is not a model and is never used in a real run.
    """
    lines: list[str] = ["# Compiled wiki"]
    budget = target_words

    if preserve:
        lines.append("\n## Preserved facts")
        for f in preserve:
            lines.append(f"- ({f.document_id}) {f.text.strip()}")
            budget -= f.n_words

    per_doc = max(8, budget // max(1, len(documents)))
    lines.append("\n## Collection")
    for d in documents:
        words = d.text.split()
        # Deterministic per-seed offset: different seeds drop different detail,
        # which is what makes compiler non-determinism visible in a seed sweep.
        offset = (seed * 7) % max(1, len(words) // 4 or 1)
        lines.append(f"\n### {d.document_id}\n" + " ".join(words[offset : offset + per_doc]))
    return "\n".join(lines)


def compile_topic(
    profile: Profile,
    topic: str,
    level: str,
    *,
    compiler: Compiler | None = None,
    preserve: Sequence[PreservedFact] = (),
    name: str | None = None,
    seed: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> CompileResult:
    """Compile one topic at a named level and persist the wiki plus metadata."""
    corpus = load_topic(profile, topic)
    ratio = profile.compiler.levels.get(level)
    if ratio is None:
        raise KeyError(f"unknown level {level!r}; choose from {sorted(profile.compiler.levels)}")
    compiler = compiler or Compiler(
        profile.compiler,
        price_in=profile.judge.price_in_per_mtok,
        price_out=profile.judge.price_out_per_mtok,
    )
    result = compiler.compile(
        corpus.documents, ratio, topic=topic, preserve=preserve, seed=seed,
        count_tokens=count_tokens,
    )
    md, meta = wiki_paths(profile, topic, name or f"blind_{level}")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(result.text)
    meta.write_text(json.dumps({**result.meta(), "level": level}, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all")
    ap.add_argument(
        "--level", action="append",
        help="light | moderate | aggressive (repeatable); default: all three",
    )
    ap.add_argument("--seed", type=int, help="compiler seed; vary it to expose non-determinism")
    ap.add_argument("--name", help="wiki name override (single topic/level only)")
    ap.add_argument("--dry-run", action="store_true", help="estimate API cost and exit")
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    topics = args.topic or list(TOPICS)
    levels = args.level or list(profile.compiler.levels)
    compiler = Compiler(
        profile.compiler,
        price_in=profile.judge.price_in_per_mtok,
        price_out=profile.judge.price_out_per_mtok,
    )

    if args.dry_run:
        total = 0.0
        for topic in topics:
            corpus = load_topic(profile, topic)
            est = compiler.estimate_cost_usd(corpus.n_words, len(levels))
            total += est
            print(f"{topic:<26} words={corpus.n_words:>7} levels={len(levels)} est=${est:.2f}")
        print(f"\nestimated total: ${total:.2f} (before prompt-cache savings)")
        return 0

    # Reader token counts are optional: report actual ratios in reader tokens
    # when a server is up, in words always.
    count_tokens = None
    try:
        from wicer_core.reader import ReaderClient

        reader = ReaderClient(profile.reader)
        if reader.health():
            count_tokens = reader.count_tokens
    except Exception:  # noqa: BLE001
        pass

    for topic in topics:
        for level in levels:
            r = compile_topic(
                profile, topic, level, compiler=compiler, seed=args.seed,
                name=args.name if len(topics) == 1 and len(levels) == 1 else None,
                count_tokens=count_tokens,
            )
            print(
                f"{topic:<26} {level:<11} target={r.target_ratio:.0%} "
                f"actual_words={r.actual_ratio_words:.1%} "
                f"({r.actual_words:,}/{r.source_words:,} words)"
                + (f" actual_tokens={r.actual_ratio_tokens:.1%}" if r.actual_ratio_tokens else "")
            )
    print(f"\ncompiler usage: {compiler.usage_summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
