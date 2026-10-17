#!/usr/bin/env python3
"""Full-context benchmark runner: TTFT, throughput, answers.

Paper Section 3.4: the whole corpus goes into a single system prompt, one
preflight request primes the KV cache, and every subsequent query sends only
the ~30-50 token question suffix.

The same runner serves the compiled-wiki conditions -- a wiki is just a
smaller context -- so FC and wiki arms are guaranteed to share a chat
template, a decoding config and a timing method.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile, resolve
from wicer_core.corpus import Corpus, QAPair, format_documents, load_topic
from wicer_core.provenance import build_run_meta, write_run_meta
from wicer_core.reader import (
    ContextOverflow,
    Generation,
    ReaderClient,
    build_user_prompt,
)
from wicer_core.records import (
    AnswerRecord,
    JsonlWriter,
    answers_path,
    completed_question_ids,
    run_dir,
    write_skip,
)

SYSTEM_PREAMBLE = (
    "You are a knowledgeable assistant. Use the reference material below to "
    "answer questions accurately.\n\n"
)


def build_fc_context(corpus: Corpus) -> str:
    """The full-context system prompt: every document, concatenated."""
    return SYSTEM_PREAMBLE + format_documents(corpus.documents)


def build_wiki_context(wiki_text: str) -> str:
    """A compiled wiki, served through the identical template."""
    return SYSTEM_PREAMBLE + wiki_text.strip()


def run_condition(
    profile: Profile,
    topic: str,
    condition: str,
    context: str,
    questions: Sequence[QAPair],
    *,
    reader: ReaderClient | None = None,
    resume: bool = False,
    progress: bool = True,
) -> dict:
    """Generate answers for one (topic, condition) pair.

    Returns a summary dict.  Raises ContextOverflow if the context does not fit
    the server's window -- callers decide whether that is fatal or a skip.
    """
    reader = reader or ReaderClient(profile.reader)
    reader.wait_for_health()

    n_context_tokens = reader.check_fits(context)

    out = run_dir(profile, topic, condition)
    out.mkdir(parents=True, exist_ok=True)
    (out / "context.txt").write_text(context)

    path = answers_path(profile, topic, condition)
    done = completed_question_ids(path) if resume else set()
    pending = [q for q in questions if q.question_id not in done]

    # Prime once; every query after this reuses the KV prefix.
    prime: Generation = reader.prime(context)

    iterator: Sequence[QAPair] = pending
    if progress and pending:
        try:
            from tqdm import tqdm

            iterator = tqdm(pending, desc=f"{topic}/{condition}", unit="q")
        except ImportError:
            pass

    with JsonlWriter(path, resume=resume) as writer:
        for q in iterator:
            gen = reader.chat(context, build_user_prompt(q.question))
            writer.write(
                AnswerRecord(
                    question_id=q.question_id,
                    document_id=q.document_id,
                    question=q.question,
                    gold=q.answer,
                    generated=gen.text.strip(),
                    ttft_s=gen.ttft_s,
                    total_s=gen.total_s,
                    throughput_tok_s=gen.throughput_tok_s,
                    n_generated_tokens=gen.n_generated_tokens,
                    n_prompt_tokens=gen.n_prompt_tokens,
                    cache_n=gen.cache_n,
                    finish_reason=gen.finish_reason,
                ).to_dict()
            )

    summary = {
        "topic": topic,
        "condition": condition,
        "n_context_tokens": n_context_tokens,
        "n_questions": len(questions),
        "n_generated": len(pending),
        "n_resumed": len(done),
        "preflight_ttft_s": prime.ttft_s,
        "preflight_total_s": prime.total_s,
    }
    write_run_meta(
        out / "run_meta.json",
        build_run_meta(
            profile,
            command="benchmark/run_benchmark.py",
            extra=summary,
            server_props=_safe_props(reader),
            hash_model=False,
        ),
    )
    return summary


def _safe_props(reader: ReaderClient) -> dict | None:
    try:
        return reader.props()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all built")
    ap.add_argument("--condition", default="fc_raw", help="condition name (default: %(default)s)")
    ap.add_argument(
        "--context-file",
        help="serve this file as the context instead of the raw corpus "
        "(required for wiki_* / wicer_* / random_pin conditions)",
    )
    ap.add_argument("--limit", type=int, help="only run the first N questions")
    ap.add_argument("--resume", action="store_true", help="skip questions already answered")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report context size and fit, generate nothing",
    )
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    topics = args.topic or list(TOPICS)
    reader = ReaderClient(profile.reader)

    rc = 0
    for topic in topics:
        corpus = load_topic(profile, topic)
        if args.context_file:
            context = build_wiki_context(resolve(args.context_file).read_text())
        else:
            context = build_fc_context(corpus)

        questions = corpus.qa[: args.limit] if args.limit else corpus.qa

        if args.dry_run:
            rc |= _dry_run(reader, profile, topic, args.condition, context, questions)
            continue

        try:
            summary = run_condition(
                profile, topic, args.condition, context, questions, reader=reader,
                resume=args.resume,
            )
        except ContextOverflow as e:
            write_skip(
                profile, topic, args.condition, "context_overflow",
                n_context_tokens=e.n_tokens, limit=e.limit,
            )
            print(
                f"{topic:<26} SKIPPED context_overflow "
                f"({e.n_tokens} tokens > {e.limit} usable)"
            )
            continue
        print(
            f"{topic:<26} {args.condition:<16} "
            f"ctx={summary['n_context_tokens']:>7} tok  "
            f"answered={summary['n_generated']}"
        )
    return rc


def _dry_run(
    reader: ReaderClient,
    profile: Profile,
    topic: str,
    condition: str,
    context: str,
    questions: Sequence[QAPair],
) -> int:
    if not reader.health():
        print(
            f"{topic:<26} cannot measure: no llama-server at {profile.reader.base_url}",
            file=sys.stderr,
        )
        return 1
    n = reader.count_tokens(context)
    limit = reader.context_limit()
    verdict = "FITS" if n <= limit else "CONTEXT OVERFLOW"
    print(
        f"{topic:<26} {condition:<16} ctx={n:>7} tok  usable={limit:>7} tok  "
        f"questions={len(questions):>4}  {verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
