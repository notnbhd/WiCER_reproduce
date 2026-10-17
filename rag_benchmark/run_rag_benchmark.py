#!/usr/bin/env python3
"""RAG benchmark runner: embed the query, retrieve top-5, generate, measure.

Paper Section 3.4: top-5 chunks by cosine similarity, no reranking, formatted
into *the same chat template* as full-context.  Retrieval accuracy -- whether
the gold source document appears among the retrieved chunks, and at what rank
-- is recorded per question (paper Section 3.5).

Runs against the same llama-server as the full-context arm, so the two
conditions differ only in what fills the system prompt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile
from wicer_core.corpus import load_topic
from wicer_core.embedding import Embedder, cosine_top_k
from wicer_core.provenance import build_run_meta, write_run_meta
from wicer_core.reader import ReaderClient, build_user_prompt
from wicer_core.records import (
    AnswerRecord,
    JsonlWriter,
    answers_path,
    completed_question_ids,
    run_dir,
)
from benchmark.run_benchmark import SYSTEM_PREAMBLE
from rag_benchmark.build_index import load_index

CONDITION = "rag_top5"


def format_retrieved(chunks: list[dict]) -> str:
    """Retrieved passages, rendered like documents so the template matches."""
    parts = [
        f"=== PASSAGE {i + 1} (source: {c['document_id']}) ===\n{c['text'].strip()}"
        for i, c in enumerate(chunks)
    ]
    return SYSTEM_PREAMBLE + "\n\n".join(parts)


def run_topic(
    profile: Profile,
    topic: str,
    *,
    reader: ReaderClient | None = None,
    embedder: Embedder | None = None,
    limit: int | None = None,
    resume: bool = False,
    progress: bool = True,
) -> dict:
    corpus = load_topic(profile, topic)
    vectors, chunks, index_meta = load_index(profile, topic)
    reader = reader or ReaderClient(profile.reader)
    reader.wait_for_health()
    embedder = embedder or Embedder(profile.rag.embed_model)
    top_k = profile.rag.top_k

    questions = corpus.qa[:limit] if limit else corpus.qa
    path = answers_path(profile, topic, CONDITION)
    done = completed_question_ids(path) if resume else set()
    pending = [q for q in questions if q.question_id not in done]

    iterator = pending
    if progress and pending:
        try:
            from tqdm import tqdm

            iterator = tqdm(pending, desc=f"{topic}/{CONDITION}", unit="q")
        except ImportError:
            pass

    n_hits = 0
    with JsonlWriter(path, resume=resume) as writer:
        for q in iterator:
            qvec = embedder.encode([q.question])[0]
            idx, _sims = cosine_top_k(qvec, vectors, top_k)
            retrieved = [chunks[int(i)] for i in idx]
            context = format_retrieved(retrieved)

            rank = next(
                (i + 1 for i, c in enumerate(retrieved) if c["document_id"] == q.document_id),
                None,
            )
            n_hits += rank is not None

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
                    retrieval_hit=rank is not None,
                    retrieval_rank=rank,
                    retrieved_document_ids=[c["document_id"] for c in retrieved],
                    retrieved_tokens=gen.n_prompt_tokens,
                ).to_dict()
            )

    summary = {
        "topic": topic,
        "condition": CONDITION,
        "n_questions": len(questions),
        "n_generated": len(pending),
        "retrieval_accuracy_pct": 100.0 * n_hits / len(pending) if pending else None,
        "index_meta": index_meta,
    }
    write_run_meta(
        run_dir(profile, topic, CONDITION) / "run_meta.json",
        build_run_meta(
            profile, command="rag_benchmark/run_rag_benchmark.py",
            extra=summary, hash_model=False,
        ),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all")
    ap.add_argument("--limit", type=int, help="only run the first N questions")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    reader = ReaderClient(profile.reader)
    embedder = Embedder(profile.rag.embed_model)
    for topic in args.topic or list(TOPICS):
        s = run_topic(profile, topic, reader=reader, embedder=embedder,
                      limit=args.limit, resume=args.resume)
        acc = s["retrieval_accuracy_pct"]
        if acc is None:
            print(f"{topic:<26} nothing to do")
        else:
            print(
                f"{topic:<26} answered={s['n_generated']:>4} "
                f"retrieval_accuracy={acc:.1f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
