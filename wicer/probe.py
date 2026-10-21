#!/usr/bin/env python3
"""Diagnostic probe selection.

Paper Algorithm 1, line 1: "Select one QA pair per source document".  With 80
documents per topic that gives 80 probes and leaves ~320 pairs as the held-out
evaluation set.

One probe per document, not eighty probes sampled at random, is the point:
coverage is uniform, so every document gets an equal chance to reveal that the
compiler dropped it.  Within a document the probe is drawn uniformly from its
questions, so the probe set looks like the full set -- RepLiQA's unanswerable
questions included, at their usual rate.  The two pilots took each document's
first question instead, which never picked an unanswerable one (they are
almost all a document's 4th or 5th) and favoured questions about a document's
main subject, which a compiler keeps.  The draw is seeded per document, so a
rerun probes the same questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile, resolve
from wicer_core.corpus import QAPair, load_topic


def split_probes(qa: list[QAPair], seed: int) -> tuple[list[QAPair], list[QAPair]]:
    """(probes, held-out).  One probe per document, drawn uniformly from its questions."""
    by_doc: dict[str, list[QAPair]] = {}
    for q in sorted(qa, key=lambda q: (q.document_id, q.question_id)):
        by_doc.setdefault(q.document_id, []).append(q)
    probes = [_draw(by_doc[d], seed, d) for d in sorted(by_doc)]
    probe_ids = {q.question_id for q in probes}
    heldout = [q for q in qa if q.question_id not in probe_ids]
    return probes, heldout


def _draw(questions: list[QAPair], seed: int, document_id: str) -> QAPair:
    """Seeded per document, so the pick depends on neither input order nor
    which other documents are present."""
    digest = hashlib.sha256(f"{seed}:{document_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).choice(questions)


def probe_paths(profile: Profile, topic: str) -> tuple[Path, Path]:
    d = resolve(profile.results_dir) / topic
    return d / "probes.json", d / "heldout.json"


def build(profile: Profile, topic: str) -> tuple[list[QAPair], list[QAPair]]:
    corpus = load_topic(profile, topic)
    probes, heldout = split_probes(corpus.qa, profile.wicer.probe_seed)
    p_path, h_path = probe_paths(profile, topic)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    p_path.write_text(json.dumps([_row(q) for q in probes], indent=2) + "\n")
    h_path.write_text(json.dumps([_row(q) for q in heldout], indent=2) + "\n")
    return probes, heldout


def load(profile: Profile, topic: str) -> tuple[list[QAPair], list[QAPair]]:
    p_path, h_path = probe_paths(profile, topic)
    if not p_path.exists():
        return build(profile, topic)
    return (
        [QAPair(**r) for r in json.loads(p_path.read_text())],
        [QAPair(**r) for r in json.loads(h_path.read_text())],
    )


def _row(q: QAPair) -> dict:
    return {
        "question_id": q.question_id,
        "document_id": q.document_id,
        "question": q.question,
        "answer": q.answer,
        "long_answer": q.long_answer,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all")
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    for topic in args.topic or list(TOPICS):
        probes, heldout = build(profile, topic)
        n_docs = len({q.document_id for q in probes})
        print(f"{topic:<26} probes={len(probes):>3} (one per {n_docs} docs) heldout={len(heldout):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
