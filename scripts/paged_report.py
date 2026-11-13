#!/usr/bin/env python3
"""Tables for the paged-cache runs, from the committed results alone.

    uv run python scripts/paged_report.py           # 80 documents a topic (run_paged_doc.sh)
    uv run python scripts/paged_report.py --full    # whole topics (run_paged_full.sh)

80 documents: held-out answerable questions -- the sweep's probes
(result-sweep/<topic>/probes.json) and questions whose gold says the answer is
not in the document are dropped -- with result-sweep's fc_raw beside them.
Whole topics: every answerable question; no probes were ever drawn there.

Differences are paired on the same questions, pooled over topics, with a
bootstrap 95% interval (and a Wilcoxon p when scipy is installed).  The
gold answers come from each arm's answers.jsonl, so the laptop's corpora,
which may be stale, are never read.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from wicer_core.corpus import is_not_in_document  # noqa: E402

ARMS_80 = {"plain verbatim": "paged_none", "document fetch": "paged_bare_reader_doc",
           "FC": "fc_raw", "oracle": "paged_bare_oracle_doc"}
PAIRS_80 = [("document fetch", "plain verbatim"), ("document fetch", "FC"),
            ("plain verbatim", "FC"), ("oracle", "document fetch")]
ARMS_FULL = {"plain verbatim": "paged_none_b88", "fetch, bare stubs": "paged_bare_b88_reader_doc",
             "fetch, cut index": "paged_index_b88_reader_doc",
             "fetch, bare, 55K": "paged_bare_b55_reader_doc", "oracle": "paged_bare_b88_oracle_doc"}
PAIRS_FULL = [("fetch, bare stubs", "plain verbatim"), ("fetch, cut index", "plain verbatim"),
              ("fetch, cut index", "fetch, bare stubs"), ("fetch, bare, 55K", "fetch, bare stubs"),
              ("oracle", "fetch, cut index")]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def _arm_dir(full: bool, topic: str, arm: str) -> Path:
    if arm == "fc_raw":
        return REPO / "result-sweep" / topic / arm
    return REPO / ("result-full" if full else "result-tact") / topic / arm


def scores(full: bool, topic: str, arm: str) -> dict[str, int]:
    return {r["question_id"]: r["score"] for r in _rows(_arm_dir(full, topic, arm) / "scores.jsonl")}


def kept_questions(full: bool, topic: str, base_arm: str) -> list[str]:
    gold = {r["question_id"]: r["gold"] for r in _rows(_arm_dir(full, topic, base_arm) / "answers.jsonl")}
    held_back: set[str] = set()
    probes = REPO / "result-sweep" / topic / "probes.json"
    if not full and probes.exists():
        held_back = {p["question_id"] for p in json.loads(probes.read_text())}
    return sorted(q for q, g in gold.items() if q not in held_back and not is_not_in_document(g))


def interval(d: list[float], n: int = 2000) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def report(full: bool) -> str:
    arms, pairs = (ARMS_FULL, PAIRS_FULL) if full else (ARMS_80, PAIRS_80)
    root = REPO / ("result-full" if full else "result-tact")
    base = arms["plain verbatim"]
    topics = sorted(p.parent.parent.name for p in root.glob(f"*/{base}/scores.jsonl"))
    table: dict[str, dict[str, dict[str, int]]] = {}
    keep: dict[str, list[str]] = {}
    for t in topics:
        keep[t] = kept_questions(full, t, base)
        table[t] = {name: scores(full, t, arm) for name, arm in arms.items()}
    present = [n for n in arms if any(table[t][n] for t in topics)]

    def mean(t: str, n: str) -> str:
        v = [table[t][n][q] for q in keep[t] if q in table[t][n]]
        return f"{sum(v) / len(v):.2f}" if v else "--"

    out = [f"| Topic | n | {' | '.join(present)} |", "|---|---|" + "---|" * len(present)]
    out += [f"| {t} | {len(keep[t])} | {' | '.join(mean(t, n) for n in present)} |" for t in topics]
    out.append("")
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    for a, b in pairs:
        if a not in present or b not in present:
            continue
        d: list[float] = []
        n_topics = 0
        for t in topics:
            A, B = table[t][a], table[t][b]
            dt = [A[q] - B[q] for q in keep[t] if q in A and q in B]
            if dt:
                d += dt
                n_topics += 1
        lo, hi = interval(d)
        p = f", p={wilcoxon(d).pvalue:.1g}" if wilcoxon and any(d) else ""
        out.append(f"- {a} - {b}: {sum(d) / len(d):+.2f} [{lo:+.2f}, {hi:+.2f}]"
                   f" (n={len(d)}, {n_topics} topics{p})")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="whole topics (result-full/)")
    print(report(ap.parse_args().full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
