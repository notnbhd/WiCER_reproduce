"""Compare two judge configurations over answers already on disk.

Paper Appendix J validates its judge against a human rater.  You cannot
cheaply repeat that, but you can ask the question that actually matters before
swapping the judge model: *does the cheaper judge track the one it replaces,
closely enough that the findings survive?*

The bar is Appendix J's per-condition row -- Pearson r >= 0.89 and
|bias| <= 0.17 -- and the reason the bias bound is the binding one is paper
Section 5: the full-context/RAG gap is 0.18 points.  A judge carrying a
condition-dependent bias of that size can erase or invent E2 on its own.
E3 and the ablation move ~0.8-1.2 points and tolerate much more.

    uv run python benchmark/judge_agreement.py --profile paper \
        --candidate-model anthropic/claude-haiku-5.5 --limit 100

Answers are read from existing runs, so nothing is regenerated; only the
candidate judge spends money.  Scores are cached content-addressed per model,
so a rerun is free and the reference pass is never repeated.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.agreement import Agreement, agreement
from wicer_core.config import TOPICS, add_profile_arg, get_profile
from wicer_core.judge import Judge, JudgeRequest
from wicer_core.records import CONDITIONS, read_jsonl, run_dir


def collect(profile, topics, conditions, limit: int, seed: int) -> list[dict]:
    """Gather (condition, question, gold, generated) rows from finished runs."""
    rows: list[dict] = []
    for topic in topics:
        for cond in conditions:
            answers = run_dir(profile, topic, cond) / "answers.jsonl"
            if not answers.exists():
                continue
            for r in read_jsonl(answers):
                if not r.get("generated"):
                    continue
                rows.append({
                    "condition": cond,
                    "topic": topic,
                    "question_id": r["question_id"],
                    "question": r["question"],
                    "gold": r["gold"],
                    "generated": r["generated"],
                })

    if limit and len(rows) > limit:
        # Stratify by condition the way Appendix J stratifies, so no single
        # condition can dominate the sample and hide a per-condition bias.
        by_cond: dict[str, list[dict]] = {}
        for r in rows:
            by_cond.setdefault(r["condition"], []).append(r)
        rng = random.Random(seed)
        per = max(1, limit // len(by_cond))
        out: list[dict] = []
        for cond, group in sorted(by_cond.items()):
            rng.shuffle(group)
            out.extend(group[:per])
        rows = out[:limit]
    return rows


def score_all(judge: Judge, rows: list[dict]) -> list[int]:
    requests = [
        JudgeRequest(question=r["question"], gold=r["gold"], generated=r["generated"],
                     key=r["question_id"])
        for r in rows
    ]
    return [j.score for j in judge.score_many(requests)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all")
    ap.add_argument("--condition", action="append", choices=list(CONDITIONS),
                    help="condition (repeatable); default: all present on disk")
    ap.add_argument("--candidate-model", required=True,
                    help="the judge model under test, e.g. anthropic/claude-haiku-5.5")
    ap.add_argument("--candidate-backend", default="openrouter")
    ap.add_argument("--reference-model", default=None,
                    help="default: the profile's configured judge model")
    ap.add_argument("--reference-backend", default=None,
                    help="default: WICER_JUDGE_BACKEND, or anthropic")
    ap.add_argument("--limit", type=int, default=100,
                    help="stratified sample size (Appendix J used 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    topics = args.topic or list(TOPICS)
    conditions = args.condition or list(CONDITIONS)

    rows = collect(profile, topics, conditions, args.limit, args.seed)
    if not rows:
        print("no answers found on disk -- run the benchmark arms first", file=sys.stderr)
        return 1

    ref_cfg = profile.judge
    if args.reference_model:
        ref_cfg = replace(ref_cfg, model=args.reference_model)
    reference = Judge(ref_cfg, backend=args.reference_backend)
    candidate = Judge(replace(profile.judge, model=args.candidate_model),
                      backend=args.candidate_backend)

    print(f"scoring {len(rows)} answers with the reference judge "
          f"({ref_cfg.model}, {reference.backend}) ...")
    ref_scores = score_all(reference, rows)
    print(f"scoring the same {len(rows)} with the candidate "
          f"({args.candidate_model}, {candidate.backend}) ...")
    cand_scores = score_all(candidate, rows)

    groups = {"overall": agreement(cand_scores, ref_scores)}
    for cond in sorted({r["condition"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["condition"] == cond]
        if len(idx) >= 2:
            groups[cond] = agreement([cand_scores[i] for i in idx],
                                     [ref_scores[i] for i in idx])

    md = [
        f"# Judge agreement: {args.candidate_model} vs {ref_cfg.model}\n",
        f"n={len(rows)} answers, stratified by condition, seed {args.seed}.\n",
        "Bar is paper Appendix J's per-condition row: Pearson r >= 0.89 and ",
        "|bias| <= 0.17. The bias bound binds because the full-context/RAG gap ",
        "the sweep must detect is 0.18 points.\n\n",
        Agreement.render(groups),
        "\n",
    ]
    overall = groups["overall"]
    inconclusive = [n for n, a in groups.items() if a.verdict == "n/a"]
    if Agreement.all_conclusive_pass(groups):
        verdict = (
            "Every conclusive group clears the bar; the swap is defensible on "
            "the paper's own terms."
        )
    else:
        verdict = (
            "At least one group misses the bar. E3 and the ablation (~0.8-1.2 "
            "points) may still hold, but E2's 0.18-point gap is not safe with "
            "this judge."
        )
    if inconclusive:
        verdict += (
            f"\n\nInconclusive (undefined correlation -- every answer in the "
            f"group drew the same score, or too few samples): "
            f"{', '.join(sorted(inconclusive))}. Raise --limit to widen them."
        )
    md.append(verdict + "\n")
    report = "".join(md)
    print("\n" + report)

    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"wrote {out}")

    print(f"reference judge: {reference.usage_summary()}")
    print(f"candidate judge: {candidate.usage_summary()}")
    return 0 if overall.meets_paper_bar() else 2


if __name__ == "__main__":
    raise SystemExit(main())
