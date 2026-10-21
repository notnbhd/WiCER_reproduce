#!/usr/bin/env python3
"""Score generated answers with the Claude LLM-as-judge (Appendix B rubric).

The identical judge and rubric are applied to every condition -- full-context,
RAG and every compiled wiki -- which is what makes the paper's cross-condition
comparisons legitimate (paper Section 3.5).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile
from wicer_core.judge import Judge, JudgeRequest, reparse_failure
from wicer_core.records import (
    JsonlWriter,
    ScoreRecord,
    answers_path,
    completed_question_ids,
    read_jsonl,
    run_dir,
    scores_path,
    skip_path,
)


def evaluate_condition(
    profile: Profile,
    topic: str,
    condition: str,
    *,
    judge: Judge | None = None,
    resume: bool = True,
    limit: int | None = None,
) -> dict:
    """Judge every answer for one (topic, condition) pair."""
    judge = judge or Judge(profile.judge)
    answers = read_jsonl(answers_path(profile, topic, condition))
    if limit:
        answers = answers[:limit]
    if not answers:
        return {"topic": topic, "condition": condition, "n_scored": 0, "note": "no answers"}

    path = scores_path(profile, topic, condition)
    done = completed_question_ids(path) if resume else set()
    pending = [a for a in answers if a["question_id"] not in done]

    requests = [
        JudgeRequest(question=a["question"], gold=a["gold"], generated=a["generated"])
        for a in pending
    ]
    judgements = judge.score_many(requests)

    with JsonlWriter(path, resume=resume and bool(done)) as writer:
        for a, j in zip(pending, judgements):
            writer.write(
                ScoreRecord(
                    question_id=a["question_id"],
                    document_id=a["document_id"],
                    score=j.score,
                    reasoning=j.reasoning,
                    judge_model=j.model,
                    parse_failure=j.parse_failure,
                ).to_dict()
            )

    usage = judge.usage_summary()
    (run_dir(profile, topic, condition) / "judge_usage.json").write_text(
        json.dumps(usage, indent=2) + "\n"
    )
    return {
        "topic": topic,
        "condition": condition,
        "n_scored": len(pending),
        "n_resumed": len(done),
        **usage,
    }


def reparse_failures(profile: Profile, topic: str, condition: str) -> list[dict]:
    """Read recorded parse failures back with the current parser.  No judge call.

    A verdict whose reasoning quoted the answer without escaping the quotes used
    to be recorded as a failure scored 1.  The judge's reply is kept in that
    row's reasoning, so its score can be recovered without asking again -- and
    without rewriting judge_usage.json, the record of what the run spent, as a
    re-run of evaluate_condition would.  Rows still holding no verdict stay
    failures.  Returns the repaired rows.
    """
    path = scores_path(profile, topic, condition)
    rows = read_jsonl(path)
    fixed = []
    for row in rows:
        if not row.get("parse_failure"):
            continue
        parsed = reparse_failure(row.get("reasoning", ""))
        if parsed is None:
            continue
        row["score"], row["reasoning"] = parsed
        row["parse_failure"] = False
        row["reparsed"] = True
        fixed.append(row)
    if fixed:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.replace(path)
    return fixed


def _conditions_present(profile: Profile, topic: str) -> list[str]:
    from wicer_core.config import resolve

    root = resolve(profile.results_dir) / topic
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "answers.jsonl").exists())


def main(argv: list[str] | None = None, *, default_condition: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all")
    ap.add_argument(
        "--condition",
        action="append",
        help="condition name (repeatable); default: every condition with answers",
    )
    ap.add_argument("--limit", type=int, help="only score the first N answers")
    ap.add_argument("--no-resume", action="store_true", help="re-score everything")
    ap.add_argument(
        "--reparse-failures",
        action="store_true",
        help="re-read recorded parse failures with the current parser and exit; "
        "no judge calls, judge_usage.json untouched",
    )
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    topics = args.topic or list(TOPICS)
    conditions = args.condition or ([default_condition] if default_condition else None)

    if args.reparse_failures:
        for topic in topics:
            for condition in conditions or _conditions_present(profile, topic):
                for row in reparse_failures(profile, topic, condition):
                    print(f"{topic:<26} {condition:<16} {row['question_id']}: "
                          f"recorded 1, verdict was {row['score']}")
        return 0

    judge = Judge(profile.judge)
    for topic in topics:
        for condition in conditions or _conditions_present(profile, topic):
            if skip_path(profile, topic, condition).exists():
                print(f"{topic:<26} {condition:<16} skipped upstream")
                continue
            res = evaluate_condition(
                profile, topic, condition, judge=judge,
                resume=not args.no_resume, limit=args.limit,
            )
            if res.get("note"):
                continue
            print(
                f"{topic:<26} {condition:<16} scored={res['n_scored']:>4} "
                f"resumed={res.get('n_resumed', 0):>4} "
                f"cache_hits={res.get('cache_hits', 0):>4}"
            )

    print(f"\njudge usage: {judge.usage_summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
