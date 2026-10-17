#!/usr/bin/env python3
"""Aggregate per-question records into the paper's tables.

Emits, into ``<results_dir>/``:

    per_question.csv     every scored question, every condition
    per_topic.csv        one row per (topic, condition)
    report.md            the paper-shaped tables plus the success-criteria check

Table shapes follow paper Tables 3 (compilation), 4 (WiCER), 5 (ablation),
12 (generalisation summary) and 13 (latency).  The success-criteria block
checks spec.md Section 8 target-by-target and prints the delta, so a run
states plainly whether it reproduced rather than leaving that to the reader.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile, resolve
from wicer_core.metrics import (
    RetrievalStats,
    ScoreStats,
    TimingStats,
    fmt,
    mean,
    median,
    recovery_rate,
    retrieval_stats,
    score_stats,
    timing_stats,
)
from wicer_core.records import (
    CONDITIONS,
    answers_path,
    load_ledger,
    load_wiki_metas,
    read_jsonl,
    run_dir,
    scores_path,
    skip_path,
)

# spec.md Section 8, over 17 RepLiQA topics.
SUCCESS_CRITERIA = [
    ("FC raw mean", "fc_raw", "mean", 3.46, 0.20),
    ("RAG mean", "rag_top5", "mean", 3.63, 0.20),
    ("Wiki-moderate (blind) mean", "wiki_moderate", "mean", 2.25, 0.30),
    ("Wiki-moderate score-1 rate %", "wiki_moderate", "pct_eq_1", 57.1, 10.0),
    ("Best WiCER mean", "wicer_best", "mean", 3.18, 0.30),
    ("Best WiCER score-1 rate %", "wicer_best", "pct_eq_1", 24.8, 10.0),
]


@dataclass
class CondResult:
    topic: str
    condition: str
    scores: ScoreStats
    timings: TimingStats
    retrieval: RetrievalStats | None
    n_context_tokens: int | None
    n_answers: int
    skipped: str | None = None

    def row(self) -> dict[str, Any]:
        d = {
            "topic": self.topic,
            "condition": self.condition,
            "skipped": self.skipped or "",
            "n_context_tokens": self.n_context_tokens,
            "n_answers": self.n_answers,
            **self.scores.to_dict(),
            **{f"ttft_{k}": v for k, v in self.timings.to_dict().items() if k.startswith("ttft")},
            "total_median": self.timings.total_median,
            "throughput_median": self.timings.throughput_median,
        }
        if self.retrieval:
            d.update({f"retrieval_{k}": v for k, v in self.retrieval.to_dict().items()})
        return d


def collect(profile: Profile, topics: Sequence[str], conditions: Sequence[str]) -> list[CondResult]:
    out: list[CondResult] = []
    for topic in topics:
        for condition in conditions:
            d = run_dir(profile, topic, condition)
            if not d.is_dir():
                continue
            skipped = None
            if skip_path(profile, topic, condition).exists():
                skipped = json.loads(skip_path(profile, topic, condition).read_text()).get(
                    "skipped", "skipped"
                )
            answers = {a["question_id"]: a for a in read_jsonl(answers_path(profile, topic, condition))}
            scores = read_jsonl(scores_path(profile, topic, condition))
            if not answers and not skipped:
                continue

            paired = [(answers[s["question_id"]], s) for s in scores if s["question_id"] in answers]
            n_ctx = _context_tokens(profile, topic, condition)
            ret = None
            if paired and paired[0][0].get("retrieval_hit") is not None:
                ret = retrieval_stats(
                    [bool(a.get("retrieval_hit")) for a, _ in paired],
                    [a.get("retrieval_rank") for a, _ in paired],
                )
            out.append(
                CondResult(
                    topic=topic,
                    condition=condition,
                    scores=score_stats(
                        [int(s["score"]) for _, s in paired],
                        sum(1 for _, s in paired if s.get("parse_failure")),
                    ),
                    timings=timing_stats(
                        [a["ttft_s"] for a in answers.values()],
                        [a["total_s"] for a in answers.values()],
                        [a["throughput_tok_s"] for a in answers.values()],
                    ),
                    retrieval=ret,
                    n_context_tokens=n_ctx,
                    n_answers=len(answers),
                    skipped=skipped,
                )
            )
    return out


def _context_tokens(profile: Profile, topic: str, condition: str) -> int | None:
    """Context size for a condition.

    Fixed-context arms record it once in run_meta. RAG builds a different
    context per query, so fall back to the median prompt size the server
    reported across that condition's answers.
    """
    meta = run_dir(profile, topic, condition) / "run_meta.json"
    if meta.exists():
        try:
            n = json.loads(meta.read_text()).get("n_context_tokens")
        except json.JSONDecodeError:
            n = None
        if n:
            return int(n)
    prompts = [
        a["n_prompt_tokens"]
        for a in read_jsonl(answers_path(profile, topic, condition))
        if a.get("n_prompt_tokens")
    ]
    return int(median(prompts)) if prompts else None


# -- rendering ----------------------------------------------------------------


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_(no data)_\n"
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return f"{line}\n{sep}\n{body}\n"


def _by(results: Sequence[CondResult], condition: str) -> dict[str, CondResult]:
    return {r.topic: r for r in results if r.condition == condition and r.scores.n}


def _wicer_best(profile: Profile, topic: str, results: Sequence[CondResult]) -> CondResult | None:
    """The best WiCER iteration for a topic, as the paper's Table 4 reports."""
    cands = [r for r in results if r.topic == topic and r.condition.startswith("wicer_iter") and r.scores.n]
    if not cands:
        return None
    return max(cands, key=lambda r: r.scores.mean)


def render(profile: Profile, results: list[CondResult], topics: Sequence[str]) -> str:
    parts: list[str] = ["# WiCER reproduction report\n"]
    parts.append(f"Profile: `{profile.name}`  ·  judge: `{profile.judge.model}`  ·  "
                 f"compiler: `{profile.compiler.model}`\n")
    parts.append(f"Reader: `{profile.reader.model_path}`, n_ctx={profile.reader.n_ctx}, "
                 f"KV={profile.reader.cache_type_k}/{profile.reader.cache_type_v}\n")

    # -- per topic x condition
    parts.append("\n## Per topic and condition\n")
    rows = []
    for r in results:
        rows.append([
            r.topic, r.condition,
            r.skipped or r.scores.n,
            fmt(r.scores.mean), fmt(r.scores.pct_ge_4, 1), fmt(r.scores.pct_eq_1, 1),
            fmt(r.timings.ttft_median, 3), fmt(r.timings.throughput_median, 1),
            r.n_context_tokens if r.n_context_tokens is not None else "--",
        ])
    parts.append(md_table(
        ["Topic", "Condition", "n", "Mean", "%>=4", "%=1", "TTFT med (s)", "tok/s", "ctx tok"],
        rows,
    ))

    # -- Table 3: compilation levels
    parts.append("\n## Compilation: compression and quality (paper Table 3)\n")
    metas = {t: load_wiki_metas(profile, t) for t in topics}
    fc = _by(results, "fc_raw")
    fc_mean = mean([r.scores.mean for r in fc.values()]) if fc else float("nan")
    rows = []
    for label, cond, wiki_key, target in [
        ("FC raw (80 docs)", "fc_raw", None, None),
        ("Wiki-light", "wiki_light", "blind_light", 0.75),
        ("Wiki-moderate", "wiki_moderate", "blind_moderate", 0.50),
        ("Wiki-aggressive", "wiki_aggressive", "blind_aggressive", 0.25),
        ("RAG", "rag_top5", None, None),
    ]:
        sel = _by(results, cond)
        if not sel:
            continue
        m = mean([r.scores.mean for r in sel.values()])
        s1 = mean([r.scores.pct_eq_1 for r in sel.values()])
        actual = "--"
        if wiki_key:
            ratios = [
                metas[t][wiki_key]["actual_ratio_words"]
                for t in sel
                if wiki_key in metas.get(t, {}) and "actual_ratio_words" in metas[t][wiki_key]
            ]
            if ratios:
                actual = f"{100 * mean(ratios):.1f}%"
        rows.append([
            label,
            f"{100 * target:.0f}%" if target else "--",
            actual,
            fmt(m), f"{s1:.1f}%",
            fmt(m - fc_mean) if fc and cond != "fc_raw" else "--",
        ])
    parts.append(md_table(["Level", "Target", "Actual", "Quality", "Score-1 %", "vs. FC raw"], rows))

    # -- Table 4: WiCER per topic, on the probe set
    #
    # Paper Table 4 reports "80 probes each" -- the same probes the loop
    # diagnoses against. Those numbers come from the loop ledger. The
    # held-out numbers over the full question set follow in the next section;
    # they are the honest generalisation figure and can disagree, because a
    # wiki tuned until its probes pass may have spent budget doing it.
    parts.append("\n## WiCER across topics, probe set (paper Table 4)\n")
    blind = _by(results, "wiki_moderate")
    ledgers = {t: load_ledger(profile, t, "wicer") for t in topics}
    rows, rec_vals, best_means, best_s1, blind_means = [], [], [], [], []
    for topic in topics:
        led = ledgers.get(topic)
        if not led or not led.get("iterations"):
            continue
        iters = led["iterations"]
        b = iters[0]
        best = max(iters, key=lambda r: r["probe_mean"])
        f = fc.get(topic)
        rec = recovery_rate(
            b["probe_mean"], best["probe_mean"], f.scores.mean if f else None
        )
        rows.append([
            topic,
            f"{fmt(b['probe_mean'])} / {fmt(b['probe_score_1_pct'], 1)}%",
            f"{fmt(best['probe_mean'])} / {fmt(best['probe_score_1_pct'], 1)}%",
            fmt(f.scores.mean) if f else "--",
            f"{rec:.0f}%" if rec is not None else "--",
            best["iteration"],
            led.get("stop_reason", "--"),
        ])
        blind_means.append(b["probe_mean"])
        best_means.append(best["probe_mean"])
        best_s1.append(best["probe_score_1_pct"])
        if rec is not None:
            rec_vals.append(rec)
    if rows:
        rows.append([
            "**Mean**", fmt(mean(blind_means)),
            f"{fmt(mean(best_means))} / {fmt(mean(best_s1), 1)}%",
            fmt(mean([r.scores.mean for r in fc.values()])) if fc else "--",
            f"{mean(rec_vals):.0f}%" if rec_vals else "--", "--", "--",
        ])
    parts.append(md_table(
        ["Topic", "Blind (mean / score-1)", "Best WiCER", "FC raw (held-out)",
         "Recovery", "Iter", "Stop"],
        rows,
    ))

    # -- iteration trajectory
    traj = []
    for topic in topics:
        led = ledgers.get(topic)
        if not led:
            continue
        for r in led["iterations"]:
            traj.append([
                topic, r["iteration"], r["wiki_name"], fmt(r["probe_mean"]),
                fmt(r["probe_score_1_pct"], 1), r["n_failures"], r["n_pinned_facts"],
                r["pinned_words"],
                # Did the compiler actually keep what it was told to keep?  A
                # low rate invalidates the row above it -- see DEVIATIONS.md 12.
                f"{r['preservation_rate']:.0%}"
                if r.get("preservation_rate") is not None else "--",
                fmt(r["actual_ratio_words"] * 100, 1) + "%",
                f"{r['improvement']:+.1%}" if r.get("improvement") is not None else "--",
            ])
    if traj:
        parts.append("\n### Iteration trajectory\n")
        parts.append(md_table(
            ["Topic", "Iter", "Wiki", "Probe mean", "Score-1 %", "Failures",
             "Pinned facts", "Pinned words", "Kept verbatim", "Actual ratio",
             "Improvement"],
            traj,
        ))

    # -- held-out evaluation
    parts.append("\n## WiCER held-out evaluation (full question set)\n")
    rows = []
    for topic in topics:
        b = blind.get(topic)
        best = _wicer_best(profile, topic, results)
        if not b and not best:
            continue
        f = fc.get(topic)
        rec = (
            recovery_rate(b.scores.mean, best.scores.mean, f.scores.mean if f else None)
            if b and best else None
        )
        rows.append([
            topic,
            f"{fmt(b.scores.mean)} / {fmt(b.scores.pct_eq_1, 1)}%" if b else "--",
            f"{fmt(best.scores.mean)} / {fmt(best.scores.pct_eq_1, 1)}%" if best else "--",
            fmt(f.scores.mean) if f else "--",
            f"{rec:.0f}%" if rec is not None else "--",
            best.condition.replace("wicer_iter", "") if best else "--",
        ])
    parts.append(md_table(
        ["Topic", "Blind (mean / score-1)", "Best WiCER", "FC raw", "Recovery", "Iter"],
        rows,
    ))
    parts.append(
        "\n> A probe-set gain that does not appear here is the loop fitting its "
        "own probes: pinned facts consume budget that would otherwise have gone "
        "to general coverage (paper Appendix I, 'random knowledge displacement').\n"
    )

    # -- Table 5: ablation, on the probe set
    parts.append("\n## Ablation: diagnosed vs. random pinning, probe set (paper Table 5)\n")
    rows, d_rand, d_wicer = [], [], []
    for topic in topics:
        led = ledgers.get(topic)
        rnd_led = load_ledger(profile, topic, "random_pin")
        if not led or not led.get("iterations"):
            continue
        b = led["iterations"][0]["probe_mean"]
        best_w = max(r["probe_mean"] for r in led["iterations"])
        best_r = (
            max(r["probe_mean"] for r in rnd_led["iterations"])
            if rnd_led and rnd_led.get("iterations") else None
        )
        rows.append([topic, fmt(b), fmt(best_r) if best_r is not None else "--", fmt(best_w)])
        d_wicer.append(best_w - b)
        if best_r is not None and rnd_led["iterations"]:
            d_rand.append(best_r - rnd_led["iterations"][0]["probe_mean"])
    if rows:
        rows.append([
            "**Mean delta vs. blind**", "--",
            f"{mean(d_rand):+.2f}" if d_rand else "--",
            f"{mean(d_wicer):+.2f}" if d_wicer else "--",
        ])
    parts.append(md_table(["Topic", "Blind", "Random Pin", "WiCER"], rows))
    parts.append(
        "\n> This is the load-bearing comparison (paper Section 7.5): it is the "
        "only thing separating 'diagnosis works' from 'pinning anything works'. "
        "The paper measures +0.16 for random pinning against +0.95 for WiCER.\n"
    )

    # -- Table 13: latency
    parts.append("\n## Latency by condition (paper Table 13)\n")
    rows = []
    for cond in CONDITIONS:
        sel = [r for r in results if r.condition == cond and r.timings.n]
        if not sel:
            continue
        rows.append([
            cond,
            f"{median([r.n_context_tokens for r in sel if r.n_context_tokens]):,.0f}"
            if any(r.n_context_tokens for r in sel) else "--",
            fmt(median([r.timings.ttft_median for r in sel]), 3),
            fmt(median([r.timings.throughput_median for r in sel]), 1),
        ])
    parts.append(md_table(["Condition", "Context tokens", "TTFT median (s)", "tok/s"], rows))

    # -- retrieval
    rag = [r for r in results if r.condition == "rag_top5" and r.retrieval]
    if rag:
        parts.append("\n## RAG retrieval accuracy (paper Section 5.1)\n")
        parts.append(md_table(
            ["Topic", "Accuracy %", "Mean rank (hits)", "MRR"],
            [[r.topic, fmt(r.retrieval.accuracy_pct, 1), fmt(r.retrieval.mean_rank),
              fmt(r.retrieval.mrr, 3)] for r in rag]
            + [["**Mean**", fmt(mean([r.retrieval.accuracy_pct for r in rag]), 1), "--", "--"]],
        ))

    # -- success criteria
    parts.append("\n## Success criteria (spec.md Section 8)\n")

    def from_conditions(condition: str, field_: str) -> dict[str, float]:
        return {t: getattr(r.scores, field_) for t, r in _by(results, condition).items()}

    def from_ledgers(field_: str) -> dict[str, float]:
        """WiCER targets come from paper Table 4, which is probe-based."""
        out: dict[str, float] = {}
        for topic in topics:
            led = ledgers.get(topic)
            if led and led.get("iterations"):
                best = max(led["iterations"], key=lambda r: r["probe_mean"])
                out[topic] = best[field_]
        return out

    measured: dict[tuple[str, str], dict[str, float]] = {
        ("fc_raw", "mean"): from_conditions("fc_raw", "mean"),
        ("rag_top5", "mean"): from_conditions("rag_top5", "mean"),
        ("wiki_moderate", "mean"): from_conditions("wiki_moderate", "mean"),
        ("wiki_moderate", "pct_eq_1"): from_conditions("wiki_moderate", "pct_eq_1"),
        ("wicer_best", "mean"): from_ledgers("probe_mean"),
        ("wicer_best", "pct_eq_1"): from_ledgers("probe_score_1_pct"),
    }

    rows = []
    for label, cond, field_, target, tol in SUCCESS_CRITERIA:
        sel = measured.get((cond, field_)) or {}
        if not sel:
            rows.append([label, fmt(target), "--", "--", "NOT RUN", "--"])
            continue
        got = mean(list(sel.values()))
        delta = got - target
        rows.append([
            label, fmt(target), fmt(got), f"{delta:+.2f}",
            "PASS" if abs(delta) <= tol else "MISS", f"n={len(sel)} topics",
        ])
    parts.append(md_table(["Criterion", "Target", "Measured", "Delta", "Verdict", "Coverage"], rows))
    parts.append(
        "\n> These targets describe the full 17-topic RepLiQA sweep. A partial or "
        "reduced-scale run will miss them by construction -- read the verdict "
        "column only when the coverage column says 17 topics.\n"
    )
    parts.append(
        "\n> Directional agreement is the real bar (spec.md Section 8): the FC/RAG "
        "crossover exists, and targeted diagnosis beats random pinning by a wide "
        "margin. Exact numbers will move -- the compiler is stochastic and TTFT "
        "changes outright on different hardware.\n"
    )
    return "\n".join(parts)


def write_csvs(
    profile: Profile, results: list[CondResult], stem: str = "report"
) -> tuple[Path, Path]:
    """CSVs are named after the report they accompany, so a narrowed report
    (rag_report.py) cannot overwrite the full sweep's tables."""
    out = resolve(profile.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "" if stem == "report" else f"{stem}_"

    per_topic = out / f"{prefix}per_topic.csv"
    rows = [r.row() for r in results]
    fields = sorted({k for r in rows for k in r})
    with per_topic.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    per_question = out / f"{prefix}per_question.csv"
    with per_question.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["topic", "condition", "question_id", "document_id", "score",
                    "ttft_s", "total_s", "throughput_tok_s", "retrieval_hit", "retrieval_rank"])
        for r in results:
            answers = {a["question_id"]: a for a in read_jsonl(answers_path(profile, r.topic, r.condition))}
            for s in read_jsonl(scores_path(profile, r.topic, r.condition)):
                a = answers.get(s["question_id"], {})
                w.writerow([r.topic, r.condition, s["question_id"], s.get("document_id", ""),
                            s["score"], a.get("ttft_s", ""), a.get("total_s", ""),
                            a.get("throughput_tok_s", ""), a.get("retrieval_hit", ""),
                            a.get("retrieval_rank", "")])
    return per_topic, per_question


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_profile_arg(ap)
    ap.add_argument("--topic", action="append", help="topic slug (repeatable); default: all")
    ap.add_argument("--condition", action="append", help="condition (repeatable); default: all known")
    ap.add_argument("--out", help="markdown output path (default: <results_dir>/report.md)")
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    topics = args.topic or list(TOPICS)
    conditions = args.condition or list(CONDITIONS)

    results = collect(profile, topics, conditions)
    if not results:
        print(f"no results under {resolve(profile.results_dir)}", file=sys.stderr)
        return 1

    present = [t for t in topics if any(r.topic == t for r in results)]
    text = render(profile, results, present)
    # A bare filename lands in the results dir; a path with separators is used as given.
    if args.out and ("/" in args.out or Path(args.out).is_absolute()):
        out = resolve(args.out)
    else:
        out = resolve(profile.results_dir) / (args.out or "report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    per_topic, per_question = write_csvs(profile, results, stem=out.stem)
    print(text)
    print(f"\nwrote {out}\nwrote {per_topic}\nwrote {per_question}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
