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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.run_benchmark import build_wiki_context
from wicer.loop import BLIND_CONDITION, BLIND_WIKI
from wicer.probe import probe_paths
from wicer_core.config import TOPICS, Profile, add_profile_arg, get_profile, resolve
from wicer_core.corpus import is_not_in_document
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
    wiki_dir,
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
    # Per-question scores and the questions RepLiQA marks unanswerable, so the
    # WiCER tables can score probes, held-out and unanswerable questions apart.
    by_question: dict[str, int] = field(default_factory=dict)
    unanswerable: frozenset[str] = frozenset()
    # The compiled wiki this condition's context was built from, if any, and
    # whether a pinning arm actually served one with nothing pinned.
    served_wiki: str | None = None
    unpinned_stand_in: bool = False

    def row(self) -> dict[str, Any]:
        d = {
            "topic": self.topic,
            "condition": self.condition,
            "served_wiki": self.served_wiki or "",
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
            served = _served_wiki(profile, topic, condition)
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
                    by_question={s["question_id"]: int(s["score"]) for _, s in paired},
                    unanswerable=frozenset(
                        q for q, a in answers.items() if is_not_in_document(a.get("gold", ""))
                    ),
                    served_wiki=served,
                    unpinned_stand_in=_unpinned_stand_in(profile, topic, condition, served),
                )
            )
    return out


def _served_wiki(profile: Profile, topic: str, condition: str) -> str | None:
    """The compiled wiki a condition's context was built from, if any."""
    ctx = run_dir(profile, topic, condition) / "context.txt"
    if not ctx.exists():
        return None
    context = ctx.read_text()
    for md in sorted(wiki_dir(profile, topic).glob("*.md")):
        if build_wiki_context(md.read_text()) == context:
            return md.stem
    return None


def _unpinned_stand_in(profile: Profile, topic: str, condition: str, served: str | None) -> bool:
    """A pinning arm's condition that served a wiki with nothing pinned in it.

    ``random_pin.py --eval-full`` scores the arm's best iteration, and best-of
    includes iteration 0, the unpinned blind -- which is what both pilots got.
    """
    if not served or not (condition == "random_pin" or condition.startswith("wicer_iter")):
        return False
    return load_wiki_metas(profile, topic).get(served, {}).get("n_preserved_facts") == 0


def _probe_ids(profile: Profile, topic: str) -> frozenset[str]:
    """The loop's probe questions; empty when no loop has run on the topic."""
    path = probe_paths(profile, topic)[0]
    if not path.exists():
        return frozenset()
    return frozenset(q["question_id"] for q in json.loads(path.read_text()))


def _stats(r: CondResult, keep: Callable[[str], bool]) -> ScoreStats:
    """Score statistics over the questions ``keep`` selects."""
    return score_stats([s for q, s in r.by_question.items() if keep(q)])


def _answerable(r: CondResult) -> ScoreStats:
    return _stats(r, lambda q: q not in r.unanswerable)


def _mean_and_s1(s: ScoreStats | None) -> str:
    return f"{fmt(s.mean)} / {fmt(s.pct_eq_1, 1)}%" if s and s.n else "--"


def _stand_in_note(results: Sequence[CondResult]) -> str:
    where = "; ".join(
        f"`{r.condition}` on {r.topic} served `{r.served_wiki}`"
        for r in results if r.unpinned_stand_in
    )
    return (
        f"\n> † Nothing pinned: {where}. `--eval-full` scores the arm's best "
        "iteration, and best-of includes iteration 0, the unpinned blind -- so the "
        "row is another blind compilation of the moderate prompt, not a pinning "
        "condition.\n"
    )


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


def _best_condition(ledger: dict[str, Any] | None) -> str | None:
    """The condition ``loop.py --eval-full`` scored the loop's best wiki under.

    The loop picks its best wiki on the probes.  Picking the best of several
    full-set conditions instead would select on the held-out questions.
    """
    name = (ledger or {}).get("best_wiki_name")
    if not name:
        return None
    return BLIND_CONDITION if name == BLIND_WIKI else name


def _result(results: Sequence[CondResult], topic: str, condition: str | None) -> CondResult | None:
    return next(
        (r for r in results if r.topic == topic and r.condition == condition and r.scores.n),
        None,
    )


def _arm_blind_and_best(ledger: dict[str, Any]) -> tuple[float, float]:
    """An ablation arm's own blind score and its best-of-iterations score.

    Each arm compiles its own blind wiki, so a gain is only meaningful against
    that arm's blind.  The ablation row used to print the WiCER arm's blind
    beside a delta computed from the random-pin arm's, which made the published
    arithmetic ``3.16 | 3.20 | +0.00`` -- three numbers that cannot all be true
    at once.
    """
    iters = ledger["iterations"]
    return iters[0]["probe_mean"], max(r["probe_mean"] for r in iters)


def _best_pinned(ledger: dict[str, Any]) -> float | None:
    """Best score among the iterations that actually pinned something.

    ``max`` over every iteration includes iteration 0, the unpinned blind, so an
    arm whose pinning only ever made things worse reports its own baseline and a
    delta of exactly zero -- reading as "pinning did nothing" when the truth is
    "pinning hurt, and selection discarded it".  Identifies pinned iterations by
    ``n_pinned_facts`` rather than by index, so it holds if the loop ever stops
    numbering from zero.  Returns ``None`` when no iteration pinned anything.
    """
    pinned = [r["probe_mean"] for r in ledger["iterations"] if r.get("n_pinned_facts")]
    return max(pinned) if pinned else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _distinct(values: Sequence[str | None]) -> list[str]:
    out: list[str] = []
    for v in values:
        if v and v not in out:
            out.append(v)
    return out


def _reader_desc(meta: dict[str, Any]) -> str | None:
    r = meta.get("reader") or {}
    if not r.get("model_path"):
        return None
    return (f"`{r['model_path']}`, n_ctx={r.get('n_ctx')}, "
            f"KV={r.get('cache_type_k')}/{r.get('cache_type_v')}")


def header_provenance(profile: Profile, topics: Sequence[str]) -> list[str]:
    """Judge, compiler and reader as the run on disk recorded them.

    These lines used to print ``profile.judge.model`` and friends -- the config
    as it stands *now*, not what ran.  Regenerating a report without the run's
    ``WICER_*`` overrides exported therefore relabelled a Haiku run as Sonnet,
    the config default, which matters because the judge's identity is what the
    bias correction hangs on.  So read it back off disk: the judge from each
    condition's ``judge_usage.json`` (the client that actually scored), the
    compiler and reader from the recorded metadata.  Distinct values are listed
    rather than collapsed -- a results tree built by two different models is a
    fact worth seeing, not something to average into one label.
    """
    metas, judges = [], []
    for topic in topics:
        for name in ("wicer", "random_pin"):
            if m := _read_json(resolve(profile.results_dir) / topic / f"{name}_run_meta.json"):
                metas.append(m)
        for condition in CONDITIONS:
            d = run_dir(profile, topic, condition)
            if m := _read_json(d / "run_meta.json"):
                metas.append(m)
            if u := _read_json(d / "judge_usage.json"):
                judges.append(u.get("judge_model"))

    judge = _distinct(judges + [m.get("judge_model_resolved") for m in metas])
    compiler = _distinct([m.get("compiler_model_resolved") for m in metas])
    reader = _distinct([_reader_desc(m) for m in metas])

    # Nothing on disk means nothing has run yet, so the live config is all there
    # is -- say so, rather than presenting it as a record of a run.
    suffix = "" if metas or judges else "  (config default -- no run metadata on disk)"
    if not judge:
        judge = [profile.judge.model]
    if not compiler:
        compiler = [profile.compiler.model]
    if not reader:
        reader = [
            (f"`{profile.reader.model_path}`, n_ctx={profile.reader.n_ctx}, "
             f"KV={profile.reader.cache_type_k}/{profile.reader.cache_type_v}")
        ]

    return [
        (f"Profile: `{profile.name}`  ·  judge: `{'`, `'.join(judge)}`  ·  "
         f"compiler: `{'`, `'.join(compiler)}`{suffix}\n"),
        f"Reader: {'; '.join(reader)}\n",
    ]


def render(profile: Profile, results: list[CondResult], topics: Sequence[str]) -> str:
    parts: list[str] = ["# WiCER reproduction report\n"]
    parts.extend(header_provenance(profile, topics))

    # -- per topic x condition
    parts.append("\n## Per topic and condition\n")
    rows = []
    for r in results:
        rows.append([
            r.topic, f"{r.condition} †" if r.unpinned_stand_in else r.condition,
            r.skipped or r.scores.n,
            fmt(r.scores.mean), fmt(r.scores.pct_ge_4, 1), fmt(r.scores.pct_eq_1, 1),
            fmt(r.timings.ttft_median, 3), fmt(r.timings.throughput_median, 1),
            r.n_context_tokens if r.n_context_tokens is not None else "--",
        ])
    parts.append(md_table(
        ["Topic", "Condition", "n", "Mean", "%>=4", "%=1", "TTFT med (s)", "tok/s", "ctx tok"],
        rows,
    ))
    stand_ins = [r for r in results if r.unpinned_stand_in]
    if stand_ins:
        parts.append(_stand_in_note(stand_ins))

    # -- Table 3: compilation levels
    parts.append("\n## Compilation: compression and quality (paper Table 3)\n")
    metas = {t: load_wiki_metas(profile, t) for t in topics}
    fc = _by(results, "fc_raw")
    fc_mean = mean([r.scores.mean for r in fc.values()]) if fc else float("nan")
    fc_ans_mean = mean([_answerable(r).mean for r in fc.values()]) if fc else float("nan")
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
        m_ans = mean([_answerable(r).mean for r in sel.values()])
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
            fmt(m_ans),
            fmt(m_ans - fc_ans_mean) if fc and cond != "fc_raw" else "--",
        ])
    parts.append(md_table(
        ["Level", "Target", "Actual", "Quality", "Score-1 %", "vs. FC raw",
         "Quality (answerable)", "vs. FC raw (answerable)"],
        rows,
    ))
    parts.append(
        "\n> The answerable columns leave out RepLiQA's unanswerable questions "
        "(gold: \"The answer is not found in the document.\"), about a fifth of each "
        "topic. A compiled wiki makes the reader abstain more, which those questions "
        "reward, so they shrink the gap the other columns measure.\n"
    )

    # -- Table 4: WiCER per topic, on the probe set
    #
    # Paper Table 4 reports "80 probes each" -- the same probes the loop
    # diagnoses against. Those numbers come from the loop ledger. The
    # held-out numbers, on questions the loop never saw, follow in the next
    # section; they are the honest generalisation figure and can disagree.
    parts.append("\n## WiCER across topics, probe set (paper Table 4)\n")
    blind = _by(results, "wiki_moderate")
    ledgers = {t: load_ledger(profile, t, "wicer") for t in topics}
    rows, rec_vals, best_means, best_s1, blind_means, fc_probe_means = [], [], [], [], [], []
    for topic in topics:
        led = ledgers.get(topic)
        if not led or not led.get("iterations"):
            continue
        iters = led["iterations"]
        b = iters[0]
        best = max(iters, key=lambda r: r["probe_mean"])
        f = fc.get(topic)
        fc_p = _stats(f, _probe_ids(profile, topic).__contains__) if f else None
        fc_probe = fc_p.mean if fc_p and fc_p.n else None
        rec = recovery_rate(b["probe_mean"], best["probe_mean"], fc_probe)
        rows.append([
            topic,
            f"{fmt(b['probe_mean'])} / {fmt(b['probe_score_1_pct'], 1)}%",
            f"{fmt(best['probe_mean'])} / {fmt(best['probe_score_1_pct'], 1)}%",
            fmt(fc_probe) if fc_probe is not None else "--",
            f"{rec:.0f}%" if rec is not None else "--",
            best["iteration"],
            led.get("stop_reason", "--"),
        ])
        blind_means.append(b["probe_mean"])
        best_means.append(best["probe_mean"])
        best_s1.append(best["probe_score_1_pct"])
        if fc_probe is not None:
            fc_probe_means.append(fc_probe)
        if rec is not None:
            rec_vals.append(rec)
    if rows:
        rows.append([
            "**Mean**", fmt(mean(blind_means)),
            f"{fmt(mean(best_means))} / {fmt(mean(best_s1), 1)}%",
            fmt(mean(fc_probe_means)),
            f"{mean(rec_vals):.0f}%" if rec_vals else "--", "--", "--",
        ])
    parts.append(md_table(
        ["Topic", "Blind (mean / score-1)", "Best WiCER", "FC raw (probes)",
         "Recovery", "Iter", "Stop"],
        rows,
    ))
    parts.append(
        "\n> FC raw is scored on the same probes as Blind and Best WiCER, so recovery "
        "compares one question set throughout. The paper divides by FC over every "
        "question, which agrees with this only when the probes are as hard as the "
        "full set.\n"
    )

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
    parts.append("\n## WiCER held-out evaluation (questions the loop never saw)\n")
    rows = []
    for topic in topics:
        led = ledgers.get(topic)
        b = blind.get(topic)
        best = _result(results, topic, _best_condition(led))
        if not b and not best:
            continue
        f = fc.get(topic)
        arms = [r for r in (b, best, f) if r]
        held = frozenset().union(*(r.by_question for r in arms)) - _probe_ids(profile, topic)
        unans = frozenset().union(*(r.unanswerable for r in arms))
        for label, subset in [("answerable", held - unans), ("unanswerable", held & unans)]:
            sb, sw, sf = (_stats(r, subset.__contains__) if r else None for r in (b, best, f))
            n = (sb or sw).n
            if not n:
                continue
            rec = (
                recovery_rate(sb.mean, sw.mean, sf.mean if sf and sf.n else None)
                if label == "answerable" and sb and sw and sw.n else None
            )
            rows.append([
                topic, label, n, _mean_and_s1(sb), _mean_and_s1(sw),
                fmt(sf.mean) if sf and sf.n else "--",
                f"{rec:.0f}%" if rec is not None else "--",
                led.get("best_iteration", "--") if led else "--",
            ])
    parts.append(md_table(
        ["Topic", "Questions", "n", "Blind (mean / score-1)", "Best WiCER", "FC raw",
         "Recovery", "Iter"],
        rows,
    ))
    parts.append(
        "\n> Every probe is left out, so these are questions the loop never saw. "
        "RepLiQA's unanswerable questions (gold: \"The answer is not found in the "
        "document.\") are scored apart and given no recovery: they reward a reader "
        "that abstains, which a smaller context encourages, so a wiki can beat FC raw "
        "on them without having kept anything. With one compilation per arm, a "
        "held-out difference smaller than the gap between two compilations of the "
        "same prompt cannot be told apart from compiler noise.\n"
    )

    # -- Table 5: ablation, on the probe set
    parts.append("\n## Ablation: diagnosed vs. random pinning, probe set (paper Table 5)\n")
    rows: list[list[Any]] = []
    d_wicer, d_wicer_pinned, d_rand, d_rand_pinned = [], [], [], []
    for topic in topics:
        led = ledgers.get(topic)
        rnd_led = load_ledger(profile, topic, "random_pin")
        if not led or not led.get("iterations"):
            continue
        b_w, best_w = _arm_blind_and_best(led)
        d_wicer.append(best_w - b_w)
        row = [topic, fmt(b_w), f"{fmt(best_w)} ({best_w - b_w:+.2f})"]
        pinned_w = _best_pinned(led)
        if pinned_w is None:
            row.append("--")
        else:
            d_wicer_pinned.append(pinned_w - b_w)
            row.append(f"{fmt(pinned_w)} ({pinned_w - b_w:+.2f})")

        if rnd_led and rnd_led.get("iterations"):
            b_r, best_r = _arm_blind_and_best(rnd_led)
            d_rand.append(best_r - b_r)
            row += [fmt(b_r), f"{fmt(best_r)} ({best_r - b_r:+.2f})"]
            pinned_r = _best_pinned(rnd_led)
            if pinned_r is None:
                row.append("--")
            else:
                d_rand_pinned.append(pinned_r - b_r)
                row.append(f"{fmt(pinned_r)} ({pinned_r - b_r:+.2f})")
        else:
            row += ["--", "--", "--"]
        rows.append(row)

    if rows:
        rows.append([
            "**Mean delta vs. own blind**", "--",
            f"{mean(d_wicer):+.2f}" if d_wicer else "--",
            f"{mean(d_wicer_pinned):+.2f}" if d_wicer_pinned else "--", "--",
            f"{mean(d_rand):+.2f}" if d_rand else "--",
            f"{mean(d_rand_pinned):+.2f}" if d_rand_pinned else "--",
        ])
    parts.append(md_table(
        ["Topic", "Blind (WiCER arm)", "WiCER (best-of)", "WiCER (pinned only)",
         "Blind (abl. arm)", "Random pin (best-of)", "Random pin (pinned only)"],
        rows,
    ))
    parts.append(
        "\n> This is the load-bearing comparison (paper Section 7.5): it is the "
        "only thing separating 'diagnosis works' from 'pinning anything works'. "
        "The paper measures +0.16 for random pinning against +0.95 for WiCER.\n"
    )
    parts.append(
        "\n> Each arm compiles its own blind wiki -- paper Section 7.5, 'each topic "
        "uses an independent blind compilation; blind baselines differ slightly "
        "from Table 4 due to LLM compiler non-determinism' -- so every delta here "
        "is measured against the blind in its own column, never the other arm's. "
        "'Best-of' follows the paper's 'best iteration per topic' and so includes "
        "iteration 0, the unpinned blind: an arm whose pinning only ever hurt "
        "therefore reports its own baseline and a delta of exactly +0.00. That is "
        "not a null effect, it is a selection artifact, and it is visible in the "
        "paper too -- seven of its seventeen topics print Random Pin identical to "
        "Blind. The pinned-only columns drop that selection and report what "
        "pinning actually did, for both arms alike -- an ablation that measured "
        "its two arms differently is what produced the incoherent row this "
        "replaced.\n"
    )
    parts.append(
        "\n> One construction difference from the paper: its Table 5 compares the "
        "Table 4 WiCER score against the ablation arm's independent blind, mixing "
        "two compilations in one delta. Here each arm is measured against its own "
        "blind, so our WiCER delta and the paper's +0.95 do not share a "
        "denominator even when the method agrees.\n"
    )

    # -- Table 13: latency
    parts.append("\n## Latency by condition (paper Table 13)\n")
    rows = []
    for cond in CONDITIONS:
        sel = [r for r in results if r.condition == cond and r.timings.n]
        if not sel:
            continue
        rows.append([
            f"{cond} †" if any(r.unpinned_stand_in for r in sel) else cond,
            f"{median([r.n_context_tokens for r in sel if r.n_context_tokens]):,.0f}"
            if any(r.n_context_tokens for r in sel) else "--",
            fmt(median([r.timings.ttft_median for r in sel]), 3),
            fmt(median([r.timings.throughput_median for r in sel]), 1),
        ])
    parts.append(md_table(["Condition", "Context tokens", "TTFT median (s)", "tok/s"], rows))
    if stand_ins:
        parts.append(_stand_in_note(stand_ins))

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
                    "ttft_s", "total_s", "throughput_tok_s", "retrieval_hit", "retrieval_rank",
                    "probe", "unanswerable"])
        probes = {t: _probe_ids(profile, t) for t in {r.topic for r in results}}
        for r in results:
            answers = {a["question_id"]: a for a in read_jsonl(answers_path(profile, r.topic, r.condition))}
            for s in read_jsonl(scores_path(profile, r.topic, r.condition)):
                a = answers.get(s["question_id"], {})
                w.writerow([r.topic, r.condition, s["question_id"], s.get("document_id", ""),
                            s["score"], a.get("ttft_s", ""), a.get("total_s", ""),
                            a.get("throughput_tok_s", ""), a.get("retrieval_hit", ""),
                            a.get("retrieval_rank", ""),
                            int(s["question_id"] in probes[r.topic]),
                            int(s["question_id"] in r.unanswerable)])
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
