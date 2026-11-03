"""TACT's arms against the reproduction, on the questions WiCER never saw.

Draft section 3.9: every arm is scored on the benchmark questions the sweep's
WiCER loop did not use as probes, answerable and unanswerable apart.  The
baseline rows (blind compile, best pinned WiCER iteration, full context) are
read from the frozen sweep directory; the TACT rows are every condition with
scores in TACT's own results directory.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from wicer_core.config import Profile, resolve
from wicer_core.corpus import is_not_in_document, load_topic
from wicer_core.metrics import mean, recovery_rate
from wicer_core.records import read_jsonl, scores_path, wiki_paths
from wicer.loop import best_pinned_iteration

BASELINE_DIR = "result-sweep"


def _scores(profile: Profile, topic: str, condition: str) -> dict[str, int]:
    path = scores_path(profile, topic, condition)
    if not path.exists():
        return {}
    return {r["question_id"]: r["score"] for r in read_jsonl(path)
            if r.get("score") is not None and not r.get("parse_failure")}


def _ratio(profile: Profile, topic: str, wiki: str | None) -> float | None:
    if wiki is None:
        return None
    meta = wiki_paths(profile, topic, wiki)[1]
    if not meta.exists():
        return None
    return json.loads(meta.read_text()).get("actual_ratio_tokens")


def compare_topic(
    profile: Profile, topic: str, baseline_dir: str | Path = BASELINE_DIR
) -> list[dict[str, Any]]:
    base = replace(profile, results_dir=str(baseline_dir))
    base_root = resolve(baseline_dir) / topic
    probe_ids = {q["question_id"] for q in json.loads((base_root / "probes.json").read_text())}
    qa = [q for q in load_topic(profile, topic).qa if q.question_id not in probe_ids]
    answerable = {q.question_id for q in qa if not is_not_in_document(q.answer)}
    unanswerable = {q.question_id for q in qa} - answerable

    arms: list[tuple[str, Profile, str, str | None]] = [
        ("blind (sweep)", base, "wiki_moderate", "blind_moderate")]
    ledger = base_root / "wicer_ledger.json"
    if ledger.exists():
        best = best_pinned_iteration(json.loads(ledger.read_text())["iterations"])
        if best is not None:
            arms.append(("wicer best (sweep)", base, best["wiki_name"], best["wiki_name"]))
    arms.append(("fc_raw (sweep)", base, "fc_raw", None))
    own = resolve(profile.results_dir) / topic
    for p in sorted(own.glob("*/scores.jsonl")):
        arms.append((p.parent.name, profile, p.parent.name, p.parent.name))

    rows = []
    for arm, prof, condition, wiki in arms:
        s = _scores(prof, topic, condition)
        ans = [v for q, v in s.items() if q in answerable]
        una = [v for q, v in s.items() if q in unanswerable]
        rows.append({
            "arm": arm, "condition": condition,
            "n_answerable": len(ans),
            "mean_answerable": mean(ans) if ans else None,
            "score1_answerable": sum(v <= 1 for v in ans) / len(ans) if ans else None,
            "n_unanswerable": len(una),
            "mean_unanswerable": mean(una) if una else None,
            "ratio_tokens": _ratio(prof, topic, wiki),
        })
    blind = rows[0]["mean_answerable"]
    fc = next((r["mean_answerable"] for r in rows if r["condition"] == "fc_raw"), None)
    for r in rows:
        m = r["mean_answerable"]
        known = m is not None and blind is not None
        r["delta_vs_blind"] = m - blind if known else None
        rec = recovery_rate(blind, m, fc) if known and r["condition"] != "fc_raw" else None
        r["recovery"] = rec / 100 if rec is not None and r is not rows[0] else None
    return rows


def _f(x: float | None, fmt: str = "{:.2f}") -> str:
    return "--" if x is None else fmt.format(x)


def render_table(topic: str, rows: list[dict[str, Any]]) -> str:
    out = [f"### {topic}: held-out questions (WiCER's probes excluded)\n",
           "| Arm | n ans | Answerable mean | score-1 | Δ vs blind | Recovery | "
           "n unans | Unanswerable mean | Tokens/source |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['arm']} | {r['n_answerable']} | {_f(r['mean_answerable'])} | "
            f"{_f(r['score1_answerable'], '{:.0%}')} | {_f(r['delta_vs_blind'], '{:+.2f}')} | "
            f"{_f(r['recovery'], '{:.0%}')} | {r['n_unanswerable']} | "
            f"{_f(r['mean_unanswerable'])} | {_f(r['ratio_tokens'], '{:.1%}')} |"
        )
    return "\n".join(out) + "\n"


# -- displacement: held-out change split by pinned documents ---------------------


def _split_delta(
    before: dict[str, int], after: dict[str, int], questions: dict[str, str], pinned: set[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, keep in (("pinned", True), ("unpinned", False)):
        ids = [q for q, doc in questions.items()
               if (doc in pinned) == keep and q in before and q in after]
        out[f"n_{label}_questions"] = len(ids)
        out[f"before_{label}"] = mean([float(before[q]) for q in ids]) if ids else None
        out[f"after_{label}"] = mean([float(after[q]) for q in ids]) if ids else None
        out[f"delta_{label}"] = (out[f"after_{label}"] - out[f"before_{label}"]) if ids else None
    return out


def displacement(
    profile: Profile, topic: str, baseline_dir: str | Path = BASELINE_DIR
) -> list[dict[str, Any]]:
    """Held-out answerable change from each method's start to its result, by
    whether the question's document had anything pinned (draft section 3.7).

    WiCER: blind -> best pinned iteration; a document is pinned when a fact
    from it was preserved in that wiki.  TACT: its loop's start wiki ->
    ``tact_loop``; pinned documents come from ``loop.json``.
    """
    from tact.loop import FINAL_WIKI, loop_dir

    base = replace(profile, results_dir=str(baseline_dir))
    base_root = resolve(baseline_dir) / topic
    probe_ids = {q["question_id"] for q in json.loads((base_root / "probes.json").read_text())}
    questions = {q.question_id: q.document_id for q in load_topic(profile, topic).qa
                 if q.question_id not in probe_ids and not is_not_in_document(q.answer)}
    rows = []
    ledger = base_root / "wicer_ledger.json"
    if ledger.exists():
        led = json.loads(ledger.read_text())
        best = best_pinned_iteration(led["iterations"])
        if best is not None:
            pinned = {f["document_id"] for f in led.get("preserved_facts", [])
                      if f["iteration"] < best["iteration"]}
            rows.append({"method": "WiCER (sweep)", "start": "wiki_moderate",
                         "result": best["wiki_name"], "n_pinned_docs": len(pinned),
                         **_split_delta(_scores(base, topic, "wiki_moderate"),
                                        _scores(base, topic, best["wiki_name"]),
                                        questions, pinned)})
    loop = loop_dir(profile, topic) / "loop.json"
    if loop.exists():
        state = json.loads(loop.read_text())
        pinned = set(state["pinned_docs"])
        rows.append({"method": "TACT", "start": state["start"], "result": FINAL_WIKI,
                     "n_pinned_docs": len(pinned),
                     **_split_delta(_scores(profile, topic, state["start"]),
                                    _scores(profile, topic, FINAL_WIKI), questions, pinned)})
    return rows


def render_displacement(topic: str, rows: list[dict[str, Any]]) -> str:
    out = [f"### {topic}: held-out answerable change, by pinned documents\n",
           "| Method | Start → result | Pinned docs | Δ pinned (n) | Δ unpinned (n) |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['method']} | {r['start']} → {r['result']} | {r['n_pinned_docs']} | "
            f"{_f(r['delta_pinned'], '{:+.2f}')} ({r['n_pinned_questions']}) | "
            f"{_f(r['delta_unpinned'], '{:+.2f}')} ({r['n_unpinned_questions']}) |")
    return "\n".join(out) + "\n"
