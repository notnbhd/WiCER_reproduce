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
