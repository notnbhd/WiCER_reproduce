"""TACT command line: the offline end-to-end path, and the held-out comparison."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from wicer_core.corpus import is_not_in_document, load_topic
from wicer_core.records import read_jsonl, scores_path
from tact import run
from tact.compare import compare_topic


def _scores(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_compare_scores_every_arm_on_the_sweeps_held_out_questions(profile, topic, tmp_path):
    qa = load_topic(profile, topic).qa
    probes = {}
    for q in qa:
        probes.setdefault(q.document_id, q)
    base = tmp_path / "sweep"
    (base / topic).mkdir(parents=True)
    (base / topic / "probes.json").write_text(json.dumps(
        [{"question_id": q.question_id} for q in probes.values()]))
    (base / topic / "wicer_ledger.json").write_text(json.dumps({"iterations": [
        {"iteration": 0, "wiki_name": "blind_moderate", "n_pinned_facts": 0, "probe_mean": 2.0},
        {"iteration": 1, "wiki_name": "wicer_iter1", "n_pinned_facts": 3, "probe_mean": 3.0},
    ]}))
    sweep = replace(profile, results_dir=str(base))
    for cond, score in [("wiki_moderate", 2), ("wicer_iter1", 3), ("fc_raw", 5)]:
        _scores(scores_path(sweep, topic, cond),
                [{"question_id": q.question_id, "score": score} for q in qa])
    # TACT scores probes 1 and held-out questions 4: probes must not count.
    probe_ids = {q.question_id for q in probes.values()}
    _scores(scores_path(profile, topic, "tact_iter0"),
            [{"question_id": q.question_id, "score": 1 if q.question_id in probe_ids else 4}
             for q in qa])

    rows = {r["arm"]: r for r in compare_topic(profile, topic, base)}
    assert list(rows) == ["blind (sweep)", "wicer best (sweep)", "fc_raw (sweep)", "tact_iter0"]
    held = [q for q in qa if q.question_id not in probe_ids]
    n_ans = sum(not is_not_in_document(q.answer) for q in held)
    t = rows["tact_iter0"]
    assert t["n_answerable"] == n_ans and t["mean_answerable"] == 4.0
    assert t["delta_vs_blind"] == 2.0 and t["recovery"] == pytest.approx(2 / 3)
    assert rows["wicer best (sweep)"]["condition"] == "wicer_iter1"


def test_offline_pipeline_extracts_renders_truncates_and_scores(profile, topic):
    argv = ["--profile", "test", "--topic", topic]
    assert run.main(["extract", *argv], profile=profile) == 0
    assert run.main(["render", *argv, "--name", "tact_iter0", "--rho", "0.1"], profile=profile) == 0
    assert run.main(["truncate", *argv], profile=profile) == 0
    for cond in ("tact_iter0", "truncate"):
        assert run.main(["score", *argv, "--condition", cond], profile=profile) == 0
        rows = read_jsonl(scores_path(profile, topic, cond))
        assert len(rows) == len(load_topic(profile, topic).qa)
