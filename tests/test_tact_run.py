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


def test_score_rejudges_rather_than_keeping_old_scores(profile, topic):
    argv = ["--profile", "test", "--topic", topic]
    run.main(["extract", *argv], profile=profile)
    run.main(["truncate", *argv], profile=profile)
    stale = [{"question_id": q.question_id, "document_id": q.document_id, "score": 5,
              "reasoning": "STALE", "judge_model": "x"} for q in load_topic(profile, topic).qa]
    _scores(scores_path(profile, topic, "truncate"), stale)
    assert run.main(["score", *argv, "--condition", "truncate"], profile=profile) == 0
    rows = read_jsonl(scores_path(profile, topic, "truncate"))
    assert rows and not any(r["reasoning"] == "STALE" for r in rows)


def test_displacement_splits_held_out_change_by_pinned_documents(profile, topic, tmp_path):
    from tact.compare import displacement
    from tact.loop import loop_dir

    qa = load_topic(profile, topic).qa
    docs = sorted({q.document_id for q in qa})
    probes = {}
    for q in qa:
        probes.setdefault(q.document_id, q)
    base = tmp_path / "sweep"
    (base / topic).mkdir(parents=True)
    (base / topic / "probes.json").write_text(json.dumps(
        [{"question_id": q.question_id} for q in probes.values()]))
    # WiCER pinned docs[0] at iteration 0, scored in wicer_iter1 (its best);
    # a fact diagnosed at iteration 1 never reached that wiki.
    (base / topic / "wicer_ledger.json").write_text(json.dumps({
        "best_iteration": 1,
        "iterations": [
            {"iteration": 0, "wiki_name": "blind_moderate", "n_pinned_facts": 0, "probe_mean": 2.0},
            {"iteration": 1, "wiki_name": "wicer_iter1", "n_pinned_facts": 1, "probe_mean": 3.0}],
        "preserved_facts": [{"document_id": docs[0], "iteration": 0},
                            {"document_id": docs[1], "iteration": 1}]}))
    sweep = replace(profile, results_dir=str(base))

    def score(prof, cond, pinned_docs, pinned, unpinned):
        _scores(scores_path(prof, topic, cond), [
            {"question_id": q.question_id,
             "score": pinned if q.document_id in pinned_docs else unpinned} for q in qa])

    score(sweep, "wiki_moderate", {docs[0]}, 2, 3)
    score(sweep, "wicer_iter1", {docs[0]}, 4, 2)          # +2 pinned, -1 unpinned
    loop_dir(profile, topic).mkdir(parents=True, exist_ok=True)
    (loop_dir(profile, topic) / "loop.json").write_text(json.dumps(
        {"start": "tact_iter0", "pinned_docs": docs[:3]}))
    score(profile, "tact_iter0", set(docs[:3]), 3, 3)
    score(profile, "tact_loop", set(docs[:3]), 4, 3)      # +1 pinned, 0 unpinned

    rows = {r["method"]: r for r in displacement(profile, topic, base)}
    w, t = rows["WiCER (sweep)"], rows["TACT"]
    assert (w["n_pinned_docs"], w["delta_pinned"], w["delta_unpinned"]) == (1, 2.0, -1.0)
    assert (t["n_pinned_docs"], t["delta_pinned"], t["delta_unpinned"]) == (3, 1.0, 0.0)


def test_offline_milestone_b_probes_loop_and_score(profile, topic):
    from tact.loop import loop_dir
    from tact.probes import probe_dir

    argv = ["--profile", "test", "--topic", topic]
    run.main(["extract", *argv], profile=profile)
    run.main(["render", *argv, "--name", "tact_iter0", "--rho", "0.1"], profile=profile)
    assert run.main(["probes", *argv, "--m", "4"], profile=profile) == 0
    assert (probe_dir(profile, topic) / "probes.jsonl").exists()
    assert run.main(["loop", *argv, "--max-iter", "2"], profile=profile) == 0
    assert json.loads((loop_dir(profile, topic) / "loop.json").read_text())["iterations"]
    assert run.main(["score", *argv, "--condition", "tact_loop"], profile=profile) == 0
