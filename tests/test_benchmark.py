"""Benchmark runner, evaluation and reporting."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from wicer_core.corpus import load_topic
from wicer_core.judge import Judge
from wicer_core.reader import ContextOverflow
from wicer_core.records import answers_path, read_jsonl, run_dir, scores_path, skip_path
from benchmark.evaluate import evaluate_condition
from benchmark.report import collect, md_table, render, write_csvs
from benchmark.run_benchmark import build_fc_context, build_wiki_context, main, run_condition


def test_full_context_prompt_contains_every_document(profile, topic):
    corpus = load_topic(profile, topic)
    context = build_fc_context(corpus)
    for doc in corpus.documents:
        assert doc.document_id in context
    assert context.count("=== DOCUMENT") == len(corpus.documents)


def test_wiki_and_full_context_share_the_same_template(profile, topic):
    """Only the knowledge differs between conditions; the framing must not."""
    corpus = load_topic(profile, topic)
    preamble = build_fc_context(corpus).split("=== DOCUMENT")[0]
    assert build_wiki_context("some wiki text").startswith(preamble)


def test_run_condition_writes_answers_context_and_provenance(profile, topic):
    corpus = load_topic(profile, topic)
    summary = run_condition(
        profile, topic, "fc_raw", build_fc_context(corpus), corpus.qa[:5], progress=False
    )
    assert summary["n_generated"] == 5
    assert summary["n_context_tokens"] > 0

    d = run_dir(profile, topic, "fc_raw")
    assert (d / "context.txt").exists()
    meta = json.loads((d / "run_meta.json").read_text())
    assert meta["command"] == "benchmark/run_benchmark.py"
    assert meta["judge_model_configured"] == profile.judge.model
    assert meta["reader"]["cache_type_k"] == "q8_0"

    rows = read_jsonl(answers_path(profile, topic, "fc_raw"))
    assert len(rows) == 5
    assert all(r["ttft_s"] > 0 and r["total_s"] >= r["ttft_s"] for r in rows)
    assert all(r["generated"] for r in rows)


def test_resume_only_generates_the_missing_questions(profile, topic):
    corpus = load_topic(profile, topic)
    context = build_fc_context(corpus)
    run_condition(profile, topic, "fc_raw", context, corpus.qa[:3], progress=False)
    summary = run_condition(
        profile, topic, "fc_raw", context, corpus.qa[:6], resume=True, progress=False
    )
    assert summary["n_resumed"] == 3
    assert summary["n_generated"] == 3
    assert len(read_jsonl(answers_path(profile, topic, "fc_raw"))) == 6


def test_oversized_corpus_raises_rather_than_silently_truncating(profile, topic):
    corpus = load_topic(profile, topic)
    tiny = replace(profile, reader=replace(profile.reader, ctx_margin_tokens=8_000))
    with pytest.raises(ContextOverflow):
        run_condition(tiny, topic, "fc_raw", build_fc_context(corpus), corpus.qa[:1], progress=False)


def test_cli_records_a_skip_instead_of_failing_the_sweep(profile, topic, monkeypatch):
    """A topic that overflows must not abort the other sixteen."""
    monkeypatch.setenv("WICER_RESULTS_DIR", profile.results_dir)
    monkeypatch.setenv("WICER_READER_PORT", str(profile.reader.port))
    monkeypatch.setenv("WICER_N_CTX", "600")  # far too small for the fixture
    rc = main(["--profile", "test", "--topic", topic, "--condition", "fc_raw"])
    assert rc == 0

    skip = json.loads(skip_path(profile, topic, "fc_raw").read_text())
    assert skip["skipped"] == "context_overflow"
    assert skip["n_context_tokens"] > skip["limit"]


def test_evaluate_scores_every_answer(profile, topic):
    corpus = load_topic(profile, topic)
    run_condition(profile, topic, "fc_raw", build_fc_context(corpus), corpus.qa[:6], progress=False)
    res = evaluate_condition(profile, topic, "fc_raw", judge=Judge(profile.judge, backend="fake"))
    assert res["n_scored"] == 6

    scores = read_jsonl(scores_path(profile, topic, "fc_raw"))
    assert len(scores) == 6
    assert all(1 <= s["score"] <= 5 for s in scores)
    assert all(s["judge_model"] == "fake" for s in scores)
    assert (run_dir(profile, topic, "fc_raw") / "judge_usage.json").exists()


def test_evaluate_is_idempotent_via_resume(profile, topic):
    corpus = load_topic(profile, topic)
    run_condition(profile, topic, "fc_raw", build_fc_context(corpus), corpus.qa[:4], progress=False)
    judge = Judge(profile.judge, backend="fake")
    evaluate_condition(profile, topic, "fc_raw", judge=judge)
    second = evaluate_condition(profile, topic, "fc_raw", judge=judge)
    assert second["n_scored"] == 0
    assert second["n_resumed"] == 4
    assert len(read_jsonl(scores_path(profile, topic, "fc_raw"))) == 4


def test_evaluate_on_missing_answers_is_a_no_op(profile, topic):
    res = evaluate_condition(profile, topic, "fc_raw", judge=Judge(profile.judge, backend="fake"))
    assert res["n_scored"] == 0


def test_md_table_renders_and_handles_emptiness():
    out = md_table(["A", "B"], [["1", "2"], ["3", "4"]])
    assert out.splitlines()[0] == "| A | B |"
    assert "| 1 | 2 |" in out
    assert "(no data)" in md_table(["A"], [])


def test_report_collects_and_renders(profile, topic):
    corpus = load_topic(profile, topic)
    run_condition(profile, topic, "fc_raw", build_fc_context(corpus), corpus.qa[:6], progress=False)
    evaluate_condition(profile, topic, "fc_raw", judge=Judge(profile.judge, backend="fake"))

    results = collect(profile, [topic], ["fc_raw", "rag_top5"])
    assert len(results) == 1
    r = results[0]
    assert r.condition == "fc_raw" and r.scores.n == 6
    assert r.n_context_tokens and r.n_context_tokens > 0
    assert r.retrieval is None

    text = render(profile, results, [topic])
    assert "# WiCER reproduction report" in text
    assert "Success criteria" in text
    assert topic in text

    per_topic, per_question = write_csvs(profile, results)
    assert per_topic.exists() and per_question.exists()
    assert len(per_question.read_text().splitlines()) == 7  # header + 6 rows


def test_report_surfaces_skipped_conditions(profile, topic):
    from wicer_core.records import write_skip

    write_skip(profile, topic, "fc_raw", "context_overflow", n_context_tokens=99, limit=10)
    results = collect(profile, [topic], ["fc_raw"])
    assert results and results[0].skipped == "context_overflow"
    assert "context_overflow" in render(profile, results, [topic])
