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


def test_reparse_reads_back_verdicts_the_old_parser_recorded_as_failures(
    profile, topic, monkeypatch
):
    """The rejected reply is in the row's own reasoning, so no judge call is needed.

    judge_usage.json is the record of what the run spent; repairing scores must
    leave it alone.  A row that still holds no verdict stays a failure.
    """
    from benchmark.evaluate import main as evaluate_main

    d = run_dir(profile, topic, "fc_raw")
    d.mkdir(parents=True)
    (d / "answers.jsonl").write_text("".join(
        json.dumps({"question_id": q, "document_id": "d", "question": "?", "gold": "g",
                    "generated": "a"}) + "\n" for q in ("q1", "q2", "q3")
    ))
    rows = [
        {"question_id": "q1", "document_id": "d", "score": 4, "reasoning": "fine",
         "judge_model": "m", "parse_failure": False},
        {"question_id": "q2", "document_id": "d", "score": 1,
         "reasoning": 'PARSE FAILURE: {"score": 3, "reasoning": "adds ("quoted") words',
         "judge_model": "m", "parse_failure": True},
        {"question_id": "q3", "document_id": "d", "score": 1,
         "reasoning": "PARSE FAILURE: anthropic/x returned no text (finish_reason='length')",
         "judge_model": "m", "parse_failure": True},
    ]
    scores_path(profile, topic, "fc_raw").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    usage = '{"api_calls": 400, "measured_cost_usd": 0.05}\n'
    (d / "judge_usage.json").write_text(usage)

    monkeypatch.setenv("WICER_RESULTS_DIR", profile.results_dir)
    assert evaluate_main(["--profile", "test", "--topic", topic, "--reparse-failures"]) == 0

    after = {r["question_id"]: r for r in read_jsonl(scores_path(profile, topic, "fc_raw"))}
    assert after["q1"] == rows[0]
    assert (after["q2"]["score"], after["q2"]["parse_failure"]) == (3, False)
    assert after["q2"]["reasoning"] == 'adds ("quoted") words'
    assert after["q2"]["reparsed"] is True
    assert (after["q3"]["score"], after["q3"]["parse_failure"]) == (1, True)
    assert (d / "judge_usage.json").read_text() == usage


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


def test_report_header_names_the_models_that_ran_not_the_live_config(profile, topic):
    """Re-reporting a finished run must not relabel it with the current config.

    The header used to print ``profile.judge.model``, so a Haiku run regenerated
    without the run's ``WICER_*`` overrides exported came out labelled Sonnet --
    and the judge's identity is exactly what the bias correction hangs on.
    """
    corpus = load_topic(profile, topic)
    run_condition(
        profile, topic, "fc_raw", build_fc_context(corpus), corpus.qa[:4], progress=False
    )
    evaluate_condition(profile, topic, "fc_raw", judge=Judge(profile.judge, backend="fake"))
    results = collect(profile, [topic], ["fc_raw"])

    decoy = replace(
        profile,
        judge=replace(profile.judge, model="decoy-judge"),
        compiler=replace(profile.compiler, model="decoy-compiler"),
        reader=replace(profile.reader, model_path="decoy-reader.gguf"),
    )
    text = render(decoy, results, [topic])

    assert "decoy-judge" not in text
    assert "decoy-compiler" not in text
    assert "decoy-reader.gguf" not in text
    assert profile.judge.model in text
    assert "no run metadata on disk" not in text


def test_report_header_marks_an_unrun_tree_as_config_default(profile, topic):
    """With nothing on disk the live config is all there is -- but say so."""
    text = render(profile, [], [topic])
    assert "no run metadata on disk" in text
    assert profile.judge.model in text


def _ledger(topic: str, name: str, pairs: list[tuple[float, int]]) -> dict:
    """A minimal WiCER ledger: (probe_mean, n_pinned_facts) per iteration."""
    return {
        "topic": topic,
        "iterations": [
            {
                "iteration": i,
                "wiki_name": f"{name}_iter{i}" if i else f"blind_{name}",
                "probe_mean": mean_,
                "probe_score_1_pct": 10.0,
                "n_failures": 1,
                "n_pinned_facts": pinned,
                "pinned_words": 100 * pinned,
                "actual_ratio_words": 0.15,
            }
            for i, (mean_, pinned) in enumerate(pairs)
        ],
    }


def test_ablation_shows_what_pinning_did_not_just_the_best_of_selection(profile, topic):
    """Random pinning that hurt must not read as a flat +0.00 and nothing more.

    ``max`` over iterations includes iteration 0, the unpinned blind, so an arm
    whose pinning only ever made things worse selects its own baseline.  The
    paper's Table 5 is built the same way, so the best-of column stays for
    comparability -- but the pinned-only column has to surface the loss it
    discards.  Each delta must also use its own arm's blind: the row used to
    print the WiCER arm's blind beside a random-pin delta.
    """
    from wicer_core.records import ledger_path

    p = ledger_path(profile, topic, "wicer")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_ledger(topic, "wicer", [(3.00, 0), (3.50, 10)])))
    ledger_path(profile, topic, "random_pin").write_text(
        json.dumps(_ledger(topic, "random_pin", [(3.20, 0), (3.06, 10)]))
    )

    text = render(profile, collect(profile, [topic], ["fc_raw"]), [topic])
    abl = text.split("## Ablation")[1].split("## Latency")[0]

    assert "3.50 (+0.50)" in abl        # WiCER against ITS blind, 3.00
    assert "3.20 (+0.00)" in abl        # best-of, exactly as the paper computes it
    assert "3.06 (-0.14)" in abl        # ... and what pinning actually did
    assert "selection artifact" in abl
    # 7 columns: topic, then blind/best-of/pinned-only for each of the two arms.
    header = next(ln for ln in abl.splitlines() if ln.startswith("| Topic"))
    assert header.count("|") == 8, header


def test_ablation_exposes_a_wicer_regression_the_same_way(profile, topic):
    """The artifact is symmetric, so the WiCER arm gets the same treatment.

    A topic where diagnosed pinning also only hurt (the paper has one, with 0%
    recovery) would otherwise report +0.00 for WiCER and look like a no-op.
    """
    from wicer_core.records import ledger_path

    p = ledger_path(profile, topic, "wicer")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_ledger(topic, "wicer", [(3.00, 0), (2.70, 10)])))
    ledger_path(profile, topic, "random_pin").write_text(
        json.dumps(_ledger(topic, "random_pin", [(3.00, 0), (2.80, 10)]))
    )

    abl = render(profile, collect(profile, [topic], ["fc_raw"]), [topic])
    abl = abl.split("## Ablation")[1].split("## Latency")[0]
    assert "3.00 (+0.00)" in abl        # both best-of columns select the blind
    assert "2.70 (-0.30)" in abl        # WiCER's real effect, no longer hidden
    assert "2.80 (-0.20)" in abl        # random pin's


# -- WiCER tables on the right question sets ----------------------------------
#
# A synthetic topic with literal scores, so every expected number below is
# worked by hand: p1/p2 are the probes, h1/h2 held-out answerable questions,
# u1/u2 RepLiQA's unanswerable ones.

_UNANSWERABLE = "The answer is not found in the document."
_GOLD = {"p1": "Ilse Hallam.", "p2": "In 1890.", "h1": "The Suden River.",
         "h2": "Six terms.", "u1": _UNANSWERABLE, "u2": _UNANSWERABLE}
_SCORES = {  #                       p1 p2 h1 h2 u1 u2
    "fc_raw":        dict(zip(_GOLD, (5, 5, 4, 4, 1, 2))),
    "wiki_moderate": dict(zip(_GOLD, (4, 3, 2, 2, 3, 3))),
    "wicer_iter1":   dict(zip(_GOLD, (5, 5, 2, 1, 2, 3))),
}


def _write_condition(profile, topic, condition, scores, context=None):
    d = run_dir(profile, topic, condition)
    d.mkdir(parents=True, exist_ok=True)
    (d / "answers.jsonl").write_text("".join(
        json.dumps({"question_id": q, "document_id": f"doc-{q}", "question": f"{q}?",
                    "gold": _GOLD[q], "generated": "an answer", "ttft_s": 0.1,
                    "total_s": 0.5, "throughput_tok_s": 50.0,
                    "n_generated_tokens": 10}) + "\n"
        for q in scores
    ))
    scores_path(profile, topic, condition).write_text("".join(
        json.dumps({"question_id": q, "document_id": f"doc-{q}", "score": s,
                    "reasoning": "", "judge_model": "fake", "parse_failure": False}) + "\n"
        for q, s in scores.items()
    ))
    if context is not None:
        (d / "context.txt").write_text(context)


def _synthetic_wicer_topic(profile, topic):
    from wicer.probe import probe_paths
    from wicer_core.records import ledger_path

    for condition, scores in _SCORES.items():
        _write_condition(profile, topic, condition, scores)
    probe_paths(profile, topic)[0].write_text(json.dumps([
        {"question_id": q, "document_id": f"doc-{q}", "question": f"{q}?",
         "answer": _GOLD[q], "long_answer": ""} for q in ("p1", "p2")
    ]))
    # The loop's own probe runs: blind 3.0, then 4.0 once two facts were pinned.
    led = _ledger(topic, "wicer", [(3.0, 0), (4.0, 2)])
    led["best_wiki_name"] = "wicer_iter1"
    ledger_path(profile, topic, "wicer").write_text(json.dumps(led))


def _section(text: str, heading: str) -> str:
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def _table(section: str) -> list[dict[str, str]]:
    """The first markdown table in a report section, one dict per body row."""
    lines = [ln for ln in section.splitlines() if ln.startswith("|")]
    cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    return [dict(zip(cells[0], row)) for row in cells[2:]]


def _render_synthetic(profile, topic):
    _synthetic_wicer_topic(profile, topic)
    return render(profile, collect(profile, [topic], list(_SCORES)), [topic])


def test_probe_recovery_divides_by_full_context_on_the_same_probes(profile, topic):
    """FC over every question carries ones the probes never include.

    FC scores 5.0 on these probes but 3.5 overall, so dividing by the overall
    figure, as the paper's Table 4 does, would print (4 - 3) / (3.5 - 3) = 200%.
    """
    text = _render_synthetic(profile, topic)
    row = _table(_section(text, "## WiCER across topics, probe set"))[0]
    assert row["FC raw (probes)"] == "5.00"
    assert row["Recovery"] == "50%"                      # (4.0 - 3.0) / (5.0 - 3.0)


def test_held_out_excludes_the_probes_and_scores_unanswerable_questions_apart(profile, topic):
    text = _render_synthetic(profile, topic)
    rows = {r["Questions"]: r for r in _table(_section(text, "## WiCER held-out"))}

    ans = rows["answerable"]
    assert ans["n"] == "2"                               # h1 and h2, never p1 or p2
    assert ans["Blind (mean / score-1)"] == "2.00 / 0.0%"
    assert ans["Best WiCER"] == "1.50 / 50.0%"
    assert ans["FC raw"] == "4.00"
    assert ans["Recovery"] == "-25%"                     # (1.5 - 2.0) / (4.0 - 2.0)

    un = rows["unanswerable"]
    assert (un["n"], un["Blind (mean / score-1)"], un["Best WiCER"], un["FC raw"]) == (
        "2", "3.00 / 0.0%", "2.50 / 0.0%", "1.50")
    # The wiki beats FC here by abstaining more; there is no lost quality to recover.
    assert un["Recovery"] == "--"


def test_held_out_scores_the_wiki_the_loop_chose_not_the_best_held_out_one(profile, topic):
    """Choosing among full-set conditions by their score would select on the test set."""
    _synthetic_wicer_topic(profile, topic)
    # A later iteration that does far better on held-out questions; the loop,
    # judging on its probes, still chose wicer_iter1.
    _write_condition(profile, topic, "wicer_iter2", dict(zip(_GOLD, (1, 1, 5, 5, 5, 5))))
    text = render(profile, collect(profile, [topic], [*_SCORES, "wicer_iter2"]), [topic])
    rows = {r["Questions"]: r for r in _table(_section(text, "## WiCER held-out"))}
    assert rows["answerable"]["Best WiCER"] == "1.50 / 50.0%"    # wicer_iter1, not 5.00


def test_compilation_table_also_reports_the_gap_on_answerable_questions(profile, topic):
    text = _render_synthetic(profile, topic)
    rows = {r["Level"]: r for r in _table(_section(text, "## Compilation"))}
    assert rows["FC raw (80 docs)"]["Quality (answerable)"] == "4.50"
    assert rows["Wiki-moderate"]["Quality"] == "2.83"    # the paper's column, unchanged
    assert rows["Wiki-moderate"]["Quality (answerable)"] == "2.75"
    assert rows["Wiki-moderate"]["vs. FC raw (answerable)"] == "-1.75"


def test_per_question_csv_marks_probes_and_unanswerable_questions(profile, topic):
    import csv

    _synthetic_wicer_topic(profile, topic)
    _, per_question = write_csvs(profile, collect(profile, [topic], list(_SCORES)))
    rows = {(r["condition"], r["question_id"]): r for r in csv.DictReader(per_question.open())}
    assert (rows["fc_raw", "p1"]["probe"], rows["fc_raw", "p1"]["unanswerable"]) == ("1", "0")
    assert (rows["fc_raw", "h1"]["probe"], rows["fc_raw", "h1"]["unanswerable"]) == ("0", "0")
    assert (rows["fc_raw", "u1"]["probe"], rows["fc_raw", "u1"]["unanswerable"]) == ("0", "1")


@pytest.mark.parametrize("served,labelled", [
    ("random_blind_moderate", True), ("random_pin_iter1", False),
])
def test_a_random_pin_row_says_when_it_served_the_unpinned_blind(
    profile, topic, served, labelled
):
    """--eval-full evaluates the arm's best iteration, which can be its blind.

    In both pilots it was, so the row labelled random pinning was a second
    blind compilation of the moderate prompt.
    """
    import csv

    from wicer_core.records import wiki_paths

    for name, n_pinned in [("random_blind_moderate", 0), ("random_pin_iter1", 3)]:
        md, meta = wiki_paths(profile, topic, name)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(f"# {name}\n\nCompiled body.")
        meta.write_text(json.dumps({"n_preserved_facts": n_pinned}))
    context = build_wiki_context(wiki_paths(profile, topic, served)[0].read_text())
    _write_condition(profile, topic, "random_pin", _SCORES["wiki_moderate"], context=context)

    results = collect(profile, [topic], ["random_pin"])
    text = render(profile, results, [topic])
    for heading in ("## Per topic and condition", "## Latency by condition"):
        assert ("random_pin †" in _section(text, heading)) is labelled
    assert ("random_blind_moderate" in text) is labelled

    per_topic, _ = write_csvs(profile, results)
    assert next(csv.DictReader(per_topic.open()))["served_wiki"] == served


def test_ablation_reports_no_pinned_column_when_nothing_was_pinned(profile, topic):
    """An arm that never got past its blind has no pinned result to report."""
    from wicer_core.records import ledger_path

    p = ledger_path(profile, topic, "wicer")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_ledger(topic, "wicer", [(3.00, 0), (3.50, 10)])))
    ledger_path(profile, topic, "random_pin").write_text(
        json.dumps(_ledger(topic, "random_pin", [(3.20, 0)]))
    )

    abl = render(profile, collect(profile, [topic], ["fc_raw"]), [topic])
    abl = abl.split("## Ablation")[1].split("## Latency")[0]
    assert "3.20 (+0.00)" in abl
    assert "| --" in abl
