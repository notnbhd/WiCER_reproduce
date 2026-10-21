"""WiCER: probe selection, compilation measurement, diagnosis, the loop."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from wicer_core.corpus import QAPair, load_topic
from wicer_core.judge import Judge
from wicer_core.reader import ReaderClient
from wicer.compile import CompileTruncated, Compiler, PreservedFact, compile_topic
from wicer.diagnose import Diagnoser, _clip_words
from wicer.loop import (
    BLIND_WIKI, LoopResult, best_pinned_iteration, evaluate_wikis_full, full_set_wikis, run_loop,
)
from wicer.probe import split_probes
from wicer.random_pin import RandomPinner, run_random_pin


# -- probe selection ---------------------------------------------------------


def test_exactly_one_probe_per_source_document(profile, topic):
    corpus = load_topic(profile, topic)
    probes, heldout = split_probes(corpus.qa, profile.wicer.probe_seed)
    doc_ids = {d.document_id for d in corpus.documents}
    assert len(probes) == len(doc_ids)
    assert {p.document_id for p in probes} == doc_ids
    assert len(probes) + len(heldout) == len(corpus.qa)


def test_probes_and_heldout_are_disjoint(profile, topic):
    probes, heldout = split_probes(load_topic(profile, topic).qa, profile.wicer.probe_seed)
    assert not ({p.question_id for p in probes} & {h.question_id for h in heldout})


def test_probe_selection_is_order_independent(profile, topic):
    qa = load_topic(profile, topic).qa
    a, _ = split_probes(qa, profile.wicer.probe_seed)
    b, _ = split_probes(list(reversed(qa)), profile.wicer.probe_seed)
    assert [q.question_id for q in a] == [q.question_id for q in b]


def _documents(n_docs: int) -> list[QAPair]:
    """``n_docs`` documents with five questions each; the fifth is unanswerable."""
    return [
        QAPair(
            question_id=f"d{d:04d}-q{i}",
            document_id=f"d{d:04d}",
            question=f"question {i}?",
            answer="The answer is not found in the document." if i == 5 else f"answer {i}",
        )
        for d in range(n_docs)
        for i in range(1, 6)
    ]


def test_each_probe_is_drawn_uniformly_from_its_documents_five_questions():
    """Always taking the first question never probed an unanswerable one and
    favoured questions about a document's main subject, which a compiler keeps."""
    probes, _ = split_probes(_documents(1000), seed=20261021)
    positions = Counter(p.question_id.rsplit("-q", 1)[1] for p in probes)
    assert sorted(positions) == ["1", "2", "3", "4", "5"]
    # 200 expected per position; 150-250 is about +/-4 standard deviations.
    assert all(150 <= n <= 250 for n in positions.values()), positions


def test_a_different_seed_draws_different_probes():
    qa = _documents(50)
    first, _ = split_probes(qa, seed=1)
    second, _ = split_probes(qa, seed=2)
    assert [p.question_id for p in first] != [p.question_id for p in second]


def test_build_draws_with_the_profiles_probe_seed(profile, topic):
    from dataclasses import replace

    from wicer.probe import build

    a, _ = build(profile, topic)
    other = replace(profile, wicer=replace(profile.wicer, probe_seed=profile.wicer.probe_seed + 1))
    b, _ = build(other, topic)
    assert [p.question_id for p in a] != [p.question_id for p in b]


# -- compilation -------------------------------------------------------------


def test_compilation_measures_the_actual_ratio_not_the_target(profile, topic):
    corpus = load_topic(profile, topic)
    compiler = Compiler(profile.compiler, backend="fake")
    result = compiler.compile(corpus.documents, 0.5, topic=topic)
    assert result.target_ratio == 0.5
    assert result.source_words == corpus.n_words
    assert result.actual_words > 0
    # The ratio is computed from the output, never assumed from the target --
    # the paper's compiler overshoots 2-3x (Section 6.2).
    assert result.actual_ratio_words == pytest.approx(
        result.actual_words / result.source_words
    )


def test_compilation_records_reader_token_ratio_when_a_server_is_up(profile, topic):
    reader = ReaderClient(profile.reader)
    corpus = load_topic(profile, topic)
    compiler = Compiler(profile.compiler, backend="fake")
    r = compiler.compile(corpus.documents, 0.5, topic=topic, count_tokens=reader.count_tokens)
    assert r.source_tokens and r.actual_tokens
    assert r.actual_ratio_tokens == pytest.approx(r.actual_tokens / r.source_tokens)


def test_compilation_without_a_tokeniser_leaves_token_ratios_unset(profile, topic):
    compiler = Compiler(profile.compiler, backend="fake")
    r = compiler.compile(load_topic(profile, topic).documents, 0.5)
    assert r.source_tokens is None and r.actual_ratio_tokens is None
    assert not math.isnan(r.actual_ratio_words)


def test_preserved_facts_appear_verbatim_in_the_next_compilation(profile, topic):
    corpus = load_topic(profile, topic)
    compiler = Compiler(profile.compiler, backend="fake")
    fact = PreservedFact("mini03", "mini03-q1", "The municipal library holds 6030 volumes.")
    r = compiler.compile(corpus.documents, 0.5, topic=topic, preserve=[fact])
    assert fact.text in r.text
    assert r.n_preserved_facts == 1
    assert r.preserved_words == fact.n_words


def test_preservation_instruction_states_a_hard_constraint(profile):
    compiler = Compiler(profile.compiler, backend="fake")
    fact = PreservedFact("d", "q", "A specific fact.")
    instruction = compiler.build_instruction(10_000, 0.5, [fact])
    assert "MANDATORY PRESERVED FACTS" in instruction
    assert "verbatim" in instruction
    assert "A specific fact." in instruction
    # Paper Section 7.1: preserved facts come before general coverage.
    assert instruction.index("Place them FIRST") < instruction.index("broad coverage")


def test_blind_instruction_never_mentions_preservation(profile):
    compiler = Compiler(profile.compiler, backend="fake")
    assert "PRESERVED" not in compiler.build_instruction(10_000, 0.5, [])


def test_compile_topic_persists_wiki_and_metadata(profile, topic):
    from wicer_core.records import wiki_paths

    compiler = Compiler(profile.compiler, backend="fake")
    compile_topic(profile, topic, "moderate", compiler=compiler)
    md, meta = wiki_paths(profile, topic, "blind_moderate")
    assert md.exists() and meta.exists()

    import json

    m = json.loads(meta.read_text())
    assert m["level"] == "moderate" and m["target_ratio"] == 0.5
    assert "text" not in m  # metadata, not a second copy of the wiki


def test_unknown_compression_level_is_rejected(profile, topic):
    with pytest.raises(KeyError):
        compile_topic(profile, topic, "extreme", compiler=Compiler(profile.compiler, backend="fake"))


# -- diagnosis ---------------------------------------------------------------


def test_diagnosis_respects_the_word_budget(profile, topic):
    corpus = load_topic(profile, topic)
    d = Diagnoser(profile.compiler.model, profile.wicer, backend="fake")
    probe = next(q for q in corpus.qa if q.document_id == "mini03")
    fact = d.diagnose(probe, corpus.doc_by_id("mini03"), "a wrong answer")
    assert fact.document_id == "mini03"
    assert fact.question_id == probe.question_id
    assert fact.n_words <= profile.wicer.diagnosis_max_words
    # Diagnosis must be far smaller than the source document; that ratio is the
    # mechanism the ablation tests (paper Section 7.2).
    assert fact.n_words < corpus.doc_by_id("mini03").n_words / 3


def test_diagnosis_targets_the_passage_the_answer_needed(profile):
    corpus = load_topic(profile, "mini_topic")
    d = Diagnoser(profile.compiler.model, profile.wicer, backend="fake")
    doc = corpus.doc_by_id("mini03")
    probe = next(q for q in corpus.qa if q.document_id == "mini03" and "volumes" in q.answer)
    fact = d.diagnose(probe, doc, "no idea")
    assert "volumes" in fact.text


def test_clip_words_is_a_hard_ceiling():
    assert _clip_words("a b c d e", 3) == "a b c"
    assert _clip_words("a b", 5) == "a b"


# -- random pinning ----------------------------------------------------------


def test_random_pin_is_length_matched_and_reproducible(profile, topic):
    corpus = load_topic(profile, topic)
    cfg = profile.wicer
    probe = corpus.qa[0]
    a = RandomPinner(corpus, cfg.diagnosis_min_words, cfg.diagnosis_max_words, seed=42)
    b = RandomPinner(corpus, cfg.diagnosis_min_words, cfg.diagnosis_max_words, seed=42)
    fa, fb = a.pin(probe, 0), b.pin(probe, 0)
    assert fa.text == fb.text
    assert cfg.diagnosis_min_words <= fa.n_words <= cfg.diagnosis_max_words


def test_random_pins_come_from_across_the_collection(profile, topic):
    corpus = load_topic(profile, topic)
    pinner = RandomPinner(corpus, 25, 40, seed=7)
    sources = {pinner.pin(corpus.qa[0], 0).document_id for _ in range(40)}
    assert len(sources) > 1


# -- the loop ----------------------------------------------------------------


def _probe_scores(profile, topic, wiki):
    """Per-probe judge scores from the loop's own run against ``wiki``."""
    import json

    from wicer_core.config import resolve

    path = resolve(profile.results_dir) / topic / "probe_runs" / f"{wiki}.jsonl"
    return {r["question_id"]: r["score"] for r in map(json.loads, path.open())}


def test_loop_diagnoses_failures_and_pins_back_what_they_needed(profile, topic):
    """At least half the probes diagnosed at iteration 0 pass at iteration 1.

    Targeted repair is the mechanism, but the offline harness cannot show it
    for every probe: the fixture's documents share sentence templates, so the
    stub reader sometimes answers a pinned question from another document's
    sentence even though the pinned fact reached the wiki.  Nor is the overall
    probe mean promised to rise -- pinning displaces other coverage (paper
    Appendix I), sharply so in the fake compiler's fixed word budget
    (DEVIATIONS.md 9) -- and across probe draws it moves either way.
    """
    reader = ReaderClient(profile.reader)
    judge = Judge(profile.judge, backend="fake")
    compiler = Compiler(profile.compiler, backend="fake")
    diagnoser = Diagnoser(profile.compiler.model, profile.wicer, backend="fake")

    res = run_loop(profile, topic, reader=reader, judge=judge,
                   compiler=compiler, diagnoser=diagnoser)

    assert res.iterations[0]["wiki_name"] == BLIND_WIKI
    assert res.iterations[0]["n_pinned_facts"] == 0
    assert res.iterations[0]["n_failures"] > 0, "fixture must produce failures to diagnose"
    assert len(res.iterations) > 1, "loop must refine at least once"

    # Pinned facts accumulate and are never discarded (paper Appendix I).
    pinned = [r["n_pinned_facts"] for r in res.iterations]
    assert pinned == sorted(pinned)

    diagnosed = {f["question_id"] for f in res.preserved_facts if f["iteration"] == 0}
    after = _probe_scores(profile, topic, res.iterations[1]["wiki_name"])
    recovered = [q for q in diagnosed if after[q] > profile.wicer.failure_threshold]
    assert recovered and 2 * len(recovered) >= len(diagnosed), {q: after[q] for q in diagnosed}
    assert res.stop_reason in {"no_failures", "insufficient_improvement", "max_iterations"}


def test_loop_stops_within_the_iteration_budget(profile, topic):
    res = run_loop(
        profile, topic,
        reader=ReaderClient(profile.reader),
        judge=Judge(profile.judge, backend="fake"),
        compiler=Compiler(profile.compiler, backend="fake"),
        diagnoser=Diagnoser(profile.compiler.model, profile.wicer, backend="fake"),
    )
    assert len(res.iterations) <= profile.wicer.max_iterations + 1


def test_loop_writes_a_ledger_with_the_full_trajectory(profile, topic):
    import json

    from wicer_core.records import ledger_path

    run_loop(profile, topic, reader=ReaderClient(profile.reader),
             judge=Judge(profile.judge, backend="fake"),
             compiler=Compiler(profile.compiler, backend="fake"),
             diagnoser=Diagnoser(profile.compiler.model, profile.wicer, backend="fake"))

    led = json.loads(ledger_path(profile, topic, "wicer").read_text())
    assert led["topic"] == topic
    assert led["best_wiki_name"]
    for r in led["iterations"]:
        assert {"probe_mean", "probe_score_1_pct", "n_failures", "pinned_words"} <= set(r)
    assert led["usage"]["judge"]["backend"] == "fake"


def test_random_pin_control_uses_an_independent_blind_compilation(profile, topic):
    """Paper Section 7.5: the control compiles its own baseline."""
    from wicer_core.records import wiki_paths

    kwargs = dict(
        reader=ReaderClient(profile.reader),
        judge=Judge(profile.judge, backend="fake"),
        compiler=Compiler(profile.compiler, backend="fake"),
        diagnoser=Diagnoser(profile.compiler.model, profile.wicer, backend="fake"),
    )
    run_loop(profile, topic, **kwargs)
    run_random_pin(profile, topic, **kwargs)

    assert wiki_paths(profile, topic, "blind_moderate")[0].exists()
    assert wiki_paths(profile, topic, "random_blind_moderate")[0].exists()

    import json

    from wicer_core.records import ledger_path

    rp = json.loads(ledger_path(profile, topic, "random_pin").read_text())
    assert rp["iterations"][0]["wiki_name"] == "random_blind_moderate"
    # The pinned text must not be the diagnosed text.
    if rp["preserved_facts"]:
        wicer_led = json.loads(ledger_path(profile, topic, "wicer").read_text())
        assert {f["text"] for f in rp["preserved_facts"]} != {
            f["text"] for f in wicer_led["preserved_facts"]
        }


def _iteration(i: int, wiki: str, probe_mean: float, n_pinned: int) -> dict:
    return {"iteration": i, "wiki_name": wiki, "probe_mean": probe_mean,
            "n_pinned_facts": n_pinned}


def _loop_result(iterations: list[dict], best: str) -> LoopResult:
    return LoopResult(
        topic="t", best_iteration=0, best_wiki_name=best, best_probe_mean=0.0,
        blind_probe_mean=0.0, stop_reason="insufficient_improvement", iterations=iterations,
    )


def test_best_pinned_iteration_never_returns_the_unpinned_blind():
    iters = [_iteration(0, "blind", 3.20, 0), _iteration(1, "first", 3.06, 13),
             _iteration(2, "second", 2.90, 17)]
    assert best_pinned_iteration(iters)["wiki_name"] == "first"
    assert best_pinned_iteration(iters[:1]) is None


def test_eval_full_scores_the_pinned_wiki_even_when_best_of_is_the_blind():
    """Both pilots' random arm peaked at its blind, so the condition labelled
    random_pin was a second blind and no random-pinned wiki was ever scored."""
    res = _loop_result(
        [_iteration(0, "random_blind_moderate", 3.20, 0),
         _iteration(1, "random_pin_iter1", 3.06, 13)],
        best="random_blind_moderate",
    )
    assert full_set_wikis(res, "random_blind", "random_pin") == [
        ("random_blind_moderate", "random_blind"),
        ("random_pin_iter1", "random_pin"),
    ]


def test_eval_full_names_a_wicer_pinned_wiki_after_itself():
    res = _loop_result(
        [_iteration(0, "blind_moderate", 3.16, 0), _iteration(1, "wicer_iter1", 3.66, 13),
         _iteration(2, "wicer_iter2", 3.30, 17)],
        best="wicer_iter1",
    )
    assert full_set_wikis(res, "wiki_moderate") == [
        ("blind_moderate", "wiki_moderate"), ("wicer_iter1", "wicer_iter1"),
    ]


def test_eval_full_scores_only_the_blind_when_nothing_was_pinned():
    res = _loop_result([_iteration(0, "blind_moderate", 4.9, 0)], best="blind_moderate")
    assert full_set_wikis(res, "wiki_moderate") == [("blind_moderate", "wiki_moderate")]


def test_random_arm_scores_its_blind_and_its_pinned_wiki_on_every_question(profile, topic):
    from benchmark.run_benchmark import build_wiki_context
    from wicer.random_pin import BLIND_CONDITION, CONDITION
    from wicer_core.records import answers_path, read_jsonl, run_dir, wiki_paths

    res = run_random_pin(
        profile, topic,
        reader=ReaderClient(profile.reader),
        judge=Judge(profile.judge, backend="fake"),
        compiler=Compiler(profile.compiler, backend="fake"),
        diagnoser=Diagnoser(profile.compiler.model, profile.wicer, backend="fake"),
    )
    pairs = full_set_wikis(res, BLIND_CONDITION, CONDITION)
    assert len(pairs) == 2, "fixture must pin something"
    evaluate_wikis_full(profile, topic, pairs, reader=ReaderClient(profile.reader))

    n_questions = len(load_topic(profile, topic).qa)
    for wiki, condition in pairs:
        served = (run_dir(profile, topic, condition) / "context.txt").read_text()
        assert served == build_wiki_context(wiki_paths(profile, topic, wiki)[0].read_text())
        assert len(read_jsonl(answers_path(profile, topic, condition))) == n_questions


# -- the openrouter backend ---------------------------------------------------


class _StubOR:
    """Stands in for wicer_core.openrouter.OpenRouterClient."""

    def __init__(self, text="# Wiki\n\nCompiled body.", finish_reason="stop"):
        self.text = text
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def chat(self, **kw):
        from wicer_core.openrouter import Reply

        self.calls.append(kw)
        return Reply(
            text=self.text,
            input_tokens=5000,
            output_tokens=900,
            cost_usd=0.042,
            served_model="anthropic/claude-sonnet-4.5-served",
            cache_read_tokens=4000,
            finish_reason=self.finish_reason,
        )


def _corpus(profile, topic):
    return load_topic(profile, topic)


def _or_compiler(profile, stub, model="anthropic/claude-sonnet-4.5"):
    from dataclasses import replace

    return Compiler(
        replace(profile.compiler, model=model), backend="openrouter", openrouter_client=stub
    )


def test_openrouter_compiler_returns_the_compiled_text(profile, topic):
    stub = _StubOR()
    result = _or_compiler(profile, stub).compile(_corpus(profile, topic).documents, 0.5)
    assert result.text == "# Wiki\n\nCompiled body."
    assert len(stub.calls) == 1


def test_openrouter_compiler_requests_a_cache_breakpoint(profile, topic):
    """The source blob is resent at three levels and once per iteration."""
    stub = _StubOR()
    _or_compiler(profile, stub).compile(_corpus(profile, topic).documents, 0.5)
    assert stub.calls[0]["cache_system"] is True


def test_openrouter_compiler_puts_the_sources_in_the_system_block(profile, topic):
    stub = _StubOR()
    _or_compiler(profile, stub).compile(_corpus(profile, topic).documents, 0.5)
    assert _corpus(profile, topic).documents[0].text[:40] in stub.calls[0]["system"]


def test_openrouter_compiler_sends_the_instruction_as_the_user_turn(profile, topic):
    stub = _StubOR()
    _or_compiler(profile, stub).compile(_corpus(profile, topic).documents, 0.5)
    assert "words" in stub.calls[0]["user"]


def test_compile_records_the_finish_reason(profile, topic):
    """Provenance: a wiki's meta says how its generation ended."""
    stub = _StubOR()
    result = _or_compiler(profile, stub).compile(_corpus(profile, topic).documents, 0.5)
    assert result.finish_reason == "stop"


def test_a_wiki_truncated_at_max_tokens_is_rejected(profile, topic):
    """Reasoning bills against max_tokens, so a wiki can be cut off mid-sentence
    while still being non-empty.  Scoring one measures the cap, not the method:
    every question past the cut has no context to answer from, which reads as
    compilation losing knowledge.  Fail and name the cap instead."""
    stub = _StubOR(finish_reason="length")
    with pytest.raises(CompileTruncated) as excinfo:
        _or_compiler(profile, stub).compile(_corpus(profile, topic).documents, 0.5)
    assert "max_tokens" in str(excinfo.value)


def test_the_anthropic_stop_reason_for_truncation_is_rejected_too(profile, topic):
    """The two backends spell it differently -- 'length' and 'max_tokens'."""
    stub = _StubOR(finish_reason="max_tokens")
    with pytest.raises(CompileTruncated):
        _or_compiler(profile, stub).compile(_corpus(profile, topic).documents, 0.5)


def test_a_truncated_compile_still_records_what_it_spent(profile, topic):
    """The call was made and billed before the truncation was found.  Dropping
    its usage would under-report the run's cost by exactly the calls that
    failed."""
    stub = _StubOR(finish_reason="length")
    compiler = _or_compiler(profile, stub)
    with pytest.raises(CompileTruncated):
        compiler.compile(_corpus(profile, topic).documents, 0.5)
    assert compiler.n_calls == 1
    assert compiler.output_tokens == 900
    assert compiler.measured_cost_usd == pytest.approx(0.042)


def test_openrouter_compiler_passes_the_configured_effort(profile, topic):
    stub = _StubOR()
    _or_compiler(profile, stub).compile(_corpus(profile, topic).documents, 0.5)
    assert stub.calls[0]["effort"] == profile.compiler.effort


def test_openrouter_compiler_records_the_served_model(profile, topic):
    result = _or_compiler(profile, _StubOR()).compile(_corpus(profile, topic).documents, 0.5)
    assert result.model == "anthropic/claude-sonnet-4.5-served"


def test_openrouter_compiler_records_cache_reads(profile, topic):
    result = _or_compiler(profile, _StubOR()).compile(_corpus(profile, topic).documents, 0.5)
    assert result.cache_read_tokens == 4000


def test_openrouter_compiler_uses_the_measured_cost(profile, topic):
    """Not the Anthropic price table, which is wrong for any other slug."""
    result = _or_compiler(profile, _StubOR()).compile(_corpus(profile, topic).documents, 0.5)
    assert result.estimated_cost_usd == pytest.approx(0.042)


def test_openrouter_compiler_accumulates_measured_cost(profile, topic):
    stub = _StubOR()
    c = _or_compiler(profile, stub)
    c.compile(_corpus(profile, topic).documents, 0.75)
    c.compile(_corpus(profile, topic).documents, 0.25)
    assert c.measured_cost_usd == pytest.approx(0.084)
    assert c.usage_summary()["cost_source"] == "openrouter"


def test_openrouter_diagnoser_returns_the_fact_text(profile, topic):
    stub = _StubOR(text="The library holds 6030 volumes.")
    d = Diagnoser(
        "anthropic/claude-sonnet-4.5",
        profile.wicer,
        backend="openrouter",
        openrouter_client=stub,
    )
    corpus = _corpus(profile, topic)
    fact = d.diagnose(corpus.qa[0], corpus.documents[0], "no idea")
    assert fact.text == "The library holds 6030 volumes."
    assert fact.document_id == corpus.documents[0].document_id


def test_openrouter_diagnoser_does_not_request_caching(profile, topic):
    """Each diagnosis carries a different single document; nothing to reuse."""
    stub = _StubOR(text="a fact")
    d = Diagnoser(
        "anthropic/claude-sonnet-4.5",
        profile.wicer,
        backend="openrouter",
        openrouter_client=stub,
    )
    corpus = _corpus(profile, topic)
    d.diagnose(corpus.qa[0], corpus.documents[0], "no idea")
    assert stub.calls[0].get("cache_system", False) is False


def test_openrouter_diagnoser_accumulates_measured_cost(profile, topic):
    stub = _StubOR(text="a fact")
    d = Diagnoser(
        "anthropic/claude-sonnet-4.5",
        profile.wicer,
        backend="openrouter",
        openrouter_client=stub,
    )
    corpus = _corpus(profile, topic)
    d.diagnose(corpus.qa[0], corpus.documents[0], "no idea")
    assert d.measured_cost_usd == pytest.approx(0.042)
    assert d.usage_summary()["cost_source"] == "openrouter"


def test_openrouter_compiler_asks_for_the_one_hour_cache(profile, topic):
    """The same source blob is recompiled 9 times per topic; 5 minutes is not
    long enough to span the loop's reader and judge work between compiles."""
    stub = _StubOR()
    _or_compiler(profile, stub).compile(_corpus(profile, topic).documents, 0.5)
    assert stub.calls[0]["cache_ttl"] == "1h"


# -- verbatim preservation ----------------------------------------------------


def test_preservation_rate_is_one_when_every_fact_survives(profile, topic):
    corpus = load_topic(profile, topic)
    compiler = Compiler(profile.compiler, backend="fake")
    facts = [PreservedFact("mini03", "mini03-q1", "The municipal library holds 6030 volumes.")]
    r = compiler.compile(corpus.documents, 0.5, topic=topic, preserve=facts)
    assert r.n_preserved_found == 1
    assert r.preservation_rate == 1.0


def test_preservation_rate_catches_a_compiler_that_drops_pinned_facts(profile, topic):
    """WiCER's mechanism is the pinning. A compiler that ignores the instruction
    produces a null result that looks exactly like 'the method does not work'."""
    corpus = load_topic(profile, topic)

    class Dropping(Compiler):
        def _compile_openrouter(self, source_text, instruction):
            return "A wiki that quietly omits what it was told to keep.", {"model": "stub"}

    c = Dropping(profile.compiler, backend="openrouter")
    facts = [
        PreservedFact("mini03", "q1", "The municipal library holds 6030 volumes."),
        PreservedFact("mini04", "q2", "A second fact that also vanishes."),
    ]
    r = c.compile(corpus.documents, 0.5, topic=topic, preserve=facts)
    assert r.n_preserved_found == 0
    assert r.preservation_rate == 0.0


def test_preservation_rate_is_none_when_nothing_was_pinned(profile, topic):
    """Blind compilation pins nothing; a rate of 0.0 would read as a failure."""
    compiler = Compiler(profile.compiler, backend="fake")
    r = compiler.compile(load_topic(profile, topic).documents, 0.5, topic=topic)
    assert r.n_preserved_found == 0
    assert r.preservation_rate is None


def test_preservation_ignores_whitespace_differences(profile, topic):
    """A compiler that reflows a fact across lines has still preserved it."""
    corpus = load_topic(profile, topic)

    class Reflowing(Compiler):
        def _compile_openrouter(self, source_text, instruction):
            return "Intro.\n\nThe municipal library\nholds   6030 volumes.\n\nMore.", {}

    c = Reflowing(profile.compiler, backend="openrouter")
    fact = PreservedFact("mini03", "q1", "The municipal library holds 6030 volumes.")
    r = c.compile(corpus.documents, 0.5, topic=topic, preserve=[fact])
    assert r.n_preserved_found == 1


def test_loop_records_the_preservation_rate_per_iteration(profile, topic):
    reader = ReaderClient(profile.reader)
    judge = Judge(profile.judge, backend="fake")
    compiler = Compiler(profile.compiler, backend="fake")
    diagnoser = Diagnoser(profile.compiler.model, profile.wicer, backend="fake")
    res = run_loop(profile, topic, reader=reader, judge=judge,
                   compiler=compiler, diagnoser=diagnoser)
    later = [r for r in res.iterations if r["n_pinned_facts"]]
    assert later, "expected at least one iteration with pinned facts"
    assert all("preservation_rate" in r for r in res.iterations)
    assert all(0.0 <= r["preservation_rate"] <= 1.0 for r in later)
