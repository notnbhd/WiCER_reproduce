"""Judge: the Appendix B rubric, the JSON parse ladder, caching, calibration."""

from __future__ import annotations


import pytest

from wicer_core.judge import (
    JUDGE_SYSTEM_PROMPT,
    Judge,
    JudgeRequest,
    _numbers,
    _parse_judgement,
)


def test_rubric_is_the_papers_text():
    # Appendix B. Reword this and every table in the paper changes meaning.
    assert JUDGE_SYSTEM_PROMPT.startswith(
        "You are an expert evaluator comparing a generated answer against a "
        "gold-standard answer."
    )
    for line in [
        "5 = Perfect -- captures all key facts, accurate",
        "4 = Good -- mostly correct, minor omissions",
        "3 = Acceptable -- partially correct, misses details",
        "2 = Poor -- significant errors or major omissions",
        "1 = Wrong -- incorrect, irrelevant, misses the point",
    ]:
        assert line in JUDGE_SYSTEM_PROMPT
    assert '{"score": <1-5>, "reasoning": "<explanation>"}' in JUDGE_SYSTEM_PROMPT


@pytest.mark.parametrize("raw,expected", [
    ('{"score": 4, "reasoning": "close enough"}', 4),
    ('```json\n{"score": 2, "reasoning": "x"}\n```', 2),
    ('Here is my verdict: {"score": 5, "reasoning": "perfect"} -- done.', 5),
    ('{"score": "3", "reasoning": "coerced from string"}', 3),
    ('\n\n  {"score": 1, "reasoning": "wrong"}  \n', 1),
])
def test_parse_ladder_recovers_valid_scores(raw, expected):
    parsed = _parse_judgement(raw)
    assert parsed is not None and parsed[0] == expected


@pytest.mark.parametrize("raw", [
    "The answer was pretty good, I'd say a four.",
    '{"reasoning": "no score field"}',
    '{"score": 9, "reasoning": "out of range"}',
    '{"score": null}',
    "",
])
def test_parse_ladder_reports_failure_rather_than_guessing(raw):
    assert _parse_judgement(raw) is None


def test_fake_judge_scores_recall(profile):
    judge = Judge(profile.judge, backend="fake")
    perfect = judge.score(JudgeRequest("q?", "the mayor was Ilse Hallam", "the mayor was Ilse Hallam"))
    assert perfect.score == 5
    wrong = judge.score(JudgeRequest("q?", "the mayor was Ilse Hallam", "completely unrelated text"))
    assert wrong.score == 1


def test_fake_judge_treats_a_wrong_figure_as_wrong(profile):
    """Token overlap alone would give partial credit for the wrong number."""
    judge = Judge(profile.judge, backend="fake")
    j = judge.score(JudgeRequest(
        "How many volumes?", "6030 volumes.", "The municipal library holds 10300 volumes."
    ))
    assert j.score == 1


def test_number_extraction_normalises_separators():
    assert _numbers("6,030 volumes in 1890.") == {"6030", "1890"}
    assert _numbers("no digits here") == set()


def test_results_are_cached_content_addressed(profile, tmp_path):
    judge = Judge(profile.judge, backend="fake")
    req = JudgeRequest("q?", "gold text here", "gold text here")
    first = judge.score(req)
    assert first.cached is False
    assert judge.n_cache_hits == 0

    second = judge.score(req)
    assert second.cached is True
    assert second.score == first.score
    assert judge.n_cache_hits == 1

    # A different answer is a different cache key.
    judge.score(JudgeRequest("q?", "gold text here", "something else entirely"))
    assert judge.n_cache_hits == 1


def test_corrupt_cache_entry_is_discarded_not_crashed(profile):
    judge = Judge(profile.judge, backend="fake")
    req = JudgeRequest("q?", "gold", "gold")
    judge.score(req)
    corrupted = next(judge.cache_dir.glob("*.json"))
    corrupted.write_text("{not json")
    again = judge.score(req)
    assert again.score == 5


def test_score_many_preserves_request_order(profile):
    judge = Judge(profile.judge, backend="fake")
    reqs = [
        JudgeRequest("q?", "alpha beta gamma", "alpha beta gamma"),
        JudgeRequest("q?", "alpha beta gamma", "nothing in common"),
        JudgeRequest("q?", "delta epsilon", "delta epsilon"),
    ]
    scores = [j.score for j in judge.score_many(reqs)]
    assert scores[0] == 5 and scores[1] == 1 and scores[2] == 5


def test_usage_summary_records_the_pinned_model(profile):
    judge = Judge(profile.judge, backend="fake")
    judge.score(JudgeRequest("q?", "g", "g"))
    summary = judge.usage_summary()
    assert summary["judge_model"] == profile.judge.model
    assert summary["backend"] == "fake"
    assert summary["parse_failures"] == 0


# -- the openrouter backend ---------------------------------------------------


class _StubOR:
    """Stands in for wicer_core.openrouter.OpenRouterClient."""

    def __init__(self, text='{"score": 4, "reasoning": "close enough"}'):
        self.text = text
        self.calls: list[dict] = []

    def chat(self, **kw):
        from wicer_core.openrouter import Reply

        self.calls.append(kw)
        return Reply(
            text=self.text,
            input_tokens=120,
            output_tokens=30,
            cost_usd=0.002,
            served_model="anthropic/claude-sonnet-4.5-served",
            cache_read_tokens=0,
        )


def _or_judge(profile, stub, model="anthropic/claude-sonnet-4.5"):
    from dataclasses import replace

    return Judge(
        replace(profile.judge, model=model), backend="openrouter", openrouter_client=stub
    )


def _req(question="q"):
    return JudgeRequest(question=question, gold="g", generated="a")


def test_openrouter_backend_scores_through_openrouter(profile):
    stub = _StubOR()
    out = _or_judge(profile, stub).score(_req())
    assert out.score == 4
    assert out.reasoning == "close enough"
    assert len(stub.calls) == 1


def test_openrouter_backend_records_the_served_model(profile):
    """spec.md Section 9: judge drift is only visible if we record it."""
    out = _or_judge(profile, _StubOR()).score(_req())
    assert out.model == "anthropic/claude-sonnet-4.5-served"


def test_openrouter_backend_sends_the_rubric_as_the_system_prompt(profile):
    stub = _StubOR()
    _or_judge(profile, stub).score(_req())
    assert stub.calls[0]["system"] == JUDGE_SYSTEM_PROMPT


def test_openrouter_backend_passes_the_configured_effort(profile):
    stub = _StubOR()
    _or_judge(profile, stub).score(_req())
    assert stub.calls[0]["effort"] == profile.judge.effort


def test_openrouter_backend_does_not_request_caching(profile):
    """Judge prompts are short and all different; a breakpoint would only cost."""
    stub = _StubOR()
    _or_judge(profile, stub).score(_req())
    assert stub.calls[0].get("cache_system", False) is False


def test_openrouter_backend_accumulates_measured_cost(profile):
    stub = _StubOR()
    j = _or_judge(profile, stub)
    j.score(_req("q1"))
    j.score(_req("q2"))
    assert j.measured_cost_usd == pytest.approx(0.004)


def test_usage_summary_prefers_measured_cost_over_the_price_table(profile):
    j = _or_judge(profile, _StubOR())
    j.score(_req())
    summary = j.usage_summary()
    assert summary["measured_cost_usd"] == pytest.approx(0.002)
    assert summary["cost_source"] == "openrouter"


def test_usage_summary_marks_estimates_on_the_non_openrouter_path(profile):
    j = Judge(profile.judge, backend="fake")
    j.score(_req())
    assert j.usage_summary()["cost_source"] == "estimate"


def test_openrouter_results_are_cached_like_any_other(profile):
    stub = _StubOR()
    j = _or_judge(profile, stub)
    j.score(_req())
    j.score(_req())
    assert len(stub.calls) == 1, "second identical request must hit the disk cache"


def test_a_different_slug_does_not_reuse_cached_scores(profile):
    """Switching models must re-score, not silently reuse another model's grade."""
    _or_judge(profile, _StubOR()).score(_req())
    second = _StubOR()
    _or_judge(profile, second, model="google/gemini-2.5-pro").score(_req())
    assert len(second.calls) == 1


def test_openrouter_parse_failure_is_recorded_not_raised(profile):
    j = _or_judge(profile, _StubOR(text="I think it's pretty good, honestly."))
    out = j.score(_req())
    assert out.parse_failure is True
    assert j.n_parse_failures == 1


def test_an_empty_openrouter_completion_is_a_parse_failure_not_a_dead_sweep(profile):
    """One bad response must not kill 54,000 judge calls."""
    from wicer_core.openrouter import OpenRouterError

    class Failing:
        def chat(self, **kw):
            raise OpenRouterError("anthropic/x returned no text (finish_reason=length)")

    j = _or_judge(profile, Failing())
    out = j.score(_req())
    assert out.parse_failure is True
    assert out.score == 1
    assert j.n_parse_failures == 1
