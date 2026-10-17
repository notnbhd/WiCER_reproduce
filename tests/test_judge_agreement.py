"""Judge agreement: the Appendix J statistics, between two judge backends.

Paper Appendix J validates its judge against a human rater (n=100): Pearson
r 0.940, exact 75%, within-1 99%, MAE 0.26, bias +0.06, and per-condition
r >= 0.89 with |dmean| <= 0.17.  Swapping the judge model is defensible only
against the same bar, measured against the judge you are replacing.
"""

from __future__ import annotations

import pytest

from wicer_core.agreement import (
    Agreement,
    agreement,
    pearson,
    spearman,
)

# -- correlation primitives ---------------------------------------------------


def test_pearson_of_identical_series_is_one():
    assert pearson([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_pearson_of_inverted_series_is_minus_one():
    assert pearson([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_detects_a_known_correlation():
    # Two adjacent pairs swapped: sum(dx*dy)=8 over sqrt(10)*sqrt(10).
    assert pearson([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)


def test_pearson_is_none_when_a_series_has_no_variance():
    """Every answer scored 4 is a degenerate judge, not a perfect correlation."""
    assert pearson([4, 4, 4, 4], [1, 2, 3, 4]) is None


def test_spearman_is_one_for_any_monotonic_relation():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 99]) == pytest.approx(1.0)


def test_spearman_handles_ties():
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


# -- the agreement report -----------------------------------------------------


def test_identical_scores_agree_perfectly():
    a = agreement([5, 4, 3, 2, 1], [5, 4, 3, 2, 1])
    assert a.n == 5
    assert a.exact == 5
    assert a.within_1 == 5
    assert a.mae == pytest.approx(0.0)
    assert a.bias == pytest.approx(0.0)
    assert a.pearson == pytest.approx(1.0)


def test_bias_is_signed_candidate_minus_reference():
    """A judge that scores a point high everywhere must report +1, not 1."""
    a = agreement([5, 5, 5], [4, 4, 4])
    assert a.bias == pytest.approx(1.0)
    b = agreement([3, 3, 3], [4, 4, 4])
    assert b.bias == pytest.approx(-1.0)


def test_mae_is_unsigned_so_opposite_errors_do_not_cancel():
    a = agreement([5, 3], [4, 4])
    assert a.mae == pytest.approx(1.0)
    assert a.bias == pytest.approx(0.0)


def test_within_1_counts_the_exact_matches_too():
    a = agreement([4, 5, 1], [4, 4, 4])
    assert a.exact == 1
    assert a.within_1 == 2


def test_empty_input_is_reported_not_crashed():
    a = agreement([], [])
    assert a.n == 0
    assert a.pearson is None
    assert a.mae is None


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        agreement([1, 2], [1])


# -- the paper's bar ----------------------------------------------------------


def test_meets_paper_bar_accepts_appendix_j_itself():
    """The paper's own judge must pass the test built from its own numbers."""
    ref = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5]
    cand = [1, 2, 2, 3, 3, 4, 4, 4, 5, 5]      # one point off on one sample
    a = agreement(cand, ref)
    assert a.meets_paper_bar() is True


def test_meets_paper_bar_rejects_a_systematically_lenient_judge():
    """Bias 0.4 would swallow the 0.18-point FC/RAG gap whole."""
    ref = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    cand = [2, 2, 4, 4, 5, 2, 3, 3, 5, 5]
    a = agreement(cand, ref)
    assert a.bias > 0.17
    assert a.meets_paper_bar() is False


def test_meets_paper_bar_rejects_a_noisy_but_unbiased_judge():
    ref = [1, 5, 1, 5, 1, 5, 1, 5]
    cand = [5, 1, 5, 1, 5, 1, 5, 1]
    assert agreement(cand, ref).meets_paper_bar() is False


def test_report_renders_a_markdown_table():
    a = agreement([5, 4, 3], [5, 4, 4])
    md = Agreement.render({"overall": a})
    assert "Pearson" in md and "MAE" in md


def test_per_condition_breakdown_flags_the_offending_condition():
    """A judge can be unbiased overall and still wreck one condition."""
    by_cond = {
        # tracks the reference closely
        "fc_raw": agreement([2, 3, 4, 5, 3], [2, 3, 4, 5, 3]),
        # same spread, shifted a point down -- bias far past the 0.17 bound
        "rag_top5": agreement([1, 2, 3, 4, 2], [2, 3, 4, 5, 3]),
    }
    md = Agreement.render(by_cond)
    assert "rag_top5" in md
    assert by_cond["fc_raw"].meets_paper_bar() is True
    assert by_cond["rag_top5"].meets_paper_bar() is False


# -- degenerate samples are not failures -------------------------------------


def test_a_constant_sample_is_inconclusive_not_a_failure():
    """Three answers all scored 5 give an undefined correlation. Calling that
    FAIL would reject a judge that actually agreed on every single item."""
    a = agreement([5, 5, 5], [5, 5, 5])
    assert a.pearson is None
    assert a.verdict == "n/a"
    assert a.meets_paper_bar() is False       # still not evidence of passing


def test_a_real_disagreement_is_still_a_failure():
    a = agreement([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert a.verdict == "FAIL"


def test_a_tracking_judge_passes():
    a = agreement([2, 3, 4, 5, 3], [2, 3, 4, 5, 3])
    assert a.verdict == "PASS"


def test_inconclusive_groups_do_not_count_as_failures_overall():
    groups = {
        "overall": agreement([2, 3, 4, 5], [2, 3, 4, 5]),
        "flat_condition": agreement([4, 4, 4], [4, 4, 4]),
    }
    assert Agreement.all_conclusive_pass(groups) is True


def test_one_failing_group_sinks_the_verdict():
    groups = {
        "overall": agreement([2, 3, 4, 5], [2, 3, 4, 5]),
        "biased": agreement([1, 2, 3, 4], [2, 3, 4, 5]),
    }
    assert Agreement.all_conclusive_pass(groups) is False


def test_render_shows_the_inconclusive_verdict():
    md = Agreement.render({"flat": agreement([4, 4, 4], [4, 4, 4])})
    assert "n/a" in md
