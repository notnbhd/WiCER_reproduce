"""Aggregation maths, checked against hand-computed values."""

from __future__ import annotations

import math

import pytest

from wicer_core.metrics import (
    compression_ratio,
    fmt,
    median,
    percentile,
    recovery_rate,
    retrieval_stats,
    score_stats,
    timing_stats,
)


def test_percentile_interpolates():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 4.0
    assert percentile(xs, 50) == pytest.approx(2.5)
    assert median(xs) == pytest.approx(2.5)


def test_percentile_handles_degenerate_inputs():
    assert math.isnan(percentile([], 50))
    assert percentile([7.0], 95) == 7.0


def test_score_stats_counts_catastrophic_failures_separately():
    # Score 1 is the paper's "catastrophic failure" and is not just "<= 2".
    s = score_stats([5, 4, 3, 2, 1, 1])
    assert s.n == 6
    assert s.mean == pytest.approx(16 / 6)
    assert s.pct_ge_4 == pytest.approx(100 * 2 / 6)
    assert s.pct_le_2 == pytest.approx(100 * 3 / 6)
    assert s.pct_eq_1 == pytest.approx(100 * 2 / 6)


def test_score_stats_empty_is_nan_not_zero():
    s = score_stats([])
    assert s.n == 0
    assert math.isnan(s.mean)


def test_timing_stats_percentiles():
    t = timing_stats([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [10.0, 20.0, 30.0])
    assert t.ttft_median == pytest.approx(2.0)
    assert t.ttft_min == 1.0 and t.ttft_max == 3.0
    assert t.total_median == pytest.approx(5.0)
    assert t.throughput_median == pytest.approx(20.0)


def test_retrieval_stats_ignores_misses_in_mean_rank():
    r = retrieval_stats([True, True, False], [1, 3, None])
    assert r.accuracy_pct == pytest.approx(100 * 2 / 3)
    assert r.mean_rank == pytest.approx(2.0)
    assert r.mrr == pytest.approx((1 / 1 + 1 / 3) / 3)


def test_recovery_rate_matches_paper_definition():
    # Paper Table 4: recovery is the share of the blind->FC gap that is closed.
    assert recovery_rate(2.0, 3.0, 4.0) == pytest.approx(50.0)
    assert recovery_rate(2.0, 4.5, 4.0) == pytest.approx(125.0)   # can exceed FC
    assert recovery_rate(2.0, 2.0, 4.0) == pytest.approx(0.0)


def test_recovery_rate_undefined_without_baseline():
    # The two topics that overflow the context window have no FC baseline.
    assert recovery_rate(2.0, 3.0, None) is None
    assert recovery_rate(3.0, 3.5, 3.0) is None   # zero gap


def test_compression_ratio_and_formatting():
    assert compression_ratio(354, 1000) == pytest.approx(0.354)
    assert math.isnan(compression_ratio(10, 0))
    assert fmt(float("nan")) == "--"
    assert fmt(3.456) == "3.46"
