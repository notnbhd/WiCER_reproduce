"""Aggregation of per-question records into the paper's reported metrics.

Paper Section 4 (per question): TTFT, total response time, generation
throughput, judge score.  RAG additionally reports retrieval accuracy.
Aggregated per topic: mean score, % >= 4, % = 1, median / P5 / P95 TTFT,
median throughput (spec.md Section 4).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile; q in [0, 100]. Empty -> nan."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


@dataclass
class ScoreStats:
    n: int
    mean: float
    pct_ge_4: float
    pct_le_2: float
    pct_eq_1: float      # the "catastrophic failure" rate -- a headline metric
    n_parse_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_stats(scores: Sequence[int], n_parse_failures: int = 0) -> ScoreStats:
    n = len(scores)
    if n == 0:
        nan = float("nan")
        return ScoreStats(0, nan, nan, nan, nan, n_parse_failures)
    return ScoreStats(
        n=n,
        mean=mean([float(s) for s in scores]),
        pct_ge_4=100.0 * sum(1 for s in scores if s >= 4) / n,
        pct_le_2=100.0 * sum(1 for s in scores if s <= 2) / n,
        pct_eq_1=100.0 * sum(1 for s in scores if s == 1) / n,
        n_parse_failures=n_parse_failures,
    )


@dataclass
class TimingStats:
    n: int
    ttft_median: float
    ttft_p5: float
    ttft_p95: float
    ttft_min: float
    ttft_max: float
    total_median: float
    throughput_median: float
    throughput_min: float
    throughput_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def timing_stats(
    ttfts: Sequence[float], totals: Sequence[float], throughputs: Sequence[float]
) -> TimingStats:
    nan = float("nan")
    if not ttfts:
        return TimingStats(0, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    return TimingStats(
        n=len(ttfts),
        ttft_median=median(ttfts),
        ttft_p5=percentile(ttfts, 5),
        ttft_p95=percentile(ttfts, 95),
        ttft_min=min(ttfts),
        ttft_max=max(ttfts),
        total_median=median(totals) if totals else nan,
        throughput_median=median(throughputs) if throughputs else nan,
        throughput_min=min(throughputs) if throughputs else nan,
        throughput_max=max(throughputs) if throughputs else nan,
    )


@dataclass
class RetrievalStats:
    n: int
    accuracy_pct: float          # gold document present in the top-k chunks
    mean_rank: float             # 1-based rank of the first gold chunk, hits only
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def retrieval_stats(hits: Sequence[bool], ranks: Sequence[int | None]) -> RetrievalStats:
    n = len(hits)
    if n == 0:
        nan = float("nan")
        return RetrievalStats(0, nan, nan, nan)
    hit_ranks = [r for r in ranks if r is not None and r > 0]
    return RetrievalStats(
        n=n,
        accuracy_pct=100.0 * sum(1 for h in hits if h) / n,
        mean_rank=mean([float(r) for r in hit_ranks]),
        mrr=sum(1.0 / r for r in hit_ranks) / n,
    )


def recovery_rate(blind: float, best: float, fc_raw: float | None) -> float | None:
    """Fraction of the blind-compilation quality gap that was recovered.

    Paper Section 7.3 / Table 4.  Undefined where there is no full-context
    baseline (the two topics that overflow the window) or where blind already
    matches FC.
    """
    if fc_raw is None:
        return None
    gap = fc_raw - blind
    if abs(gap) < 1e-9:
        return None
    return 100.0 * (best - blind) / gap


def compression_ratio(compiled_units: int, source_units: int) -> float:
    """Actual compression, as a fraction of the source.

    Paper Section 6.2: targets of 75/50/25% produce actual ratios of
    35.4/12.2/8.2%.  Always measure; never assume the target held.
    """
    return compiled_units / source_units if source_units else float("nan")


def fmt(value: float, places: int = 2) -> str:
    """Table-friendly float formatting that survives nan."""
    if value != value:  # nan
        return "--"
    return f"{value:.{places}f}"
