"""Judge-agreement statistics, shaped after the paper's Appendix J.

Appendix J validates the paper's judge against a human rater on 100 stratified
samples: Pearson r 0.940, exact 75%, within-1 99%, MAE 0.26, bias +0.06, and
per-condition r >= 0.89 with |dmean| <= 0.17.

Swapping the judge model is defensible only against the same bar -- measured,
here, against the judge being replaced rather than against a human.  The
threshold that matters is the *bias*: paper Section 5 reports the full-context
/ RAG gap as 0.18 points (FC 3.47, RAG 3.64), so a judge carrying a
condition-dependent bias of 0.17 can erase or invent that result outright.
E3 and the ablation move ~0.8-1.2 points and tolerate far more noise.

No scipy: the repo ships four dependencies and these are twenty lines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Appendix J, per-condition row.
MIN_PEARSON = 0.89
MAX_ABS_BIAS = 0.17


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Product-moment correlation.  None when either series is constant."""
    if len(a) < 2:
        return None
    ma, mb = _mean(a), _mean(b)
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    na = math.sqrt(sum(x * x for x in da))
    nb = math.sqrt(sum(y * y for y in db))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(da, db)) / (na * nb)


def _ranks(xs: Sequence[float]) -> list[float]:
    """Average ranks, so tied scores -- common on a 1-5 rubric -- share one."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    return pearson(_ranks(a), _ranks(b))


@dataclass(frozen=True)
class Agreement:
    """How closely a candidate judge tracks the reference judge."""

    n: int
    exact: int
    within_1: int
    mae: float | None
    bias: float | None          # candidate - reference; signed on purpose
    pearson: float | None
    spearman: float | None
    candidate_mean: float | None
    reference_mean: float | None

    @property
    def exact_pct(self) -> float:
        return 100.0 * self.exact / self.n if self.n else 0.0

    @property
    def within_1_pct(self) -> float:
        return 100.0 * self.within_1 / self.n if self.n else 0.0

    def meets_paper_bar(self) -> bool:
        """Appendix J's per-condition standard: r >= 0.89 and |bias| <= 0.17.

        False for an inconclusive group too -- absence of evidence is not a
        pass.  Use `verdict` to tell the two apart.
        """
        if self.pearson is None or self.bias is None:
            return False
        return self.pearson >= MIN_PEARSON and abs(self.bias) <= MAX_ABS_BIAS

    @property
    def verdict(self) -> str:
        """PASS, FAIL, or n/a.

        n/a means the correlation is undefined -- fewer than two samples, or
        every answer in the group drew the same score.  That is a property of
        the sample, not of the judge: reporting it as FAIL would reject a judge
        that agreed on every item.  Widen the sample instead.
        """
        if self.pearson is None:
            return "n/a"
        return "PASS" if self.meets_paper_bar() else "FAIL"

    @staticmethod
    def all_conclusive_pass(by_group: dict[str, Agreement]) -> bool:
        """True when nothing conclusive failed, and something was conclusive."""
        verdicts = [a.verdict for a in by_group.values()]
        return "FAIL" not in verdicts and "PASS" in verdicts

    @staticmethod
    def render(by_group: dict[str, Agreement]) -> str:
        head = (
            "| Group | n | Pearson r | Spearman | Exact | Within 1 | MAE | "
            "Bias | Verdict |\n|---|---|---|---|---|---|---|---|---|\n"
        )

        def f(v, spec=".3f"):
            return "--" if v is None else format(v, spec)

        rows = []
        for name, a in by_group.items():
            rows.append(
                f"| {name} | {a.n} | {f(a.pearson)} | {f(a.spearman)} | "
                f"{a.exact_pct:.1f}% | {a.within_1_pct:.1f}% | {f(a.mae, '.2f')} | "
                f"{'' if a.bias is None else '+' if a.bias >= 0 else ''}"
                f"{f(a.bias, '.2f')} | "
                f"{a.verdict} |"
            )
        return head + "\n".join(rows) + "\n"


def agreement(candidate: Sequence[float], reference: Sequence[float]) -> Agreement:
    """Compare a candidate judge's scores against the reference judge's."""
    if len(candidate) != len(reference):
        raise ValueError(
            f"candidate and reference must be the same length: "
            f"{len(candidate)} != {len(reference)}"
        )
    n = len(candidate)
    if n == 0:
        return Agreement(0, 0, 0, None, None, None, None, None, None)

    diffs = [c - r for c, r in zip(candidate, reference)]
    return Agreement(
        n=n,
        exact=sum(1 for d in diffs if d == 0),
        within_1=sum(1 for d in diffs if abs(d) <= 1),
        mae=_mean([abs(d) for d in diffs]),
        bias=_mean(diffs),
        pearson=pearson(candidate, reference),
        spearman=spearman(candidate, reference),
        candidate_mean=_mean(candidate),
        reference_mean=_mean(reference),
    )
