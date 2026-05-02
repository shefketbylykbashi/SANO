# stats.py
"""Statistical tests used in Section VI: Mann-Whitney U + Cliff's delta."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

try:
    from scipy.stats import mannwhitneyu  # type: ignore
except Exception:  # pragma: no cover
    mannwhitneyu = None


@dataclass
class StatResult:
    metric: str
    system_a: str
    system_b: str
    n_a: int
    n_b: int
    u: float
    p_value: float
    cliffs_delta: float
    magnitude: str


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    n_a, n_b = len(a), len(b)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (n_a * n_b)


def magnitude_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def compare(a: Sequence[float], b: Sequence[float], *,
            metric: str, system_a: str, system_b: str,
            alternative: str = "two-sided") -> StatResult:
    if mannwhitneyu is None:
        raise RuntimeError("scipy required for Mann-Whitney U")
    res = mannwhitneyu(a, b, alternative=alternative)
    d = cliffs_delta(a, b)
    return StatResult(metric=metric, system_a=system_a, system_b=system_b,
                      n_a=len(a), n_b=len(b),
                      u=float(res.statistic), p_value=float(res.pvalue),
                      cliffs_delta=d, magnitude=magnitude_label(d))