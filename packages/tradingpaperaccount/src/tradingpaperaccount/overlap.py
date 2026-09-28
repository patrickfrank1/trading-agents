"""Cross-portfolio overlap math (pure, no SDK imports).

Measures how much two long-only books have in common. The default metric is
``nav_overlap``: the sum, over shared tickers, of the smaller of the two NAV
weights — i.e. the fraction of NAV invested in the same names by both accounts.
A value of ``0.10`` means 10% of NAV is duplicated across the two portfolios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class OverlapResult:
    """Overlap between two weight vectors (``symbol -> fraction of NAV``)."""

    shared: tuple[str, ...]
    count_a: int
    count_b: int
    union_count: int
    nav_overlap: float
    invested_overlap: float
    jaccard: float
    shared_fraction_smaller: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared": list(self.shared),
            "shared_count": len(self.shared),
            "count_a": self.count_a,
            "count_b": self.count_b,
            "union_count": self.union_count,
            "nav_overlap": self.nav_overlap,
            "invested_overlap": self.invested_overlap,
            "jaccard": self.jaccard,
            "shared_fraction_smaller": self.shared_fraction_smaller,
        }


def _long_weights(weights: Mapping[str, float]) -> dict[str, float]:
    return {str(symbol): float(value) for symbol, value in weights.items() if value > 0}


def portfolio_overlap(
    weights_a: Mapping[str, float],
    weights_b: Mapping[str, float],
) -> OverlapResult:
    """Compute overlap between two long-only weight vectors.

    Values in ``weights_a`` / ``weights_b`` are expected to be fractions of NAV
    (they need not sum to 1). Shorts (negative weights) are ignored.
    """
    long_a = _long_weights(weights_a)
    long_b = _long_weights(weights_b)

    total_a = sum(long_a.values())
    total_b = sum(long_b.values())
    shared = tuple(sorted(set(long_a) & set(long_b)))
    union = set(long_a) | set(long_b)

    nav_overlap = sum(min(long_a[s], long_b[s]) for s in shared)

    if total_a > 0 and total_b > 0:
        invested_overlap = sum(
            min(long_a[s] / total_a, long_b[s] / total_b) for s in shared
        )
    else:
        invested_overlap = 0.0

    jaccard = len(shared) / len(union) if union else 0.0
    smaller = min(len(long_a), len(long_b))
    shared_fraction_smaller = len(shared) / smaller if smaller else 0.0

    return OverlapResult(
        shared=shared,
        count_a=len(long_a),
        count_b=len(long_b),
        union_count=len(union),
        nav_overlap=nav_overlap,
        invested_overlap=invested_overlap,
        jaccard=jaccard,
        shared_fraction_smaller=shared_fraction_smaller,
    )


DEFAULT_METRIC = "nav_overlap"

_METRICS = ("nav_overlap", "invested_overlap", "jaccard", "shared_fraction_smaller")


def metric_value(result: OverlapResult, metric: str = DEFAULT_METRIC) -> float:
    """Return one of the overlap metrics by name."""
    if metric not in _METRICS:
        raise ValueError(f"unknown overlap metric: {metric!r} (expected one of {_METRICS})")
    return float(getattr(result, metric))


__all__ = [
    "DEFAULT_METRIC",
    "OverlapResult",
    "metric_value",
    "portfolio_overlap",
]
