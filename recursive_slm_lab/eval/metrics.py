from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class MetricSummary:
    pass_at_1: float
    pass_at_k: float


def compute_pass_rates(outcomes: Iterable[list[bool]], k: int) -> MetricSummary:
    outcomes = list(outcomes)
    if not outcomes:
        return MetricSummary(pass_at_1=0.0, pass_at_k=0.0)
    pass_at_1 = sum(1 for outcome in outcomes if outcome[:1] and outcome[0]) / len(outcomes)
    pass_at_k = sum(1 for outcome in outcomes if any(outcome[:k])) / len(outcomes)
    return MetricSummary(pass_at_1=pass_at_1, pass_at_k=pass_at_k)
