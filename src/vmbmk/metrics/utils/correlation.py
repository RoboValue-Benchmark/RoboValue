"""Pure Spearman calculations with explicit tie and constant conventions."""
from __future__ import annotations

import math
from typing import Sequence
from vmbmk.errors import VMBMKError


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end + 1) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks

def _validate_inputs(left: Sequence[float], right: Sequence[float], metric: str) -> None:
    if len(left) != len(right) or len(left) < 2:
        raise VMBMKError(f"{metric} Spearman requires equal lengths and at least 2 points")
    if any(not math.isfinite(value) for values in (left, right) for value in values):
        raise VMBMKError(f"{metric} Spearman requires finite values")


def spearman(left: Sequence[float], right: Sequence[float], metric: str) -> float:
    _validate_inputs(left, right, metric)
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    left_var = sum((value - left_mean) ** 2 for value in left_ranks)
    right_var = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_var * right_var)
    if denominator == 0:
        raise VMBMKError(f"{metric} Spearman is undefined for constant values")
    return numerator / denominator

def spearman_or_zero(left: Sequence[float], right: Sequence[float], metric: str) -> float:
    """Score a VOC episode, mapping an undefined constant correlation to zero."""
    _validate_inputs(left, right, metric)
    if len(set(left)) < 2 or len(set(right)) < 2:
        return 0.0
    return spearman(left, right, metric)
