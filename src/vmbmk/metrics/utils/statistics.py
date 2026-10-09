from __future__ import annotations

import math
from typing import Sequence
from vmbmk.errors import VMBMKError


def edge_frames(num_frames: int) -> tuple[list[int], list[int]]:
    count = max(1, math.ceil(num_frames * 0.02))
    return list(range(count)), list(range(num_frames - count, num_frames))

def mean(values: Sequence[float]) -> float:
    if not values:
        raise VMBMKError("mean requires at least one value")
    return sum(float(value) for value in values) / len(values)

def percentile_higher(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise VMBMKError("percentile requires values and a quantile in [0, 1]")
    selected = sorted(float(value) for value in values)
    return selected[max(0, math.ceil(quantile * len(selected)) - 1)]
