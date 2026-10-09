from __future__ import annotations

import math
from abc import ABC
from typing import Any, Callable, Mapping, Sequence

from vmbmk.errors import ConfigurationError
from vmbmk.inference.queries import CompareQuery, SubtaskQuery, ValueQuery


def compare_value_difference(
    queries: Sequence[CompareQuery],
    value: Callable[[Sequence[ValueQuery]], list[float]],
) -> list[float]:
    """Compare value-based models with all left states before all right states."""
    if not queries:
        return []
    left = [
        ValueQuery(f"{query.query_id}:a", query.state_a, query.instruction)
        for query in queries
    ]
    right = [
        ValueQuery(f"{query.query_id}:b", query.state_b, query.instruction)
        for query in queries
    ]
    values = value([*left, *right])
    split = len(queries)
    return [values[split + index] - values[index] for index in range(split)]


def require_config(
    config: Mapping[str, Any], adapter: str, optional_fields: set[str]
) -> None:
    expected = {"adapter", "python", "checkpoint", *optional_fields}
    missing = sorted({"adapter", "python", "checkpoint"} - set(config))
    unknown = sorted(set(config) - expected)
    if missing or unknown:
        raise ConfigurationError(
            f"{adapter} config fields do not match; "
            f"missing={missing}, unknown={unknown}"
        )
    if config.get("adapter") != adapter:
        raise ConfigurationError(
            f"config adapter must be {adapter!r}; got {config.get('adapter')!r}"
        )


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ConfigurationError(f"{label} must be a positive number")
    return result


class Adapter(ABC):
    """Ordered inference contract implemented by isolated model adapters."""

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the value interface"
        )

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the compare interface"
        )

    def tga(self, queries: Sequence[CompareQuery]) -> list[float]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the native TGA interface"
        )

    def subtask(self, queries: Sequence[SubtaskQuery]) -> list[str]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the subtask interface"
        )
