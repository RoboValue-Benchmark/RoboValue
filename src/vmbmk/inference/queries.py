from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from vmbmk.errors import VMBMKError
from vmbmk.serialization import read_jsonl


def _keys(row: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(row))
    unknown = sorted(set(row) - expected)
    if missing or unknown:
        raise VMBMKError(
            f"{label} fields do not match; missing={missing}, unknown={unknown}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VMBMKError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class StateRef:
    """Identify an episode anchor; the Adapter determines its history context."""

    task_id: str
    episode_id: str
    anchor_frame: int

    @classmethod
    def from_dict(cls, row: Any, label: str) -> "StateRef":
        if not isinstance(row, dict):
            raise VMBMKError(f"{label} must be an object")
        _keys(row, {"task_id", "episode_id", "anchor_frame"}, label)
        frame = row["anchor_frame"]
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise VMBMKError(f"{label}.anchor_frame must be an integer")
        return cls(
            _text(row["task_id"], f"{label}.task_id"),
            _text(row["episode_id"], f"{label}.episode_id"),
            frame,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "anchor_frame": self.anchor_frame,
        }


@dataclass(frozen=True)
class ValueQuery:
    query_id: str
    state: StateRef
    instruction: str
    playback: Literal["forward", "reverse", "cycle"] = "forward"
    op: Literal["value"] = "value"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "op": self.op,
            "state": self.state.to_dict(),
            "instruction": self.instruction,
            "playback": self.playback,
        }


@dataclass(frozen=True)
class CompareQuery:
    """Compare ordered states; value-difference adapters return value(b) - value(a)."""

    query_id: str
    state_a: StateRef
    state_b: StateRef
    instruction: str
    op: Literal["compare"] = "compare"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "op": self.op,
            "state_a": self.state_a.to_dict(),
            "state_b": self.state_b.to_dict(),
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class SubtaskQuery:
    query_id: str
    state: StateRef
    instruction: str
    op: Literal["subtask"] = "subtask"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "op": self.op,
            "state": self.state.to_dict(),
            "instruction": self.instruction,
        }


Query: TypeAlias = ValueQuery | CompareQuery | SubtaskQuery


def query_state(query: Query) -> StateRef:
    """Return the source state that owns a query and its task/domain coverage."""
    return query.state_a if isinstance(query, CompareQuery) else query.state


def query_from_dict(row: dict[str, Any], label: str) -> Query:
    op = row.get("op")
    if op == "value":
        expected = {"query_id", "op", "state", "instruction", "playback"}
        unknown = sorted(set(row) - expected)
        missing = sorted((expected - {"playback"}) - set(row))
        if missing or unknown:
            raise VMBMKError(
                f"{label} fields do not match; missing={missing}, unknown={unknown}"
            )
        playback = row.get("playback", "forward")
        if playback not in {"forward", "reverse", "cycle"}:
            raise VMBMKError(
                f"{label}.playback must be 'forward', 'reverse', or 'cycle'"
            )
        return ValueQuery(
            _text(row["query_id"], f"{label}.query_id"),
            StateRef.from_dict(row["state"], f"{label}.state"),
            _text(row["instruction"], f"{label}.instruction"),
            playback,
        )
    if op == "compare":
        _keys(
            row,
            {"query_id", "op", "state_a", "state_b", "instruction"},
            label,
        )
        return CompareQuery(
            _text(row["query_id"], f"{label}.query_id"),
            StateRef.from_dict(row["state_a"], f"{label}.state_a"),
            StateRef.from_dict(row["state_b"], f"{label}.state_b"),
            _text(row["instruction"], f"{label}.instruction"),
        )
    if op == "subtask":
        _keys(row, {"query_id", "op", "state", "instruction"}, label)
        return SubtaskQuery(
            _text(row["query_id"], f"{label}.query_id"),
            StateRef.from_dict(row["state"], f"{label}.state"),
            _text(row["instruction"], f"{label}.instruction"),
        )
    raise VMBMKError(f"{label}.op must be 'value', 'compare', or 'subtask'")


def read_queries(path: str | Path) -> list[Query]:
    queries = [
        query_from_dict(row, f"{path}:{index}")
        for index, row in enumerate(read_jsonl(path), start=1)
    ]
    if not queries:
        raise VMBMKError(f"query file is empty: {path}")
    identifiers = [query.query_id for query in queries]
    if len(identifiers) != len(set(identifiers)):
        raise VMBMKError(f"query file contains duplicate query_id values: {path}")
    return queries


@dataclass(frozen=True)
class Result:
    query_id: str
    op: Literal["value", "compare", "subtask"]
    score: float | None = None
    output: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any], label: str) -> "Result":
        op = row.get("op")
        expected = (
            {"query_id", "op", "output"}
            if op == "subtask"
            else {"query_id", "op", "score"}
        )
        _keys(row, expected, label)
        if op not in {"value", "compare", "subtask"}:
            raise VMBMKError(f"{label}.op is invalid")
        if op == "subtask":
            return cls(
                _text(row["query_id"], f"{label}.query_id"),
                op,
                output=_text(row["output"], f"{label}.output"),
            )
        score = row["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise VMBMKError(f"{label}.score must be a number")
        score = float(score)
        if not math.isfinite(score):
            raise VMBMKError(f"{label}.score must be finite")
        return cls(_text(row["query_id"], f"{label}.query_id"), op, score=score)

    def to_dict(self) -> dict[str, Any]:
        if self.op == "subtask":
            return {"query_id": self.query_id, "op": self.op, "output": self.output}
        return {"query_id": self.query_id, "op": self.op, "score": self.score}


def read_results(path: str | Path) -> list[Result]:
    return [
        Result.from_dict(row, f"{path}:{index}")
        for index, row in enumerate(read_jsonl(path), start=1)
    ]
