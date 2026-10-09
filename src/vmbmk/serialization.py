from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def read_yaml_object(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping with source context and no metadata fallback."""
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"unable to read YAML config {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a YAML object")
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object, retaining source context in parse errors."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return value


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects in file order, ignoring blank lines."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write compact UTF-8 objects in iteration order with LF delimiters."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def replace_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file using a sibling temporary file."""
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    try:
        write_jsonl(temporary, rows)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
