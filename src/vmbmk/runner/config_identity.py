"""Shared configuration identity for completed and interrupted evaluations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def resolved_run_config_paths(config: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Preserve the config-relative meaning of documented top-level paths."""
    resolved = deepcopy(dict(config))
    for key in ("python", "checkpoint", "data", "output", "reference_data", "sia_responses"):
        value = resolved.get(key)
        if value is not None:
            path = Path(value)
            resolved[key] = str(path if path.is_absolute() else (directory / path).resolve())
    return resolved


def normalized_result_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Ignore only recorded execution controls that permit result reuse."""
    normalized = deepcopy(dict(config))
    normalized.pop("gpu", None)
    normalized.pop("batch_size", None)
    options = normalized.get("model_options")
    if isinstance(options, dict):
        options.pop("sia_workers", None)
        options.pop("sia_timeout", None)
    return normalized
