#!/usr/bin/env python3
"""Inspect configured runs without launching inference or publishing results."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from vmbmk.data.dataset import Dataset
from vmbmk.runner.run import RunConfig
from vmbmk.metrics.registry import QUERY_METRICS
from .validation import inspect_result


def build_summary(
    config_root: Path,
    results_root: Path,
    dataset: Path,
    *,
    paused: bool = False,
) -> dict[str, Any]:
    configs = sorted(
        path
        for path in config_root.glob("*.yaml")
        if path.name != "generalconfigs.yaml"
    )
    if not configs:
        raise ValueError(f"no run configs found in {config_root}")

    selections = {path: RunConfig.load(path) for path in configs}
    metric_names = sorted({metric for config in selections.values() for metric, _, _ in config.metrics})
    loaded_dataset = Dataset.load(dataset, metrics=metric_names)
    completed: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    remaining: list[str] = []
    unsupported: list[dict[str, Any]] = []
    for config_path in configs:
        stem = config_path.stem
        unsupported_metrics = sorted({metric for metric, _, _ in selections[config_path].metrics} - QUERY_METRICS)
        if unsupported_metrics:
            unsupported.append({"run": stem, "metrics": unsupported_metrics})
            continue
        run_dir = results_root / stem
        details = inspect_result(config_path, run_dir, loaded_dataset)
        row = {"run": stem, "result_dir": str(run_dir), **details}
        row.pop("valid")
        row.pop("state")
        if details["valid"]:
            completed.append(row)
        else:
            remaining.append(stem)
            if details["state"] == "stale":
                stale.append(row)

    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "status": (
            "complete"
            if not remaining and not unsupported
            else "unsupported" if unsupported and not remaining
            else "evaluation_paused" if paused else "incomplete"
        ),
        "expected_runs": len(configs),
        "completed_count": len(completed),
        "remaining_count": len(remaining),
        "stale_count": len(stale),
        "unsupported_runs": unsupported,
        "remaining_runs": remaining,
        "completed_runs": completed,
        "stale_runs": stale,
    }


def status_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--paused",
        action="store_true",
        help="record an incomplete evaluation as paused rather than running",
    )
    args = parser.parse_args(argv)

    summary = build_summary(
        args.config_root.resolve(),
        args.results_root.resolve(),
        args.dataset.resolve(),
        paused=args.paused,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(output)
    return 0
