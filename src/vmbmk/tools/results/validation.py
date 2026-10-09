#!/usr/bin/env python3
"""Validate completed runs against configuration, query identity and coverage."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from vmbmk.data.dataset import Dataset
from vmbmk.inference.queries import read_results
from vmbmk.errors import VMBMKError
from vmbmk.runner.run import RunConfig
from vmbmk.metrics.registry import QUERY_METRICS, build_metric_queries, selected_task_ids
from vmbmk.metrics.csvc import CSVC_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.vs import VS_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.forward import VOC_PROTOCOL
from vmbmk.inference.queries import query_state


def operation_count(path: Path) -> int:
    with path.open("rb") as rows:
        return sum(1 for _ in rows)


def metric_means(metrics: dict[str, Any], *, allow_missing: bool = False) -> dict[str, float | None]:
    means: dict[str, float | None] = {}
    for name, row in metrics.items():
        if not isinstance(row, dict):
            raise ValueError(f"metrics.{name}: expected an object")
        value = row.get("mean")
        if value is None and allow_missing and "mean" in row:
            means[name] = None
            continue
        if value is None and name == "csvc" and row.get("score") is None and "score" in row:
            means[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metrics.{name}.mean: expected a number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"metrics.{name}.mean: expected a finite number")
        means[name] = value
    return means


def _load_yaml(path: Path) -> dict[str, Any]:
    row = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError(f"{path}: expected a YAML object")
    return row


def _normalized_config(path: Path) -> dict[str, Any]:
    row = _load_yaml(path)
    # Completion lanes may remap a run to another physical GPU.
    row.pop("gpu", None)
    return row


def expected_result(
    config_path: Path, dataset: Dataset
) -> tuple[int, dict[str, dict[str, set[str]]]]:
    """Derive operation counts and coverage from the registered metric planners."""
    config = RunConfig.load(config_path)
    unsupported = sorted({metric for metric, _, _ in config.metrics} - QUERY_METRICS)
    if unsupported:
        raise ValueError(f"query validation is unsupported for metrics: {unsupported}")
    expected_operations = 0
    expected_tasks: dict[str, dict[str, set[str]]] = {}
    for metric, _, domains in config.metrics:
        requested = (config.metric_tasks or {}).get(metric, config.tasks)
        tasks = selected_task_ids(metric, dataset, requested, domains)
        queries = build_metric_queries(
            metric, dataset, tasks, domains,
            candidate_tasks=(config.candidate_tasks or {}).get(metric),
        )
        expected_operations += len(queries)
        coverage: dict[str, set[str]] = {domain: set() for domain in domains}
        for query in queries:
            state = query_state(query)
            domain = dataset.episode(state.task_id, state.episode_id).domain
            coverage[domain].add(state.task_id)
        if metric in {"vs", "fpl", "trr", "csvc"}:
            coverage = {domain: tasks for domain, tasks in coverage.items() if tasks}
        expected_tasks[metric] = coverage
    return expected_operations, expected_tasks


def _validate_metric_coverage(
    metrics: dict[str, Any], expected: dict[str, dict[str, set[str]]]
) -> None:
    if set(metrics) != set(expected):
        raise ValueError(
            f"metric mismatch: expected={sorted(expected)}, actual={sorted(metrics)}"
        )
    metric_means(metrics)
    for metric, domains in expected.items():
        row = metrics[metric]
        if metric == "csvc" and row.get("protocol") != CSVC_PROTOCOL:
            raise ValueError("CSVC result uses an obsolete scoring protocol")
        if metric == "voc" and row.get("protocol") != VOC_PROTOCOL:
            raise ValueError("VOC result predates cycle-forward scoring")
        if metric == "vs" and row.get("protocol") != VS_PROTOCOL:
            raise ValueError("VS result uses an obsolete input protocol")
        actual_domains = row.get("domains")
        if not isinstance(actual_domains, dict):
            raise ValueError(f"metrics.{metric}.domains: expected an object")
        if set(actual_domains) != set(domains):
            raise ValueError(
                f"metrics.{metric} domain mismatch: expected={sorted(domains)}, "
                f"actual={sorted(actual_domains)}"
            )
        for domain, task_ids in domains.items():
            domain_row = actual_domains[domain]
            if not isinstance(domain_row, dict):
                raise ValueError(
                    f"metrics.{metric}.domains.{domain}: expected an object"
                )
            tasks = domain_row.get("tasks")
            if not isinstance(tasks, dict):
                raise ValueError(
                    f"metrics.{metric}.domains.{domain}.tasks: expected an object"
                )
            if set(tasks) != task_ids:
                raise ValueError(
                    f"metrics.{metric}.{domain} task mismatch: "
                    f"expected={sorted(task_ids)}, actual={sorted(tasks)}"
                )
            scores = {f"{metric}.{domain}": domain_row}
            metric_means(scores, allow_missing=metric == "csvc")
            for task_id, score in tasks.items():
                if metric == "csvc" and score is None:
                    if row["task_results"][task_id].get("score") is not None:
                        raise ValueError(f"CSVC missing score disagrees with task result: {task_id}")
                    continue
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise ValueError(
                        f"metrics.{metric}.{domain}.{task_id}: "
                        "expected a finite number"
                    )



def validate_operation_identity(
    config_path: Path,
    operations_path: Path,
    dataset: Dataset,
    expected_tasks: Mapping[str, Mapping[str, Sequence[str]]],
) -> None:
    config = RunConfig.load(config_path)
    expected = {}
    for metric, _, domains in config.metrics:
        tasks = sorted(set().union(*expected_tasks[metric].values()))
        queries = build_metric_queries(
            metric, dataset, tasks, domains,
            candidate_tasks=(config.candidate_tasks or {}).get(metric),
        )
        for query in queries:
            if query.query_id in expected:
                raise ValueError(f"duplicate planned query: {query.query_id}")
            expected[query.query_id] = query.op
    seen = set()
    for result in read_results(operations_path):
        if result.query_id in seen:
            raise ValueError(f"duplicate result query: {result.query_id}")
        if result.query_id not in expected:
            raise ValueError(f"unexpected result query: {result.query_id}")
        if result.op != expected[result.query_id]:
            raise ValueError(f"incorrect operation for query: {result.query_id}")
        seen.add(result.query_id)
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"missing result queries: {len(missing)}, sample={sorted(missing)[:3]}")


def inspect_result(
    config_path: Path, run_dir: Path, dataset: Dataset
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    operations_path = run_dir / "operations.jsonl"
    saved_config_path = run_dir / "config.yaml"
    if not run_dir.exists():
        return {"valid": False, "state": "missing", "reason": "result directory missing"}
    missing = [
        path.name
        for path in (saved_config_path, operations_path, metrics_path)
        if not path.is_file()
    ]
    if missing:
        return {
            "valid": False,
            "state": "stale",
            "reason": f"missing result files: {', '.join(missing)}",
        }
    try:
        if _normalized_config(saved_config_path) != _normalized_config(config_path):
            raise ValueError("saved config does not match current config (excluding gpu)")
        expected_operations, expected_tasks = expected_result(config_path, dataset)
        actual_operations = operation_count(operations_path)
        if actual_operations != expected_operations:
            raise ValueError(
                f"operation count mismatch: expected={expected_operations}, "
                f"actual={actual_operations}"
            )
        validate_operation_identity(config_path, operations_path, dataset, expected_tasks)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json: expected an object")
        _validate_metric_coverage(metrics, expected_tasks)
        return {
            "valid": True,
            "state": "complete",
            "operations": actual_operations,
            "metrics": metric_means(metrics),
        }
    except (KeyError, OSError, TypeError, ValueError, VMBMKError, yaml.YAMLError) as exc:
        details: dict[str, Any] = {
            "valid": False,
            "state": "stale",
            "reason": str(exc),
        }
        if operations_path.is_file():
            details["operations"] = operation_count(operations_path)
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(metrics, dict):
                    details["metrics"] = metric_means(metrics)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return details



def validate_run(config_path: Path, run_dir: Path, data_root: Path) -> dict[str, Any]:
    """Inspect saved artifacts using the configured metric-scoped dataset."""
    config = RunConfig.load(config_path)
    dataset = Dataset.load(data_root, metrics=[metric for metric, _, _ in config.metrics])
    return inspect_result(config_path, run_dir, dataset)
