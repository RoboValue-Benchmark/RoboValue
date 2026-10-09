"""Conservative compatibility records for incremental task-level evaluation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from vmbmk.data.dataset import Dataset
from .run import RunConfig
from vmbmk.metrics.registry import METRICS, build_metric_queries
from vmbmk.inference.queries import query_state


CONTRACT_VERSION = "batch-resume-v2"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _implementation_digest(metric: str, adapter: str) -> str:
    """Hash explicit scoring/model dependencies, not unrelated implementations."""
    package = Path(__file__).resolve().parents[1]
    metric_modules = {METRICS[metric].module, "utils.statistics", "utils.results", "utils.aggregation"}
    if metric in {"voc", "voc_mem", "cycle_voc"}:
        metric_modules.add("utils.correlation")
    if metric == "sia":
        metric_modules.update({"sia.scoring", "sia.judge", "sia.config"})
    if metric == "vs":
        metric_modules.update({
            "cycle_voc_vs.vs_scoring", "cycle_voc_vs.cycle", "utils.correlation",
        })
    if metric in {"cycle_voc", "vs"}:
        metric_modules.add("cycle_voc_vs.planning")
    if metric == "voc":
        metric_modules.update({"cycle_voc_vs.cycle", "cycle_voc_vs.planning"})
    adapter_modules = {adapter, "base", "video_inputs", "__init__"}
    if adapter == "remote_api":
        adapter_modules.update({"robometer", "rynnvalue", "procvlm"})
    paths = {
        package / "metrics" / (name.replace(".", "/") + ".py")
        for name in metric_modules
    }
    paths.update(package / "adapters" / f"{name}.py" for name in adapter_modules)
    paths.update(package / name for name in (
        "data/dataset.py", "data/playback.py", "inference/queries.py",
        "inference/execution.py", "inference/worker.py", "inference/dispatch.py", "serialization.py",
        "runner/run.py", "runner/provenance.py", "runner/policy.py", "metrics/registry.py",
        "adapters/registry.py", "errors.py",
        "data/__init__.py", "inference/__init__.py", "runner/__init__.py",
        "metrics/__init__.py",
    ))
    if metric in {"voc", "voc_mem", "cycle_voc"}:
        paths.add(package / "metrics/utils/__init__.py")
    elif metric == "sia":
        paths.add(package / "metrics/sia/__init__.py")
    if metric in {"voc", "cycle_voc", "vs"}:
        paths.update({
            package / "metrics/cycle_voc_vs/__init__.py",
            package / "metrics/utils/__init__.py",
        })
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resume_contract(
    config_path: Path, dataset: Dataset, metric: str,
    task_ids: Sequence[str], candidate_tasks: Sequence[str] | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Fingerprint effective settings, planner code and task/domain query inputs."""
    config = RunConfig.load(config_path)
    if data_root is not None:
        config = replace(config, data=data_root.resolve())
    mode, domains = next((mode, domains) for name, mode, domains in config.metrics if name == metric)
    model = config.model_config()
    model.pop("batch_size", None)
    code_identity = _implementation_digest(metric, "remote_api" if config.backend == "remote_api" else config.model)
    identity = _digest({
        "version": CONTRACT_VERSION, "metric": metric, "mode": mode,
        "model": model, "data": str(config.data), "code": code_identity,
        "candidate_tasks": sorted(candidate_tasks or ()) if metric in {"tga_easy", "tga_hard"} else None,
    })
    planned = build_metric_queries(
        metric, dataset, task_ids, domains, candidate_tasks=candidate_tasks,
    )
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for query in planned:
        state = query_state(query)
        episode = dataset.episode(state.task_id, state.episode_id)
        row = query.to_dict()
        if metric == "trr":
            row["query_id"] = ":".join((state.task_id, state.episode_id, *query.query_id.split(":")[2:]))
        grouped.setdefault(episode.domain, {}).setdefault(state.task_id, []).append(row)
    fingerprints: dict[str, dict[str, str]] = {}
    for domain, tasks in grouped.items():
        fingerprints[domain] = {}
        for task, queries in tasks.items():
            task_root = config.data / task
            metadata = [task_root / "metadata.json"]
            for episode_id, episode in sorted(dataset.tasks[task].episodes.items()):
                if episode.domain == domain:
                    episode_root = task_root / "episodes" / episode_id
                    metadata.extend((episode_root / "metadata.json", episode_root / "annotation.json"))
            inputs = [(str(path.relative_to(task_root)), path.read_text(encoding="utf-8")) for path in metadata]
            fingerprints[domain][task] = _digest({"queries": queries, "inputs": inputs})
    return {"version": CONTRACT_VERSION, "identity": identity, "coverage": fingerprints}


def missing_tasks(
    summary_path: Path, requested: dict[str, Any], tasks: Sequence[str],
) -> tuple[str, ...]:
    """Reuse only recorded compatible cells; legacy/changed summaries are errors."""
    if not summary_path.exists():
        return tuple(tasks)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"published summary must be an object: {summary_path}")
    previous = summary.get("resume_contract")
    if not isinstance(previous, dict) or previous.get("version") != CONTRACT_VERSION:
        raise ValueError(f"published result has no compatible resume contract: {summary_path}; rerun explicitly")
    if previous.get("identity") != requested["identity"]:
        raise ValueError(f"published configuration or protocol is incompatible: {summary_path}; rerun explicitly")
    result_domains = summary.get("result", {}).get("domains")
    coverage = previous.get("coverage")
    if not isinstance(result_domains, dict) or not isinstance(coverage, dict):
        raise ValueError(f"published summary has no task/domain coverage: {summary_path}")
    missing = []
    for task in tasks:
        expected = {domain: rows[task] for domain, rows in requested["coverage"].items() if task in rows}
        complete = bool(expected)
        for domain, fingerprint in expected.items():
            old = previous.get("coverage", {}).get(domain, {}).get(task)
            measured = task in result_domains.get(domain, {}).get("tasks", {})
            if old is not None and old != fingerprint:
                raise ValueError(f"published query/data inputs changed for {task}/{domain}; rerun explicitly")
            complete = complete and measured and old == fingerprint
        if not complete:
            missing.append(task)
    return tuple(missing)
