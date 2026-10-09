#!/usr/bin/env python3
"""Batch configured baselines with GPU scheduling, retries and incremental publication."""

from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from vmbmk.data.dataset import Dataset
from vmbmk.metrics.registry import (
    QUERY_METRICS,
    build_metric_queries,
    canonical_metric,
    dataset_metric_name,
    task_query_reason,
)
from .policy import evaluation_policy
from vmbmk.adapters.registry import canonical_baseline
from .resume import missing_tasks, resume_contract


DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_FINAL_RESULTS_ROOT = Path("final_results")
METRIC_DATA_NAMES = {metric: dataset_metric_name(metric) for metric in sorted(QUERY_METRICS)}


@dataclass
class Job:
    baseline: str
    metrics: tuple[str, ...]
    tasks: tuple[str, ...]
    template_path: Path
    config_path: Path
    run_name: str
    metric_tasks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Full task vocabulary used to construct TGA-easy negatives.  This can
    # be larger than metric_tasks during an incremental repair.
    candidate_tasks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    status: str = "pending"
    gpu: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result_dir: str | None = None
    result: dict[str, Any] | None = None
    attempts: int = 0
    resume_contracts: dict[str, dict[str, Any]] = field(default_factory=dict)
    execution_mode: str = "normal"
    publication_baseline: str | None = None

    def manifest_row(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "publication_baseline": self.publication_baseline or self.baseline,
            "metrics": list(self.metrics),
            "tasks": list(self.tasks),
            "template": str(self.template_path),
            "config": str(self.config_path),
            "run_name": self.run_name,
            "metric_tasks": {
                metric: list(self.metric_tasks.get(metric, self.tasks))
                for metric in self.metrics
            },
            "candidate_tasks": {
                metric: list(self.candidate_tasks.get(metric, self.tasks))
                for metric in self.metrics
            },
            "status": self.status,
            "gpu": self.gpu,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_dir": self.result_dir,
            "result": self.result,
            "attempts": self.attempts,
            "execution_mode": self.execution_mode,
        }


from vmbmk.tools.results.publication import (
    atomic_json,
    now_iso,
    publish_result,
    PublicationRequest,
)


def split_values(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(item for item in value.split(",") if item)
    return result


def normalize_task(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_metric(value: str) -> str:
    return canonical_metric(value)


def expand_metrics(values: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for value in split_values(values):
        metric = normalize_metric(value)
        if metric == "tga":
            expanded.extend(("tga_easy", "tga_hard"))
        else:
            expanded.append(metric)
    return list(dict.fromkeys(expanded))


def discover_templates(config_root: Path) -> dict[str, Path]:
    templates: dict[str, Path] = {}
    for path in sorted(config_root.glob("*.yaml")):
        baseline = path.stem.removesuffix("configs")
        if baseline == "general":
            continue
        templates[baseline] = path.resolve()
    return templates


def select_names(
    raw_values: Iterable[str], available: Iterable[str], *, exclude_from_all: Iterable[str] = ()
) -> list[str]:
    choices = sorted(available)
    values = split_values(raw_values)
    if values == ["all"]:
        excluded = set(exclude_from_all)
        return [value for value in choices if value not in excluded]
    unknown = sorted(set(values) - set(choices))
    if unknown:
        raise ValueError(f"unknown values {unknown}; available={choices}")
    return list(dict.fromkeys(values))







def _planned_job(
    baseline: str, publication_baseline: str, metrics: list[str],
    metric_tasks: dict[str, tuple[str, ...]], candidate_tasks: dict[str, tuple[str, ...]],
    template_path: Path, config_dir: Path,
) -> Job:
    """Build one execution job without duplicating coalesced/single-metric layouts."""
    tasks = tuple(dict.fromkeys(task for metric in metrics for task in metric_tasks[metric]))
    run_name = f"{baseline}__{'_'.join(metrics)}"
    return Job(
        baseline=baseline, publication_baseline=publication_baseline,
        metrics=tuple(metrics), tasks=tasks, template_path=template_path,
        config_path=config_dir / f"{run_name}.yaml", run_name=run_name,
        metric_tasks={metric: metric_tasks[metric] for metric in metrics},
        candidate_tasks={
            metric: candidate_tasks.get(metric, metric_tasks[metric])
            if metric in {"tga_easy", "tga_hard"} else metric_tasks[metric]
            for metric in metrics
        },
    )


def plan_metric_tasks(
    dataset: Dataset,
    baselines: list[str],
    tasks: list[str],
    metrics: list[str],
    templates: dict[str, Path],
    config_dir: Path,
    *,
    coalesce_metrics: bool = False,
    incremental: bool = False,
    execution_mode: str = "normal",
    final_results_root: Path = DEFAULT_FINAL_RESULTS_ROOT,
    data_root: Path | None = None,
) -> tuple[list[Job], list[dict[str, str]]]:
    jobs: list[Job] = []
    unsupported = sorted(set(metrics) - set(METRIC_DATA_NAMES))
    if unsupported:
        raise ValueError(f"unsupported batch metrics: {unsupported}; use single-config run for SIA")
    if execution_mode not in {"normal", "missing", "rerun"}:
        raise ValueError(f"invalid execution mode: {execution_mode}")
    if incremental and execution_mode == "rerun":
        raise ValueError("incremental cannot be combined with rerun")
    if incremental:
        execution_mode = "missing"
    skipped: list[dict[str, str]] = []
    for baseline in baselines:
        template = yaml.safe_load(templates[baseline].read_text(encoding="utf-8"))
        if not isinstance(template, dict):
            raise ValueError(f"{templates[baseline]} must contain a YAML object")
        canonical_name = canonical_baseline(template, fallback=baseline)
        policy = evaluation_policy(template)
        publication_baseline = canonical_name
        metric_to_tasks: dict[str, tuple[str, ...]] = {}
        metric_all_tasks: dict[str, tuple[str, ...]] = {}
        for metric in metrics:
            compatible: list[str] = []
            selected_tasks = template.get("metric_tasks", {}).get(metric, tasks)
            tga_candidates = template.get("candidate_tasks", {}).get(metric)
            for task_id in tasks:
                if task_id not in selected_tasks:
                    continue
                reason = policy.reason(template, metric, task_id)
                if reason is not None:
                    skipped.append({"baseline": baseline, "task": task_id, "metric": metric, "reason": reason})
                    continue
                reason = task_query_reason(
                    metric, dataset, task_id, template["metrics"][metric]["domains"],
                    candidate_tasks=(tga_candidates or selected_tasks)
                    if metric in {"tga_easy", "tga_hard"} else None,
                )
                if reason is None:
                    compatible.append(task_id)
                else:
                    skipped.append(
                        {
                            "baseline": baseline,
                            "task": task_id,
                            "metric": metric,
                            "reason": reason,
                        }
                    )
            if metric == "tga_easy" and tga_candidates is None and len(compatible) < 2:
                for task_id in compatible:
                    skipped.append(
                        {
                            "baseline": baseline,
                            "task": task_id,
                            "metric": metric,
                            "reason": "TGA-easy requires at least two selected compatible tasks",
                        }
                    )
                compatible = []
            if compatible:
                settings = template["metrics"][metric]
                candidates = (tga_candidates or compatible) if metric in {"tga_easy", "tga_hard"} else None
                build_metric_queries(
                    metric, dataset, compatible, settings["domains"],
                    candidate_tasks=candidates,
                )
                metric_all_tasks[metric] = tuple(candidates or compatible)
                target_tasks = tuple(compatible)
                if execution_mode == "missing":
                    contract = resume_contract(
                        templates[baseline], dataset, metric, compatible,
                        candidates,
                        data_root=data_root,
                    )
                    target_tasks = missing_tasks(
                        final_results_root / publication_baseline / metric / "summary.json",
                        contract, compatible,
                    )
                    if not target_tasks:
                        continue
                metric_to_tasks[metric] = target_tasks

        if coalesce_metrics and metric_to_tasks:
            groups = [list(metric_to_tasks)]
        else:
            by_tasks: dict[tuple[str, ...], list[str]] = {}
            for metric, compatible in metric_to_tasks.items():
                by_tasks.setdefault(compatible, []).append(metric)
            groups = list(by_tasks.values())
        for grouped_metrics in groups:
            jobs.append(_planned_job(
                baseline, publication_baseline, grouped_metrics, metric_to_tasks,
                metric_all_tasks, templates[baseline], config_dir,
            ))
    for job in jobs:
        job.execution_mode = execution_mode
    return jobs, skipped


def available_gpus(max_used_mb: int) -> list[int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found; pass explicit --gpus IDs") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown nvidia-smi error"
        raise RuntimeError(f"cannot query GPUs with nvidia-smi: {detail}") from exc
    rows: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"unexpected nvidia-smi output: {line!r}")
        index, used = (int(item) for item in fields)
        if used <= max_used_mb:
            rows.append((used, index))
    return [index for _, index in sorted(rows)]


class RunState:
    def __init__(
        self,
        batch_dir: Path,
        run_id: str,
        request: dict[str, Any],
        jobs: list[Job],
        skipped: list[dict[str, str]],
        final_path: Path,
    ) -> None:
        self.batch_dir = batch_dir
        self.run_id = run_id
        self.request = request
        self.jobs = jobs
        self.skipped = skipped
        self.final_path = final_path
        self.final_results_root = Path(request["final_results_root"])
        self.lock = threading.Lock()

    def manifest(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for job in self.jobs:
            statuses[job.status] = statuses.get(job.status, 0) + 1
        return {
            "run_id": self.run_id,
            "updated_at": now_iso(),
            "request": self.request,
            "status_counts": statuses,
            "jobs": [job.manifest_row() for job in self.jobs],
            "skipped": self.skipped,
        }

    def save(self) -> None:
        with self.lock:
            manifest = self.manifest()
            atomic_json(self.batch_dir / "manifest.json", manifest)
            self._update_final(manifest)

    def _update_final(self, manifest: dict[str, Any]) -> None:
        import fcntl

        lock_path = self.final_path.with_suffix(self.final_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if self.final_path.is_file():
                final = json.loads(self.final_path.read_text(encoding="utf-8"))
            else:
                final = {"version": 1, "runs": [], "latest": {}}
            runs = [row for row in final.get("runs", []) if row.get("run_id") != self.run_id]
            runs.append(manifest)
            latest = final.setdefault("latest", {})
            for job in manifest["jobs"]:
                if job["status"] != "completed" or not isinstance(job["result"], dict):
                    continue
                baseline_latest = latest.setdefault(job["baseline"], {})
                for metric in job["metrics"]:
                    baseline_latest[metric] = {
                        "run_id": self.run_id,
                        "finished_at": job["finished_at"],
                        "tasks": job.get("metric_tasks", {}).get(
                            metric, job["tasks"]
                        ),
                        "result_dir": job["result_dir"],
                        "result": job["result"].get(metric),
                    }
            final.update({"updated_at": now_iso(), "runs": runs, "latest": latest})
            atomic_json(self.final_path, final)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def generated_config(
    job: Job,
    gpu: int,
    output_root: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    template = yaml.safe_load(job.template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError(f"{job.template_path} must contain a YAML object")
    selected_metrics = {}
    for metric in job.metrics:
        settings = template.get("metrics", {}).get(metric)
        if settings is None:
            raise ValueError(f"{job.template_path} has no settings for metric {metric}")
        selected_metrics[metric] = copy.deepcopy(settings)
    template["gpu"] = gpu
    for field_name in ("python", "checkpoint", "data", "output", "reference_data", "sia_responses"):
        if template.get(field_name) is not None:
            path = Path(template[field_name])
            if not path.is_absolute():
                template[field_name] = str((job.template_path.parent / path).resolve())
    if data_root is not None:
        template["data"] = str(data_root)
    template["output"] = str(output_root)
    template["metrics"] = selected_metrics
    template["tasks"] = list(job.tasks)
    batch_size = evaluation_policy(template).batch_override(job.metrics)
    if batch_size is not None:
        template["batch_size"] = batch_size
    metric_tasks = {
        metric: list(job.metric_tasks.get(metric, job.tasks))
        for metric in job.metrics
    }
    template["metric_tasks"] = metric_tasks
    template.pop("candidate_tasks", None)
    if any(job.candidate_tasks.get(metric, job.tasks) != job.metric_tasks.get(metric, job.tasks)
           for metric in job.metrics if metric in {"tga_easy", "tga_hard"}):
        tga_candidates = {
            metric: list(job.candidate_tasks.get(metric, job.tasks))
            for metric in job.metrics
            if metric in {"tga_easy", "tga_hard"}
        }
        if tga_candidates:
            template["candidate_tasks"] = tga_candidates
    return template


def execution_environment() -> dict[str, str]:
    """Expose the package to children without depending on a repository checkout."""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        environment.setdefault(name, "1")
    return environment


def _validate_job_result(job: Job, result_dir: Path, data_root: Path, log_path: Path) -> None:
    """Run CPU artifact validation before making a job eligible for publication."""
    from vmbmk.tools.results.validation import validate_run

    report = validate_run(job.config_path, result_dir, data_root)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n----- {job.run_name} validation -----\n"
            + json.dumps(report, ensure_ascii=False, sort_keys=True)
            + f"\n===== {job.run_name} finished =====\n"
        )
    if not report["valid"]:
        raise RuntimeError(f"result validation failed; see {log_path} section {job.run_name}")
    job.result = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))


def run_job(job: Job, gpu: int, batch_dir: Path, state: RunState) -> None:
    job.attempts += 1
    job.gpu = gpu
    job.status = "running"
    job.started_at = now_iso()
    job.error = None
    job.result = None
    state.save()
    # Keep one log per selected run. A run can contain many tasks/jobs, and
    # creating a separate file for each one makes long retry queues unwieldy.
    log_path = batch_dir / "logs" / "run.log"
    result_dir = batch_dir / "runs" / job.run_name
    job.result_dir = str(result_dir)
    print(
        f"[{job.started_at}] START baseline={job.baseline} "
        f"metrics={','.join(job.metrics)} gpu={gpu} log={log_path}",
        flush=True,
    )
    try:
        config = generated_config(
            job,
            gpu,
            batch_dir / "runs",
            data_root=Path(state.request["data"]),
        )
        job.config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        configured_data = Dataset.load(config["data"], metrics=list(job.metrics))
        job.resume_contracts = {
            metric: resume_contract(
                job.config_path, configured_data, metric, job.metric_tasks.get(metric, job.tasks),
                job.candidate_tasks.get(metric, job.tasks) if metric in {"tga_easy", "tga_hard"} else None,
            )
            for metric in job.metrics
        }
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n===== {job.run_name} (gpu={gpu}, started={job.started_at}) =====\n"
            )
            log.flush()
            completed = subprocess.run(
                [sys.executable, "-m", "vmbmk.cli", "run", str(job.config_path), "--mode",
                 "rerun" if job.execution_mode == "rerun" else "normal"],
                env=execution_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"evaluation exited with code {completed.returncode}; "
                f"see {log_path} section {job.run_name}"
            )
        _validate_job_result(job, result_dir, Path(config["data"]), log_path)
        publish_result(
            PublicationRequest(
                baseline=job.publication_baseline or job.baseline,
                config_path=job.config_path,
                run_name=job.run_name,
                result=job.result,
                resume_contracts=job.resume_contracts,
                execution_mode=job.execution_mode,
            ),
            state.final_results_root,
            merge_existing=(
                job.execution_mode != "rerun"
                and (
                    os.environ.get("VMBMK_PUBLISH_MERGE") == "1"
                    or job.execution_mode == "missing"
                    or any(metric in {"tga_easy", "tga_hard"} for metric in job.metrics)
                )
            ),
        )
        job.status = "completed"
    except Exception as exc:  # Keep independent jobs running and record the failure.
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.finished_at = now_iso()
        state.save()
        stream = sys.stdout if job.status == "completed" else sys.stderr
        detail = "" if job.error is None else f" error={job.error}"
        print(
            f"[{job.finished_at}] {job.status.upper()} baseline={job.baseline} "
            f"gpu={gpu} attempt={job.attempts}{detail}",
            file=stream,
            flush=True,
        )


def archive_partial_output(job: Job, batch_dir: Path) -> None:
    """Move incomplete output aside so a retry starts from an empty run."""
    if job.result_dir is None:
        return
    result_dir = Path(job.result_dir)
    temporary = result_dir.parent / f".{result_dir.name}.tmp"
    batch_root = batch_dir.resolve()
    for path in (result_dir, temporary):
        if not path.resolve().is_relative_to(batch_root):
            raise ValueError(f"retry output escapes batch directory: {path}")
    stale_root = batch_dir / "stale"
    stale_root.mkdir(parents=True, exist_ok=True)
    for path in (result_dir, temporary):
        if not path.exists():
            continue
        stale = stale_root / f"{job.run_name}_attempt_{job.attempts}"
        suffix = 1
        while stale.exists():
            stale = stale_root / f"{job.run_name}_attempt_{job.attempts}_{suffix}"
            suffix += 1
        shutil.move(str(path), str(stale))


def run_with_retries(
    job: Job,
    gpu: int,
    batch_dir: Path,
    state: RunState,
    *,
    max_retries: int,
    retry_backoff_s: float,
) -> None:
    """Run one job and retry transient failures on the same worker GPU."""
    max_attempts = max_retries + 1
    for attempt in range(max_attempts):
        run_job(job, gpu, batch_dir, state)
        if job.status == "completed":
            return
        if attempt + 1 >= max_attempts:
            return

        archive_partial_output(job, batch_dir)
        delay = retry_backoff_s * (2**attempt)
        job.status = "retrying"
        state.save()
        print(
            f"[{now_iso()}] RETRY baseline={job.baseline} gpu={gpu} "
            f"next_attempt={job.attempts + 1}/{max_attempts} "
            f"sleep={delay:g}s error={job.error}",
            file=sys.stderr,
            flush=True,
        )
        if delay > 0:
            time.sleep(delay)


def unique_batch_dir(root: Path) -> tuple[str, Path]:
    base = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while True:
        run_id = base if suffix == 1 else f"{base}_{suffix}"
        path = root / run_id
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            suffix += 1
            continue
        return run_id, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--metrics", nargs="+", help="explicit metrics or all metrics declared by selected configs")
    parser.add_argument("--gpus", nargs="+", default=["auto"])
    parser.add_argument("--max-gpu-memory-used-mb", type=int, default=2048)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="number of retries after the initial attempt (default: 2)",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=10.0,
        help="initial retry delay in seconds; doubles after each failure (default: 10)",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--final-results-root", type=Path, default=DEFAULT_FINAL_RESULTS_ROOT
    )
    parser.add_argument("--config-root", type=Path, default=Path("configs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--coalesce-metrics",
        action="store_true",
        help="run all compatible metrics once per baseline to avoid reloading models",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="alias for --mode missing; reject incompatible or unrecorded results",
    )
    parser.add_argument("--mode", choices=("normal", "missing", "rerun"), default="normal")
    parser.add_argument("--list-baselines", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.incremental and args.mode == "rerun":
        parser.error("--incremental cannot be combined with --mode rerun")
    if args.incremental:
        args.mode = "missing"
    templates = discover_templates(args.config_root.resolve())
    if args.list_baselines:
        print("\n".join(sorted(templates)))
        return 0
    if not templates:
        parser.error(f"no baseline templates found in {args.config_root}")
    if args.baselines is None or args.tasks is None or args.metrics is None:
        parser.error("--baselines, --tasks, and --metrics are required")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.retry_backoff_s < 0:
        parser.error("--retry-backoff-s must be >= 0")
    try:
        baselines = select_names(args.baselines, templates)
        raw_metrics = split_values(args.metrics)
        metrics = (
            list(dict.fromkeys(
                metric
                for baseline in baselines
                for metric in yaml.safe_load(templates[baseline].read_text(encoding="utf-8"))["metrics"]
            ))
            if raw_metrics == ["all"]
            else expand_metrics(raw_metrics)
        )
        if not metrics:
            raise ValueError("selected configs contain no batch-supported metrics")
        unknown_metrics = sorted(set(metrics) - set(METRIC_DATA_NAMES))
        if unknown_metrics:
            raise ValueError(f"unsupported metrics {unknown_metrics}")
        dataset = Dataset.load(args.data.resolve(), metrics=metrics)
        raw_tasks = split_values(args.tasks)
        tasks = (
            sorted(dataset.tasks)
            if raw_tasks == ["all"]
            else list(dict.fromkeys(normalize_task(value) for value in raw_tasks))
        )
        unknown_tasks = sorted(set(tasks) - set(dataset.tasks))
        if unknown_tasks:
            raise ValueError(f"unknown tasks {unknown_tasks}")
    except ValueError as exc:
        parser.error(str(exc))

    run_id, batch_dir = unique_batch_dir(args.results_root.resolve() / "selected_runs")
    for name in ("configs", "logs", "runs"):
        (batch_dir / name).mkdir()
    jobs, skipped = plan_metric_tasks(
        dataset,
        baselines,
        tasks,
        metrics,
        templates,
        batch_dir / "configs",
        coalesce_metrics=args.coalesce_metrics,
        incremental=args.incremental,
        execution_mode=args.mode,
        final_results_root=args.final_results_root.resolve(),
        data_root=args.data.resolve(),
    )
    request = {
        "baselines": baselines,
        "tasks": tasks,
        "metrics": metrics,
        "data": str(args.data.resolve()),
        "dry_run": args.dry_run,
        "coalesce_metrics": args.coalesce_metrics,
        "incremental": args.incremental,
        "execution_mode": args.mode,
        "max_retries": args.max_retries,
        "retry_backoff_s": args.retry_backoff_s,
        "final_results_root": str(args.final_results_root.resolve()),
    }
    state = RunState(
        batch_dir,
        run_id,
        request,
        jobs,
        skipped,
        args.results_root.resolve() / "final_results.json",
    )
    print(
        f"batch_dir={batch_dir} planned_jobs={len(jobs)} "
        f"skipped_combinations={len(skipped)}",
        flush=True,
    )
    planned_metrics = {metric for job in jobs for metric in job.metrics}
    missing_metrics = [metric for metric in metrics if metric not in planned_metrics]
    if missing_metrics:
        print(
            "warning: no executable jobs for requested metrics: "
            + ", ".join(missing_metrics)
            + f"; inspect {batch_dir / 'manifest.json'} for skip reasons",
            file=sys.stderr,
        )
    if args.dry_run:
        for job in jobs:
            job.status = "planned"
        atomic_json(batch_dir / "manifest.json", state.manifest())
        print(batch_dir)
        return 0

    if args.gpus == ["auto"]:
        try:
            gpus = available_gpus(args.max_gpu_memory_used_mb)
        except RuntimeError as exc:
            parser.error(str(exc))
    else:
        try:
            gpus = list(dict.fromkeys(int(value) for value in split_values(args.gpus)))
        except ValueError:
            parser.error("--gpus must be 'auto' or integer GPU indices")
    if not gpus:
        parser.error("no GPU satisfies the requested availability threshold")

    state.request["gpus"] = gpus
    state.save()
    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                run_with_retries(
                    job,
                    gpu,
                    batch_dir,
                    state,
                    max_retries=args.max_retries,
                    retry_backoff_s=args.retry_backoff_s,
                )
            finally:
                pending.task_done()

    workers = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    state.save()
    print(batch_dir)
    return 1 if any(job.status == "failed" for job in jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
