"""Shared metric identities and lazy entry points, without model imports."""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Sequence

from vmbmk.errors import VMBMKError

if TYPE_CHECKING:
    from vmbmk.data.dataset import Dataset
    from vmbmk.inference.queries import Query


@dataclass(frozen=True)
class MetricSpec:
    dataset_name: str
    module: str
    runner: str
    builder: str | None


METRICS = {
    "sa": MetricSpec("SA", "sa", "run_sa", "build_sa_queries"),
    "sia": MetricSpec("SIA", "sia.evaluation", "run_sia", None),
    "tga_easy": MetricSpec("TGA", "tga", "run_tga_easy", "build_tga_queries"),
    "tga_hard": MetricSpec("TGA", "tga", "run_tga_hard", "build_tga_queries"),
    "voc": MetricSpec("VOC", "cycle_voc_vs.forward", "run_voc", "build_voc_queries"),
    "voc_mem": MetricSpec("VOC-MEM", "mem_voc", "run_voc_mem", "build_voc_mem_queries"),
    "cycle_voc": MetricSpec("VOC", "cycle_voc_vs.cycle", "run_cycle_voc", "build_cycle_voc_queries"),
    "fpl": MetricSpec("FPL", "fpl", "run_fpl", "build_fpl_queries"),
    "trr": MetricSpec("TRR", "trr", "run_trr_timeweighted", "build_trr_timeweighted_queries"),
    "vs": MetricSpec("VS", "cycle_voc_vs.vs", "run_vs", "build_vs_queries"),
    "csvc": MetricSpec("CSVC", "csvc", "run_csvc", "build_csvc_queries"),
}
SUPPORTED_METRICS = frozenset(METRICS)
QUERY_METRICS = frozenset(name for name, spec in METRICS.items() if spec.builder)


def canonical_metric(value: str) -> str:
    """Normalize external aliases while retaining distinct legacy metric names."""
    name = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return {"cspc": "csvc", "vocmem": "voc_mem", "mem_voc": "voc_mem", "vocf1": "voc_f1"}.get(name, name)


def dataset_metric_name(metric: str) -> str:
    return METRICS[metric].dataset_name


def selected_task_ids(
    metric: str, dataset: Dataset, tasks: Sequence[str], domains: Sequence[str],
) -> tuple[str, ...]:
    """Select declared task/domain coverage; episode eligibility stays in planners."""
    declared = dataset_metric_name(metric)
    return tuple(
        task_id for task_id in tasks
        if (declared in dataset.tasks[task_id].metrics
            or metric in {"voc", "cycle_voc", "vs"}
            and {"VOC", "CYCLE-VOC"}.intersection(dataset.tasks[task_id].metrics))
        and any(episode.domain in domains for episode in dataset.tasks[task_id].episodes.values())
    )


def metric_runner(metric: str) -> Callable[..., dict[str, Any]]:
    spec = METRICS[metric]
    return getattr(import_module(f".{spec.module}", __package__), spec.runner)


def build_metric_queries(
    metric: str, dataset: Dataset, tasks: Sequence[str], domains: Sequence[str],
    *, candidate_tasks: Sequence[str] | None = None,
) -> list[Query]:
    """Use each metric's own planner; candidate scope is not a model option."""
    spec = METRICS[metric]
    if spec.builder is None:
        raise ValueError(f"query validation is unsupported for metric: {metric}")
    builder = getattr(import_module(f".{spec.module}", __package__), spec.builder)
    if metric in {"tga_easy", "tga_hard"}:
        return builder(
            dataset, tasks, domains,
            variant=metric.removeprefix("tga_"), candidate_task_ids=candidate_tasks,
        )
    return builder(dataset, tasks, domains)


def task_query_reason(
    metric: str, dataset: Dataset, task_id: str, domains: Sequence[str],
    *, candidate_tasks: Sequence[str] | None = None,
) -> str | None:
    """Use the metric planner as batch preflight, not a second eligibility algorithm."""
    declared = dataset_metric_name(metric)
    if not selected_task_ids(metric, dataset, [task_id], domains):
        return f"no declared {declared} coverage in the selected domains"
    task_domains = [
        domain for domain in domains
        if any(episode.domain == domain for episode in dataset.tasks[task_id].episodes.values())
    ]
    try:
        queries = build_metric_queries(
            metric, dataset, [task_id], task_domains, candidate_tasks=candidate_tasks,
        )
    except VMBMKError as exc:
        return str(exc)
    return None if queries else f"no eligible {declared} queries"
