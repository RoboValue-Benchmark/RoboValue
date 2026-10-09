"""VOC is the forward half of the shared Cycle-VOC protocol."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.dispatch import validate_results
from vmbmk.inference.queries import Result, ValueQuery, read_results
from vmbmk.metrics.utils.results import result_scores
from vmbmk.inference.execution import run_queries
from vmbmk.metrics.utils.aggregation import summarize_domain_tasks
from vmbmk.serialization import replace_jsonl
from .cycle import score_forward_voc_values
from .planning import CyclePlan, plan_cycle

VOC_PROTOCOL = "voc-cycle-forward-v2"


def build_voc_queries(
    dataset: Dataset, task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None, *, plan: CyclePlan | None = None,
) -> list[ValueQuery]:
    """Select the exact forward queries, including the shared turn."""
    plan = plan if plan is not None else plan_cycle(dataset, task_ids, domains)
    return [
        replace(query, query_id=query.query_id.replace("cycle_voc:", "voc:", 1))
        for query in plan.queries if query.playback == "forward"
    ]


def score_voc_by_domain(
    dataset: Dataset, results: Sequence[Result],
    task_ids: Sequence[str] | None = None, domains: Sequence[str] | None = None,
    *, plan: CyclePlan | None = None,
) -> dict[str, dict[str, float]]:
    plan = plan if plan is not None else plan_cycle(dataset, task_ids, domains)
    queries = build_voc_queries(dataset, plan=plan)
    scores = result_scores(queries, results, "VOC", "value")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (domain, task, _), trajectory in plan.trajectories.items():
        values = [scores[query_id.replace("cycle_voc:", "voc:", 1)] for query_id in trajectory.query_ids[:len(trajectory.forward_frames)]]
        grouped[(domain, task)].append(score_forward_voc_values(values))
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for (domain, task), values in sorted(grouped.items()):
        output[domain][task] = sum(values) / len(values)
    return dict(output)


def score_voc_by_task(
    dataset: Dataset, results: Sequence[Result],
    task_ids: Sequence[str] | None = None, domains: Sequence[str] | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for tasks in score_voc_by_domain(dataset, results, task_ids, domains).values():
        for task, value in tasks.items():
            grouped[task].append(value)
    return {task: sum(values) / len(values) for task, values in sorted(grouped.items())}


def score_voc(
    dataset: Dataset, results: Sequence[Result],
    task_ids: Sequence[str] | None = None, domains: Sequence[str] | None = None,
) -> float:
    return summarize_domain_tasks(score_voc_by_domain(dataset, results, task_ids, domains))["mean"]


def run_voc(
    data_root: str | Path, model: Mapping[str, Any], task_ids: Sequence[str],
    operation_path: str | Path, gpu: int, *, mode: str, domains: Sequence[str],
    plan: CyclePlan | None = None, cycle_operation_path: str | Path | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    """Infer only the forward half, or reuse a completed base-mode cycle."""
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=["voc"])
    plan = plan if plan is not None else plan_cycle(dataset, task_ids, domains)
    queries = build_voc_queries(dataset, plan=plan)
    if cycle_operation_path is None:
        results = run_queries(
            data_root, queries, model, operation_path, gpu,
            metric="VOC", mode=mode, native_metric="voc",
        )
    else:
        if mode != "base":
            raise VMBMKError("VOC cannot reuse metric-native Cycle-VOC predictions")
        validate_results(plan.queries, cycle_operation_path)
        predictions = result_scores(plan.queries, read_results(cycle_operation_path), "CYCLE-VOC", "value")
        results = [Result(query.query_id, "value", predictions[query.query_id.replace("voc:", "cycle_voc:", 1)]) for query in queries]
        replace_jsonl(operation_path, (result.to_dict() for result in results))
    summary = summarize_domain_tasks(score_voc_by_domain(dataset, results, plan=plan))
    return {"protocol": VOC_PROTOCOL, **summary}
