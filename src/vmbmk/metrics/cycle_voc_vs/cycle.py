from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.dispatch import run_inference
from vmbmk.inference.queries import Result, ValueQuery, read_results
from vmbmk.metrics.utils.results import result_scores
from vmbmk.metrics.utils.aggregation import summarize_domain_tasks
from vmbmk.metrics.utils.correlation import spearman
from vmbmk.serialization import write_jsonl
from .planning import CyclePlan, build_cycle_frame_indices, plan_cycle


def build_cycle_voc_queries(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    stride: int | None = None,
) -> list[ValueQuery]:
    """Build annotated VOC frames followed by VROC in one virtual timeline.

    Forward queries see S0..Sj. Descending queries see S0..SK..Sj, represented
    by a logical playback view rather than a materialized duplicate video.
    Thus a video-prefix model receives the complete cycle prefix while the
    public value operation remains unchanged.
    """
    return plan_cycle(dataset, task_ids, domains, stride).queries


def _score_cycle_segment(
    values: Sequence[float], targets: Sequence[float]
) -> tuple[float | None, float]:
    """Return raw Spearman and the benchmark score for one cycle half.

    Spearman is mathematically undefined for a constant prediction because
    its ranks have zero variance. Cycle-VOC keeps such model output as a valid
    benchmark result: the raw correlation is reported as ``None`` and the
    segment score is defined as zero.
    """
    if len(set(values)) < 2 or len(set(targets)) < 2:
        return None, 0.0
    rho = spearman(values, targets, "CYCLE-VOC")
    return rho, rho


def score_forward_voc_values(values: Sequence[float]) -> float:
    """Score the cycle's forward half, with constants receiving zero."""
    if len(values) < 3:
        raise VMBMKError("VOC requires at least 3 cycle milestones")
    return _score_cycle_segment(values, list(range(len(values))))[1]


def score_cycle_voc_values(values: Sequence[float]) -> dict[str, float | None]:
    """Score one ordered 2K+1 value sequence and expose its components."""
    if len(values) < 5 or len(values) % 2 == 0:
        raise VMBMKError("CYCLE-VOC requires 2K+1 values with K >= 2")
    k = (len(values) - 1) // 2
    up_values = list(values[: k + 1])
    down_values = list(values[k:])
    rho_up, score_up = _score_cycle_segment(
        up_values,
        [index / k for index in range(k + 1)],
    )
    rho_down, score_down = _score_cycle_segment(
        down_values,
        [1 - index / k for index in range(k + 1)],
    )
    cycle_voc = (score_up + score_down) / 2
    return {
        "rho_up": rho_up,
        "rho_down": rho_down,
        "score_up": score_up,
        "score_down": score_down,
        "cycle_voc": cycle_voc,
    }

def score_cycle_voc_by_domain(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    *,
    plan: CyclePlan | None = None,
) -> dict[str, dict[str, float]]:
    plan = plan if plan is not None else plan_cycle(dataset, task_ids, domains)
    scores = result_scores(plan.queries, results, "CYCLE-VOC", "value")
    by_domain_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (domain, task_id, _), trajectory in plan.trajectories.items():
        details = score_cycle_voc_values([scores[query_id] for query_id in trajectory.query_ids])
        by_domain_task[(domain, task_id)].append(details["cycle_voc"])
    by_domain: dict[str, dict[str, float]] = defaultdict(dict)
    for (domain, task_id), values in sorted(by_domain_task.items()):
        by_domain[domain][task_id] = sum(values) / len(values)
    return {domain: tasks for domain, tasks in sorted(by_domain.items())}


def score_cycle_voc(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> float:
    by_domain = score_cycle_voc_by_domain(dataset, results, task_ids, domains)
    domain_scores = [sum(tasks.values()) / len(tasks) for tasks in by_domain.values()]
    return sum(domain_scores) / len(domain_scores)


def run_cycle_voc(
    data_root: str | Path,
    model: Mapping[str, Any],
    task_ids: Sequence[str],
    operation_path: str | Path,
    gpu: int,
    *,
    mode: str,
    domains: Sequence[str],
    plan: CyclePlan | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    if mode not in {"base", "native"}:
        raise VMBMKError("CYCLE-VOC mode must be base or native")
    native_metric = "cycle_voc" if mode == "native" else None
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=["cycle_voc"])
    plan = plan if plan is not None else plan_cycle(dataset, task_ids, domains)
    queries = plan.queries
    with tempfile.TemporaryDirectory(prefix="vmbmk-cycle-voc-") as directory:
        query_path = Path(directory) / "queries.jsonl"
        write_jsonl(query_path, (query.to_dict() for query in queries))
        run_inference(
            data_root,
            query_path,
            model,
            operation_path,
            gpu=gpu,
            native_metric=native_metric,
        )
    scores = score_cycle_voc_by_domain(
        dataset, read_results(operation_path), task_ids, domains, plan=plan
    )
    return summarize_domain_tasks(scores)
