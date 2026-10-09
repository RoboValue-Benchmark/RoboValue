"""MEM-VOC planning, scoring and execution with unchanged history inputs."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import Result, StateRef, ValueQuery
from vmbmk.metrics.utils.results import result_scores
from vmbmk.metrics.utils.aggregation import summarize_domain_tasks
from vmbmk.metrics.utils.correlation import spearman_or_zero
from vmbmk.inference.execution import run_queries


def _plan(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> tuple[
    list[ValueQuery],
    dict[str, float],
    dict[tuple[str, str, str], list[str]],
]:
    if task_ids is None:
        tasks = sorted(dataset.tasks.items())
    else:
        tasks = []
        for task_id in task_ids:
            try:
                tasks.append((task_id, dataset.tasks[task_id]))
            except KeyError as exc:
                raise VMBMKError(f"unknown task {task_id!r}") from exc

    selected_domains = set(domains) if domains is not None else None
    observed_domains: set[str] = set()
    queries: list[ValueQuery] = []
    targets: dict[str, float] = {}
    groups: dict[tuple[str, str, str], list[str]] = {}
    for task_id, task in tasks:
        if "VOC-MEM" not in task.metrics:
            continue
        episodes = [
            episode
            for episode in task.episodes.values()
            if episode.success
            and (selected_domains is None or episode.domain in selected_domains)
            and episode.voc_mem_frames
        ]
        for episode in sorted(episodes, key=lambda item: item.episode_id):
            frames = sorted(episode.voc_mem_frames)
            if len(frames) < 2:
                raise VMBMKError(
                    f"MEM-VOC episode {task_id}/{episode.episode_id} "
                    "needs at least 2 points"
                )
            observed_domains.add(episode.domain)
            group = (episode.domain, task_id, episode.episode_id)
            groups[group] = []
            for frame in frames:
                query_id = f"voc_mem:{task_id}:{episode.episode_id}:{frame}"
                queries.append(
                    ValueQuery(
                        query_id,
                        StateRef(task_id, episode.episode_id, frame),
                        task.instruction,
                    )
                )
                targets[query_id] = frame / episode.num_frames
                groups[group].append(query_id)

    if not groups:
        raise VMBMKError(
            f"dataset has no task/domain with annotated MEM-VOC episodes"
        )
    if selected_domains is not None and observed_domains != selected_domains:
        missing = sorted(selected_domains - observed_domains)
        raise VMBMKError(f"MEM-VOC domains have no annotated episodes: {missing}")
    return queries, targets, groups


def build_voc_mem_queries(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> list[ValueQuery]:
    return _plan(
        dataset,
        task_ids,
        domains,
    )[0]


def score_voc_mem_by_domain(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    queries, targets, groups = _plan(
        dataset,
        task_ids,
        domains,
    )
    scores = result_scores(queries, results, "MEM-VOC", "value")

    by_domain_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (domain, task_id, _), query_ids in groups.items():
        predicted = [scores[query_id] for query_id in query_ids]
        target = [targets[query_id] for query_id in query_ids]
        by_domain_task[(domain, task_id)].append(
            spearman_or_zero(predicted, target, "MEM-VOC")
        )
    by_domain: dict[str, dict[str, float]] = defaultdict(dict)
    for (domain, task_id), values in sorted(by_domain_task.items()):
        by_domain[domain][task_id] = sum(values) / len(values)
    return {domain: tasks for domain, tasks in sorted(by_domain.items())}


def score_voc_mem_by_task(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> dict[str, float]:
    by_domain = score_voc_mem_by_domain(
        dataset,
        results,
        task_ids,
        domains,
    )
    values: dict[str, list[float]] = defaultdict(list)
    for tasks in by_domain.values():
        for task_id, value in tasks.items():
            values[task_id].append(value)
    return {
        task_id: sum(task_scores) / len(task_scores)
        for task_id, task_scores in sorted(values.items())
    }


def score_voc_mem(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> float:
    by_domain = score_voc_mem_by_domain(
        dataset,
        results,
        task_ids,
        domains,
    )
    domain_scores = [sum(tasks.values()) / len(tasks) for tasks in by_domain.values()]
    return sum(domain_scores) / len(domain_scores)


def run_voc_mem(
    data_root: str | Path,
    model: Mapping[str, Any],
    task_ids: Sequence[str],
    operation_path: str | Path,
    gpu: int,
    *,
    mode: str,
    domains: Sequence[str],
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=["voc_mem"])
    queries = build_voc_mem_queries(
        dataset,
        task_ids,
        domains,
    )
    results = run_queries(
        data_root, queries, model, operation_path, gpu,
        metric="MEM-VOC", mode=mode, native_metric="voc_mem",
    )
    scores = score_voc_mem_by_domain(
        dataset,
        results,
        task_ids,
        domains,
    )
    return summarize_domain_tasks(scores)
