from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import Result, StateRef, ValueQuery
from vmbmk.metrics.utils.statistics import edge_frames, mean
from vmbmk.metrics.utils.results import result_scores
from vmbmk.inference.execution import run_queries


def _plan(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> tuple[
    list[ValueQuery],
    dict[tuple[str, str, str], list[str]],
    dict[tuple[str, str, str], bool],
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
    groups: dict[tuple[str, str, str], list[str]] = {}
    labels: dict[tuple[str, str, str], bool] = {}
    for task_id, task in tasks:
        if "SA" not in task.metrics:
            continue
        episodes_by_domain: dict[str, list[Any]] = defaultdict(list)
        for episode in task.episodes.values():
            if selected_domains is None or episode.domain in selected_domains:
                episodes_by_domain[episode.domain].append(episode)
        for domain, episodes in sorted(episodes_by_domain.items()):
            classes = {episode.success for episode in episodes}
            if classes != {False, True}:
                raise VMBMKError(
                    f"SA task/domain {task_id}/{domain} requires both ST and FRT episodes"
                )
            observed_domains.add(domain)
            for episode in sorted(episodes, key=lambda item: item.episode_id):
                group = (domain, task_id, episode.episode_id)
                groups[group] = []
                labels[group] = episode.success
                _, tail = edge_frames(episode.num_frames)
                for frame in tail:
                    query_id = f"sa:{task_id}:{episode.episode_id}:{frame}"
                    queries.append(
                        ValueQuery(
                            query_id,
                            StateRef(task_id, episode.episode_id, frame),
                            task.instruction,
                        )
                    )
                    groups[group].append(query_id)

    if not groups:
        raise VMBMKError("dataset has no task/domain with SA episodes")
    if selected_domains is not None and observed_domains != selected_domains:
        missing = sorted(selected_domains - observed_domains)
        raise VMBMKError(f"SA domains have no eligible episodes: {missing}")
    return queries, groups, labels


def build_sa_queries(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> list[ValueQuery]:
    return _plan(dataset, task_ids, domains)[0]


def _score_sa_details(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float | None]]:
    queries, groups, labels = _plan(dataset, task_ids, domains)
    scores = result_scores(queries, results, "SA", "value")

    terminal_scores = {
        group: mean([scores[query_id] for query_id in query_ids])
        for group, query_ids in groups.items()
    }
    grouped: dict[tuple[str, str], dict[bool, list[float]]] = defaultdict(
        lambda: {False: [], True: []}
    )
    for group, value in terminal_scores.items():
        domain, task_id, _ = group
        grouped[(domain, task_id)][labels[group]].append(value)

    by_domain: dict[str, dict[str, float]] = defaultdict(dict)
    success_means: dict[str, dict[str, float]] = defaultdict(dict)
    failure_means: dict[str, dict[str, float]] = defaultdict(dict)
    for (domain, task_id), classes in sorted(grouped.items()):
        st_scores = classes[True]
        frt_scores = classes[False]
        wins = sum(st > frt for st in st_scores for frt in frt_scores)
        by_domain[domain][task_id] = wins / (len(st_scores) * len(frt_scores))
        success_means[domain][task_id] = mean(st_scores)
        failure_means[domain][task_id] = mean(frt_scores)

    def macro_average(values: Mapping[str, Mapping[str, float]]) -> float:
        return mean([mean(tasks.values()) for _, tasks in sorted(values.items())])

    success_macro_mean = macro_average(success_means)
    failure_macro_mean = macro_average(failure_means)
    gap = success_macro_mean - failure_macro_mean
    return (
        {domain: tasks for domain, tasks in sorted(by_domain.items())},
        {
            "success_macro_mean": success_macro_mean,
            "failure_macro_mean": failure_macro_mean,
            "gap": gap,
            "range": success_macro_mean,
            "gap_over_range_percent": (
                100 * gap / success_macro_mean if success_macro_mean else None
            ),
        },
    )


def score_sa_by_domain(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    return _score_sa_details(dataset, results, task_ids, domains)[0]


def score_sa_by_task(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> dict[str, float]:
    by_domain = score_sa_by_domain(dataset, results, task_ids, domains)
    values: dict[str, list[float]] = defaultdict(list)
    for tasks in by_domain.values():
        for task_id, value in tasks.items():
            values[task_id].append(value)
    return {
        task_id: sum(task_scores) / len(task_scores)
        for task_id, task_scores in sorted(values.items())
    }


def score_sa(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> float:
    by_domain = score_sa_by_domain(dataset, results, task_ids, domains)
    domain_scores = [sum(tasks.values()) / len(tasks) for tasks in by_domain.values()]
    return sum(domain_scores) / len(domain_scores)


def run_sa(
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
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=["sa"])
    queries = build_sa_queries(dataset, task_ids, domains)
    results = run_queries(
        data_root,
        queries,
        model,
        operation_path,
        gpu,
        metric="sa",
        mode=mode,
    )
    scores, terminal_values = _score_sa_details(
        dataset,
        results,
        task_ids,
        domains,
    )
    domain_results = {
        domain: {
            "mean": sum(tasks.values()) / len(tasks),
            "tasks": tasks,
        }
        for domain, tasks in scores.items()
    }
    return {
        "protocol": "terminal-value-tail2pct-average-sa-v5",
        "primary": "strict_st_over_frt_accuracy",
        "mean": sum(item["mean"] for item in domain_results.values())
        / len(domain_results),
        "domains": domain_results,
        "edge_fraction": 0.02,
        "gap": terminal_values["gap"],
        "terminal_values": {
            "success_macro_mean": terminal_values["success_macro_mean"],
            "failure_macro_mean": terminal_values["failure_macro_mean"],
            "range": terminal_values["range"],
            "gap_over_range_percent": terminal_values["gap_over_range_percent"],
        },
    }
