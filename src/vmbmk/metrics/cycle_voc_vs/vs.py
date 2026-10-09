from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.dispatch import validate_results
from vmbmk.inference.queries import Result, StateRef, ValueQuery, read_results
from vmbmk.metrics.utils.statistics import mean
from vmbmk.metrics.utils.results import result_scores
from vmbmk.metrics.utils.aggregation import summarize_task_scores
from vmbmk.serialization import replace_jsonl
from .cycle import run_cycle_voc
from .planning import CyclePlan, plan_cycle
from .vs_scoring import score_curve

VS_PROTOCOL = "vs_v1-cycle-forward-5hz-v2"
VSGroupKey = tuple[str, str, str]
VSGroup = tuple[list[str], list[int], float]


def _plan(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    cycle_plan: CyclePlan | None = None,
) -> tuple[list[ValueQuery], dict[VSGroupKey, VSGroup], list[dict[str, str]]]:
    plan = cycle_plan if cycle_plan is not None else plan_cycle(dataset, task_ids, domains)
    selected_domains = set(domains) if domains is not None else None
    selected_tasks = set(task_ids) if task_ids is not None else None
    queries, groups = [], {}
    exclusions = [
        item for item in plan.exclusions
        if (selected_tasks is None or item["task_id"] in selected_tasks)
        and (selected_domains is None or item["domain"] in selected_domains)
    ]
    for group, trajectory in plan.trajectories.items():
        domain, task_id, episode_id = group
        if selected_tasks is not None and task_id not in selected_tasks:
            continue
        if selected_domains is not None and domain not in selected_domains:
            continue
        episode = trajectory.episode
        frames = []
        sample_index = 0
        while True:
            frame = round(sample_index * episode.fps / 5)
            if frame >= episode.num_frames:
                break
            if not frames or frames[-1] != frame:
                frames.append(frame)
            sample_index += 1
        if len(frames) < 7:
            exclusions.append({
                "task_id": task_id, "episode_id": episode_id,
                "domain": domain, "reason": "fewer_than_seven_samples",
            })
            continue
        missing_frames = sorted(set(frames) - set(trajectory.forward_frames))
        if missing_frames:
            raise VMBMKError(
                f"VS {task_id}/{episode_id}: Cycle-VOC forward annotations "
                f"do not cover the 5 Hz grid; missing frames={missing_frames}"
            )
        query_ids = []
        for frame in frames:
            query_id = f"vs:{task_id}:{episode_id}:{frame}"
            queries.append(ValueQuery(
                query_id, StateRef(task_id, episode_id, frame), trajectory.instruction
            ))
            query_ids.append(query_id)
        groups[group] = (query_ids, frames, episode.fps)
    if not groups:
        raise VMBMKError("dataset has no eligible VS expert trajectories")
    task_domains = defaultdict(set)
    for domain, task_id, _ in groups:
        task_domains[task_id].add(domain)
    covered_domains = set.union(*task_domains.values())
    if selected_domains is not None and covered_domains != selected_domains:
        raise VMBMKError("VS requested domains lack eligible expert trajectories")
    if any(coverage != covered_domains for coverage in task_domains.values()):
        raise VMBMKError("VS requires the same covered domains for every eligible task")
    return queries, groups, exclusions


def build_vs_queries(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    *,
    cycle_plan: CyclePlan | None = None,
) -> list[ValueQuery]:
    """Select the native 5 Hz grid covered by Cycle-VOC forward annotations."""
    return _plan(dataset, task_ids, domains, cycle_plan)[0]


def score_vs(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    *,
    cycle_plan: CyclePlan | None = None,
) -> dict[str, Any]:
    """Average episode VS scores using the existing task/domain protocol."""
    queries, groups, exclusions = _plan(dataset, task_ids, domains, cycle_plan)
    scores = result_scores(queries, results, "VS", "value")
    components = {name: defaultdict(list) for name in ("vs", "er", "scale")}
    episodes = []
    for (domain, task_id, episode_id), (query_ids, frames, fps) in groups.items():
        try:
            record = score_curve(
                [scores[query_id] for query_id in query_ids],
                [frame / fps for frame in frames],
            ).to_dict()
        except ValueError as exc:
            raise VMBMKError(f"VS {task_id}/{episode_id}: {exc}") from exc
        for name, grouped in components.items():
            grouped[(domain, task_id)].append(record[name])
        episodes.append(
            {
                "domain": domain,
                "task_id": task_id,
                "episode_id": episode_id,
                "frames": frames,
                **record,
            }
        )
    summaries = {
        name: summarize_task_scores({key: mean(values) for key, values in grouped.items()})
        for name, grouped in components.items()
    }
    return {
        "protocol": VS_PROTOCOL,
        "prediction_source": "cycle_voc_forward",
        "formula_version": "vs_v1",
        "sample_hz": 5,
        "endpoint_samples": 3,
        "primary": "vs_higher_is_better",
        **summaries["vs"],
        "er": summaries["er"],
        "nonflat_time_fraction": summaries["scale"],
        "episodes": episodes,
        "n_episodes": len(episodes),
        "exclusions": exclusions,
    }


def run_vs(
    data_root: str | Path,
    model: Mapping[str, Any],
    task_ids: Sequence[str],
    operation_path: str | Path,
    gpu: int,
    *,
    mode: str,
    domains: Sequence[str],
    cycle_operation_path: str | Path | None = None,
    cycle_task_ids: Sequence[str] | None = None,
    cycle_domains: Sequence[str] | None = None,
    cycle_plan: CyclePlan | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    """Score Cycle-VOC's forward predictions without another model invocation."""
    if mode != "base":
        raise VMBMKError("VS requires Cycle-VOC mode=base predictions")
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=["vs"])
    source_tasks = cycle_task_ids if cycle_task_ids is not None else task_ids
    source_domains = cycle_domains if cycle_domains is not None else domains
    cycle_plan = (
        cycle_plan if cycle_plan is not None
        else plan_cycle(dataset, source_tasks, source_domains)
    )
    queries = build_vs_queries(dataset, task_ids, domains, cycle_plan=cycle_plan)
    if cycle_operation_path is None:
        cycle_operation_path = Path(operation_path).with_name(
            f"{Path(operation_path).name}.cycle.jsonl"
        )
        run_cycle_voc(
            data_root, model, source_tasks, cycle_operation_path, gpu,
            mode="base", domains=source_domains, plan=cycle_plan,
        )
    cycle_queries = cycle_plan.queries
    validate_results(cycle_queries, cycle_operation_path)
    predictions = result_scores(
        cycle_queries, read_results(cycle_operation_path), "CYCLE-VOC", "value"
    )
    forward_queries = {
        query.state: query
        for query in cycle_queries if query.playback == "forward"
    }
    results = []
    for query in queries:
        source = forward_queries.get(query.state)
        if source is None or source.instruction != query.instruction:
            raise VMBMKError(
                f"VS {query.query_id}: no matching Cycle-VOC forward prediction"
            )
        results.append(Result(query.query_id, "value", predictions[source.query_id]))
    replace_jsonl(operation_path, (result.to_dict() for result in results))
    return score_vs(dataset, results, task_ids, domains, cycle_plan=cycle_plan)
