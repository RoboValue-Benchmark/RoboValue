"""Full-trajectory, time-weighted TRR scorer frozen from the Acc protocol."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import Result, StateRef, ValueQuery
from vmbmk.metrics.utils.statistics import mean
from vmbmk.metrics.utils.results import result_scores
from vmbmk.inference.execution import run_queries
from vmbmk.metrics.utils.aggregation import summarize_task_scores

TARGET_HZ = 10.0
MIN_SAMPLES = 10
TRR_PROTOCOL = "trr-binary-timeweighted-10hz-k10-v2"

TRR_DEFINITION = {
    "sampling": (
        "episode-anchored round(i*fps/10) grid; native top-up to min(10, "
        "stage length), earliest seed and ties"
    ),
    "weighting": "frame-center Voronoi weights clipped to stage boundaries",
    "direction": (
        "Sw=sum_{i<j} wi*wj*(vj-vi) / sum_{i<j} wi*wj*abs(vj-vi), "
        "with constant stage=0"
    ),
    "stage_signs": (
        "failure<0; recovery>0; successful_result>0; failed_result<0; "
        "error_continuation<=0"
    ),
    "aggregation": (
        "binary conjunction within branch, equal branches/groups/tasks/covered domains"
    ),
}
ROLES = ("r_plus", "r_zero", "r_minus")
STAGES = {
    "r_plus": (("failure_span", -1.0), ("recovery_span", 1.0), ("recovery_success_span", 1.0)),
    "r_zero": (("failure_span", -1.0), ("recovery_span", 1.0), ("recovery_failed_span", -1.0)),
    "r_minus": (("failure_span", -1.0), ("continue_span", None)),
}


@dataclass(frozen=True)
class StagePlan:
    role: str
    name: str
    sign: float | None
    start: int
    end: int
    frames: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class GroupPlan:
    domain: str
    task_id: str
    group_id: str
    branches: Mapping[str, Episode]
    stages: Mapping[str, tuple[StagePlan, ...]]


def _frames(start: int, end: int, fps: float) -> tuple[int, ...]:
    if not 0 <= start < end:
        raise VMBMKError(f"invalid TRR stage [{start}, {end})")
    selected = set()
    sample_index = 0
    while True:
        frame = round(sample_index * fps / TARGET_HZ)
        if frame >= end:
            break
        if frame >= start:
            selected.add(frame)
        sample_index += 1
    target = min(MIN_SAMPLES, end - start)
    while len(selected) < target:
        candidates = [frame for frame in range(start, end) if frame not in selected]
        choice = (
            candidates[0]
            if not selected
            else max(
                candidates,
                key=lambda frame: (min(abs(frame - other) for other in selected), -frame),
            )
        )
        selected.add(choice)
    return tuple(sorted(selected))


def _weights(frames: Sequence[int], start: int, end: int) -> tuple[float, ...]:
    result = []
    for index, frame in enumerate(frames):
        left = float(start) if index == 0 else max(float(start), (frames[index - 1] + frame) / 2)
        right = (
            float(end)
            if index + 1 == len(frames)
            else min(float(end), (frame + frames[index + 1]) / 2)
        )
        if right <= left:
            raise VMBMKError(f"non-positive TRR frame weight at {frame}")
        result.append(right - left)
    return tuple(result)


def _branch(episode: Episode) -> tuple[StagePlan, ...]:
    if episode.trr_role not in ROLES:
        raise VMBMKError(f"{episode.episode_id}: invalid TRR role {episode.trr_role!r}")
    output = []
    for name, sign in STAGES[episode.trr_role]:
        if name not in episode.trr_spans:
            raise VMBMKError(f"{episode.episode_id}: missing TRR stage {name}")
        start, end = episode.trr_spans[name]
        frames = _frames(start, end, episode.fps)
        output.append(
            StagePlan(
                episode.trr_role, name, sign, start, end, frames, _weights(frames, start, end)
            )
        )
    return tuple(output)


def _plans(
    dataset: Dataset, task_ids: Sequence[str] | None, domains: Sequence[str] | None
) -> list[GroupPlan]:
    wanted_tasks = sorted(dataset.tasks) if task_ids is None else list(task_ids)
    wanted_domains = set(domains) if domains is not None else None
    grouped: dict[tuple[str, str, str], dict[str, Episode]] = defaultdict(dict)
    for task_id in wanted_tasks:
        if task_id not in dataset.tasks:
            raise VMBMKError(f"unknown task {task_id!r}")
        task = dataset.tasks[task_id]
        if "TRR" not in task.metrics:
            continue
        for episode in task.episodes.values():
            if episode.trr_group_id is None or episode.trr_role is None:
                continue
            if wanted_domains is not None and episode.domain not in wanted_domains:
                continue
            key = (episode.domain, task_id, episode.trr_group_id)
            if episode.trr_role in grouped[key]:
                raise VMBMKError(f"TRR group {key!r} repeats {episode.trr_role}")
            grouped[key][episode.trr_role] = episode
    plans = [
        GroupPlan(
            domain,
            task_id,
            group_id,
            dict(branches),
            {role: _branch(branches[role]) for role in ROLES},
        )
        for (domain, task_id, group_id), branches in sorted(grouped.items())
        if set(branches) == set(ROLES)
    ]
    if not plans:
        raise VMBMKError("dataset has no complete TRR groups")
    observed = {plan.domain for plan in plans}
    if wanted_domains is not None and observed != wanted_domains:
        raise VMBMKError(f"TRR domains have no complete groups: {sorted(wanted_domains - observed)}")
    return plans


def build_trr_timeweighted_queries(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> list[ValueQuery]:
    """Plan ordered branch/stage queries without changing frozen sampling."""
    queries = []
    for group_index, plan in enumerate(_plans(dataset, task_ids, domains)):
        instruction = dataset.tasks[plan.task_id].instruction
        for role in ROLES:
            episode = plan.branches[role]
            for stage in plan.stages[role]:
                for index, frame in enumerate(stage.frames):
                    queries.append(
                        ValueQuery(
                            f"trr_tw:{group_index}:{role}:{stage.name}:{index}",
                            StateRef(plan.task_id, episode.episode_id, frame),
                            instruction,
                        )
                    )
    return queries


def _direction(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights) or not values:
        raise VMBMKError("TRR score/weight cardinality mismatch")
    numerator = denominator = 0.0
    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            mass = weights[left_index] * weights[right_index]
            delta = values[right_index] - values[left_index]
            numerator += mass * delta
            denominator += mass * abs(delta)
    return 0.0 if denominator == 0.0 else numerator / denominator


def score_trr_timeweighted(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score binary branch conjunctions, then aggregate groups and tasks."""
    plans = _plans(dataset, task_ids, domains)
    scores = result_scores(
        build_trr_timeweighted_queries(dataset, task_ids, domains), results, "TRR", "value"
    )
    task_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    records = []
    for group_index, plan in enumerate(plans):
        branch_scores, branch_stages = {}, {}
        for role in ROLES:
            stages = []
            for stage in plan.stages[role]:
                values = [
                    scores[f"trr_tw:{group_index}:{role}:{stage.name}:{index}"]
                    for index in range(len(stage.frames))
                ]
                raw = _direction(values, stage.weights)
                signed = int(raw <= 0 if stage.sign is None else stage.sign * raw > 0)
                stages.append(
                    {
                        "name": stage.name,
                        "start_frame": stage.start,
                        "end_frame_exclusive": stage.end,
                        "frames": list(stage.frames),
                        "weights": list(stage.weights),
                        "direction": raw,
                        "score": signed,
                    }
                )
            branch_stages[role] = stages
            branch_scores[role] = int(all(stage["score"] for stage in stages))
        group_score = mean(list(branch_scores.values()))
        task_groups[(plan.domain, plan.task_id)].append(group_score)
        records.append(
            {
                "domain": plan.domain,
                "task_id": plan.task_id,
                "group_id": plan.group_id,
                "branch_scores": branch_scores,
                "group_score": group_score,
                "stages": branch_stages,
            }
        )
    task_scores = {
        f"{domain}::{task}": mean(values)
        for (domain, task), values in sorted(task_groups.items())
    }
    return {
        "protocol": TRR_PROTOCOL,
        "definition": dict(TRR_DEFINITION),
        "task_scores": task_scores,
        **summarize_task_scores({key: mean(values) for key, values in task_groups.items()}),
        "n_groups": len(records),
        "groups": records,
    }


def run_trr_timeweighted(
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
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=["trr"])
    queries = build_trr_timeweighted_queries(dataset, task_ids, domains)
    results = run_queries(data_root, queries, model, operation_path, gpu, metric="trr", mode=mode)
    return score_trr_timeweighted(dataset, results, task_ids, domains)
