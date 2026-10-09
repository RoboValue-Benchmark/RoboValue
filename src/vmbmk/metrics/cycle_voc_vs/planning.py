"""Shared normal-ST selection and continuous Cycle-VOC&VS query planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import StateRef, ValueQuery


@dataclass(frozen=True)
class CycleTrajectory:
    """One original trajectory with its forward grid and shared-turn query IDs."""

    episode: Episode
    instruction: str
    forward_frames: list[int]
    query_ids: list[str]


@dataclass(frozen=True)
class CyclePlan:
    """A shared cohort and complete forward/reverse inference plan."""

    queries: list[ValueQuery]
    trajectories: dict[tuple[str, str, str], CycleTrajectory]
    exclusions: list[dict[str, str]]


def build_cycle_frame_indices(num_frames: int, stride: int = 10) -> list[int]:
    """Return evenly sampled source frames in the 2K+1 Cycle order."""
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise VMBMKError("CYCLE-VOC num_frames must be a positive integer")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise VMBMKError("CYCLE-VOC stride must be a positive integer")
    forward = list(range(0, num_frames, stride))
    if forward[-1] != num_frames - 1:
        forward.append(num_frames - 1)
    if len(forward) < 3:
        raise VMBMKError("CYCLE-VOC needs at least 3 sampled milestones (K >= 2)")
    return forward + list(reversed(forward[:-1]))


def plan_cycle(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    stride: int | None = None,
) -> CyclePlan:
    """Select normal successful trajectories once, preserving annotation frames.

    Forward prefixes and continuous descending prefixes share one turn. An
    explicit stride is reserved for curve tools; benchmark queries use exactly
    the annotated VOC frames, without synthesizing a start or endpoint.
    """
    selected_domains = set(domains) if domains is not None else None
    observed_domains: set[str] = set()
    queries: list[ValueQuery] = []
    trajectories: dict[tuple[str, str, str], CycleTrajectory] = {}
    exclusions: list[dict[str, str]] = []
    for task_id in sorted(dataset.tasks) if task_ids is None else task_ids:
        if task_id not in dataset.tasks:
            raise VMBMKError(f"unknown task {task_id!r}")
        task = dataset.tasks[task_id]
        if not {"CYCLE-VOC", "VOC"}.intersection(task.metrics):
            continue
        for episode in sorted(task.episodes.values(), key=lambda item: item.episode_id):
            if selected_domains is not None and episode.domain not in selected_domains:
                continue
            reason = None
            if not episode.success:
                reason = "unsuccessful"
            elif episode.trr_role is not None:
                reason = "trr_branch"
            elif episode.cspc_solution_type == "diverse":
                reason = "diverse_solution"
            elif not episode.voc_frames:
                reason = "no_cycle_voc_annotations"
            if reason is not None:
                exclusions.append({
                    "task_id": task_id, "episode_id": episode.episode_id,
                    "domain": episode.domain, "reason": reason,
                })
                continue
            if stride is None:
                frames = sorted(set(episode.voc_frames))
            else:
                cycle_indices = build_cycle_frame_indices(episode.num_frames, stride)
                frames = cycle_indices[: (len(cycle_indices) + 1) // 2]
            if len(frames) < 3:
                raise VMBMKError(
                    f"CYCLE-VOC episode {task_id}/{episode.episode_id} "
                    "needs at least 3 milestones (K >= 2)"
                )
            observed_domains.add(episode.domain)
            query_ids = []
            cycle_frames = frames + list(reversed(frames[:-1]))
            for index, frame in enumerate(cycle_frames):
                half = "up" if index < len(frames) else "down"
                half_index = index if half == "up" else index - len(frames)
                query_id = f"cycle_voc:{task_id}:{episode.episode_id}:{half}:{half_index}"
                queries.append(ValueQuery(
                    query_id, StateRef(task_id, episode.episode_id, frame),
                    task.instruction, playback="forward" if half == "up" else "cycle",
                ))
                query_ids.append(query_id)
            trajectories[(episode.domain, task_id, episode.episode_id)] = CycleTrajectory(
                episode, task.instruction, frames, query_ids
            )
    if not trajectories:
        raise VMBMKError("dataset has no task/domain with annotated CYCLE-VOC episodes")
    if selected_domains is not None and observed_domains != selected_domains:
        missing = sorted(selected_domains - observed_domains)
        raise VMBMKError(f"CYCLE-VOC domains have no annotated episodes: {missing}")
    return CyclePlan(queries, trajectories, exclusions)
