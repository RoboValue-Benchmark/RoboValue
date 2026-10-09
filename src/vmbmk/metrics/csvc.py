from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import CompareQuery, Result, StateRef
from vmbmk.metrics.utils.results import result_scores
from vmbmk.inference.execution import run_queries

EpisodeDeltas = Mapping[str, Mapping[str, float]]
CSVC_PROTOCOL = "semantic-gain-symmetric-ratio-v2"


def _validated(episodes: EpisodeDeltas, label: str) -> list[dict[str, float]]:
    if not episodes:
        raise VMBMKError(f"CSVC {label} episodes are empty")
    rows = []
    for episode, values in sorted(episodes.items()):
        if not isinstance(episode, str) or not episode.strip() or not values:
            raise VMBMKError(f"CSVC invalid {label} episode")
        row = {}
        for semantic, value in values.items():
            if not isinstance(semantic, str) or not semantic.strip():
                raise VMBMKError("CSVC requires stable semantic IDs")
            if isinstance(
                value,
                bool
            ) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise VMBMKError("CSVC requires finite deltas")
            row[semantic] = float(value)
        rows.append(row)
    return rows


def _by_semantic(rows: Sequence[Mapping[str, float]]) -> dict[str, list[float]]:
    values = {}
    for row in rows:
        for semantic, delta in row.items():
            values.setdefault(semantic, []).append(delta)
    return values


def _cross_semantic_null(
    rows: Sequence[Mapping[str, float]], semantics: Sequence[str]
) -> float | None:
    """Return the deterministic cross-solution, mismatched-semantic null.

    This uses precisely the unique-solution pairs used by cross-solution
    diagnostics, but replaces the semantic match with every feasible ``a != b``
    mismatch. Both orientations are included so partial semantic coverage does
    not depend on arbitrary row ordering. It is diagnostic-only and never
    changes the CSVC primary score.
    """
    values = []
    allowed = set(semantics)
    for left_index, left in enumerate(rows):
        left_semantics = sorted(allowed & set(left))
        for right in rows[left_index + 1:]:
            right_semantics = sorted(allowed & set(right))
            values.extend(
                (left[a] - right[b]) ** 2
                for a in left_semantics
                for b in right_semantics
                if a != b
            )
            values.extend(
                (right[a] - left[b]) ** 2
                for a in right_semantics
                for b in left_semantics
                if a != b
            )
    return mean(values) if values else None


def _score_solutions(
    rows: Sequence[Mapping[str, float]], semantics: Sequence[str]
) -> dict[str, Any]:
    values = _by_semantic(rows)
    # Keep the original semantic estimand during bootstrap. Missing coverage
    # invalidates a replicate instead of silently changing measured semantics.
    if not semantics or any(len(values.get(s, [])) < 2 for s in semantics):
        return {"score": None, "raw_score": None, "identifiable": False,
                "unidentifiable_reason": "insufficient_unique_solution_coverage"}
    centers = {s: median(values[s]) for s in semantics}
    errors = {s: mean((v - centers[s]) ** 2 for v in values[s]) for s in semantics}
    gain = math.sqrt(mean(v ** 2 for v in centers.values()))
    distance = math.sqrt(mean(errors.values()))
    denominator = gain + distance
    score = (gain - distance) / denominator if denominator else -1.0
    result = {"score": score, "raw_score": score, "identifiable": denominator > 0,
            "gain_scale": gain, "cross_rmse": distance,
            "semantic_centers": centers,
            "semantic_rmse": {s: math.sqrt(v) for s, v in errors.items()}}
    if not denominator:
        result["unidentifiable_reason"] = "zero_total_magnitude"
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    values = sorted(values)
    position = probability * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def _csvc_diagnostics(
    result: Mapping[str, Any], solutions: Sequence[Mapping[str, float]], semantics: Sequence[str],
    normal_values: Mapping[str, list[float]], values: Mapping[str, list[float]],
    bootstrap_samples: int, bootstrap_seed: int,
) -> dict[str, Any]:
    """Compute uncertainty and auxiliary repeat/null statistics, not the primary score."""
    rng = random.Random(bootstrap_seed)
    boot = []
    if result["score"] is not None:
        for _ in range(bootstrap_samples):
            sample = [solutions[rng.randrange(len(solutions))] for _ in solutions]
            scored = _score_solutions(sample, semantics)
            if scored["score"] is not None:
                boot.append(scored["score"])
    repeat = [mean((v[i] - v[j]) ** 2 for i in range(len(v)) for j in range(i + 1, len(v)))
              for s, v in normal_values.items() if s in semantics and len(v) >= 2]
    cross = [mean((v[i] - v[j]) ** 2 for i in range(len(v)) for j in range(i + 1, len(v)))
             for s, v in values.items() if s in semantics]
    diagnostics = {
        "score_ci95": [_percentile(boot, .025), _percentile(boot, .975)] if boot else None,
        "bootstrap_valid_samples": len(boot),
        "semantic_coverage": {s: len(v) for s, v in sorted(values.items())},
        "Q_repeat": mean(repeat) if repeat else None,
        "Q_cross": mean(cross) if cross else None,
        "Q_null": _cross_semantic_null(solutions, semantics),
    }
    diagnostics["semantic_groups"] = semantics
    return diagnostics


def score_csvc_symmetric(normal: EpisodeDeltas, diverse: EpisodeDeltas, *,
                         epsilon: float = 1e-12, bootstrap_samples: int = 2000,
                         bootstrap_seed: int = 42) -> dict[str, object]:
    """Score (G-D)/(G+D), retaining negative values and assigning -1 at G+D=0.

    Normal repeats collapse to one semantic median profile; semantics have equal
    weight even with partial coverage. Missing semantic coverage is not scored.
    The retained epsilon argument does not alter the paper's exact ratio.
    """
    if isinstance(
        epsilon,
        bool
    ) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise VMBMKError("CSVC epsilon must be finite and positive")
    if isinstance(
        bootstrap_samples,
        bool
    ) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise VMBMKError("CSVC bootstrap_samples must be a nonnegative integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise VMBMKError("CSVC bootstrap_seed must be an integer")
    normal_rows = _validated(normal, "normal")
    diverse_rows = _validated(diverse, "diverse")
    normal_values = _by_semantic(normal_rows)
    solutions = [{s: median(v) for s, v in normal_values.items()}, *diverse_rows]
    values = _by_semantic(solutions)
    semantics = sorted(s for s, v in values.items() if len(v) >= 2)
    result = _score_solutions(solutions, semantics)
    result.update(_csvc_diagnostics(
        result, solutions, semantics, normal_values, values, bootstrap_samples, bootstrap_seed,
    ))
    return result


def aggregate_csvc(
    task_results: Mapping[str, Mapping[str, object]],
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Macro-average covered task scores, including the defined zero-denominator -1.

    Tasks lacking matched semantic coverage remain N/A rather than receiving an
    invented score. Identifiability is diagnostic, not a score-exclusion rule.
    """
    if not task_results:
        raise VMBMKError("CSVC task results are empty")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 0
    ):
        raise VMBMKError("CSVC bootstrap_samples must be a nonnegative integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise VMBMKError("CSVC bootstrap_seed must be an integer")

    scores: list[float] = []
    raw_scores: list[float] = []
    task_scores: dict[str, float | None] = {}
    identifiable_tasks = 0
    for task_id, result in sorted(task_results.items()):
        identifiable = result.get("identifiable")
        value = result.get("score")
        if not isinstance(
            task_id,
            str
        ) or not task_id.strip() or not isinstance(identifiable, bool):
            raise VMBMKError(f"CSVC invalid task result for {task_id!r}")
        identifiable_tasks += int(identifiable)
        if value is None:
            if identifiable:
                raise VMBMKError(f"CSVC identifiable task has no score: {task_id!r}")
            task_scores[task_id] = None
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -1.0 <= float(value) <= 1.0
        ):
            raise VMBMKError(f"CSVC invalid score for {task_id!r}")
        score = float(value)
        scores.append(score)
        task_scores[task_id] = score
        raw_value = result.get("raw_score")
        if (
            isinstance(raw_value, (int, float))
            and not isinstance(raw_value, bool)
            and math.isfinite(float(raw_value))
        ):
            raw_scores.append(float(raw_value))

    if not scores:
        return {
            "score": None,
            "raw_score": None,
            "score_ci95": None,
            "task_scores": task_scores,
            "tasks": len(task_results),
            "identifiable_tasks": 0,
            "unidentifiable_task_rate": 1.0,
            "bootstrap_samples": bootstrap_samples,
        }

    macro_score = mean(scores)
    if bootstrap_samples:
        rng = random.Random(bootstrap_seed)
        bootstrap_scores = [
            mean(scores[rng.randrange(len(scores))] for _ in range(len(scores)))
            for _ in range(bootstrap_samples)
        ]
        score_ci95: list[float] | None = [
            _percentile(bootstrap_scores, 0.025),
            _percentile(bootstrap_scores, 0.975),
        ]
    else:
        score_ci95 = None
    return {
        "score": macro_score,
        "raw_score": mean(raw_scores) if raw_scores else None,
        "score_ci95": score_ci95,
        "task_scores": task_scores,
        "tasks": len(task_results),
        "identifiable_tasks": identifiable_tasks,
        "unidentifiable_task_rate": 1.0 - identifiable_tasks / len(task_results),
        "bootstrap_samples": bootstrap_samples,
    }


WINDOW_SAMPLES = 4
WINDOW_STEP_SECONDS = 0.15


def _semantic_spans(episode: Episode) -> dict[str, tuple[int, int]]:
    """Return stable-semantic execution spans for one CSVC episode.

    The delta protocol requires an interval, not only a completion point. A
    one-frame span is how the loader represents completion-only ``subtask_frame`` data,
    which is intentionally rejected here because its start state is ambiguous
    when legal solutions reorder subtasks.
    """
    spans = {
        semantic: (int(start), int(end))
        for semantic, (start, end) in episode.subtask_spans.items()
    }
    if not spans:
        raise VMBMKError(
            f"{episode.task_id}/{episode.episode_id}: CSVC requires semantic "
            "subtask spans"
        )
    point_only = sorted(
        semantic for semantic, (start, end) in spans.items() if end - start <= 1
    )
    if point_only:
        raise VMBMKError(
            f"{episode.task_id}/{episode.episode_id}: CSVC requires start/end "
            f"spans, not completion-only points: {point_only}"
        )
    return spans


def _window_pairs(
    episode: Episode,
    semantic: str,
    span: tuple[int, int],
    *,
    window_samples: int,
    window_step_seconds: float,
) -> list[tuple[int, int]]:
    start, end = span
    pairs: list[tuple[int, int]] = []
    for sample_index in range(window_samples):
        # Keep short annotated spans valid without crossing the two endpoints.
        inward = min(
            round(sample_index * episode.fps * window_step_seconds),
            (end - start - 2) // 2,
        )
        before = start + inward
        after = end - 1 - inward
        if before >= after:
            raise VMBMKError(
                f"{episode.task_id}/{episode.episode_id}/{semantic}: semantic "
                "span is too short for the CSVC boundary windows"
            )
        pairs.append((before, after))
    return pairs


def _plan(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    *,
    window_samples: int = WINDOW_SAMPLES,
    window_step_seconds: float = WINDOW_STEP_SECONDS,
) -> tuple[
    list[CompareQuery],
    dict[str, tuple[str, str, str, str, int]],
    dict[str, set[str]],
]:
    if window_samples <= 0 or window_step_seconds <= 0:
        raise VMBMKError("CSVC window parameters must be positive")
    selected_domains = {"id"} if domains is None else set(domains)
    if selected_domains != {"id"}:
        raise VMBMKError("CSVC supports only the ID domain")
    selected_tasks = sorted(dataset.tasks) if task_ids is None else list(task_ids)
    queries: list[CompareQuery] = []
    labels: dict[str, tuple[str, str, str, str, int]] = {}
    task_domains: dict[str, set[str]] = defaultdict(set)

    for task_id in selected_tasks:
        try:
            task = dataset.tasks[task_id]
        except KeyError as exc:
            raise VMBMKError(f"unknown task {task_id!r}") from exc
        if not {"CSVC", "CSPC"} & set(task.metrics):
            continue
        episodes = [
            episode
            for episode in task.episodes.values()
            if episode.success
            and episode.cspc_solution_type in {"normal", "diverse"}
            and episode.domain in selected_domains
        ]
        roles = {episode.cspc_solution_type for episode in episodes}
        if roles != {"normal", "diverse"}:
            raise VMBMKError(
                f"CSVC {task_id!r} requires normal and diverse successful episodes"
            )
        for episode in sorted(episodes, key=lambda row: row.episode_id):
            role = str(episode.cspc_solution_type)
            task_domains[task_id].add(episode.domain)
            for semantic, span in sorted(_semantic_spans(episode).items()):
                pairs = _window_pairs(
                    episode,
                    semantic,
                    span,
                    window_samples=window_samples,
                    window_step_seconds=window_step_seconds,
                )
                for sample_index, (before, after) in enumerate(pairs):
                    query_id = ":".join(
                        (
                            "csvc",
                            task_id,
                            episode.episode_id,
                            semantic,
                            str(sample_index),
                        )
                    )
                    queries.append(
                        CompareQuery(
                            query_id,
                            StateRef(task_id, episode.episode_id, before),
                            StateRef(task_id, episode.episode_id, after),
                            task.instruction,
                        )
                    )
                    labels[query_id] = (
                        task_id,
                        episode.episode_id,
                        role,
                        semantic,
                        sample_index,
                    )
    if not queries:
        raise VMBMKError("dataset has no scorable CSVC semantic spans")
    return queries, labels, task_domains


def build_csvc_queries(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> list[CompareQuery]:
    return _plan(dataset, task_ids, domains)[0]


def score_csvc(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> dict[str, Any]:
    queries, labels, task_domains = _plan(dataset, task_ids, domains)
    scores = result_scores(queries, results, "CSPC", "compare")
    grouped: dict[
        str, dict[str, dict[str, dict[str, list[float]]]]
    ] = defaultdict(
        lambda: {
            "normal": defaultdict(lambda: defaultdict(list)),
            "diverse": defaultdict(lambda: defaultdict(list)),
        }
    )
    for query_id, (task_id, episode_id, role, semantic, _) in labels.items():
        grouped[task_id][role][episode_id][semantic].append(scores[query_id])

    task_results: dict[str, dict[str, object]] = {}
    for task_id, roles in sorted(grouped.items()):
        deltas = {
            role: {
                episode_id: {
                    semantic: float(median(values))
                    for semantic, values in semantic_rows.items()
                }
                for episode_id, semantic_rows in episodes.items()
            }
            for role, episodes in roles.items()
        }
        result = score_csvc_symmetric(
            deltas["normal"],
            deltas["diverse"],
            bootstrap_samples=2000,
            bootstrap_seed=42,
        )
        result.update(
            {
                "task_id": task_id,
                "domains": sorted(task_domains[task_id]),
                "delta_signal": "median_compare_boundary_windows",
                "window_samples": WINDOW_SAMPLES,
                "window_step_seconds": WINDOW_STEP_SECONDS,
                "short_span_policy": "clamp_inward_to_ordered_endpoints",
                "semantic_alignment": "stable_subtask_id",
                "null_status": "configured_cross_solution_semantic_mismatch_all_pairs",
            }
        )
        task_results[task_id] = result

    aggregate = aggregate_csvc(
        task_results,
        bootstrap_samples=2000,
        bootstrap_seed=42,
    )
    return {
        "protocol": CSVC_PROTOCOL,
        "mean": aggregate["score"],
        **aggregate,
        "task_results": task_results,
        "domains": {
            "id": {
                "mean": aggregate["score"],
                "tasks": {task: result["score"] for task, result in task_results.items()},
            }
        },
        "alignment": "stable semantic subtask ID, never chronological rank",
    }


def run_csvc(
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
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=["csvc"])
    queries = build_csvc_queries(dataset, task_ids, domains)
    results = run_queries(
        data_root,
        queries,
        model,
        operation_path,
        gpu,
        metric="csvc",
        mode=mode,
    )
    return score_csvc(dataset, results, task_ids, domains)
