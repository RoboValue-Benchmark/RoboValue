from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import Result, StateRef, ValueQuery
from vmbmk.metrics.utils.statistics import mean
from vmbmk.metrics.utils.results import result_scores
from vmbmk.inference.execution import run_queries
from vmbmk.metrics.utils.aggregation import summarize_task_scores


FPL_SAMPLE_RATE_HZ = 5.0
FPL_ZIGZAG_ALPHA = 0.2
FPL_PROTOCOL = "zigzag-onset-normalized-alpha0.2-v1"


def _anchor_frames(episode: Episode) -> list[int]:
    """Uniformly sample the full trajectory from frame zero at 5 Hz."""
    frame_period = episode.fps / FPL_SAMPLE_RATE_HZ
    anchors: list[int] = []
    sample_index = 0
    while True:
        frame = int(round(sample_index * frame_period))
        if frame >= episode.num_frames:
            break
        if not anchors or anchors[-1] != frame:
            anchors.append(frame)
        sample_index += 1
    final_frame = episode.num_frames - 1
    if anchors[-1] != final_frame:
        anchors.append(final_frame)
    return anchors


def _maximum_drawdown(frames_and_values: Sequence[tuple[int, float]]) -> float:
    """Return the largest historical-peak to later-value decrease."""
    if len(frames_and_values) < 2:
        raise VMBMKError("FPL maximum drawdown requires at least two anchors")
    peak_value = frames_and_values[0][1]
    maximum_drop = 0.0
    for _, value in frames_and_values[1:]:
        maximum_drop = max(maximum_drop, peak_value - value)
        peak_value = max(peak_value, value)
    return maximum_drop


def _zigzag_descents(
    frames_and_values: Sequence[tuple[int, float]], threshold: float
) -> list[tuple[int, int, float]]:
    """Return threshold-confirmed descents plus an unfinished terminal descent."""
    if len(frames_and_values) < 2:
        raise VMBMKError("FPL ZigZag requires at least two anchors")
    if threshold <= 0:
        return []

    direction = "up"
    peak_index = 0
    peak_value = frames_and_values[0][1]
    trough_index = 0
    trough_value = peak_value
    descents: list[tuple[int, int, float]] = []

    for index in range(1, len(frames_and_values)):
        value = frames_and_values[index][1]
        if direction == "up":
            # The last point on a flat peak gives the shortest eventual descent.
            if value >= peak_value:
                peak_index, peak_value = index, value
            elif peak_value - value >= threshold:
                direction = "down"
                trough_index, trough_value = index, value
        else:
            # The first point on a flat trough gives the shortest descent.
            if value < trough_value:
                trough_index, trough_value = index, value
            elif value - trough_value >= threshold:
                descents.append(
                    (
                        frames_and_values[peak_index][0],
                        frames_and_values[trough_index][0],
                        peak_value - trough_value,
                    )
                )
                direction = "up"
                peak_index, peak_value = index, value

    if direction == "down":
        descents.append(
            (
                frames_and_values[peak_index][0],
                frames_and_values[trough_index][0],
                peak_value - trough_value,
            )
        )
    else:
        # Preserve a terminal decline even when its rebound threshold has not
        # yet been reached before the video ends.
        terminal_trough = peak_index
        terminal_value = peak_value
        for index in range(peak_index + 1, len(frames_and_values)):
            value = frames_and_values[index][1]
            if value < terminal_value:
                terminal_trough, terminal_value = index, value
        if terminal_value < peak_value:
            descents.append(
                (
                    frames_and_values[peak_index][0],
                    frames_and_values[terminal_trough][0],
                    peak_value - terminal_value,
                )
            )
    return descents


def _select_descent(
    descents: Sequence[tuple[int, int, float]],
) -> tuple[int, int, float] | None:
    if not descents:
        return None
    return min(descents, key=lambda row: (-row[2], row[1] - row[0], row[0]))


def _plan(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> tuple[
    list[ValueQuery],
    dict[str, tuple[str, str, str, int]],
    dict[tuple[str, str, str], tuple[int, int, int]],
]:
    selected_tasks = sorted(dataset.tasks) if task_ids is None else list(task_ids)
    selected_domains = set(domains) if domains is not None else None
    observed_domains: set[str] = set()
    queries: list[ValueQuery] = []
    anchors: dict[str, tuple[str, str, str, int]] = {}
    intervals: dict[tuple[str, str, str], tuple[int, int, int]] = {}
    for task_id in selected_tasks:
        try:
            task = dataset.tasks[task_id]
        except KeyError as exc:
            raise VMBMKError(f"unknown task {task_id!r}") from exc
        if "FPL" not in task.metrics:
            continue
        for episode in sorted(task.episodes.values(), key=lambda row: row.episode_id):
            if episode.fpl_interval is None:
                continue
            # An invalid recovery has two distinct errors: the initial failure
            # and the failed recovery. FPL localizes one dominant event only.
            if episode.trr_role == "r_zero":
                continue
            if selected_domains is not None and episode.domain not in selected_domains:
                continue
            episode_key = (episode.domain, task_id, episode.episode_id)
            intervals[episode_key] = (*episode.fpl_interval, episode.num_frames)
            observed_domains.add(episode.domain)
            for frame in _anchor_frames(episode):
                query_id = f"fpl:{task_id}:{episode.episode_id}:{frame}"
                queries.append(
                    ValueQuery(
                        query_id,
                        StateRef(task_id, episode.episode_id, frame),
                        task.instruction,
                    )
                )
                anchors[query_id] = (*episode_key, frame)
    if not queries:
        raise VMBMKError("dataset has no annotated FPL failure intervals")
    if selected_domains is not None and observed_domains != selected_domains:
        raise VMBMKError(
            f"FPL domains have no annotated failures: {sorted(selected_domains - observed_domains)}"
        )
    return queries, anchors, intervals


def build_fpl_queries(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> list[ValueQuery]:
    return _plan(dataset, task_ids, domains)[0]


def score_fpl(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> dict[str, Any]:
    queries, anchors, intervals = _plan(dataset, task_ids, domains)
    query_ids = {query.query_id for query in queries}
    scores = result_scores(
        queries,
        [result for result in results if result.query_id in query_ids],
        "FPL",
        "value",
    )
    curves: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for query_id, (domain, task_id, episode_id, frame) in anchors.items():
        curves[(domain, task_id, episode_id)].append((frame, scores[query_id]))

    task_distances: dict[tuple[str, str], list[float]] = defaultdict(list)
    episodes = []
    for key, curve in sorted(curves.items()):
        domain, task_id, episode_id = key
        curve.sort()
        start, _end, num_frames = intervals[key]
        last_frame = num_frames - 1
        if last_frame <= 0:
            raise VMBMKError(f"FPL episode {task_id}/{episode_id} has zero duration")
        maximum_drop = _maximum_drawdown(curve)
        threshold = FPL_ZIGZAG_ALPHA * maximum_drop
        descents = _zigzag_descents(curve, threshold) if maximum_drop > 0 else []
        selected = _select_descent(descents)
        if selected is None:
            predicted_start = None
            frame_error = max(start, last_frame - start)
        else:
            predicted_start = selected[0]
            frame_error = abs(predicted_start - start)
        normalized_error = frame_error / last_frame
        task_distances[(domain, task_id)].append(normalized_error)
        episodes.append(
            {
                "domain": domain,
                "task_id": task_id,
                "episode_id": episode_id,
                "ground_truth_onset_frame": start,
                "predicted_onset_frame": predicted_start,
                "last_frame": last_frame,
                "normalized_error": normalized_error,
                "maximum_drawdown": maximum_drop,
                "zigzag_threshold": threshold,
                "zigzag_descents": [list(row) for row in descents],
                "selected_descent": list(selected) if selected is not None else None,
                "sampled_value_curve": [
                    {"frame_index": frame, "value": value}
                    for frame, value in curve
                ],
            }
        )
    task_scores = {
        key: mean(values) for key, values in sorted(task_distances.items())
    }
    return {
        "protocol": FPL_PROTOCOL,
        "primary": "normalized_onset_error_lower_is_better",
        **summarize_task_scores(task_scores),
        "zigzag_alpha": FPL_ZIGZAG_ALPHA,
        "sampling": "uniform-from-frame-zero",
        "sample_rate_hz": FPL_SAMPLE_RATE_HZ,
        "sample_period_s": 1.0 / FPL_SAMPLE_RATE_HZ,
        "episodes": episodes,
    }


def run_fpl(
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
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=["fpl"])
    queries = build_fpl_queries(dataset, task_ids, domains)
    results = run_queries(
        data_root,
        queries,
        model,
        operation_path,
        gpu,
        metric="fpl",
        mode=mode,
    )
    return score_fpl(dataset, results, task_ids, domains)
