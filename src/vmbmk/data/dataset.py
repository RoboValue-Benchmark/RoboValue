from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.errors import DataValidationError


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataValidationError(f"{label} must be an object")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise DataValidationError(f"{label} has unknown fields: {unknown}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise DataValidationError(f"{label} must be finite")
    return number


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"missing file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{path}: invalid JSON: {exc}") from exc
    return _mapping(value, str(path))


def _asset_path(root: Path, value: Any, label: str) -> Path:
    """Resolve declared relative assets, including shared external raw-data trees."""
    text = _text(value, label)
    source = Path(text)
    if source.is_absolute():
        raise DataValidationError(f"{label} must be a relative path")
    path = (root / source).resolve()
    if not path.is_file():
        raise DataValidationError(f"{label} does not exist: {path}")
    return path


def _frame(value: Any, num_frames: int, label: str) -> int:
    frame = _integer(value, label)
    if not 0 <= frame < num_frames:
        raise DataValidationError(
            f"{label} {frame} is outside [0, {num_frames})"
        )
    return frame


def _span(value: Any, num_frames: int, label: str) -> None:
    row = _mapping(value, label)
    _keys(row, {"start_frame", "end_frame_exclusive"}, label)
    start = _integer(row.get("start_frame"), f"{label}.start_frame")
    end = _integer(
        row.get("end_frame_exclusive"), f"{label}.end_frame_exclusive"
    )
    if not 0 <= start < end <= num_frames:
        raise DataValidationError(
            f"{label} interval [{start}, {end}) is invalid for {num_frames} frames"
        )


def _span_value(value: Any, num_frames: int, label: str) -> tuple[int, int]:
    _span(value, num_frames, label)
    return int(value["start_frame"]), int(value["end_frame_exclusive"])


def _validate_task_metadata(
    row: Mapping[str, Any],
    task_id: str,
    label: str,
    requested_metrics: frozenset[str] | None = None,
) -> tuple[str, frozenset[str], frozenset[str]]:
    if requested_metrics is None:
        _keys(row, {"task_id", "instruction", "supports", "subtasks", "tga"}, label)
    if _text(row.get("task_id"), f"{label}.task_id") != task_id:
        raise DataValidationError(f"{label}.task_id must match directory {task_id!r}")
    instruction = _text(row.get("instruction"), f"{label}.instruction")
    supports = _mapping(row.get("supports"), f"{label}.supports")
    _keys(supports, {"metrics"}, f"{label}.supports")
    supported_metrics = supports.get("metrics")
    if not isinstance(supported_metrics, list) or any(
        not isinstance(item, str) or not item for item in supported_metrics
    ):
        raise DataValidationError(f"{label}.supports.metrics must be a string array")
    # Subtask descriptions are only part of the input contract for metrics that
    # classify or compare subtasks.  In a metric-scoped load, malformed or
    # unrelated subtask metadata must not prevent another metric from running.
    validate_subtasks = requested_metrics is None or bool(
        {"sia", "cspc"} & requested_metrics
    )
    subtasks = row.get("subtasks", []) if validate_subtasks else []
    if not isinstance(subtasks, list):
        raise DataValidationError(f"{label}.subtasks must be an array")
    subtask_ids = []
    for index, item in enumerate(subtasks):
        item_label = f"{label}.subtasks[{index}]"
        item = _mapping(item, item_label)
        _keys(item, {"id", "description"}, item_label)
        subtask_ids.append(_text(item.get("id"), f"{item_label}.id"))
        _text(item.get("description"), f"{item_label}.description")
    if len(subtask_ids) != len(set(subtask_ids)):
        raise DataValidationError(f"{label}.subtasks contains duplicate IDs")
    validate_tga = requested_metrics is None or bool(
        {"tga_easy", "tga_hard"} & requested_metrics
    )
    if validate_tga and row.get("tga") is not None:
        tga = _mapping(row["tga"], f"{label}.tga")
        _keys(tga, {"counterfactual_instructions"}, f"{label}.tga")
        items = tga.get("counterfactual_instructions")
        if not isinstance(items, list):
            raise DataValidationError(
                f"{label}.tga.counterfactual_instructions must be an array"
            )
        identifiers = []
        for index, item in enumerate(items):
            item_label = f"{label}.tga.counterfactual_instructions[{index}]"
            item = _mapping(item, item_label)
            _keys(item, {"id", "description"}, item_label)
            identifiers.append(_text(item.get("id"), f"{item_label}.id"))
            _text(item.get("description"), f"{item_label}.description")
        if len(identifiers) != len(set(identifiers)):
            raise DataValidationError(
                f"{label}.tga.counterfactual_instructions contains duplicate IDs"
            )
    return instruction, frozenset(subtask_ids), frozenset(
        "CSVC" if metric.upper() in {"CSPC", "CSVC"} else metric
        for metric in supported_metrics
    )


def _validate_robot_data(row: Mapping[str, Any], label: str) -> None:
    _keys(row, {"uri", "format", "state", "action"}, label)
    if "format" in row:
        _text(row["format"], f"{label}.format")
    for field in ("state", "action"):
        if field not in row:
            continue
        item_label = f"{label}.{field}"
        item = _mapping(row[field], item_label)
        _keys(item, {"key", "indices"}, item_label)
        _text(item.get("key"), f"{item_label}.key")
        indices = item.get("indices")
        if not isinstance(indices, list) or not indices:
            raise DataValidationError(f"{item_label}.indices must be a non-empty array")
        for index, value in enumerate(indices):
            value = _integer(value, f"{item_label}.indices[{index}]")
            if value < 0:
                raise DataValidationError(
                    f"{item_label}.indices[{index}] must be non-negative"
                )


@dataclass(frozen=True)
class Annotation:
    """Validated metric inputs; legacy-only fields do not enter runtime episodes."""
    voc_frames: tuple[int, ...]
    voc_mem_frames: tuple[int, ...]
    subtask_frames: Mapping[str, int]
    subtask_spans: Mapping[str, tuple[int, int]]
    fpl_interval: tuple[int, int] | None
    trr_spans: Mapping[str, tuple[int, int]]


def _validate_legacy_lpd(
    row: Mapping[str, Any], num_frames: int, label: str, enabled: bool,
) -> None:
    """Validate archived annotations without carrying retired LPD runtime fields."""
    lpd = row.get("lpd", []) if enabled else []
    if not isinstance(lpd, list):
        raise DataValidationError(f"{label}.lpd must be an array")
    lpd_ids = []
    for index, item in enumerate(lpd):
        item_label = f"{label}.lpd[{index}]"
        item = _mapping(item, item_label)
        _keys(item, {"id", "frameA", "frameB", "type"}, item_label)
        lpd_ids.append(_text(item.get("id"), f"{item_label}.id"))
        _frame(item.get("frameA"), num_frames, f"{item_label}.frameA")
        _frame(item.get("frameB"), num_frames, f"{item_label}.frameB")
        if item.get("type") not in {"forward", "stagnation", "backward"}:
            raise DataValidationError(f"{item_label}.type is invalid")
    if len(lpd_ids) != len(set(lpd_ids)):
        raise DataValidationError(f"{label}.lpd contains duplicate IDs")


def _validate_trr_annotation(
    trr: Any, num_frames: int, trr_role: str | None, label: str,
) -> dict[str, tuple[int, int]]:
    """Normalize only supported legacy/unified TRR span contracts."""
    if trr is None:
        return {}
    trr = _mapping(trr, f"{label}.trr")
    if trr_role is None:
        accepted = (
            {"failure_span", "recovery_span"},
            {"failure_span", "recovery_success_span"},
            {"failure_span", "recovery_span", "recovery_success_span"},
            {"useless_span"},
            {"failure_span", "recovery_failed_span"},
            {"failure_span", "recovery_span", "recovery_failed_span"},
            {"continue_span"},
            {"failure_span", "continue_span"},
        )
        fields = set(trr)
        if fields not in accepted:
            raise DataValidationError(
                f"{label}.trr has unsupported fields {sorted(fields)}"
            )
        trr_spans = {
            field: _span_value(trr[field], num_frames, f"{label}.trr.{field}")
            for field in fields
        }
        return trr_spans
    legacy_fields = {
        "r_plus": {"failure_span", "recovery_span"},
        "r_zero": {"useless_span"},
        "r_minus": {"continue_span"},
    }[trr_role]
    unified_required = {
        "r_plus": {"failure_span", "recovery_success_span"},
        "r_zero": {"failure_span", "recovery_failed_span"},
        "r_minus": {"failure_span", "continue_span"},
    }[trr_role]
    unified_optional = {"recovery_span"} if trr_role != "r_minus" else set()
    fields = set(trr)
    is_legacy = fields == legacy_fields
    is_unified = unified_required <= fields <= unified_required | unified_optional
    if not is_legacy and not is_unified:
        raise DataValidationError(
            f"{label}.trr for {trr_role} must match legacy "
            f"{sorted(legacy_fields)} or unified required "
            f"{sorted(unified_required)} with optional "
            f"{sorted(unified_optional)}"
        )
    trr_spans = {
        field: _span_value(trr[field], num_frames, f"{label}.trr.{field}")
        for field in fields
    }
    if is_unified:
        result_field = {
            "r_plus": "recovery_success_span",
            "r_zero": "recovery_failed_span",
            "r_minus": "continue_span",
        }[trr_role]
        ordered_fields = ["failure_span"]
        if "recovery_span" in fields:
            ordered_fields.append("recovery_span")
        ordered_fields.append(result_field)
        for left, right in zip(ordered_fields, ordered_fields[1:]):
            if trr_spans[left][1] != trr_spans[right][0]:
                raise DataValidationError(
                    f"{label}.trr spans must be contiguous; {left} ends at "
                    f"{trr_spans[left][1]} but {right} starts at "
                    f"{trr_spans[right][0]}"
                )
    return trr_spans


def _validate_annotation(
    row: Mapping[str, Any],
    num_frames: int,
    trr_role: str | None,
    subtask_ids: frozenset[str],
    label: str,
    metrics: frozenset[str] | None = None,
) -> Annotation:
    if metrics is None:
        _keys(
            row,
            {"subtask_segments", "voc", "voc_mem", "fpl", "lpd", "trr"},
            label,
        )
    validate_subtasks = metrics is None or bool(
        {"sia", "cspc"} & metrics
    )
    validate_voc = metrics is None or bool({"voc", "cycle_voc"} & metrics)
    validate_voc_mem = metrics is None or "voc_mem" in metrics
    validate_fpl = metrics is None or "fpl" in metrics
    validate_lpd = metrics is None or "lpd" in metrics
    validate_trr = metrics is None or "trr" in metrics

    segments = row.get("subtask_segments", []) if validate_subtasks else []
    if not isinstance(segments, list):
        raise DataValidationError(f"{label}.subtask_segments must be an array")
    segment_ids = []
    subtask_frames: dict[str, int] = {}
    subtask_spans: dict[str, tuple[int, int]] = {}
    for index, item in enumerate(segments):
        item_label = f"{label}.subtask_segments[{index}]"
        item = _mapping(item, item_label)
        allowed = {"id", "subtask_frame", "start_frame", "end_frame_exclusive"}
        _keys(item, allowed, item_label)
        segment_id = _text(item.get("id"), f"{item_label}.id")
        if segment_id not in subtask_ids:
            raise DataValidationError(
                f"{item_label}.id references unknown subtask {segment_id!r}"
            )
        segment_ids.append(segment_id)
        has_point = "subtask_frame" in item
        has_span = {"start_frame", "end_frame_exclusive"} <= set(item)
        if has_point == has_span:
            raise DataValidationError(
                f"{item_label} must contain exactly one of subtask_frame or a span"
            )
        if has_point:
            _frame(item.get("subtask_frame"), num_frames, f"{item_label}.subtask_frame")
            start = int(item["subtask_frame"])
            end = start + 1
        else:
            _span(
                {
                    "start_frame": item["start_frame"],
                    "end_frame_exclusive": item["end_frame_exclusive"],
                },
                num_frames,
                item_label,
            )
            start = int(item["start_frame"])
            end = int(item["end_frame_exclusive"])
        subtask_frames[segment_id] = start
        subtask_spans[segment_id] = (start, end)
    if len(segment_ids) != len(set(segment_ids)):
        raise DataValidationError(f"{label}.subtask_segments contains duplicate IDs")
    point_frames: dict[str, tuple[int, ...]] = {}
    for field, enabled in (("voc", validate_voc), ("voc_mem", validate_voc_mem)):
        if not enabled:
            point_frames[field] = ()
            continue
        items = row.get(field, [])
        if not isinstance(items, list):
            raise DataValidationError(f"{label}.{field} must be an array")
        frames = []
        for index, item in enumerate(items):
            item_label = f"{label}.{field}[{index}]"
            item = _mapping(item, item_label)
            _keys(item, {"frame_index"}, item_label)
            frames.append(
                _frame(item.get("frame_index"), num_frames, f"{item_label}.frame_index")
            )
        if len(frames) != len(set(frames)):
            raise DataValidationError(f"{label}.{field} contains duplicate frames")
        point_frames[field] = tuple(frames)
    fpl_interval = None
    if validate_fpl and row.get("fpl") is not None:
        fpl = _mapping(row["fpl"], f"{label}.fpl")
        if set(fpl) == {"failure_span"}:
            fpl_interval = _span_value(
                fpl["failure_span"], num_frames, f"{label}.fpl.failure_span"
            )
        elif set(fpl) == {"failure_start", "failure_end"}:
            start = _integer(
                fpl.get("failure_start"), f"{label}.fpl.failure_start"
            )
            end = _integer(fpl.get("failure_end"), f"{label}.fpl.failure_end")
            if not 0 <= start < end <= num_frames:
                raise DataValidationError(f"{label}.fpl interval is invalid")
            fpl_interval = (start, end)
        else:
            raise DataValidationError(
                f"{label}.fpl must contain failure_span or "
                "failure_start/failure_end"
            )
    _validate_legacy_lpd(row, num_frames, label, validate_lpd)
    trr_spans = _validate_trr_annotation(
        row.get("trr") if validate_trr else None, num_frames, trr_role, label,
    )
    return Annotation(
        point_frames["voc"], point_frames["voc_mem"], subtask_frames,
        subtask_spans, fpl_interval, trr_spans,
    )


@dataclass(frozen=True)
class Episode:
    task_id: str
    episode_id: str
    root: Path
    success: bool
    domain: str
    fps: float
    num_frames: int
    videos: Mapping[str, Path]
    voc_frames: tuple[int, ...]
    voc_mem_frames: tuple[int, ...]
    subtask_frames: Mapping[str, int] = field(default_factory=dict)
    subtask_spans: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    fpl_interval: tuple[int, int] | None = None
    trr_group_id: str | None = None
    trr_role: str | None = None
    trr_spans: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    cspc_solution_type: str | None = None

    def video(self, view: str) -> Path:
        try:
            return self.videos[view]
        except KeyError as exc:
            raise DataValidationError(
                f"{self.task_id}/{self.episode_id} has no {view!r} video"
            ) from exc


@dataclass(frozen=True)
class Task:
    task_id: str
    instruction: str
    episodes: Mapping[str, Episode]
    metrics: frozenset[str]
    subtasks: Mapping[str, str] = field(default_factory=dict)
    counterfactual_instructions: Mapping[str, str] = field(default_factory=dict)


def _load_episode(
    task_id: str, episode_root: Path, subtask_ids: frozenset[str],
    requested_metrics: frozenset[str] | None,
) -> Episode:
    """Read and validate one episode without coupling it to task discovery."""
    episode_id = episode_root.name
    metadata_file = episode_root / "metadata.json"
    label = str(metadata_file)
    row = _read_json(metadata_file)
    if requested_metrics is None:
        _keys(
            row,
            {
                "episode_id",
                "world_type",
                "domain",
                "success",
                "fps",
                "num_frames",
                "assets",
                "trr",
                "cspc",
            },
            label,
        )
    if _text(row.get("episode_id"), f"{label}.episode_id") != episode_id:
        raise DataValidationError(
            f"{label}.episode_id must match directory {episode_id!r}"
        )
    if row.get("world_type") not in {"real", "sim"}:
        raise DataValidationError(f"{label}.world_type is invalid")
    if row.get("domain") not in {"id", "env", "emb"}:
        raise DataValidationError(f"{label}.domain is invalid")
    if not isinstance(row.get("success"), bool):
        raise DataValidationError(f"{label}.success must be boolean")
    fps = _number(row.get("fps"), f"{label}.fps")
    num_frames = _integer(row.get("num_frames"), f"{label}.num_frames")
    if fps <= 0 or num_frames <= 0:
        raise DataValidationError(f"{label} has invalid fps/num_frames")
    assets = _mapping(row.get("assets"), f"{label}.assets")
    if requested_metrics is None:
        _keys(assets, {"videos", "robot_data"}, f"{label}.assets")
    raw_videos = _mapping(assets.get("videos"), f"{label}.assets.videos")
    if not raw_videos:
        raise DataValidationError(f"{label}.assets.videos is empty")
    videos = {
        _text(view, f"{label}.assets.videos view"): _asset_path(
            episode_root, path, f"{label}.assets.videos.{view}"
        )
        for view, path in raw_videos.items()
    }
    if requested_metrics is None:
        robot_data = _mapping(
            assets.get("robot_data"), f"{label}.assets.robot_data"
        )
        _validate_robot_data(robot_data, f"{label}.assets.robot_data")
        _asset_path(
            episode_root,
            robot_data.get("uri"),
            f"{label}.assets.robot_data.uri",
        )
    trr_role = None
    validate_trr_metadata = requested_metrics is None or bool(
        {
            "vs",
            "trr",
            "cycle_voc",
            "tga_easy",
            "tga_hard",
        }
        & requested_metrics
    )
    if validate_trr_metadata and row.get("trr") is not None:
        trr = _mapping(row["trr"], f"{label}.trr")
        _keys(trr, {"group_id", "role"}, f"{label}.trr")
        _text(trr.get("group_id"), f"{label}.trr.group_id")
        trr_role = trr.get("role")
        if trr_role not in {"r_plus", "r_minus", "r_zero"}:
            raise DataValidationError(f"{label}.trr.role is invalid")
        if row["success"] != (trr_role == "r_plus"):
            raise DataValidationError(
                f"{label}.success conflicts with trr.role"
            )
    validate_cspc_metadata = requested_metrics is None or bool(
        {"vs", "cspc", "sia", "cycle_voc", "tga_easy", "tga_hard"} & requested_metrics
    )
    if validate_cspc_metadata and row.get("cspc") is not None:
        cspc = _mapping(row["cspc"], f"{label}.cspc")
        _keys(cspc, {"solution_type"}, f"{label}.cspc")
        if cspc.get("solution_type") not in {"normal", "diverse"}:
            raise DataValidationError(
                f"{label}.cspc.solution_type is invalid"
            )
    annotation_metrics = (
        None
        if requested_metrics is None
        else requested_metrics
        & {
            "voc",
            "cycle_voc",
            "voc_mem",
            "sia",
            "cspc",
            "fpl",
            "lpd",
            "trr",
        }
    )
    annotation = (
        _read_json(episode_root / "annotation.json")
        if requested_metrics is None or annotation_metrics
        else {}
    )
    parsed = _validate_annotation(
        annotation,
        num_frames,
        trr_role,
        subtask_ids,
        str(episode_root / "annotation.json"),
        requested_metrics,
    )
    return Episode(
        task_id=task_id,
        episode_id=episode_id,
        root=episode_root,
        success=bool(row["success"]),
        domain=str(row["domain"]),
        fps=fps,
        num_frames=num_frames,
        videos=videos,
        voc_frames=parsed.voc_frames,
        voc_mem_frames=parsed.voc_mem_frames,
        subtask_frames=parsed.subtask_frames,
        subtask_spans=parsed.subtask_spans,
        fpl_interval=parsed.fpl_interval,
        trr_group_id=(
            str(row["trr"]["group_id"])
            if validate_trr_metadata and row.get("trr") is not None
            else None
        ),
        trr_role=trr_role,
        trr_spans=parsed.trr_spans,
        cspc_solution_type=(
            str(row["cspc"]["solution_type"])
            if validate_cspc_metadata and row.get("cspc") is not None
            else None
        ),
    )


@dataclass(frozen=True)
class Dataset:
    root: Path
    tasks: Mapping[str, Task]

    @classmethod
    def load(
        cls,
        root: str | Path,
        metrics: Sequence[str] | None = None,
    ) -> "Dataset":
        """Load a dataset, optionally validating only selected metric inputs.

        ``metrics=None`` keeps the strict, whole-dataset validation used by the
        standalone ``vmbmk validate`` command and by callers that need a schema
        audit.  Evaluation passes its configured metric names so annotations and
        metadata belonging exclusively to other metrics cannot make a run fail.
        """
        requested_metrics = (
            None
            if metrics is None
            else frozenset(
                "cspc" if str(metric).lower() == "csvc"
                else str(metric).lower().replace("-", "_")
                for metric in metrics
            )
        )
        if requested_metrics is not None and {"voc", "vs"} & requested_metrics:
            requested_metrics = requested_metrics | {"cycle_voc"}
        source = Path(root).resolve()
        if not source.is_dir():
            raise DataValidationError(f"dataset directory does not exist: {source}")
        tasks: dict[str, Task] = {}
        for task_root in sorted(path for path in source.iterdir() if path.is_dir()):
            metadata_path = task_root / "metadata.json"
            if not metadata_path.is_file():
                continue
            task_id = task_root.name
            task_metadata = _read_json(metadata_path)
            instruction, subtask_ids, metrics = _validate_task_metadata(
                task_metadata,
                task_id,
                str(metadata_path),
                requested_metrics,
            )
            subtasks = (
                {
                    str(item["id"]): str(item["description"])
                    for item in task_metadata.get("subtasks", [])
                }
                if requested_metrics is None
                or bool({"sia", "cspc"} & requested_metrics)
                else {}
            )
            counterfactual_instructions = (
                {
                    str(item["id"]): str(item["description"])
                    for item in task_metadata.get("tga", {}).get(
                        "counterfactual_instructions", []
                    )
                }
                if requested_metrics is None
                or bool({"tga_easy", "tga_hard"} & requested_metrics)
                else {}
            )
            episodes_root = task_root / "episodes"
            if not episodes_root.is_dir():
                raise DataValidationError(f"missing directory: {episodes_root}")
            episodes: dict[str, Episode] = {}
            for episode_root in sorted(
                path for path in episodes_root.iterdir() if path.is_dir()
            ):
                episodes[episode_root.name] = _load_episode(
                    task_id, episode_root, subtask_ids, requested_metrics,
                )
            if not episodes:
                raise DataValidationError(f"task {task_id!r} has no episodes")
            tasks[task_id] = Task(
                task_id=task_id,
                instruction=instruction,
                episodes=episodes,
                metrics=metrics,
                subtasks=subtasks,
                counterfactual_instructions=counterfactual_instructions,
            )
        if not tasks:
            raise DataValidationError(f"dataset has no task directories: {source}")
        return cls(source, tasks)

    def episode(self, task_id: str, episode_id: str) -> Episode:
        try:
            return self.tasks[task_id].episodes[episode_id]
        except KeyError as exc:
            raise DataValidationError(
                f"unknown episode {task_id!r}/{episode_id!r}"
            ) from exc

    def reference_episode(self, task_id: str) -> Episode:
        try:
            task = self.tasks[task_id]
        except KeyError as exc:
            raise DataValidationError(f"unknown task {task_id!r}") from exc
        eligible = sorted(
            (episode for episode in task.episodes.values() if episode.success),
            key=lambda episode: episode.episode_id,
        )
        if not eligible:
            raise DataValidationError(f"task {task_id!r} has no successful reference")
        return eligible[0]
