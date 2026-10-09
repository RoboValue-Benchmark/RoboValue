from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.data.playback import PlaybackView
from vmbmk.inference.queries import CompareQuery, StateRef, ValueQuery
from .base import Adapter, positive_int, require_config
from .video_inputs import query_view, read_frame, task_reference_clip

if TYPE_CHECKING:
    from PIL import Image


_OFFICIAL_SAMPLE_FPS = 5.0
_OFFICIAL_SKIP = 5


@dataclass(frozen=True)
class _ReferenceClip:
    identifier: str
    video: Path
    start_frame: int
    num_frames: int


def _robodojo_reference(
    root: Path, instruction: str, view: str
) -> _ReferenceClip:
    info_path = root / "meta" / "info.json"
    episodes_root = root / "meta" / "episodes"
    if not info_path.is_file() or not episodes_root.is_dir():
        raise ConfigurationError(
            f"VLAC reference_data is not a LeRobot v3 dataset: {root}"
        )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        fps = float(info["fps"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid RoboDojo metadata: {info_path}") from exc
    if fps <= 0:
        raise ConfigurationError(f"invalid RoboDojo fps in {info_path}")
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise ConfigurationError(
            "VLAC one-shot references require pyarrow in the model environment"
        ) from exc

    prefix = f"videos/{view}"
    columns = [
        "episode_index",
        "tasks",
        "length",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
    ]
    selected: dict[str, Any] | None = None
    for metadata_path in sorted(episodes_root.rglob("*.parquet")):
        try:
            rows = parquet.read_table(metadata_path, columns=columns).to_pylist()
        except (KeyError, OSError) as exc:
            raise ConfigurationError(
                f"cannot read RoboDojo episode metadata: {metadata_path}"
            ) from exc
        for row in rows:
            if instruction not in row["tasks"]:
                continue
            if selected is None or row["episode_index"] < selected["episode_index"]:
                selected = row
    if selected is None:
        raise ConfigurationError(
            f"RoboDojo reference_data has no episode for task: {instruction}"
        )

    chunk_index = int(selected[f"{prefix}/chunk_index"])
    file_index = int(selected[f"{prefix}/file_index"])
    video = root / prefix / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    if not video.is_file():
        raise ConfigurationError(f"missing RoboDojo reference video: {video}")
    num_frames = int(selected["length"])
    if num_frames < 2:
        raise ConfigurationError(
            f"RoboDojo reference episode {selected['episode_index']} is too short"
        )
    return _ReferenceClip(
        identifier=f"robodojo:{selected['episode_index']}",
        video=video,
        start_frame=round(float(selected[f"{prefix}/from_timestamp"]) * fps),
        num_frames=num_frames,
    )


def _official_sample_step(source_fps: float) -> int:
    """Match VLAC's official compress_video sampling interval."""
    output_fps = min(source_fps, _OFFICIAL_SAMPLE_FPS)
    return int(source_fps / output_fps) if output_fps < source_fps else 1


def _critic_to_value_simple(deltas: Sequence[float]) -> float:
    """Apply VLAC's official simple=True accumulation in normalized units."""
    value = 0.0
    for delta in deltas:
        value = value + (1.0 - value) * delta
    return value


class VLACAdapter(Adapter):
    default_shot_mode = "one_shot"
    default_reference_view = "observation.images.cam_high"
    default_ref_num = 6

    @classmethod
    def resolve_shot_mode(cls, config: Mapping[str, Any]) -> str:
        """Use references only when configured, unless the mode is explicit."""
        return str(
            config.get(
                "shot_mode",
                "one_shot" if config.get("reference_data") is not None else "zero_shot",
            )
        )

    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str,
        *,
        shot_mode: str,
        reference_data: Path | None,
        reference_view: str,
        ref_num: int,
        reference_instruction_overrides: Mapping[str, str],
        batch_size: int,
    ) -> None:
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.shot_mode = shot_mode
        self.reference_data = reference_data
        self.reference_view = reference_view
        self.ref_num = ref_num
        self.reference_instruction_overrides = dict(reference_instruction_overrides)
        self.batch_size = batch_size
        self._model = None
        self._references: dict[str, tuple[str, list[Any]]] = {}
        self._value_cache: dict[tuple[Any, ...], float] = {}
        self._pair_cache: dict[tuple[Any, ...], float] = {}
        self._compare_cache: dict[tuple[Any, ...], float] = {}
        self._prepared_images: OrderedDict[int, tuple[Any, Any]] = OrderedDict()

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "VLACAdapter":
        require_config(
            config,
            "vlac",
            {
                "shot_mode",
                "reference_data",
                "reference_view",
                "ref_num",
                "reference_instruction_overrides",
                "batch_size",
            },
        )
        checkpoint = str(config["checkpoint"])
        if not checkpoint:
            raise ConfigurationError("vlac.checkpoint must be non-empty")
        shot_mode = cls.resolve_shot_mode(config)
        if shot_mode not in {"zero_shot", "one_shot"}:
            raise ConfigurationError(
                "vlac.shot_mode must be 'zero_shot' or 'one_shot'"
            )
        raw_reference_data = config.get("reference_data")
        if shot_mode == "one_shot":
            if raw_reference_data is None:
                reference_data = (dataset.root / "reference").resolve()
            elif not isinstance(raw_reference_data, str) or not raw_reference_data:
                raise ConfigurationError(
                    "vlac.reference_data must be a non-empty path"
                )
            else:
                reference_data = Path(raw_reference_data).resolve()
            if not reference_data.is_dir():
                raise ConfigurationError(
                    f"vlac.reference_data does not exist: {reference_data}"
                )
        else:
            if raw_reference_data is not None:
                raise ConfigurationError(
                    "vlac.reference_data is only valid for one_shot"
                )
            reference_data = None
        reference_view = str(
            config.get(
                "reference_view",
                cls.default_reference_view if shot_mode == "one_shot" else "front",
            )
        )
        if not reference_view:
            raise ConfigurationError("vlac.reference_view must be non-empty")
        ref_num = positive_int(
            config.get("ref_num", cls.default_ref_num), "vlac.ref_num"
        )
        if ref_num < 2:
            raise ConfigurationError("vlac.ref_num must be at least 2")
        raw_overrides = config.get("reference_instruction_overrides", {})
        if not isinstance(raw_overrides, Mapping):
            raise ConfigurationError("vlac.reference_instruction_overrides must be a mapping")
        reference_instruction_overrides: dict[str, str] = {}
        for task_id, instruction in raw_overrides.items():
            if task_id not in dataset.tasks:
                raise ConfigurationError(
                    f"vlac.reference_instruction_overrides has unknown task: {task_id}"
                )
            if not isinstance(instruction, str) or not instruction:
                raise ConfigurationError(
                    "vlac.reference_instruction_overrides values must be non-empty strings"
                )
            reference_instruction_overrides[task_id] = instruction
        return cls(
            dataset,
            checkpoint,
            shot_mode=shot_mode,
            reference_data=reference_data,
            reference_view=reference_view,
            ref_num=ref_num,
            reference_instruction_overrides=reference_instruction_overrides,
            batch_size=positive_int(
                config.get("batch_size", 5), "vlac.batch_size"
            ),
        )

    def _load(self) -> Any:
        if self._model is None:
            from evo_vlac import GAC_model

            self._model = GAC_model(tag="critic")
            self._model.init_model(
                model_path=self.checkpoint,
                model_type="internvl2",
                device_map="cuda:0",
            )
            self._model.temperature = 0.5
            self._model.top_k = 1
            self._model.set_config()
            self._model.set_system_prompt()
        return self._model

    def _episode(self, state: StateRef) -> Episode:
        episode = self.dataset.episode(state.task_id, state.episode_id)
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"{state.task_id}/{state.episode_id} frame {state.anchor_frame} "
                f"is outside [0, {episode.num_frames})"
            )
        episode.video("front")
        return episode

    @lru_cache(maxsize=4096)
    def _frame(
        self, task_id: str, episode_id: str, frame_index: int
    ) -> Any:
        episode = self.dataset.episode(task_id, episode_id)
        return read_frame(episode.video("front"), frame_index)

    @lru_cache(maxsize=4096)
    def _external_frame(self, source: str, frame_index: int) -> Any:
        return read_frame(source, frame_index)

    def _reference(self, task_id: str) -> tuple[str, list[Any]]:
        if task_id not in self._references:
            if self.reference_data is None:
                raise ConfigurationError(
                    "VLAC one-shot reference_data is not configured"
                )
            if (
                (self.reference_data / "meta" / "info.json").is_file()
                and (self.reference_data / "meta" / "episodes").is_dir()
            ):
                clip = _robodojo_reference(
                    self.reference_data,
                    self.reference_instruction_overrides.get(
                        task_id, self.dataset.tasks[task_id].instruction
                    ),
                    self.reference_view,
                )
            else:
                video, num_frames = task_reference_clip(
                    self.reference_data, task_id, self.reference_view
                )
                clip = _ReferenceClip(
                    identifier=f"reference:{video}",
                    video=video,
                    start_frame=0,
                    num_frames=num_frames,
                )
            indices = tuple(
                round((clip.num_frames - 1) * index / (self.ref_num - 1))
                for index in range(self.ref_num)
            )
            frames = [
                self._external_frame(str(clip.video), clip.start_frame + frame)
                for frame in indices
            ]
            self._references[task_id] = (clip.identifier, frames)
        return self._references[task_id]

    def _native_trajectory(
        self,
        instruction: str,
        images: Sequence[Any],
        reference_images: Sequence[Any],
        *,
        skip: int,
    ) -> list[float]:
        requests = [
            (
                instruction,
                images[0],
                images[index - skip],
                images[index],
                reference_images,
            )
            for index in range(skip, len(images), skip)
        ]
        return self._native_pairs(requests)

    def _native_pairs(
        self,
        requests: Sequence[tuple[str, Any, Any, Any, Sequence[Any]]],
    ) -> list[float]:
        """Use official pair construction and preserve trajectory batch boundaries."""
        if not requests:
            return []
        model = self._load()
        outputs: list[float] = []
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start : start + self.batch_size]
            prompts = []
            image_batches = []
            for instruction, trajectory_start, before, after, references in chunk:
                prompts.append(
                    model.get_score_prompt(
                        task=instruction,
                        trajectory_len=(self.ref_num if references else 0),
                        think=False,
                    )
                )
                if references:
                    image_batches.append(
                        list(references) + [trajectory_start, before, after]
                    )
                else:
                    image_batches.append([before, after])
            # Cache the official RGB/resize operation only. Each request still
            # receives its own image copy and the original prompt and batching.
            original_process = model._process_image_to_pil
            def prepare(image: Image.Image) -> Any:
                key = id(image)
                cached = self._prepared_images.get(key)
                if cached is None or cached[0] is not image:
                    cached = (image, original_process(image))
                    self._prepared_images[key] = cached
                self._prepared_images.move_to_end(key)
                while len(self._prepared_images) > 64:
                    self._prepared_images.popitem(last=False)
                return cached[1].copy()
            model._process_image_to_pil = prepare
            try:
                infer_requests = model.get_infer_requests(
                    prompt=prompts,
                    images=image_batches,
                )
            finally:
                model._process_image_to_pil = original_process
            response_list, _infer_time = model.chat(infer_requests)
            answers, _complete = model.results_format(
                response_list,
                infer_requests,
                rich=True,
            )
            if len(answers) != len(chunk):
                raise VMBMKError("VLAC returned the wrong critic batch size")
            parsed = [float(answer) / 100.0 for answer in answers]
            if any(not math.isfinite(value) for value in parsed):
                raise VMBMKError("VLAC returned a non-finite prediction")
            outputs.extend(parsed)
        return outputs

    @staticmethod
    def _base_key(
        query: ValueQuery, reference_id: str
    ) -> tuple[Any, ...]:
        return (
            query.state.task_id,
            query.state.episode_id,
            query.instruction,
            query.playback,
            reference_id,
        )

    @staticmethod
    def _grid_pair_key(
        base_key: tuple[Any, ...], before: int, after: int
    ) -> tuple[Any, ...]:
        return ("value", *base_key, before, after)

    def _view_frame(
        self, episode: Episode, view: PlaybackView, timeline_index: int
    ) -> Any:
        return self._frame(
            episode.task_id,
            episode.episode_id,
            view.source_index(timeline_index),
        )

    def _ensure_grid(
        self,
        base_key: tuple[Any, ...],
        episode: Episode,
        view: PlaybackView,
        timeline_anchor: int,
        reference_images: Sequence[Any],
    ) -> tuple[int, int]:
        sample_step = _official_sample_step(view.fps)
        grid_stride = sample_step * _OFFICIAL_SKIP
        grid_end = (timeline_anchor // grid_stride) * grid_stride
        pair_count = grid_end // grid_stride
        first_missing = pair_count
        for index in range(pair_count):
            before = index * grid_stride
            after = before + grid_stride
            if self._grid_pair_key(base_key, before, after) not in self._pair_cache:
                first_missing = index
                break
        if first_missing == pair_count:
            return grid_stride, grid_end

        if first_missing == 0:
            timeline_indices = list(range(0, grid_end + 1, sample_step))
            discard = 0
        else:
            cached_end = first_missing * grid_stride
            if reference_images:
                # Keep image_list[0] as the episode start, as official one-shot
                # trajectory inference does, while extending only the suffix.
                timeline_indices = [0] * _OFFICIAL_SKIP + [cached_end]
                timeline_indices.extend(
                    range(cached_end + sample_step, grid_end + 1, sample_step)
                )
                discard = 1
            else:
                timeline_indices = list(range(cached_end, grid_end + 1, sample_step))
                discard = 0
        # The critic only consumes every skip-th image. Preserve the original
        # timeline positions and batch boundaries without decoding unused frames.
        images = [
            self._view_frame(episode, view, index)
            if position % _OFFICIAL_SKIP == 0 else None
            for position, index in enumerate(timeline_indices)
        ]
        deltas = self._native_trajectory(
            base_key[2],
            images,
            reference_images,
            skip=_OFFICIAL_SKIP,
        )[discard:]
        expected_new = pair_count - first_missing
        if len(deltas) != expected_new:
            raise VMBMKError("VLAC returned the wrong grid extension shape")
        for offset, delta in enumerate(deltas, start=first_missing):
            before = offset * grid_stride
            after = before + grid_stride
            self._pair_cache[self._grid_pair_key(base_key, before, after)] = delta
        return grid_stride, grid_end

    def _terminal_delta(
        self,
        base_key: tuple[Any, ...],
        episode: Episode,
        view: PlaybackView,
        before: int,
        after: int,
        reference_images: Sequence[Any],
    ) -> float:
        key = ("terminal", *base_key, before, after)
        if key not in self._pair_cache:
            if reference_images and before != 0:
                timeline_indices = [0, before, after]
            else:
                timeline_indices = [before, after]
            images = [
                self._view_frame(episode, view, index)
                for index in timeline_indices
            ]
            deltas = self._native_trajectory(
                base_key[2], images, reference_images, skip=1
            )
            self._pair_cache[key] = deltas[-1]
        return self._pair_cache[key]

    def _value_at(
        self,
        base_key: tuple[Any, ...],
        episode: Episode,
        view: PlaybackView,
        timeline_anchor: int,
        reference_images: Sequence[Any],
    ) -> float:
        grid_stride, grid_end = self._ensure_grid(
            base_key,
            episode,
            view,
            timeline_anchor,
            reference_images,
        )
        deltas = [
            self._pair_cache[
                self._grid_pair_key(base_key, before, before + grid_stride)
            ]
            for before in range(0, grid_end, grid_stride)
        ]
        if grid_end != timeline_anchor:
            deltas.append(
                self._terminal_delta(
                    base_key,
                    episode,
                    view,
                    grid_end,
                    timeline_anchor,
                    reference_images,
                )
            )
        return _critic_to_value_simple(deltas)

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        if not queries:
            return []
        owned: list[tuple[Any, ...]] = []
        pending: dict[
            tuple[Any, ...], tuple[tuple[Any, ...], Episode, PlaybackView, int, list[Any]]
        ] = {}
        for query in queries:
            episode = self._episode(query.state)
            view = query_view(self.dataset, query, "front")
            timeline_anchor = view.timeline_anchor(query.state.anchor_frame)
            if self.shot_mode == "one_shot":
                reference_id, reference_images = self._reference(query.state.task_id)
            else:
                reference_id, reference_images = "none", []
            base_key = self._base_key(query, reference_id)
            key = (*base_key, timeline_anchor)
            if key not in self._value_cache:
                pending[key] = (
                    base_key,
                    episode,
                    view,
                    timeline_anchor,
                    reference_images,
                )
            owned.append(key)

        # Shorter prefixes first let longer prefixes extend cached deltas without
        # changing a query's result based on query order or query-set membership.
        for key, item in sorted(
            pending.items(), key=lambda entry: (entry[1][0], entry[1][3])
        ):
            base_key, episode, view, timeline_anchor, reference_images = item
            self._value_cache[key] = self._value_at(
                base_key,
                episode,
                view,
                timeline_anchor,
                reference_images,
            )
        return [self._value_cache[key] for key in owned]

    def _compare_key(
        self, query: CompareQuery, reference_id: str
    ) -> tuple[Any, ...]:
        return (
            query.state_a.task_id,
            query.state_a.episode_id,
            query.state_a.anchor_frame,
            query.state_b.episode_id,
            query.state_b.anchor_frame,
            query.instruction,
            reference_id,
        )

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        if not queries:
            return []
        owned: list[tuple[Any, ...]] = []
        for query in queries:
            if query.state_a.task_id != query.state_b.task_id:
                raise VMBMKError(
                    f"{query.query_id}: compare states must belong to the same task"
                )
            before = self._episode(query.state_a)
            after = self._episode(query.state_b)
            if self.shot_mode == "one_shot":
                reference_id, reference_images = self._reference(
                    query.state_a.task_id
                )
            else:
                reference_id, reference_images = "none", []
            key = self._compare_key(query, reference_id)
            if query.state_a == query.state_b:
                self._compare_cache[key] = 0.0
            elif key not in self._compare_cache:
                images = [
                    self._frame(
                        before.task_id,
                        before.episode_id,
                        query.state_a.anchor_frame,
                    ),
                    self._frame(
                        after.task_id,
                        after.episode_id,
                        query.state_b.anchor_frame,
                    ),
                ]
                self._compare_cache[key] = self._native_trajectory(
                    query.instruction,
                    images,
                    reference_images,
                    skip=1,
                )[0]
            owned.append(key)
        return [self._compare_cache[key] for key in owned]
