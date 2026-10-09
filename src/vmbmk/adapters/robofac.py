from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache
from typing import Any, Iterator, Mapping, Sequence, TYPE_CHECKING

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.data.playback import PlaybackView
from vmbmk.inference.queries import CompareQuery, StateRef, SubtaskQuery, ValueQuery
from .base import Adapter, compare_value_difference, positive_int, positive_number, require_config
from .video_inputs import query_view, read_frame

if TYPE_CHECKING:
    from PIL import Image


_LABELS = ("A", "B", "C", "D", "E")
_IMAGE_SIZE = 512
_ROBOFAC_FRAME_CACHE_SIZE = 4096
# Runtime scheduling guard only. It never changes which frames are scored.
_ROBOFAC_BATCH_FRAME_BUDGET = 64


_PROMPT = """Evaluate the robot's current still-valid progress toward the task at the end of the shown trajectory prefix.

Task: {instruction}

Judge only task-relevant progress that is visibly established and still valid in the final shown state. A past mistake that has already been recovered does not by itself force a low score. Mere robot or object motion, an attempt, or activity without lasting task-relevant progress does not count as progress. If a mistake is currently visible, retain credit for prerequisite progress that remains valid, but do not credit invalidated or undone progress.

Choose exactly one level:
A - No valid task-relevant progress is currently established.
B - A small amount of valid progress is established, but most requirements remain.
C - Meaningful partial completion is established, with major requirements remaining.
D - The task is nearly complete; most requirements are valid and only a minor or final requirement remains.
E - All task requirements are visibly satisfied in the current state.

Answer with exactly one letter: A, B, C, D, or E."""


def _round_fraction(value: Fraction) -> int:
    return (2 * value.numerator + value.denominator) // (
        2 * value.denominator
    )


def _uniform_limit(indices: tuple[int, ...], max_frames: int) -> tuple[int, ...]:
    if len(indices) <= max_frames:
        return indices
    if max_frames < 2:
        raise VMBMKError(
            "RoboFAC max_frames must be at least 2 when a prefix has "
            "more than one frame"
        )
    last = len(indices) - 1
    denominator = max_frames - 1
    positions = tuple(
        _round_fraction(Fraction(position * last, denominator))
        for position in range(max_frames)
    )
    result = tuple(indices[position] for position in positions)
    if result[0] != indices[0] or result[-1] != indices[-1]:
        raise VMBMKError("RoboFAC max_frames sampling lost a prefix endpoint")
    return result


def _sample_indices(
    anchor: int,
    source_fps: float,
    sample_fps: float,
    max_frames: int | None,
) -> tuple[int, ...]:
    if anchor < 0:
        raise VMBMKError("RoboFAC anchor must be non-negative")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise VMBMKError("RoboFAC source fps must be a positive finite number")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise VMBMKError("RoboFAC sample_fps must be a positive finite number")
    if max_frames is not None and max_frames <= 0:
        raise VMBMKError("RoboFAC max_frames must be positive")

    ratio = Fraction(str(source_fps)) / Fraction(str(sample_fps))
    frames: list[int] = []
    sample_number = 0
    while True:
        frame = _round_fraction(sample_number * ratio)
        if frame > anchor:
            break
        if not frames or frame != frames[-1]:
            frames.append(frame)
        sample_number += 1
    if not frames or frames[0] != 0:
        frames.insert(0, 0)
    if frames[-1] != anchor:
        frames.append(anchor)

    indices = tuple(frames)
    if max_frames is not None:
        indices = _uniform_limit(indices, max_frames)
    if (
        not indices
        or indices[0] != 0
        or indices[-1] != anchor
        or len(indices) != len(set(indices))
        or any(index < 0 or index > anchor for index in indices)
    ):
        raise VMBMKError(
            f"RoboFAC sampled invalid prefix for anchor frame {anchor}: {indices}"
        )
    return indices


def _progress_from_candidate_logits(logits: Sequence[float]) -> float:
    if len(logits) != len(_LABELS):
        raise VMBMKError(
            f"RoboFAC expected {len(_LABELS)} candidate logits; got {len(logits)}"
        )
    values = [float(value) for value in logits]
    if any(not math.isfinite(value) for value in values):
        raise VMBMKError("RoboFAC candidate logits must all be finite")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    return sum(index * probability for index, probability in enumerate(probabilities)) / 4.0


class RoboFACAdapter(Adapter):
    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str,
        *,
        view: str,
        sample_fps: float,
        max_frames: int | None,
        batch_size: int,
    ) -> None:
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.view = view
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._cache: dict[tuple[Any, ...], float] = {}

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "RoboFACAdapter":
        require_config(
            config,
            "robofac",
            {"view", "sample_fps", "max_frames", "batch_size"},
        )
        checkpoint = str(config["checkpoint"])
        if not checkpoint:
            raise ConfigurationError("robofac.checkpoint must be non-empty")
        view = config.get("view", "front")
        if not isinstance(view, str) or not view:
            raise ConfigurationError("robofac.view must be a non-empty string")
        max_frames = config.get("max_frames")
        if max_frames is not None:
            max_frames = positive_int(max_frames, "robofac.max_frames")
        return cls(
            dataset,
            checkpoint,
            view=view,
            sample_fps=positive_number(
                config.get("sample_fps", 1.0), "robofac.sample_fps"
            ),
            max_frames=max_frames,
            batch_size=positive_int(
                config.get("batch_size", 1), "robofac.batch_size"
            ),
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._torch = torch
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            local_files_only=True,
            use_fast=False,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        self._model.eval()

    def _episode(self, state: StateRef) -> Episode:
        episode = self.dataset.episode(state.task_id, state.episode_id)
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"RoboFAC {state.task_id}/{state.episode_id} anchor "
                f"{state.anchor_frame} is outside [0, {episode.num_frames})"
            )
        episode.video(self.view)
        return episode

    @lru_cache(maxsize=_ROBOFAC_FRAME_CACHE_SIZE)
    def _frame(
        self,
        task_id: str,
        episode_id: str,
        view: str,
        frame_index: int,
    ) -> Image.Image:
        episode = self.dataset.episode(task_id, episode_id)
        frame = read_frame(episode.video(view), frame_index)
        return frame.resize((_IMAGE_SIZE, _IMAGE_SIZE))

    def _key(self, query: ValueQuery) -> tuple[Any, ...]:
        return (
            query.state.task_id,
            query.state.episode_id,
            query.state.anchor_frame,
            query.instruction,
            query.playback,
            self.view,
            self.sample_fps,
            self.max_frames,
            self.checkpoint,
        )

    @lru_cache(maxsize=_ROBOFAC_FRAME_CACHE_SIZE)
    def _prompt(self, instruction: str, frame_count: int) -> str:
        """Return the processor template for a structural image prefix.

        Chat templating only consumes the content types and their order; it
        does not inspect image pixels. The actual images remain in
        ``batch_messages`` for vision preprocessing below.
        """
        if frame_count <= 0:
            raise VMBMKError("RoboFAC prompt requires at least one frame")
        messages = [{
            "role": "user",
            "content": [
                *({"type": "image", "image": None} for _ in range(frame_count)),
                {"type": "text", "text": _PROMPT.format(instruction=instruction)},
            ],
        }]
        return self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @lru_cache(maxsize=_ROBOFAC_FRAME_CACHE_SIZE)
    def _candidate_ids(self, prompt: str) -> tuple[int, ...]:
        tokenizer = self._processor.tokenizer
        base_ids = tokenizer(
            prompt, add_special_tokens=False
        ).input_ids
        candidate_ids = []
        for label in _LABELS:
            combined = tokenizer(
                f"{prompt}{label}", add_special_tokens=False
            ).input_ids
            if (
                combined[: len(base_ids)] != base_ids
                or len(combined) != len(base_ids) + 1
            ):
                raise VMBMKError(
                    f"RoboFAC label {label!r} is not one next token at the "
                    "actual Qwen assistant generation position"
                )
            candidate_ids.append(combined[-1])
        if len(candidate_ids) != len(set(candidate_ids)):
            raise VMBMKError("RoboFAC A-E labels do not have distinct token IDs")
        return tuple(candidate_ids)

    def _score_batch(
        self,
        items: Sequence[
            tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]]
        ],
    ) -> list[float]:
        if not items:
            return []
        self._load()
        batch_messages = []
        prompts = []
        candidate_id_rows = []
        for query, episode, view, indices in items:
            frames = [
                self._frame(
                    episode.task_id,
                    episode.episode_id,
                    self.view,
                    view.source_index(frame_index),
                )
                for frame_index in indices
            ]
            messages = [{
                "role": "user",
                "content": [
                    *({"type": "image", "image": frame} for frame in frames),
                    {
                        "type": "text",
                        "text": _PROMPT.format(instruction=query.instruction),
                    },
                ],
            }]
            prompt = self._prompt(query.instruction, len(frames))
            batch_messages.append(messages)
            prompts.append(prompt)
            candidate_id_rows.append(self._candidate_ids(prompt))

        image_inputs, video_inputs = self._process_vision_info(batch_messages)
        inputs = self._processor(
            text=prompts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(inputs["input_ids"])
        positions = self._torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).expand_as(attention_mask)
        last_positions = positions.masked_fill(attention_mask == 0, -1).max(
            dim=1
        ).values
        if (last_positions < 0).any():
            raise VMBMKError("RoboFAC processor returned an empty prompt")
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        rows = self._torch.arange(len(items), device=outputs.logits.device)
        next_token_logits = outputs.logits[rows, last_positions]
        candidate_index = self._torch.tensor(
            candidate_id_rows,
            dtype=self._torch.long,
            device=next_token_logits.device,
        )
        candidate_logits = next_token_logits.gather(1, candidate_index)
        scores = []
        for (query, episode, _view, _indices), logits in zip(
            items, candidate_logits, strict=True
        ):
            score = _progress_from_candidate_logits(
                logits.float().detach().cpu().tolist()
            )
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise VMBMKError(
                    f"RoboFAC returned an invalid progress score for "
                    f"{episode.task_id}/{episode.episode_id} anchor "
                    f"{query.state.anchor_frame}: {score}"
                )
            scores.append(score)
        return scores

    def _microbatches(
        self,
        items: Sequence[
            tuple[
                tuple[Any, ...],
                tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]],
            ]
        ],
    ) -> Iterator[list[Any]]:
        """Batch only shape-compatible prefixes within the frame budget."""
        groups: dict[
            tuple[int, str],
            list[
                tuple[
                    tuple[Any, ...],
                    tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]],
                ]
            ],
        ] = {}
        for item in items:
            _key, (query, _episode, _view, indices) = item
            # Each image has the same fixed resolution. Equal frame counts
            # plus the same instruction produce equal prompt shapes, avoiding
            # padding-dependent numerical drift.
            shape = (len(indices), query.instruction)
            groups.setdefault(shape, []).append(item)
        for group in groups.values():
            batch = []
            frame_count = 0
            for item in group:
                item_frames = len(item[1][3])
                if batch and (
                    len(batch) >= self.batch_size
                    or frame_count + item_frames > _ROBOFAC_BATCH_FRAME_BUDGET
                ):
                    yield batch
                    batch = []
                    frame_count = 0
                batch.append(item)
                frame_count += item_frames
            if batch:
                yield batch

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        if not queries:
            return []
        pending: dict[
            tuple[Any, ...], tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]]
        ] = {}
        owned: list[tuple[Any, ...]] = []
        for query in queries:
            episode = self._episode(query.state)
            view = query_view(self.dataset, query, self.view)
            indices = _sample_indices(
                view.timeline_anchor(query.state.anchor_frame),
                episode.fps,
                self.sample_fps,
                self.max_frames,
            )
            key = self._key(query)
            owned.append(key)
            if key not in self._cache:
                pending.setdefault(key, (query, episode, view, indices))

        items = list(pending.items())
        for batch in self._microbatches(items):
            scores = self._score_batch([item for _key, item in batch])
            for (key, _item), score in zip(batch, scores, strict=True):
                self._cache[key] = score
        return [self._cache[key] for key in owned]

    def subtask(self, queries: Sequence[SubtaskQuery]) -> list[str]:
        if not queries:
            return []
        import json

        self._load()
        responses = [""] * len(queries)
        groups: dict[
            tuple[str, int], list[tuple[int, SubtaskQuery, tuple[int, ...]]]
        ] = {}
        for index, query in enumerate(queries):
            episode = self._episode(query.state)
            indices = _sample_indices(
                query.state.anchor_frame, episode.fps,
                self.sample_fps, self.max_frames,
            )
            groups.setdefault((query.instruction, len(indices)), []).append(
                (index, query, indices)
            )
        # Match value() batching by prompt shape and total frame budget.
        for (instruction, frame_count), group in groups.items():
            batch_size = min(
                self.batch_size, max(1, _ROBOFAC_BATCH_FRAME_BUDGET // frame_count)
            )
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                messages = []
                for _, query, indices in batch:
                    state = query.state
                    frames = [
                        self._frame(
                            state.task_id, state.episode_id, self.view, frame_index
                        )
                        for frame_index in indices
                    ]
                    messages.append([{
                        "role": "user",
                        "content": [
                            *({"type": "image", "image": frame} for frame in frames),
                            {
                                "type": "text",
                                "text": (
                                    f"Task: {instruction}\n"
                                    "Describe the subtask the robot is currently "
                                    "performing in the image."
                                ),
                            },
                        ],
                    }])
                prompts = [self._processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                ) for message in messages]
                images, videos = self._process_vision_info(messages)
                inputs = self._processor(
                    text=prompts, images=images, videos=videos,
                    padding=True, return_tensors="pt",
                ).to(self._model.device)
                with self._torch.inference_mode():
                    output_ids = self._model.generate(
                        **inputs, do_sample=False, max_new_tokens=256,
                    )
                texts = self._processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                if len(texts) != len(batch):
                    raise VMBMKError("RoboFAC subtask response count mismatch")
                for (index, query, _indices), text in zip(batch, texts, strict=True):
                    # Preserve IDs, free-form descriptions, and explanations for
                    # the semantic judge instead of imposing a response grammar.
                    response = text.strip()
                    print("[RoboFAC subtask] " + json.dumps({
                        "query_id": query.query_id, "response": response,
                    }, ensure_ascii=False), flush=True)
                    if not response:
                        raise VMBMKError(
                            f"RoboFAC returned an empty subtask for {query.query_id}"
                        )
                    responses[index] = response
        return responses

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        if not queries:
            return []
        for query in queries:
            if query.state_a.task_id != query.state_b.task_id:
                raise VMBMKError(
                    f"{query.query_id}: RoboFAC compare states must belong "
                    "to the same task"
                )
        return compare_value_difference(queries, self.value)
