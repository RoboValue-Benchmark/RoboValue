from __future__ import annotations

import math
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.data.playback import PlaybackView
from vmbmk.inference.queries import CompareQuery, StateRef, ValueQuery
from .base import Adapter, compare_value_difference, positive_int, positive_number, require_config
from .video_inputs import query_view, read_frame, read_frames

if TYPE_CHECKING:
    from PIL import Image


_TOPREWARD_BATCH_SIZE = 2


_TOPREWARD_SHORT_PREFIX_BATCH_SIZE = 4
_TOPREWARD_SHORT_PREFIX_MAX_FRAMES = 64
_TOPREWARD_FRAME_CACHE_SIZE = 2048
_TOPREWARD_VISION_CACHE_BYTES = 4 * 1024 * 1024 * 1024

class _VideoMetadataProcessor:
    """Forward real video timing so Qwen performs its native FPS sampling."""

    def __init__(self, processor: Any, *, sample_fps: float) -> None:
        self._processor = processor
        self._sample_fps = sample_fps
        self._video_timings: list[tuple[float, int]] | None = None
        self._pre_sampled_indices: list[tuple[int, ...]] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    @staticmethod
    def _frame_count(video: Any) -> int:
        shape = getattr(video, "shape", None)
        if shape is not None and len(shape) >= 4:
            return int(shape[0])
        try:
            return len(video)
        except TypeError as exc:
            raise VMBMKError(
                "TOPReward processor received a video with no frame dimension"
            ) from exc

    def set_video_timing(self, *, source_fps: float, total_num_frames: int) -> None:
        self.set_video_timings([(source_fps, total_num_frames)])

    def set_video_timings(
        self,
        timings: Sequence[tuple[float, int]],
    ) -> None:
        checked: list[tuple[float, int]] = []
        for source_fps, total_num_frames in timings:
            if not math.isfinite(source_fps) or source_fps <= 0:
                raise VMBMKError(
                    "TOPReward source fps must be positive and finite"
                )
            if total_num_frames <= 0:
                raise VMBMKError(
                    "TOPReward prefix must contain at least one frame"
                )
            checked.append((source_fps, total_num_frames))
        if not checked:
            raise VMBMKError("TOPReward video timings must not be empty")
        self._video_timings = checked

        self._pre_sampled_indices = None

    def set_pre_sampled_video_timings(
        self,
        timings: Sequence[tuple[float, int, Sequence[int]]],
    ) -> None:
        checked: list[tuple[float, int]] = []
        sampled: list[tuple[int, ...]] = []
        for source_fps, total_num_frames, frame_indices in timings:
            if not math.isfinite(source_fps) or source_fps <= 0:
                raise VMBMKError(
                    "TOPReward source fps must be positive and finite"
                )
            if total_num_frames <= 0:
                raise VMBMKError(
                    "TOPReward prefix must contain at least one frame"
                )
            indices = tuple(int(index) for index in frame_indices)
            if not indices or any(
                index < 0 or index >= total_num_frames for index in indices
            ):
                raise VMBMKError(
                    "TOPReward sampled frame indices are outside the prefix"
                )
            checked.append((source_fps, total_num_frames))
            sampled.append(indices)
        if not checked:
            raise VMBMKError("TOPReward video timings must not be empty")
        self._video_timings = checked
        self._pre_sampled_indices = sampled

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        videos = kwargs.get("videos")
        if videos is not None:
            if self._video_timings is None:
                raise VMBMKError("TOPReward video timing was not configured")
            if isinstance(videos, (list, tuple)):
                video_items = list(videos)
            else:
                video_items = [videos]
            if len(video_items) != len(self._video_timings):
                raise VMBMKError(
                    "TOPReward processor received "
                    f"{len(video_items)} videos for "
                    f"{len(self._video_timings)} timing records"
                )
            metadata = []
            for index, (video, (source_fps, total_num_frames)) in enumerate(zip(
                video_items,
                self._video_timings,
            )):
                actual_frames = self._frame_count(video)
                if self._pre_sampled_indices is None:
                    if actual_frames not in {
                        total_num_frames,
                        total_num_frames + 1,
                    }:
                        raise VMBMKError(
                            "TOPReward processor received an unexpected number of "
                            f"frames: {actual_frames} for a "
                            f"{total_num_frames}-frame prefix"
                        )
                    frame_indices = list(range(total_num_frames))
                    frame_indices.extend(
                        [total_num_frames - 1]
                        * (actual_frames - total_num_frames)
                    )
                else:
                    frame_indices = list(self._pre_sampled_indices[index])
                    if actual_frames == len(frame_indices) + 1:
                        frame_indices.append(frame_indices[-1])
                    elif actual_frames != len(frame_indices):
                        raise VMBMKError(
                            "TOPReward processor received an unexpected number of "
                            "pre-sampled frames"
                        )
                metadata.append(
                    {
                        "fps": source_fps,
                        "frames_indices": frame_indices,
                        "total_num_frames": total_num_frames,
                    }
                )
            kwargs["video_metadata"] = metadata
            kwargs["do_sample_frames"] = self._pre_sampled_indices is None
            if self._pre_sampled_indices is None:
                kwargs["fps"] = self._sample_fps
        return self._processor(*args, **kwargs)


class TOPRewardAdapter(Adapter):
    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str,
        *,
        view: str,
        sample_fps: float,
        reduction: str,
        use_video_description: bool,
        add_chat_template: bool,
        batch_size: int,
        backend: str = "qwen",
    ) -> None:
        if backend == "qwen" and batch_size != _TOPREWARD_BATCH_SIZE:
            raise ConfigurationError(
                "TOPReward batch_size is fixed to "
                f"{_TOPREWARD_BATCH_SIZE}; got {batch_size}"
            )
        self.backend = backend
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.view = view
        self.sample_fps = sample_fps
        self.reduction = reduction
        self.use_video_description = use_video_description
        self.add_chat_template = add_chat_template
        self.batch_size = batch_size
        self._client = None
        self._video_processor: _VideoMetadataProcessor | None = None
        self._cache: dict[tuple[Any, ...], float] = {}
        self._vision_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._vision_cache_bytes = 0

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "TOPRewardAdapter":
        require_config(
            config,
            "topreward",
            {
                "view",
                "backend",
                "max_frames",
                "max_input_length",
                "sample_fps",
                "reduction",
                "use_video_description",
                "add_chat_template",
                "batch_size",
            },
        )
        backend = config.get("backend", "qwen")
        if backend not in {"qwen", "molmo"}:
            raise ConfigurationError("topreward.backend must be qwen or molmo")
        if backend == "qwen" and {"max_frames", "max_input_length"}.intersection(config):
            raise ConfigurationError("max_frames/max_input_length are Molmo-only")
        checkpoint = str(config["checkpoint"])
        if not checkpoint:
            raise ConfigurationError("topreward.checkpoint must be non-empty")
        view = config.get("view", "front")
        if not isinstance(view, str) or not view:
            raise ConfigurationError("topreward.view must be a non-empty string")
        reduction = config.get("reduction", "mean")
        if reduction not in {"mean", "sum"}:
            raise ConfigurationError("topreward.reduction must be 'mean' or 'sum'")
        for field in ("use_video_description", "add_chat_template"):
            if not isinstance(config.get(field, False), bool):
                raise ConfigurationError(f"topreward.{field} must be a boolean")
        adapter = cls(
            dataset,
            checkpoint,
            backend=backend,
            view=view,
            sample_fps=positive_number(
                config.get("sample_fps", 2.0), "topreward.sample_fps"
            ),
            reduction=str(reduction),
            use_video_description=config.get("use_video_description", False),
            add_chat_template=config.get("add_chat_template", False),
            batch_size=positive_int(
                config.get("batch_size", 1 if backend == "molmo" else _TOPREWARD_BATCH_SIZE),
                "topreward.batch_size",
            ),
        )

        if backend == "molmo":
            adapter.max_frames = positive_int(config.get("max_frames", 128), "topreward.max_frames")
            adapter.max_input_length = positive_int(config.get("max_input_length", 32768), "topreward.max_input_length")
        return adapter

    def _load(self) -> None:
        if self.backend == "molmo":
            return self._load_molmo()
        if self._client is not None:
            return
        from topreward.clients.qwen import QwenClient

        self._client = QwenClient(model_name=self.checkpoint)
        self._video_processor = _VideoMetadataProcessor(
            self._client.processor,
            sample_fps=self.sample_fps,
        )
        self._client.processor = self._video_processor

    def _episode(self, state: StateRef) -> Episode:
        episode = self.dataset.episode(state.task_id, state.episode_id)
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"TOPReward {state.task_id}/{state.episode_id} anchor "
                f"{state.anchor_frame} is outside [0, {episode.num_frames})"
            )
        episode.video(self.view)
        return episode

    @lru_cache(maxsize=_TOPREWARD_FRAME_CACHE_SIZE)
    def _frame(
        self,
        task_id: str,
        episode_id: str,
        view: str,
        frame_index: int,
    ) -> Image.Image:
        episode = self.dataset.episode(task_id, episode_id)
        return read_frame(episode.video(view), frame_index)

    @lru_cache(maxsize=_TOPREWARD_FRAME_CACHE_SIZE)
    def _pil_frame(
        self,
        task_id: str,
        episode_id: str,
        view: str,
        frame_index: int,
    ) -> Image.Image:
        from topreward.utils.images import to_pil

        return to_pil(self._frame(task_id, episode_id, view, frame_index))

    def _prefetch_frame_references(
        self,
        frame_references: Sequence[tuple[str, str, str, int]],
    ) -> None:
        """Populate the decoded-frame cache with one sequential read per video."""
        by_video: dict[object, list[int]] = {}
        for task_id, episode_id, view, frame_index in frame_references:
            source = self.dataset.episode(task_id, episode_id).video(view)
            by_video.setdefault(source, []).append(frame_index)
        for source, indices in by_video.items():
            read_frames(source, indices)

    def _vision_key(
        self,
        frame_references: Sequence[tuple[str, str, str, int]],
        fps: float,
        frame_count: int,
    ) -> tuple[Any, ...]:
        return (
            tuple(frame_references),
            fps,
            frame_count,
            self._source_video_max_pixels(frame_count),
        )

    def _cached_video_input(self, key: tuple[Any, ...]) -> Any:
        cached = self._vision_cache.pop(key, None)
        if cached is None:
            return None
        self._vision_cache[key] = cached
        return cached.clone()

    def _cache_video_input(self, key: tuple[Any, ...], video: Any) -> None:
        cached = video.detach().cpu().clone()
        size = cached.numel() * cached.element_size()
        if size > _TOPREWARD_VISION_CACHE_BYTES:
            return
        previous = self._vision_cache.pop(key, None)
        if previous is not None:
            self._vision_cache_bytes -= (
                previous.numel() * previous.element_size()
            )
        self._vision_cache[key] = cached
        self._vision_cache_bytes += size
        while self._vision_cache_bytes > _TOPREWARD_VISION_CACHE_BYTES:
            _old_key, old_video = self._vision_cache.popitem(last=False)
            self._vision_cache_bytes -= (
                old_video.numel() * old_video.element_size()
            )

    def _sample_frame_indices(
        self,
        total_num_frames: int,
        source_fps: float,
    ) -> tuple[int, ...]:
        import numpy as np

        if self._video_processor is None:
            if self.backend == "molmo":
                return self._sample_molmo_frames(total_num_frames, source_fps)
            raise VMBMKError("TOPReward batch processor was not initialized")
        video_processor = self._video_processor.video_processor
        count = int(total_num_frames / source_fps * self.sample_fps)
        count = min(
            max(count, int(video_processor.min_frames)),
            int(video_processor.max_frames),
            total_num_frames,
        )
        return tuple(
            np.linspace(0, total_num_frames - 1, count).round().astype(int)
        )

    @staticmethod
    def _source_video_max_pixels(total_num_frames: int) -> int:
        from qwen_vl_utils.vision_process import (
            FRAME_FACTOR,
            MODEL_SEQ_LEN,
            SPATIAL_MERGE_SIZE,
            VIDEO_MAX_TOKEN_NUM,
            VIDEO_MIN_TOKEN_NUM,
            ceil_by_factor,
        )

        image_factor = 14 * SPATIAL_MERGE_SIZE
        min_pixels = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
        max_pixels = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
        processed_frames = ceil_by_factor(total_num_frames, FRAME_FACTOR)
        total_pixels = MODEL_SEQ_LEN * image_factor * image_factor * 0.9
        return int(
            max(
                min(max_pixels, total_pixels / processed_frames * FRAME_FACTOR),
                int(min_pixels * 1.05),
            )
        )

    def _key(self, query: ValueQuery) -> tuple[Any, ...]:
        return (
            query.state.task_id,
            query.state.episode_id,
            query.state.anchor_frame,
            query.instruction,
            query.playback,
            self.view,
            self.sample_fps,
            self.reduction,
            self.use_video_description,
            self.add_chat_template,
            self.checkpoint,
        )

    def _compute_instruction_rewards_batch(
        self,
        frame_reference_batches: Sequence[
            Sequence[tuple[str, str, str, int]]
        ],
        instructions: Sequence[str],
        source_fps: Sequence[float],
    ) -> list[float]:
        """Run one Qwen forward for a batch of independent prefixes."""
        if self._video_processor is None:
            raise VMBMKError("TOPReward batch processor was not initialized")
        if not (
            len(frame_reference_batches) == len(instructions) == len(source_fps)
        ):
            raise VMBMKError("TOPReward batch inputs have inconsistent lengths")

        messages, texts, timings, keys, videos = self._prepare_reward_messages(
            frame_reference_batches, instructions, source_fps,
        )
        inputs = self._encode_reward_inputs(messages, texts, timings, keys, videos)
        return self._reward_from_inputs(inputs)

    def _prepare_reward_messages(
        self, frame_reference_batches: Sequence[Sequence[tuple[str, str, str, int]]],
        instructions: Sequence[str], source_fps: Sequence[float],
    ) -> tuple[list[Any], list[str], list[Any], list[Any], list[Any]]:
        batch_messages = []
        full_texts: list[str] = []
        timings: list[tuple[float, int, tuple[int, ...]]] = []
        vision_keys: list[tuple[Any, ...]] = []
        cached_video_inputs: list[Any | None] = []
        eos_token = self._client.processor.tokenizer.eos_token

        for frame_references, instruction, fps in zip(
            frame_reference_batches,
            instructions,
            source_fps,
        ):
            frame_count = len(frame_references)
            selected_indices = self._sample_frame_indices(frame_count, fps)
            selected_references = [
                frame_references[index] for index in selected_indices
            ]
            vision_key = self._vision_key(
                frame_references, fps, frame_count
            )
            cached_video = (
                None
                if self.use_video_description
                else self._cached_video_input(vision_key)
            )
            if cached_video is None:
                self._prefetch_frame_references(selected_references)
                video_payload = [
                    self._pil_frame(*reference)
                    for reference in selected_references
                ]
            else:
                # The Qwen chat template depends on the video item, not its
                # frame payload.  The verified cached tensor supplies pixels.
                video_payload = ()
            trajectory_description = None
            if self.use_video_description:
                # Description generation is optional and disabled by every checked-in
                # config.  Preserve its official behavior before batching the reward
                # forward itself.
                self._video_processor.set_video_timing(
                    source_fps=fps,
                    total_num_frames=frame_count,
                )
                trajectory_description = (
                    self._client.generate_object_state_reasoning(
                        [self._frame(*reference) for reference in frame_references],
                        fps=fps,
                    )
                )

            vision_messages, full_text = self._reward_prompt_messages(
                video_payload, instruction, fps, frame_count, trajectory_description, eos_token,
            )

            batch_messages.append(vision_messages)
            full_texts.append(full_text)
            timings.append((fps, frame_count, selected_indices))
            vision_keys.append(vision_key)
            cached_video_inputs.append(cached_video)
        return batch_messages, full_texts, timings, vision_keys, cached_video_inputs

    def _reward_prompt_messages(
        self, video_payload: Any, instruction: str, fps: float, frame_count: int,
        trajectory_description: str | None, eos_token: str | None,
    ) -> tuple[list[Any], str]:
        """Preserve upstream prompt text and final supervised token placement."""
        if trajectory_description is None:
            prompt_text = (
                "The above video shows a robot manipulation trajectory that "
                "completes the following task: "
            )
        else:
            prompt_text = (
                f"{trajectory_description} Therefore given the above "
                "description and the video, the video shows a robot "
                "manipulation trajectory that **completes** the following "
                "instruction: "
            )

        content = [
            {
                "type": "video",
                "video": video_payload,
                "fps": fps,
                "max_pixels": self._source_video_max_pixels(frame_count),
            },
            {"type": "text", "text": prompt_text},
        ]
        user_messages = [{"role": "user", "content": content}]
        if self.add_chat_template:
            instruction_suffix = (
                f"{instruction} Decide whether the above statement is True "
                "or not. The answer is:"
            )
            templated_messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_payload,
                        "fps": fps,
                        "max_pixels": self._source_video_max_pixels(
                            frame_count
                        ),
                    },
                    {
                        "type": "text",
                        "text": f"{prompt_text}{instruction_suffix}",
                    },
                ],
            }]
            prompt_chat = self._client.processor.apply_chat_template(
                templated_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = f"{prompt_chat}True"
            vision_messages = templated_messages
        else:
            instruction_suffix = (
                f"{instruction} Decide whether the above statement is True "
                "or not. The answer is: True"
            )
            prompt_chat = self._client.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if eos_token is not None:
                prompt_chat = prompt_chat.split(eos_token)[0]
            full_text = f"{prompt_chat}{instruction_suffix}"
            vision_messages = user_messages
        return vision_messages, full_text

    def _encode_reward_inputs(
        self, batch_messages: Sequence[Any], full_texts: Sequence[str], timings: Sequence[Any],
        vision_keys: Sequence[Any], cached_video_inputs: list[Any],
    ) -> Any:
        from qwen_vl_utils import process_vision_info

        video_inputs = cached_video_inputs
        missing_messages = []
        missing_positions = []
        missing_keys = []
        for index, cached in enumerate(video_inputs):
            if cached is None:
                missing_messages.append(batch_messages[index])
                missing_positions.append(index)
                missing_keys.append(vision_keys[index])
        if missing_messages:
            _image_inputs, decoded_videos = process_vision_info(missing_messages)
            if (
                decoded_videos is None
                or len(decoded_videos) != len(missing_messages)
            ):
                raise VMBMKError(
                    "TOPReward video preprocessing returned invalid inputs"
                )
            for index, key, video in zip(
                missing_positions, missing_keys, decoded_videos
            ):
                if not self.use_video_description:
                    self._cache_video_input(key, video)
                video_inputs[index] = video
        if any(video is None for video in video_inputs):
            raise VMBMKError(
                "TOPReward video preprocessing did not fill every input"
            )
        self._video_processor.set_pre_sampled_video_timings(timings)
        inputs = self._client.processor(
            text=full_texts,
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._client.model.device)
        return inputs

    def _reward_from_inputs(self, inputs: Any) -> list[float]:
        """Read only the supervised final True token, preserving native scoring."""
        import torch
        import torch.nn.functional as F

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).expand_as(input_ids)
        answer_positions = positions.masked_fill(
            attention_mask == 0,
            -1,
        ).max(dim=1).values
        if (answer_positions < 1).any():
            raise VMBMKError("TOPReward batch contained an empty token sequence")

        labels = torch.full_like(input_ids, -100)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        labels[rows, answer_positions] = input_ids[rows, answer_positions]

        self._client.model.eval()
        with torch.no_grad():
            outputs = self._client.model(**inputs, labels=labels)

        prediction_positions = answer_positions - 1
        answer_logits = outputs.logits[rows, prediction_positions, :]
        answer_ids = input_ids[rows, answer_positions]
        answer_log_probs = F.log_softmax(answer_logits, dim=-1).gather(
            1,
            answer_ids.unsqueeze(1),
        ).squeeze(1)
        # The native implementation supervises exactly the final ``True`` token,
        # so mean and sum are identical.  Keep the reduction validation explicit.
        if self.reduction not in {"mean", "sum"}:
            raise VMBMKError(f"Unknown TOPReward reduction {self.reduction!r}")
        return [float(value) for value in answer_log_probs.detach().cpu()]

    def _score_batch(
        self,
        items: Sequence[
            tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]]
        ],
    ) -> list[float]:
        if self.backend == "molmo":
            return self._score_molmo_batch(items)
        self._load()
        frame_reference_batches = [
            [
                (
                    episode.task_id,
                    episode.episode_id,
                    self.view,
                    view.source_index(frame_index),
                )
                for frame_index in indices
            ]
            for _query, episode, view, indices in items
        ]

        # Unit tests inject a minimal client without the real processor/model.
        # Real Qwen clients always take the batched path below.
        if self._video_processor is None:
            raw_scores = [
                self._client.compute_instruction_reward(
                    frames=[self._frame(*reference) for reference in references],
                    instruction=query.instruction,
                    reduction=self.reduction,
                    fps=episode.fps,
                    use_video_description=self.use_video_description,
                    add_chat_template=self.add_chat_template,
                ).reward
                for (query, episode, _view, _indices), references in zip(
                    items,
                    frame_reference_batches,
                )
            ]
        else:
            raw_scores = self._compute_instruction_rewards_batch(
                frame_reference_batches,
                [query.instruction for query, _episode, _view, _indices in items],
                [episode.fps for _query, episode, _view, _indices in items],
            )

        scores: list[float] = []
        for (query, episode, _view, _indices), raw_score in zip(
            items,
            raw_scores,
        ):
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise VMBMKError(
                    f"TOPReward returned an invalid reward for "
                    f"{episode.task_id}/{episode.episode_id} anchor "
                    f"{query.state.anchor_frame}"
                ) from exc
            if not math.isfinite(score):
                raise VMBMKError(
                    f"TOPReward returned a non-finite reward for "
                    f"{episode.task_id}/{episode.episode_id} anchor "
                    f"{query.state.anchor_frame}: {score}"
                )
            scores.append(score)
        if len(scores) != len(items):
            raise VMBMKError(
                f"TOPReward returned {len(scores)} rewards for {len(items)} queries"
            )
        return scores

    def _score(
        self,
        query: ValueQuery,
        episode: Episode,
        indices: tuple[int, ...],
    ) -> float:
        view = query_view(self.dataset, query, self.view)
        return self._score_batch([(query, episode, view, indices)])[0]

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
            indices = tuple(range(view.timeline_anchor(query.state.anchor_frame) + 1))
            # Qwen3-VL video preprocessing requires at least two frames.  The
            # initial state has a one-frame causal prefix, so duplicate that
            # same observed state rather than advancing the trajectory.
            if len(indices) == 1:
                indices = (0, 0)
            key = self._key(query)
            owned.append(key)
            if key not in self._cache:
                pending.setdefault(key, (query, episode, view, indices))

        items = list(pending.items())
        for short_prefix in (True, False):
            matching = [
                pair
                for pair in items
                if (len(pair[1][3]) <= _TOPREWARD_SHORT_PREFIX_MAX_FRAMES)
                == short_prefix
            ]
            size = (
                _TOPREWARD_SHORT_PREFIX_BATCH_SIZE
                if short_prefix
                else self.batch_size
            )
            for start in range(0, len(matching), size):
                chunk = matching[start : start + size]
                scores = self._score_batch([item for _key, item in chunk])
                for (key, _item), score in zip(chunk, scores):
                    self._cache[key] = score
        return [self._cache[key] for key in owned]

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        if not queries:
            return []
        for query in queries:
            if query.state_a.task_id != query.state_b.task_id:
                raise VMBMKError(
                    f"{query.query_id}: TOPReward compare states must belong "
                    "to the same task"
                )
        return compare_value_difference(queries, self.value)


    def _load_molmo(self) -> None:
        if self._client is not None:
            return
        from types import SimpleNamespace

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model.eval()
        self._client = SimpleNamespace(
            model=model,
            processor=AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True),
        )

    def _sample_molmo_frames(self, total_num_frames: int, source_fps: float) -> tuple[int, ...]:
        import numpy as np

        if not math.isfinite(source_fps) or source_fps <= 0:
            raise VMBMKError("TOPReward Molmo source fps must be positive and finite")
        count = min(
            total_num_frames,
            self.max_frames,
            max(2, int(total_num_frames / source_fps * self.sample_fps)),
        )
        if count == 1:
            return (total_num_frames - 1,)
        return tuple(int(i) for i in np.linspace(0, total_num_frames - 1, count).round())

    def _inputs(
        self,
        video: Any,
        metadata: Mapping[str, Any],
        text: str,
        *,
        generation: bool,
        answer: bool = False
    ) -> Any:
        processor = self._client.processor
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video},
            {"type": "text", "text": text},
        ]}]
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation,
        )
        if not generation:
            eos = processor.tokenizer.eos_token
            if eos is not None:
                rendered = rendered.split(eos)[0]
        if answer:
            rendered += "True"
        inputs = processor(
            videos=[video], video_metadata=[dict(metadata)], text=rendered,
            do_sample_frames=False, padding=True, return_tensors="pt",
        )
        return {key: value.to(self._client.model.device) for key, value in inputs.items()}

    def _check_length(self, inputs: Any) -> None:
        length = inputs["input_ids"].shape[-1]
        if length > self.max_input_length:
            raise VMBMKError(
                f"TOPReward Molmo input has {length} tokens, exceeding "
                f"max_input_length={self.max_input_length}; reduce max_frames"
            )

    def _reward(self, video: Any, metadata: Mapping[str, Any], instruction: str) -> float:
        import torch

        description = None
        if self.use_video_description:
            inputs = self._inputs(
                video, metadata, "Describe the robot manipulation trajectory in the video.",
                generation=True,
            )
            self._check_length(inputs)
            with torch.inference_mode():
                output = self._client.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            description = self._client.processor.tokenizer.decode(
                output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
            ).strip()
        # Match the prompt in TOPReward/topreward/clients/molmo.py.
        prompt = (
            "The above video shows a complete robot manipulation trajectory "
            "of the following task with all the motion: "
            if description is None else
            f"{description} Therefore given the above description and the video, "
            "the video shows a robot manipulation trajectory that **completes** "
            "the following instruction: "
        )
        text = (
            f"{prompt}{instruction}\n"
            " Decide whether the above statement is True or not. The answer is:"
        )
        if not self.add_chat_template:
            text += " True"
        inputs = self._inputs(
            video, metadata, text,
            generation=self.add_chat_template, answer=self.add_chat_template,
        )
        self._check_length(inputs)
        ids = inputs["input_ids"]
        if ids.shape[-1] < 2 or self._client.processor.tokenizer.decode(
            ids[0, -1:].tolist(), skip_special_tokens=True,
        ).strip() != "True":
            raise VMBMKError("TOPReward Molmo expected a final True answer token")
        with torch.inference_mode():
            output = self._client.model(**inputs, use_cache=False)
            # Only this token is supervised by the official reward. Avoid a
            # second full sequence-by-vocabulary tensor for log_softmax.
            logits = output.logits[0, -2].float()
            score = logits.log_softmax(-1)[ids[0, -1]].item()
        if not math.isfinite(score):
            raise VMBMKError(f"TOPReward Molmo returned non-finite reward: {score}")
        return score

    def _score_molmo_batch(
        self,
        items: Sequence[tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]]],
    ) -> list[float]:
        import numpy as np

        self._load()
        scores = []
        # Molmo2's custom processor handles one video per forward. The runner
        # may still group queries; inherited value() deduplicates their keys.
        for query, episode, view, _indices in items:
            count = view.timeline_anchor(query.state.anchor_frame) + 1
            selected = self._sample_frame_indices(count, view.fps)
            references = [
                (episode.task_id, episode.episode_id, self.view, view.source_index(i))
                for i in selected
            ]
            self._prefetch_frame_references(references)
            video = np.stack([self._frame(*reference) for reference in references])
            metadata = {
                "fps": view.fps, "total_num_frames": count,
                "frames_indices": list(selected),
                "height": video.shape[1], "width": video.shape[2],
            }
            scores.append(self._reward(video, metadata, query.instruction))
        return scores
