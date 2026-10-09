from __future__ import annotations

import math
import random
import re
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.data.playback import PlaybackView
from vmbmk.inference.queries import CompareQuery, StateRef, ValueQuery
from .base import Adapter, compare_value_difference, positive_int, require_config


_PROMPT = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.

Task: {task}"""
_SCORE = re.compile(r"ANSWER\s*:\s*([1-5])\b", re.IGNORECASE)


def _normalize_score(score: int) -> float:
    if not 1 <= score <= 5:
        raise VMBMKError(f"RoboReward score is outside [1, 5]: {score}")
    return (score - 1) / 4.0


class RoboRewardAdapter(Adapter):
    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str,
        *,
        batch_size: int,
        max_new_tokens: int,
        seed: int,
        view: str,
    ) -> None:
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.view = view
        self._model = None
        self._processor = None
        self._video_decoder = None
        self._video_decoder_path: str | None = None
        self._cache: dict[tuple[Any, ...], float] = {}
        self._sa_cache: dict[tuple[Any, ...], float] = {}

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "RoboRewardAdapter":
        require_config(
            config,
            "roboreward",
            {"batch_size", "max_new_tokens", "seed", "view"},
        )
        checkpoint = str(config["checkpoint"])
        if not checkpoint:
            raise ConfigurationError("roboreward.checkpoint must be non-empty")
        seed = config.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("roboreward.seed must be an integer")
        view = config.get("view", "front")
        if not isinstance(view, str) or not view:
            raise ConfigurationError("roboreward.view must be a non-empty string")
        return cls(
            dataset,
            checkpoint,
            batch_size=positive_int(
                config.get("batch_size", 1), "roboreward.batch_size"
            ),
            max_new_tokens=positive_int(
                config.get("max_new_tokens", 32),
                "roboreward.max_new_tokens",
            ),
            seed=seed,
            view=view,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import numpy as np
        import torch
        from torchcodec.decoders import VideoDecoder
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.video_utils import VideoMetadata

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self._torch = torch
        self._VideoDecoder = VideoDecoder
        self._VideoMetadata = VideoMetadata
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint,
            dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            local_files_only=True,
        )
        self._processor.tokenizer.padding_side = "left"

    def _episode(self, state: StateRef) -> Episode:
        episode = self.dataset.episode(state.task_id, state.episode_id)
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"{state.task_id}/{state.episode_id} frame {state.anchor_frame} "
                f"is outside [0, {episode.num_frames})"
            )
        episode.video(self.view)
        return episode

    def _key(self, query: ValueQuery) -> tuple[Any, ...]:
        return (
            query.state.task_id,
            query.state.episode_id,
            query.state.anchor_frame,
            query.instruction,
            query.playback,
            self.view,
            self.max_new_tokens,
            self.seed,
        )

    def _parse(self, text: str) -> float:
        return _normalize_score(self._parse_native(text))

    def _parse_native(self, text: str) -> int:
        match = _SCORE.search(text)
        if not match:
            raise VMBMKError(f"cannot parse RoboReward output {text!r}")
        return int(match.group(1))

    def _native_fps(self) -> float:
        fps = self._processor.video_processor.fps
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
            raise VMBMKError(
                f"RoboReward checkpoint video fps must be positive; got {fps!r}"
            )
        return float(fps)

    def _video_metadata(
        self,
        prefix_frames: int,
        fps: float,
        timeline_indices: Sequence[int],
    ) -> Any:
        return self._VideoMetadata(
            total_num_frames=prefix_frames,
            fps=fps,
            duration=prefix_frames / fps,
            frames_indices=list(timeline_indices),
        )

    def _decoder(self, view: PlaybackView) -> tuple[Any, float]:
        path = str(view.path)
        if path != self._video_decoder_path:
            self._video_decoder = self._VideoDecoder(
                path,
                num_ffmpeg_threads=8,
            )
            self._video_decoder_path = path
        decoder_fps = float(self._video_decoder.metadata.average_fps)
        decoder_frames = int(self._video_decoder.metadata.num_frames)
        if not math.isclose(decoder_fps, view.fps, rel_tol=1e-3, abs_tol=1e-3):
            raise VMBMKError(
                f"RoboReward video fps {decoder_fps} does not match "
                f"metadata fps {view.fps} for {view.path}"
            )
        if decoder_frames < view.num_frames:
            raise VMBMKError(
                f"RoboReward video has {decoder_frames} frames but metadata "
                f"declares {view.num_frames} for {view.path}"
            )
        return self._video_decoder, decoder_fps

    def _sampled_video(
        self,
        query: ValueQuery,
        view: PlaybackView,
    ) -> tuple[Any, Any]:
        anchor = view.timeline_anchor(query.state.anchor_frame)
        decoder, fps = self._decoder(view)
        prefix_frames = anchor + 1
        if prefix_frames == 1:
            # Qwen3-VL requires at least one temporal pair. Repeating frame 0
            # is the only valid video input that does not cross the anchor.
            timeline_indices = [0, 0]
        else:
            prefix_metadata = self._video_metadata(
                prefix_frames,
                fps,
                range(prefix_frames),
            )
            timeline_indices = [
                int(index)
                for index in self._processor.video_processor.sample_frames(
                    prefix_metadata,
                )
            ]
        source_indices = list(view.source_indices(timeline_indices))
        video = decoder.get_frames_at(indices=source_indices).data
        metadata = self._video_metadata(
            prefix_frames,
            fps,
            timeline_indices,
        )
        return video, metadata

    def _generate_inputs(self, inputs: Any, expected: int) -> list[str]:
        inputs = inputs.to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = generated[:, inputs.input_ids.shape[1] :]
        outputs = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(outputs) != expected:
            raise VMBMKError(
                "RoboReward returned the wrong number of output rows"
            )
        return list(outputs)

    def _generate(
        self,
        conversations: Sequence[list[dict[str, Any]]],
        processor_kwargs: Mapping[str, Any],
    ) -> list[str]:
        inputs = self._processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=dict(processor_kwargs),
        )
        return self._generate_inputs(inputs, len(conversations))

    def _generate_prepared(
        self,
        conversations: Sequence[list[dict[str, Any]]],
        videos: Sequence[Any],
        video_metadata: Sequence[Any],
    ) -> list[str]:
        prompts = [
            self._processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        inputs = self._processor(
            text=prompts,
            videos=list(videos),
            video_metadata=list(video_metadata),
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        )
        return self._generate_inputs(inputs, len(conversations))

    def _native_generate(
        self, conversations: Sequence[list[dict[str, Any]]]
    ) -> list[str]:
        try:
            return self._generate(
                conversations,
                {"padding": True, "fps": self._native_fps()},
            )
        except self._torch.OutOfMemoryError:
            if len(conversations) == 1:
                raise
            midpoint = len(conversations) // 2
            self._torch.cuda.empty_cache()
            return self._native_generate(
                conversations[:midpoint]
            ) + self._native_generate(conversations[midpoint:])

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        """Score each causal prefix as a rollout ending at its anchor frame."""
        if not queries:
            return []
        pending: dict[tuple[Any, ...], tuple[ValueQuery, PlaybackView]] = {}
        owned: list[tuple[Any, ...]] = []
        for query in queries:
            episode = self._episode(query.state)
            view = PlaybackView(
                path=episode.video(self.view),
                num_frames=episode.num_frames,
                fps=episode.fps,
                playback=query.playback,
            )
            key = self._key(query)
            owned.append(key)
            if key not in self._cache:
                pending.setdefault(key, (query, view))

        items = list(pending.items())
        if items:
            self._load()
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            conversations = []
            videos = []
            video_metadata = []
            for _, (query, view) in chunk:
                video, metadata = self._sampled_video(query, view)
                videos.append(video)
                video_metadata.append(metadata)
                conversations.append([{
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {
                            "type": "text",
                            "text": _PROMPT.format(task=query.instruction),
                        },
                    ],
                }])
            outputs = self._generate_prepared(
                conversations,
                videos,
                video_metadata,
            )
            for (key, _), output in zip(chunk, outputs):
                self._cache[key] = self._parse(output)
        return [self._cache[key] for key in owned]

    def sa(self, queries: Sequence[ValueQuery]) -> list[float]:
        if not queries:
            return []
        pending: dict[tuple[Any, ...], tuple[ValueQuery, Episode]] = {}
        owned: list[tuple[Any, ...]] = []
        for query in queries:
            episode = self._episode(query.state)
            key = (
                episode.task_id,
                episode.episode_id,
                query.instruction,
                self.view,
                self.max_new_tokens,
                self.seed,
            )
            owned.append(key)
            if key not in self._sa_cache:
                pending.setdefault(key, (query, episode))

        items = list(pending.items())
        if items:
            self._load()
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            conversations = [
                [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(episode.video(self.view)),
                        },
                        {
                            "type": "text",
                            "text": _PROMPT.format(task=query.instruction),
                        },
                    ],
                }]
                for _, (query, episode) in chunk
            ]
            outputs = self._native_generate(conversations)
            for (key, _), output in zip(chunk, outputs):
                self._sa_cache[key] = float(self._parse_native(output))
        return [self._sa_cache[key] for key in owned]

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        if not queries:
            return []
        return compare_value_difference(queries, self.value)
