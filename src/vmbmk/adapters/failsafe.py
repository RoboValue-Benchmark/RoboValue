from __future__ import annotations

import copy
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.inference.queries import StateRef, SubtaskQuery
from .base import Adapter, positive_int, require_config
from .video_inputs import read_frame


_SUBTASK_LABEL = re.compile(r"\bsubtask\s*:\s*", re.IGNORECASE)
_STRUCTURED_ANSWER = re.compile(
    r"(?:^|(?<=[.!?]))[ \t]*(yes|no)\b",
    re.IGNORECASE | re.MULTILINE,
)
_ANSWER = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _prefix_indices(anchor: int, max_length: int) -> tuple[int, ...]:
    if anchor < 0 or max_length < 2:
        raise VMBMKError(
            "FailSafe prefix requires a non-negative anchor and max_length >= 2"
        )
    count = anchor + 1
    if count <= max_length:
        indices = tuple([*range(count), *([anchor] * (max_length - count))])
    else:
        indices = tuple([*range(count - 10, count)])
    if (
        len(indices) != max_length
        or indices[-1] != anchor
        or any(left > right for left, right in zip(indices, indices[1:]))
    ):
        raise VMBMKError(f"FailSafe prefix does not end at frame {anchor}")
    return indices


class FailSafeAdapter(Adapter):
    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str,
        *,
        batch_size: int,
        encoder_batch_size: int,
        seed: int,
        view: str,
        failsafe_root: str | None,
        max_length: int | None,
    ) -> None:
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.encoder_batch_size = encoder_batch_size
        self.seed = seed
        self.view = view
        self.failsafe_root = failsafe_root
        self.max_length: int | None = max_length
        self._model = None
        self._response_cache: dict[tuple[Any, ...], str] = {}

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "FailSafeAdapter":
        require_config(
            config,
            "failsafe",
            {"batch_size", "encoder_batch_size", "seed", "view", "failsafe_root"},
        )
        checkpoint = str(config["checkpoint"])
        if not checkpoint:
            raise ConfigurationError("failsafe.checkpoint must be non-empty")
        seed = config.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("failsafe.seed must be an integer")
        view = config.get("view", "front")
        if not isinstance(view, str) or not view:
            raise ConfigurationError("failsafe.view must be a non-empty string")
        failsafe_root = config.get("failsafe_root")
        if failsafe_root is not None and (
            not isinstance(failsafe_root, str) or not failsafe_root
        ):
            raise ConfigurationError(
                "failsafe.failsafe_root must be a non-empty string or null"
            )
        return cls(
            dataset,
            checkpoint,
            batch_size=positive_int(
                config.get("batch_size", 64), "failsafe.batch_size"
            ),
            encoder_batch_size=positive_int(
                config.get("encoder_batch_size", 16),
                "failsafe.encoder_batch_size",
            ),
            seed=seed,
            view=view,
            failsafe_root=failsafe_root,
            max_length=10,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        # This deployment is intentionally offline and the merged checkpoint
        # contains all language and vision weights.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import numpy as np
        import torch
        from llava.model.builder import load_pretrained_model

        # Set seed everywhere.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        checkpoint_path = Path(self.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_dir():
            raise ConfigurationError(
                f"FailSafe checkpoint does not exist: {checkpoint_path}"
            )
        index = json.loads(
            (checkpoint_path / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        embedding_shard = checkpoint_path / index["weight_map"][
            "model.embed_tokens.weight"
        ]
        from safetensors import safe_open

        with safe_open(str(embedding_shard), framework="pt", device="cpu") as handle:
            checkpoint_vocab_size = handle.get_slice(
                "model.embed_tokens.weight"
            ).get_shape()[0]
        tokenizer, model, image_processor, _ = load_pretrained_model(
            str(checkpoint_path),
            None,
            "llava_qwen",
            device_map="auto",
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
            multimodal=True,
            overwrite_config={"vocab_size": checkpoint_vocab_size},
        )

        self._model = model
        self._model.eval()
        self._tokenizer = tokenizer
        self._image_processor = image_processor

    def _episode(self, state: StateRef) -> Episode:
        episode = self.dataset.episode(state.task_id, state.episode_id)
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"{state.task_id}/{state.episode_id} frame {state.anchor_frame} "
                f"is outside [0, {episode.num_frames})"
            )
        episode.video(self.view)
        return episode

    def _key(
        self,
        query: SubtaskQuery,
        indices: tuple[int, ...],
    ) -> tuple[Any, ...]:
        return (
            query.state.task_id,
            query.state.episode_id,
            query.state.anchor_frame,
            query.instruction,
            self.view,
            indices,
        )

    @staticmethod
    def _answer_match(response: str) -> re.Match[str] | None:
        label = _SUBTASK_LABEL.search(response)
        start = label.end() if label is not None else 0
        return (
            _STRUCTURED_ANSWER.search(response, start)
            or _ANSWER.search(response, start)
        )

    @classmethod
    def _parse_subtask(cls, response: str) -> str:
        label = _SUBTASK_LABEL.search(response)
        start = label.end() if label is not None else 0
        # Unlabelled generations may start directly with the subtask. Require
        # a sentence-boundary verdict so words inside the task are not split.
        answer = (
            cls._answer_match(response)
            if label is not None
            else _STRUCTURED_ANSWER.search(response)
        )
        if answer is None or answer.start() <= start:
            raise VMBMKError(
                "FailSafe model response did not contain a parseable subtask: "
                f"{response!r}"
            )
        subtask = response[start : answer.start()].strip()
        if not subtask:
            raise VMBMKError(
                "FailSafe model response contained an empty subtask: "
                f"{response!r}"
            )
        return subtask

    def _generate_responses(self, video_batch: Sequence[Any], input_ids_batch: Sequence[Any]) -> list[str]:
        """Left-pad one SIA batch and decode the upstream deterministic generation."""
        import torch

        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        max_length = max(ids.shape[-1] for ids in input_ids_batch)
        batch_length = len(input_ids_batch)
        padded_input_ids = torch.full(
            (batch_length, max_length),
            fill_value=pad_token_id,
            dtype=input_ids_batch[0].dtype,
            device=self._model.device,
        )
        batch_attention_mask = torch.zeros(
            (batch_length, max_length),
            dtype=torch.long,
            device=self._model.device,
        )
        for batch_index, ids in enumerate(input_ids_batch):
            sequence_length = ids.shape[-1]

            padded_input_ids[batch_index, -sequence_length:] = ids[0]
            batch_attention_mask[batch_index, -sequence_length:] = 1

        with torch.inference_mode():
            output_ids = self._model.generate(
                inputs=padded_input_ids,
                attention_mask=batch_attention_mask,
                images=video_batch,
                modalities=["video"] * len(video_batch),
                do_sample=False,
                temperature=0,
                max_new_tokens=64,
            )

        responses = self._tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        return responses

    def _responses(
        self, queries: Sequence[SubtaskQuery]
    ) -> list[str]:
        if not queries:
            return []
        episodes = [self._episode(query.state) for query in queries]
        if self.max_length is None:
            raise VMBMKError("FailSafe checkpoint max_length was not initialized")
        indexed = [
            _prefix_indices(query.state.anchor_frame, self.max_length)
            for query in queries
        ]
        keys = [self._key(query, indices) for query, indices in zip(queries, indexed)]
        pending: dict[
            tuple[Any, ...],
            tuple[SubtaskQuery, Episode, tuple[int, ...]],
        ] = {}
        for key, query, episode, indices in zip(keys, queries, episodes, indexed):
            if key not in self._response_cache:
                pending.setdefault(key, (query, episode, indices))

        items = list(pending.items())
        if not items:
            return [self._response_cache[key] for key in keys]

        self._load()
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        inference_batch_size = min(self.batch_size, self.encoder_batch_size)
        for start in range(0, len(items), inference_batch_size):
            chunk = items[start : start + inference_batch_size]
            print(
                f"[FailSafe] batch {start // inference_batch_size + 1}/"
                f"{math.ceil(len(items) / inference_batch_size)} "
                f"({len(chunk)} queries)",
                flush=True,
            )
            video_batch = []
            input_ids_batch = []
            for _, (query, episode, indices) in chunk:
                frame_batch = []
                video_path = episode.video(self.view)
                for index in indices:
                    frame_batch.append(read_frame(video_path, index))

                if len(frame_batch) != self.max_length:
                    raise VMBMKError("FailSafe prepared the wrong number of frames")
                video_tensor = self._image_processor.preprocess(
                    frame_batch,
                    return_tensors="pt",
                )["pixel_values"]
                video_tensor = video_tensor.to(
                    device=self._model.device,
                    dtype=self._model.dtype,
                )

                conversation = copy.deepcopy(conv_templates["qwen_1_5"])
                question = (
                    f"{DEFAULT_IMAGE_TOKEN}\n"
                    f"The task is {query.instruction}. First identify the current "
                    "subtask robot is executing, then determine whether a failure "
                    "is likely at this stage by choosing from ['yes', 'no'], and "
                    "if it is 'yes', also output a corrective action that could "
                    "help the robot return to the correct state."
                )

                conversation.append_message(conversation.roles[0], question)
                conversation.append_message(conversation.roles[1], None)

                prompt = conversation.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt,
                    self._tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(self._model.device)
                video_batch.append(video_tensor)
                input_ids_batch.append(input_ids)

            responses = self._generate_responses(video_batch, input_ids_batch)

            if len(responses) != len(chunk):
                raise VMBMKError(
                    "FailSafe model returned the wrong number of responses"
                )
            for (key, _), response in zip(chunk, responses):
                self._response_cache[key] = response.strip()
        return [self._response_cache[key] for key in keys]

    def subtask(self, queries: Sequence[SubtaskQuery]) -> list[str]:
        return [
            self._parse_subtask(response) for response in self._responses(queries)
        ]
