from __future__ import annotations

import logging
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from html import unescape
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.data.playback import PlaybackView
from vmbmk.inference.queries import CompareQuery, StateRef, SubtaskQuery, ValueQuery
from .base import Adapter, compare_value_difference, positive_int, require_config
from .video_inputs import query_view, read_frame

if TYPE_CHECKING:
    from PIL import Image


_LOGGER = logging.getLogger(__name__)


_VLLM_DTYPES = {
    "auto": "auto",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
}
_FINISHED_MARKER = "this task is finished because"
_ALL_TASKS_COMPLETED = "All tasks are completed."
# ProcVLM has emitted all of these list styles in practice:
# ``- text``, ``1. text``, ``1.) text``, ``001: text`` and ``001 - text``.
# Keep the complete prefix in one expression so the same grammar is used to
# detect the first item and the boundary before the next item.  ``1.)`` is the
# list style used by legacy SIA prompts, so it must be matched as a unit
# before extracting the action text.
_LIST_PREFIX = (
    r"(?:[-*\u2022][ \t]+|\d+\.\)[ \t]+|\d+[.)][ \t]+|"
    r"\d+[ \t]*(?:[-\u2013\u2014]|:)[ \t]+)"
)
_LIST_ITEM = re.compile(
    rf"^[ \t]*{_LIST_PREFIX}(.+?)"
    rf"(?=\n[ \t]*(?:{_LIST_PREFIX}|Therefore\b)|\Z)",
    re.MULTILINE | re.DOTALL,
)
_COMPLETED_PROGRESS = re.compile(
    r"\\?\s*<\s*progr(?:\\?\s*)ess\s*>\s*100(?:\.0+)?\s*%\s*"
    r"\\?\s*<\s*/\s*progr(?:\\?\s*)ess\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _uniform_indices(total_frames: int, target_count: int) -> tuple[int, ...]:
    if total_frames <= 0 or target_count <= 0:
        return ()
    if target_count >= total_frames:
        return tuple(range(total_frames))
    if target_count == 1:
        return (0,)
    values = (
        round(index * (total_frames - 1) / (target_count - 1))
        for index in range(target_count)
    )
    return tuple(dict.fromkeys(int(value) for value in values))


def _window_indices(
    anchor: int,
    total_frames: int,
    window_size: int,
    max_sampled_frames: int = 512,
) -> tuple[int, ...]:
    if anchor < 0:
        raise VMBMKError("ProcVLM anchor must be non-negative")
    if total_frames <= 0:
        raise VMBMKError("ProcVLM trajectory must contain at least one frame")
    if anchor >= total_frames:
        raise VMBMKError("ProcVLM anchor must be inside the trajectory")
    if window_size <= 0:
        raise VMBMKError("ProcVLM window size must be positive")
    if max_sampled_frames <= 0:
        raise VMBMKError("ProcVLM max sampled frames must be positive")
    selected = _uniform_indices(
        total_frames, min(total_frames, max_sampled_frames)
    )
    history = tuple(index for index in selected if index <= anchor)
    if history[-1] != anchor:
        history += (anchor,)
    window = history[-window_size:]
    return (window[0],) * (window_size - len(window)) + window


class ProcVLMAdapter(Adapter):
    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str,
        *,
        window_size: int,
        batch_size: int,
        max_new_tokens: int,
        torch_dtype: str,
        seed: int,
        view: str,
        max_model_len: int | None = None,
        use_lora: bool = False,
        enable_value_head: bool = False,
        max_sampled_frames: int = 512,
        image_decode_workers: int = 1,
        checkpoint_by_task: Mapping[str, str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.window_size = window_size
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self.max_model_len = max_model_len
        self.seed = seed
        self.view = view
        self.use_lora = use_lora
        self.enable_value_head = enable_value_head
        self.max_sampled_frames = max_sampled_frames
        self.image_decode_workers = positive_int(
            image_decode_workers, "procvlm.image_decode_workers"
        )
        self.checkpoint_by_task = {
            str(task): str(path) for task, path in (checkpoint_by_task or {}).items()
        }
        self._infer = None
        self._prompt = None
        self._parse_progress = None
        self._image_executor: ThreadPoolExecutor | None = None
        self._cache: dict[tuple[Any, ...], float] = {}
        self._subtask_cache: dict[tuple[Any, ...], str] = {}

    def close(self) -> None:
        """Release the cached vLLM engine before the worker interpreter exits."""
        if self._image_executor is not None:
            self._image_executor.shutdown(wait=True)
            self._image_executor = None
        if self._infer is None or self.use_lora:
            return
        from core.backends.vllm import shutdown_vllm

        shutdown_vllm()

    @staticmethod
    def resolve_use_lora(config: Mapping[str, Any]) -> bool:
        """Use task checkpoints by default; checkpoint names do not select a mode."""
        use_lora = config.get("use_lora", bool(config.get("checkpoint_by_task")))
        if not isinstance(use_lora, bool):
            raise ConfigurationError("procvlm.use_lora must be a boolean")
        return use_lora

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "ProcVLMAdapter":
        require_config(
            config,
            "procvlm",
            {
                "window_size",
                "batch_size",
                "max_new_tokens",
                "torch_dtype",
                "max_model_len",
                "seed",
                "view",
                "use_lora",
                "enable_value_head",
                "source_root",
                "max_sampled_frames",
                "image_decode_workers",
                "checkpoint_by_task",
            },
        )
        checkpoint = str(config["checkpoint"])
        if not checkpoint:
            raise ConfigurationError("procvlm.checkpoint must be non-empty")
        seed = config.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("procvlm.seed must be an integer")
        torch_dtype = config.get("torch_dtype", "bf16")
        if torch_dtype not in {"auto", "bf16", "bfloat16", "fp16", "float16"}:
            raise ConfigurationError(
                "procvlm.torch_dtype must be auto, bf16, bfloat16, fp16, or float16"
            )
        view = config.get("view", "front")
        if not isinstance(view, str) or not view:
            raise ConfigurationError("procvlm.view must be a non-empty string")
        use_lora = cls.resolve_use_lora(config)
        enable_value_head = config.get("enable_value_head", False)
        if not isinstance(enable_value_head, bool):
            raise ConfigurationError(
                "procvlm.enable_value_head must be a boolean"
            )
        max_model_len = config.get("max_model_len", 32768)
        if max_model_len is not None:
            max_model_len = positive_int(max_model_len, "procvlm.max_model_len")
        checkpoint_by_task = config.get("checkpoint_by_task")
        if checkpoint_by_task is not None:
            if not isinstance(checkpoint_by_task, Mapping):
                raise ConfigurationError(
                    "procvlm.checkpoint_by_task must be an object"
                )
            unknown = sorted(set(map(str, checkpoint_by_task)) - set(dataset.tasks))
            if unknown:
                raise ConfigurationError(
                    f"procvlm.checkpoint_by_task has unknown tasks: {unknown}"
                )
        return cls(
            dataset,
            checkpoint,
            window_size=positive_int(
                config.get("window_size", 8), "procvlm.window_size"
            ),
            batch_size=cls._batch_size(config.get("batch_size", -1)),
            max_new_tokens=positive_int(
                config.get("max_new_tokens", 4096),
                "procvlm.max_new_tokens",
            ),
            torch_dtype=str(torch_dtype),
            max_model_len=max_model_len,
            seed=seed,
            view=view,
            use_lora=use_lora,
            enable_value_head=enable_value_head,
            max_sampled_frames=positive_int(
                config.get("max_sampled_frames", 512),
                "procvlm.max_sampled_frames",
            ),
            image_decode_workers=positive_int(
                config.get("image_decode_workers", 1),
                "procvlm.image_decode_workers",
            ),
            checkpoint_by_task=checkpoint_by_task,
        )

    @staticmethod
    def _batch_size(value: Any) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value == 0
            or value < -1
        ):
            raise ConfigurationError(
                "procvlm.batch_size must be -1 or a positive integer"
            )
        return value

    def _load(self) -> None:
        if self._infer is not None:
            return
        import numpy as np
        import torch
        from evqa.inference import extract_progress, load_prompt_template
        from evqa.model import batch_chat_with_value_head, batch_chat_with_vllm

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self._infer = (
            batch_chat_with_value_head if self.use_lora else batch_chat_with_vllm
        )
        self._prompt = load_prompt_template
        self._parse_progress = extract_progress

    def _episode(self, state: StateRef) -> Episode:
        episode = self.dataset.episode(state.task_id, state.episode_id)
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"{state.task_id}/{state.episode_id} frame {state.anchor_frame} "
                f"is outside [0, {episode.num_frames})"
            )
        episode.video(self.view)
        return episode

    def _decode_frame_key(self, key: tuple[str, str, int]) -> Image.Image:
        return self._frame(*key)

    def _frame_keys(
        self, query: ValueQuery | SubtaskQuery, episode: Episode, view: PlaybackView
    ) -> list[tuple[str, str, int]]:
        """Preserve the model's ordered window, including repeated frames."""
        indices = _window_indices(
            view.timeline_anchor(query.state.anchor_frame),
            view.timeline_num_frames,
            self.window_size,
            self.max_sampled_frames,
        )
        return [
            (episode.task_id, episode.episode_id, view.source_index(index))
            for index in indices
        ]

    def _checkpoint_groups(
        self,
        items: Sequence[tuple[tuple[Any, ...], Any]],
    ) -> dict[str, list[tuple[tuple[Any, ...], Any]]]:
        """Group queries by checkpoint in first-appearance order."""
        groups: dict[str, list[tuple[tuple[Any, ...], Any]]] = {}
        for key, item in items:
            checkpoint = self.checkpoint_by_task.get(item[0].state.task_id, self.checkpoint)
            groups.setdefault(checkpoint, []).append((key, item))
        return groups

    def _batch_items(
        self,
        chunk: Sequence[tuple[tuple[Any, ...], tuple[ValueQuery, Episode, PlaybackView]]],
    ) -> list[dict[str, Any]]:
        """Build an ordered model batch after de-duplicated frame prefetch."""
        windows: list[tuple[Any, list[tuple[str, str, int]]]] = []
        frame_keys: list[tuple[str, str, int]] = []
        for _, (query, episode, view) in chunk:
            keys = self._frame_keys(query, episode, view)
            windows.append((query, keys))
            frame_keys.extend(keys)
        unique_keys = list(dict.fromkeys(frame_keys))
        if self.image_decode_workers == 1 or len(unique_keys) < 2:
            images = [self._decode_frame_key(key) for key in unique_keys]
        else:
            if self._image_executor is None:
                self._image_executor = ThreadPoolExecutor(
                    max_workers=self.image_decode_workers,
                    thread_name_prefix="procvlm-frame",
                )
            images = list(self._image_executor.map(self._decode_frame_key, unique_keys))
        frames = dict(zip(unique_keys, images))
        return [
            {
                "image": [frames[key] for key in keys],
                "conversations": [{"from": "human", "value": self._prompt(query.instruction)}],
            }
            for query, keys in windows
        ]

    @lru_cache(maxsize=4096)
    def _frame(self, task_id: str, episode_id: str, frame_index: int) -> Image.Image:
        episode = self.dataset.episode(task_id, episode_id)
        return read_frame(episode.video(self.view), frame_index)

    def _key(self, query: ValueQuery | SubtaskQuery) -> tuple[Any, ...]:
        return (
            query.state.task_id,
            query.state.episode_id,
            query.state.anchor_frame,
            query.instruction,
            getattr(query, "playback", "forward"),
            self.view,
            self.window_size,
            self.max_new_tokens,
            self.torch_dtype,
            self.seed,
            self.use_lora,
            self.enable_value_head,
            self.max_sampled_frames,
        )

    def _subtask_key(self, query: SubtaskQuery) -> tuple[Any, ...]:
        return ("subtask", *self._key(query))

    @staticmethod
    def _parse_subtask(output: str) -> str | None:
        # API wrappers occasionally serialize line breaks and HTML-escape the
        # model response.  Decode those transport artifacts before parsing the
        # actual procedural list.
        output = output.replace("\\r\\n", "\n")
        output = output.replace("\\n", "\n").replace("\\r", "\r")
        output = unescape(output)
        # A tokenized response can split a numeric item ID across a line
        # break (for example ``00\n5:``).  Join digit runs of any length
        # immediately before the item's colon so IDs such as ``003``, ``123``
        # and longer IDs are handled uniformly while ordinary prose spacing
        # is preserved.
        output = re.sub(r"(?<!\d)(\d+)\s+(\d+)(?=\s*:)", r"\1\2", output)
        if (
            _FINISHED_MARKER in output.lower()
            or _COMPLETED_PROGRESS.search(output)
        ):
            return _ALL_TASKS_COMPLETED
        match = _LIST_ITEM.search(output)
        if match is None:
            # Native responses sometimes describe one remaining action in prose.
            # Keep that prediction verbatim for the judge, without its progress.
            prose = re.fullmatch(
                r"(.*?)\s*Therefore,\s*the estimated progress is\s*"
                r"<progress>\s*\d+(?:\.\d+)?\s*%\s*</progress>\s*\.?\s*",
                output.strip(),
                re.IGNORECASE | re.DOTALL,
            )
            if prose is not None:
                description = re.sub(r"\s+", " ", prose.group(1)).strip()
                if description and re.search(r"[A-Za-z]", description):
                    return description
            direct_action = output.strip()
            if re.fullmatch(
                r"(?:Place|Pick|Put|Grasp|Grab|Move|Lift|Hang|Insert|Fold|Close|Open|"
                r"Press|Push|Pull|Stack|Sweep|Rotate|Turn|Release|Slide|Align|"
                r"Remove|Take|Set|Hold|Carry|Position|Arrange|Store)\b[^\n<>]+[.!]",
                direct_action,
                re.IGNORECASE,
            ):
                return direct_action
            return None
        action = re.sub(r"\s+", " ", match.group(1)).strip()
        action = re.sub(r"^\d+\s*[:\-\u2013\u2014]\s*", "", action)
        return action or None

    def _parse(self, output: str) -> float | None:
        progress = self._parse_progress(output)
        if progress is None:
            if _FINISHED_MARKER in output.lower():
                return 1.0
            return None
        value = float(progress) / 100.0
        if not math.isfinite(value):
            raise VMBMKError("ProcVLM returned a non-finite prediction")
        return value

    def _infer_items(
        self, batch_items: list[dict[str, Any]], checkpoint: str | None = None
    ) -> list[str]:
        checkpoint = checkpoint or self.checkpoint
        if self.use_lora:
            outputs = self._infer(
                batch_items=batch_items,
                model_path=checkpoint,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                torch_dtype=self.torch_dtype,
                dp=1,
                enable_value_head=self.enable_value_head,
            )
        else:
            # Leave GPU cache sizing to the ProcVLM vLLM backend.  Passing a
            # fixed utilization here makes startup fail when the device is
            # already partially occupied by another workload.
            engine_kwargs: dict[str, Any] = {
                "dtype": _VLLM_DTYPES[self.torch_dtype],
            }
            if self.max_model_len is not None:
                engine_kwargs["max_model_len"] = self.max_model_len
            outputs = self._infer(
                batch_items=batch_items,
                model_path=checkpoint,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                tp=1,
                sampling_kwargs={"seed": self.seed},
                engine_kwargs=engine_kwargs,
            )
        if len(outputs) != len(batch_items):
            raise VMBMKError(
                "ProcVLM returned the wrong number of output rows"
            )
        return outputs

    def _release_task_lora(self) -> None:
        """Release a task-specific LoRA before loading the next task.

        ProcVLM's helper caches one full merged base model per LoRA path.  A
        multi-task metric would otherwise retain every task model and exhaust
        GPU memory even though the task groups are evaluated sequentially.
        """
        if not self.use_lora or not self.checkpoint_by_task:
            return
        try:
            import gc
            import torch
            import evqa.model as procvlm_model

            for cache_name in ("_PROCVLM_CACHE", "_ROBOT_VLM_CACHE"):
                cache = getattr(procvlm_model, cache_name, None)
                if isinstance(cache, dict):
                    cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            _LOGGER.warning(
                "Failed to release task-specific ProcVLM LoRA caches; "
                "continuing with possible retained GPU memory",
                exc_info=True,
            )

    def _forward_fill_failed_values(
        self,
        failed: Sequence[tuple[Any, ...]],
        pending: Mapping[
            tuple[Any, ...], tuple[ValueQuery, Episode, PlaybackView]
        ],
    ) -> None:
        ordered = sorted(
            failed,
            key=lambda key: pending[key][2].timeline_anchor(
                pending[key][0].state.anchor_frame
            ),
        )
        for key in ordered:
            query, _, view = pending[key]
            frame = view.timeline_anchor(query.state.anchor_frame)
            previous = max(
                (
                    (
                        view.timeline_anchor(candidate_key[2]),
                        candidate_value,
                    )
                    for candidate_key, candidate_value in self._cache.items()
                    if candidate_key[0] == key[0]
                    and candidate_key[1] == key[1]
                    and candidate_key[3:] == key[3:]
                    and view.timeline_anchor(candidate_key[2]) < frame
                ),
                default=None,
                key=lambda item: item[0],
            )
            self._cache[key] = 0.0 if previous is None else previous[1]

    def _pending_queries(
        self,
        queries: Sequence[ValueQuery | SubtaskQuery],
        cache: Mapping[tuple[Any, ...], Any],
        key_function: Callable[[ValueQuery | SubtaskQuery], tuple[Any, ...]],
    ) -> tuple[
        dict[tuple[Any, ...], tuple[Any, Episode, PlaybackView]],
        list[tuple[Any, ...]],
    ]:
        """Deduplicate uncached queries while retaining original output order."""
        pending = {}
        owned = []
        for query in queries:
            episode = self._episode(query.state)
            key = key_function(query)
            owned.append(key)
            if key not in cache:
                pending.setdefault(key, (query, episode, query_view(self.dataset, query, self.view)))
        return pending, owned

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        if not queries:
            return []
        pending, owned = self._pending_queries(queries, self._cache, self._key)

        items = list(pending.items())
        failed: list[tuple[Any, ...]] = []
        if items:
            self._load()
        chunk_size = len(items) if self.batch_size == -1 else self.batch_size
        # One-shot ProcVLM uses a task-specific LoRA.  Keep each inference
        # batch on a single adapter; the underlying loader caches adapters by
        # path, so this also works for mixed-task metric query batches.
        for checkpoint, grouped_items in self._checkpoint_groups(items).items():
            for start in range(0, len(grouped_items), chunk_size):
                chunk = grouped_items[start : start + chunk_size]
                batch_items = self._batch_items(chunk)
                outputs = self._infer_items(batch_items, checkpoint)
                retry_keys = []
                retry_items = []
                for (key, _), batch_item, output in zip(
                    chunk, batch_items, outputs
                ):
                    value = self._parse(output)
                    if value is None:
                        retry_keys.append(key)
                        retry_items.append(batch_item)
                    else:
                        self._cache[key] = value
                if retry_items:
                    retry_outputs = self._infer_items(retry_items, checkpoint)
                    for key, retry_output in zip(retry_keys, retry_outputs):
                        value = self._parse(retry_output)
                        if value is None:
                            failed.append(key)
                            continue
                        self._cache[key] = value
            self._release_task_lora()
        if failed:
            self._forward_fill_failed_values(failed, pending)
        return [self._cache[key] for key in owned]

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        return compare_value_difference(queries, self.value)

    def subtask(self, queries: Sequence[SubtaskQuery]) -> list[str]:
        if not queries:
            return []
        pending, owned = self._pending_queries(queries, self._subtask_cache, self._subtask_key)
        if pending:
            self._load()
        items = list(pending.items())
        chunk_size = len(items) if self.batch_size == -1 else self.batch_size
        for checkpoint, grouped_items in self._checkpoint_groups(items).items():
            for start in range(0, len(grouped_items), chunk_size):
                chunk = grouped_items[start : start + chunk_size]
                batch_items = []
                for _, (query, episode, view) in chunk:
                    images = [
                        self._frame(*key)
                        for key in self._frame_keys(query, episode, view)
                    ]
                    batch_items.append({
                        "image": images,
                        "conversations": [{
                            "from": "human",
                            "value": self._prompt(query.instruction),
                        }],
                    })
                outputs = self._infer_items(batch_items, checkpoint)
                retry_keys = []
                retry_items = []
                first_outputs = []
                for (key, _), batch_item, output in zip(chunk, batch_items, outputs):
                    subtask = self._parse_subtask(output)
                    if subtask is None:
                        retry_keys.append(key)
                        retry_items.append(batch_item)
                        first_outputs.append(output)
                    else:
                        self._subtask_cache[key] = subtask
                if retry_items:
                    retry_outputs = self._infer_items(retry_items, checkpoint)
                    for key, first_output, retry_output in zip(
                        retry_keys, first_outputs, retry_outputs
                    ):
                        subtask = self._parse_subtask(retry_output)
                        if subtask is None:
                            raise VMBMKError(
                                "cannot parse ProcVLM subtask after one retry; "
                                f"first={first_output!r}, retry={retry_output!r}"
                            )
                        self._subtask_cache[key] = subtask
            self._release_task_lora()
        return [self._subtask_cache[key] for key in owned]
