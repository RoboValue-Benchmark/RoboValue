from __future__ import annotations

from typing import Any, Mapping, Sequence, TYPE_CHECKING

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.data.playback import PlaybackView
from vmbmk.inference.queries import CompareQuery, StateRef, ValueQuery
from .base import Adapter, positive_int, require_config
from .video_inputs import query_view, read_frame

if TYPE_CHECKING:
    from PIL import Image


def _prefix_indices(anchor: int, max_frames: int) -> tuple[int, ...]:
    indices = tuple(
        anchor * index // (max_frames - 1)
        for index in range(max_frames)
    )
    if (
        len(indices) != max_frames
        or indices[0] != 0
        or indices[-1] != anchor
        or any(index < 0 or index > anchor for index in indices)
    ):
        raise VMBMKError(f"RynnValue prefix does not end at frame {anchor}")
    return indices


class RynnValueAdapter(Adapter):
    """Adapter for RynnValue's remaining-time value head.

    RynnValue predicts remaining seconds.  VMBMK values use the negated
    absolute prediction by default, so larger values consistently mean more
    progress.
    """

    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str,
        *,
        num_frames: int,
        batch_size: int,
        mode: str,
        view: str,
        robot_description: str | None,
        camera_description: str | None,
    ) -> None:
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.mode = mode
        self.view = view
        self.robot_description = robot_description
        self.camera_description = camera_description
        self._model = None
        self._processor = None
        self._cache: dict[tuple[Any, ...], float] = {}
        self._frame_cache: dict[tuple[str, str, str, int], Any] = {}
        self._frame_episode: tuple[str, str, str] | None = None
        self._episode_cache: dict[tuple[str, str], Episode] = {}

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "RynnValueAdapter":
        require_config(
            config,
            "rynnvalue",
            {
                "num_frames",
                "batch_size",
                "mode",
                "view",
                "robot_description",
                "camera_description",
            },
        )
        mode = config.get("mode", "absolute")
        if mode not in {"absolute", "remaining_time"}:
            raise ConfigurationError(
                "rynnvalue.mode must be 'absolute' or 'remaining_time'"
            )
        view = config.get("view", "front")
        if not isinstance(view, str) or not view.strip():
            raise ConfigurationError("rynnvalue.view must be a non-empty string")

        def optional_text(name: str, default: str) -> str | None:
            value = config.get(name, default)
            if value is None:
                return None
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"rynnvalue.{name} must be a non-empty string or null"
                )
            return value.strip()

        checkpoint = str(config["checkpoint"])
        if not checkpoint:
            raise ConfigurationError("rynnvalue.checkpoint must be non-empty")
        num_frames = positive_int(
            config.get("num_frames", 8), "rynnvalue.num_frames"
        )
        if num_frames < 2:
            raise ConfigurationError("rynnvalue.num_frames must be at least 2")
        return cls(
            dataset,
            checkpoint,
            num_frames=num_frames,
            batch_size=positive_int(
                config.get("batch_size", 4), "rynnvalue.batch_size"
            ),
            mode=mode,
            view=view.strip(),
            robot_description=optional_text("robot_description", "a robot arm"),
            camera_description=optional_text(
                "camera_description", "a front-facing camera"
            ),
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor

        self._torch = torch
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        config = AutoConfig.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )
        config._attn_implementation = "pred_slot_isolated_eager"
        self._model = AutoModel.from_pretrained(
            self.checkpoint,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self._model.to(device=self._device, dtype=dtype)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )
        if getattr(self._processor, "use_meta", False):
            if (
                self.robot_description is None
                and self.camera_description is None
            ):
                self.robot_description = "a robot arm"
                self.camera_description = "a front-facing camera"

    def _episode(self, state: StateRef) -> Episode:
        key = (state.task_id, state.episode_id)
        episode = self._episode_cache.get(key)
        if episode is None:
            episode = self.dataset.episode(state.task_id, state.episode_id)
            self._episode_cache[key] = episode
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"{state.task_id}/{state.episode_id} frame {state.anchor_frame} "
                f"is outside [0, {episode.num_frames})"
            )
        episode.video(self.view)
        return episode

    def _frame(self, episode: Episode, index: int) -> Image.Image:
        current_episode = (episode.task_id, episode.episode_id, self.view)
        if self._frame_episode != current_episode:
            self._frame_cache.clear()
            self._frame_episode = current_episode
        key = (episode.task_id, episode.episode_id, self.view, index)
        if key not in self._frame_cache:
            self._frame_cache[key] = read_frame(episode.video(self.view), index)
        return self._frame_cache[key]

    def _key(
        self,
        query: ValueQuery,
        indices: tuple[int, ...],
    ) -> tuple[Any, ...]:
        return (
            query.state.task_id,
            query.state.episode_id,
            query.state.anchor_frame,
            query.instruction,
            query.playback,
            indices,
            self.view,
            self.mode,
            self.robot_description,
            self.camera_description,
        )

    @staticmethod
    def _collate(
        processed: Sequence[Mapping[str, Any]], device: Any
    ) -> dict[str, Any]:
        import torch

        # ``process_episode`` may return sequences with different instruction
        # lengths.  Pad them to one batch width before concatenation; the
        # processor's attention mask tells the model which positions count.
        sequence_lengths = [item["input_ids"].shape[-1] for item in processed]
        max_length = max(sequence_lengths)

        def padded(name: str) -> list[Any]:
            values = []
            for item in processed:
                value = item[name]
                padding = max_length - value.shape[-1]
                if padding:
                    value = torch.nn.functional.pad(value, (0, padding))
                values.append(value)
            return values

        result: dict[str, Any] = {
            "input_ids": torch.cat(padded("input_ids"), dim=0),
            "attention_mask": torch.cat(padded("attention_mask"), dim=0),
        }
        for name in ("pixel_values", "pixel_values_videos"):
            values = [item.get(name) for item in processed]
            if values[0] is not None:
                result[name] = torch.cat(
                    [value.flatten(0, 1) for value in values], dim=0
                )
        for name in ("image_grid_thw", "video_grid_thw"):
            values = [item.get(name) for item in processed]
            if values[0] is not None:
                result[name] = torch.cat(
                    [value.flatten(0, 1) for value in values], dim=0
                )
        return {
            name: value.to(device)
            if hasattr(value, "to")
            else value
            for name, value in result.items()
        }

    def _reduce_batch(self, prediction: Any, batch_size: int) -> list[float]:
        torch = self._torch
        prediction = prediction.detach()
        if prediction.dim() == 3:
            prediction = prediction.mean(dim=0)
        if prediction.dim() == 2 and prediction.shape[0] == 1:
            prediction = prediction.reshape(batch_size, -1)
        elif prediction.dim() == 2 and prediction.shape[0] == batch_size:
            pass
        elif prediction.dim() == 2 and prediction.shape[1] == batch_size:
            prediction = prediction.transpose(0, 1)
        elif prediction.dim() == 2:
            prediction = prediction.mean(dim=0).reshape(batch_size, -1)
        elif prediction.dim() == 1:
            prediction = prediction.reshape(batch_size, -1)
        else:
            raise VMBMKError(
                "RynnValue returned unsupported value shape "
                f"{tuple(prediction.shape)}"
            )
        if prediction.shape[0] != batch_size or prediction.shape[1] < 1:
            raise VMBMKError(
                f"RynnValue returned {tuple(prediction.shape)} for batch size {batch_size}"
            )
        values = prediction[:, -1]
        if not torch.isfinite(values).all():
            raise VMBMKError("RynnValue returned non-finite values")
        return [float(value) for value in values.detach().cpu()]

    def _predict(self, queries: Sequence[ValueQuery]) -> list[tuple[Any, ...]]:
        if not queries:
            return []
        pending: dict[
            tuple[Any, ...], tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]]
        ] = {}
        owned: list[tuple[Any, ...]] = []
        for query in queries:
            episode = self._episode(query.state)
            view = query_view(self.dataset, query, self.view)
            indices = _prefix_indices(
                view.timeline_anchor(query.state.anchor_frame), self.num_frames
            )
            key = self._key(query, indices)
            pending.setdefault(key, (query, episode, view, indices))
            owned.append(key)
        missing = [(key, item) for key, item in pending.items() if key not in self._cache]
        if missing:
            self._load()
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start : start + self.batch_size]
                processed = []
                frame_windows: dict[tuple[Any, ...], list[Any]] = {}
                for _key, (query, episode, view, indices) in chunk:
                    source_indices = tuple(
                        view.source_index(index) for index in indices
                    )
                    window_key = (
                        episode.task_id,
                        episode.episode_id,
                        self.view,
                        source_indices,
                    )
                    images = frame_windows.get(window_key)
                    if images is None:
                        images = [
                            self._frame(episode, index) for index in source_indices
                        ]
                        frame_windows[window_key] = images
                    processed.append(
                        self._processor.process_episode(
                            instruction=query.instruction,
                            images=images,
                            robot_description=self.robot_description,
                            camera_description=self.camera_description,
                        )
                    )
                batch = self._collate(processed, self._device)
                with self._torch.inference_mode():
                    outputs = self._model(**batch)
                value_output = getattr(outputs, "value", None)
                prediction = getattr(value_output, "pred_value", None)
                if prediction is None:
                    raise VMBMKError(
                        "RynnValue output does not contain value.pred_value"
                    )
                values = self._reduce_batch(prediction, len(chunk))
                for (key, _item), value in zip(chunk, values):
                    self._cache[key] = -value if self.mode == "absolute" else value
        return owned

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        return [self._cache[key] for key in self._predict(queries)]

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        left = [
            ValueQuery(f"{query.query_id}:a", query.state_a, query.instruction)
            for query in queries
        ]
        right = [
            ValueQuery(f"{query.query_id}:b", query.state_b, query.instruction)
            for query in queries
        ]
        values = self.value([*left, *right])
        split = len(queries)
        if self.mode == "remaining_time":
            return [values[index] - values[split + index] for index in range(split)]
        return [values[split + index] - values[index] for index in range(split)]


__all__ = ["RynnValueAdapter", "_prefix_indices"]
