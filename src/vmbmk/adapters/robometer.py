from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.data.playback import PlaybackView
from vmbmk.inference.queries import CompareQuery, StateRef, ValueQuery
from .base import Adapter, compare_value_difference, positive_int, require_config
from .video_inputs import query_view, read_frame


def _prefix_indices(anchor: int, max_frames: int) -> tuple[int, ...]:
    count = anchor + 1
    if count <= max_frames:
        indices = tuple([*range(count), *([anchor] * (max_frames - count))])
    else:
        indices = tuple(
            round(anchor * index / (max_frames - 1))
            for index in range(max_frames)
        )
    if (
        len(indices) != max_frames
        or indices[0] != 0
        or indices[-1] != anchor
        or any(index > anchor for index in indices)
    ):
        raise VMBMKError(f"Robometer prefix does not end at frame {anchor}")
    return indices


class RobometerAdapter(Adapter):
    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str,
        *,
        max_frames: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.seed = seed
        self._model = None
        self._cache: dict[tuple[Any, ...], float] = {}
        self._success_cache: dict[tuple[Any, ...], float] = {}

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "RobometerAdapter":
        require_config(
            config,
            "robometer",
            {"max_frames", "batch_size", "seed"},
        )
        seed = config.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("robometer.seed must be an integer")
        checkpoint = str(config["checkpoint"])
        if not checkpoint:
            raise ConfigurationError("robometer.checkpoint must be non-empty")
        max_frames = positive_int(
            config.get("max_frames", 8), "robometer.max_frames"
        )
        if max_frames < 2:
            raise ConfigurationError("robometer.max_frames must be at least 2")
        return cls(
            dataset,
            checkpoint,
            max_frames=max_frames,
            batch_size=positive_int(
                config.get("batch_size", 32), "robometer.batch_size"
            ),
            seed=seed,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import numpy as np
        import torch
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self._torch = torch
        self._ProgressSample = ProgressSample
        self._Trajectory = Trajectory
        self._compute_batch_outputs = compute_batch_outputs
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._cfg, self._tokenizer, processor, self._model = load_model_from_hf(
            model_path=self.checkpoint,
            device=self._device,
        )
        self._model.eval()
        self._collator = setup_batch_collator(
            processor,
            self._tokenizer,
            self._cfg,
            is_eval=True,
        )
        self._discrete = self._cfg.loss.progress_loss_type.lower() == "discrete"
        self._num_bins = int(self._cfg.loss.progress_discrete_bins)

    def _episode(self, state: StateRef) -> Episode:
        episode = self.dataset.episode(state.task_id, state.episode_id)
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"{state.task_id}/{state.episode_id} frame {state.anchor_frame} "
                f"is outside [0, {episode.num_frames})"
            )
        episode.video("front")
        return episode

    def _key(
        self, query: ValueQuery, episode: Episode, indices: tuple[int, ...]
    ) -> tuple[Any, ...]:
        return (
            episode.task_id,
            episode.episode_id,
            query.state.anchor_frame,
            query.instruction,
            query.playback,
            indices,
            self.seed,
        )

    def _predict_batch(
        self, chunk: Sequence[tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]]],
    ) -> Any:
        """Prepare a same-length batch and run the upstream progress model."""
        import numpy as np

        samples = []
        frame_windows: dict[tuple[Any, ...], Any] = {}
        for query, episode, view, indices in chunk:
            source_indices = tuple(
                view.source_index(index) for index in indices
            )
            window_key = (str(view.path), source_indices)
            frames = frame_windows.get(window_key)
            if frames is None:
                frames = np.asarray(
                    [read_frame(view.path, index) for index in source_indices]
                )
                frame_windows[window_key] = frames
            trajectory = self._Trajectory(
                frames=frames,
                frames_shape=tuple(frames.shape),
                task=query.instruction,
                id=f"{episode.task_id}:{episode.episode_id}:{query.state.anchor_frame}",
                quality_label=None,
                metadata={"subsequence_length": len(indices)},
                video_embeddings=None,
            )
            samples.append(self._ProgressSample(trajectory=trajectory))
        batch = self._collator(samples)["progress_inputs"]
        batch = {
            name: value.to(self._device)
            if isinstance(value, self._torch.Tensor)
            else value
            for name, value in batch.items()
        }
        with self._torch.inference_mode():
            result = self._compute_batch_outputs(
                self._model,
                self._tokenizer,
                batch,
                "progress",
                is_discrete_mode=self._discrete,
                num_bins=self._num_bins,
            )
        return result

    def _store_predictions(
        self, chunk: Sequence[tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]]],
        result: Mapping[str, Any], need_success: bool,
    ) -> None:
        import numpy as np

        rows = result.get("progress_pred", [])
        if len(rows) != len(chunk):
            raise VMBMKError(
                "Robometer returned the wrong number of progress rows"
            )
        success_rows = result.get("outputs_success", {}).get("success_probs")
        if success_rows is not None and len(success_rows) != len(chunk):
            raise VMBMKError(
                "Robometer returned the wrong number of success rows"
            )
        if need_success and success_rows is None:
            raise VMBMKError(
                "Robometer did not return native success probabilities"
            )
        for index, (item, row) in enumerate(zip(chunk, rows)):
            query, episode, _view, indices = item
            values = np.asarray(row, dtype=float).reshape(-1)
            if len(values) != len(indices):
                raise VMBMKError(
                    f"Robometer returned {len(values)} values for "
                    f"{len(indices)} prefix frames"
                )
            key = self._key(query, episode, indices)
            self._cache[key] = float(values[-1])
            if success_rows is not None:
                success = np.asarray(success_rows[index], dtype=float).reshape(
                    -1
                )
                if len(success) != len(indices):
                    raise VMBMKError(
                        f"Robometer returned {len(success)} success values "
                        f"for {len(indices)} prefix frames"
                    )
                self._success_cache[key] = float(success[-1])

    def _predict(
        self,
        queries: Sequence[ValueQuery],
        *,
        need_success: bool,
    ) -> list[tuple[Any, ...]]:
        if not queries:
            return []
        pending: dict[
            tuple[Any, ...], tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]]
        ] = {}
        owned: list[tuple[Any, ...]] = []
        for query in queries:
            episode = self._episode(query.state)
            view = query_view(self.dataset, query, "front")
            indices = _prefix_indices(
                view.timeline_anchor(query.state.anchor_frame),
                self.max_frames,
            )
            key = self._key(query, episode, indices)
            pending.setdefault(key, (query, episode, view, indices))
            owned.append(key)
        missing = [
            item
            for key, item in pending.items()
            if key not in self._cache
            or (need_success and key not in self._success_cache)
        ]
        if missing:
            self._load()
        by_length: dict[
            int, list[tuple[ValueQuery, Episode, PlaybackView, tuple[int, ...]]]
        ] = defaultdict(list)
        for item in missing:
            by_length[len(item[3])].append(item)
        for items in by_length.values():
            for start in range(0, len(items), self.batch_size):
                chunk = items[start : start + self.batch_size]
                result = self._predict_batch(chunk)
                self._store_predictions(chunk, result, need_success)
                del result
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        return owned

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        keys = self._predict(queries, need_success=False)
        return [self._cache[key] for key in keys]

    def sa(self, queries: Sequence[ValueQuery]) -> list[float]:
        keys = self._predict(queries, need_success=True)
        return [self._success_cache[key] for key in keys]

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        return compare_value_difference(queries, self.value)
