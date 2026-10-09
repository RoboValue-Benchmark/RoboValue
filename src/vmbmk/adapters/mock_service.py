from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset
from vmbmk.errors import ConfigurationError
from vmbmk.inference.queries import CompareQuery, SubtaskQuery, ValueQuery
from .base import Adapter, compare_value_difference, require_config
from .video_inputs import query_view

if TYPE_CHECKING:
    from PIL import Image


class MockServiceAdapter(Adapter):
    """CPU example: native context stays in the adapter; only prediction is mocked."""

    requires_gpu = False
    requires_checkpoint = False

    def __init__(self, dataset: Dataset, *, view: str, model_version: str) -> None:
        self.dataset = dataset
        self.view = view
        self.model_version = model_version

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset,
    ) -> MockServiceAdapter:
        require_config(
            config, "mock_service", {"batch_size", "view", "model_version"},
            requires_checkpoint=cls.requires_checkpoint,
        )
        view = config.get("view", "front")
        model_version = config.get("model_version", "mock-1")
        for name, value in (("view", view), ("model_version", model_version)):
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"mock_service.{name} must be non-empty text")
        return cls(dataset, view=view, model_version=model_version)

    def _frames(self, query: ValueQuery | SubtaskQuery) -> list[Image.Image]:
        playback = query_view(self.dataset, query, self.view)
        anchor = playback.timeline_anchor(query.state.anchor_frame)
        return playback.read_many(range(anchor + 1))

    def _predict_value(self, frames: Sequence[Image.Image], instruction: str) -> float:
        """Toy prediction over the full RGB prefix, not a robotic value model."""
        from PIL import ImageStat

        return sum(ImageStat.Stat(frame).mean[0] for frame in frames) / (
            255 * len(frames)
        )

    def _predict_subtask(self, frames: Sequence[Image.Image], instruction: str) -> str:
        """Placeholder text; a real service must return its native prediction."""
        return f"Mock prediction for: {instruction} ({len(frames)} frames)"

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        return [
            self._predict_value(self._frames(query), query.instruction)
            for query in queries
        ]

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        return compare_value_difference(queries, self.value)

    def subtask(self, queries: Sequence[SubtaskQuery]) -> list[str]:
        return [
            self._predict_subtask(self._frames(query), query.instruction)
            for query in queries
        ]
