from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TYPE_CHECKING

from vmbmk.errors import VMBMKError

if TYPE_CHECKING:
    from PIL import Image


Playback = Literal["forward", "reverse", "cycle"]


def _validate_frame(frame: int, num_frames: int) -> None:
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise VMBMKError("playback frame must be an integer")
    if num_frames <= 0 or frame < 0 or frame >= num_frames:
        raise VMBMKError(
            f"playback frame {frame} is outside [0, {num_frames})"
        )


@dataclass(frozen=True)
class PlaybackView:
    path: Path
    num_frames: int
    fps: float
    playback: Playback = "forward"

    @property
    def timeline_num_frames(self) -> int:
        """Return the number of logical frames exposed by this view."""
        if self.playback == "cycle":
            return 2 * self.num_frames - 1
        return self.num_frames

    def timeline_anchor(self, source_anchor: int) -> int:
        _validate_frame(source_anchor, self.num_frames)

        if self.playback == "forward":
            return source_anchor
        if self.playback == "reverse":
            return self.num_frames - 1 - source_anchor
        if self.playback == "cycle":
            # The peak is not repeated: 0..T-1,T-2..0. A cycle query denotes
            # the descending occurrence of the source frame.
            return 2 * (self.num_frames - 1) - source_anchor
        raise VMBMKError(f"unsupported playback mode: {self.playback!r}")

    def source_index(self, timeline_index: int) -> int:
        _validate_frame(timeline_index, self.timeline_num_frames)

        if self.playback == "forward":
            return timeline_index
        if self.playback == "reverse":
            return self.num_frames - 1 - timeline_index
        if self.playback == "cycle":
            peak = self.num_frames - 1
            return (
                timeline_index
                if timeline_index <= peak
                else 2 * peak - timeline_index
            )
        raise VMBMKError(f"unsupported playback mode: {self.playback!r}")

    def source_indices(
        self, timeline_indices: Sequence[int]
    ) -> tuple[int, ...]:
        """Map logical indices without changing order or duplicate entries."""
        return tuple(self.source_index(index) for index in timeline_indices)

    def read(self, timeline_index: int) -> Image.Image:
        # Import lazily to avoid the playback <-> adapter utility import cycle.
        from vmbmk.adapters.video_inputs import read_frame

        source_index = self.source_index(timeline_index)
        return read_frame(self.path, source_index)

    def read_many(self, timeline_indices: Sequence[int]) -> list[Image.Image]:
        # Preserve the caller's order and duplicate entries.
        return [self.read(index) for index in timeline_indices]
