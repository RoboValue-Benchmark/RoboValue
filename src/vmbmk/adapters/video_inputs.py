from __future__ import annotations

from types import ModuleType
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, TYPE_CHECKING

from vmbmk.data.dataset import Dataset
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.inference.queries import ValueQuery, SubtaskQuery
from vmbmk.data.playback import PlaybackView

if TYPE_CHECKING:
    from PIL import Image


_PYAV_ONLY: dict[str, bool] = {}
_IMAGEIO_CACHE = threading.local()
_VIDEO_READER = threading.local()
_VIDEO_READER_LIMIT = 4
_PYAV_SEQUENTIAL_GAP = 32

# Keep decoded images bounded across the process. This is large enough to
# retain ordinary model windows without letting long videos accumulate in RAM.
# Video prefixes are reused heavily by TGA (the same episode/prefix is scored
# against several instructions). Keep enough decoded frames to avoid seeking
# and decoding the same prefix repeatedly across adjacent batches.
_FRAME_CACHE_BUDGET = 512 * 1024 * 1024
_FRAME_CACHE: OrderedDict[tuple[str, int], tuple[Any, int]] = OrderedDict()
_FRAME_CACHE_BYTES = 0
_FRAME_CACHE_LOCK = threading.Lock()


def task_reference_clip(
    root: Path, task_id: str, view: str = "front"
) -> tuple[Path, int]:
    names = [task_id]
    if task_id == "sweep_blocks":
        names.insert(0, "sweep_objects")
    if task_id == "store_laptop_and_headphone":
        names.insert(0, "store_laptop_and_headphones")

    checked: list[Path] = []
    for name in names:
        task = root / name
        video = task / f"{view}.mp4"
        checked.append(video)
        if video.is_file():
            metadata = task / "metadata.json"
            if metadata.is_file():
                return video, _reference_frame_count(metadata, "num_frames")
            return video, _inspect_video_frame_count(video)

        video_key = {
            "front": "observation.images.cam_high",
            "wrist_left": "observation.images.cam_left_wrist",
            "wrist_right": "observation.images.cam_right_wrist",
        }.get(view, view)
        video = task / "videos" / video_key / "episode_000000.mp4"
        checked.append(video)
        if video.is_file():
            metadata = task / "meta" / "info.json"
            return video, _reference_frame_count(metadata, "total_frames")

    raise ConfigurationError(
        f"reference video not found for task {task_id!r}; checked={checked}"
    )


def _reference_frame_count(metadata: Path, field: str) -> int:
    if not metadata.is_file():
        raise ConfigurationError(f"reference metadata not found: {metadata}")
    try:
        row = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid reference metadata: {metadata}") from exc
    if field == "num_frames" and set(row) != {field}:
        raise ConfigurationError(
            f"reference metadata must contain only {field}: {metadata}"
        )
    num_frames = row.get(field)
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 2:
        raise ConfigurationError(
            f"reference {field} must be an integer of at least 2: {metadata}"
        )
    return num_frames


def _inspect_video_frame_count(video: Path) -> int:
    source = str(video)
    frame_count = 0
    try:
        import cv2

        capture = cv2.VideoCapture(source)
        if capture.isOpened():
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
    except Exception:
        pass
    if frame_count < 2:
        try:
            import imageio.v3 as imageio

            candidate = imageio.immeta(source).get("nframes")
            if isinstance(candidate, (int, float)) and candidate != float("inf"):
                frame_count = int(candidate)
        except Exception:
            pass
    if frame_count < 2:
        try:
            import imageio.v3 as imageio

            frame_count = sum(1 for _ in imageio.imiter(source))
        except Exception as exc:
            raise ConfigurationError(
                f"cannot inspect reference video: {video}"
            ) from exc
    if frame_count < 2:
        raise ConfigurationError(f"reference video is too short: {video}")
    return frame_count


def _cached_frame(source: str, index: int) -> Image.Image | None:
    key = (source, index)
    with _FRAME_CACHE_LOCK:
        cached = _FRAME_CACHE.pop(key, None)
        if cached is None:
            return None
        _FRAME_CACHE[key] = cached
        image, _size = cached
    return image.copy()


def _store_frame(source: str, index: int, image: Image.Image) -> None:
    global _FRAME_CACHE_BYTES

    size = image.width * image.height * len(image.getbands())
    if size > _FRAME_CACHE_BUDGET:
        return
    stored = image.copy()
    key = (source, index)
    with _FRAME_CACHE_LOCK:
        previous = _FRAME_CACHE.pop(key, None)
        if previous is not None:
            _FRAME_CACHE_BYTES -= previous[1]
        _FRAME_CACHE[key] = (stored, size)
        _FRAME_CACHE_BYTES += size
        while _FRAME_CACHE_BYTES > _FRAME_CACHE_BUDGET:
            _old_key, (_old_image, old_size) = _FRAME_CACHE.popitem(last=False)
            _FRAME_CACHE_BYTES -= old_size


def _close_reader(state: dict[str, Any]) -> None:
    capture = state.get("cv2_capture")
    if capture is not None:
        capture.release()
    container = state.get("av_container")
    if container is not None:
        container.close()


def _reader_state(source: str) -> dict[str, Any]:
    readers = getattr(_VIDEO_READER, "readers", None)
    if readers is None:
        readers = OrderedDict()
        _VIDEO_READER.readers = readers
    state = readers.pop(source, None)
    if state is not None:
        readers[source] = state
        return state
    state = {
        "source": source,
        "cv2_capture": None,
        "cv2_next": None,
        "av_container": None,
        "av_stream": None,
        "av_decoder": None,
        "av_position": None,
    }
    readers[source] = state
    while len(readers) > _VIDEO_READER_LIMIT:
        _old_source, old_state = readers.popitem(last=False)
        _close_reader(old_state)
    return state


def _read_cv2_frame(source: str, index: int, cv2: ModuleType) -> Image.Image | None:
    from PIL import Image

    state = _reader_state(source)
    capture = state["cv2_capture"]
    if capture is None:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            return None
        state["cv2_capture"] = capture
    if state["cv2_next"] != index:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    state["cv2_next"] = index + 1 if ok else None
    if not ok or bgr is None:
        return None
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _read_cv2_frames(
    source: str,
    indices: tuple[int, ...],
    cv2: ModuleType
) -> list[Image.Image] | None:
    """Decode strictly increasing frame indices in one forward CV2 pass."""
    from PIL import Image

    if not indices or any(left >= right for left, right in zip(indices, indices[1:])):
        return None
    state = _reader_state(source)
    capture = state["cv2_capture"]
    if capture is None:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            return None
        state["cv2_capture"] = capture
    capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    wanted = iter(indices)
    target = next(wanted)
    decoded = {}
    for position in range(indices[0], indices[-1] + 1):
        if position == target:
            ok, bgr = capture.read()
            if not ok or bgr is None:
                state["cv2_next"] = None
                return None
            decoded[position] = Image.fromarray(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            )
            target = next(wanted, None)
        elif not capture.grab():
            state["cv2_next"] = None
            return None
    state["cv2_next"] = indices[-1] + 1
    return [decoded[index] for index in indices]


def _read_pyav_frame(source: str, index: int, av: ModuleType) -> Image.Image:
    from PIL import Image

    state = _reader_state(source)
    container = state["av_container"]
    stream = state["av_stream"]
    if container is None:
        container = av.open(source)
        stream = container.streams.video[0]
        state["av_container"] = container
        state["av_stream"] = stream
    if stream.average_rate is not None and stream.time_base is not None:
        fps = float(stream.average_rate)
        decoder = state["av_decoder"]
        position = state["av_position"]
        if (
            decoder is None
            or position is None
            or index <= position
            or index - position > _PYAV_SEQUENTIAL_GAP
        ):
            target = int((index / fps) / float(stream.time_base))
            container.seek(target, stream=stream, backward=True)
            decoder = container.decode(stream)
            state["av_decoder"] = decoder
            state["av_position"] = None
        for frame in decoder:
            if frame.pts is None:
                continue
            current = round(float(frame.pts * stream.time_base) * fps)
            state["av_position"] = current
            if current >= index:
                return Image.fromarray(frame.to_rgb().to_ndarray())
    else:
        container.seek(0, stream=stream, backward=True)
        for current, frame in enumerate(container.decode(stream)):
            if current == index:
                return Image.fromarray(frame.to_rgb().to_ndarray())
    return None


def _read_imageio_frame(source: str, index: int) -> Image.Image:
    from PIL import Image
    import imageio.v3 as imageio

    if getattr(_IMAGEIO_CACHE, "source", None) != source:
        _IMAGEIO_CACHE.source = source
        _IMAGEIO_CACHE.frames = imageio.imread(source, index=None)
    frames = _IMAGEIO_CACHE.frames
    if index < 0 or index >= len(frames):
        raise VMBMKError(f"cannot read frame {index} from {source}")
    return Image.fromarray(frames[index])


def _requires_pyav(source: str) -> bool:
    if source not in _PYAV_ONLY:
        try:
            import av
        except ModuleNotFoundError:
            _PYAV_ONLY[source] = False
            return False

        with av.open(source) as container:
            codec = container.streams.video[0].codec_context.name
        _PYAV_ONLY[source] = codec in {"av1", "libdav1d", "libaom-av1"}
    return _PYAV_ONLY[source]


def read_frame(path: str | Path, index: int) -> Image.Image:
    source = str(path)
    if index < 0:
        raise VMBMKError(f"cannot read frame {index} from {source}")
    cached = _cached_frame(source, index)
    if cached is not None:
        return cached
    try:
        import cv2
    except ModuleNotFoundError:
        cv2 = None

    if cv2 is not None and not _requires_pyav(source):
        image = _read_cv2_frame(source, index, cv2)
        if image is not None:
            _store_frame(source, index, image)
            return image

    try:
        import av
    except ModuleNotFoundError:
        try:
            return _read_imageio_frame(source, index)
        except Exception as exc:
            raise VMBMKError(f"cannot read frame {index} from {source}") from exc

    from av.error import FFmpegError

    try:
        image = _read_pyav_frame(source, index, av)
    except (FFmpegError, IndexError) as exc:
        readers = getattr(_VIDEO_READER, "readers", {})
        state = readers.get(source)
        if state is not None:
            container = state.get("av_container")
            if container is not None:
                container.close()
            state["av_container"] = None
            state["av_stream"] = None
            state["av_decoder"] = None
            state["av_position"] = None
        raise VMBMKError(f"cannot open video {source}") from exc
    if image is not None:
        _store_frame(source, index, image)
        return image
    raise VMBMKError(f"cannot read frame {index} from {source}")


def read_frames(path: str | Path, indices: Sequence[int]) -> list[Image.Image]:
    """Read several frame indices, using one sequential CV2 decode when safe."""
    source = str(path)
    requested = tuple(indices)
    if any(index < 0 for index in requested):
        raise VMBMKError(f"cannot read negative frame indices from {source}")
    if not requested:
        return []

    resolved = {}
    missing = []
    missing_set = set()
    for index in requested:
        cached = _cached_frame(source, index)
        if cached is not None:
            resolved[index] = cached
        elif index not in missing_set:
            missing.append(index)
            missing_set.add(index)
    if missing:
        try:
            import cv2
        except ModuleNotFoundError:
            cv2 = None
        decoded = None
        ordered_missing = tuple(missing)
        if (
            cv2 is not None
            and not _requires_pyav(source)
            and all(left < right for left, right in zip(ordered_missing, ordered_missing[1:]))
        ):
            decoded = _read_cv2_frames(source, ordered_missing, cv2)
        if decoded is not None:
            for index, image in zip(ordered_missing, decoded):
                _store_frame(source, index, image)
                resolved[index] = image
        else:
            for index in ordered_missing:
                resolved[index] = read_frame(source, index)
    return [resolved[index] for index in requested]


def query_view(
    dataset: Dataset,
    query: ValueQuery | SubtaskQuery,
    view: str,
) -> PlaybackView:
    episode = dataset.episode(
        query.state.task_id,
        query.state.episode_id,
    )
    return PlaybackView(
        path=episode.video(view),
        num_frames=episode.num_frames,
        fps=episode.fps,
        playback=getattr(query, "playback", "forward"),
    )
