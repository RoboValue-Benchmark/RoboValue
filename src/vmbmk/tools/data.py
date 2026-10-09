"""Read-only checks of dataset video metadata and annotated frame bounds."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _video_frames(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    output = subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout
    streams = json.loads(output).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected one video stream")
    for key in ("nb_read_frames", "nb_frames"):
        value = streams[0].get(key)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    raise ValueError(f"{path}: ffprobe did not report a frame count")


def _audit_video(item: tuple[Path, str, Path]) -> dict[str, Any]:
    episode_root, view, video = item
    metadata = _read_json(episode_root / "metadata.json")
    annotation = _read_json(episode_root / "annotation.json")
    declared = int(metadata["num_frames"])
    annotated = [
        int(row["frame_index"])
        for field in ("voc", "voc_mem")
        for row in annotation.get(field, [])
    ]
    maximum = max(annotated, default=-1)
    actual = _video_frames(video)
    unsafe = []
    if declared > actual:
        unsafe.append("metadata_exceeds_video")
    if maximum >= actual:
        unsafe.append("annotation_exceeds_video")
    return {
        "task": episode_root.parent.parent.name,
        "episode": episode_root.name,
        "view": view,
        "declared_frames": declared,
        "actual_frames": actual,
        "max_annotated_frame": maximum,
        "status": unsafe or (["video_longer_than_metadata"] if declared < actual else ["ok"]),
    }


def selected_tasks(dataset: Path, requested: list[str] | None) -> list[str]:
    """Select existing task directories, preserving an explicit selection order."""
    available = sorted(path.name for path in dataset.iterdir() if path.is_dir())
    tasks = available if requested is None else list(dict.fromkeys(requested))
    unknown = sorted(set(tasks) - set(available))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    if not tasks:
        raise ValueError(f"no tasks found in {dataset}")
    return tasks


def audit_frames(
    dataset: Path, tasks: list[str] | None = None,
    views: list[str] | None = None, workers: int = 16,
) -> dict[str, Any]:
    """Check declared and VOC/MEM-VOC annotated frame bounds against video assets."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    selected = selected_tasks(dataset, tasks)
    items = []
    for task in selected:
        for episode_root in sorted((dataset / task / "episodes").iterdir()):
            if not episode_root.is_dir():
                continue
            videos = _read_json(episode_root / "metadata.json")["assets"]["videos"]
            requested_views = sorted(videos) if views is None else list(dict.fromkeys(views))
            unknown = sorted(set(requested_views) - set(videos))
            if unknown:
                raise ValueError(f"{episode_root}: unknown views {unknown}")
            for view in requested_views:
                video = (episode_root / videos[view]).resolve()
                if not video.is_relative_to(episode_root.resolve()):
                    raise ValueError(f"video asset escapes episode directory: {video}")
                items.append((episode_root, view, video))
    if not items:
        raise ValueError("selection has no video assets")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = sorted(executor.map(_audit_video, items), key=lambda row: (
            row["task"], row["episode"], row["view"],
        ))
    unsafe_labels = {"metadata_exceeds_video", "annotation_exceeds_video"}
    unsafe = [row for row in rows if unsafe_labels.intersection(row["status"])]
    mismatches = [row for row in rows if row["status"] != ["ok"]]
    return {
        "tasks": selected,
        "episodes": len({(row["task"], row["episode"]) for row in rows}),
        "videos": len(rows), "unsafe_count": len(unsafe),
        "mismatch_count": len(mismatches), "unsafe": unsafe, "mismatches": mismatches,
        "annotation_fields_checked": ["voc", "voc_mem"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frames = commands.add_parser("check-frames", help="Audit video frame bounds without model inference.")
    frames.add_argument("dataset", type=Path)
    frames.add_argument("--tasks", "--task", nargs="+")
    frames.add_argument("--views", nargs="+")
    frames.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    report = audit_frames(args.dataset.resolve(), args.tasks, args.views, args.workers)
    print(json.dumps(report, indent=2))
    return 1 if report["unsafe_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
