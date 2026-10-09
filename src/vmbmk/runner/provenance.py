"""Local run identities for safe reuse of completed and interrupted results."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from vmbmk.data.dataset import Dataset


def run_identity(dataset: Dataset) -> dict[str, Any]:
    """Fingerprint package code, dataset metadata and asset size/mtime identities."""
    package = Path(__file__).resolve().parents[1]
    code = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        code.update(path.relative_to(package).as_posix().encode())
        code.update(path.read_bytes())
    data = hashlib.sha256()
    scanned_assets: set[Path] = set()
    for task_id, task in sorted(dataset.tasks.items()):
        task_root = dataset.root / task_id
        paths = [task_root / "metadata.json"]
        for episode_id in sorted(task.episodes):
            paths.extend(sorted((task_root / "episodes" / episode_id).rglob("*")))
        for path in paths:
            if not path.is_file():
                continue
            data.update(path.relative_to(dataset.root).as_posix().encode())
            if path.suffix == ".json":
                data.update(path.read_bytes())
            else:
                scanned_assets.add(path.resolve())
                stat = path.stat()
                data.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    videos = {
        video.resolve()
        for task in dataset.tasks.values()
        for episode in task.episodes.values()
        for video in episode.videos.values()
    }
    for video in sorted(videos - scanned_assets):
        data.update(str(video).encode())
        stat = video.stat()
        data.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return {
        "protocol": "local-run-identity-v2",
        "code_sha256": code.hexdigest(),
        "dataset_sha256": data.hexdigest(),
        "dataset_root": str(dataset.root),
        "asset_identity": "size-and-mtime-not-content-hash",
    }
