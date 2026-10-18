"""Local run identities for safe reuse of completed and interrupted results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from vmbmk.data.dataset import Dataset
from .config_identity import normalized_result_config


def model_identity(config: Mapping[str, Any], *, data_root: Path | None = None) -> str:
    """Fingerprint inference config and local checkpoint/reference size/mtime."""
    model = normalized_result_config(config)
    for key in ("sia_base_url", "sia_model", "sia_seed", "sia_workers", "sia_timeout"):
        model.pop(key, None)
    reference = model.get("reference_data")
    if reference is None and data_root is not None:
        from vmbmk.adapters.registry import adapter_class

        adapter = adapter_class(model["adapter"])
        if hasattr(adapter, "resolve_shot_mode"):
            mode = adapter.resolve_shot_mode(model)
        else:
            mode = model.get("shot_mode", getattr(adapter, "default_shot_mode", None))
        if mode == "one_shot":
            reference = data_root / "reference"
    asset_roots = [
        model.get("checkpoint"), reference,
        *(model.get("checkpoint_by_task") or {}).values(),
    ]
    assets = []
    for asset_root in asset_roots:
        if asset_root is None:
            continue
        path = Path(str(asset_root))
        if path.exists():
            paths = path.rglob("*") if path.is_dir() else [path]
            for asset in paths:
                if asset.is_file():
                    stat = asset.stat()
                    assets.append((str(asset.resolve()), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(
        {"config": model, "assets": sorted(assets)},
        sort_keys=True, ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def run_identity(
    dataset: Dataset, model: Mapping[str, Any] | None = None,
    *, sia_responses: Path | None = None,
) -> dict[str, Any]:
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
        "protocol": "local-run-identity-v3",
        "code_sha256": code.hexdigest(),
        "dataset_sha256": data.hexdigest(),
        "dataset_root": str(dataset.root),
        "asset_identity": "size-and-mtime-not-content-hash",
        **({"model_sha256": model_identity(model, data_root=dataset.root)} if model is not None else {}),
        **({"sia_responses_sha256": hashlib.sha256(sia_responses.read_bytes()).hexdigest()}
           if sia_responses is not None else {}),
    }
