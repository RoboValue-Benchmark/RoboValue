from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def set_task_metadata(
    data: Path, metrics: list[str], subtasks: list[dict[str, Any]] | None = None,
) -> Path:
    """Set one synthetic task's metric declarations and subtask vocabulary."""
    task = data / "task_001"
    path = task / "metadata.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["supports"]["metrics"] = metrics
    row["subtasks"] = subtasks or []
    write_json(path, row)
    return task


def make_dataset(root: Path, episodes: Sequence[tuple[str, bool]] | None = None) -> Path:
    task = root / "task_001"
    write_json(
        task / "metadata.json",
        {
            "task_id": "task_001",
            "instruction": "Put the cup on the tray",
            "supports": {"metrics": []},
            "subtasks": [],
        },
    )
    rows = episodes or [
        ("ep_success", True),
        ("ep_failure", False),
    ]
    for episode_id, success in rows:
        episode = task / "episodes" / episode_id
        (episode / "front.mp4").parent.mkdir(parents=True, exist_ok=True)
        for name in ("front.mp4", "wrist_left.mp4", "wrist_right.mp4", "robot.parquet"):
            (episode / name).write_bytes(b"fixture")
        write_json(
            episode / "metadata.json",
            {
                "episode_id": episode_id,
                "world_type": "real",
                "domain": "id",
                "success": success,
                "fps": 10,
                "num_frames": 100,
                "assets": {
                    "videos": {
                        "front": "front.mp4",
                        "wrist_left": "wrist_left.mp4",
                        "wrist_right": "wrist_right.mp4",
                    },
                    "robot_data": {"uri": "robot.parquet"},
                },
            },
        )
        write_json(episode / "annotation.json", {})
    return root
