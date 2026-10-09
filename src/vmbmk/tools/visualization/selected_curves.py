#!/usr/bin/env python3
"""Render value curves for a reusable selection of task trajectories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .value_curves import (
    load_model_config,
    _render_svg,
    _sample_frames,
)

from vmbmk.data.dataset import Dataset
from vmbmk.runner.run import RunConfig
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.inference.dispatch import run_inference
from vmbmk.inference.queries import StateRef, ValueQuery, read_results
from vmbmk.serialization import write_jsonl


def _load_selection(path: Path) -> tuple[int, list[dict[str, Any]]]:
    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ConfigurationError(f"{path}: selection must be an object")
    stride = row.get("stride_frames")
    trajectories = row.get("trajectories")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ConfigurationError(f"{path}.stride_frames must be positive")
    if not isinstance(trajectories, list) or not trajectories:
        raise ConfigurationError(f"{path}.trajectories must be a non-empty list")
    return stride, trajectories



def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    selection_path = Path(args.selection).resolve()
    config = RunConfig.load(config_path)
    data_root, model_config, config_gpu, output_root = load_model_config(config)
    gpu = config_gpu if args.gpu is None else args.gpu
    if args.batch_size is not None:
        model_config["batch_size"] = args.batch_size
    if args.max_new_tokens is not None:
        model_config["max_new_tokens"] = args.max_new_tokens
    dataset = Dataset.load(data_root)
    stride, selected = _load_selection(selection_path)
    allowed_tasks = set(config.tasks)
    native_metric = args.native_metric

    points: list[tuple[str, str, int, str | None, str | None]] = []
    queries: list[ValueQuery] = []
    instructions: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise ConfigurationError(f"{selection_path}.trajectories[{index}] is invalid")
        task_id = item.get("task")
        episode_id = item.get("episode")
        domain = item.get("domain")
        st = item.get("st")
        if not isinstance(task_id, str) or not isinstance(episode_id, str):
            raise ConfigurationError(
                f"{selection_path}.trajectories[{index}] needs task and episode"
            )
        if task_id not in allowed_tasks:
            continue
        key = (task_id, episode_id)
        if key in seen:
            raise ConfigurationError(f"duplicate selected trajectory: {key}")
        seen.add(key)
        task = dataset.tasks[task_id]
        episode = dataset.episode(task_id, episode_id)
        instructions[task_id] = task.instruction
        for frame in _sample_frames(episode.num_frames, stride):
            points.append((task_id, episode_id, frame, domain, st))
            queries.append(
                ValueQuery(
                    query_id=f"curve:{task_id}:{episode_id}:{frame}",
                    state=StateRef(task_id, episode_id, frame),
                    instruction=task.instruction,
                )
            )
    if not queries:
        raise ConfigurationError("selection has no trajectories allowed by the config")

    output = (
        Path(args.output).resolve()
        if args.output
        else output_root / "value_curves_5hz" / config_path.stem
    )
    if output.exists():
        raise ConfigurationError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    if temporary_output.exists():
        raise ConfigurationError(f"temporary output already exists: {temporary_output}")

    try:
        temporary_output.mkdir()
        with tempfile.TemporaryDirectory(prefix="vmbmk-selected-curves-") as directory:
            working = Path(directory)
            query_path = working / "queries.jsonl"
            result_path = working / "results.jsonl"
            write_jsonl(query_path, (query.to_dict() for query in queries))
            run_inference(
                data_root,
                query_path,
                model_config,
                result_path,
                gpu=gpu,
                native_metric=native_metric,
            )
            results = read_results(result_path)
        values = [(*point, result.score) for point, result in zip(points, results)]
        task_ids = sorted({task_id for task_id, _ in seen})
        for task_id in task_ids:
            task_output = temporary_output / task_id
            task_output.mkdir()
            write_jsonl(
                task_output / "values.jsonl",
                (
                    {
                        "episode_id": episode_id,
                        "frame_index": frame,
                        "score": score,
                    }
                    for value_task, episode_id, frame, _, _, score in values
                    if value_task == task_id
                ),
            )
            (task_output / "curves").mkdir()
        for task_id, episode_id in sorted(seen):
            _render_svg(
                temporary_output / task_id / "curves" / f"{episode_id}.svg",
                [
                    (frame, score)
                    for value_task, value_episode, frame, _, _, score in values
                    if value_task == task_id and value_episode == episode_id
                ],
                task_id=task_id,
                episode_id=episode_id,
                instruction=instructions[task_id],
            )
        metadata = {
            "config": str(config_path),
            "selection": str(selection_path),
            "gpu": gpu,
            "batch_size": model_config["batch_size"],
            "max_new_tokens": model_config.get("max_new_tokens"),
            "inference_mode": "native" if native_metric else "value-point",
            "stride_frames": stride,
            "trajectories": len(seen),
            "sample_points": len(values),
        }
        (temporary_output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_output, output)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--native-metric", choices=("voc", "voc_mem", "sa"),
                        help="explicit native adapter interface; omitted uses value queries")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be a non-negative integer")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be a positive integer")
    try:
        print(run(args))
        return 0
    except (OSError, ValueError, VMBMKError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
