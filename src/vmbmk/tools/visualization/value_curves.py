#!/usr/bin/env python3
"""Sample one episode with the value interface and render a score curve."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from vmbmk.data.dataset import Dataset
from vmbmk.runner.run import RunConfig
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.inference.dispatch import run_inference
from vmbmk.inference.queries import StateRef, ValueQuery, read_results
from vmbmk.serialization import write_jsonl
from .svg_grid import grid_elements


def load_model_config(
    source: Path | RunConfig,
) -> tuple[Path, dict[str, Any], int | tuple[int, ...], Path]:
    config = source if isinstance(source, RunConfig) else RunConfig.load(source)
    if config.model == "failsafe":
        raise ConfigurationError("FailSafe has no value interface for curves")
    return config.data, config.model_config(), config.gpu, config.output


def _sample_frames(num_frames: int, stride: int) -> list[int]:
    frames = list(range(0, num_frames, stride))
    if frames[-1] != num_frames - 1:
        frames.append(num_frames - 1)
    return frames


def _render_svg(
    path: Path,
    values: list[tuple[int, float]],
    *,
    task_id: str,
    episode_id: str,
    instruction: str,
) -> None:
    width, height = 1200, 640
    left, right, top, bottom = 85, 35, 55, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_frame = max(frame for frame, _ in values)
    scores = [score for _, score in values]
    low, high = min(scores), max(scores)
    if low == high:
        padding = max(abs(low) * 0.05, 0.5)
    else:
        padding = (high - low) * 0.08
    y_min, y_max = low - padding, high + padding

    def x(frame: int) -> float:
        return left + plot_width * frame / max(max_frame, 1)

    def y(score: float) -> float:
        return top + plot_height * (y_max - score) / (y_max - y_min)

    points = " ".join(f"{x(frame):.2f},{y(score):.2f}" for frame, score in values)
    title = html.escape(f"{task_id} / {episode_id}")
    subtitle = html.escape(instruction)
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.5}.curve{fill:none;stroke:#2563eb;stroke-width:2.5}.node{fill:#2563eb;stroke:white;stroke-width:1}</style>',
        f'<text x="{left}" y="25" font-size="20" font-weight="600">{title}</text>',
        f'<text x="{left}" y="45" font-size="13" fill="#555">{subtitle}</text>',
    ]
    elements.extend(grid_elements(
        width, height, (left, right, top, bottom), (y_min, y_max),
        max_frame,
    ))
    elements.extend(
        (
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<polyline class="curve" points="{points}"/>',
        )
    )
    for frame, score in values:
        elements.append(
            f'<circle class="node" cx="{x(frame):.2f}" cy="{y(score):.2f}" r="4"><title>frame {frame}: {score:.8g}</title></circle>'
        )
    elements.extend(
        (
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 20}" font-size="14" text-anchor="middle">frame_index</text>',
            f'<text x="22" y="{top + plot_height / 2:.2f}" font-size="14" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.2f})">value</text>',
            "</svg>",
        )
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    data_root, model_config, gpu, output_root = (
        load_model_config(config_path)
    )
    if args.gpu is not None:
        gpu = args.gpu
    dataset = Dataset.load(data_root)
    task_id = args.task
    try:
        task = dataset.tasks[task_id]
    except KeyError as exc:
        raise ConfigurationError(f"unknown task: {task_id!r}") from exc
    if args.all:
        if args.episodes:
            raise ConfigurationError("--all cannot be combined with episode IDs")
        episode_ids = tuple(sorted(task.episodes))
    else:
        if not args.episodes:
            raise ConfigurationError("provide episode IDs or use --all")
        if len(args.episodes) != len(set(args.episodes)):
            raise ConfigurationError("episode IDs contain duplicates")
        episode_ids = tuple(args.episodes)
    instruction = args.instruction or task.instruction
    points: list[tuple[str, int]] = []
    queries: list[ValueQuery] = []
    for episode_id in episode_ids:
        episode = dataset.episode(task_id, episode_id)
        for frame in _sample_frames(episode.num_frames, args.stride):
            points.append((episode_id, frame))
            queries.append(
                ValueQuery(
                    query_id=f"curve:{task_id}:{episode_id}:{frame}",
                    state=StateRef(task_id, episode_id, frame),
                    instruction=instruction,
                )
            )

    output = (
        Path(args.output).resolve()
        if args.output
        else output_root / f"{config_path.stem}_{task_id}_value_curves"
    )
    if output.exists():
        raise ConfigurationError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    if temporary_output.exists():
        raise ConfigurationError(f"temporary output already exists: {temporary_output}")

    try:
        temporary_output.mkdir()
        with tempfile.TemporaryDirectory(prefix="vmbmk-value-curve-") as directory:
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
            )
            results = read_results(result_path)
        values = [
            (episode_id, frame, result.score)
            for (episode_id, frame), result in zip(points, results)
        ]
        write_jsonl(
            temporary_output / "values.jsonl",
            (
                {
                    "episode_id": episode_id,
                    "frame_index": frame,
                    "score": score,
                }
                for episode_id, frame, score in values
            ),
        )
        curves = temporary_output / "curves"
        curves.mkdir()
        for episode_id in episode_ids:
            _render_svg(
                curves / f"{episode_id}.svg",
                [
                    (frame, score)
                    for value_episode, frame, score in values
                    if value_episode == episode_id
                ],
                task_id=task_id,
                episode_id=episode_id,
                instruction=instruction,
            )
        os.replace(temporary_output, output)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="sample trajectories with value() and draw score curves"
    )
    parser.add_argument("--config", required=True, help="existing VMBMK run YAML")
    parser.add_argument("task", help="task ID")
    parser.add_argument(
        "stride",
        type=int,
        help="sample every N frames; the final frame is always included",
    )
    parser.add_argument("episodes", nargs="*", help="episode IDs")
    parser.add_argument(
        "--all",
        action="store_true",
        help="process every episode in the task",
    )
    parser.add_argument("--instruction", help="override the dataset instruction")
    parser.add_argument("--output", help="override the automatic output directory")
    parser.add_argument(
        "--gpu",
        type=int,
        help="override the GPU index from the run config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stride <= 0:
        parser.error("stride must be a positive integer")
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be a non-negative integer")
    try:
        print(run(args))
        return 0
    except (OSError, ValueError, VMBMKError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
