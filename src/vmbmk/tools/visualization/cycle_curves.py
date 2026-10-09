#!/usr/bin/env python3
"""Run Cycle-VOC on continuous forward-then-reverse prefixes and render curves.

The source video is not copied: a logical view maps the virtual timeline
0..T-1,T-2..0 back to source frames. Each descending video prefix therefore
retains the complete forward history before traversing backward.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from vmbmk.data.dataset import Dataset
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.inference.dispatch import run_inference
from vmbmk.metrics.cycle_voc_vs.cycle import (
    build_cycle_frame_indices,
    score_cycle_voc_values,
)
from vmbmk.inference.queries import StateRef, ValueQuery, read_results
from vmbmk.serialization import write_jsonl
from .svg_grid import grid_elements
from .value_curves import load_model_config


def _render_svg(
    path: Path,
    values: list[tuple[int, int, float, float]],
    *,
    task_id: str,
    episode_id: str,
    instruction: str,
    details: dict[str, float | None],
) -> None:
    width, height = 1200, 680
    left, right, top, bottom = 85, 35, 55, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    positions = [position for position, _, _, _ in values]
    scores = [score for _, _, score, _ in values]
    targets = [target for _, _, _, target in values]
    low = min(min(scores), min(targets))
    high = max(max(scores), max(targets))
    padding = max((high - low) * 0.08, 0.05)
    y_min, y_max = low - padding, high + padding
    max_position = max(positions, default=1)

    def x(position: int) -> float:
        return left + plot_width * position / max(max_position, 1)

    def y(score: float) -> float:
        return top + plot_height * (y_max - score) / (y_max - y_min)

    predicted_points = " ".join(
        f"{x(position):.2f},{y(score):.2f}"
        for position, _, score, _ in values
    )
    target_points = " ".join(
        f"{x(position):.2f},{y(target):.2f}"
        for position, _, _, target in values
    )
    title = html.escape(f"{task_id} / {episode_id} / Cycle-VOC")
    subtitle = html.escape(instruction)

    def format_rho(name: str) -> str:
        rho = details[name]
        if rho is None:
            return f"{name}=N/A (all-zero; score=0)"
        return f"{name}={rho:.4f}"

    cycle_score = details["cycle_voc"]
    assert cycle_score is not None
    summary = (
        f"{format_rho('rho_up')}  "
        f"{format_rho('rho_down')}  "
        f"cycle_voc={cycle_score:.4f}"
    )
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.5}.pred{fill:none;stroke:#2563eb;stroke-width:2.5}.target{fill:none;stroke:#dc2626;stroke-width:2;stroke-dasharray:7 5}.node{fill:#2563eb;stroke:white;stroke-width:1}.peak{stroke:#16a34a;stroke-width:1.5;stroke-dasharray:4 4}</style>',
        f'<text x="{left}" y="25" font-size="20" font-weight="600">{title}</text>',
        f'<text x="{left}" y="44" font-size="13" fill="#555">{subtitle}</text>',
        f'<text x="{left}" y="64" font-size="13" fill="#444">{html.escape(summary)}</text>',
    ]
    elements.extend(grid_elements(
        width, height, (left, right, top, bottom), (y_min, y_max),
        max_position,
    ))
    peak = max_position / 2
    elements.extend(
        (
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<line class="peak" x1="{x(peak):.2f}" y1="{top}" x2="{x(peak):.2f}" y2="{height - bottom}"/>',
            f'<polyline class="target" points="{target_points}"/>',
            f'<polyline class="pred" points="{predicted_points}"/>',
        )
    )
    for position, source_frame, score, target in values:
        elements.append(
            f'<circle class="node" cx="{x(position):.2f}" cy="{y(score):.2f}" r="4"><title>cycle {position}, source frame {source_frame}: value {score:.8g}, target {target:.8g}</title></circle>'
        )
    elements.extend(
        (
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 20}" font-size="14" text-anchor="middle">cycle_position (green = peak)</text>',
            f'<text x="22" y="{top + plot_height / 2:.2f}" font-size="14" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.2f})">value</text>',
            '<text x="1000" y="50" font-size="13" fill="#2563eb">prediction</text>',
            '<text x="1000" y="68" font-size="13" fill="#dc2626">ideal cycle</text>',
            "</svg>",
        )
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    data_root, model_config, gpu, output_root = load_model_config(config_path)
    dataset = Dataset.load(data_root)
    try:
        task = dataset.tasks[args.task]
    except KeyError as exc:
        raise ConfigurationError(f"unknown task: {args.task!r}") from exc
    if "VOC" not in task.metrics and "CYCLE-VOC" not in task.metrics:
        raise ConfigurationError(
            f"task {args.task!r} has no VOC annotation contract"
        )
    if args.all:
        if args.episodes:
            raise ConfigurationError("--all cannot be combined with episode IDs")
        episode_ids = tuple(
            sorted(
                episode_id
                for episode_id, episode in task.episodes.items()
                if episode.success
            )
        )
        if not episode_ids:
            raise ConfigurationError("task has no successful episodes")
    else:
        if not args.episodes:
            raise ConfigurationError("provide episode IDs or use --all")
        episode_ids = tuple(args.episodes)
    instruction = args.instruction or task.instruction
    queries: list[ValueQuery] = []
    points: list[tuple[str, int, int, float]] = []
    for episode_id in episode_ids:
        episode = dataset.episode(args.task, episode_id)
        if not episode.success:
            raise ConfigurationError(
                f"Cycle-VOC requires successful episodes: {args.task}/{episode_id}"
            )
        cycle_frames = build_cycle_frame_indices(episode.num_frames, args.stride)
        k = (len(cycle_frames) - 1) // 2
        for position, frame in enumerate(cycle_frames):
            target = min(position, 2 * k - position) / k
            points.append((episode_id, position, frame, target))
            # Descending queries retain the forward history and then traverse
            # backward to this source frame: S0..SK..Sj.
            playback = "forward" if position <= k else "cycle"
            queries.append(
                ValueQuery(
                    query_id=f"cycle_curve:{args.task}:{episode_id}:{position}",
                    state=StateRef(args.task, episode_id, frame),
                    instruction=instruction,
                    playback=playback,
                )
            )
    output = (
        Path(args.output).resolve()
        if args.output
        else output_root / f"{config_path.stem}_{args.task}_cycle_voc_curves"
    )
    if output.exists():
        raise ConfigurationError(f"output already exists: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise ConfigurationError(f"temporary output already exists: {temporary}")
    try:
        temporary.mkdir(parents=True)
        working = temporary / "_working"
        working.mkdir()
        query_path = working / "queries.jsonl"
        result_path = working / "results.jsonl"
        write_jsonl(query_path, (query.to_dict() for query in queries))
        run_inference(data_root, query_path, model_config, result_path, gpu=gpu)
        results = read_results(result_path)
        scores = [result.score for result in results]
        if len(scores) != len(points) or any(score is None for score in scores):
            raise VMBMKError("Cycle-VOC inference returned incomplete values")
        write_jsonl(
            temporary / "values.jsonl",
            (
                {
                    "episode_id": episode_id,
                    "cycle_position": position,
                    "source_frame": frame,
                    "target": target,
                    "score": score,
                }
                for (episode_id, position, frame, target), score in zip(points, scores)
            ),
        )
        curves = temporary / "curves"
        curves.mkdir()
        summaries: dict[str, Any] = {}
        for episode_id in episode_ids:
            curve = [
                (position, frame, float(score), target)
                for (value_episode, position, frame, target), score in zip(points, scores)
                if value_episode == episode_id
            ]
            details = score_cycle_voc_values([score for _, _, score, _ in curve])
            summaries[episode_id] = details
            _render_svg(
                curves / f"{episode_id}.svg",
                curve,
                task_id=args.task,
                episode_id=episode_id,
                instruction=instruction,
                details=details,
            )
        (temporary / "summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(working)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="sample a simple Cycle-VOC sequence and draw prediction curves"
    )
    parser.add_argument("--config", required=True, help="existing VMBMK run YAML")
    parser.add_argument("task", help="task ID")
    parser.add_argument("episodes", nargs="*", help="successful episode IDs")
    parser.add_argument("--all", action="store_true", help="process every episode")
    parser.add_argument("--stride", type=int, default=10, help="sample every N frames")
    parser.add_argument("--instruction", help="override the dataset instruction")
    parser.add_argument("--output", help="override the output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stride <= 0:
        parser.error("stride must be a positive integer")
    try:
        print(run(args))
        return 0
    except (OSError, ValueError, VMBMKError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
