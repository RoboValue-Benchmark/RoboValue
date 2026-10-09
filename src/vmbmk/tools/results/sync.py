#!/usr/bin/env python3
"""Synchronize one result file into the stable ``final_results`` directory.

The normal result layout is a run directory containing ``metrics.json`` and
``config.yaml``.  Such a file is published using the same normalized layout as
the evaluation runner (``<baseline>/<metric>/``).  Files without the run
metadata are copied verbatim while preserving their path relative to the
results root.

Examples::

    # A normal run: publish metrics and overwrite the existing final result.
    python -m vmbmk.cli results publish \
        results/my_run/metrics.json

    # An arbitrary result artifact: mirror it under final_results.
    python -m vmbmk.cli results publish \
        results/my_run/trace.json \
        --mode mirror

Use ``--mode publish`` to require the normalized publication path, or
``--mode mirror`` to always copy the input file verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vmbmk.metrics.registry import canonical_metric
from vmbmk.adapters.registry import canonical_baseline


DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_FINAL_RESULTS_ROOT = Path("final_results")


def _sia_judge_baseline(config: dict[str, Any], baseline: str, metrics: set[str]) -> str:
    """Add the SIA judge model to the stable baseline identity.

    SIA scores are judge-dependent, so publishing them under the plain model
    name would make results from different judge base models overwrite one
    another.  Keep the suffix filesystem-safe and deterministic.
    """
    if "sia" not in metrics:
        return baseline
    options = config.get("model_options")
    judge = options.get("sia_model") if isinstance(options, dict) else None
    if judge is None:
        judge = config.get("sia_model")
    judge = re.sub(r"[^A-Za-z0-9._-]+", "-", str(judge or "unknown"))
    return f"{baseline}__sia-judge-{judge}"


@dataclass(frozen=True)
class SyncResult:
    """Description of one synchronization operation."""

    mode: str
    source: Path
    destinations: tuple[Path, ...]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON result {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_yaml_object(path: Path) -> dict[str, Any]:
    from vmbmk.serialization import read_yaml_object

    return read_yaml_object(path)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy one file and atomically replace an existing destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        # ``copy2`` can fail before the temporary file is replaced.
        temporary.unlink(missing_ok=True)


def _mirror_destination(source: Path, results_root: Path, final_root: Path) -> Path:
    """Return the safe mirror destination for ``source``."""

    try:
        relative = source.relative_to(results_root)
    except ValueError:
        # Absolute paths outside results are still supported, but never use
        # their absolute components as destination path components.
        relative = Path(source.name)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"invalid relative result path: {relative}")
    destination = (final_root / relative).resolve()
    final_root_resolved = final_root.resolve()
    if destination != final_root_resolved and final_root_resolved not in destination.parents:
        raise ValueError(f"refusing to write outside final-results root: {destination}")
    return destination


def _run_metadata(source: Path) -> tuple[Path, Path] | None:
    """Find ``(run_directory, config_path)`` for a metrics result file."""

    if source.is_dir():
        run_directory = source
        metrics_path = run_directory / "metrics.json"
    else:
        run_directory = source.parent
        metrics_path = source
    if metrics_path.name != "metrics.json" or not metrics_path.is_file():
        return None
    config_path = run_directory / "config.yaml"
    if not config_path.is_file():
        return None
    return run_directory, config_path


def _resolve_source_path(result_path: Path, results_root: Path) -> Path:
    """Resolve a convenient relative path such as ``results/run/metrics.json``."""

    candidate = result_path.expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    parts = candidate.parts
    if parts and parts[0] == results_root.name:
        candidate = results_root.joinpath(*parts[1:])
    else:
        candidate = results_root / candidate
    return candidate.resolve()


def _publish_metrics(
    metrics_path: Path,
    config_path: Path,
    final_root: Path,
    *,
    baseline_override: str | None = None,
    merge_existing: bool = False,
) -> tuple[Path, ...]:
    """Publish one metrics/config pair in the canonical final-results tree."""

    metrics = _read_json_object(metrics_path)
    config = _read_yaml_object(config_path)
    baseline = str(
        baseline_override
        or canonical_baseline(config, fallback=metrics_path.parent.name)
    )
    from .publication import PublicationRequest, publish_result

    # Preserve explicit legacy F1 keys rather than relabelling them as VOC.
    # Normalize result keys and configured task maps consistently.
    normalized_metrics: dict[str, Any] = {}
    for raw_metric, metric_result in metrics.items():
        metric = canonical_metric(raw_metric)
        # If both aliases are present, prefer the canonical key.
        if metric in normalized_metrics and raw_metric != metric:
            continue
        normalized_metrics[metric] = metric_result
    baseline = _sia_judge_baseline(config, baseline, set(normalized_metrics))

    job = PublicationRequest(
        baseline=baseline,
        config_path=config_path,
        run_name=metrics_path.parent.name,
        result=normalized_metrics,
    )
    publish_result(job, final_root, merge_existing=merge_existing)

    return tuple(
        final_root / baseline / canonical_metric(str(metric)) / "summary.json"
        for metric in normalized_metrics
    )


def synchronize(
    result_path: Path,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    final_root: Path = DEFAULT_FINAL_RESULTS_ROOT,
    mode: str = "auto",
    baseline: str | None = None,
    merge_existing: bool = False,
) -> SyncResult:
    """Synchronize ``result_path`` and return the written destinations.

    ``result_path`` may be a single file or a run directory.  A directory is
    accepted as a convenience and resolves to its ``metrics.json`` file.
    """

    result_path = Path(result_path)
    results_root = Path(results_root).expanduser().resolve()
    final_root = Path(final_root).expanduser().resolve()
    source = _resolve_source_path(result_path, results_root)
    if not source.exists():
        raise FileNotFoundError(f"result path does not exist: {source}")
    if source.is_dir():
        metadata = _run_metadata(source)
        if metadata is None:
            raise ValueError(f"result directory must contain metrics.json and config.yaml: {source}")
        source = metadata[0] / "metrics.json"
    if not source.is_file():
        raise ValueError(f"result path is not a regular file: {source}")
    if mode not in {"auto", "publish", "mirror"}:
        raise ValueError(f"unsupported mode {mode!r}; choose auto, publish, or mirror")

    metadata = _run_metadata(source)
    should_publish = mode == "publish" or (mode == "auto" and metadata is not None)
    if should_publish:
        if metadata is None:
            raise ValueError(
                "publish mode requires a metrics.json next to its run's config.yaml"
            )
        destinations = _publish_metrics(
            source,
            metadata[1],
            final_root,
            baseline_override=baseline,
            merge_existing=merge_existing,
        )
        return SyncResult("publish", source, destinations)

    destination = _mirror_destination(source, results_root, final_root)
    if destination == source:
        raise ValueError("source and destination are the same file")
    _atomic_copy(source, destination)
    return SyncResult("mirror", source, (destination,))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_path",
        type=Path,
        help="result file, or a run directory containing metrics.json/config.yaml",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--final-results-root",
        "--final-root",
        "--finalresults-root",
        type=Path,
        default=DEFAULT_FINAL_RESULTS_ROOT,
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "publish", "mirror"),
        default="auto",
        help="auto publishes complete runs and mirrors other files (default: auto)",
    )
    parser.add_argument(
        "--baseline",
        help="override the model name used by publish mode",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge task-level output into an existing metric summary",
    )
    args = parser.parse_args(argv)
    try:
        result = synchronize(
            args.result_path,
            results_root=args.results_root,
            final_root=args.final_results_root,
            mode=args.mode,
            baseline=args.baseline,
            merge_existing=args.merge,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"synchronized ({result.mode}): {result.source}")
    for destination in result.destinations:
        print(f"  -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
