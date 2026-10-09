from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from vmbmk.data.dataset import Dataset
from vmbmk.adapters.registry import adapter_class
from vmbmk.errors import ConfigurationError, VMBMKError
from .queries import (
    CompareQuery,
    Query,
    Result,
    StateRef,
    SubtaskQuery,
    ValueQuery,
    read_queries,
    read_results,
)
from vmbmk.serialization import read_jsonl, replace_jsonl, write_jsonl


def _validate_state(dataset: Dataset, query_id: str, state: StateRef) -> None:
    episode = dataset.episode(state.task_id, state.episode_id)
    if not 0 <= state.anchor_frame < episode.num_frames:
        raise VMBMKError(
            f"{query_id}: frame {state.anchor_frame} is outside "
            f"[0, {episode.num_frames})"
        )


def validate_queries(dataset: Dataset, queries: list[Query]) -> None:
    for query in queries:
        if isinstance(query, ValueQuery):
            _validate_state(dataset, query.query_id, query.state)
        elif isinstance(query, CompareQuery):
            if query.state_a.task_id != query.state_b.task_id:
                raise VMBMKError(
                    f"{query.query_id}: compare states must belong to the same task"
                )
            _validate_state(dataset, query.query_id, query.state_a)
            _validate_state(dataset, query.query_id, query.state_b)
        elif isinstance(query, SubtaskQuery):
            _validate_state(dataset, query.query_id, query.state)


def validate_results(queries: list[Query], result_path: str | Path) -> None:
    expected = {query.query_id: query.op for query in queries}
    seen: set[str] = set()
    extra: list[str] = []
    mismatched: list[str] = []
    for index, row in enumerate(read_jsonl(result_path), start=1):
        result = Result.from_dict(row, f"{result_path}:{index}")
        if result.query_id in seen:
            raise VMBMKError("results contain duplicate query_id values")
        seen.add(result.query_id)
        expected_op = expected.get(result.query_id)
        if expected_op is None:
            extra.append(result.query_id)
        elif expected_op != result.op:
            mismatched.append(result.query_id)
    missing = sorted(set(expected) - seen)
    extra.sort()
    mismatched.sort()
    if missing or extra or mismatched:
        raise VMBMKError(
            "result coverage mismatch; "
            f"missing={missing}, extra={extra}, op_mismatch={mismatched}"
        )


def _normalize_gpus(gpu: int | Sequence[int] | None) -> tuple[int, ...]:
    values = (gpu,) if isinstance(gpu, int) and not isinstance(gpu, bool) else gpu
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in values
        )
    ):
        raise ConfigurationError(
            "gpu must be a non-negative integer or a non-empty array of them"
        )
    selected = tuple(values)
    if len(selected) != len(set(selected)):
        raise ConfigurationError("gpu array contains duplicates")
    return selected


def _episode_refs(query: Query) -> tuple[tuple[str, str], ...]:
    if isinstance(query, CompareQuery):
        return tuple(
            sorted(
                {
                    (query.state_a.task_id, query.state_a.episode_id),
                    (query.state_b.task_id, query.state_b.episode_id),
                }
            )
        )
    return ((query.state.task_id, query.state.episode_id),)


def _partition_queries(queries: list[Query], count: int) -> list[list[Query]]:
    """Balance episode-connected query groups without splitting video locality."""
    parent: dict[tuple[str, str], tuple[str, str]] = {}
    episode_refs = {query.query_id: _episode_refs(query) for query in queries}

    def find(item: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for query in queries:
        refs = episode_refs[query.query_id]
        for ref in refs[1:]:
            union(refs[0], ref)
        find(refs[0])

    grouped: dict[tuple[str, str], list[Query]] = defaultdict(list)
    for query in queries:
        grouped[find(episode_refs[query.query_id][0])].append(query)

    order = {query.query_id: index for index, query in enumerate(queries)}
    groups = sorted(
        grouped.values(),
        key=lambda rows: (
            -sum(len(episode_refs[query.query_id]) for query in rows),
            min(order[query.query_id] for query in rows),
        ),
    )
    shard_count = min(count, len(groups))
    shards: list[list[Query]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for rows in groups:
        target = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[target].extend(rows)
        loads[target] += sum(len(episode_refs[query.query_id]) for query in rows)
    for shard in shards:
        shard.sort(key=lambda query: order[query.query_id])
    return shards


def _run_worker_process(
    data_root: str | Path,
    query_path: str | Path,
    config: Mapping[str, Any],
    result_path: Path,
    queries: list[Query],
    gpu: int | None,
    native_metric: str | None,
) -> None:
    if result_path.exists():
        try:
            validate_results(queries, result_path)
            return
        except (ValueError, VMBMKError):
            # The worker validates and resumes an incomplete result prefix.
            pass

    python = Path(str(config.get("python", "")))
    if not python.is_file():
        raise ConfigurationError(f"configured python does not exist: {python}")
    if adapter_class(config["adapter"]).requires_checkpoint:
        checkpoint = Path(str(config.get("checkpoint", "")))
        if not config.get("checkpoint") or not checkpoint.exists():
            raise ConfigurationError(f"configured checkpoint does not exist: {checkpoint}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    source_roots = [str(Path(__file__).resolve().parents[2])]
    adapter_source = config.get("source_root")
    if adapter_source is not None:
        adapter_source_path = Path(str(adapter_source)).resolve()
        if not adapter_source_path.is_dir():
            raise ConfigurationError(
                f"configured source_root does not exist: {adapter_source_path}"
            )
        source_roots.append(str(adapter_source_path))
    current = environment.get("PYTHONPATH")
    if current:
        source_roots.append(current)
    environment["PYTHONPATH"] = os.pathsep.join(source_roots)
    with tempfile.TemporaryDirectory(prefix="vmbmk-config-") as directory:
        config_path = Path(directory) / "config.json"
        config_path.write_text(
            json.dumps(dict(config), ensure_ascii=False),
            encoding="utf-8",
        )
        command = [
            str(python),
            "-m",
            "vmbmk.inference.worker",
            "--data",
            str(Path(data_root).resolve()),
            "--queries",
            str(Path(query_path).resolve()),
            "--config",
            str(config_path),
            "--output",
            str(result_path),
        ]
        if native_metric is not None:
            command.extend(("--native-metric", native_metric))
        completed = subprocess.run(command, env=environment)
        if completed.returncode != 0:
            device = "CPU" if gpu is None else f"gpu {gpu}"
            raise VMBMKError(
                f"adapter worker on {device} failed with exit code "
                f"{completed.returncode}"
            )
        validate_results(queries, result_path)


def _read_resume_results(
    queries: list[Query],
    paths: Sequence[Path],
) -> dict[str, Result]:
    """Read valid partial results without requiring complete coverage."""
    expected = {query.query_id: query.op for query in queries}
    results: dict[str, Result] = {}
    for path in paths:
        for result in read_results(path):
            expected_op = expected.get(result.query_id)
            if expected_op is None:
                raise VMBMKError(
                    f"resume output contains unknown query ID: {result.query_id!r}"
                )
            if result.op != expected_op:
                raise VMBMKError(
                    "resume output operation mismatch for "
                    f"{result.query_id!r}: expected {expected_op!r}, got {result.op!r}"
                )
            if result.query_id in results:
                raise VMBMKError(
                    "resume outputs contain duplicate result "
                    f"{result.query_id!r}"
                )
            results[result.query_id] = result
    return results


def _execute_shards(
    data_root: str | Path, config: Mapping[str, Any],
    jobs: Sequence[tuple[int, list[Query], Path, Path]], native_metric: str | None,
) -> None:
    """Wait for each shard worker; failures retain their resumable outputs."""
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [
            executor.submit(
                _run_worker_process,
                data_root,
                shard_queries,
                config,
                shard_output,
                shard,
                gpu,
                native_metric,
            )
            for gpu, shard, shard_queries, shard_output in jobs
        ]
        for future in futures:
            future.result()


def _collect_shards(
    result_path: Path, queries: Sequence[Query], resumed: Mapping[str, Result],
    legacy_parts: Sequence[Path], jobs: Sequence[tuple[int, list[Query], Path, Path]],
) -> None:
    """Validate and assemble original query order before removing partial files."""
    results = dict(resumed)
    for _gpu, _shard, _shard_queries, shard_output in jobs:
        for result in read_results(shard_output):
            if result.query_id in results:
                raise VMBMKError(
                    "sharded inference produced duplicate result "
                    f"{result.query_id!r}"
                )
            results[result.query_id] = result
    missing = [query.query_id for query in queries if query.query_id not in results]
    if missing:
        raise VMBMKError(
            f"sharded inference result coverage mismatch; missing={missing}"
        )
    replace_jsonl(
        result_path,
        (results[query.query_id].to_dict() for query in queries),
    )
    validate_results(queries, result_path)
    for partial in {*legacy_parts, *(job[3] for job in jobs)}:
        partial.unlink(missing_ok=True)


def _run_sharded(
    data_root: str | Path,
    config: Mapping[str, Any],
    result_path: Path,
    queries: list[Query],
    gpus: tuple[int, ...],
    native_metric: str | None,
) -> None:
    """Run episode-local shards, retaining valid work when the GPU set changes."""
    legacy_parts = sorted(result_path.parent.glob(f"{result_path.name}.part-*"))
    default_prefix = f"{result_path.name}.part-{len(gpus):03d}-"
    reuse_existing_layout = (
        not result_path.exists()
        and all(
            path.name.startswith(default_prefix)
            and path.name[len(default_prefix):].isdigit()
            for path in legacy_parts
        )
    )
    resumed: dict[str, Result] = {}
    if reuse_existing_layout:
        pending = queries
    else:
        resume_paths = list(legacy_parts)
        if result_path.exists():
            resume_paths.append(result_path)
        resumed = _read_resume_results(queries, resume_paths)
        pending = [
            query for query in queries if query.query_id not in resumed
        ]
        if not pending:
            replace_jsonl(
                result_path,
                (resumed[query.query_id].to_dict() for query in queries),
            )
            validate_results(queries, result_path)
            for partial in legacy_parts:
                partial.unlink(missing_ok=True)
            return

    shards = _partition_queries(pending, len(gpus))
    resume_attempt = 1
    if not reuse_existing_layout:
        while any(
            path.name.startswith(
                f"{result_path.name}.part-{len(shards):03d}-resume-"
                f"{resume_attempt:03d}-"
            )
            for path in legacy_parts
        ):
            resume_attempt += 1
    with tempfile.TemporaryDirectory(prefix="vmbmk-shards-") as directory:
        query_root = Path(directory)
        jobs = []
        for index, (gpu, shard) in enumerate(zip(gpus, shards)):
            shard_queries = query_root / f"queries-{index:03d}.jsonl"
            if reuse_existing_layout:
                shard_name = f"{result_path.name}.part-{len(shards):03d}-{index:03d}"
            else:
                shard_name = (
                    f"{result_path.name}.part-{len(shards):03d}-resume-"
                    f"{resume_attempt:03d}-{index:03d}"
                )
            shard_output = result_path.with_name(shard_name)
            write_jsonl(shard_queries, (query.to_dict() for query in shard))
            jobs.append((gpu, shard, shard_queries, shard_output))
            print(
                f"inference shard {index + 1}/{len(shards)}: "
                f"gpu={gpu} queries={len(shard)}",
                flush=True,
            )

        _execute_shards(data_root, config, jobs, native_metric)
        _collect_shards(result_path, queries, resumed, legacy_parts, jobs)


def run_inference(
    data_root: str | Path,
    query_path: str | Path,
    config: Mapping[str, Any],
    output_path: str | Path,
    *,
    gpu: int | Sequence[int] | None,
    native_metric: str | None = None,
) -> Path:
    result_path = Path(output_path).resolve()
    if adapter_class(config["adapter"]).requires_gpu:
        gpus = _normalize_gpus(gpu)
    else:
        if gpu is not None:
            raise ConfigurationError("CPU adapters do not accept a gpu selection")
        gpus = ()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    queries = read_queries(query_path)
    if result_path.exists():
        try:
            validate_results(queries, result_path)
            return result_path
        except (ValueError, VMBMKError):
            # The worker validates the resumable prefix, truncates only an
            # interrupted final line, and appends the missing queries.
            pass

    if len(gpus) <= 1:
        _run_worker_process(
            data_root,
            query_path,
            config,
            result_path,
            queries,
            gpus[0] if gpus else None,
            native_metric,
        )
    else:
        _run_sharded(
            data_root,
            config,
            result_path,
            queries,
            gpus,
            native_metric,
        )
    return result_path
