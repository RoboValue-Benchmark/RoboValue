from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.adapters import create_adapter
from vmbmk.adapters.base import Adapter
from vmbmk.data.dataset import Dataset
from vmbmk.errors import ConfigurationError, VMBMKError
from .dispatch import validate_queries
from .queries import (
    CompareQuery,
    Query,
    Result,
    SubtaskQuery,
    ValueQuery,
    read_queries,
)
from vmbmk.serialization import read_json, read_jsonl


def _read_checkpoint(output: Path, queries: Sequence[Query]) -> set[str]:
    if not output.exists():
        return set()
    if not output.is_file():
        raise VMBMKError(f"resume output is not a file: {output}")

    expected = {query.query_id: query.op for query in queries}

    def scan() -> set[str]:
        completed: set[str] = set()
        for index, row in enumerate(read_jsonl(output), start=1):
            result = Result.from_dict(row, f"{output}:{index}")
            if result.query_id in completed:
                raise VMBMKError(
                    "resume output contains duplicate query_id values"
                )
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
            completed.add(result.query_id)
        return completed

    try:
        completed = scan()
    except ValueError:
        # Chunk commits always end in a newline. Only a non-newline-terminated
        # tail can be an interrupted write; corruption in committed lines must
        # remain visible instead of being silently discarded.
        data = output.read_bytes()
        if not data or data.endswith(b"\n"):
            raise
        last_newline = data.rfind(b"\n")
        with output.open("rb+") as handle:
            handle.truncate(last_newline + 1 if last_newline >= 0 else 0)
            handle.flush()
            os.fsync(handle.fileno())
        return scan()
    has_unterminated_line = False
    if output.stat().st_size:
        with output.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            has_unterminated_line = handle.read(1) != b"\n"
    if has_unterminated_line:
        # A complete JSON object without its newline is valid, but appending
        # directly would concatenate the next object onto the same line.
        with output.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return completed


def _append_checkpoint(output: Path, results: Sequence[Result]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_chunk_size(batch_size: int) -> int:
    configured = os.environ.get("VMBMK_CHECKPOINT_CHUNK_SIZE")
    if configured is not None:
        try:
            requested = int(configured)
        except ValueError as exc:
            raise ConfigurationError("VMBMK_CHECKPOINT_CHUNK_SIZE must be a positive integer") from exc
        if requested <= 0:
            raise ConfigurationError("VMBMK_CHECKPOINT_CHUNK_SIZE must be a positive integer")
        return min(2048, requested)
    return min(512, max(1, batch_size * 16))


def run_worker(
    data_root: str | Path,
    query_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    native_metric: str | None = None,
) -> None:
    config = read_json(config_path)
    remote = config.get("adapter") == "remote_api"
    if remote and native_metric is not None:
        raise ConfigurationError("Remote API v2 supports only mode=base")
    if not remote:
        _configure_torch_threads()
    queries = read_queries(query_path)
    from vmbmk.metrics.registry import SUPPORTED_METRICS

    query_metrics = {query.query_id.split(":", 1)[0] for query in queries}
    dataset = Dataset.load(
        data_root,
        metrics=sorted(query_metrics) if query_metrics and query_metrics <= SUPPORTED_METRICS else None,
    )
    validate_queries(dataset, queries)
    output = Path(output_path)
    if remote:
        adapter = create_adapter(config, dataset)
        adapter.prepare_run(output, queries)
    completed_ids = _read_checkpoint(output, queries)
    pending = [query for query in queries if query.query_id not in completed_ids]
    if not pending:
        return

    if not remote:
        adapter = create_adapter(config, dataset)
    try:
        if remote:
            chunk_size = adapter.batch_size
        else:
            configured_batch = config.get("batch_size", 1)
            if (
                isinstance(configured_batch, bool)
                or not isinstance(configured_batch, int)
                or configured_batch < 1
            ):
                configured_batch = 1
            # Amortize model-call and fsync overhead while keeping recovery loss
            # bounded. This is derived from the consumed adapter batch size and
            # does not add another run-config field.
            chunk_size = _checkpoint_chunk_size(configured_batch)
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]
            results = _evaluate_chunk(adapter, chunk, native_metric, config)
            _append_checkpoint(output, results)
            completed = len(completed_ids) + min(start + chunk_size, len(pending))
            print(
                f"worker checkpoint: {completed}/{len(queries)} "
                f"({len(completed_ids)} previously completed)",
                flush=True,
            )
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


def _configure_torch_threads() -> None:
    """Avoid CPU oversubscription starving the GPU input pipeline.

    A worker owns one model and runs one inference stream. The default torch
    thread pools can otherwise create hundreds of runnable threads when two
    workers share a host, causing context-switch overhead and long gaps between
    CUDA launches. Environment variables remain the escape hatch for machines
    with a different CPU budget.
    """
    cpu_count = os.cpu_count() or 1
    configured = os.environ.get("VMBMK_TORCH_THREADS")
    try:
        threads = int(configured) if configured is not None else min(8, cpu_count)
    except ValueError:
        threads = min(8, cpu_count)
    threads = max(1, threads)
    # Set these before importing torch so its native pools and linked BLAS
    # libraries do not eagerly create a host-sized pool for every worker.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, str(threads))
    try:
        import torch
    except ImportError:
        return
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(min(2, threads))
    except RuntimeError:
        # Torch may already have initialized its pools in an embedding process.
        pass


def _evaluate_chunk(
    adapter: Adapter,
    queries: Sequence[Query],
    native_metric: str | None,
    config: Mapping[str, Any],
) -> list[Result]:
    if native_metric is not None:
        method = getattr(adapter, native_metric, None)
        if not callable(method):
            raise VMBMKError(
                f"adapter {config.get('adapter')!r} has no native "
                f"{native_metric!r} interface"
            )
        native_scores = method(queries)
        if len(native_scores) != len(queries):
            raise VMBMKError(
                "native metric interface returned the wrong number of scores"
            )
        results = []
        for query, score in zip(queries, native_scores):
            if isinstance(query, SubtaskQuery):
                if not isinstance(score, str) or not score.strip():
                    raise VMBMKError(
                        "native subtask outputs must be non-empty strings"
                    )
                results.append(Result(query.query_id, query.op, output=score.strip()))
            else:
                results.append(Result(query.query_id, query.op, score=score))
        return results

    values = [query for query in queries if isinstance(query, ValueQuery)]
    comparisons = [query for query in queries if isinstance(query, CompareQuery)]
    subtasks = [query for query in queries if isinstance(query, SubtaskQuery)]
    value_scores = adapter.value(values) if values else []
    compare_scores = adapter.compare(comparisons) if comparisons else []
    try:
        subtask_outputs = adapter.subtask(subtasks) if subtasks else []
    except NotImplementedError as exc:
        raise VMBMKError(
            f"adapter {config.get('adapter')!r} has no subtask interface"
        ) from exc
    if len(value_scores) != len(values) or len(compare_scores) != len(comparisons):
        raise VMBMKError("adapter returned the wrong number of scores")
    if len(subtask_outputs) != len(subtasks):
        raise VMBMKError("adapter returned the wrong number of subtask outputs")
    scores = {
        query.query_id: score for query, score in zip(values, value_scores)
    }
    scores.update(
        {
            query.query_id: score
            for query, score in zip(comparisons, compare_scores)
        }
    )
    outputs = {
        query.query_id: output
        for query, output in zip(subtasks, subtask_outputs)
    }
    results = []
    for query in queries:
        if isinstance(query, SubtaskQuery):
            if not isinstance(outputs[query.query_id], str):
                raise VMBMKError("adapter subtask outputs must be strings")
            results.append(
                Result(query.query_id, query.op, output=outputs[query.query_id])
            )
        else:
            results.append(Result(query.query_id, query.op, score=scores[query.query_id]))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-metric")
    args = parser.parse_args(argv)
    run_worker(
        args.data,
        args.queries,
        args.config,
        args.output,
        native_metric=args.native_metric,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
