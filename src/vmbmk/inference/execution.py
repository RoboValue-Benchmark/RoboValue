"""Execute planned queries through the configured inference runtime."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from vmbmk.errors import VMBMKError
from vmbmk.inference.dispatch import run_inference
from vmbmk.inference.queries import Query, Result, read_results
from vmbmk.serialization import write_jsonl


def run_queries(
    data_root: str | Path,
    queries: Sequence[Query],
    model: Mapping[str, Any],
    operation_path: str | Path,
    gpu: int,
    *,
    metric: str,
    mode: str,
    native_metric: str | None = None,
) -> list[Result]:
    if mode == "base":
        active_native_metric = None
    elif mode == "native" and native_metric is not None:
        active_native_metric = native_metric
    else:
        raise VMBMKError(
            f"{metric} has no separate metric-native adapter contract; use mode=base"
        )
    with tempfile.TemporaryDirectory(prefix=f"vmbmk-{metric}-") as directory:
        query_path = Path(directory) / "queries.jsonl"
        write_jsonl(query_path, (query.to_dict() for query in queries))
        run_inference(
            data_root,
            query_path,
            model,
            operation_path,
            gpu=gpu,
            native_metric=active_native_metric,
        )
    return read_results(operation_path)
