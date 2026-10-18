"""Bind prediction outputs and resumable shards to their actual inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import Query
from vmbmk.runner.provenance import model_identity, run_identity


def load_query_dataset(data_root: str | Path, queries: Sequence[Query]) -> Dataset:
    """Load the same metric-scoped metadata in dispatch and worker processes."""
    from vmbmk.metrics.registry import SUPPORTED_METRICS

    metrics = {query.query_id.split(":", 1)[0] for query in queries}
    return Dataset.load(
        data_root,
        metrics=sorted(metrics) if metrics and metrics <= SUPPORTED_METRICS else None,
    )


def _prediction_identity(
    data_root: str | Path,
    queries: Sequence[Query],
    config: Mapping[str, Any],
    native_metric: str | None,
    dataset: Dataset | None,
) -> dict[str, Any]:
    if dataset is None:
        dataset = load_query_dataset(data_root, queries)
    payload = json.dumps(
        [query.to_dict() for query in queries],
        sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode()

    return {
        "protocol": "prediction-input-identity-v2",
        "queries_sha256": hashlib.sha256(payload).hexdigest(),
        "model_sha256": model_identity(config, data_root=dataset.root),
        "native_metric": native_metric,
        "dataset": run_identity(dataset),
    }


def validate_prediction_cache(
    data_root: str | Path, queries: Sequence[Query], config: Mapping[str, Any],
    output: Path, native_metric: str | None, *, dataset: Dataset | None = None,
) -> None:
    """Read-only check of saved prediction provenance; never bind old outputs."""
    identity_path = output.parent / ".inference-identities" / f"{output.name}.json"
    if not identity_path.is_file():
        raise VMBMKError("prediction provenance is missing; use a new output path")
    expected = _prediction_identity(data_root, queries, config, native_metric, dataset)
    if json.loads(identity_path.read_text(encoding="utf-8")) != expected:
        raise VMBMKError("prediction inputs or model changed; use a new output path")


def bind_prediction_cache(
    data_root: str | Path, queries: Sequence[Query], config: Mapping[str, Any],
    output: Path, native_metric: str | None, *, dataset: Dataset | None = None,
) -> None:
    """Bind new outputs and reject changed or unbound saved predictions.

    Model assets use file size/mtime identities, not full checkpoint hashes.
    Existing unbound artifacts must be rescored separately or use a new output.
    """
    identity = _prediction_identity(data_root, queries, config, native_metric, dataset)
    identity_path = output.parent / ".inference-identities" / f"{output.name}.json"
    has_predictions = output.exists() or any(output.parent.glob(f"{output.name}.part-*"))
    if identity_path.exists():
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded == identity:
            return
        if has_predictions:
            raise VMBMKError("prediction inputs or model changed; use a new output path")
    elif has_predictions:
        raise VMBMKError("prediction provenance is missing; use a new output path")
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = identity_path.with_name(f"{identity_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, identity_path)
