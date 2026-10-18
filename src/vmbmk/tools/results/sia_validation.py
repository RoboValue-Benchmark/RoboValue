"""Validate saved SIA generation, judge decisions and geometric aggregation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vmbmk.data.dataset import Dataset
from vmbmk.inference.cache import validate_prediction_cache
from vmbmk.inference.dispatch import validate_results
from vmbmk.inference.queries import read_results
from vmbmk.metrics.sia.evaluation import (
    _generation_queries, _read_bundle, build_sia_queries, select_sia_responses,
    sia_episode_subtasks,
)
from vmbmk.metrics.sia.scoring import score_sia_mappings
from vmbmk.runner.run import RunConfig
from vmbmk.serialization import read_jsonl


def validate_sia_artifacts(
    config_path: Path, run_dir: Path, dataset: Dataset, metric: dict[str, Any],
    *, operations_path: Path | None = None,
) -> None:
    """Reproduce saved judge scores without regenerating random query frames."""
    config = RunConfig.load(config_path)
    domains = next(domains for name, _, domains in config.metrics if name == "sia")
    tasks = (config.metric_tasks or {}).get("sia", config.tasks)
    cache = run_dir / "sia"
    query_path = cache / "queries.jsonl"
    source = json.loads((cache / "response_source.json").read_text(encoding="utf-8"))["external"]
    if source is not None and (config.sia_responses is None or Path(source) != config.sia_responses.resolve()):
        raise ValueError("SIA recorded response source differs from configuration")
    bundle_path = Path(source) if source is not None else cache / "responses.json"
    items = select_sia_responses(_read_bundle(bundle_path), dataset, tasks, domains)
    item_ids = {item["query"].query_id for item in items}
    if source is None:
        if not query_path.is_file():
            raise ValueError("SIA saved generation queries are missing")
        queries = _generation_queries(query_path, dataset, tasks, domains, seed=0)
        validate_prediction_cache(dataset.root, queries, config.model_config(), cache / "subtask_results.jsonl", None, dataset=dataset)
        validate_results(queries, cache / "subtask_results.jsonl")
        generated_queries = {query.query_id: query for query in queries}
        predictions = {prediction.query_id: prediction for prediction in read_results(cache / "subtask_results.jsonl")}
        if item_ids != set(generated_queries):
            raise ValueError("SIA response bundle coverage differs from saved generation")
        for item in items:
            query_id = item["query"].query_id
            if item["query"] != generated_queries[query_id] or item["prediction"] != predictions[query_id]:
                raise ValueError("SIA response bundle differs from saved generation")
    else:
        expected_ids = {query.query_id for query in build_sia_queries(dataset, tasks, domains, seed=0)}
        if item_ids != expected_ids:
            raise ValueError("SIA response bundle coverage differs from current annotations")
    for item in items:
        query = item["query"]
        task = dataset.tasks[query.state.task_id]
        episode = dataset.episode(query.state.task_id, query.state.episode_id)
        spans, candidates = sia_episode_subtasks(task, episode)
        target = query.query_id.rsplit(":", 1)[-1]
        start, end = spans[target]
        if (not start <= query.state.anchor_frame < end
                or query.instruction != task.instruction
                or item["task_instruction"] != task.instruction
                or item["target_subtask_id"] != target
                or item["candidates"] != candidates):
            raise ValueError("SIA response bundle disagrees with current subtask contracts")
    decisions = list(read_jsonl(cache / "decisions.jsonl"))
    mapped = {record["query_id"]: record for record in decisions}
    if len(mapped) != len(decisions) or set(mapped) != item_ids:
        raise ValueError("SIA judge decision coverage mismatch")
    records, recomputed = score_sia_mappings(items, mapped)
    if {record["query_id"]: record for record in records} != mapped:
        raise ValueError("SIA saved judge decisions contain inconsistent scores")
    if recomputed != metric:
        raise ValueError("SIA metric does not match saved judge decisions")
    rows = list(read_jsonl(operations_path or run_dir / "operations.jsonl"))
    if operations_path is not None and any(not row["query_id"].startswith("sia:") for row in rows):
        raise ValueError("SIA checkpoint contains unexpected saved predictions")
    sia_rows = [row for row in rows if row["query_id"].startswith("sia:")]
    operations = {row["query_id"]: row for row in sia_rows}
    if len(operations) != len(sia_rows) or any(row["op"] != "value" for row in sia_rows):
        raise ValueError("SIA saved predictions contain duplicate IDs or incorrect operations")
    if set(operations) != set(mapped) or any(operations[record["query_id"]]["score"] != record["score"] for record in records):
        raise ValueError("SIA operations disagree with saved judge decisions")
