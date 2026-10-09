from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset, Episode, Task
from vmbmk.errors import VMBMKError
from vmbmk.inference.dispatch import run_inference
from vmbmk.inference.queries import Result, StateRef, SubtaskQuery, query_from_dict, read_queries, read_results
from vmbmk.serialization import replace_jsonl
from .judge import (
    DeepSeekCandidateJudge,
    _candidate_labels,
    _target_logprob,
    classification_prompt,
)
from .scoring import (
    SIA_AGGREGATION,
    SIA_EPISODE_SELECTION,
    SIA_PROTOCOL,
    score_sia_mappings,
)


def is_sia_episode(episode: Episode) -> bool:
    """Include normal, diverse, and unmarked trajectories."""
    return episode.cspc_solution_type in {None, "normal", "diverse"}


def _needs_diverse_remapping(episode: Episode) -> bool:
    ordinal_ids = {f"{index:03d}" for index in range(1, 10)}
    return episode.cspc_solution_type == "diverse" and not (
        set(episode.subtask_spans) <= ordinal_ids
    )


def sia_episode_subtasks(
    task: Task, episode: Episode,
) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
    """Return evaluation spans/candidates without changing dataset annotations.

    Diverse episodes with semantic IDs use chronological 001..N labels, carrying
    each original ID's description along with it. Numeric-only diverse
    episodes and all ordinary episodes keep their original taxonomy.
    """
    if not _needs_diverse_remapping(episode):
        return dict(episode.subtask_spans), dict(task.subtasks)
    ordered = sorted(episode.subtask_spans.items(), key=lambda item: item[1])
    if any(left[1][1] > right[1][0] for left, right in zip(ordered, ordered[1:])):
        raise VMBMKError(
            f"{task.task_id}/{episode.episode_id}: cannot order overlapping "
            "diverse subtask spans"
        )
    missing = sorted(set(episode.subtask_spans) - set(task.subtasks))
    if missing:
        raise VMBMKError(
            f"{task.task_id}/{episode.episode_id}: diverse SIA remapping requires "
            f"original subtask descriptions in task metadata; missing={missing}"
        )
    spans = {f"{index:03d}": span for index, (_, span) in enumerate(ordered, start=1)}
    candidates = {
        f"{index:03d}": task.subtasks[original_id]
        for index, (original_id, _) in enumerate(ordered, start=1)
    }
    _candidate_labels(candidates)
    return spans, candidates


def _remap_diverse_response(
    item: dict[str, Any], task: Task, episode: Episode,
) -> dict[str, Any]:
    if not _needs_diverse_remapping(episode):
        return item
    spans, candidates = sia_episode_subtasks(task, episode)
    query = item["query"]
    frame = query.state.anchor_frame
    matches = [key for key, (start, end) in spans.items() if start <= frame < end]
    if len(matches) != 1:
        raise VMBMKError(
            f"{query.query_id}: frame {frame} matches {len(matches)} current "
            "diverse subtask spans; cannot remap this answer"
        )
    target = matches[0]
    query_id = f"sia:{query.state.task_id}:{query.state.episode_id}:{target}"
    return {
        **item,
        "query": replace(query, query_id=query_id),
        "prediction": replace(item["prediction"], query_id=query_id),
        "target_subtask_id": target,
        "task_instruction": task.instruction,
        "candidates": candidates,
    }


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_sia_responses(
    dataset: Dataset,
    queries: Sequence[SubtaskQuery],
    predictions: Sequence[Result],
    bundle_path: str | Path,
) -> None:
    """Persist immutable GPU outputs and the exact candidate taxonomy to judge."""
    by_id = {result.query_id: result for result in predictions}
    expected_ids = {query.query_id for query in queries}
    if len(by_id) != len(predictions) or set(by_id) != expected_ids:
        raise VMBMKError("SIA result coverage mismatch")
    items = []
    for query in queries:
        prediction = by_id[query.query_id]
        if prediction.op != "subtask" or not prediction.output:
            raise VMBMKError(
                f"SIA result {query.query_id!r} is not a subtask output"
            )
        task = dataset.tasks[query.state.task_id]
        target_id = query.query_id.rsplit(":", 1)[-1]
        episode = dataset.episode(query.state.task_id, query.state.episode_id)
        item = _remap_diverse_response({
            "query": query,
            "prediction": prediction,
            "target_subtask_id": target_id,
            "task_instruction": task.instruction,
            "candidates": dict(task.subtasks),
            "domain": episode.domain,
        }, task, episode)
        if item["target_subtask_id"] not in item["candidates"]:
            raise VMBMKError(
                f"SIA query {query.query_id!r} references an unknown subtask"
            )
        items.append({
            **item,
            "query": item["query"].to_dict(),
            "prediction": item["prediction"].to_dict(),
        })
    if len({item["query"]["query_id"] for item in items}) != len(items):
        raise VMBMKError("SIA duplicate subtask sample after diverse remapping")
    _save_json(Path(bundle_path), {"version": 1, "items": items})


def _read_bundle(responses: str | Path) -> list[dict[str, Any]]:
    try:
        bundle = json.loads(Path(responses).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VMBMKError("invalid SIA response bundle") from exc
    if bundle.get("version") != 1 or not isinstance(bundle.get("items"), list):
        raise VMBMKError("invalid SIA response bundle")
    items = bundle["items"]
    if not items:
        raise VMBMKError("invalid or empty SIA response bundle")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise VMBMKError("invalid SIA response item")
        expected = {
            "query",
            "prediction",
            "target_subtask_id",
            "task_instruction",
            "candidates",
            "domain",
        }
        if set(item) != expected:
            raise VMBMKError("invalid SIA response item")
        try:
            query = query_from_dict(item["query"], f"SIA item {index}")
            prediction = Result.from_dict(
                item["prediction"], f"SIA item {index}"
            )
        except (TypeError, VMBMKError) as exc:
            raise VMBMKError("invalid SIA response item") from exc
        candidates = item["candidates"]
        valid = (
            isinstance(query, SubtaskQuery)
            and prediction.op == "subtask"
            and query.query_id == prediction.query_id
            and query.query_id not in seen
            and isinstance(prediction.output, str)
            and prediction.output.strip()
            and isinstance(candidates, dict)
            and candidates
            and isinstance(item["target_subtask_id"], str)
            and item["target_subtask_id"] in candidates
            and isinstance(item["task_instruction"], str)
            and item["task_instruction"].strip()
            and isinstance(item["domain"], str)
            and item["domain"].strip()
            and all(
                isinstance(key, str)
                and key.strip()
                and isinstance(value, str)
                and value.strip()
                for key, value in candidates.items()
            )
        )
        if not valid:
            raise VMBMKError("invalid SIA response item")
        seen.add(query.query_id)
        validated.append(
            {
                "query": query,
                "prediction": prediction,
                "target_subtask_id": item["target_subtask_id"],
                "task_instruction": item["task_instruction"],
                "candidates": candidates,
                "domain": item["domain"],
            }
        )
    return validated


def judge_sia_responses(
    responses: str | Path,
    operation_path: str | Path,
    *,
    model: Mapping[str, Any],
    cache_dir: str | Path,
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select and remap saved answers by episode annotations, without a GPU."""
    items = select_sia_responses(_read_bundle(responses), dataset, task_ids, domains)
    workers = model.get("sia_workers", 50)
    timeout = model.get("sia_timeout", 120)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise VMBMKError("SIA sia_workers must be a positive integer")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise VMBMKError("SIA sia_timeout must be a positive finite number")
    judge = DeepSeekCandidateJudge(
        base_url=model.get("sia_base_url"),
        model=model.get("sia_model"),
        timeout=timeout,
    )
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    def evaluate(index: int, item: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        query = item["query"]
        prediction = item["prediction"]
        query_id = query.query_id
        prompt = classification_prompt(
            item["task_instruction"], item["candidates"], prediction.output
        )
        identity = {
            "query_id": query_id,
            "protocol": SIA_PROTOCOL,
            "target_subtask_id": item["target_subtask_id"],
            "candidates": item["candidates"],
            "top_logprobs": 20,
            "thinking": {"type": "disabled"},
            "prompt": prompt,
            "model": judge.model,
            "url": judge._completion_url(),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        checkpoint = cache / f"{digest}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("identity") != identity:
                raise VMBMKError("SIA judgement cache identity mismatch")
            result = saved.get("result", {})
            _target_logprob(result, item["candidates"], item["target_subtask_id"])
            return query_id, result
        started = time.monotonic()
        print(
            f"[SIA] judging started: item={index}/{len(items)} "
            f"query_id={query_id}",
            file=sys.stderr,
            flush=True,
        )
        result = judge.classify(
            item["task_instruction"], item["candidates"], prediction.output
        )
        _target_logprob(result, item["candidates"], item["target_subtask_id"])
        _save_json(
            checkpoint,
            {"identity": identity, "result": result},
        )
        print(
            f"[SIA] judging completed: item={index}/{len(items)} "
            f"query_id={query_id} predicted={result['predicted_subtask_id']} "
            f"elapsed={time.monotonic() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return query_id, result

    mapped: dict[str, dict[str, Any]] = {}
    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = iter(enumerate(items, start=1))
        futures: dict[Future[tuple[str, dict[str, Any]]], str] = {}
        exhausted = False
        while futures or not exhausted:
            while not exhausted and len(futures) < workers * 2:
                item = next(pending, None)
                if item is None:
                    exhausted = True
                    break
                index, response = item
                futures[executor.submit(evaluate, index, response)] = response["query"].query_id
            if not futures:
                break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                response_id = futures.pop(future)
                try:
                    query_id, result = future.result()
                    mapped[query_id] = result
                except Exception as exc:
                    failures.append((response_id, exc))
                    print(
                        f"[SIA] judging failed: query_id={response_id} error={exc}",
                        file=sys.stderr,
                        flush=True,
                    )
    if failures:
        raise VMBMKError(
            f"SIA judging failed for {len(failures)}/{len(items)} "
            f"responses; successful mappings are saved in {cache}; first "
            f"failure: {failures[0][0]}: {failures[0][1]}"
        ) from failures[0][1]

    records, result = score_sia_mappings(items, mapped)
    replace_jsonl(
        operation_path,
        (
            {
                "query_id": record["query_id"],
                "op": "value",
                "score": record["score"],
            }
            for record in records
        ),
    )
    replace_jsonl(Path(cache).parent / "decisions.jsonl", records)
    return result


def select_sia_responses(
    items: Sequence[dict[str, Any]],
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select and remap legacy and freshly generated bundles in the same way."""
    selected_tasks = set(task_ids) if task_ids is not None else None
    selected_domains = set(domains) if domains is not None else None
    selected = []
    seen: set[str] = set()
    for item in items:
        state = item["query"].state
        if selected_tasks is not None and state.task_id not in selected_tasks:
            continue
        episode = dataset.episode(state.task_id, state.episode_id)
        if not is_sia_episode(episode):
            continue
        if selected_domains is not None and episode.domain not in selected_domains:
            continue
        item = _remap_diverse_response(
            {**item, "domain": episode.domain}, dataset.tasks[state.task_id], episode,
        )
        query_id = item["query"].query_id
        if query_id in seen:
            raise VMBMKError(f"SIA duplicate subtask sample after diverse remapping: {query_id}")
        seen.add(query_id)
        selected.append(item)
    if not selected:
        raise VMBMKError("SIA responses contain no trajectories for the selected tasks/domains")
    return selected


def build_sia_queries(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    *,
    seed: int | None = None,
) -> list[SubtaskQuery]:
    selected = set(domains) if domains is not None else None
    ids = task_ids if task_ids is not None else sorted(dataset.tasks)
    rng = random.Random(seed)
    queries: list[SubtaskQuery] = []
    for task_id in ids:
        task = dataset.tasks[task_id]
        if "SIA" not in task.metrics:
            continue
        if not task.subtasks:
            raise VMBMKError(f"SIA task {task_id!r} has no subtasks in metadata")
        for episode_id, episode in sorted(task.episodes.items()):
            if not is_sia_episode(episode):
                continue
            if selected is not None and episode.domain not in selected:
                continue
            spans, candidates = sia_episode_subtasks(task, episode)
            for subtask_id, span in sorted(spans.items()):
                if subtask_id not in candidates:
                    raise VMBMKError(
                        f"{task_id}/{episode_id} references unknown subtask "
                        f"{subtask_id!r}"
                    )
                start, end = span
                length = end - start
                middle_start = start + math.ceil(length * 0.2)
                middle_end = start + math.floor(length * 0.8)
                if middle_end <= middle_start:
                    middle_start, middle_end = start, end
                frame = rng.randrange(middle_start, middle_end)
                query_id = f"sia:{task_id}:{episode_id}:{subtask_id}"
                queries.append(
                    SubtaskQuery(
                        query_id,
                        StateRef(task_id, episode_id, frame),
                        task.instruction,
                    )
                )
    if not queries:
        raise VMBMKError(
            "dataset has no trajectory with SIA subtask annotations in the selected tasks/domains"
        )
    return queries


def _generation_queries(
    query_path: Path, dataset: Dataset, task_ids: Sequence[str], domains: Sequence[str],
    seed: int | None,
) -> list[SubtaskQuery]:
    """Reuse only queries matching current episode, subtask and instruction contracts."""
    if query_path.exists():
        queries = read_queries(query_path)
        if any(not isinstance(query, SubtaskQuery) for query in queries):
            raise VMBMKError(
                "SIA saved queries are not subtask queries"
            )
        if any(
            query.state.task_id not in task_ids
            or not is_sia_episode(dataset.episode(query.state.task_id, query.state.episode_id))
            or dataset.episode(query.state.task_id, query.state.episode_id).domain not in domains
            for query in queries
        ):
            raise VMBMKError(
                "SIA saved queries include excluded trajectories; use a new output directory"
            )
        expected_ids = {
            query.query_id
            for query in build_sia_queries(dataset, task_ids, domains, seed=0)
        }
        if {query.query_id for query in queries} != expected_ids:
            raise VMBMKError(
                "SIA saved queries have obsolete subtask IDs or trajectory coverage; "
                "use a new output directory"
            )
        for query in queries:
            task = dataset.tasks[query.state.task_id]
            episode = dataset.episode(query.state.task_id, query.state.episode_id)
            spans, _ = sia_episode_subtasks(task, episode)
            start, end = spans[query.query_id.rsplit(":", 1)[-1]]
            if (not start <= query.state.anchor_frame < end
                    or query.instruction != task.instruction):
                raise VMBMKError(
                    "SIA saved queries do not match current annotations/instructions; "
                    "use a new output directory"
                )
    else:
        queries = build_sia_queries(dataset, task_ids, domains, seed=seed)
        replace_jsonl(query_path, (query.to_dict() for query in queries))
    return queries


def run_sia(
    data_root: str | Path,
    model: Mapping[str, Any],
    task_ids: Sequence[str],
    operation_path: str | Path,
    gpu: int | Sequence[int],
    *,
    mode: str,
    domains: Sequence[str],
    stage: str = "all",
    responses_path: str | Path | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    if mode != "base":
        raise VMBMKError("SIA currently supports only mode=base")
    if stage not in {"all", "generate", "judge"}:
        raise VMBMKError(f"unknown SIA stage: {stage}")
    cache = Path(operation_path).parent / "sia"
    bundle_path = cache / "responses.json"
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=["sia"])
    if stage == "judge":
        return judge_sia_responses(
            responses_path or bundle_path,
            operation_path,
            model=model,
            cache_dir=cache / "judgements",
            dataset=dataset,
            task_ids=task_ids,
            domains=domains,
        )
    seed = model.get("sia_seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise VMBMKError("SIA sia_seed must be an integer")
    cache.mkdir(parents=True, exist_ok=True)
    query_path = cache / "queries.jsonl"
    result_path = cache / "subtask_results.jsonl"
    adapter_model = {
        key: value
        for key, value in model.items()
        if key
        not in {"sia_base_url", "sia_model", "sia_seed", "sia_workers", "sia_timeout"}
    }
    manifest = {
        "data": str(Path(data_root).resolve()),
        "episode_selection": SIA_EPISODE_SELECTION,
        "tasks": list(task_ids),
        "domains": list(domains),
        "seed": seed,
        "model": adapter_model,
        "instruction_protocol": (
            "task-only-robofac-subtask-v1"
            if adapter_model.get("adapter") == "robofac"
            else "task-only-v1"
        ),
    }
    manifest_path = cache / "generation.json"
    if manifest_path.exists() and json.loads(
        manifest_path.read_text(encoding="utf-8")
    ) != manifest:
        raise VMBMKError(
            "SIA saved generation does not match the requested configuration"
        )
    queries = _generation_queries(query_path, dataset, task_ids, domains, seed)
    _save_json(manifest_path, manifest)
    run_inference(data_root, query_path, adapter_model, result_path, gpu=gpu)
    predictions = read_results(result_path)
    save_sia_responses(dataset, queries, predictions, bundle_path)
    print(
        f"[SIA] saved {len(queries)} responses: {bundle_path}",
        flush=True,
    )
    if stage == "generate":
        return {
            "status": "generated",
            "protocol": SIA_PROTOCOL,
            "responses_path": str(bundle_path),
            "n_queries": len(queries),
        }
    return judge_sia_responses(
        bundle_path,
        operation_path,
        model=model,
        cache_dir=cache / "judgements",
        dataset=dataset,
        task_ids=task_ids,
        domains=domains,
    )
