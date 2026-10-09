from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import CompareQuery, Result, StateRef
from vmbmk.metrics.utils.statistics import edge_frames, mean
from vmbmk.metrics.utils.results import result_scores
from vmbmk.inference.execution import run_queries
from vmbmk.metrics.utils.aggregation import summarize_task_scores


TGAVariant = Literal["easy", "hard"]

# Keep the type mapping in the metric contract so metadata stays descriptive
# and lean.  Metadata IDs are normalized below because released snapshots use
# both bare (``001``) and prefixed (``counterfactual_1``) spellings.
TGA_COUNTERFACTUAL_TYPES: Mapping[str, str] = {
    "001": "objects",
    "002": "actions",
    "003": "placement",
    "004": "constraints",
}

_COUNTERFACTUAL_ID = re.compile(r"^counterfactual[-_](\d+)$", re.IGNORECASE)


def _canonical_counterfactual_id(candidate_id: str) -> str:
    """Normalize metadata IDs to the public three-digit TGA identifiers.

    Dataset snapshots have used both ``001`` and ``counterfactual_1`` forms.
    Keep the metric contract stable by accepting either spelling and emitting
    ``001``--``004`` in query IDs and summaries.
    """
    match = _COUNTERFACTUAL_ID.fullmatch(candidate_id)
    if match is not None:
        return f"{int(match.group(1)):03d}"
    return candidate_id


def _public_candidate_id(candidate_id: str, variant: TGAVariant) -> str:
    prefix = "counterfactual-"
    if variant == "hard" and candidate_id.startswith(prefix):
        return candidate_id[len(prefix) :]
    return candidate_id


def _candidate_type(candidate_id: str, variant: TGAVariant) -> str | None:
    if variant != "hard":
        return None
    public_id = _public_candidate_id(candidate_id, variant)
    return TGA_COUNTERFACTUAL_TYPES.get(public_id)


def _selected_tasks(
    dataset: Dataset, task_ids: Sequence[str] | None
) -> list[tuple[str, Any]]:
    ids = sorted(dataset.tasks) if task_ids is None else list(task_ids)
    tasks = []
    for task_id in ids:
        try:
            task = dataset.tasks[task_id]
        except KeyError as exc:
            raise VMBMKError(f"unknown task {task_id!r}") from exc
        if "TGA" in task.metrics and "VOC-MEM" not in task.metrics:
            tasks.append((task_id, task))
    if not tasks:
        raise VMBMKError(
            "dataset has no TGA task after excluding tasks that support VOC-MEM"
        )
    return tasks


def _candidate_instructions(
    tasks: Sequence[tuple[str, Any]], task_id: str, variant: TGAVariant
) -> list[tuple[str, str, bool]]:
    task_by_id = dict(tasks)
    task = task_by_id[task_id]
    candidates = [("correct", task.instruction, True)]
    if variant == "easy":
        seen = {task.instruction}
        for other_id, other in tasks:
            if other_id == task_id or other.instruction in seen:
                continue
            candidates.append((f"task-{other_id}", other.instruction, False))
            seen.add(other.instruction)
    elif variant == "hard":
        canonical: dict[str, str] = {}
        for raw_id, instruction in task.counterfactual_instructions.items():
            candidate_id = _canonical_counterfactual_id(str(raw_id))
            if candidate_id in canonical:
                raise VMBMKError(
                    f"TGA-hard task {task_id!r} has duplicate counterfactual ID "
                    f"{candidate_id!r} after normalization"
                )
            canonical[candidate_id] = instruction
        unknown_ids = sorted(set(canonical) - set(TGA_COUNTERFACTUAL_TYPES))
        if unknown_ids:
            raise VMBMKError(
                f"TGA-hard task {task_id!r} has unsupported counterfactual IDs: "
                f"{unknown_ids}; expected IDs are {sorted(TGA_COUNTERFACTUAL_TYPES)}"
            )
        for candidate_id, instruction in sorted(canonical.items()):
            if instruction == task.instruction:
                raise VMBMKError(
                    f"TGA-hard task {task_id!r} has a counterfactual identical "
                    "to the correct instruction"
                )
            candidates.append((f"counterfactual-{candidate_id}", instruction, False))
    else:
        raise VMBMKError(f"unknown TGA variant {variant!r}")
    if len(candidates) < 2:
        raise VMBMKError(
            f"TGA-{variant} task {task_id!r} has no usable negative instruction"
        )
    return candidates


def _plan(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    *,
    variant: TGAVariant,
    candidate_task_ids: Sequence[str] | None = None,
) -> tuple[
    list[CompareQuery],
    dict[str, tuple[str, str, str, str, bool]],
]:
    tasks = _selected_tasks(dataset, task_ids)
    # TGA-easy uses instructions from other tasks as negatives.  During an
    # incremental repair we may score only the missing target rows while
    # retaining the complete instruction vocabulary from the original run.
    candidate_tasks = (
        tasks
        if candidate_task_ids is None
        else _selected_tasks(dataset, candidate_task_ids)
    )
    missing_candidates = sorted({task_id for task_id, _ in tasks} - {task_id for task_id, _ in candidate_tasks})
    if missing_candidates:
        raise VMBMKError(f"TGA candidate tasks omit selected targets: {missing_candidates}")
    selected_domains = set(domains) if domains is not None else None
    observed_domains: set[str] = set()
    queries: list[CompareQuery] = []
    labels: dict[str, tuple[str, str, str, str, bool]] = {}

    for task_id, task in tasks:
        candidates = _candidate_instructions(candidate_tasks, task_id, variant)
        eligible = [
            episode
            for episode in task.episodes.values()
            if episode.success
            and episode.trr_group_id is None
            and episode.cspc_solution_type != "diverse"
            and (selected_domains is None or episode.domain in selected_domains)
        ]
        for episode in sorted(eligible, key=lambda row: row.episode_id):
            head, tail = edge_frames(episode.num_frames)
            observed_domains.add(episode.domain)
            for candidate_id, instruction, correct in candidates:
                for index, (frame_a, frame_b) in enumerate(zip(head, tail)):
                    query_id = (
                        f"tga_{variant}:{task_id}:{episode.episode_id}:"
                        f"{candidate_id}:{index}"
                    )
                    queries.append(
                        CompareQuery(
                            query_id,
                            StateRef(task_id, episode.episode_id, frame_a),
                            StateRef(task_id, episode.episode_id, frame_b),
                            instruction,
                        )
                    )
                    labels[query_id] = (
                        episode.domain,
                        task_id,
                        episode.episode_id,
                        candidate_id,
                        correct,
                    )

    if not queries:
        raise VMBMKError(f"dataset has no successful episodes for TGA-{variant}")
    if selected_domains is not None and observed_domains != selected_domains:
        raise VMBMKError(
            f"TGA-{variant} domains have no eligible episodes: "
            f"{sorted(selected_domains - observed_domains)}"
        )
    return queries, labels


def build_tga_queries(
    dataset: Dataset,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    *,
    variant: TGAVariant,
    candidate_task_ids: Sequence[str] | None = None,
) -> list[CompareQuery]:
    return _plan(
        dataset,
        task_ids,
        domains,
        variant=variant,
        candidate_task_ids=candidate_task_ids,
    )[0]


def _candidate_diagnostics(
    variant: TGAVariant,
    episodes: Mapping[tuple[str, str, str], list[tuple[str, bool, float]]],
    pairwise_by_candidate_task: Mapping[tuple[str, str, str], list[float]],
    margin_by_candidate_task: Mapping[tuple[str, str, str], list[float]],
    pairwise_by_type_task: Mapping[tuple[str, str, str], list[float]],
    margin_by_type_task: Mapping[tuple[str, str, str], list[float]],
) -> dict[str, Any]:
    """Package auxiliary candidate statistics separately from strict top1 scoring."""
    counterfactual_by_id: dict[str, Any] = {}
    candidate_ids = sorted(
        {candidate_id for candidate_id, _, _ in pairwise_by_candidate_task}
    )
    for candidate_id in candidate_ids:
        pairwise_values = {
            (domain, task_id): mean(values)
            for (candidate, domain, task_id), values in pairwise_by_candidate_task.items()
            if candidate == candidate_id
        }
        margin_values = {
            (domain, task_id): mean(values)
            for (candidate, domain, task_id), values in margin_by_candidate_task.items()
            if candidate == candidate_id
        }
        counterfactual_by_id[_public_candidate_id(candidate_id, variant)] = {
            "pairwise_accuracy": summarize_task_scores(pairwise_values),
            "mean_margin": summarize_task_scores(margin_values),
        }

    counterfactual_by_type: dict[str, Any] = {}
    for type_name in sorted(
        {type_name for type_name, _, _ in pairwise_by_type_task}
    ):
        pairwise_values = {
            (domain, task_id): mean(values)
            for (candidate_type, domain, task_id), values in pairwise_by_type_task.items()
            if candidate_type == type_name
        }
        margin_values = {
            (domain, task_id): mean(values)
            for (candidate_type, domain, task_id), values in margin_by_type_task.items()
            if candidate_type == type_name
        }
        counterfactual_by_type[type_name] = {
            "counterfactual_id": next(
                candidate_id
                for candidate_id, mapped_type in TGA_COUNTERFACTUAL_TYPES.items()
                if mapped_type == type_name
            ),
            "pairwise_accuracy": summarize_task_scores(pairwise_values),
            "mean_margin": summarize_task_scores(margin_values),
        }
    # A lossless, task-by-candidate view is useful when a TGA run is resumed:
    # rows from a new task can be merged without losing the old matrix.  Keep
    # scores (rather than only top-1 decisions) so the matrix remains useful
    # for diagnostics and reproduces the published episode aggregates.
    confusion_matrix: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(dict)
    for (domain, task_id, _episode_id), candidates in sorted(episodes.items()):
        target = confusion_matrix[domain].setdefault(task_id, {})
        for candidate_id, _correct, score in candidates:
            target.setdefault(candidate_id, []).append(score)
    normalized_matrix: dict[str, dict[str, dict[str, float]]] = {}
    for domain, task_rows in sorted(confusion_matrix.items()):
        normalized_matrix[domain] = {}
        for task_id, candidate_rows in sorted(task_rows.items()):
            normalized_matrix[domain][task_id] = {
                candidate_id: mean(values)
                for candidate_id, values in sorted(candidate_rows.items())
            }
    return {
        "counterfactual_by_id": counterfactual_by_id,
        "counterfactual_by_type": counterfactual_by_type,
        "confusion_matrix": normalized_matrix,
    }


def score_tga(
    dataset: Dataset,
    results: Sequence[Result],
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    *,
    variant: TGAVariant,
    candidate_task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    queries, labels = _plan(
        dataset,
        task_ids,
        domains,
        variant=variant,
        candidate_task_ids=candidate_task_ids,
    )
    scores = result_scores(queries, results, f"TGA-{variant}", "compare")
    candidate_values: dict[
        tuple[str, str, str, str, bool], list[float]
    ] = defaultdict(list)
    for query_id, label in labels.items():
        candidate_values[label].append(scores[query_id])

    episodes: dict[
        tuple[str, str, str], list[tuple[str, bool, float]]
    ] = defaultdict(list)
    for (domain, task_id, episode_id, candidate_id, correct), values in sorted(
        candidate_values.items()
    ):
        episodes[(domain, task_id, episode_id)].append(
            (candidate_id, correct, mean(values))
        )

    pairwise_by_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    top1_by_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    pairwise_by_candidate_task: dict[
        tuple[str, str, str], list[float]
    ] = defaultdict(list)
    margin_by_candidate_task: dict[
        tuple[str, str, str], list[float]
    ] = defaultdict(list)
    pairwise_by_type_task: dict[
        tuple[str, str, str], list[float]
    ] = defaultdict(list)
    margin_by_type_task: dict[
        tuple[str, str, str], list[float]
    ] = defaultdict(list)
    records = []
    for (domain, task_id, episode_id), candidates in sorted(episodes.items()):
        correct_rows = [row for row in candidates if row[1]]
        negative_rows = [row for row in candidates if not row[1]]
        if len(correct_rows) != 1 or not negative_rows:
            raise VMBMKError(
                f"TGA-{variant} {task_id}/{episode_id} needs exactly one correct "
                "instruction and at least one negative"
            )
        correct_score = correct_rows[0][2]
        credits = []
        public_scores: dict[str, float] = {}
        public_types: dict[str, str] = {}
        for candidate_id, _, score in negative_rows:
            credit = (
                1.0
                if correct_score > score
                else 0.5
                if correct_score == score
                else 0.0
            )
            credits.append(credit)
            public_id = _public_candidate_id(candidate_id, variant)
            public_scores[public_id] = score
            candidate_type = _candidate_type(candidate_id, variant)
            if candidate_type is not None:
                public_types[public_id] = candidate_type
            candidate_key = (candidate_id, domain, task_id)
            pairwise_by_candidate_task[candidate_key].append(credit)
            margin_by_candidate_task[candidate_key].append(correct_score - score)
            if candidate_type is not None:
                type_key = (candidate_type, domain, task_id)
                pairwise_by_type_task[type_key].append(credit)
                margin_by_type_task[type_key].append(correct_score - score)
        pairwise_accuracy = mean(credits)
        strict_top1 = correct_score > max(score for _, _, score in negative_rows)
        key = (domain, task_id)
        pairwise_by_task[key].append(pairwise_accuracy)
        top1_by_task[key].append(float(strict_top1))
        records.append(
            {
                "domain": domain,
                "task_id": task_id,
                "episode_id": episode_id,
                "correct_gain": correct_score,
                "correct_score": correct_score,
                "negative_gains": {
                    candidate_id: score
                    for candidate_id, _, score in negative_rows
                },
                "counterfactual_scores": public_scores,
                "counterfactual_types": public_types,
                "pairwise_accuracy": pairwise_accuracy,
                "strict_top1": strict_top1,
            }
        )

    pairwise_tasks = {
        key: mean(values) for key, values in sorted(pairwise_by_task.items())
    }
    top1_tasks = {
        key: mean(values) for key, values in sorted(top1_by_task.items())
    }
    pairwise_summary = summarize_task_scores(pairwise_tasks)
    top1_summary = summarize_task_scores(top1_tasks)

    diagnostics = _candidate_diagnostics(
        variant, episodes, pairwise_by_candidate_task, margin_by_candidate_task,
        pairwise_by_type_task, margin_by_type_task,
    )

    return {
        "protocol": f"trajectory-gain-edge2pct-{variant}-v1",
        "primary": "strict_top1_accuracy",
        "mean": top1_summary["mean"],
        "domains": top1_summary["domains"],
        "pairwise_mean": pairwise_summary["mean"],
        "pairwise_domains": pairwise_summary["domains"],
        **diagnostics,
        "episodes": records,
        "tie_credit": 0.5,
        "edge_fraction": 0.02,
    }


def _run_tga(
    data_root: str | Path,
    model: Mapping[str, Any],
    task_ids: Sequence[str],
    operation_path: str | Path,
    gpu: int,
    *,
    mode: str,
    domains: Sequence[str],
    variant: TGAVariant,
    candidate_task_ids: Sequence[str] | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    dataset = dataset if dataset is not None else Dataset.load(data_root, metrics=[f"tga_{variant}"])
    queries = build_tga_queries(
        dataset,
        task_ids,
        domains,
        variant=variant,
        candidate_task_ids=candidate_task_ids,
    )
    results = run_queries(
        data_root,
        queries,
        model,
        operation_path,
        gpu,
        metric=f"tga_{variant}",
        mode=mode,
        native_metric="tga",
    )
    return score_tga(
        dataset,
        results,
        task_ids,
        domains,
        variant=variant,
        candidate_task_ids=candidate_task_ids,
    )


def run_tga_easy(
    data_root: str | Path,
    model: Mapping[str, Any],
    task_ids: Sequence[str],
    operation_path: str | Path,
    gpu: int,
    *,
    mode: str,
    domains: Sequence[str],
    candidate_task_ids: Sequence[str] | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    return _run_tga(
        data_root,
        model,
        task_ids,
        operation_path,
        gpu,
        mode=mode,
        domains=domains,
        variant="easy",
        dataset=dataset,
        candidate_task_ids=candidate_task_ids,
    )


def run_tga_hard(
    data_root: str | Path,
    model: Mapping[str, Any],
    task_ids: Sequence[str],
    operation_path: str | Path,
    gpu: int,
    *,
    mode: str,
    domains: Sequence[str],
    candidate_task_ids: Sequence[str] | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    return _run_tga(
        data_root,
        model,
        task_ids,
        operation_path,
        gpu,
        mode=mode,
        domains=domains,
        variant="hard",
        dataset=dataset,
        candidate_task_ids=candidate_task_ids,
    )
