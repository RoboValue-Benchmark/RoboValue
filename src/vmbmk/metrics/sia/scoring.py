"""Pure SIA scoring: average logprobs within task, then aggregate probabilities."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .judge import _candidate_labels, _target_logprob


SIA_PROTOCOL = "candidate-logprob-mean-exp-single-digit-v3"
SIA_AGGREGATION = "task-exp-mean-logprob-domain-macro-v1"
SIA_EPISODE_SELECTION = "all-trajectories-diverse-ordinal-v1"


def score_sia_mappings(
    items: Sequence[Mapping[str, Any]], mapped: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ordered audit records and scores without inference or file I/O."""
    records: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in items:
        query = item["query"]
        result = mapped[query.query_id]
        predicted_id = result["predicted_subtask_id"]
        logprob = _target_logprob(result, item["candidates"], item["target_subtask_id"])
        records.append(
            {
                "query_id": query.query_id,
                "target_subtask_id": item["target_subtask_id"],
                "predicted_subtask_id": predicted_id,
                "candidate_labels": _candidate_labels(item["candidates"]),
                "candidate_logprobs": result["candidate_logprobs"],
                "target_logprob": logprob,
                "score": logprob,
            }
        )
        grouped[item["domain"]][query.state.task_id].append(logprob)
    domains_result = {}
    for domain, tasks in sorted(grouped.items()):
        task_means = {
            task: math.exp(sum(values) / len(values))
            for task, values in sorted(tasks.items())
        }
        domains_result[domain] = {
            "mean": sum(task_means.values()) / len(task_means),
            "tasks": task_means,
        }
    result = {
        "protocol": SIA_PROTOCOL,
        "aggregation": SIA_AGGREGATION,
        "episode_selection": SIA_EPISODE_SELECTION,
        "primary": "score",
        "higher_is_better": True,
        "mean": sum(item["mean"] for item in domains_result.values())
        / len(domains_result),
        "domains": domains_result,
        "items": len(records),
    }

    return records, result
