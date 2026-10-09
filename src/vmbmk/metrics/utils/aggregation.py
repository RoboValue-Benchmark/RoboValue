from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping
from vmbmk.errors import VMBMKError
from .statistics import mean


def summarize_task_scores(
    task_scores: Mapping[tuple[str, str], float]
) -> dict[str, Any]:
    if not task_scores:
        raise VMBMKError("metric produced no task scores")
    by_domain: dict[str, dict[str, float]] = defaultdict(dict)
    for (domain, task_id), score in sorted(task_scores.items()):
        by_domain[domain][task_id] = float(score)
    domains = {
        domain: {"mean": mean(list(tasks.values())), "tasks": tasks}
        for domain, tasks in sorted(by_domain.items())
    }
    return {
        "mean": mean([row["mean"] for row in domains.values()]),
        "domains": domains,
    }

def summarize_domain_tasks(scores: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Preserve insertion-order macro averaging for VOC-family run outputs."""
    domains = {
        domain: {"mean": sum(tasks.values()) / len(tasks), "tasks": tasks}
        for domain, tasks in scores.items()
    }
    return {
        "mean": sum(item["mean"] for item in domains.values()) / len(domains),
        "domains": domains,
    }
