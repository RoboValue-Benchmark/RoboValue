"""Evaluation eligibility and workload constraints, separate from GPU scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vmbmk.adapters.registry import canonical_baseline


@dataclass(frozen=True)
class EvaluationPolicy:
    """Recorded restrictions; absence of a restriction is not runtime certification."""

    excluded_metrics: frozenset[str] = frozenset()
    only_metrics: frozenset[str] | None = None
    excluded_tasks: frozenset[str] = frozenset()
    task_metrics: frozenset[str] = frozenset()
    batch_size: int | None = None
    batch_metrics: frozenset[str] = frozenset()
    requires_task_checkpoint: bool = False

    def reason(self, config: Mapping[str, Any], metric: str, task: str) -> str | None:
        """Explain an unsupported combination before querying a dataset/model."""
        if metric in self.excluded_metrics or (
            self.only_metrics is not None and metric not in self.only_metrics
        ):
            return f"{config['model']} does not support {metric} evaluation"
        if metric in self.task_metrics and task in self.excluded_tasks:
            return "no task-specific ProcVLM one-shot LoRA is available"
        if self.requires_task_checkpoint:
            checkpoints = (config.get("model_options") or {}).get("checkpoint_by_task") or {}
            if task not in checkpoints:
                return f"no task checkpoint configured for {task}"
        return None

    def batch_override(self, metrics: tuple[str, ...]) -> int | None:
        """Return a recorded workload constraint, not an automatic GPU estimate."""
        return self.batch_size if self.batch_metrics.intersection(metrics) else None


_POLICIES = {
    "failsafe": EvaluationPolicy(only_metrics=frozenset({"sia"})),
    "roboreward": EvaluationPolicy(excluded_metrics=frozenset({"voc", "voc_mem", "cycle_voc"})),
    "topreward": EvaluationPolicy(batch_size=2, batch_metrics=frozenset({"tga_easy", "tga_hard"})),
    "procvlm_one-shot": EvaluationPolicy(
        excluded_tasks=frozenset({"press_by_number", "swap_blocks"}),
        task_metrics=frozenset({"voc", "voc_mem"}), requires_task_checkpoint=True,
    ),
}


def evaluation_policy(config: Mapping[str, Any]) -> EvaluationPolicy:
    """Resolve effective model mode independently of config filename or checkpoint size."""
    if config.get("backend") == "remote_api":
        return EvaluationPolicy()
    identity = canonical_baseline(dict(config))
    return _POLICIES.get(identity, _POLICIES.get(str(config.get("model")), EvaluationPolicy()))
