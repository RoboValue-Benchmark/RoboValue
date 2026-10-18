"""Reproduce saved metric scores from operations without inference or I/O."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset
from vmbmk.inference.queries import Result
from vmbmk.metrics.registry import build_metric_queries, selected_task_ids
from vmbmk.metrics.utils.aggregation import summarize_domain_tasks
from vmbmk.runner.run import RunConfig


def _recomputed_score(
    metric: str, dataset: Dataset, results: Sequence[Result], tasks: Sequence[str],
    domains: Sequence[str], config: RunConfig,
) -> dict[str, Any]:
    if metric == "sa":
        from vmbmk.metrics.sa import _score_sa_details

        scores, terminal = _score_sa_details(dataset, results, tasks, domains)
        return {
            **summarize_domain_tasks(scores), "gap": terminal["gap"],
            "terminal_values": {key: terminal[key] for key in (
                "success_macro_mean", "failure_macro_mean", "range", "gap_over_range_percent",
            )},
        }
    if metric in {"tga_easy", "tga_hard"}:
        from vmbmk.metrics.tga import score_tga

        return score_tga(
            dataset, results, tasks, domains, variant=metric.removeprefix("tga_"),
            candidate_task_ids=(config.candidate_tasks or {}).get(metric),
        )
    if metric == "voc":
        from vmbmk.metrics.cycle_voc_vs.forward import score_voc_by_domain

        return summarize_domain_tasks(score_voc_by_domain(dataset, results, tasks, domains))
    if metric == "voc_mem":
        from vmbmk.metrics.mem_voc import score_voc_mem_by_domain

        return summarize_domain_tasks(score_voc_mem_by_domain(dataset, results, tasks, domains))
    if metric == "cycle_voc":
        from vmbmk.metrics.cycle_voc_vs.cycle import (
            CYCLE_COMPONENT_PROTOCOL, score_cycle_voc_components,
        )

        components = score_cycle_voc_components(dataset, results, tasks, domains)
        return {
            **components["cycle"],
            "components": {
                "protocol": CYCLE_COMPONENT_PROTOCOL,
                "forward": components["forward"], "reverse": components["reverse"],
            },
        }
    if metric == "vs":
        from vmbmk.metrics.cycle_voc_vs.planning import plan_cycle
        from vmbmk.metrics.cycle_voc_vs.vs import score_vs

        settings = {name: domain for name, _, domain in config.metrics}
        plan = None
        if "cycle_voc" in settings:
            requested = (config.metric_tasks or {}).get("cycle_voc", config.tasks)
            cycle_tasks = selected_task_ids("cycle_voc", dataset, requested, settings["cycle_voc"])
            plan = plan_cycle(dataset, cycle_tasks, settings["cycle_voc"])
        return score_vs(dataset, results, tasks, domains, cycle_plan=plan)
    if metric == "fpl":
        from vmbmk.metrics.fpl import score_fpl

        return score_fpl(dataset, results, tasks, domains)
    if metric == "trr":
        from vmbmk.metrics.trr import score_trr_timeweighted

        return score_trr_timeweighted(dataset, results, tasks, domains)
    if metric == "csvc":
        from vmbmk.metrics.csvc import score_csvc

        return score_csvc(dataset, results, tasks, domains)
    raise ValueError(f"score validation is unsupported for {metric}")


def validate_metric_scores(
    config: RunConfig, dataset: Dataset, metrics: Mapping[str, Any],
    results: Sequence[Result],
) -> None:
    """Check numerical artifacts using the original pure metric implementations.

    SIA is validated separately against its saved responses and judge decisions.
    A partial metric mapping is supported for interrupted-run checkpoints.
    """
    indexed = {result.query_id: result for result in results}
    if len(indexed) != len(results):
        raise ValueError("saved predictions contain duplicate query IDs")
    plans = []
    expected_ids = set()
    for metric, _, domains in config.metrics:
        if metric not in metrics:
            continue
        requested = (config.metric_tasks or {}).get(metric, config.tasks)
        tasks = selected_task_ids(metric, dataset, requested, domains)
        if metric == "sia":
            from vmbmk.metrics.sia.evaluation import build_sia_queries

            queries = build_sia_queries(dataset, tasks, domains, seed=0)
        else:
            queries = build_metric_queries(
                metric, dataset, tasks, domains,
                candidate_tasks=(config.candidate_tasks or {}).get(metric),
            )
        expected_ids.update(query.query_id for query in queries)
        plans.append((metric, tasks, domains, queries))
    if set(indexed) != expected_ids:
        raise ValueError("saved predictions have missing or unexpected query IDs")
    for metric, tasks, domains, queries in plans:
        if metric == "sia":
            continue
        predictions = [indexed[query.query_id] for query in queries]
        expected = _recomputed_score(metric, dataset, predictions, tasks, domains, config)
        expected = json.loads(json.dumps(expected, allow_nan=False))
        for key, value in expected.items():
            if metrics[metric].get(key) != value:
                raise ValueError(f"{metric}.{key}: score disagrees with saved predictions")
