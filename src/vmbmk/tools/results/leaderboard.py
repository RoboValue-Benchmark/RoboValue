"""Export the paper's leaderboard from versioned, already-scored artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from vmbmk.metrics.csvc import CSVC_PROTOCOL
from vmbmk.metrics.fpl import FPL_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.forward import VOC_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.cycle import CYCLE_COMPONENT_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.vs import VS_PROTOCOL
from vmbmk.metrics.sia.scoring import SIA_PROTOCOL


INPUT_SCHEMA = "robovalue-leaderboard-input-v1"
OUTPUT_SCHEMA = "robovalue-leaderboard-v1"
PROTOCOL = "paper-f3-equal-world-v1"
METRICS = ("sa", "tga_easy", "tga_hard", "voc", "voc_mem", "fpl", "trr", "vs", "csvc", "vroc")
VERSIONED = {"fpl": FPL_PROTOCOL, "csvc": CSVC_PROTOCOL, "voc": VOC_PROTOCOL, "vs": VS_PROTOCOL, "sia": SIA_PROTOCOL}


def _domain_scores(metric: str, result: Mapping[str, Any]) -> tuple[dict[str, float | None], dict[str, list[str]]]:
    if metric in VERSIONED and result.get("protocol") != VERSIONED[metric]:
        raise ValueError(f"{metric}: incompatible metric protocol")
    domains = result.get("domains")
    if metric == "csvc" and domains is None:
        domains = {"id": {"mean": result["score"], "tasks": {task: row["score"] for task, row in result["task_results"].items()}}}
    if not isinstance(domains, dict) or not domains or set(domains) - {"id", "env", "emb"}:
        raise ValueError(f"{metric}: invalid domain coverage")
    if metric in {"trr", "csvc"} and set(domains) != {"id"}:
        raise ValueError(f"{metric}: the paper protocol is ID-only")
    if metric not in {"trr", "csvc", "sia"} and set(domains) != {"id", "env", "emb"}:
        raise ValueError(f"{metric}: incomplete leaderboard domain coverage; provide complete ID/ENV/EMB artifacts")
    means, coverage = {}, {}
    lower = -1 if metric in {"voc", "voc_mem", "vroc", "csvc", "cycle_voc"} else 0
    for domain, row in domains.items():
        tasks = row["tasks"]
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError(f"{metric}/{domain}: empty task coverage")
        valid = {}
        for task, value in tasks.items():
            if value is None and metric == "csvc":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= 1:
                raise ValueError(f"{metric}/{domain}/{task}: invalid fractional score")
            valid[task] = float(value)
        mean = sum(valid.values()) / len(valid) if valid else None
        reported = row["mean"]
        if mean is None:
            if reported is not None:
                raise ValueError(f"{metric}/{domain}: missing task scores disagree with mean")
        elif isinstance(reported, bool) or not isinstance(reported, (int, float)) or not math.isclose(mean, reported, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{metric}/{domain}: task macro disagrees with reported mean")
        means[domain], coverage[domain] = mean, sorted(valid)
    return means, coverage


def score_baseline(worlds: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Apply Appendix F.3 without pooling tasks across worlds or rounding."""
    if set(worlds) != {"sim", "real"}:
        raise ValueError("leaderboard input requires separate sim and real worlds")
    raw_worlds, coverage = {}, {}
    for world, artifacts in worlds.items():
        if set(artifacts) - (set(METRICS) | {"cycle_voc", "sia"}):
            raise ValueError(f"{world}: unknown leaderboard metric")
        raw_worlds[world], coverage[world] = {}, {}
        for metric, result in artifacts.items():
            if metric == "cycle_voc":
                cycle, cycle_tasks = _domain_scores(metric, result)
                raw_worlds[world][metric], coverage[world][metric] = cycle, cycle_tasks
                halves = result.get("components", {})
                if halves.get("protocol") != CYCLE_COMPONENT_PROTOCOL:
                    raise ValueError("rescore cached Cycle-VOC predictions to export verified VROC components")
                forward, forward_tasks = _domain_scores("vroc", halves["forward"])
                reverse, reverse_tasks = _domain_scores("vroc", halves["reverse"])
                if cycle_tasks != forward_tasks or cycle_tasks != reverse_tasks:
                    raise ValueError("Cycle-VOC half coverage mismatch")
                for domain in cycle:
                    for task in cycle_tasks[domain]:
                        expected = (halves["forward"]["domains"][domain]["tasks"][task] + halves["reverse"]["domains"][domain]["tasks"][task]) / 2
                        if not math.isclose(expected, result["domains"][domain]["tasks"][task], rel_tol=1e-12, abs_tol=1e-12):
                            raise ValueError("Cycle-VOC components disagree with combined score")
                raw_worlds[world]["vroc"], coverage[world]["vroc"] = reverse, reverse_tasks
            elif metric == "vroc":
                raise ValueError("VROC must come from verified continuous Cycle-VOC components")
            else:
                raw_worlds[world][metric], coverage[world][metric] = _domain_scores(metric, result)
    reported = METRICS + tuple(
        metric for metric in ("cycle_voc", "sia")
        if any(metric in raw_worlds[world] for world in ("sim", "real"))
    )
    raw = {}
    for metric in reported:
        domains = ("id",) if metric in {"trr", "csvc"} else ("id", "env", "emb")
        raw[metric] = {}
        for domain in domains:
            values = [raw_worlds[world].get(metric, {}).get(domain) for world in ("sim", "real")]
            raw[metric][domain] = None if any(value is None for value in values) else sum(values) * 50
    normalized, missing = {}, []
    for metric in METRICS:
        values = raw[metric]
        if any(value is None for value in values.values()):
            normalized[metric] = 0.0
            missing.append(metric)
            continue
        if metric in {"trr", "csvc"}:
            combined = values["id"]
        elif metric == "vroc":
            combined = sum(values.values()) / 3
        else:
            combined = values["id"] / 2 + values["env"] / 4 + values["emb"] / 4
        if metric in {"voc", "voc_mem", "vroc", "csvc"}:
            normalized[metric] = (combined + 100) / 2
        elif metric == "fpl":
            normalized[metric] = 100 - combined
        else:
            normalized[metric] = combined
    dimensions = {
        "task_state_understanding": (normalized["sa"] + (normalized["tga_easy"] + normalized["tga_hard"]) / 2) / 2,
        "temporal_progress_monitoring": (normalized["voc"] + normalized["voc_mem"]) / 4 + normalized["vroc"] / 2,
        "failure_and_recovery_reasoning": (normalized["fpl"] + normalized["trr"]) / 2,
        "value_consistency": (normalized["vs"] + normalized["csvc"]) / 2,
    }
    return {"raw_worlds": raw_worlds, "task_coverage": coverage, "raw_equal_world_x100": raw,
            "normalized_components_x100": normalized, "missing_components": missing,
            "dimensions_x100": dimensions, "overall_x100": sum(dimensions.values()) / 4}


def export_leaderboard(manifest_path: Path) -> dict[str, Any]:
    """Load explicit artifact references and retain source hashes and protocols."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("incompatible leaderboard input schema")
    entries, seen = [], set()
    for entry in manifest["baselines"]:
        baseline, track = entry["baseline"], entry["track"]
        if not isinstance(baseline, str) or not baseline or not isinstance(track, str) or not track or (baseline, track) in seen:
            raise ValueError("baseline and explicit track must be nonempty and unique")
        seen.add((baseline, track))
        worlds, sources = {}, {}
        for world, references in entry["worlds"].items():
            worlds[world], sources[world] = {}, {}
            for metric, reference in references.items():
                path = (manifest_path.parent / reference).resolve()
                payload = path.read_bytes()
                saved = json.loads(payload)
                if "result" in saved:
                    if saved.get("metric") != metric or saved.get("baseline") != baseline:
                        raise ValueError("published artifact identity disagrees with manifest")
                    result = saved["result"]
                else:
                    result = saved[metric]
                worlds[world][metric] = result
                sources[world][metric] = {"path": reference, "sha256": hashlib.sha256(payload).hexdigest(), "protocol": result.get("protocol")}
        entries.append({"baseline": baseline, "track": track, "sources": sources, **score_baseline(worlds)})
    if not entries:
        raise ValueError("leaderboard manifest has no baselines")
    rankings = {track: [entry["baseline"] for entry in sorted((entry for entry in entries if entry["track"] == track), key=lambda entry: (-entry["overall_x100"], entry["baseline"]))] for track in sorted({entry["track"] for entry in entries})}
    return {"schema_version": OUTPUT_SCHEMA, "protocol": PROTOCOL, "baselines": entries, "rankings_by_track": rankings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export_leaderboard(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return 0
