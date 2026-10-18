from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from vmbmk.metrics.fpl import FPL_PROTOCOL
from vmbmk.metrics.csvc import CSVC_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.forward import VOC_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.vs import VS_PROTOCOL
from vmbmk.tools.results.leaderboard import INPUT_SCHEMA, score_baseline, export_leaderboard, main
from vmbmk.tools.results.publication import _merge_metric_result


def result(value: float, domains=("id", "env", "emb"), protocol=None) -> dict:
    row = {"mean": value, "domains": {domain: {"mean": value, "tasks": {"task": value}} for domain in domains}}
    if protocol is not None:
        row["protocol"] = protocol
    return row


def worlds() -> dict:
    metrics = {
        "sa": result(0.6), "tga_easy": result(0.8), "tga_hard": result(0.4),
        "voc": result(0.2, protocol=VOC_PROTOCOL), "voc_mem": result(0.4),
        "fpl": result(0.3, protocol=FPL_PROTOCOL), "trr": result(0.5, ("id",)),
        "vs": result(0.8, protocol=VS_PROTOCOL), "csvc": result(-0.2, ("id",), CSVC_PROTOCOL),
        "cycle_voc": {**result(0.4), "components": {"protocol": "continuous-cycle-halves-v1", "forward": result(0.2), "reverse": result(0.6)}},
    }
    return {"sim": deepcopy(metrics), "real": deepcopy(metrics)}


class LeaderboardTest(unittest.TestCase):
    def test_rejects_partial_domain_artifact_instead_of_zeroing_valid_id_score(self):
        inputs = worlds()
        inputs["sim"]["sa"] = result(0.8, ("id",))
        with self.assertRaisesRegex(ValueError, "domain coverage"):
            score_baseline(inputs)

    def test_incremental_cycle_publication_preserves_both_half_cohorts(self):
        existing = worlds()["sim"]["cycle_voc"]
        incoming = {**result(0.0), "components": {"protocol": "continuous-cycle-halves-v1", "forward": result(1.0), "reverse": result(-1.0)}}
        for record in (incoming, incoming["components"]["forward"], incoming["components"]["reverse"]):
            for domain in record["domains"].values():
                domain["tasks"] = {"other": domain["mean"]}
        merged = _merge_metric_result("cycle_voc", existing, incoming)
        inputs = worlds()
        inputs["sim"]["cycle_voc"] = merged
        self.assertEqual(score_baseline(inputs)["task_coverage"]["sim"]["vroc"]["id"], ["other", "task"])
        del incoming["components"]
        with self.assertRaisesRegex(ValueError, "unverified Cycle-VOC"):
            _merge_metric_result("cycle_voc", existing, incoming)

    def test_independent_dimension_reference_and_no_duplicate_cycle_weight(self):
        scored = score_baseline(worlds())
        self.assertEqual(scored["missing_components"], [])
        self.assertEqual(scored["dimensions_x100"], {"task_state_understanding": 60.0, "temporal_progress_monitoring": 72.5, "failure_and_recovery_reasoning": 60.0, "value_consistency": 60.0})
        self.assertEqual(scored["overall_x100"], 63.125)

    def test_equal_world_macro_not_pooled_tasks_and_vroc_equal_domain_weight(self):
        inputs = worlds()
        inputs["sim"]["sa"]["domains"]["id"] = {"mean": 0.2, "tasks": {"a": 0.1, "b": 0.3}}
        inputs["real"]["sa"]["domains"]["id"] = {"mean": 0.8, "tasks": {"c": 0.8}}
        for world in inputs.values():
            for domain, reverse in (("id", 0.0), ("env", 0.3), ("emb", 0.9)):
                world["cycle_voc"]["components"]["reverse"]["domains"][domain] = {"mean": reverse, "tasks": {"task": reverse}}
                world["cycle_voc"]["domains"][domain] = {"mean": (0.2 + reverse) / 2, "tasks": {"task": (0.2 + reverse) / 2}}
        scored = score_baseline(inputs)
        self.assertAlmostEqual(scored["raw_equal_world_x100"]["sa"]["id"], 50)
        self.assertAlmostEqual(scored["normalized_components_x100"]["vroc"], 70)

    def test_missing_is_na_in_raw_and_zero_only_after_normalization(self):
        inputs = worlds()
        for world in inputs.values():
            del world["vs"]
            del world["csvc"]
        scored = score_baseline(inputs)
        self.assertIsNone(scored["raw_equal_world_x100"]["csvc"]["id"])
        self.assertEqual(scored["normalized_components_x100"]["csvc"], 0)
        self.assertEqual(scored["dimensions_x100"]["value_consistency"], 0)

    def test_rejects_obsolete_protocol_inconsistent_macro_and_unverified_vroc(self):
        for mutation in ("protocol", "macro", "halves", "coverage", "range"):
            with self.subTest(mutation=mutation):
                inputs = worlds()
                if mutation == "protocol":
                    inputs["sim"]["fpl"]["protocol"] = "old"
                elif mutation == "macro":
                    inputs["sim"]["sa"]["domains"]["id"]["mean"] = 0.1
                elif mutation == "halves":
                    del inputs["sim"]["cycle_voc"]["components"]
                elif mutation == "coverage":
                    inputs["sim"]["cycle_voc"]["components"]["reverse"]["domains"]["id"]["tasks"] = {"different": 0.6}
                else:
                    inputs["sim"]["sa"]["domains"]["id"]["tasks"]["task"] = float("nan")
                with self.assertRaises(ValueError):
                    score_baseline(inputs)

    def test_manifest_exports_hashes_separate_tracks_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = worlds()
            references = {}
            for world, metrics in inputs.items():
                (root / f"{world}.json").write_text(json.dumps(metrics), encoding="utf-8")
                references[world] = {metric: f"{world}.json" for metric in metrics}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": INPUT_SCHEMA, "baselines": [
                {"baseline": "model-a", "track": "zero-shot", "worlds": references},
                {"baseline": "model-b", "track": "one-shot", "worlds": references},
            ]}), encoding="utf-8")
            exported = export_leaderboard(manifest)
            self.assertEqual(exported["rankings_by_track"], {"one-shot": ["model-b"], "zero-shot": ["model-a"]})
            self.assertEqual(len(exported["baselines"][0]["sources"]["sim"]["sa"]["sha256"]), 64)
            output = root / "leaderboard.json"
            self.assertEqual(main([str(manifest), "--output", str(output)]), 0)
            with self.assertRaises(FileExistsError):
                main([str(manifest), "--output", str(output)])
