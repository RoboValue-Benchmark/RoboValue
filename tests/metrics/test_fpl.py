from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synthetic_data import make_dataset, set_task_metadata, write_json
from vmbmk.data.dataset import Dataset
from vmbmk.metrics.fpl import (
    FPL_PROTOCOL,
    _select_descent,
    _zigzag_descents,
    build_fpl_queries,
    score_fpl,
)
from vmbmk.inference.queries import Result





class FPLTest(unittest.TestCase):
    def test_fpl_uses_zigzag_onset_and_normalized_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("failure", False)])
            task = set_task_metadata(data, ["FPL"])
            write_json(
                task / "episodes" / "failure" / "annotation.json",
                {"fpl": {"failure_start": 40, "failure_end": 50}},
            )
            dataset = Dataset.load(data)
            queries = build_fpl_queries(dataset)
            self.assertTrue(all(query.op == "value" for query in queries))
            self.assertEqual(
                [query.state.anchor_frame for query in queries],
                [*range(0, 100, 2), 99],
            )
            def value(frame: int) -> float:
                if frame <= 30:
                    return 1.0
                if frame == 34:
                    return 0.65
                if frame <= 54:
                    return 0.6
                return 0.2 if frame == 56 else 0.4

            results = [
                Result(query.query_id, "value", value(query.state.anchor_frame))
                for query in queries
            ]
            scored = score_fpl(dataset, results)
            self.assertEqual(scored["protocol"], FPL_PROTOCOL)
            self.assertEqual(scored["primary"], "normalized_onset_error_lower_is_better")
            self.assertEqual(scored["zigzag_alpha"], 0.2)
            self.assertEqual(scored["sampling"], "uniform-from-frame-zero")
            self.assertEqual(scored["sample_rate_hz"], 5.0)
            self.assertEqual(scored["sample_period_s"], 0.2)
            self.assertEqual(scored["episodes"][0]["predicted_onset_frame"], 30)
            self.assertAlmostEqual(scored["episodes"][0]["maximum_drawdown"], 0.8)
            self.assertAlmostEqual(scored["episodes"][0]["zigzag_threshold"], 0.16)
            self.assertAlmostEqual(scored["mean"], 10 / 99)
            self.assertEqual(
                scored["episodes"][0]["sampled_value_curve"][0],
                {"frame_index": 0, "value": 1.0},
            )
            self.assertEqual(
                scored["episodes"][0]["sampled_value_curve"][-1],
                {"frame_index": 99, "value": 0.4},
            )

    def test_fpl_excludes_invalid_recovery_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(
                Path(directory),
                [("single_failure", False), ("invalid_recovery", False)],
            )
            task = set_task_metadata(data, ["FPL"])
            write_json(
                task / "episodes" / "single_failure" / "annotation.json",
                {"fpl": {"failure_start": 40, "failure_end": 50}},
            )
            invalid_metadata = (
                task / "episodes" / "invalid_recovery" / "metadata.json"
            )
            row = json.loads(invalid_metadata.read_text(encoding="utf-8"))
            row["trr"] = {"group_id": "recovery_001", "role": "r_zero"}
            write_json(invalid_metadata, row)
            write_json(
                task / "episodes" / "invalid_recovery" / "annotation.json",
                {
                    "fpl": {"failure_start": 20, "failure_end": 40},
                    "trr": {
                        "failure_span": {
                            "start_frame": 20,
                            "end_frame_exclusive": 40,
                        },
                        "recovery_failed_span": {
                            "start_frame": 40,
                            "end_frame_exclusive": 60,
                        },
                    },
                },
            )
            for metrics in (None, ["fpl"]):
                with self.subTest(metrics=metrics):
                    dataset = Dataset.load(data, metrics=metrics)
                    self.assertEqual(
                        dataset.episode("task_001", "invalid_recovery").trr_role,
                        "r_zero",
                    )
                    queries = build_fpl_queries(dataset)
                    self.assertTrue(
                        all(query.state.episode_id == "single_failure" for query in queries)
                    )
                    results = [Result(query.query_id, "value", 0.0) for query in queries]
                    results.append(
                        Result("fpl:task_001:invalid_recovery:0", "value", 0.0)
                    )
                    scored = score_fpl(dataset, results)
                    self.assertEqual(
                        [episode["episode_id"] for episode in scored["episodes"]],
                        ["single_failure"],
                    )

    def test_fpl_scoped_loading_retains_successful_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("successful_recovery", True)])
            task = set_task_metadata(data, ["FPL"])
            metadata = task / "episodes" / "successful_recovery" / "metadata.json"
            row = json.loads(metadata.read_text(encoding="utf-8"))
            row["trr"] = {"group_id": "recovery_001", "role": "r_plus"}
            write_json(metadata, row)
            write_json(
                metadata.with_name("annotation.json"),
                {"fpl": {"failure_start": 20, "failure_end": 40}, "trr": "unused"},
            )
            dataset = Dataset.load(data, metrics=["fpl"])
            self.assertEqual(
                dataset.episode("task_001", "successful_recovery").trr_role,
                "r_plus",
            )
            queries = build_fpl_queries(dataset)
            self.assertTrue(queries)
            scored = score_fpl(
                dataset, [Result(query.query_id, "value", 0.0) for query in queries]
            )
            self.assertEqual(
                [episode["episode_id"] for episode in scored["episodes"]],
                ["successful_recovery"],
            )

    def test_completed_fpl_rejects_legacy_cohort_protocol(self) -> None:
        from vmbmk.tools.results.publication import _merge_metric_result
        from vmbmk.tools.results.validation import _validate_metric_coverage

        score = {
            "protocol": "zigzag-onset-normalized-alpha0.2-v1",
            "mean": 0.5,
            "domains": {"id": {"mean": 0.5, "tasks": {"task_001": 0.5}}},
        }
        coverage = {"fpl": {"id": {"task_001"}}}
        with self.assertRaisesRegex(ValueError, "FPL result uses an obsolete cohort protocol"):
            _validate_metric_coverage({"fpl": score}, coverage)
        with self.assertRaisesRegex(ValueError, "obsolete cohort protocol"):
            _merge_metric_result("fpl", score, score)
        score["protocol"] = FPL_PROTOCOL
        _validate_metric_coverage({"fpl": score}, coverage)
        self.assertEqual(_merge_metric_result("fpl", score, score)["protocol"], FPL_PROTOCOL)

    def test_fpl_resume_rejects_legacy_cohort_protocol(self) -> None:
        from vmbmk.errors import ConfigurationError
        from vmbmk.runner.run import _read_metrics

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "metrics.json"
            write_json(checkpoint, {"fpl": {"protocol": "zigzag-onset-normalized-alpha0.2-v1"}})
            with self.assertRaisesRegex(ConfigurationError, "obsolete cohort protocol"):
                _read_metrics(checkpoint, {"fpl"}, "test")
            write_json(checkpoint, {"fpl": {"protocol": FPL_PROTOCOL}})
            self.assertEqual(_read_metrics(checkpoint, {"fpl"}, "test")["fpl"]["protocol"], FPL_PROTOCOL)

    def test_fpl_publication_rejects_legacy_protocol_before_writing(self) -> None:
        from vmbmk.tools.results.publication import PublicationRequest, publish_result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            write_json(config, {"model": "topreward"})
            score = {
                "protocol": "zigzag-onset-normalized-alpha0.2-v1",
                "mean": 0.5,
                "domains": {"id": {"mean": 0.5, "tasks": {"task_001": 0.5}}},
            }
            job = PublicationRequest("topreward_qwen32b", config, "test", {"fpl": score})
            output = root / "published"
            with self.assertRaisesRegex(ValueError, "obsolete cohort protocol"):
                publish_result(job, output)
            self.assertFalse(output.exists())
            score["protocol"] = FPL_PROTOCOL
            publish_result(job, output)
            self.assertTrue((output / "index.json").is_file())

    def test_fpl_zigzag_ties_use_shortest_span_then_earliest_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("failure", False)])
            task = set_task_metadata(data, ["FPL"])
            write_json(
                task / "episodes" / "failure" / "annotation.json",
                {"fpl": {"failure_start": 70, "failure_end": 80}},
            )
            dataset = Dataset.load(data)
            queries = build_fpl_queries(dataset)
            results = []
            for query in queries:
                frame = query.state.anchor_frame
                if frame <= 30 or frame == 42:
                    value = 1.0
                else:
                    value = 0.5
                results.append(Result(query.query_id, "value", value))
            scored = score_fpl(dataset, results)
            episode = scored["episodes"][0]
            self.assertEqual(episode["selected_descent"], [30, 32, 0.5])
            self.assertEqual(episode["predicted_onset_frame"], 30)
            self.assertAlmostEqual(scored["mean"], 40 / 99)

    def test_fpl_zigzag_keeps_small_rebounds_and_terminal_declines(self) -> None:
        curve = [(0, 1.0), (2, 0.6), (4, 0.68), (6, 0.2), (8, 0.5), (10, 0.45)]
        descents = _zigzag_descents(curve, 0.16)
        self.assertEqual([row[:2] for row in descents], [(0, 6), (8, 10)])
        self.assertAlmostEqual(descents[0][2], 0.8)
        self.assertAlmostEqual(descents[1][2], 0.05)
        self.assertEqual(_select_descent(descents), (0, 6, 0.8))

    def test_fpl_zigzag_confirms_reversal_at_the_threshold(self) -> None:
        curve = [(0, 1.0), (2, 0.6), (4, 0.8)]
        descents = _zigzag_descents(curve, 0.2)
        self.assertEqual([row[:2] for row in descents], [(0, 2)])
        self.assertAlmostEqual(descents[0][2], 0.4)

    def test_fpl_descent_ties_use_span_then_start_time(self) -> None:
        descents = [(10, 20, 0.5), (30, 35, 0.5), (5, 10, 0.5)]
        self.assertEqual(_select_descent(descents), (5, 10, 0.5))

    def test_fpl_no_decline_uses_farthest_boundary_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("failure", False)])
            task = set_task_metadata(data, ["FPL"])
            write_json(
                task / "episodes" / "failure" / "annotation.json",
                {"fpl": {"failure_start": 40, "failure_end": 50}},
            )
            dataset = Dataset.load(data)
            queries = build_fpl_queries(dataset)
            results = [
                Result(query.query_id, "value", float(index))
                for index, query in enumerate(queries)
            ]
            scored = score_fpl(dataset, results)
            episode = scored["episodes"][0]
            self.assertIsNone(episode["predicted_onset_frame"])
            self.assertIsNone(episode["selected_descent"])
            self.assertAlmostEqual(scored["mean"], 59 / 99)

    def test_fpl_macro_averages_episodes_then_tasks_then_domains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(
                Path(directory),
                [("id_a", False), ("id_b", False), ("env_a", False)],
            )
            task = set_task_metadata(data, ["FPL"])
            for episode_id in ("id_a", "id_b", "env_a"):
                write_json(
                    task / "episodes" / episode_id / "annotation.json",
                    {
                        "fpl": {
                            "failure_start": 0 if episode_id == "env_a" else 40,
                            "failure_end": 50,
                        }
                    },
                )
            env_metadata = task / "episodes" / "env_a" / "metadata.json"
            row = json.loads(env_metadata.read_text(encoding="utf-8"))
            row["domain"] = "env"
            write_json(env_metadata, row)

            dataset = Dataset.load(data)
            queries = build_fpl_queries(dataset)
            results = []
            for query in queries:
                frame = query.state.anchor_frame
                value = (
                    float(frame)
                    if query.state.episode_id == "env_a"
                    else (1.0 if frame <= 40 else 0.0)
                )
                results.append(Result(query.query_id, "value", value))
            scored = score_fpl(dataset, results)
            self.assertAlmostEqual(scored["domains"]["id"]["mean"], 0.0)
            self.assertAlmostEqual(scored["domains"]["env"]["mean"], 1.0)
            self.assertAlmostEqual(scored["mean"], 0.5)

    def test_fpl_uses_a_time_uniform_grid_at_non_divisible_frame_rates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("failure", False)])
            task = set_task_metadata(data, ["FPL"])
            write_json(
                task / "episodes" / "failure" / "annotation.json",
                {"fpl": {"failure_start": 40, "failure_end": 50}},
            )
            metadata_path = task / "episodes" / "failure" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["fps"] = 12.0
            write_json(metadata_path, metadata)
            queries = build_fpl_queries(Dataset.load(data))
            self.assertEqual(
                [query.state.anchor_frame for query in queries[:6]],
                [0, 2, 5, 7, 10, 12],
            )
