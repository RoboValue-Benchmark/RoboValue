from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synthetic_data import make_dataset, write_json
from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import Result
from vmbmk.metrics.cycle_voc_vs.cycle import build_cycle_frame_indices, build_cycle_voc_queries, score_cycle_voc, score_cycle_voc_values
from vmbmk.metrics.cycle_voc_vs.forward import build_voc_queries


def make_cycle_dataset(root: Path) -> Path:
    data = make_dataset(root)
    task = data / "task_001"
    metadata_path = task / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["supports"]["metrics"] = ["CYCLE-VOC"]
    write_json(metadata_path, metadata)
    write_json(
        task / "episodes" / "ep_success" / "annotation.json",
        {"voc": [{"frame_index": 0}, {"frame_index": 50}, {"frame_index": 90}]},
    )
    return data


class CycleVOCTest(unittest.TestCase):
    def test_samples_every_ten_frames_and_mirrors_without_peak_duplicate(self) -> None:
        self.assertEqual(
            build_cycle_frame_indices(35, 10),
            [0, 10, 20, 30, 34, 30, 20, 10, 0],
        )

    def test_builds_cycle_without_repeating_peak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_cycle_dataset(root)
            write_json(
                data / "task_001" / "episodes" / "ep_success" / "annotation.json",
                {"voc": [{"frame_index": 11}, {"frame_index": 50}, {"frame_index": 90}]},
            )
            dataset = Dataset.load(data)
            queries = build_cycle_voc_queries(dataset)
            self.assertEqual(len(queries), 5)
            self.assertEqual(
                [query.state.anchor_frame for query in queries],
                [11, 50, 90, 50, 11],
            )
            self.assertEqual(
                [query.playback for query in queries],
                ["forward"] * 3 + ["cycle"] * 2,
            )
            self.assertEqual(len({query.query_id for query in queries}), 5)

    def test_excludes_diverse_and_trr_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_dataset(
                root,
                episodes=[
                    ("normal", True),
                    ("diverse", True),
                    ("trr", True),
                    ("failure", False),
                ],
            )
            task = data / "task_001"
            metadata_path = task / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["supports"]["metrics"] = ["CYCLE-VOC"]
            write_json(metadata_path, metadata)
            for episode_id in ("normal", "diverse", "trr"):
                write_json(
                    task / "episodes" / episode_id / "annotation.json",
                    {"voc": [{"frame_index": 0}, {"frame_index": 50}, {"frame_index": 90}]},
                )
            diverse_metadata_path = task / "episodes" / "diverse" / "metadata.json"
            diverse_metadata = json.loads(diverse_metadata_path.read_text(encoding="utf-8"))
            diverse_metadata["cspc"] = {"solution_type": "diverse"}
            write_json(diverse_metadata_path, diverse_metadata)
            trr_metadata_path = task / "episodes" / "trr" / "metadata.json"
            trr_metadata = json.loads(trr_metadata_path.read_text(encoding="utf-8"))
            trr_metadata["trr"] = {"group_id": "group_001", "role": "r_plus"}
            write_json(trr_metadata_path, trr_metadata)

            for metrics in (None, ["cycle_voc"], ["voc"]):
                with self.subTest(metrics=metrics):
                    dataset = Dataset.load(data, metrics=metrics)
                    queries = build_cycle_voc_queries(dataset)
                    self.assertEqual(
                        {query.state.episode_id for query in queries}, {"normal"}
                    )
                    self.assertEqual({query.state.episode_id for query in build_voc_queries(dataset)}, {"normal"})
                    self.assertEqual(
                        dataset.episode("task_001", "diverse").cspc_solution_type,
                        "diverse",
                    )
                    self.assertEqual(
                        dataset.episode("task_001", "trr").trr_role, "r_plus"
                    )
                    results = [
                        Result(query.query_id, "value", value)
                        for query, value in zip(queries, [0.0, 0.5, 1.0, 0.5, 0.0])
                    ]
                    self.assertAlmostEqual(score_cycle_voc(dataset, results), 1.0)

    def test_perfect_cycle_is_one(self) -> None:
        details = score_cycle_voc_values([0.0, 0.5, 1.0, 0.5, 0.0])
        self.assertAlmostEqual(details["rho_up"], 1.0)
        self.assertAlmostEqual(details["rho_down"], 1.0)
        self.assertAlmostEqual(details["cycle_voc"], 1.0)

    def test_one_wrong_half_is_penalized(self) -> None:
        up_wrong = score_cycle_voc_values([1.0, 0.5, 0.0, 0.5, 1.0])
        down_wrong = score_cycle_voc_values([0.0, 0.3, 0.6, 1.0, 0.0, 0.2, 0.4])
        self.assertLess(up_wrong["cycle_voc"], 0.1)
        self.assertLess(down_wrong["cycle_voc"], 1.0)

    def test_monotonic_rise_has_raw_negative_return_score(self) -> None:
        details = score_cycle_voc_values([0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertAlmostEqual(details["score_up"], 1.0)
        self.assertAlmostEqual(details["score_down"], -1.0)
        self.assertAlmostEqual(details["cycle_voc"], 0.0)

    def test_double_negative_cycle_keeps_negative_mean(self) -> None:
        details = score_cycle_voc_values([1.0, 0.5, 0.0, 0.5, 1.0])
        self.assertAlmostEqual(details["score_up"], -1.0)
        self.assertAlmostEqual(details["score_down"], -1.0)
        self.assertAlmostEqual(details["cycle_voc"], -1.0)

    def test_all_zero_sequence_is_valid_with_zero_score(self) -> None:
        details = score_cycle_voc_values([0.0] * 5)
        self.assertIsNone(details["rho_up"])
        self.assertIsNone(details["rho_down"])
        self.assertEqual(details["score_up"], 0.0)
        self.assertEqual(details["score_down"], 0.0)
        self.assertEqual(details["cycle_voc"], 0.0)

    def test_all_zero_half_is_valid_and_forces_zero_cycle_score(self) -> None:
        details = score_cycle_voc_values([0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertIsNone(details["rho_up"])
        self.assertEqual(details["score_up"], 0.0)
        # The descending half includes the turning point: [0, 1, 0].
        self.assertAlmostEqual(details["rho_down"], 0.0)
        self.assertEqual(details["cycle_voc"], 0.0)

    def test_constant_nonzero_is_zero_and_short_sequences_are_invalid(self) -> None:
        details = score_cycle_voc_values([1.0] * 5)
        self.assertIsNone(details["rho_up"])
        self.assertIsNone(details["rho_down"])
        self.assertEqual(details["cycle_voc"], 0.0)
        for values in ([0.0, 1.0, 0.0], [0.0] * 4):
            with self.subTest(values=values), self.assertRaisesRegex(
                VMBMKError, r"2K\+1"
            ):
                score_cycle_voc_values(values)

    def test_ties_are_stable(self) -> None:
        first = score_cycle_voc_values([0.0, 0.5, 0.5, 0.5, 0.0])
        second = score_cycle_voc_values([0.0, 0.5, 0.5, 0.5, 0.0])
        self.assertEqual(first, second)

    def test_macro_scores_cycle_values_from_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_cycle_dataset(Path(directory)))
            queries = build_cycle_voc_queries(dataset)
            results = [
                Result(query.query_id, "value", value)
                for query, value in zip(
                    queries, [0.0, 0.5, 1.0, 0.5, 0.0]
                )
            ]
            self.assertAlmostEqual(score_cycle_voc(dataset, results), 1.0)

    def test_macro_includes_all_zero_episode_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_cycle_dataset(Path(directory)))
            queries = build_cycle_voc_queries(dataset)
            results = [Result(query.query_id, "value", 0.0) for query in queries]
            self.assertEqual(score_cycle_voc(dataset, results), 0.0)
