from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synthetic_data import make_dataset, write_json
from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import Result
from vmbmk.metrics.sa import _score_sa_details, build_sa_queries, score_sa, score_sa_by_domain


def make_sa_dataset(root: Path, episodes=None, num_frames: int = 100) -> Path:
    data = make_dataset(
        root,
        episodes=episodes
        or [("st_a", True), ("st_b", True), ("frt_a", False), ("frt_b", False)],
    )
    task = data / "task_001"
    metadata_path = task / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["supports"]["metrics"] = ["SA"]
    write_json(metadata_path, metadata)
    for episode in (task / "episodes").iterdir():
        episode_metadata = episode / "metadata.json"
        row = json.loads(episode_metadata.read_text(encoding="utf-8"))
        row["num_frames"] = num_frames
        write_json(episode_metadata, row)
    return data


class SATest(unittest.TestCase):
    def test_builds_value_queries_for_last_two_percent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(
                make_sa_dataset(
                    Path(directory),
                    episodes=[("st", True), ("frt", False)],
                    num_frames=101,
                )
            )
            queries = build_sa_queries(dataset)
            by_episode: dict[str, list[int]] = {}
            for query in queries:
                by_episode.setdefault(query.state.episode_id, []).append(
                    query.state.anchor_frame
                )
            self.assertEqual(
                by_episode,
                {"frt": [98, 99, 100], "st": [98, 99, 100]},
            )
            self.assertTrue(all(query.op == "value" for query in queries))

    def test_averages_tail_values_then_all_st_frt_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_sa_dataset(Path(directory)))
            queries = build_sa_queries(dataset, domains=["id"])
            frame_scores = {
                "st_a": [0.8, 1.0],
                "st_b": [0.3, 0.5],
                "frt_a": [0.4, 0.6],
                "frt_b": [0.2, 0.4],
            }
            offsets = {episode: 0 for episode in frame_scores}
            results = []
            for query in queries:
                episode = query.state.episode_id
                index = offsets[episode]
                offsets[episode] += 1
                results.append(
                    Result(query.query_id, "value", frame_scores[episode][index])
                )
            self.assertEqual(
                score_sa_by_domain(dataset, results, domains=["id"]),
                {"id": {"task_001": 0.75}},
            )
            self.assertEqual(score_sa(dataset, results), 0.75)

    def test_ties_do_not_count_as_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(
                make_sa_dataset(
                    Path(directory), episodes=[("st", True), ("frt", False)]
                )
            )
            results = [
                Result(query.query_id, "value", 0.5)
                for query in build_sa_queries(dataset)
            ]
            self.assertEqual(score_sa(dataset, results), 0.0)

    def test_rejects_compare_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(
                make_sa_dataset(
                    Path(directory), episodes=[("st", True), ("frt", False)]
                )
            )
            results = [
                Result(query.query_id, "compare", 0.5)
                for query in build_sa_queries(dataset)
            ]
            with self.assertRaisesRegex(VMBMKError, "must use op=value"):
                score_sa(dataset, results)


    def test_reports_raw_terminal_gap_and_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_sa_dataset(root)
            dataset = Dataset.load(data)
            queries = build_sa_queries(dataset, domains=["id"])
            frame_scores = {
                "st_a": [0.8, 1.0],
                "st_b": [0.3, 0.5],
                "frt_a": [0.4, 0.6],
                "frt_b": [0.2, 0.4],
            }
            offsets = {episode: 0 for episode in frame_scores}
            results = []
            for query in queries:
                episode = query.state.episode_id
                index = offsets[episode]
                offsets[episode] += 1
                results.append(Result(query.query_id, "value", frame_scores[episode][index]))
            _, scored = _score_sa_details(dataset, results, domains=["id"])
            self.assertEqual(scored["gap"], 0.25)
            self.assertEqual(scored["success_macro_mean"], 0.65)
            self.assertEqual(scored["failure_macro_mean"], 0.4)
            self.assertEqual(scored["range"], 0.65)
            self.assertAlmostEqual(
                scored["gap_over_range_percent"],
                100 * 0.25 / 0.65,
            )

    def test_reports_no_gap_percentage_when_terminal_range_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_sa_dataset(root, episodes=[("st", True), ("frt", False)])
            results = [
                Result(
                    query.query_id,
                    "value",
                    0.0 if query.state.episode_id == "st" else -0.5,
                )
                for query in build_sa_queries(Dataset.load(data), domains=["id"])
            ]
            _, scored = _score_sa_details(Dataset.load(data), results, domains=["id"])
            self.assertEqual(scored["gap"], 0.5)
            self.assertEqual(scored["range"], 0.0)
            self.assertIsNone(scored["gap_over_range_percent"])

    def test_requires_st_and_frt_in_each_task_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(
                make_sa_dataset(Path(directory), episodes=[("st", True)])
            )
            with self.assertRaisesRegex(VMBMKError, "both ST and FRT"):
                build_sa_queries(dataset)
