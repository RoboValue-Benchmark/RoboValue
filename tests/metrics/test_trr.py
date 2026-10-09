from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synthetic_data import make_dataset, write_json
from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.metrics.trr import _frames, _weights, _direction, build_trr_timeweighted_queries as build_trr_queries, score_trr_timeweighted as score_trr
from vmbmk.inference.queries import Result


def make_trr_dataset(
    root: Path,
    *,
    failure_span: tuple[int, int] = (30, 50),
) -> Path:
    data = make_dataset(
        root,
        [("plus", True), ("zero", False), ("minus", False)],
    )
    task = data / "task_001"
    metadata_path = task / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["supports"]["metrics"] = ["TRR"]
    write_json(metadata_path, metadata)
    roles = {
        "plus": ("r_plus", "recovery_success_span"),
        "zero": ("r_zero", "recovery_failed_span"),
        "minus": ("r_minus", "continue_span"),
    }
    for episode_id, (role, result_span) in roles.items():
        episode = task / "episodes" / episode_id
        episode_metadata_path = episode / "metadata.json"
        episode_metadata = json.loads(
            episode_metadata_path.read_text(encoding="utf-8")
        )
        episode_metadata["num_frames"] = 100
        episode_metadata["trr"] = {"group_id": "FRT-1-id-G01", "role": role}
        write_json(episode_metadata_path, episode_metadata)
        failure_start, failure_end = failure_span
        recovery_end = failure_end + 20
        result_end = recovery_end + 20
        trr = {
            "failure_span": {
                "start_frame": failure_start,
                "end_frame_exclusive": failure_end,
            },
        }
        if role == "r_minus":
            trr[result_span] = {
                "start_frame": failure_end,
                "end_frame_exclusive": result_end,
            }
        else:
            trr["recovery_span"] = {
                "start_frame": failure_end,
                "end_frame_exclusive": recovery_end,
            }
            trr[result_span] = {
                "start_frame": recovery_end,
                "end_frame_exclusive": result_end,
            }
        write_json(episode / "annotation.json", {"trr": trr})
    return data


class TrrTimeweightedTest(unittest.TestCase):
    def test_sampling_covers_short_stages_and_time_mass(self) -> None:
        for start, end in [(1, 4), (30, 50), (31, 100)]:
            frames = _frames(start, end, 30)
            self.assertTrue(all(start <= f < end for f in frames))
            self.assertGreaterEqual(len(frames), min(10, end-start))
            self.assertTrue(set(range(((start+2)//3)*3, end, 3)) <= set(frames))
            self.assertAlmostEqual(sum(_weights(frames, start, end)), end-start)

    def test_weighted_direction_matches_hand_calculation(self) -> None:
        self.assertAlmostEqual(_direction([0, 2, 1], [1, 2, 3]), 1/13)
        self.assertEqual(_direction([2, 2, 2], [1, 2, 3]), 0)

    def test_constant_rising_and_falling_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_trr_dataset(Path(directory)), metrics=['trr'])
            queries = build_trr_queries(dataset)
            self.assertEqual(len({q.query_id for q in queries}), len(queries))
            for slope, expected in [(0, 0), (1, 0), (-1, 1/3)]:
                rows = [Result(q.query_id, 'value', slope*q.state.anchor_frame) for q in queries]
                self.assertAlmostEqual(score_trr(dataset, rows)['mean'], expected)
            with self.assertRaises(VMBMKError):
                score_trr(dataset, rows[:-1])


    def test_noninteger_grid_and_strict_conjunction(self) -> None:
        self.assertTrue({0, 2, 5, 8, 10, 12, 15, 18, 20, 22} <= set(_frames(0, 25, 25)))
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_trr_dataset(Path(directory)), metrics=['trr'])
            rows = []
            for query in build_trr_queries(dataset):
                role, stage = query.query_id.split(':')[2:4]
                slope = -1 if stage in {'failure_span', 'recovery_failed_span'} else 1
                if role == 'r_minus' and stage == 'continue_span':
                    slope = 0
                rows.append(Result(query.query_id, 'value', slope * query.state.anchor_frame))
            self.assertEqual(score_trr(dataset, rows)['mean'], 1)
            bad_rows = [Result(row.query_id, 'value', 0 if ':r_zero:recovery_failed_span:' in row.query_id else row.score) for row in rows]
            self.assertAlmostEqual(score_trr(dataset, bad_rows)['mean'], 2/3)
