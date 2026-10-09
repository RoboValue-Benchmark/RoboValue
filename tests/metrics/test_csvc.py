from __future__ import annotations

import json
import tempfile
from pathlib import Path
import math
import unittest
from types import SimpleNamespace

from vmbmk.metrics.csvc import _window_pairs
from synthetic_data import make_dataset, set_task_metadata, write_json
from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.metrics.csvc import (
    aggregate_csvc,
    CSVC_PROTOCOL,
    score_csvc_symmetric,
)
from vmbmk.metrics.csvc import build_csvc_queries, score_csvc
from vmbmk.inference.queries import Result


def score(normal, diverse, **options):
    options.setdefault('bootstrap_samples', 0)
    return score_csvc_symmetric(normal, diverse, **options)


class CSVCScoringTest(unittest.TestCase):
    def test_exact_consistency(self) -> None:
        result = score({'n': {'a': 1, 'b': 2}}, {'d': {'a': 1, 'b': 2}})
        self.assertEqual(result['score'], 1)
        self.assertTrue(result['identifiable'])

    def test_hand_computed_symmetric_score(self) -> None:
        result = score({'n': {'a': 1}}, {'d': {'a': 3}})
        self.assertAlmostEqual(result['score'], 1 / 3)
        self.assertEqual(result['semantic_centers'], {'a': 2})
        self.assertEqual(result['cross_rmse'], 1)

    def test_normal_repeats_have_one_vote(self) -> None:
        one = score({'n': {'a': 1}}, {'d': {'a': 3}})
        many = score({str(i): {'a': 1} for i in range(100)}, {'d': {'a': 3}})
        self.assertEqual(one['score'], many['score'])
        self.assertEqual(many['semantic_coverage'], {'a': 2})

    def test_normal_uses_median_not_mean(self) -> None:
        result = score({'a': {'s': 1}, 'b': {'s': 1}, 'c': {'s': 100}}, {'d': {'s': 1}})
        self.assertEqual(result['score'], 1)

    def test_exchange_unique_solutions(self) -> None:
        a = score({'n': {'s': 1}}, {'d1': {'s': 2}, 'd2': {'s': 4}})
        b = score({'n': {'s': 4}}, {'d1': {'s': 1}, 'd2': {'s': 2}})
        self.assertEqual(a['score'], b['score'])

    def test_semantic_equal_weights_with_partial_coverage(self) -> None:
        result = score({'n': {'a': 1, 'b': 2}},
                       {'d1': {'a': 3, 'b': 4}, 'd2': {'a': 5}})
        self.assertAlmostEqual(result['cross_rmse'], math.sqrt((8/3 + 1)/2))
        self.assertEqual(result['gain_scale'], 3)

        self.assertEqual(result['semantic_groups'], ['a', 'b'])
    def test_zero_denominator_receives_minus_one(self) -> None:
        result = score({'n': {'a': 0}}, {'d': {'a': 0}})
        self.assertFalse(result['identifiable'])
        self.assertEqual(result['score'], -1)

    def test_macro_includes_zero_denominator_penalty(self) -> None:
        result = aggregate_csvc(
            {
                'identifiable': {'score': 1.0, 'raw_score': 1.0, 'identifiable': True},
                'zero_denominator': {'score': -1.0, 'raw_score': -1.0, 'identifiable': False},
            },
            bootstrap_samples=0,
        )
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['task_scores']['zero_denominator'], -1.0)
        self.assertEqual(result['identifiable_tasks'], 1)
        self.assertEqual(result['unidentifiable_task_rate'], 0.5)

    def test_single_solution_semantics_excluded(self) -> None:
        result = score({'n': {'a': 1}}, {'d': {'b': 1}})
        self.assertFalse(result['identifiable'])

    def test_negative_consistency_is_not_direction_accuracy(self) -> None:
        self.assertEqual(score({'n': {'a': -1}}, {'d': {'a': -1}})['score'], 1)


    def test_negative_consistency_is_not_clipped(self) -> None:
        result = score({'n': {'a': -5}}, {'d1': {'a': 1}, 'd2': {'a': 1}})
        distance = math.sqrt(12)
        self.assertAlmostEqual(result['score'], (1 - distance) / (1 + distance))
        self.assertEqual(result['score'], result['raw_score'])
        self.assertLess(result['raw_score'], 0)

    def test_scale_invariance(self) -> None:
        a = score({'n': {'s': 1}}, {'d': {'s': 3}})
        b = score({'n': {'s': 100}}, {'d': {'s': 300}})
        self.assertAlmostEqual(a['score'], b['score'])


    def test_rejects_invalid_inputs(self) -> None:
        for bad in [float('nan'), float('inf'), True]:
            with self.assertRaises(VMBMKError):
                score({'n': {'a': bad}}, {'d': {'a': 1}})
        with self.assertRaises(VMBMKError):
            score({'n': {'a': 1}}, {'d': {'a': 1}}, epsilon=float('inf'))

    def test_metric_builds_original_four_boundary_pairs_and_scores(self) -> None:
        def episode(name, role):
            return SimpleNamespace(task_id='task', episode_id=name, success=True,
                domain='id', cspc_solution_type=role, fps=30,
                subtask_spans={'place': (10, 110)})
        task = SimpleNamespace(metrics=['CSPC'], instruction='Place the object',
            episodes={'n': episode('n', 'normal'), 'd': episode('d', 'diverse')})
        dataset = SimpleNamespace(tasks={'task': task})
        queries = build_csvc_queries(dataset, domains=['id'])
        pairs = [(q.state_a.anchor_frame, q.state_b.anchor_frame) for q in queries[:4]]
        self.assertEqual(pairs, [(10, 109), (14, 105), (19, 100), (24, 95)])
        # Median of four compare calls, not their arithmetic average.
        values = [1., 1., 1., 100.]
        results = [Result(q.query_id, 'compare', values[i % 4]) for i, q in enumerate(queries)]
        result = score_csvc(dataset, results, domains=['id'])
        self.assertEqual(result['protocol'], CSVC_PROTOCOL)
        self.assertEqual(result['mean'], 1.)
        self.assertEqual(result['domains'], {'id': {'mean': 1., 'tasks': {'task': 1.}}})
        self.assertEqual(
            result['task_results']['task']['null_status'],
            'configured_cross_solution_semantic_mismatch_all_pairs',
        )

    def test_zero_center_with_nonzero_deviation_is_measured(self) -> None:
        result = score({'n': {'a': -1}}, {'d': {'a': 1}})
        self.assertEqual(result['score'], -1)
        self.assertTrue(result['identifiable'])

    def test_small_gains_do_not_trigger_an_epsilon_cutoff(self) -> None:
        result = score({'n': {'a': 1e-15}}, {'d': {'a': 3e-15}}, epsilon=1)
        self.assertAlmostEqual(result['score'], 1 / 3)
        self.assertTrue(result['identifiable'])

    def test_missing_semantic_coverage_remains_na(self) -> None:
        missing = score({'n': {'a': 1}}, {'d': {'b': 1}})
        self.assertIsNone(missing['score'])
        aggregate = aggregate_csvc({
            'covered': {'score': -0.5, 'raw_score': -0.5, 'identifiable': True},
            'missing': missing,
        }, bootstrap_samples=0)
        self.assertEqual(aggregate['score'], -0.5)
        self.assertIsNone(aggregate['task_scores']['missing'])


class CSVCBoundaryTest(unittest.TestCase):
    def test_short_span_keeps_four_ordered_pairs_inside_annotation(self) -> None:
        episode = SimpleNamespace(fps=25, task_id='fill_pen_holder', episode_id='ep')
        for start, end in [(547, 569), (448, 471), (500, 521)]:
            pairs = _window_pairs(episode, '006', (start, end),
                                  window_samples=4, window_step_seconds=.15)
            self.assertEqual(len(pairs), 4)
            self.assertTrue(all(start <= before < after < end for before, after in pairs))
        self.assertEqual(_window_pairs(episode, '006', (100, 200),
                         window_samples=4, window_step_seconds=.15),
                         [(100, 199), (104, 195), (108, 191), (111, 188)])





class CSVCSemanticTest(unittest.TestCase):
    def test_cspc_aligns_stable_semantic_ids_after_subtask_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(
                Path(directory),
                [("normal_a", True), ("normal_b", True), ("diverse", True)],
            )
            task = set_task_metadata(
                data,
                ["CSPC"],
                [
                    {"id": "A", "description": "place object A"},
                    {"id": "B", "description": "place object B"},
                ],
            )
            spans = {
                "normal_a": {"A": (10, 20), "B": (50, 60)},
                "normal_b": {"A": (12, 22), "B": (52, 62)},
                "diverse": {"A": (70, 80), "B": (30, 40)},
            }
            for episode_id, solution_type in (
                ("normal_a", "normal"),
                ("normal_b", "normal"),
                ("diverse", "diverse"),
            ):
                metadata_path = task / "episodes" / episode_id / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["cspc"] = {"solution_type": solution_type}
                write_json(metadata_path, metadata)
                write_json(
                    task / "episodes" / episode_id / "annotation.json",
                    {
                        "subtask_segments": [
                            {
                                "id": subtask,
                                "start_frame": span[0],
                                "end_frame_exclusive": span[1],
                            }
                            for subtask, span in spans[episode_id].items()
                        ]
                    },
                )
            dataset = Dataset.load(data)
            queries = build_csvc_queries(dataset)
            self.assertEqual(len(queries), 24)
            results = [
                Result(
                    query.query_id,
                    "compare",
                    0.25 if ":A:" in query.query_id else 0.75,
                )
                for query in queries
            ]
            scored = score_csvc(dataset, results)
            self.assertEqual(scored["mean"], 1.0)
            self.assertEqual(
                scored["task_results"]["task_001"]["semantic_groups"],
                ["A", "B"],
            )
            self.assertEqual(
                scored["alignment"],
                "stable semantic subtask ID, never chronological rank",
            )
