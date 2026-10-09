from __future__ import annotations

import random
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthetic_data import make_dataset, set_task_metadata, write_json
from vmbmk.data.dataset import Dataset
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.metrics.cycle_voc_vs.vs import VS_PROTOCOL, build_vs_queries, run_vs, score_vs
from vmbmk.metrics.cycle_voc_vs.vs_scoring import score_curve
from vmbmk.metrics.cycle_voc_vs.cycle import build_cycle_voc_queries
from vmbmk.metrics.cycle_voc_vs.planning import plan_cycle
from vmbmk.metrics.registry import selected_task_ids
from vmbmk.inference.queries import Result, read_queries, read_results
from vmbmk.runner.run import _read_metrics, run_evaluation
from vmbmk.tools.results.validation import _validate_metric_coverage, inspect_result
from vmbmk.serialization import write_jsonl


def make_vs_dataset(root: Path) -> Path:
    data = make_dataset(root, [('expert', True)])
    set_task_metadata(data, ['VOC'])
    write_json(
        data / 'task_001/episodes/expert/annotation.json',
        {'voc': [{'frame_index': frame} for frame in range(100)]},
    )
    return data


def cycle_predictions(dataset: Dataset) -> list[Result]:
    return [
        Result(query.query_id, 'value', query.state.anchor_frame if query.playback == 'forward' else -1000)
        for query in build_cycle_voc_queries(dataset)
    ]


class VSScoringTest(unittest.TestCase):
    def test_native_controls(self) -> None:
        controls = [
            ([0, 1, 2, 3, 4, 5, 6], 1),
            ([0, 0, 0, 1, 1, 1, 1], 1 / 6),
            ([0, 0, 0, 1, 0, 0, 0], 0),
            ([1] * 7, 0),
            ([0, 1, 2, 1, 1, 1, 1], 1 / 2),
        ]
        for values, expected in controls:
            before = list(values)
            self.assertAlmostEqual(score_curve(values).vs, expected)
            self.assertEqual(values, before)

    def test_time_weighting(self) -> None:
        self.assertAlmostEqual(score_curve([0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 5, 6, 7, 8]).scale, 3/8)
        with self.assertRaises(ValueError):
            score_curve([0]*6)

    def test_scoped_loader_excludes_branches_and_diverse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_dataset(
                Path(directory),
                [('expert', True), ('branch', True), ('diverse', True)]
            )
            set_task_metadata(root, ['VOC'])
            write_json(
                root / 'task_001/episodes/expert/annotation.json',
                {'voc': [{'frame_index': frame} for frame in range(0, 100, 2)]},
            )
            for episode_id, extra in [('branch', {'trr': {'group_id': 'group', 'role': 'r_plus'}}),
                                      ('diverse', {'cspc': {'solution_type': 'diverse'}})]:
                path = root / 'task_001' / 'episodes' / episode_id / 'metadata.json'
                metadata = json.loads(path.read_text())
                metadata.update(extra)
                write_json(path, metadata)
            dataset = Dataset.load(root, metrics=['vs'])
            queries = build_vs_queries(dataset)
            self.assertEqual({query.state.episode_id for query in queries}, {'expert'})
            result = score_vs(
                dataset,
                [Result(query.query_id, 'value', query.state.anchor_frame) for query in queries]
            )
            self.assertEqual(result['mean'], 1)
            self.assertEqual(len(result['exclusions']), 2)

    def test_reuses_only_cycle_forward_native_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_vs_dataset(root / 'data')
            dataset = Dataset.load(data, metrics=['vs'])
            source = root / 'cycle.jsonl'
            output = root / 'vs.jsonl'
            write_jsonl(source, (result.to_dict() for result in cycle_predictions(dataset)))
            self.assertEqual(selected_task_ids('vs', dataset, ['task_001'], ['id']), ('task_001',))
            with patch('vmbmk.metrics.cycle_voc_vs.vs.run_cycle_voc') as inference:
                result = run_vs(data, {}, ['task_001'], output, 0, mode='base', domains=['id'], cycle_operation_path=source)
                inference.assert_not_called()
            self.assertEqual(result['protocol'], VS_PROTOCOL)
            self.assertEqual(result['prediction_source'], 'cycle_voc_forward')
            self.assertEqual(result['mean'], 1)
            self.assertEqual(result['episodes'][0]['frames'], list(range(0, 100, 2)))
            self.assertEqual([row.score for row in read_results(output)], list(range(0, 100, 2)))

    def test_missing_grid_and_cycle_prediction_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_vs_dataset(root / 'data')
            dataset = Dataset.load(data, metrics=['vs'])
            source = root / 'cycle.jsonl'
            write_jsonl(source, (result.to_dict() for result in cycle_predictions(dataset)[1:]))
            with self.assertRaisesRegex(VMBMKError, 'coverage mismatch'):
                run_vs(data, {}, ['task_001'], root / 'vs.jsonl', 0, mode='base', domains=['id'], cycle_operation_path=source)
            write_json(data / 'task_001/episodes/expert/annotation.json', {'voc': [{'frame_index': frame} for frame in range(1, 100)]})
            with self.assertRaisesRegex(VMBMKError, 'do not cover the 5 Hz grid'):
                build_vs_queries(Dataset.load(data, metrics=['vs']))

    def test_vs_only_computes_cycle_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_vs_dataset(root / 'data')
            dataset = Dataset.load(data, metrics=['vs'])

            def infer_cycle(data_root, model, task_ids, operation_path, gpu, *, mode, domains, plan):
                self.assertEqual(mode, 'base')
                write_jsonl(operation_path, (result.to_dict() for result in cycle_predictions(dataset)))

            with patch('vmbmk.metrics.cycle_voc_vs.vs.run_cycle_voc', side_effect=infer_cycle) as inference:
                result = run_vs(data, {}, ['task_001'], root / 'vs.jsonl', 0, mode='base', domains=['id'])
                inference.assert_called_once()
            self.assertEqual(result['mean'], 1)

    def test_shared_cli_runs_cycle_first_and_infers_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_vs_dataset(root / 'data')
            source = root / 'configs.json'
            write_json(source, {
                'model': 'robometer', 'gpu': 0, 'batch_size': 2,
                'python': 'python', 'checkpoint': str(root / 'checkpoint'),
                'data': str(data), 'output': str(root / 'runs'), 'tasks': ['task_001'],
                'metrics': {metric: {'mode': 'base', 'domains': ['id']} for metric in ('voc', 'vs', 'cycle_voc')},
            })

            def infer(data_root, query_path, model, operation_path, *, gpu, native_metric):
                queries = read_queries(query_path)
                self.assertTrue(all(query.query_id.startswith('cycle_voc:') for query in queries))
                write_jsonl(operation_path, (
                    Result(query.query_id, 'value', query.state.anchor_frame if query.playback == 'forward' else -1000).to_dict()
                    for query in queries
                ))

            with patch('vmbmk.metrics.cycle_voc_vs.cycle.run_inference', side_effect=infer) as inference, patch(
                'vmbmk.runner.run.plan_cycle', wraps=plan_cycle
            ) as cohort, patch('vmbmk.metrics.cycle_voc_vs.cycle.plan_cycle') as cycle_selection, patch(
                'vmbmk.metrics.cycle_voc_vs.vs.plan_cycle'
            ) as vs_selection:
                completed = run_evaluation(source)
                inference.assert_called_once()
                cohort.assert_called_once()
                cycle_selection.assert_not_called()
                vs_selection.assert_not_called()
            result = json.loads((completed / 'metrics.json').read_text())
            self.assertEqual(result['vs']['mean'], 1)
            self.assertEqual(result['voc']['mean'], 1)
            self.assertEqual(len(read_results(completed / 'operations.jsonl')), 199 + 100 + 50)
            self.assertFalse(any(path.name.startswith('.') for path in completed.iterdir()))

    def test_native_source_and_legacy_vs_cache_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(VMBMKError, 'mode=base'):
                run_vs(root, {}, ['task_001'], root / 'vs.jsonl', 0, mode='native', domains=['id'])
            cached = root / 'metrics.json'
            write_json(cached, {'vs': {'protocol': 'vs_v1-expert-forward-5hz-v1', 'mean': 1}})
            with self.assertRaisesRegex(ConfigurationError, 'obsolete protocol'):
                _read_metrics(cached, {'vs'}, 'completed run')
            with self.assertRaisesRegex(ValueError, 'obsolete input protocol'):
                _validate_metric_coverage(
                    {'vs': {'protocol': 'vs_v1-expert-forward-5hz-v1', 'mean': 1}},
                    {'vs': {'id': {'task_001'}}},
                )

    def test_id_env_emb_separate_runs_match_combined_domain_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts = {'id': 1, 'env': 3, 'emb': 2}
            episodes = [(f'{domain}_st_{index}', True) for domain, count in counts.items() for index in range(count)]
            data = make_dataset(root / 'data', episodes)
            set_task_metadata(data, ['VOC'])
            for episode_id, _ in episodes:
                episode_root = data / 'task_001/episodes' / episode_id
                metadata = json.loads((episode_root / 'metadata.json').read_text())
                metadata.update(domain=episode_id.split('_')[0], num_frames=14)
                write_json(episode_root / 'metadata.json', metadata)
                write_json(episode_root / 'annotation.json', {'voc': [{'frame_index': frame} for frame in range(0, 14, 2)]})
            dataset = Dataset.load(data, metrics=['cycle_voc', 'vs'])

            def infer(data_root, query_path, model, operation_path, *, gpu, native_metric):
                queries = read_queries(query_path)
                self.assertTrue(all(query.query_id.startswith('cycle_voc:') for query in queries))
                values = {'id': 1, 'env': 0, 'emb': -1}
                write_jsonl(operation_path, (
                    Result(query.query_id, 'value', values[dataset.episode(query.state.task_id, query.state.episode_id).domain] * query.state.anchor_frame).to_dict()
                    for query in queries
                ))

            runs = {}
            with patch('vmbmk.metrics.cycle_voc_vs.cycle.run_inference', side_effect=infer) as inference:
                for label, domains in [('all', list(counts)), *[(domain, [domain]) for domain in counts]]:
                    source = root / f'{label}configs.json'
                    write_json(source, {
                        'model': 'robometer', 'gpu': 0, 'batch_size': 2,
                        'python': 'python', 'checkpoint': str(root / 'checkpoint'),
                        'data': str(data), 'output': str(root / 'runs'), 'tasks': ['task_001'],
                        'metrics': {metric: {'mode': 'base', 'domains': domains} for metric in ('vs', 'cycle_voc')},
                    })
                    completed = run_evaluation(source)
                    self.assertTrue(inspect_result(source, completed, dataset)['valid'])
                    runs[label] = json.loads((completed / 'metrics.json').read_text())
                    results = read_results(completed / 'operations.jsonl')
                    self.assertEqual(len(results), sum(counts[domain] for domain in domains) * (13 + 7))
                    actual_domains = {dataset.episode('task_001', row.query_id.split(':')[2]).domain for row in results}
                    self.assertEqual(actual_domains, set(domains))
                self.assertEqual(inference.call_count, 4)
            for metric in ('cycle_voc', 'vs'):
                self.assertEqual(set(runs['all'][metric]['domains']), set(counts))
                for domain in counts:
                    separate = runs[domain][metric]
                    self.assertEqual(set(separate['domains']), {domain})
                    self.assertEqual(separate['domains'][domain], runs['all'][metric]['domains'][domain])
                    self.assertEqual(separate['mean'], separate['domains'][domain]['mean'])
            self.assertEqual([runs[domain]['cycle_voc']['mean'] for domain in counts], [1, 0, -1])
            self.assertEqual([runs[domain]['vs']['mean'] for domain in counts], [1, 0, 1])
            self.assertEqual(runs['all']['cycle_voc']['mean'], 0)
            self.assertAlmostEqual(runs['all']['vs']['mean'], 2 / 3)

    def test_ood_without_normal_st_is_missing_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [('id_st', True), ('env_failure', False)])
            set_task_metadata(data, ['VOC'])
            write_json(data / 'task_001/episodes/id_st/annotation.json', {'voc': [{'frame_index': frame} for frame in range(0, 100, 2)]})
            metadata_path = data / 'task_001/episodes/env_failure/metadata.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['domain'] = 'env'
            write_json(metadata_path, metadata)
            dataset = Dataset.load(data, metrics=['cycle_voc', 'vs'])
            plan = plan_cycle(dataset, ['task_001'], ['id'])
            self.assertEqual({domain for domain, _, _ in plan.trajectories}, {'id'})
            for domains in (['id', 'env'], ['id', 'emb']):
                with self.subTest(domains=domains), self.assertRaisesRegex(VMBMKError, 'domains have no annotated episodes'):
                    plan_cycle(dataset, ['task_001'], domains)


class VSReferenceTest(unittest.TestCase):
    def test_seeded_reference_and_nonmutation(self) -> None:
        rng = random.Random(20261001)
        for _ in range(300):
            values = [
                rng.choice([0.0, 1.0, rng.uniform(-10, 10)])
                for _ in range(rng.randint(7, 80))
            ]
            before = list(values)
            times = [0.0]
            for _ in values[1:]:
                times.append(times[-1] + rng.uniform(0.1, 2))
            smoothed = list(values)
            smoothed[:3] = [sum(values[:3]) / 3] * 3
            smoothed[-3:] = [sum(values[-3:]) / 3] * 3
            variation = sum(abs(right - left) for left, right in zip(smoothed, smoothed[1:]))
            er = abs(smoothed[-1] - smoothed[0]) / variation if variation else 1
            coverage = sum(
                right - left
                for left, right, left_value, right_value in zip(
                    times, times[1:], values, values[1:]
                )
                if left_value != right_value
            ) / (times[-1] - times[0])
            self.assertAlmostEqual(score_curve(values, times).vs, er * coverage)
            self.assertEqual(values, before)
