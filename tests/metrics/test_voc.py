from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from synthetic_data import make_dataset, write_json
from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import Result, read_queries, read_results
from vmbmk.metrics.cycle_voc_vs.cycle import build_cycle_voc_queries, score_cycle_voc_values
from vmbmk.metrics.cycle_voc_vs.forward import VOC_PROTOCOL
from vmbmk.metrics.utils.correlation import spearman, spearman_or_zero
from vmbmk.metrics.registry import selected_task_ids
from vmbmk.runner.run import run_evaluation, _read_metrics
from vmbmk.errors import ConfigurationError
from vmbmk.serialization import write_jsonl
from vmbmk.metrics.cycle_voc_vs.forward import (
    build_voc_queries,
    score_voc,
    score_voc_by_domain,
    score_voc_by_task,
)


def make_voc_dataset(root: Path) -> Path:
    data = make_dataset(root)
    task = data / "task_001"
    metadata_path = task / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["supports"]["metrics"] = ["VOC"]
    write_json(metadata_path, metadata)
    write_json(
        task / "episodes" / "ep_success" / "annotation.json",
        {"voc": [{"frame_index": 0}, {"frame_index": 50}, {"frame_index": 90}]},
    )
    return data


class VOCTest(unittest.TestCase):
    def test_correlation_rejects_invalid_inputs(self) -> None:
        for function in (spearman, spearman_or_zero):
            for left, right in (([], []), ([1], [1]), ([0, 1, 2], [0, 1]), ([0, float('nan')], [0, 1])):
                with self.subTest(function=function.__name__, left=left), self.assertRaises(VMBMKError):
                    function(left, right, 'VOC')

    def test_cycle_declaration_supports_voc_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_voc_dataset(Path(directory))
            path = data / 'task_001/metadata.json'
            row = json.loads(path.read_text())
            row['supports']['metrics'] = ['CYCLE-VOC']
            write_json(path, row)
            dataset = Dataset.load(data, metrics=['voc'])
            self.assertEqual(selected_task_ids('voc', dataset, ['task_001'], ['id']), ('task_001',))

    def test_forward_scores_equal_cycle_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_voc_dataset(Path(directory)))
            forward = build_voc_queries(dataset)
            cycle = build_cycle_voc_queries(dataset)
            self.assertEqual(
                [(query.state, query.instruction, query.playback) for query in forward],
                [(query.state, query.instruction, query.playback) for query in cycle[:3]],
            )
            for values in ([2, 2, 2], [1, 3, 2], [3, 2, 1], [1, 1, 2]):
                results = [Result(query.query_id, 'value', value) for query, value in zip(forward, values)]
                self.assertEqual(score_voc(dataset, results), score_cycle_voc_values(list(values) + [9, 8])['score_up'])

    def test_shared_run_reuses_forward_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_voc_dataset(root / 'data')
            source = root / 'configs.json'
            write_json(source, {
                'model': 'robometer', 'gpu': 0, 'batch_size': 2,
                'python': 'python', 'checkpoint': str(root / 'checkpoint'),
                'data': str(data), 'output': str(root / 'runs'), 'tasks': ['task_001'],
                'metrics': {metric: {'mode': 'base', 'domains': ['id']} for metric in ('voc', 'cycle_voc')},
            })

            def infer(data_root, query_path, model, operation_path, *, gpu, native_metric):
                queries = read_queries(query_path)
                write_jsonl(operation_path, (Result(query.query_id, 'value', query.state.anchor_frame).to_dict() for query in queries))

            with patch('vmbmk.metrics.cycle_voc_vs.cycle.run_inference', side_effect=infer) as inference, patch('vmbmk.metrics.cycle_voc_vs.forward.run_queries') as extra:
                completed = run_evaluation(source)
                inference.assert_called_once()
                extra.assert_not_called()
            metrics = json.loads((completed / 'metrics.json').read_text())
            self.assertEqual(metrics['voc']['mean'], 1)
            self.assertEqual(metrics['voc']['protocol'], VOC_PROTOCOL)
            results = read_results(completed / 'operations.jsonl')
            self.assertEqual(len(results), 8)
            self.assertEqual(len({result.query_id for result in results}), 8)

    def test_rejects_old_voc_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.json'
            write_json(path, {'voc': {'mean': 1}})
            with self.assertRaisesRegex(ConfigurationError, 'predates cycle-forward'):
                _read_metrics(path, {'voc'}, 'test')

    def test_requires_cycle_milestones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_voc_dataset(Path(directory))
            write_json(data / 'task_001/episodes/ep_success/annotation.json', {'voc': [{'frame_index': 0}, {'frame_index': 90}]})
            with self.assertRaisesRegex(VMBMKError, 'at least 3'):
                build_voc_queries(Dataset.load(data))

    def test_builds_only_value_queries_from_annotated_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_voc_dataset(Path(directory)))
            queries = build_voc_queries(dataset)
            self.assertEqual([query.op for query in queries], ["value"] * 3)
            self.assertEqual(
                [query.state.anchor_frame for query in queries],
                [0, 50, 90],
            )

    def test_filters_and_groups_by_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_voc_dataset(Path(directory))
            episodes = data / "task_001" / "episodes"
            shutil.copytree(episodes / "ep_success", episodes / "ep_env")
            metadata = episodes / "ep_env" / "metadata.json"
            row = json.loads(metadata.read_text(encoding="utf-8"))
            row["episode_id"] = "ep_env"
            row["domain"] = "env"
            write_json(metadata, row)
            dataset = Dataset.load(data)
            queries = build_voc_queries(dataset, domains=["env"])
            self.assertEqual(len(queries), 3)
            results = [
                Result(query.query_id, "value", index)
                for index, query in enumerate(queries)
            ]
            self.assertEqual(
                score_voc_by_domain(dataset, results, domains=["env"]),
                {"env": {"task_001": 1.0}},
            )

    def test_rejects_selected_domain_without_voc_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_voc_dataset(Path(directory)))
            with self.assertRaisesRegex(VMBMKError, "no task/domain"):
                build_voc_queries(dataset, domains=["env"])

    def test_scores_episode_spearman(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_voc_dataset(Path(directory)))
            queries = build_voc_queries(dataset)
            increasing = [
                Result(query.query_id, "value", score)
                for query, score in zip(queries, [0.1, 0.2, 0.3])
            ]
            decreasing = [
                Result(query.query_id, "value", score)
                for query, score in zip(queries, [0.3, 0.2, 0.1])
            ]
            self.assertEqual(score_voc(dataset, increasing), 1.0)
            self.assertEqual(score_voc(dataset, decreasing), -1.0)

    def test_constant_predictions_are_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_voc_dataset(Path(directory)))
            results = [
                Result(query.query_id, "value", 0.5)
                for query in build_voc_queries(dataset)
            ]
            self.assertEqual(score_voc(dataset, results), 0.0)

    def test_returns_per_task_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_voc_dataset(Path(directory)))
            queries = build_voc_queries(dataset, ["task_001"])
            results = [
                Result(query.query_id, "value", index)
                for index, query in enumerate(queries)
            ]
            self.assertEqual(
                score_voc_by_task(dataset, results, ["task_001"]),
                {"task_001": 1.0},
            )
