from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from synthetic_data import make_dataset, write_json
from vmbmk.data.dataset import Dataset
from vmbmk.inference.queries import Result
from vmbmk.runner.run import run_evaluation
from vmbmk.errors import ConfigurationError
from vmbmk.serialization import write_jsonl
from vmbmk.metrics.mem_voc import (
    build_voc_mem_queries,
    score_voc_mem,
    score_voc_mem_by_domain,
)


def make_voc_mem_dataset(root: Path) -> Path:
    data = make_dataset(root)
    task = data / "task_001"
    metadata_path = task / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["supports"]["metrics"] = ["VOC-MEM"]
    write_json(metadata_path, metadata)
    write_json(
        task / "episodes" / "ep_success" / "annotation.json",
        {
            "voc": [{"frame_index": 5}, {"frame_index": 10}],
            "voc_mem": [
                {"frame_index": 0},
                {"frame_index": 50},
                {"frame_index": 90},
            ],
        },
    )
    return data


class VOCMemTest(unittest.TestCase):
    def test_cli_dispatches_mem_voc_with_current_runner_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_voc_mem_dataset(root / 'data')
            config = root / 'memconfigs.json'
            write_json(config, {
                'model': 'robometer', 'gpu': 0, 'batch_size': 2,
                'python': 'python', 'checkpoint': str(root / 'checkpoint'),
                'data': str(data), 'output': str(root / 'runs'), 'tasks': ['task_001'],
                'metrics': {'voc_mem': {'mode': 'base', 'domains': ['id']}},
            })

            def infer(data_root, queries, model, operation_path, gpu, **settings):
                self.assertEqual(settings, {'metric': 'MEM-VOC', 'mode': 'base', 'native_metric': 'voc_mem'})
                results = [Result(query.query_id, 'value', query.state.anchor_frame) for query in queries]
                write_jsonl(operation_path, (result.to_dict() for result in results))
                return results

            with patch('vmbmk.metrics.mem_voc.run_queries', side_effect=infer) as inference, patch.object(Dataset, 'load', wraps=Dataset.load) as loading:
                completed = run_evaluation(config)
                inference.assert_called_once()
                loading.assert_called_once()
                self.assertEqual(run_evaluation(config), completed)
                inference.assert_called_once()
            self.assertEqual(json.loads((completed / 'metrics.json').read_text())['voc_mem']['mean'], 1)
            annotation = data / 'task_001/episodes/ep_success/annotation.json'
            row = json.loads(annotation.read_text())
            row['voc_mem'][1]['frame_index'] = 40
            write_json(annotation, row)
            with self.assertRaisesRegex(ConfigurationError, 'inputs or implementation changed'):
                run_evaluation(config)

    def test_builds_queries_only_from_voc_mem_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_voc_mem_dataset(Path(directory)))
            queries = build_voc_mem_queries(dataset)
            self.assertEqual(
                [query.state.anchor_frame for query in queries],
                [0, 50, 90],
            )
            self.assertTrue(
                all(query.query_id.startswith("voc_mem:") for query in queries)
            )

    def test_uses_same_episode_spearman_as_voc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_voc_mem_dataset(Path(directory)))
            queries = build_voc_mem_queries(dataset, domains=["id"])
            increasing = [
                Result(query.query_id, "value", value)
                for query, value in zip(queries, [0.1, 0.2, 0.3])
            ]
            decreasing = [
                Result(query.query_id, "value", value)
                for query, value in zip(queries, [0.3, 0.2, 0.1])
            ]
            self.assertEqual(score_voc_mem(dataset, increasing), 1.0)
            self.assertEqual(score_voc_mem(dataset, decreasing), -1.0)
            self.assertEqual(
                score_voc_mem_by_domain(dataset, increasing, domains=["id"]),
                {"id": {"task_001": 1.0}},
            )

    def test_constant_predictions_are_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset.load(make_voc_mem_dataset(Path(directory)))
            queries = build_voc_mem_queries(dataset, domains=["id"])
            constant = [Result(query.query_id, "value", -0.0) for query in queries]
            self.assertEqual(score_voc_mem(dataset, constant), 0.0)
