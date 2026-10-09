from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from vmbmk.errors import VMBMKError
from vmbmk.metrics.sia.config import load_judge_config
from vmbmk.metrics.sia.judge import DeepSeekCandidateJudge, _target_logprob
from vmbmk.metrics.sia.scoring import score_sia_mappings
from vmbmk.inference.queries import StateRef, SubtaskQuery


class SIATest(unittest.TestCase):
    def test_private_judge_file_is_read_without_exporting_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text(yaml.safe_dump({
                'api_key': 'test-only-key', 'base_url': 'https://judge.invalid/v1',
                'model': 'test-only-model',
            }), encoding='utf-8')
            config.chmod(0o600)
            for environment in ({}, {'VMBMK_SIA_CONFIG': str(config)}):
                with self.subTest(environment=environment), patch(
                    'vmbmk.metrics.sia.config.__file__', str(config.with_suffix('.py'))
                ), patch.dict(os.environ, environment, clear=True):
                    judge = DeepSeekCandidateJudge()
                    self.assertEqual(judge.auth_token, 'test-only-key')
                    self.assertEqual(judge._completion_url(), 'https://judge.invalid/v1/chat/completions')
                    self.assertEqual(judge.model, 'test-only-model')
                    self.assertNotIn('DEEPSEEK_API_KEY', os.environ)

    def test_judge_settings_precedence(self) -> None:
        settings = {'api_key': 'file-key', 'base_url': 'https://file.invalid', 'model': 'file-model'}
        environment = {'DEEPSEEK_API_KEY': 'env-key', 'DEEPSEEK_BASE_URL': 'https://env.invalid', 'DEEPSEEK_MODEL': 'env-model'}
        with patch('vmbmk.metrics.sia.judge.load_judge_config', return_value=settings), patch.dict(os.environ, environment, clear=True):
            judge = DeepSeekCandidateJudge()
            self.assertEqual((judge.auth_token, judge.base_url, judge.model), ('env-key', 'https://env.invalid', 'env-model'))
            explicit = DeepSeekCandidateJudge(auth_token='explicit-key', base_url='https://explicit.invalid', model='explicit-model')
            self.assertEqual((explicit.auth_token, explicit.base_url, explicit.model), ('explicit-key', 'https://explicit.invalid', 'explicit-model'))

    def test_private_judge_config_missing_and_invalid_are_not_silent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('vmbmk.metrics.sia.config.__file__', str(root / 'config.py')), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_judge_config(), {})
            config = root / 'config.yaml'
            with patch.dict(os.environ, {'VMBMK_SIA_CONFIG': str(config)}, clear=True):
                with self.assertRaisesRegex(VMBMKError, 'Cannot read SIA private config'):
                    load_judge_config()
                for contents in ('api_key: [test-secret', 'api_key: test-secret\nunknown: test-secret', 'api_key: 123', 'api_key: ""'):
                    config.write_text(contents, encoding='utf-8')
                    config.chmod(0o600)
                    with self.subTest(contents=contents), self.assertRaises(VMBMKError) as failure:
                        load_judge_config()
                    self.assertNotIn('test-secret', str(failure.exception))
                config.write_text('api_key: null\nbase_url: null\nmodel: null', encoding='utf-8')
                self.assertEqual(load_judge_config(), {})
                if os.name == 'posix':
                    config.chmod(0o644)
                    with self.assertRaisesRegex(VMBMKError, 'chmod 600'):
                        load_judge_config()

    def _score(self, rows):
        items = []
        mapped = {}
        for index, (domain, task, logprob) in enumerate(rows):
            query = SubtaskQuery(str(index), StateRef(task, "episode", index), "Move the cup")
            items.append({"query": query, "domain": domain,
                          "candidates": {"001": "Pick cup", "002": "Place cup"},
                          "target_subtask_id": "002"})
            mapped[query.query_id] = {
                "predicted_subtask_id": "001",
                "candidate_logprobs": {"001": -0.1, "002": logprob},
            }
        return score_sia_mappings(items, mapped)

    def test_uses_ground_truth_probability_not_predicted_label_accuracy(self) -> None:
        records, result = self._score([("id", "task", math.log(0.25))])
        self.assertAlmostEqual(result["mean"], 0.25)
        self.assertEqual(records[0]["target_subtask_id"], "002")
        self.assertEqual(records[0]["predicted_subtask_id"], "001")
        self.assertAlmostEqual(records[0]["target_logprob"], math.log(0.25))

    def test_geometric_mean_within_task(self) -> None:
        _, result = self._score([("id", "task", math.log(0.25)),
                                 ("id", "task", math.log(0.81))])
        self.assertAlmostEqual(result["mean"], 0.45)

    def test_averages_tasks_then_domains_not_all_items(self) -> None:
        _, result = self._score([("id", "task_a", math.log(0.25)),
                                 ("id", "task_a", math.log(0.81)),
                                 ("id", "task_b", math.log(0.8)),
                                 ("env", "task_c", math.log(0.2))])
        self.assertAlmostEqual(result["domains"]["id"]["tasks"]["task_a"], 0.45)
        self.assertAlmostEqual(result["domains"]["id"]["mean"], 0.625)
        self.assertAlmostEqual(result["domains"]["env"]["mean"], 0.2)
        self.assertAlmostEqual(result["mean"], 0.4125)

    def test_probability_one_and_small_probability(self) -> None:
        for logprob in (0.0, -700.0):
            with self.subTest(logprob=logprob):
                _, result = self._score([("id", "task", logprob)])
                self.assertEqual(result["mean"], math.exp(logprob))

    def test_missing_or_invalid_ground_truth_logprob_is_rejected(self) -> None:
        candidates = {"001": "Pick cup", "002": "Place cup"}
        with self.assertRaisesRegex(VMBMKError, "missing from top_logprobs"):
            _target_logprob({"predicted_subtask_id": "001",
                             "candidate_logprobs": {"001": -0.1}}, candidates, "002")
        for logprob in (True, 0.1, float("nan"), float("-inf"), -9999):
            with self.subTest(logprob=logprob), self.assertRaises(VMBMKError):
                self._score([("id", "task", logprob)])
