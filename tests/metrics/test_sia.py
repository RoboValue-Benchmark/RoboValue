from __future__ import annotations

import math
import json
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
from vmbmk.inference.queries import Result, StateRef, SubtaskQuery
from vmbmk.data.dataset import Dataset
from vmbmk.metrics.sia.evaluation import (
    build_sia_queries, judge_sia_responses, run_sia, save_sia_responses,
    select_sia_responses, _read_bundle, _generation_queries,
)
from synthetic_data import make_dataset, set_task_metadata, write_json


class SavedCandidateJudge:
    def __init__(self, **settings):
        self.model = "test-judge"

    def _completion_url(self):
        return "https://judge.invalid/completions"

    def classify(self, instruction, candidates, response):
        return {"predicted_subtask_id": "001", "candidate_logprobs": {
            "001": math.log(0.8), "002": math.log(0.2),
        }}


class SIAResponseBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        data = make_dataset(self.root / "data", [("episode", True)])
        task = set_task_metadata(data, ["SIA"], [
            {"id": "001", "description": "Pick cup"},
            {"id": "002", "description": "Place cup"},
        ])
        write_json(task / "episodes/episode/annotation.json", {"subtask_segments": [
            {"id": "001", "start_frame": 0, "end_frame_exclusive": 50},
            {"id": "002", "start_frame": 50, "end_frame_exclusive": 100},
        ]})
        self.dataset = Dataset.load(data, metrics=["sia"])
        queries = build_sia_queries(self.dataset, seed=0)
        self.bundle = self.root / "responses.json"
        save_sia_responses(self.dataset, queries, [
            Result(query.query_id, "subtask", output="Pick cup") for query in queries
        ], self.bundle)

    def test_offline_judging_matches_dataset_backed_scores_and_cache(self) -> None:
        original = self.bundle.read_bytes()
        with patch("vmbmk.metrics.sia.evaluation.DeepSeekCandidateJudge", SavedCandidateJudge):
            backed = judge_sia_responses(self.bundle, self.root / "backed.jsonl",
                model={}, cache_dir=self.root / "sia/judgements")
            with patch.object(Dataset, "load", side_effect=AssertionError("dataset access")), patch.object(
                SavedCandidateJudge, "classify", side_effect=AssertionError("cache not reused")
            ):
                offline = run_sia(self.root / "missing", {}, ["task_001"],
                    self.root / "offline.jsonl", None, mode="base", domains=["id"],
                    stage="judge", responses_path=self.bundle)
                cached = judge_sia_responses(self.bundle, self.root / "cached.jsonl",
                    model={}, cache_dir=self.root / "sia/judgements")
        self.assertEqual(backed, cached)
        self.assertEqual(backed, offline)
        self.assertAlmostEqual(offline["mean"], 0.4)
        self.assertEqual(self.bundle.read_bytes(), original)

    def test_filters_saved_tasks_and_domains(self) -> None:
        items = _read_bundle(self.bundle)
        self.assertEqual(select_sia_responses(items, task_ids=["task_001"], domains=["id"]), items)
        for settings in ({"task_ids": ["other"]}, {"domains": ["env"]}):
            with self.subTest(settings=settings), self.assertRaisesRegex(VMBMKError, "no trajectories"):
                select_sia_responses(items, **settings)

    def test_diverse_taxonomy_is_saved_before_offline_judging(self) -> None:
        task = self.dataset.tasks["task_001"]
        episode = self.dataset.episode("task_001", "episode")
        from dataclasses import replace

        diverse = replace(episode, cspc_solution_type="diverse",
                          subtask_spans={"place": (0, 50), "pick": (50, 100)})
        self.dataset = replace(self.dataset, tasks={"task_001": replace(task,
            subtasks={"pick": "Pick cup", "place": "Place cup"}, episodes={"episode": diverse})})
        queries = build_sia_queries(self.dataset, seed=0)
        save_sia_responses(self.dataset, queries, [
            Result(query.query_id, "subtask", output="Place cup") for query in queries
        ], self.bundle)
        offline = select_sia_responses(_read_bundle(self.bundle))
        backed = select_sia_responses(_read_bundle(self.bundle), self.dataset)
        self.assertEqual(offline, backed)
        self.assertEqual(offline[0]["candidates"], {"001": "Place cup", "002": "Pick cup"})
        self.assertEqual([item["target_subtask_id"] for item in offline], ["001", "002"])

    def test_selection_keeps_saved_instructions_and_candidates(self) -> None:
        from dataclasses import replace

        items = _read_bundle(self.bundle)
        task = self.dataset.tasks["task_001"]
        self.dataset = replace(self.dataset, tasks={"task_001": replace(task,
            instruction="A different instruction", subtasks={"001": "Different candidate"})})
        selected = select_sia_responses(items, self.dataset)
        self.assertEqual(selected, items)
        self.assertEqual(selected[0]["task_instruction"], "Put the cup on the tray")
        self.assertEqual(selected[0]["candidates"], {"001": "Pick cup", "002": "Place cup"})

    def test_cli_accepts_bundle_without_dataset(self) -> None:
        from vmbmk.cli import main

        with patch("vmbmk.metrics.sia.evaluation.DeepSeekCandidateJudge", SavedCandidateJudge), patch.object(
            Dataset, "load", side_effect=AssertionError("dataset access")
        ):
            self.assertEqual(main(["sia-judge", str(self.bundle), "--output", str(self.root / "cli")]), 0)
        result = json.loads((self.root / "cli/metrics.json").read_text())
        self.assertAlmostEqual(result["sia"]["mean"], 0.4)

    def test_incomplete_bundle_is_rejected(self) -> None:
        saved = json.loads(self.bundle.read_text())
        del saved["items"][0]["candidates"]
        write_json(self.bundle, saved)
        with self.assertRaisesRegex(VMBMKError, "invalid SIA response item"):
            _read_bundle(self.bundle)

    def test_old_generation_instruction_is_not_reused(self) -> None:
        from dataclasses import replace
        from vmbmk.serialization import replace_jsonl

        queries = build_sia_queries(self.dataset, seed=0)
        saved = self.root / "queries.jsonl"
        replace_jsonl(saved, [replace(query, instruction="Old task label").to_dict()
                             for query in queries])
        with self.assertRaisesRegex(VMBMKError, "current annotations/instructions"):
            _generation_queries(saved, self.dataset, ["task_001"], ["id"], seed=0)


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
