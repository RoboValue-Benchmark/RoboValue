from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from synthetic_data import make_dataset, set_task_metadata, write_json
from vmbmk.adapters import create_adapter
from vmbmk.adapters.mock_service import MockServiceAdapter
from vmbmk.adapters.registry import adapter_class
from vmbmk.data.dataset import Dataset
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.inference.dispatch import run_inference
from vmbmk.inference.queries import (
    CompareQuery, StateRef, SubtaskQuery, ValueQuery, read_results,
)
from vmbmk.inference.worker import run_worker
from vmbmk.metrics.sa import build_sa_queries, score_sa
from vmbmk.runner.run import RunConfig, run_evaluation
from vmbmk.serialization import write_jsonl
from vmbmk.inference.cache import bind_prediction_cache
from vmbmk.tools.results.validation import inspect_result
from vmbmk.tools.results.sync import synchronize
from vmbmk.tools.results.status import build_summary


WORKER_FIXTURE = """
import builtins
import os
import sys
from pathlib import Path
from PIL import Image

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('CPU service adapter must not import Torch')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from vmbmk.adapters import video_inputs
from vmbmk.inference.worker import main
def frame(path, index):
    red = index // 2 if Path(path).parent.name == 'ep_failure' else index
    return Image.new('RGB', (2, 2), (red, 1, 3))
video_inputs.read_frame = frame
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
main()
assert 'torch' not in sys.modules
"""


class FakeCandidateJudge:
    def __init__(self, **kwargs):
        self.model = "test-judge"

    def _completion_url(self):
        return "https://judge.invalid/completions"

    def classify(self, instruction, candidates, response):
        return {
            "predicted_subtask_id": "001",
            "candidate_logprobs": {"001": math.log(0.8), "002": math.log(0.2)},
        }


class MockServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = make_dataset(self.root / "data")
        set_task_metadata(self.data, ["SA"])
        self.dataset = Dataset.load(self.data, metrics=["sa"])
        self.config = {
            "adapter": "mock_service", "python": sys.executable, "batch_size": 2,
            "view": "front", "model_version": "mock-1",
        }
        self.adapter = create_adapter(self.config, self.dataset)
        reader = patch("vmbmk.adapters.video_inputs.read_frame", side_effect=self.frame)
        reader.start()
        self.addCleanup(reader.stop)
        self.launch = subprocess.run

    @staticmethod
    def frame(path: Path, index: int) -> Image.Image:
        red = index // 2 if path.parent.name == "ep_failure" else index
        return Image.new("RGB", (2, 2), (red, 1, 3))

    @staticmethod
    def query(anchor: int, name: str = "value", playback: str = "forward") -> ValueQuery:
        return ValueQuery(
            name, StateRef("task_001", "ep_success", anchor), "Move the cup", playback,
        )

    def launch_fixture_worker(self, command, *, env):
        """Launch the real worker with only video decoding replaced by synthetic RGB."""
        self.assertEqual(command[:3], [sys.executable, "-m", "vmbmk.inference.worker"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        completed = self.launch(
            [command[0], "-c", WORKER_FIXTURE, *command[3:]],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return completed

    def write_queries(self, queries):
        path = self.root / "queries.jsonl"
        write_jsonl(path, (query.to_dict() for query in queries))
        return path

    def write_run_config(self, **overrides):
        row = {
            "model": "mock_service", "python": sys.executable, "batch_size": 2,
            "data": str(self.data), "output": str(self.root / "results"),
            "metrics": {"sa": {"mode": "base", "domains": ["id"]}},
            "tasks": ["task_001"],
            "model_options": {"view": "front", "model_version": "mock-1"},
        }
        row.update(overrides)
        path = self.root / "mock.yaml"
        path.write_text(json.dumps(row), encoding="utf-8")
        return path

    def test_context_contains_only_ordered_rgb_and_instruction(self) -> None:
        query = self.query(3, "private:label:path")
        with patch.object(self.adapter, "_predict_value", return_value=0.4) as predict:
            self.assertEqual(self.adapter.value([query]), [0.4])
        frames, instruction = predict.call_args.args
        self.assertEqual(instruction, query.instruction)
        self.assertEqual([frame.mode for frame in frames], ["RGB"] * 4)
        self.assertEqual([frame.getpixel((0, 0)) for frame in frames], [
            (index, 1, 3) for index in range(4)
        ])
        self.assertEqual(predict.call_args.kwargs, {})

    def test_cache_rejects_changed_query_model_and_native_mode(self) -> None:
        queries = [self.query(4)]
        path = self.write_queries(queries)
        output = self.root / "bound.jsonl"
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            run_inference(self.data, path, self.config, output, gpu=None)
        original = output.read_bytes()
        for query, config, native in (
            (self.query(80), self.config, None),
            (queries[0], {**self.config, "model_version": "mock-2"}, None),
            (queries[0], self.config, "sa"),
        ):
            with self.subTest(query=query, config=config, native=native):
                self.write_queries([query])
                with patch("vmbmk.inference.dispatch.subprocess.run") as launch:
                    with self.assertRaisesRegex(VMBMKError, "inputs or model changed"):
                        run_inference(self.data, path, config, output, gpu=None, native_metric=native)
                    launch.assert_not_called()
                self.assertEqual(output.read_bytes(), original)

    def test_cache_rejects_changed_dataset_and_unbound_outputs(self) -> None:
        path = self.write_queries([self.query(4)])
        output = self.root / "bound.jsonl"
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            run_inference(self.data, path, self.config, output, gpu=None)
        metadata = self.data / "task_001" / "metadata.json"
        row = json.loads(metadata.read_text())
        row["instruction"] = "Different goal"
        write_json(metadata, row)
        with self.assertRaisesRegex(VMBMKError, "inputs or model changed"):
            run_inference(self.data, path, self.config, output, gpu=None)
        legacy = self.root / "unbound.jsonl"
        legacy.write_bytes(output.read_bytes())
        with self.assertRaisesRegex(VMBMKError, "provenance is missing"):
            run_inference(self.data, path, self.config, legacy, gpu=None)

    def test_cache_checks_parent_identity_before_partial_shard_reuse(self) -> None:
        queries = [self.query(4)]
        output = self.root / "shards.jsonl"
        bind_prediction_cache(self.data, queries, self.config, output, None)
        write_jsonl(output.with_name(output.name + ".part-002-000"), [{"query_id": "value", "op": "value", "score": 0.5}])
        path = self.write_queries([self.query(80)])
        with patch("vmbmk.inference.dispatch._run_sharded") as shard:
            with self.assertRaisesRegex(VMBMKError, "inputs or model changed"):
                run_inference(self.data, path, self.config, output, gpu=None)
            shard.assert_not_called()

    def test_empty_cache_binding_can_change_without_discarding_predictions(self) -> None:
        output = self.root / "empty.jsonl"
        bind_prediction_cache(self.data, [self.query(4)], self.config, output, None)
        bind_prediction_cache(self.data, [self.query(80)], self.config, output, None)
        self.assertFalse(output.exists())

    def test_completed_run_allows_batch_size_change_but_rejects_stale_publish(self) -> None:
        source = self.write_run_config()
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            output = run_evaluation(source)
        self.write_run_config(batch_size=1)
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch:
            self.assertEqual(run_evaluation(source), output)
            launch.assert_not_called()
        status = build_summary(self.root, output.parent, self.data)
        self.assertEqual(status["completed_count"], 1)
        published = synchronize(output, results_root=output.parent, final_root=self.root / "valid_publication", mode="publish")
        self.assertTrue(all(path.is_file() for path in published.destinations))
        metadata = self.data / "task_001" / "metadata.json"
        row = json.loads(metadata.read_text())
        row["instruction"] = "Changed task goal"
        write_json(metadata, row)
        dataset = Dataset.load(self.data, metrics=["sa"])
        report = inspect_result(source, output, dataset)
        self.assertFalse(report["valid"])
        self.assertEqual(report["state"], "stale")
        self.assertEqual(build_summary(self.root, output.parent, self.data)["stale_count"], 1)
        final = self.root / "published"
        with self.assertRaisesRegex(ValueError, "unverified current run"):
            synchronize(output, results_root=output.parent, final_root=final, mode="publish")
        self.assertFalse(final.exists())

    def test_relative_run_paths_survive_saved_config_and_publication(self) -> None:
        source = self.write_run_config(data="data", output="relative-results")
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            output = run_evaluation(source)
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch:
            self.assertEqual(run_evaluation(source), output)
            launch.assert_not_called()
        saved = RunConfig.load(output / "config.yaml")
        self.assertEqual(saved.data, self.data.resolve())
        self.assertEqual(saved.output, output.parent.resolve())
        published = synchronize(output, results_root=output.parent,
                                final_root=self.root / "relative-publication", mode="publish")
        self.assertTrue(all(path.is_file() for path in published.destinations))

    def test_completed_run_rejects_metric_scores_inconsistent_with_operations(self) -> None:
        source = self.write_run_config()
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            output = run_evaluation(source)
        path = output / "metrics.json"
        original = json.loads(path.read_text())
        for mutation in ("global_mean", "consistent_but_wrong_task_scores"):
            with self.subTest(mutation=mutation):
                metrics = json.loads(json.dumps(original))
                metrics["sa"]["mean"] = 0.25
                if mutation == "consistent_but_wrong_task_scores":
                    metrics["sa"]["domains"]["id"] = {"mean": 0.25, "tasks": {"task_001": 0.25}}
                write_json(path, metrics)
                report = inspect_result(source, output, self.dataset)
                self.assertFalse(report["valid"])
                with self.assertRaisesRegex(ConfigurationError, "saved predictions"):
                    run_evaluation(source)
                final = self.root / f"invalid-publication-{mutation}"
                with self.assertRaisesRegex(ValueError, "unverified current run"):
                    synchronize(output, results_root=output.parent, final_root=final, mode="publish")
                self.assertFalse(final.exists())

    def test_resume_rejects_corrupted_completed_metric_checkpoint(self) -> None:
        from vmbmk.metrics.registry import metric_runner

        task = set_task_metadata(self.data, ["SA", "VOC"])
        write_json(task / "episodes/ep_success/annotation.json", {
            "voc": [{"frame_index": frame} for frame in (0, 50, 90)],
        })
        source = self.write_run_config(data="data", output="partial-results", metrics={
            "sa": {"mode": "base", "domains": ["id"]},
            "voc": {"mode": "base", "domains": ["id"]},
        })

        def select(metric):
            if metric == "voc":
                def interrupted(*args, **kwargs):
                    raise RuntimeError("interrupted after SA checkpoint")
                return interrupted
            return metric_runner(metric)

        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker), patch("vmbmk.runner.run.metric_runner", side_effect=select):
            with self.assertRaisesRegex(RuntimeError, "after SA checkpoint"):
                run_evaluation(source)
        checkpoint = self.root / "partial-results/.mock.tmp/.metrics.partial.json"
        original_metrics = checkpoint.read_bytes()
        operations = checkpoint.parent / ".sa.operations.jsonl"
        original_operations = operations.read_bytes()
        for mutation in ("mean", "unexpected_query"):
            with self.subTest(mutation=mutation):
                checkpoint.write_bytes(original_metrics)
                operations.write_bytes(original_operations)
                if mutation == "mean":
                    metrics = json.loads(checkpoint.read_text())
                    metrics["sa"]["mean"] = 0.25
                    write_json(checkpoint, metrics)
                else:
                    with operations.open("a", encoding="utf-8") as output:
                        output.write(json.dumps({"query_id": "unexpected", "op": "value", "score": 0.5}) + "\n")
                with patch("vmbmk.inference.dispatch.subprocess.run") as launch:
                    with self.assertRaisesRegex(ConfigurationError, "saved predictions"):
                        run_evaluation(source)
                    launch.assert_not_called()
        self.assertFalse((self.root / "partial-results/mock").exists())
        checkpoint.write_bytes(original_metrics)
        operations.write_bytes(original_operations)
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker) as launch:
            completed = run_evaluation(source)
            launch.assert_called_once()
        metrics = json.loads((completed / "metrics.json").read_text())
        self.assertEqual(metrics["sa"]["mean"], 1.0)
        self.assertEqual(metrics["voc"]["mean"], 1.0)
        self.assertTrue(inspect_result(source, completed, Dataset.load(self.data, metrics=["sa", "voc"]))["valid"])

    def test_bound_partial_predictions_resume_with_new_batch_size(self) -> None:
        queries = [self.query(4, "first"), self.query(80, "second")]
        output = self.root / "prefix.jsonl"
        bind_prediction_cache(self.data, queries, self.config, output, None)
        write_jsonl(output, [{"query_id": "first", "op": "value", "score": 4 / (2 * 255)}])
        prefix = output.read_bytes()
        path = self.write_queries(queries)
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            run_inference(self.data, path, {**self.config, "batch_size": 1}, output, gpu=None)
        self.assertTrue(output.read_bytes().startswith(prefix))
        self.assertEqual(len(read_results(output)), 2)

    def test_completed_run_rejects_checkpoint_replacement_at_same_path(self) -> None:
        checkpoint = self.root / "weights.bin"
        checkpoint.write_bytes(b"original weights")
        source = self.write_run_config(model="robometer", gpu=0, checkpoint=str(checkpoint), model_options={})

        def infer(data, query_path, model, output, **kwargs):
            from vmbmk.inference.queries import read_queries

            write_jsonl(output, ({"query_id": query.query_id, "op": "value", "score": 0.5} for query in read_queries(query_path)))

        with patch("vmbmk.inference.execution.run_inference", side_effect=infer), patch("vmbmk.inference.dispatch.subprocess.run", side_effect=AssertionError("must not launch GPU inference")):
            output = run_evaluation(source)
        checkpoint.write_bytes(b"replacement weights with a different size")
        report = inspect_result(source, output, Dataset.load(self.data, metrics=["sa"]))
        self.assertFalse(report["valid"])
        with self.assertRaisesRegex(ConfigurationError, "inputs or implementation changed"):
            run_evaluation(source)

    def test_multigpu_parent_binding_survives_gpu_layout_change(self) -> None:
        queries = [self.query(4, "success"), ValueQuery("failure", StateRef("task_001", "ep_failure", 4), "Move cup")]
        path = self.write_queries(queries)
        output = self.root / "multigpu.jsonl"

        def worker(data, query_path, config, result_path, shard, gpu, native_metric):
            bind_prediction_cache(data, shard, config, result_path, native_metric)
            if gpu == 1:
                raise RuntimeError("interrupted shard")
            write_jsonl(result_path, ({"query_id": query.query_id, "op": "value", "score": query.state.anchor_frame / 100} for query in shard))

        with patch.object(MockServiceAdapter, "requires_gpu", True), patch("vmbmk.inference.dispatch._run_worker_process", side_effect=worker):
            with self.assertRaisesRegex(RuntimeError, "interrupted shard"):
                run_inference(self.data, path, self.config, output, gpu=[0, 1])
            run_inference(self.data, path, self.config, output, gpu=[2, 3, 4])
        self.assertEqual([result.query_id for result in read_results(output)], ["success", "failure"])
        self.assertFalse(list(output.parent.glob(output.name + ".part-*")))

    def test_mixed_sia_completed_run_and_saved_decisions_are_validated(self) -> None:
        task = set_task_metadata(self.data, ["SA", "SIA"], [
            {"id": "001", "description": "Move cup"},
            {"id": "002", "description": "Place cup"},
        ])
        for episode in ("ep_success", "ep_failure"):
            write_json(task / "episodes" / episode / "annotation.json", {"subtask_segments": [
                {"id": "001", "start_frame": 0, "end_frame_exclusive": 50},
                {"id": "002", "start_frame": 50, "end_frame_exclusive": 100},
            ]})
        source = self.write_run_config(metrics={
            "sa": {"mode": "base", "domains": ["id"]},
            "sia": {"mode": "base", "domains": ["id"]},
        })

        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker), patch("vmbmk.metrics.sia.evaluation.DeepSeekCandidateJudge", FakeCandidateJudge):
            output = run_evaluation(source)
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch:
            self.assertEqual(run_evaluation(source), output)
            launch.assert_not_called()
        identity_path = output / "sia/.inference-identities/subtask_results.jsonl.json"
        original_identity = identity_path.read_bytes()
        identity_path.unlink()
        with self.assertRaisesRegex(ConfigurationError, "prediction provenance is missing"):
            run_evaluation(source)
        self.assertFalse(identity_path.exists())
        identity_path.write_bytes(original_identity)
        query_path = output / "sia" / "queries.jsonl"
        bundle_path = output / "sia" / "responses.json"
        original_queries, original_bundle = query_path.read_bytes(), bundle_path.read_bytes()
        queries = [json.loads(line) for line in query_path.read_text().splitlines()]
        bundle = json.loads(bundle_path.read_text())
        queries[0]["state"]["anchor_frame"] += 1
        bundle["items"][0]["query"]["state"]["anchor_frame"] += 1
        write_jsonl(query_path, queries)
        write_json(bundle_path, bundle)
        with self.assertRaisesRegex(ConfigurationError, "prediction inputs or model changed"):
            run_evaluation(source)
        query_path.write_bytes(original_queries)
        bundle_path.write_bytes(original_bundle)
        decisions = output / "sia" / "decisions.jsonl"
        records = [json.loads(line) for line in decisions.read_text().splitlines()]
        records[0]["candidate_logprobs"]["001"] = math.log(0.5)
        write_jsonl(decisions, records)
        with self.assertRaisesRegex(ConfigurationError, "saved judge decisions"):
            run_evaluation(source)

    def test_standalone_sia_stages_preserve_saved_frames_and_allow_batch_change(self) -> None:
        task = set_task_metadata(self.data, ["SIA"], [
            {"id": "001", "description": "Move cup"},
            {"id": "002", "description": "Place cup"},
        ])
        for episode in ("ep_success", "ep_failure"):
            write_json(task / "episodes" / episode / "annotation.json", {"subtask_segments": [
                {"id": "001", "start_frame": 0, "end_frame_exclusive": 50},
                {"id": "002", "start_frame": 50, "end_frame_exclusive": 100},
            ]})
        settings = {"metrics": {"sia": {"mode": "base", "domains": ["id"]}}}
        source = self.write_run_config(**settings)

        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            generated = run_evaluation(source, sia_stage="generate")
        queries = (generated / "sia" / "queries.jsonl").read_bytes()
        predictions = (generated / "sia" / "subtask_results.jsonl").read_bytes()
        self.assertFalse((generated / "metrics.json").exists())
        self.write_run_config(**settings, batch_size=8)
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch, patch("vmbmk.metrics.sia.evaluation.DeepSeekCandidateJudge", FakeCandidateJudge), patch("vmbmk.runner.run._finish_run", side_effect=RuntimeError("interrupted after SIA checkpoint")):
            with self.assertRaisesRegex(RuntimeError, "after SIA checkpoint"):
                run_evaluation(source, sia_stage="judge")
            launch.assert_not_called()
        self.assertTrue((generated / ".metrics.partial.json").is_file())
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch, patch("vmbmk.metrics.sia.evaluation.DeepSeekCandidateJudge", side_effect=AssertionError("must not rejudge a valid checkpoint")):
            completed = run_evaluation(source, sia_stage="judge")
            launch.assert_not_called()
        self.assertEqual((completed / "sia" / "queries.jsonl").read_bytes(), queries)
        self.assertEqual((completed / "sia" / "subtask_results.jsonl").read_bytes(), predictions)
        self.assertAlmostEqual(json.loads((completed / "metrics.json").read_text())["sia"]["mean"], 0.4)
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch:
            self.assertEqual(run_evaluation(source), completed)
            launch.assert_not_called()
        bundle = completed / "sia" / "responses.json"
        external = self.write_run_config(**settings, output=str(self.root / "external"),
                                         sia_responses=str(bundle), sia_stage="judge")
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch, patch("vmbmk.metrics.sia.evaluation.DeepSeekCandidateJudge", FakeCandidateJudge):
            external_run = run_evaluation(external)
            self.assertEqual(run_evaluation(external), external_run)
            launch.assert_not_called()
        bundle.write_text(bundle.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "run inputs or implementation changed"):
            run_evaluation(external)
        source = self.write_run_config(**settings, output=str(self.root / "all-stage"),
                                       sia_responses=str(bundle))
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker), patch("vmbmk.metrics.sia.evaluation.DeepSeekCandidateJudge", FakeCandidateJudge):
            internal_run = run_evaluation(source)
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch:
            self.assertEqual(run_evaluation(source), internal_run)
            launch.assert_not_called()
        self.assertIsNone(json.loads((internal_run / "sia/response_source.json").read_text())["external"])

    def test_short_prefix_and_memory_keep_all_history(self) -> None:
        for anchor in (0, 1, 20):
            with self.subTest(anchor=anchor):
                frames = self.adapter._frames(self.query(anchor))
                self.assertEqual([frame.getpixel((0, 0))[0] for frame in frames],
                                 list(range(anchor + 1)))
                self.assertAlmostEqual(self.adapter.value([self.query(anchor)])[0],
                                       anchor / (2 * 255))

    def test_cycle_keeps_shared_turn_and_descending_history(self) -> None:
        frames = self.adapter._frames(self.query(97, playback="cycle"))
        self.assertEqual([frame.getpixel((0, 0))[0] for frame in frames],
                         [*range(100), 98, 97])

    def test_reverse_uses_reversed_prefix(self) -> None:
        frames = self.adapter._frames(self.query(97, playback="reverse"))
        self.assertEqual([frame.getpixel((0, 0))[0] for frame in frames], [99, 98, 97])

    def test_comparison_reuses_native_value_difference(self) -> None:
        query = CompareQuery("pair", self.query(20).state, self.query(70).state, "Move")
        self.assertAlmostEqual(self.adapter.compare([query])[0], 25 / 255)
        self.assertEqual(self.adapter.compare([]), [])

    def test_subtask_returns_mock_text_not_annotation(self) -> None:
        query = SubtaskQuery("subtask", self.query(2).state, "Move")
        self.assertEqual(self.adapter.subtask([query]), ["Mock prediction for: Move (3 frames)"])

    def test_config_omits_local_weights_and_gpu_without_changing_baselines(self) -> None:
        parsed = RunConfig.load(self.write_run_config())
        self.assertIsNone(parsed.gpu)
        self.assertIsNone(parsed.checkpoint)
        self.assertNotIn("checkpoint", parsed.model_config())
        for name in ("robometer", "vlac", "procvlm"):
            self.assertTrue(adapter_class(name).requires_gpu)
            self.assertTrue(adapter_class(name).requires_checkpoint)
        with self.assertRaisesRegex(ConfigurationError, "missing=.*checkpoint.*gpu"):
            RunConfig.load(self.write_run_config(model="robometer"))
        with self.assertRaisesRegex(ConfigurationError, "unknown=.*gpu"):
            RunConfig.load(self.write_run_config(gpu=0))

    def test_mock_config_rejects_unknown_fields_and_empty_version(self) -> None:
        with self.assertRaises(ConfigurationError):
            create_adapter(dict(self.config, checkpoint="unused"), self.dataset)
        with self.assertRaises(ConfigurationError):
            create_adapter(dict(self.config, model_version=""), self.dataset)

    def test_run_config_rejects_removed_backend_fields(self) -> None:
        for field, value in (("backend", "remote_api"), ("api", {})):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ConfigurationError, f"unknown=.*{field}"):
                    RunConfig.load(self.write_run_config(**{field: value}))

    def test_infer_cli_accepts_cpu_adapter_without_gpu_or_checkpoint(self) -> None:
        from vmbmk.cli import _inference_model, build_parser

        args = build_parser().parse_args([
            "infer", "--data", str(self.data), "--queries", "queries.jsonl",
            "--model", "mock_service", "--python", sys.executable,
            "--output", "predictions.jsonl",
        ])
        self.assertIsNone(args.gpu)
        self.assertNotIn("checkpoint", _inference_model(args))

    def test_gpu_batch_scheduler_rejects_cpu_adapter(self) -> None:
        from vmbmk.runner.batch import plan_metric_tasks

        source = self.write_run_config()
        with self.assertRaisesRegex(ValueError, "CPU adapters"):
            plan_metric_tasks(
                self.dataset, ["mock_service"], ["task_001"], ["sa"],
                {"mock_service": source}, self.root / "configs",
            )

    def test_dispatch_runs_real_cpu_worker_and_restores_operation_order(self) -> None:
        queries = [
            SubtaskQuery("text", self.query(2).state, "Move"),
            self.query(4),
            CompareQuery("pair", self.query(2).state, self.query(4).state, "Move"),
        ]
        path = self.write_queries(queries)
        output = self.root / "predictions.jsonl"
        with patch("vmbmk.inference.dispatch.subprocess.run",
                   side_effect=self.launch_fixture_worker) as launch:
            run_inference(self.data, path, self.config, output, gpu=None)
            launch.assert_called_once()
        results = read_results(output)
        self.assertEqual([result.query_id for result in results], [query.query_id for query in queries])
        self.assertEqual(results[0].output, "Mock prediction for: Move (3 frames)")
        self.assertAlmostEqual(results[1].score, 2 / 255)
        self.assertAlmostEqual(results[2].score, 1 / 255)
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch:
            run_inference(self.data, path, self.config, output, gpu=None)
            launch.assert_not_called()

    def test_worker_failure_keeps_checkpoint_and_resumes_pending_queries(self) -> None:
        queries = [self.query(index, f"value-{index}") for index in (1, 2, 3)]
        path = self.write_queries(queries)
        config_path = self.root / "adapter.json"
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        output = self.root / "predictions.jsonl"
        with (
            patch.dict(os.environ, {"VMBMK_CHECKPOINT_CHUNK_SIZE": "1"}),
            patch("vmbmk.inference.worker._configure_torch_threads") as torch,
            patch.object(MockServiceAdapter, "_predict_value",
                         side_effect=[1 / (2 * 255), RuntimeError("mock unavailable")]),
        ):
            with self.assertRaisesRegex(RuntimeError, "mock unavailable"):
                run_worker(self.data, path, config_path, output)
            torch.assert_not_called()
        prefix = output.read_bytes()
        self.assertEqual(len(read_results(output)), 1)
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            run_inference(self.data, path, self.config, output, gpu=None)
        self.assertTrue(output.read_bytes().startswith(prefix))
        results = read_results(output)
        self.assertEqual([result.query_id for result in results], [query.query_id for query in queries])
        for result, anchor in zip(results, (1, 2, 3)):
            self.assertAlmostEqual(result.score, anchor / (2 * 255))

    def test_cpu_gpu_selection_and_gpu_model_without_gpu_are_rejected(self) -> None:
        path = self.write_queries([self.query(2)])
        with self.assertRaisesRegex(ConfigurationError, "CPU adapters"):
            run_inference(self.data, path, self.config, self.root / "out.jsonl", gpu=0)
        with self.assertRaisesRegex(ConfigurationError, "gpu must"):
            run_inference(self.data, path, {"adapter": "robometer"},
                          self.root / "out.jsonl", gpu=None)

    def test_worker_exit_failure_is_not_converted_to_a_score(self) -> None:
        path = self.write_queries([self.query(2)])
        output = self.root / "predictions.jsonl"
        with patch("vmbmk.inference.dispatch.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 1)):
            with self.assertRaisesRegex(VMBMKError, "exit code 1"):
                run_inference(self.data, path, self.config, output, gpu=None)
        self.assertFalse(output.exists())

    def test_normal_runner_resumes_after_prediction_failure(self) -> None:
        source = self.write_run_config()
        planned = build_sa_queries(self.dataset, ["task_001"], ["id"])
        first_score = self.adapter.value(planned[:1])[0]

        def interrupted_worker(command, *, env):
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            run_worker(
                command[command.index("--data") + 1],
                command[command.index("--queries") + 1],
                command[command.index("--config") + 1],
                command[command.index("--output") + 1],
            )

        with (
            patch.dict(os.environ, {"VMBMK_CHECKPOINT_CHUNK_SIZE": "1"}),
            patch("vmbmk.inference.dispatch.subprocess.run", side_effect=interrupted_worker),
            patch.object(MockServiceAdapter, "_predict_value",
                         side_effect=[first_score, RuntimeError("mock unavailable")]),
        ):
            with self.assertRaisesRegex(RuntimeError, "mock unavailable"):
                run_evaluation(source)
        partial = self.root / "results" / ".mock.tmp"
        self.assertFalse((self.root / "results" / "mock").exists())
        prefix = (partial / ".sa.operations.jsonl").read_bytes()
        self.assertEqual(len(read_results(partial / ".sa.operations.jsonl")), 1)
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            output = run_evaluation(source)
        self.assertTrue((output / "operations.jsonl").read_bytes().startswith(prefix))
        self.assertEqual(len(read_results(output / "operations.jsonl")), len(planned))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["sa"]["mean"], 1.0)

    def test_normal_runner_matches_independent_sa_and_reuses_completed_run(self) -> None:
        source = self.write_run_config()
        with patch("vmbmk.inference.dispatch.subprocess.run", side_effect=self.launch_fixture_worker):
            output = run_evaluation(source)
        results = read_results(output / "operations.jsonl")
        planned = build_sa_queries(self.dataset, ["task_001"], ["id"])
        self.assertEqual([result.query_id for result in results], [query.query_id for query in planned])
        for query, result in zip(planned, results):
            red_values = [
                index // 2 if query.state.episode_id == "ep_failure" else index
                for index in range(query.state.anchor_frame + 1)
            ]
            self.assertAlmostEqual(result.score, sum(red_values) / (255 * len(red_values)))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["sa"]["mean"], 1.0)
        self.assertEqual(score_sa(self.dataset, results, ["task_001"], ["id"]), 1.0)
        with patch("vmbmk.inference.dispatch.subprocess.run") as launch:
            self.assertEqual(run_evaluation(source), output)
            launch.assert_not_called()
        for options in (
            {"view": "front", "model_version": "mock-2"},
            {"view": "wrist_left", "model_version": "mock-1"},
        ):
            with self.subTest(options=options):
                self.write_run_config(model_options=options)
                with self.assertRaisesRegex(ConfigurationError, "run inputs or implementation changed"):
                    run_evaluation(source)


if __name__ == "__main__":
    unittest.main()
