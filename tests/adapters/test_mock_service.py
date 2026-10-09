from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from synthetic_data import make_dataset, set_task_metadata
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
                with self.assertRaisesRegex(ConfigurationError, "config does not match"):
                    run_evaluation(source)


if __name__ == "__main__":
    unittest.main()
