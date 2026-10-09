from __future__ import annotations

import base64
import copy
import importlib.util
import io
import json
import os
import ssl
import sys
import tempfile
import threading
import unittest
import urllib.error
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from synthetic_data import make_dataset
from vmbmk.adapters.registry import canonical_baseline
from vmbmk.adapters.remote_api import (
    PROTOCOL, RemoteAPIAdapter, observation_context, profile_indices,
    validate_api_config, validate_metric_capabilities,
)
from vmbmk.data.dataset import Dataset
from vmbmk.data.playback import PlaybackView
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.inference.dispatch import run_inference
from vmbmk.inference.queries import (
    CompareQuery, Result, StateRef, SubtaskQuery, ValueQuery, read_results,
)
from vmbmk.serialization import write_jsonl
from vmbmk.tools.mock_api import make_server, mock_prediction


def api_config() -> dict:
    return {
        "url": "https://example.invalid/inference", "model_version": "1",
        "preprocessing_version": "mock-rgb-v1", "capabilities": ["value", "compare", "subtask"],
        "input_profile": {"name": "current_frame_v1", "views": ["front"]},
        "compare_mode": "native", "shot_mode": "zero_shot",
    }


def prediction_response(payload: dict) -> dict:
    return {
        name: payload[name]
        for name in (
            "protocol", "model_id", "model_version", "preprocessing_version", "shot_mode"
        )
    } | {"results": [
        {"request_id": item["request_id"], "op": item["op"], **mock_prediction(item)}
        for item in reversed(payload["items"])
    ]}


class RemoteAPITest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = make_dataset(self.root / "data")
        self.dataset = Dataset.load(self.data)
        self.config = {
            "adapter": "remote_api", "model_id": "mock-value",
            "batch_size": 2, "api": api_config(),
        }
        self.reader = patch("vmbmk.adapters.remote_api.read_frame", side_effect=self.frame)
        self.reader.start()
        self.addCleanup(self.reader.stop)

    @staticmethod
    def frame(path: Path, index: int) -> Image.Image:
        return Image.new("RGB", (2, 2), (index, 1 if path.stem == "front" else 2, 3))

    def query(self, frame: int = 8, name: str = "private:correct:stage") -> ValueQuery:
        return ValueQuery(name, StateRef("task_001", "ep_success", frame), "Move the cup")

    def adapter(self, queries: list, filename: str = "predictions.jsonl") -> RemoteAPIAdapter:
        adapter = RemoteAPIAdapter.from_config(self.config, self.dataset)
        adapter.prepare_run(self.root / filename, queries)
        return adapter

    def serve(self, server: ThreadingHTTPServer) -> str:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()

        def close() -> None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(close)
        return f"http://127.0.0.1:{server.server_port}/inference"

    def test_three_operations_over_http(self) -> None:
        self.config["api"].update(url=self.serve(make_server(0)), allow_local_http=True)
        before, after = StateRef("task_001", "ep_success", 1), StateRef("task_001", "ep_success", 9)
        queries = [
            self.query(8, "value:correct"),
            self.query(3, "value:negative"),
            CompareQuery("trr:stage:role", before, after, "Move the cup"),
            SubtaskQuery("sia:ground_truth", after, "Move the cup"),
        ]
        adapter = self.adapter(queries)
        self.assertEqual(adapter.value(queries[:2]), [8, 3])
        self.assertEqual(adapter.compare(queries[2:3]), [8])
        self.assertEqual(adapter.subtask(queries[3:]), ["Synthetic mock subtask."])
        self.assertEqual(adapter.value(queries[:2]), [8, 3])

    def test_request_ids_are_generated_once_and_restored(self) -> None:
        query = self.query()
        adapter = self.adapter([query])
        with (
            patch("vmbmk.adapters.remote_api.uuid.uuid4", wraps=uuid.uuid4) as generate_id,
            patch.object(RemoteAPIAdapter, "_post", side_effect=prediction_response) as request,
        ):
            self.assertEqual(adapter.value([query]), [8])
            self.assertEqual(adapter.value([query]), [8])
            resumed = self.adapter([query])
            self.assertEqual(resumed.value([query]), [8])
        generate_id.assert_called_once()
        identifiers = {
            call.args[0]["items"][0]["request_id"]
            for call in request.call_args_list
        }
        self.assertEqual(len(identifiers), 1)

    def test_lossless_multiview_and_sampling(self) -> None:
        from vmbmk.adapters.procvlm import _window_indices
        from vmbmk.adapters.robometer import _prefix_indices as robometer_indices
        from vmbmk.adapters.rynnvalue import _prefix_indices as rynn_indices

        cases = [
            ({"name": "current_frame_v1"}, (1,)),
            ({"name": "full_prefix_v1"}, (0, 1)),
            ({"name": "robometer_prefix_v1", "num_frames": 4}, robometer_indices(1, 4)),
            ({"name": "rynnvalue_prefix_v1", "num_frames": 4}, rynn_indices(1, 4)),
            (
                {"name": "procvlm_window_v1", "window_size": 4, "max_sampled_frames": 12},
                _window_indices(1, 100, 4, 12),
            ),
        ]
        views = {name: PlaybackView(Path(name), 100, 10) for name in ("front", "wrist_left")}
        for profile, expected in cases:
            with self.subTest(profile=profile):
                indices = profile_indices(profile, 1, 100)
                self.assertEqual(indices, expected)
                context = observation_context(views, indices)
                for frame, source_index in zip(context["frames"], expected):
                    for name, encoded in frame.items():
                        pixels = Image.open(io.BytesIO(base64.b64decode(encoded)))
                        self.assertEqual(
                            pixels.tobytes(), self.frame(Path(name), source_index).tobytes()
                        )
        self.assertNotEqual(robometer_indices(7, 4), rynn_indices(7, 4))

    def test_memory_and_cycle_history(self) -> None:
        self.config["api"]["input_profile"] = {"name": "full_prefix_v1", "views": ["front"]}
        forward = self.query(3, "voc_mem:history")
        cycle = ValueQuery(
            "cycle_voc:reverse", StateRef("task_001", "ep_success", 98),
            "Move the cup", "cycle",
        )
        adapter = self.adapter([forward, cycle])
        for query, expected in ((forward, [0, 1, 2, 3]), (cycle, [*range(100), 98])):
            context = adapter._context(query)
            values = [
                Image.open(io.BytesIO(base64.b64decode(frame["front"]))).getpixel((0, 0))[0]
                for frame in context["frames"]
            ]
            self.assertEqual(values, expected)

    def test_one_shot_uses_service_without_reference_inputs(self) -> None:
        self.config["api"].update(
            shot_mode="one_shot",
            url=self.serve(make_server(0, shot_mode="one_shot")),
            allow_local_http=True,
        )
        query = self.query()
        adapter = self.adapter([query])
        captured = []
        post = adapter._post

        def respond(payload):
            captured.append(payload)
            return post(payload)

        with patch.object(adapter, "_post", side_effect=respond):
            self.assertEqual(adapter.value([query]), [8])
        self.assertEqual(captured[0]["shot_mode"], "one_shot")
        self.assertEqual(
            set(captured[0]["items"][0]),
            {"request_id", "op", "instruction", "contexts"},
        )
        self.assertNotIn("training", json.dumps(adapter.state))
        self.config["api"]["preprocessing_version"] = "changed-service-references"
        with self.assertRaisesRegex(VMBMKError, "changed"):
            self.adapter([query])

    def test_scalar_difference_preserves_left_right_order(self) -> None:
        self.config["api"].update(compare_mode="value_difference", capabilities=["value"])
        queries = [CompareQuery(
            "compare:correct", StateRef("task_001", "ep_success", 3),
            StateRef("task_001", "ep_success", 8), "Move the cup",
        )]
        adapter = self.adapter(queries)
        with patch.object(adapter, "_post", side_effect=prediction_response):
            self.assertEqual(adapter.compare(queries), [5])
        identifiers = adapter.state["request_ids"]
        self.assertIn("compare:correct:a", identifiers)
        self.assertIn("compare:correct:b", identifiers)

    def test_reference_configuration_is_rejected(self) -> None:
        for mode in ("zero_shot", "one_shot"):
            config = api_config() | {
                "shot_mode": mode,
                "reference": {"root": str(self.data)},
            }
            with self.subTest(mode=mode), self.assertRaises(ConfigurationError):
                validate_api_config(config)

    def test_service_rejects_reference_payload(self) -> None:
        self.config["api"].update(
            url=self.serve(make_server(0)), allow_local_http=True,
        )
        query = self.query()
        adapter = self.adapter([query])
        post = adapter._post

        def send_reference(payload):
            payload["items"][0]["reference"] = payload["items"][0]["contexts"][0]
            return post(payload)

        with patch.object(adapter, "_post", side_effect=send_reference):
            with self.assertRaisesRegex(VMBMKError, "HTTP 400"):
                adapter.value([query])

    def test_wire_does_not_export_private_metadata(self) -> None:
        query = self.query()
        adapter = self.adapter([query])
        payloads = []

        def respond(payload):
            payloads.append(payload)
            return prediction_response(payload)

        with patch.object(adapter, "_post", side_effect=respond):
            adapter.value([query])
        wire = json.dumps(payloads[0])
        for sensitive in (
            query.query_id, "task_001", "ep_success", str(self.root),
            "anchor_frame", "playback", "success", "annotation",
        ):
            self.assertNotIn(sensitive, wire)
        self.assertEqual(
            set(payloads[0]["items"][0]), {"request_id", "op", "instruction", "contexts"}
        )

    def test_strict_response_validation(self) -> None:
        def corrupt(response, case):
            if case == "version":
                response["model_version"] = "wrong"
            elif case == "missing":
                response["results"] = []
            elif case == "duplicate":
                response["results"] *= 2
            elif case == "extra":
                response["results"][0]["request_id"] = "unexpected"
            elif case == "operation":
                response["results"][0]["op"] = "compare"
            elif case == "extra_field":
                response["results"][0]["query_id"] = "private"
            elif case == "boolean":
                response["results"][0]["score"] = True
            elif case == "nan":
                response["results"][0]["score"] = float("nan")
            elif case == "infinity":
                response["results"][0]["score"] = float("inf")
            elif case == "operation_type":
                response["results"][0]["op"] = []
            elif case == "protocol":
                response["protocol"] = "wrong"
            elif case == "preprocessing":
                response["preprocessing_version"] = "wrong"
            elif case == "envelope":
                response["unexpected"] = True
            elif case == "shot_mode":
                response["shot_mode"] = "one_shot"
            return response

        query = self.query()
        for case in (
            "version", "missing", "duplicate", "extra", "operation", "extra_field",
            "boolean", "nan", "infinity", "operation_type", "protocol",
            "preprocessing", "envelope", "shot_mode",
        ):
            with self.subTest(case=case):
                adapter = self.adapter([query], f"{case}.jsonl")
                with patch.object(
                    adapter, "_post",
                    side_effect=lambda payload: corrupt(prediction_response(payload), case),
                ):
                    with self.assertRaises(VMBMKError):
                        adapter.value([query])

    def test_empty_subtask_text_is_rejected(self) -> None:
        query = SubtaskQuery("sia:private", self.query().state, "Move the cup")
        adapter = self.adapter([query])

        def respond(payload):
            response = prediction_response(payload)
            response["results"][0]["output"] = ""
            return response

        with patch.object(adapter, "_post", side_effect=respond):
            with self.assertRaises(VMBMKError):
                adapter.subtask([query])

    def test_config_rejects_insecure_or_ambiguous_settings(self) -> None:
        for changes in (
            {"url": "http://example.invalid"}, {"url": "https://user:secret@example.invalid"},
            {"url": "https://example.invalid?token=secret"}, {"timeout": 0},
            {"max_attempts": 4}, {"capabilities": ["unknown"]}, {"api_key": "secret"},
            {"input_profile": {"name": "unknown", "views": ["front"]}},
            {"input_profile": {"name": "current_frame_v1", "views": []}},
            {"input_profile": {"name": "rynnvalue_prefix_v1", "views": ["front"], "num_frames": 1}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                validate_api_config(api_config() | changes)
        with self.assertRaises(ConfigurationError):
            validate_metric_capabilities(
                api_config() | {"capabilities": ["subtask"]}, [("sa", "base", ("id",))]
            )
        with self.assertRaises(ConfigurationError):
            validate_metric_capabilities(api_config(), [("sa", "native", ("id",))])

    def test_model_identity_includes_version(self) -> None:
        config = {"model": "new-model", "backend": "remote_api", "api": api_config()}
        original = canonical_baseline(config)
        config["api"]["model_version"] = "2"
        self.assertNotEqual(original, canonical_baseline(config))
        self.assertTrue(original.startswith("new-model__api__"))

    def test_dispatch_resume_and_changed_cache(self) -> None:
        query = self.query()
        second = self.query(4, "private:second")
        queries = [query, second]
        query_path, output = self.root / "queries.jsonl", self.root / "output.jsonl"
        write_jsonl(query_path, (item.to_dict() for item in queries))
        self.config["batch_size"] = 1
        captured = []

        def fail_second(payload):
            captured.append(payload)
            if len(captured) == 2:
                raise VMBMKError("synthetic interruption")
            return prediction_response(payload)

        with (
            patch(
                "vmbmk.inference.worker._configure_torch_threads",
                side_effect=AssertionError("torch must not load"),
            ),
            patch(
                "vmbmk.inference.dispatch._run_worker_process",
                side_effect=AssertionError("GPU worker must not run"),
            ),
        ):
            with patch.object(RemoteAPIAdapter, "_post", side_effect=fail_second):
                with self.assertRaisesRegex(VMBMKError, "interruption"):
                    run_inference(self.data, query_path, self.config, output, gpu=None)
            self.assertEqual(read_results(output), [Result(query.query_id, "value", 8)])
            resumed = []

            def respond(payload):
                resumed.append(payload)
                return prediction_response(payload)

            with patch.object(RemoteAPIAdapter, "_post", side_effect=respond):
                run_inference(self.data, query_path, self.config, output, gpu=None)
                run_inference(self.data, query_path, self.config, output, gpu=None)
            self.assertEqual(len(resumed), 1)
            self.assertEqual(
                captured[1]["items"][0]["request_id"], resumed[0]["items"][0]["request_id"]
            )
            self.assertEqual(
                read_results(output),
                [Result(query.query_id, "value", 8), Result(second.query_id, "value", 4)],
            )
            for field, value in (
                ("model_version", "2"),
                ("input_profile", {"name": "full_prefix_v1", "views": ["front"]}),
                ("shot_mode", "one_shot"),
            ):
                changed = copy.deepcopy(self.config)
                changed["api"][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(VMBMKError, "changed"):
                    run_inference(self.data, query_path, changed, output, gpu=None)
        self.assertNotIn("torch", sys.modules)

    def test_local_and_remote_predictions_and_sa_scores_match(self) -> None:
        from synthetic_data import set_task_metadata
        from vmbmk.adapters.base import Adapter
        from vmbmk.inference.worker import run_worker
        from vmbmk.metrics.sa import build_sa_queries, score_sa, score_sa_by_domain

        set_task_metadata(self.data, ["SA"])
        dataset = Dataset.load(self.data, metrics=["sa"])
        queries = build_sa_queries(dataset, ["task_001"], ["id"])
        query_path = self.root / "sa-queries.jsonl"
        write_jsonl(query_path, (query.to_dict() for query in queries))
        local_config = self.root / "local.json"
        local_config.write_text(json.dumps({"adapter": "synthetic", "batch_size": 2}))
        local_output, remote_output = self.root / "local.jsonl", self.root / "remote.jsonl"

        class SyntheticAdapter(Adapter):
            def value(self, planned):
                return [
                    float(
                        query.state.anchor_frame // 2
                        if query.state.episode_id == "ep_failure"
                        else query.state.anchor_frame
                    )
                    for query in planned
                ]

        with (
            patch("vmbmk.inference.worker._configure_torch_threads"),
            patch("vmbmk.inference.worker.create_adapter", return_value=SyntheticAdapter()),
        ):
            run_worker(self.data, query_path, local_config, local_output)
        def synthetic_frame(path, index):
            return self.frame(path, index // 2 if path.parent.name == "ep_failure" else index)

        with (
            patch.object(RemoteAPIAdapter, "_post", side_effect=prediction_response),
            patch("vmbmk.adapters.remote_api.read_frame", side_effect=synthetic_frame),
        ):
            run_inference(self.data, query_path, self.config, remote_output, gpu=None)
        local, remote = read_results(local_output), read_results(remote_output)
        self.assertEqual(local, remote)
        self.assertEqual(score_sa(dataset, local), score_sa(dataset, remote))
        self.assertEqual(score_sa(dataset, remote), 1)
        self.assertEqual(score_sa_by_domain(dataset, local), score_sa_by_domain(dataset, remote))

    def test_data_change_rejects_cached_results(self) -> None:
        query = self.query()
        output = self.root / "predictions.jsonl"
        adapter = self.adapter([query])
        with patch.object(adapter, "_post", side_effect=prediction_response):
            adapter.value([query])
        write_jsonl(output, [Result(query.query_id, "value", 8).to_dict()])
        video = self.data / "task_001" / "episodes" / "ep_success" / "front.mp4"
        video.write_bytes(b"changed synthetic asset")
        with self.assertRaisesRegex(VMBMKError, "changed"):
            self.adapter([query])

    def test_external_video_change_rejects_cached_results(self) -> None:
        video = self.root / "shared.mp4"
        video.write_bytes(b"synthetic external asset")
        metadata = self.data / "task_001" / "episodes" / "ep_success" / "metadata.json"
        row = json.loads(metadata.read_text())
        row["assets"]["videos"]["front"] = os.path.relpath(video, metadata.parent)
        metadata.write_text(json.dumps(row), encoding="utf-8")
        self.dataset = Dataset.load(self.data)
        query = self.query()
        output = self.root / "predictions.jsonl"
        self.adapter([query])
        write_jsonl(output, [Result(query.query_id, "value", 8).to_dict()])
        video.write_bytes(b"changed external asset with a different size")
        with self.assertRaisesRegex(VMBMKError, "changed"):
            self.adapter([query])

    def test_oversized_response_fails_without_retry(self) -> None:
        self.config["api"].update(
            url=self.serve(make_server(0)), allow_local_http=True,
        )
        query = self.query()
        adapter = self.adapter([query])
        with (
            patch("vmbmk.adapters.remote_api.MAX_RESPONSE_BYTES", 4),
            patch("vmbmk.adapters.remote_api.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(VMBMKError, "response exceeds"):
                adapter.value([query])
        sleep.assert_not_called()

    def test_invalid_credentials_never_reach_temporary_config(self) -> None:
        query_path = self.root / "queries.jsonl"
        write_jsonl(query_path, [self.query().to_dict()])
        invalid = copy.deepcopy(self.config)
        invalid["api"]["api_key"] = "secret-token"
        output = self.root / "untouched" / "result.jsonl"
        with self.assertRaises(ConfigurationError):
            run_inference(self.data, query_path, invalid, output, gpu=None)
        self.assertFalse(output.parent.exists())

    def test_retry_reuses_ids_and_keeps_tokens_out_of_state(self) -> None:
        captured = []
        headers = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captured.append(payload)
                headers.append(self.headers.get("Authorization"))
                if len(captured) == 1:
                    self.send_error(503)
                else:
                    body = json.dumps(prediction_response(payload)).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        self.config["api"].update(
            url=self.serve(ThreadingHTTPServer(("127.0.0.1", 0), Handler)),
            allow_local_http=True, token_env="ROBOVALUE_TEST_TOKEN",
        )
        query = self.query()
        adapter = self.adapter([query])
        with (
            patch.dict(os.environ, {"ROBOVALUE_TEST_TOKEN": "secret-token"}),
            patch("vmbmk.adapters.remote_api.time.sleep") as sleep,
        ):
            self.assertEqual(adapter.value([query]), [8])
        self.assertEqual(captured[0], captured[1])
        self.assertEqual(headers, ["Bearer secret-token", "Bearer secret-token"])
        self.assertEqual(sleep.call_count, 1)
        self.assertNotIn("secret-token", adapter.state_path.read_text())

    def test_redirect_is_not_followed(self) -> None:
        received = []
        destination = self.serve(make_server(0))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                received.append(True)
                self.send_response(302)
                self.send_header("Location", destination)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.config["api"].update(
            url=self.serve(ThreadingHTTPServer(("127.0.0.1", 0), Handler)),
            allow_local_http=True,
        )
        query = self.query()
        adapter = self.adapter([query])
        with self.assertRaisesRegex(VMBMKError, "HTTP 302"):
            adapter.value([query])
        self.assertEqual(received, [True])

    def test_connection_errors_are_bounded_and_redacted(self) -> None:
        query = self.query()
        adapter = self.adapter([query])
        failure = urllib.error.URLError("sensitive diagnostic secret-token")
        with (
            patch("urllib.request.OpenerDirector.open", side_effect=failure) as request,
            patch("vmbmk.adapters.remote_api.time.sleep"),
        ):
            with self.assertRaises(VMBMKError) as raised:
                adapter.value([query])
        self.assertEqual(request.call_count, 3)
        self.assertNotIn("secret-token", str(raised.exception))
        failure = urllib.error.URLError(ssl.SSLCertVerificationError("bad certificate"))
        with patch("urllib.request.OpenerDirector.open", side_effect=failure) as request:
            with self.assertRaisesRegex(VMBMKError, "TLS"):
                adapter.value([query])
        self.assertEqual(request.call_count, 1)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "existing PyYAML environment required")
    def test_run_configuration_without_local_runtime(self) -> None:
        from vmbmk.runner.run import RunConfig

        config = {
            "backend": "remote_api", "model": "mock-value", "batch_size": 2,
            "data": str(self.data), "output": str(self.root / "results"),
            "metrics": {"sa": {"mode": "base", "domains": ["id"]}},
            "tasks": ["task_001"], "api": api_config(),
        }
        source = self.root / "run.yaml"
        for mode in ("zero_shot", "one_shot"):
            config["api"]["shot_mode"] = mode
            source.write_text(json.dumps(config))
            with self.subTest(mode=mode):
                parsed = RunConfig.load(source)
                self.assertIsNone(parsed.python)
                self.assertIsNone(parsed.checkpoint)
                self.assertIsNone(parsed.gpu)
                self.assertEqual(parsed.model_config()["adapter"], "remote_api")
                self.assertEqual(parsed.api["shot_mode"], mode)
                self.assertNotIn("reference", parsed.api)
        config["gpu"] = 0
        source.write_text(json.dumps(config))
        with self.assertRaises(ConfigurationError):
            RunConfig.load(source)


if __name__ == "__main__":
    unittest.main()
