"""Synthetic model service for CPU API contract tests."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from vmbmk.adapters.remote_api import PROTOCOL


def mock_prediction(item: dict[str, Any]) -> dict[str, Any]:
    """Return toy predictions from PNG pixels, not benchmark measurements."""
    from PIL import Image

    if item["op"] == "subtask":
        return {"output": "Synthetic mock subtask."}
    values = []
    for context in item["contexts"]:
        frame = context["frames"][-1]
        pixels = base64.b64decode(next(iter(frame.values())), validate=True)
        with Image.open(io.BytesIO(pixels)) as image:
            values.append(float(image.convert("RGB").getpixel((0, 0))[0]))
    return {"score": values[-1] - values[0] if item["op"] == "compare" else values[0]}


def make_server(
    port: int = 8765,
    *,
    model_id: str = "mock-value",
    model_version: str = "1",
    preprocessing_version: str = "mock-rgb-v1",
    shot_mode: str = "zero_shot",
) -> ThreadingHTTPServer:
    """Bind a loopback-only mock with replay-stable opaque request IDs."""
    identity = {
        "protocol": PROTOCOL,
        "model_id": model_id,
        "model_version": model_version,
        "preprocessing_version": preprocessing_version,
        "shot_mode": shot_mode,
    }
    cache: dict[str, tuple[str, dict[str, Any]]] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("invalid request length")
                payload = json.loads(self.rfile.read(length))
                if set(payload) != {*identity, "items"} or any(
                    payload[name] != value for name, value in identity.items()
                ):
                    raise ValueError("identity mismatch")
                items = payload["items"]
                if not isinstance(items, list) or not items:
                    raise ValueError("items required")
                results = []
                with lock:
                    identifiers = set()
                    for item in items:
                        expected = {"request_id", "op", "instruction", "contexts"}
                        if set(item) != expected:
                            raise ValueError("unexpected item fields")
                        request_id = item["request_id"]
                        if not isinstance(request_id, str):
                            raise ValueError("request ID must be a string")
                        uuid.UUID(request_id)
                        if request_id in identifiers:
                            raise ValueError("duplicate request ID")
                        identifiers.add(request_id)
                        operation = item["op"]
                        if operation not in {"value", "compare", "subtask"}:
                            raise ValueError("unknown operation")
                        if len(item["contexts"]) != (2 if operation == "compare" else 1):
                            raise ValueError("context count mismatch")
                        if (
                            not isinstance(item["instruction"], str)
                            or not item["instruction"]
                        ):
                            raise ValueError("instruction required")
                        for context in item["contexts"]:
                            if set(context) != {"frames"} or not context["frames"]:
                                raise ValueError("context frames required")
                            for frame in context["frames"]:
                                if not isinstance(frame, dict) or not frame:
                                    raise ValueError("frame views required")
                                for image in frame.values():
                                    pixels = base64.b64decode(image, validate=True)
                                    if not pixels.startswith(b"\x89PNG\r\n\x1a\n"):
                                        raise ValueError("PNG image required")
                        digest = hashlib.sha256(
                            json.dumps(item, sort_keys=True).encode()
                        ).hexdigest()
                        if request_id in cache:
                            old_digest, result = cache[request_id]
                            if old_digest != digest:
                                raise ValueError("request ID reused with different inputs")
                        else:
                            result = {
                                "request_id": request_id,
                                "op": operation,
                                **mock_prediction(item),
                            }
                            cache[request_id] = (digest, result)
                        results.append(result)
                body = json.dumps(
                    {**identity, "results": list(reversed(results))}, allow_nan=False
                ).encode()
            except (ValueError, TypeError, KeyError, OSError):
                self.send_error(400, "Invalid RoboValue inference request")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-id", default="mock-value")
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--preprocessing-version", default="mock-rgb-v1")
    parser.add_argument(
        "--shot-mode", choices=("zero_shot", "one_shot"), default="zero_shot"
    )
    args = parser.parse_args()
    with make_server(
        args.port,
        model_id=args.model_id,
        model_version=args.model_version,
        preprocessing_version=args.preprocessing_version,
        shot_mode=args.shot_mode,
    ) as server:
        print(
            f"Synthetic mock listening on http://127.0.0.1:{args.port}; "
            "not for real test data",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
