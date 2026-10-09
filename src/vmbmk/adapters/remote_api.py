"""RGB inference with model-side reference preparation over RoboValue v2."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from vmbmk.data.dataset import Dataset
from vmbmk.data.playback import PlaybackView
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.inference.queries import CompareQuery, Query, Result, SubtaskQuery, ValueQuery
from vmbmk.runner.provenance import run_identity
from vmbmk.serialization import read_json

from .base import Adapter, compare_value_difference, positive_int, positive_number
from .video_inputs import query_view, read_frame


PROTOCOL = "robovalue-inference-v2"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
PROFILE_FIELDS = {
    "current_frame_v1": set(),
    "full_prefix_v1": set(),
    "robometer_prefix_v1": {"num_frames"},
    "rynnvalue_prefix_v1": {"num_frames"},
    "procvlm_window_v1": {"window_size", "max_sampled_frames"},
}


class ObservationContext(TypedDict):
    """Ordered frames; each frame maps declared view names to base64 PNGs."""

    frames: list[dict[str, str]]


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a nonempty string")
    return value


def validate_profile(value: Any) -> dict[str, Any]:
    """Validate an explicitly selected, versioned observation sampling policy."""
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("name"), str)
        or value["name"] not in PROFILE_FIELDS
    ):
        raise ConfigurationError("API input_profile requires a known versioned name")
    name = value["name"]
    if set(value) != {"name", "views", *PROFILE_FIELDS[name]}:
        raise ConfigurationError(f"API profile {name} fields do not match")
    views = value["views"]
    if not isinstance(views, list) or not views:
        raise ConfigurationError("API profile views must be a nonempty list")
    for view in views:
        _text(view, "API profile view")
    if len(set(views)) != len(views):
        raise ConfigurationError("API profile views must be unique")
    for field in PROFILE_FIELDS[name]:
        positive_int(value[field], f"API profile {field}")
    if "num_frames" in value and value["num_frames"] < 2:
        raise ConfigurationError("API prefix num_frames must be at least two")
    return dict(value, views=list(views))


def validate_api_config(value: Any) -> dict[str, Any]:
    """Validate external API settings without loading credentials or model code."""
    required = {
        "url",
        "model_version",
        "preprocessing_version",
        "capabilities",
        "input_profile",
        "compare_mode",
        "shot_mode",
    }
    optional = {"token_env", "timeout", "max_attempts", "allow_local_http"}
    if (
        not isinstance(value, dict)
        or required - set(value)
        or set(value) - required - optional
    ):
        raise ConfigurationError("API configuration fields do not match the v2 contract")
    config = dict(value)
    url = urllib.parse.urlsplit(_text(config["url"], "API url"))
    if any(character.isspace() for character in config["url"]):
        raise ConfigurationError("API url must not contain whitespace")
    try:
        url.port
    except ValueError:
        raise ConfigurationError("API url contains an invalid port") from None
    allow_local = config.get("allow_local_http", False)
    if not isinstance(allow_local, bool):
        raise ConfigurationError("API allow_local_http must be a boolean")
    local_http = (
        allow_local and url.scheme == "http" and url.hostname in {"127.0.0.1", "::1"}
    )
    if (
        not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or (url.scheme != "https" and not local_http)
    ):
        raise ConfigurationError(
            "API url requires HTTPS without credentials/query/fragment; "
            "HTTP is only for explicit loopback mocks"
        )
    for name in ("model_version", "preprocessing_version"):
        _text(config[name], f"API {name}")
    capabilities = config["capabilities"]
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or any(
            not isinstance(operation, str)
            or operation not in {"value", "compare", "subtask"}
            for operation in capabilities
        )
        or len(set(capabilities)) != len(capabilities)
    ):
        raise ConfigurationError("API capabilities must be unique supported operation names")
    if config["compare_mode"] not in ("native", "value_difference"):
        raise ConfigurationError("API compare_mode must be native or value_difference")
    if config["compare_mode"] == "value_difference" and "value" not in capabilities:
        raise ConfigurationError("API value_difference requires the value capability")
    config["input_profile"] = validate_profile(config["input_profile"])
    if config["shot_mode"] not in ("zero_shot", "one_shot"):
        raise ConfigurationError("API shot_mode must be zero_shot or one_shot")
    if "token_env" in config:
        name = _text(config["token_env"], "API token_env")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ConfigurationError("API token_env must name an environment variable")
    config["timeout"] = positive_number(config.get("timeout", 120), "API timeout")
    config["max_attempts"] = positive_int(config.get("max_attempts", 3), "API max_attempts")
    if config["max_attempts"] > 3:
        raise ConfigurationError("API max_attempts must not exceed three")
    return config


def validate_metric_capabilities(
    api: Mapping[str, Any], metrics: Sequence[tuple]
) -> None:
    """Reject unsupported metric operations before contacting a model service."""
    for metric, mode, _domains in metrics:
        if mode != "base":
            raise ConfigurationError("Remote API v2 supports only mode=base")
        if metric == "sia":
            operation = "subtask"
        elif metric in {"tga_easy", "tga_hard", "csvc"}:
            operation = "value" if api["compare_mode"] == "value_difference" else "compare"
        else:
            operation = "value"
        if operation not in api["capabilities"]:
            raise ConfigurationError(
                f"Remote API lacks the {operation} capability required by {metric}"
            )


def profile_indices(
    profile: Mapping[str, Any], anchor: int, total_frames: int
) -> tuple[int, ...]:
    """Reuse native sampling rules, including their rounding and padding."""
    name = profile["name"]
    if name == "current_frame_v1":
        return (anchor,)
    if name == "full_prefix_v1":
        return tuple(range(anchor + 1))
    if name == "robometer_prefix_v1":
        from .robometer import _prefix_indices

        return _prefix_indices(anchor, profile["num_frames"])
    if name == "rynnvalue_prefix_v1":
        from .rynnvalue import _prefix_indices

        return _prefix_indices(anchor, profile["num_frames"])
    if name == "procvlm_window_v1":
        from .procvlm import _window_indices

        return _window_indices(
            anchor, total_frames, profile["window_size"], profile["max_sampled_frames"]
        )
    raise ConfigurationError("unknown API sampling profile")


def observation_context(
    views: Mapping[str, PlaybackView], indices: Sequence[int]
) -> ObservationContext:
    """Encode selected decoded pixels losslessly without exporting source metadata."""
    frames = []
    encoded: dict[tuple[str, int], str] = {}
    for index in indices:
        images = {}
        for name, view in views.items():
            source_index = view.source_index(index)
            key = (name, source_index)
            if key not in encoded:
                image = read_frame(view.path, source_index)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded[key] = base64.b64encode(buffer.getvalue()).decode("ascii")
            images[name] = encoded[key]
        frames.append(images)
    return {"frames": frames}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, request, file, code, message, headers, new_url
    ) -> None:
        return None


def _parse_response(
    response: Any,
    identity: Mapping[str, str],
    items: Sequence[Mapping[str, Any]],
) -> list[Result]:
    """Validate external results and restore the requested order."""
    if not isinstance(response, dict) or set(response) != {*identity, "results"}:
        raise VMBMKError("Remote API response envelope fields do not match")
    for field, expected in identity.items():
        if response[field] != expected:
            raise VMBMKError(f"Remote API response {field} mismatch")
    rows = response["results"]
    if not isinstance(rows, list):
        raise VMBMKError("Remote API results must be a list")
    expected_ops = {item["request_id"]: item["op"] for item in items}
    parsed: dict[str, Result] = {}
    for row in rows:
        if not isinstance(row, dict) or "request_id" not in row:
            raise VMBMKError("Remote API result requires request_id")
        if "query_id" in row:
            raise VMBMKError("Remote API results must not contain internal query_id")
        if not isinstance(row.get("op"), str):
            raise VMBMKError("Remote API result op must be a string")
        internal_row = dict(row)
        internal_row["query_id"] = internal_row.pop("request_id")
        result = Result.from_dict(internal_row, "remote result")
        if (
            result.query_id not in expected_ops
            or result.query_id in parsed
            or result.op != expected_ops[result.query_id]
        ):
            raise VMBMKError("Remote API result IDs or operations do not match")
        parsed[result.query_id] = result
    if set(parsed) != set(expected_ops):
        raise VMBMKError("Remote API result coverage mismatch")
    return [parsed[item["request_id"]] for item in items]


class RemoteAPIAdapter(Adapter):
    """Return ordinary adapter predictions; never accept benchmark scores."""

    def __init__(self, config: Mapping[str, Any], dataset: Dataset) -> None:
        self.model_id = _text(config["model_id"], "API model_id")
        self.api = validate_api_config(config["api"])
        self.batch_size = positive_int(config["batch_size"], "API batch_size")
        self.dataset = dataset
        self.state_path: Path | None = None
        self.state: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "RemoteAPIAdapter":
        return cls(config, dataset)

    def prepare_run(self, output: Path, queries: Sequence[Query]) -> None:
        """Bind resumable predictions and opaque request IDs to frozen inputs."""
        self.state_path = output.with_name(output.name + ".remote.json")
        settings = {
            name: value
            for name, value in self.api.items()
            if name not in {"timeout", "max_attempts", "token_env"}
        }
        identity: dict[str, Any] = {
            "protocol": PROTOCOL,
            "model_id": self.model_id,
            "api": settings,
            "queries": [query.to_dict() for query in queries],
            "inputs": run_identity(self.dataset),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if set(self.state) != {"identity", "request_ids"} or self.state["identity"] != digest:
                raise VMBMKError(
                    "Remote API inputs or model protocol changed; use a new output path"
                )
            identifiers = self.state["request_ids"]
            if not isinstance(identifiers, dict) or any(
                not isinstance(name, str) or not isinstance(value, str)
                for name, value in identifiers.items()
            ):
                raise VMBMKError("invalid remote request ID checkpoint")
            try:
                parsed = [uuid.UUID(value) for value in identifiers.values()]
            except ValueError:
                raise VMBMKError("invalid opaque request ID") from None
            if len(set(parsed)) != len(parsed):
                raise VMBMKError("duplicate opaque request IDs")
        else:
            if output.exists() and output.stat().st_size:
                raise VMBMKError("Remote predictions have no matching request checkpoint")
            self.state = {"identity": digest, "request_ids": {}}
            self._save_state()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.state, handle, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.state_path)

    def _context(self, query: ValueQuery | SubtaskQuery) -> ObservationContext:
        profile = self.api["input_profile"]
        views = {
            name: query_view(self.dataset, query, name) for name in profile["views"]
        }
        view = next(iter(views.values()))
        indices = profile_indices(
            profile,
            view.timeline_anchor(query.state.anchor_frame),
            view.timeline_num_frames,
        )
        return observation_context(views, indices)

    def _post(self, payload: dict[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        if "token_env" in self.api:
            token = os.environ.get(self.api["token_env"])
            if not token or "\r" in token or "\n" in token:
                raise ConfigurationError(
                    "API credential environment variable is missing or invalid"
                )
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.api["url"],
            data=json.dumps(payload, allow_nan=False).encode(),
            headers=headers,
            method="POST",
        )
        handlers = [_NoRedirect()]
        if urllib.parse.urlsplit(self.api["url"]).scheme == "http":
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        for attempt in range(self.api["max_attempts"]):
            try:
                with opener.open(request, timeout=self.api["timeout"]) as response:
                    body = response.read(MAX_RESPONSE_BYTES + 1)
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                if (
                    status not in {429, 500, 502, 503, 504}
                    or attempt + 1 == self.api["max_attempts"]
                ):
                    raise VMBMKError(f"Remote API request failed (HTTP {status})") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if isinstance(error, urllib.error.URLError) and isinstance(
                    error.reason, ssl.SSLCertVerificationError
                ):
                    raise VMBMKError("Remote API TLS certificate verification failed") from None
                if attempt + 1 == self.api["max_attempts"]:
                    raise VMBMKError("Remote API connection failed") from None
            else:
                if len(body) > MAX_RESPONSE_BYTES:
                    raise VMBMKError("Remote API response exceeds the 8 MiB limit")
                try:
                    return json.loads(body)
                except (ValueError, UnicodeError):
                    raise VMBMKError("Remote API returned invalid JSON") from None
            time.sleep(2 ** attempt)
        raise VMBMKError("Remote API attempts exhausted")

    def _predict(self, queries: Sequence[Query]) -> list[Result]:
        if not queries:
            return []
        if self.state_path is None:
            raise VMBMKError("Remote API requires a prepared request checkpoint")
        identity = {
            "protocol": PROTOCOL,
            "model_id": self.model_id,
            "model_version": self.api["model_version"],
            "preprocessing_version": self.api["preprocessing_version"],
            "shot_mode": self.api["shot_mode"],
        }
        request_ids = self.state["request_ids"]
        predictions = []
        for start in range(0, len(queries), self.batch_size):
            items = []
            for query in queries[start : start + self.batch_size]:
                if query.op not in self.api["capabilities"]:
                    raise ConfigurationError(f"Remote API does not support {query.op}")
                if query.query_id not in request_ids:
                    request_ids[query.query_id] = str(uuid.uuid4())
                request_id = request_ids[query.query_id]
                if isinstance(query, CompareQuery):
                    contexts = [
                        self._context(ValueQuery(query.query_id, state, query.instruction))
                        for state in (query.state_a, query.state_b)
                    ]
                else:
                    contexts = [self._context(query)]
                item = {
                    "request_id": request_id,
                    "op": query.op,
                    "instruction": query.instruction,
                    "contexts": contexts,
                }
                items.append(item)
            self._save_state()
            response = self._post({**identity, "items": items})
            predictions.extend(_parse_response(response, identity, items))
        return predictions

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        return [result.score for result in self._predict(queries)]

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        if self.api["compare_mode"] == "value_difference":
            return compare_value_difference(queries, self.value)
        return [result.score for result in self._predict(queries)]

    def subtask(self, queries: Sequence[SubtaskQuery]) -> list[str]:
        return [result.output for result in self._predict(queries)]
