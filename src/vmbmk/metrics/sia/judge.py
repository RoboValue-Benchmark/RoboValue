"""Forced-choice SIA judge API and candidate logprob contracts."""
from __future__ import annotations

import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from vmbmk.errors import VMBMKError
from .config import load_judge_config


def _candidate_labels(subtasks: Mapping[str, str]) -> dict[str, str]:
    if not 1 <= len(subtasks) <= 9:
        raise VMBMKError("SIA requires between 1 and 9 candidate subtasks")
    return {
        str(index): subtask_id
        for index, subtask_id in enumerate(sorted(subtasks), start=1)
    }


def _logprob(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value > 0
        or value <= -9999
    ):
        raise VMBMKError("SIA requires a finite, non-positive, non-sentinel logprob")
    return float(value)


def _target_logprob(
    result: Mapping[str, Any], subtasks: Mapping[str, str], target_id: str
) -> float:
    if not isinstance(result, dict) or result.get("predicted_subtask_id") not in subtasks:
        raise VMBMKError("SIA judge returned an invalid candidate ID")
    probabilities = result.get("candidate_logprobs")
    if not isinstance(probabilities, dict) or target_id not in probabilities:
        raise VMBMKError(
            f"SIA correct candidate {target_id!r} is missing from top_logprobs; "
            "exact CE cannot be computed"
        )
    return _logprob(probabilities[target_id])


def classification_prompt(
    instruction: str,
    subtasks: Mapping[str, str],
    prediction: str,
) -> str:
    choices = "\n".join(
        f"- {label}: {subtasks[subtask_id]}"
        for label, subtask_id in _candidate_labels(subtasks).items()
    )
    return (
        "You are an exacting evaluator for robot subtask classification.\n"
        "This is a forced-choice classification task: select exactly one "
        "candidate subtask for the model answer. Always choose one of the "
        "listed candidates, even when the answer is incomplete, contradictory, "
        "or unclear. Do not reject the answer or return any value outside the "
        "candidate labels.\n\n"
        f"Task:\n{instruction}\n\n"
        f"Candidate subtasks:\n{choices}\n\n"
        f"Model's current-subtask answer:\n{prediction}\n\n"
        "Output exactly one digit: the selected candidate label as listed above. "
        "Do not output spaces, newlines, quotes, JSON, punctuation, or explanations."
    )


class DeepSeekCandidateJudge:
    """Map a free-form model answer to one candidate subtask ID."""

    def __init__(
        self,
        *,
        auth_token: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_attempts: int = 20,
        retry_delay: float = 1.0,
        request: Callable[
            [str, dict[str, Any], dict[str, str], float], dict[str, Any]
        ]
        | None = None,
    ) -> None:
        settings = load_judge_config()
        self.auth_token = auth_token or os.getenv("DEEPSEEK_API_KEY") or settings.get("api_key")
        self.base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or settings.get("base_url")
            or "https://api.deepseek.com/chat/completions"
        ).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or settings.get("model") or "deepseek-v4-pro"
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self._request = request or self._post

    def _post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            print(
                f"[SIA] API request started: attempt={attempt}/"
                f"{self.max_attempts} model={payload.get('model')} "
                f"url={url} timeout={timeout:g}s",
                file=sys.stderr,
                flush=True,
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw_body = response.read()
                    status = getattr(response, "status", "unknown")
                elapsed = time.monotonic() - started
                print(
                    f"[SIA] API response received: attempt={attempt}/"
                    f"{self.max_attempts} status={status} bytes={len(raw_body)} "
                    f"elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                body = json.loads(raw_body.decode("utf-8"))
                return body
            except urllib.error.HTTPError as exc:
                cause = exc
                detail = exc.read().decode("utf-8", errors="replace").strip()
                if len(detail) > 500:
                    detail = detail[:500] + "..."
                error = VMBMKError(
                    f"SIA judge API returned HTTP {exc.code}: {detail}"
                )
                if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                    print(
                        f"[SIA] API request failed without retry: "
                        f"attempt={attempt}/{self.max_attempts} "
                        f"error_type={type(exc).__name__} "
                        f"elapsed={time.monotonic() - started:.1f}s error={error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise error from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                ssl.SSLError,
                json.JSONDecodeError,
            ) as exc:
                cause = exc
                error = VMBMKError(f"SIA judge API request failed: {exc}")
            if attempt == self.max_attempts:
                print(
                    f"[SIA] API request permanently failed: attempt={attempt}/"
                    f"{self.max_attempts} error_type={type(cause).__name__} "
                    f"elapsed={time.monotonic() - started:.1f}s error={error}",
                    file=sys.stderr,
                    flush=True,
                )
                raise error from cause
            delay = min(60.0, self.retry_delay * (2 ** (attempt - 1)))
            print(
                f"[SIA] API attempt {attempt}/{self.max_attempts} failed; "
                f"error_type={type(cause).__name__} "
                f"elapsed={time.monotonic() - started:.1f}s "
                f"retrying in {delay:g}s: {error}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _completion_url(self) -> str:
        if self.base_url.endswith(("/responses", "/chat/completions")):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _prediction(body: Any, subtasks: Mapping[str, str]) -> dict[str, Any]:
        labels = _candidate_labels(subtasks)
        try:
            if not isinstance(body, dict):
                raise ValueError("response is not an object")
            if "choices" in body:
                choices = body["choices"]
                if len(choices) != 1:
                    raise ValueError("expected one completion")
                text = choices[0]["message"]["content"]
                tokens = choices[0]["logprobs"]["content"]
            else:
                parts = [
                    part for item in body["output"] if item.get("type") == "message"
                    for part in item.get("content", [])
                    if part.get("type") == "output_text"
                ]
                text = "".join(part["text"] for part in parts)
                tokens = [token for part in parts for token in part["logprobs"]]
            if text not in labels or len(tokens) != 1 or tokens[0]["token"] != text:
                raise ValueError("answer must be exactly one single-digit token")
            token = tokens[0]
            probabilities = {}
            for entry in [*token.get("top_logprobs", []), token]:
                label = entry["token"]
                if label in labels:
                    probabilities[labels[label]] = _logprob(entry["logprob"])
            return {
                "predicted_subtask_id": labels[text],
                "candidate_logprobs": probabilities,
            }
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise VMBMKError(
                "SIA requires one digit token with token logprobs from the judge API"
            ) from exc

    def classify(
        self,
        instruction: str,
        subtasks: Mapping[str, str],
        prediction: str,
    ) -> dict[str, Any]:
        if not self.auth_token:
            raise VMBMKError(
                "SIA requires api_key in metrics/sia/config.yaml or DEEPSEEK_API_KEY"
            )
        if not self.model:
            raise VMBMKError("SIA requires DEEPSEEK_MODEL or model_options.sia_model")
        prompt = classification_prompt(instruction, subtasks, prediction)
        url = self._completion_url()
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "top_logprobs": 20,
            "thinking": {"type": "disabled"},
        }
        # Responses and Chat Completions have different logprob request fields.
        if url.endswith("/responses"):
            payload.update(input=prompt, include=["message.output_text.logprobs"])
        else:
            payload.update(messages=[{"role": "user", "content": prompt}], logprobs=True)
        body = self._request(
            url, payload,
            {
                "Authorization": f"Bearer {self.auth_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "curl/8.5.0",
            },
            self.timeout,
        )
        return self._prediction(body, subtasks)
