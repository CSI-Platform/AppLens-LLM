from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from applens_llm.blackboard import append_event


def run_lane_once(
    blackboard_path: Path,
    *,
    experiment_id: str,
    task_id: str,
    prompt: str,
    lane: dict[str, Any],
    timeout_seconds: int,
    max_tokens: int = 512,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = _post_chat_completion(lane, prompt, timeout_seconds, max_tokens)
    except HTTPError as exc:
        return _append_failure(
            blackboard_path,
            experiment_id=experiment_id,
            task_id=task_id,
            lane=lane,
            outcome="http_error",
            latency_ms=_elapsed_ms(started),
            error=f"HTTP {exc.code}",
        )
    except (TimeoutError, URLError, OSError) as exc:
        return _append_failure(
            blackboard_path,
            experiment_id=experiment_id,
            task_id=task_id,
            lane=lane,
            outcome="connection_error",
            latency_ms=_elapsed_ms(started),
            error=str(exc),
        )
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        return _append_failure(
            blackboard_path,
            experiment_id=experiment_id,
            task_id=task_id,
            lane=lane,
            outcome="invalid_response",
            latency_ms=_elapsed_ms(started),
            error=str(exc),
        )

    message = response["choices"][0]["message"]
    content = message["content"]
    payload = {
        "task_id": task_id,
        "lane_id": lane["lane_id"],
        "model_label": lane["model"]["label"],
        "engine": lane["engine"],
        "backend": lane["backend"],
        "device_selector": lane["device"]["selector"],
        "accelerator_ids": lane["device"]["accelerator_ids"],
        "latency_ms": _elapsed_ms(started),
        "outcome": "success",
        "content": content,
        "reasoning_content": message.get("reasoning_content", ""),
        "usage": response.get("usage", {}),
    }
    return append_event(
        blackboard_path,
        experiment_id=experiment_id,
        event_type="model_response",
        payload=payload,
        commit_safe=False,
    )


def _post_chat_completion(
    lane: dict[str, Any],
    prompt: str,
    timeout_seconds: int,
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": lane["model"]["label"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    request = Request(
        _chat_completion_url(lane["endpoint"]),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _chat_completion_url(endpoint: str) -> str:
    cleaned = endpoint.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _append_failure(
    blackboard_path: Path,
    *,
    experiment_id: str,
    task_id: str,
    lane: dict[str, Any],
    outcome: str,
    latency_ms: int,
    error: str,
) -> dict[str, Any]:
    payload = {
        "task_id": task_id,
        "lane_id": lane["lane_id"],
        "model_label": lane["model"]["label"],
        "engine": lane["engine"],
        "backend": lane["backend"],
        "device_selector": lane["device"]["selector"],
        "accelerator_ids": lane["device"]["accelerator_ids"],
        "latency_ms": latency_ms,
        "outcome": outcome,
        "error": error,
    }
    return append_event(
        blackboard_path,
        experiment_id=experiment_id,
        event_type="failure",
        payload=payload,
        commit_safe=False,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
