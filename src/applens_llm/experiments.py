from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from applens_llm.blackboard import append_event, start_experiment
from applens_llm.handoff_contracts import BLACKBOARD_CONTRACT, build_deep_review_prompt, build_fast_lane_prompt
from applens_llm.lane_processes import endpoint_host_port, start_lane, stop_lane
from applens_llm.orchestrator import run_lane_once
from applens_llm.runtime_lanes import get_lane


StartFn = Callable[..., dict[str, Any]]
StopFn = Callable[..., dict[str, Any]]
WaitFn = Callable[[str, int], bool]
RunFn = Callable[..., dict[str, Any]]


def run_two_lane_experiment(
    *,
    config: dict[str, Any],
    fast_lane_id: str,
    deep_lane_id: str,
    experiment_id: str,
    prompt: str,
    blackboard_path: Path,
    summary_path: Path,
    state_path: Path,
    logs_dir: Path,
    timeout_seconds: int = 120,
    deep_timeout_seconds: int = 600,
    fast_max_tokens: int = 256,
    deep_max_tokens: int = 512,
    driver_evidence: list[dict[str, Any]] | None = None,
    skip_start: bool = False,
    keep_running: bool = False,
    start_fn: StartFn = start_lane,
    stop_fn: StopFn = stop_lane,
    wait_fn: WaitFn = None,
    run_fn: RunFn = run_lane_once,
) -> dict[str, Any]:
    wait = wait_fn or wait_for_endpoint
    fast_lane = get_lane(config, fast_lane_id)
    deep_lane = get_lane(config, deep_lane_id)
    started_lane_ids: list[str] = []
    start_records: list[dict[str, Any]] = []
    stop_records: list[dict[str, Any]] = []

    start_experiment(blackboard_path, experiment_id=experiment_id, title="Two-lane runtime experiment")
    append_event(
        blackboard_path,
        experiment_id=experiment_id,
        event_type="task",
        payload={
            "task_id": "fast-task",
            "prompt": prompt,
            "metadata": {"lane_id": fast_lane_id, "blackboard_contract": BLACKBOARD_CONTRACT},
        },
    )

    try:
        if not skip_start:
            for lane in (fast_lane, deep_lane):
                start_records.append(start_fn(lane, state_path=state_path, logs_dir=logs_dir))
                started_lane_ids.append(lane["lane_id"])
                wait(lane["endpoint"], timeout_seconds)

        fast_prompt = build_fast_lane_prompt(
            original_prompt=prompt,
            fast_lane_id=fast_lane_id,
            deep_lane_id=deep_lane_id,
            iteration_label="Single handoff experiment",
            fast_backend=fast_lane.get("backend"),
            deep_backend=deep_lane.get("backend"),
        )
        fast_event = run_fn(
            blackboard_path,
            experiment_id=experiment_id,
            task_id="fast-task",
            prompt=fast_prompt,
            lane=fast_lane,
            timeout_seconds=timeout_seconds,
            max_tokens=fast_max_tokens,
        )
        fast_content = fast_event["payload"].get("content") or fast_event["payload"].get("reasoning_content") or ""
        deep_prompt = build_deep_review_prompt(
            original_prompt=prompt,
            fast_content=fast_content,
            fast_lane_id=fast_lane_id,
            deep_lane_id=deep_lane_id,
            iteration_label="Single handoff experiment",
            fast_backend=fast_lane.get("backend"),
            deep_backend=deep_lane.get("backend"),
        )
        append_event(
            blackboard_path,
            experiment_id=experiment_id,
            event_type="handoff",
            payload={
                "from_lane_id": fast_lane_id,
                "to_lane_id": deep_lane_id,
                "source_task_id": "fast-task",
                "target_task_id": "deep-review",
                "prompt": deep_prompt,
                "blackboard_contract": BLACKBOARD_CONTRACT,
            },
        )
        deep_event = run_fn(
            blackboard_path,
            experiment_id=experiment_id,
            task_id="deep-review",
            prompt=deep_prompt,
            lane=deep_lane,
            timeout_seconds=deep_timeout_seconds,
            max_tokens=deep_max_tokens,
        )
    finally:
        if not skip_start and not keep_running:
            for lane_id in reversed(started_lane_ids):
                stop_records.append(stop_fn(lane_id, state_path=state_path))

    summary = {
        "schema_version": "0.1",
        "experiment_id": experiment_id,
        "created_at": _utc_now(),
        "lanes": {"fast": fast_lane_id, "deep": deep_lane_id},
        "driver_evidence": driver_evidence or [],
        "blackboard_contract": BLACKBOARD_CONTRACT,
        "blackboard": str(blackboard_path),
        "responses": {
            "fast": _response_summary(fast_event),
            "deep": _response_summary(deep_event),
        },
        "lifecycle": {
            "skip_start": skip_start,
            "keep_running": keep_running,
            "started": start_records,
            "stopped": stop_records,
        },
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def wait_for_endpoint(endpoint: str, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    health_url = _health_url(endpoint)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=min(2.0, timeout_seconds)) as response:
                if response.status == 200:
                    return True
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(f"endpoint did not become ready: {endpoint}") from last_error


def _health_url(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{scheme}://{host}{port}/health"


def _response_summary(event: dict[str, Any]) -> dict[str, Any]:
    payload = event["payload"]
    return {
        "event_type": event["event_type"],
        "lane_id": payload.get("lane_id"),
        "outcome": payload.get("outcome"),
        "latency_ms": payload.get("latency_ms"),
        "content_length": len(str(payload.get("content", ""))),
        "reasoning_content_length": len(str(payload.get("reasoning_content", ""))),
        "usage": payload.get("usage", {}),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
