from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from applens_llm.blackboard import append_event, start_experiment
from applens_llm.experiments import wait_for_endpoint
from applens_llm.handoff_contracts import BLACKBOARD_CONTRACT, build_deep_review_prompt, build_fast_lane_prompt
from applens_llm.lane_processes import start_lane, stop_lane
from applens_llm.orchestrator import run_lane_once
from applens_llm.runtime_lanes import get_lane


StartFn = Callable[..., dict[str, Any]]
StopFn = Callable[..., dict[str, Any]]
WaitFn = Callable[[str, int], bool]
RunFn = Callable[..., dict[str, Any]]


def load_loop_prompts(*, prompt_args: list[str], prompt_file: Path | None) -> list[str]:
    prompts = [prompt for prompt in prompt_args if prompt.strip()]
    if prompt_file:
        text = prompt_file.read_text(encoding="utf-8")
        prompts.extend(_parse_prompt_file(text, prompt_file))
    if not prompts:
        raise ValueError("overnight loop requires at least one prompt or prompt file entry")
    return prompts


def run_overnight_loop(
    *,
    config: dict[str, Any],
    fast_lane_id: str,
    deep_lane_id: str,
    experiment_id: str,
    prompts: list[str],
    blackboard_path: Path,
    summary_path: Path,
    state_path: Path,
    logs_dir: Path,
    max_iterations: int,
    max_runtime_minutes: float,
    sleep_seconds: float,
    timeout_seconds: int = 120,
    deep_timeout_seconds: int = 600,
    fast_max_tokens: int = 256,
    deep_max_tokens: int = 512,
    driver_evidence: list[dict[str, Any]] | None = None,
    skip_start: bool = False,
    keep_running: bool = False,
    continue_on_failure: bool = False,
    start_fn: StartFn = start_lane,
    stop_fn: StopFn = stop_lane,
    wait_fn: WaitFn | None = None,
    run_fn: RunFn = run_lane_once,
) -> dict[str, Any]:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if max_runtime_minutes <= 0:
        raise ValueError("max_runtime_minutes must be greater than 0")
    if not prompts:
        raise ValueError("overnight loop requires at least one prompt")

    wait = wait_fn or wait_for_endpoint
    fast_lane = get_lane(config, fast_lane_id)
    deep_lane = get_lane(config, deep_lane_id)
    started_lane_ids: list[str] = []
    start_records: list[dict[str, Any]] = []
    stop_records: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    completed_iterations = 0
    stop_reason = "max_iterations"
    started_at = _utc_now()
    deadline = time.monotonic() + (max_runtime_minutes * 60)

    start_experiment(blackboard_path, experiment_id=experiment_id, title="Overnight handoff loop")

    try:
        if not skip_start:
            for lane in (fast_lane, deep_lane):
                start_records.append(start_fn(lane, state_path=state_path, logs_dir=logs_dir))
                started_lane_ids.append(lane["lane_id"])
                wait_timeout = deep_timeout_seconds if lane["lane_id"] == deep_lane_id else timeout_seconds
                wait(lane["endpoint"], wait_timeout)

        for index in range(1, max_iterations + 1):
            if time.monotonic() >= deadline:
                stop_reason = "time_budget"
                break
            prompt = prompts[(index - 1) % len(prompts)]
            iteration = _run_iteration(
                blackboard_path=blackboard_path,
                experiment_id=experiment_id,
                iteration_index=index,
                prompt=prompt,
                fast_lane=fast_lane,
                deep_lane=deep_lane,
                timeout_seconds=timeout_seconds,
                deep_timeout_seconds=deep_timeout_seconds,
                fast_max_tokens=fast_max_tokens,
                deep_max_tokens=deep_max_tokens,
                run_fn=run_fn,
            )
            iterations.append(iteration)
            fast_outcome = iteration["fast"]["outcome"]
            deep_outcome = (iteration.get("deep") or {}).get("outcome")
            if fast_outcome != "success":
                stop_reason = "fast_failure"
                if not continue_on_failure:
                    break
            elif deep_outcome != "success":
                stop_reason = "deep_failure"
                if not continue_on_failure:
                    break
            else:
                completed_iterations += 1
            if sleep_seconds and index < max_iterations:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stop_reason = "time_budget"
                    break
                time.sleep(min(sleep_seconds, remaining))
    finally:
        if not skip_start and not keep_running:
            for lane_id in reversed(started_lane_ids):
                stop_records.append(stop_fn(lane_id, state_path=state_path))

    summary = {
        "schema_version": "0.1",
        "experiment_id": experiment_id,
        "created_at": started_at,
        "completed_at": _utc_now(),
        "lanes": {"fast": fast_lane_id, "deep": deep_lane_id},
        "driver_evidence": driver_evidence or [],
        "blackboard_contract": BLACKBOARD_CONTRACT,
        "blackboard": str(blackboard_path),
        "prompt_count": len(prompts),
        "max_iterations": max_iterations,
        "max_runtime_minutes": max_runtime_minutes,
        "completed_iterations": completed_iterations,
        "attempted_iterations": len(iterations),
        "stop_reason": stop_reason,
        "iterations": iterations,
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


def _run_iteration(
    *,
    blackboard_path: Path,
    experiment_id: str,
    iteration_index: int,
    prompt: str,
    fast_lane: dict[str, Any],
    deep_lane: dict[str, Any],
    timeout_seconds: int,
    deep_timeout_seconds: int,
    fast_max_tokens: int,
    deep_max_tokens: int,
    run_fn: RunFn,
) -> dict[str, Any]:
    fast_task_id = f"loop-{iteration_index}-fast"
    deep_task_id = f"loop-{iteration_index}-deep"
    append_event(
        blackboard_path,
        experiment_id=experiment_id,
        event_type="task",
        payload={
            "task_id": fast_task_id,
            "prompt": prompt,
            "metadata": {
                "iteration_index": iteration_index,
                "lane_id": fast_lane["lane_id"],
                "blackboard_contract": BLACKBOARD_CONTRACT,
            },
        },
    )
    fast_prompt = build_fast_lane_prompt(
        original_prompt=prompt,
        fast_lane_id=fast_lane["lane_id"],
        deep_lane_id=deep_lane["lane_id"],
        iteration_label=f"Overnight loop iteration {iteration_index}",
        fast_backend=fast_lane.get("backend"),
        deep_backend=deep_lane.get("backend"),
    )
    fast_event = run_fn(
        blackboard_path,
        experiment_id=experiment_id,
        task_id=fast_task_id,
        prompt=fast_prompt,
        lane=fast_lane,
        timeout_seconds=timeout_seconds,
        max_tokens=fast_max_tokens,
    )
    fast_summary = _response_summary(fast_event)
    if fast_summary["outcome"] != "success":
        iteration = {"iteration_index": iteration_index, "prompt": prompt, "fast": fast_summary, "deep": None}
        _append_iteration_verdict(blackboard_path, experiment_id, iteration, completed=False)
        return iteration

    fast_content = fast_event["payload"].get("content") or fast_event["payload"].get("reasoning_content") or ""
    deep_prompt = build_deep_review_prompt(
        original_prompt=prompt,
        fast_lane_id=fast_lane["lane_id"],
        deep_lane_id=deep_lane["lane_id"],
        fast_content=fast_content,
        iteration_label=f"Overnight loop iteration {iteration_index}",
        fast_backend=fast_lane.get("backend"),
        deep_backend=deep_lane.get("backend"),
    )
    handoff_event = append_event(
        blackboard_path,
        experiment_id=experiment_id,
        event_type="handoff",
        payload={
            "iteration_index": iteration_index,
            "from_lane_id": fast_lane["lane_id"],
            "to_lane_id": deep_lane["lane_id"],
            "source_task_id": fast_task_id,
            "target_task_id": deep_task_id,
            "prompt": deep_prompt,
            "blackboard_contract": BLACKBOARD_CONTRACT,
        },
    )
    deep_event = run_fn(
        blackboard_path,
        experiment_id=experiment_id,
        task_id=deep_task_id,
        prompt=deep_prompt,
        lane=deep_lane,
        timeout_seconds=deep_timeout_seconds,
        max_tokens=deep_max_tokens,
    )
    deep_summary = _response_summary(deep_event)
    completed = deep_summary["outcome"] == "success"
    iteration = {
        "iteration_index": iteration_index,
        "prompt": prompt,
        "handoff_event_id": handoff_event["event_id"],
        "fast": fast_summary,
        "deep": deep_summary,
    }
    _append_iteration_verdict(blackboard_path, experiment_id, iteration, completed=completed)
    return iteration


def _append_iteration_verdict(
    blackboard_path: Path,
    experiment_id: str,
    iteration: dict[str, Any],
    *,
    completed: bool,
) -> None:
    append_event(
        blackboard_path,
        experiment_id=experiment_id,
        event_type="verdict",
        payload={
            "verdict_type": "loop_iteration",
            "iteration_index": iteration["iteration_index"],
            "completed": completed,
            "fast_outcome": iteration["fast"]["outcome"],
            "deep_outcome": (iteration.get("deep") or {}).get("outcome"),
        },
    )


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


def _parse_prompt_file(text: str, path: Path) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("[") or stripped.startswith("{"):
        payload = json.loads(stripped)
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
            return [str(item).strip() for item in payload["prompts"] if str(item).strip()]
        raise ValueError(f"unsupported prompt JSON shape: {path}")
    return [line.strip() for line in stripped.splitlines() if line.strip()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
