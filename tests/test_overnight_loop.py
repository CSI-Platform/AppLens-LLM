from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from applens_llm.blackboard import append_event, read_events
from applens_llm.handoff_contracts import BLACKBOARD_CONTRACT
from applens_llm.overnight_loop import load_loop_prompts, run_overnight_loop


def test_run_overnight_loop_routes_multiple_fast_to_deep_handoffs(tmp_path: Path) -> None:
    blackboard = tmp_path / "overnight.jsonl"
    summary_path = tmp_path / "overnight-summary.json"
    calls: list[tuple[str, str, str]] = []

    def fake_run(path: Path, **kwargs: Any) -> dict[str, Any]:
        lane = kwargs["lane"]
        task_id = kwargs["task_id"]
        prompt = kwargs["prompt"]
        calls.append((task_id, lane["lane_id"], prompt))
        content = f"{lane['lane_id']} answered {task_id}"
        return append_event(
            path,
            experiment_id=kwargs["experiment_id"],
            event_type="model_response",
            payload={
                "task_id": task_id,
                "lane_id": lane["lane_id"],
                "model_label": lane["model"]["label"],
                "backend": lane["backend"],
                "accelerator_ids": lane["device"]["accelerator_ids"],
                "latency_ms": 100,
                "outcome": "success",
                "content": content,
                "reasoning_content": "",
                "usage": {"total_tokens": 12},
            },
        )

    summary = run_overnight_loop(
        config=_config(),
        fast_lane_id="fast-nvidia",
        deep_lane_id="deep-amd-vgm",
        experiment_id="overnight-test",
        prompts=["first task", "second task"],
        blackboard_path=blackboard,
        summary_path=summary_path,
        state_path=tmp_path / "state.json",
        logs_dir=tmp_path / "logs",
        max_iterations=3,
        max_runtime_minutes=60,
        sleep_seconds=0,
        skip_start=True,
        run_fn=fake_run,
    )

    events = read_events(blackboard)
    handoffs = [event for event in events if event["event_type"] == "handoff"]
    verdicts = [event for event in events if event["event_type"] == "verdict"]

    assert summary["stop_reason"] == "max_iterations"
    assert summary["completed_iterations"] == 3
    assert len(handoffs) == 3
    assert len(verdicts) == 3
    assert calls[0][0] == "loop-1-fast"
    assert calls[1][0] == "loop-1-deep"
    assert BLACKBOARD_CONTRACT not in calls[0][2]
    assert "Do not imply pooled VRAM or shared GPU memory" in calls[0][2]
    assert "fast-nvidia backend=cuda" in calls[0][2]
    assert "deep-amd-vgm backend=vulkan" in calls[0][2]
    assert "fast-nvidia answered loop-1-fast" in handoffs[0]["payload"]["prompt"]
    assert BLACKBOARD_CONTRACT not in handoffs[0]["payload"]["prompt"]
    assert handoffs[0]["payload"]["blackboard_contract"] == BLACKBOARD_CONTRACT
    assert "Do not penalize the fast lane for omitting blackboard details" in handoffs[0]["payload"]["prompt"]
    assert "first task" in calls[4][2]
    assert json.loads(summary_path.read_text(encoding="utf-8"))["completed_iterations"] == 3


def test_run_overnight_loop_stops_on_fast_failure_by_default(tmp_path: Path) -> None:
    def failing_run(path: Path, **kwargs: Any) -> dict[str, Any]:
        lane = kwargs["lane"]
        return append_event(
            path,
            experiment_id=kwargs["experiment_id"],
            event_type="failure",
            payload={
                "task_id": kwargs["task_id"],
                "lane_id": lane["lane_id"],
                "model_label": lane["model"]["label"],
                "backend": lane["backend"],
                "accelerator_ids": lane["device"]["accelerator_ids"],
                "latency_ms": 1,
                "outcome": "connection_error",
                "error": "offline",
            },
        )

    summary = run_overnight_loop(
        config=_config(),
        fast_lane_id="fast-nvidia",
        deep_lane_id="deep-amd-vgm",
        experiment_id="overnight-failure",
        prompts=["first task"],
        blackboard_path=tmp_path / "overnight.jsonl",
        summary_path=tmp_path / "summary.json",
        state_path=tmp_path / "state.json",
        logs_dir=tmp_path / "logs",
        max_iterations=3,
        max_runtime_minutes=60,
        sleep_seconds=0,
        skip_start=True,
        run_fn=failing_run,
    )

    assert summary["stop_reason"] == "fast_failure"
    assert summary["completed_iterations"] == 0
    assert summary["iterations"][0]["fast"]["outcome"] == "connection_error"


def test_run_overnight_loop_uses_deep_timeout_while_waiting_for_deep_lane(tmp_path: Path) -> None:
    waits: list[tuple[str, int]] = []

    def fake_start(lane: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {"lane_id": lane["lane_id"], "pid": 100, "status": "running"}

    def fake_stop(lane_id: str, **kwargs: Any) -> dict[str, Any]:
        return {"lane_id": lane_id, "status": "stopped"}

    def fake_wait(endpoint: str, timeout_seconds: int) -> bool:
        waits.append((endpoint, timeout_seconds))
        return True

    def fake_run(path: Path, **kwargs: Any) -> dict[str, Any]:
        lane = kwargs["lane"]
        return append_event(
            path,
            experiment_id=kwargs["experiment_id"],
            event_type="model_response",
            payload={
                "task_id": kwargs["task_id"],
                "lane_id": lane["lane_id"],
                "model_label": lane["model"]["label"],
                "backend": lane["backend"],
                "accelerator_ids": lane["device"]["accelerator_ids"],
                "latency_ms": 100,
                "outcome": "success",
                "content": "ok",
                "reasoning_content": "",
                "usage": {"total_tokens": 1},
            },
        )

    run_overnight_loop(
        config=_config(),
        fast_lane_id="fast-nvidia",
        deep_lane_id="deep-amd-vgm",
        experiment_id="overnight-wait",
        prompts=["first task"],
        blackboard_path=tmp_path / "overnight.jsonl",
        summary_path=tmp_path / "summary.json",
        state_path=tmp_path / "state.json",
        logs_dir=tmp_path / "logs",
        max_iterations=1,
        max_runtime_minutes=60,
        sleep_seconds=0,
        timeout_seconds=12,
        deep_timeout_seconds=90,
        start_fn=fake_start,
        stop_fn=fake_stop,
        wait_fn=fake_wait,
        run_fn=fake_run,
    )

    assert waits == [("http://127.0.0.1:18081/v1", 12), ("http://127.0.0.1:18082/v1", 90)]


def test_load_loop_prompts_accepts_json_and_text(tmp_path: Path) -> None:
    json_path = tmp_path / "prompts.json"
    text_path = tmp_path / "prompts.txt"
    json_path.write_text('["json one", "json two"]', encoding="utf-8")
    text_path.write_text("text one\n\ntext two\n", encoding="utf-8")

    assert load_loop_prompts(prompt_args=["inline"], prompt_file=None) == ["inline"]
    assert load_loop_prompts(prompt_args=[], prompt_file=json_path) == ["json one", "json two"]
    assert load_loop_prompts(prompt_args=[], prompt_file=text_path) == ["text one", "text two"]
    assert load_loop_prompts(prompt_args=["inline"], prompt_file=text_path) == ["inline", "text one", "text two"]


def _config() -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "lanes": [
            {
                "lane_id": "fast-nvidia",
                "role": "fast",
                "engine": "llama.cpp",
                "backend": "cuda",
                "endpoint": "http://127.0.0.1:18081/v1",
                "model": {"label": "jan-v35-4b", "path": "models/jan.gguf"},
                "device": {"selector": "CUDA0", "accelerator_ids": ["nvidia-dgpu-0"]},
                "launch": {
                    "server_binary": "llama-server",
                    "context_tokens": 2048,
                    "gpu_layers": 99,
                    "threads": 12,
                    "environment": {},
                },
            },
            {
                "lane_id": "deep-amd-vgm",
                "role": "deep",
                "engine": "llama.cpp",
                "backend": "vulkan",
                "endpoint": "http://127.0.0.1:18082/v1",
                "model": {"label": "qwen-27b", "path": "models/qwen.gguf"},
                "device": {"selector": "Vulkan0", "accelerator_ids": ["amd-igpu-0"]},
                "launch": {
                    "server_binary": "llama-server",
                    "context_tokens": 2048,
                    "gpu_layers": 99,
                    "threads": 12,
                    "environment": {},
                },
            },
        ],
    }
