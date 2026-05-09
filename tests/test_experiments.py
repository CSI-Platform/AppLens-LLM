from __future__ import annotations

import json
import socket
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from applens_llm.blackboard import append_event, read_events
from applens_llm.experiments import run_two_lane_experiment, wait_for_endpoint


def test_run_two_lane_experiment_routes_fast_output_to_deep_lane(tmp_path: Path) -> None:
    blackboard = tmp_path / "experiment.jsonl"
    summary_path = tmp_path / "summary.json"
    calls: list[tuple[str, str]] = []

    def fake_start(lane: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(("start", lane["lane_id"]))
        return {"lane_id": lane["lane_id"], "pid": 100, "status": "running"}

    def fake_stop(lane_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(("stop", lane_id))
        return {"lane_id": lane_id, "status": "stopped"}

    def fake_wait(endpoint: str, timeout_seconds: int) -> bool:
        calls.append(("wait", endpoint))
        return True

    def fake_run(
        path: Path,
        *,
        experiment_id: str,
        task_id: str,
        prompt: str,
        lane: dict[str, Any],
        timeout_seconds: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        calls.append(("run", lane["lane_id"]))
        content = "fast answer" if lane["lane_id"] == "fast-nvidia" else f"deep reviewed: {prompt}"
        return append_event(
            path,
            experiment_id=experiment_id,
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
                "usage": {"total_tokens": 10},
            },
        )

    summary = run_two_lane_experiment(
        config=_config(),
        fast_lane_id="fast-nvidia",
        deep_lane_id="deep-amd-vgm",
        experiment_id="exp-test",
        prompt="explain lanes",
        blackboard_path=blackboard,
        summary_path=summary_path,
        state_path=tmp_path / "state.json",
        logs_dir=tmp_path / "logs",
        start_fn=fake_start,
        stop_fn=fake_stop,
        wait_fn=fake_wait,
        run_fn=fake_run,
        driver_evidence=[
            {
                "vendor": "nvidia",
                "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                "driver_version": "576.88",
                "driver_branch": "game_ready",
                "benchmark_invalidates_on_change": True,
            }
        ],
    )

    events = read_events(blackboard)
    deep_event = [event for event in events if event["payload"].get("lane_id") == "deep-amd-vgm"][0]
    assert ("start", "fast-nvidia") in calls
    assert ("start", "deep-amd-vgm") in calls
    assert ("stop", "deep-amd-vgm") in calls
    assert ("stop", "fast-nvidia") in calls
    assert "fast answer" in deep_event["payload"]["content"]
    assert summary["responses"]["fast"]["outcome"] == "success"
    assert summary["responses"]["deep"]["outcome"] == "success"
    assert summary["driver_evidence"][0]["driver_branch"] == "game_ready"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["experiment_id"] == "exp-test"


def test_run_two_lane_experiment_skip_start_does_not_start_or_stop(tmp_path: Path) -> None:
    calls: list[str] = []

    def fail_start(lane: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(f"start:{lane['lane_id']}")
        return {}

    def fail_stop(lane_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(f"stop:{lane_id}")
        return {}

    def fake_run(path: Path, **kwargs: Any) -> dict[str, Any]:
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

    summary = run_two_lane_experiment(
        config=_config(),
        fast_lane_id="fast-nvidia",
        deep_lane_id="deep-amd-vgm",
        experiment_id="exp-test",
        prompt="hello",
        blackboard_path=tmp_path / "experiment.jsonl",
        summary_path=tmp_path / "summary.json",
        state_path=tmp_path / "state.json",
        logs_dir=tmp_path / "logs",
        skip_start=True,
        start_fn=fail_start,
        stop_fn=fail_stop,
        wait_fn=lambda endpoint, timeout_seconds: True,
        run_fn=fake_run,
    )

    assert calls == []
    assert summary["responses"]["fast"]["outcome"] == "connection_error"


def test_wait_for_endpoint_retries_until_port_listens() -> None:
    reserved = socket.socket()
    reserved.bind(("127.0.0.1", 0))
    host, port = reserved.getsockname()
    reserved.close()

    def delayed_server() -> None:
        time.sleep(1.0)
        server = HTTPServer((host, port), ReadyHealthHandler)
        server.handle_request()
        server.server_close()

    thread = Thread(target=delayed_server, daemon=True)
    thread.start()

    assert wait_for_endpoint(f"http://{host}:{port}/v1", timeout_seconds=3)


def test_wait_for_endpoint_raises_timeout_for_closed_port() -> None:
    reserved = socket.socket()
    reserved.bind(("127.0.0.1", 0))
    host, port = reserved.getsockname()
    reserved.close()

    with pytest.raises(TimeoutError):
        wait_for_endpoint(f"http://{host}:{port}/v1", timeout_seconds=1)


def test_wait_for_endpoint_requires_http_health_ok() -> None:
    HealthHandler.calls = 0
    server = HTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    try:
        assert wait_for_endpoint(f"http://{host}:{port}/v1", timeout_seconds=5)
    finally:
        server.shutdown()

    assert HealthHandler.calls >= 3


class HealthHandler(BaseHTTPRequestHandler):
    calls = 0

    def do_GET(self) -> None:
        type(self).calls += 1
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        if type(self).calls < 3:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"status":"loading"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, format: str, *args: object) -> None:
        return


class ReadyHealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


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
