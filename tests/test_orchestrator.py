from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

from applens_llm.blackboard import read_events, start_experiment
from applens_llm.orchestrator import run_lane_once


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        response = {
            "choices": [
                {
                    "message": {
                        "content": f"lane answered: {body['messages'][-1]['content']}; max={body.get('max_tokens')}"
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def _server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), ChatHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/v1"


class ReasoningOnlyHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        response = {
            "choices": [{"message": {"content": "", "reasoning_content": "reasoning-only answer"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def _reasoning_server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), ReasoningOnlyHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/v1"


def test_run_lane_once_records_success(tmp_path: Path) -> None:
    server, endpoint = _server()
    try:
        blackboard = tmp_path / "run.jsonl"
        start_experiment(blackboard, experiment_id="exp-test", title="Unit")
        lane = {
            "lane_id": "fast-nvidia",
            "role": "fast",
            "engine": "llama.cpp",
            "backend": "cuda",
            "endpoint": endpoint,
            "model": {"label": "jan-v35-4b", "path": "models/jan.gguf"},
            "device": {"selector": "cuda:0", "accelerator_ids": ["nvidia-dgpu-0"]},
            "launch": {
                "server_binary": "llama-server",
                "context_tokens": 4096,
                "gpu_layers": 99,
                "threads": 12,
                "environment": {},
            },
        }

        event = run_lane_once(
            blackboard,
            experiment_id="exp-test",
            task_id="task-1",
            prompt="hello",
            lane=lane,
            timeout_seconds=5,
            max_tokens=42,
        )

        events = read_events(blackboard)
        assert event["payload"]["outcome"] == "success"
        assert event["payload"]["content"] == "lane answered: hello; max=42"
        assert event["payload"]["backend"] == "cuda"
        assert event["payload"]["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        assert events[-1]["event_type"] == "model_response"
    finally:
        server.shutdown()


def test_run_lane_once_preserves_reasoning_content(tmp_path: Path) -> None:
    server, endpoint = _reasoning_server()
    try:
        blackboard = tmp_path / "run.jsonl"
        start_experiment(blackboard, experiment_id="exp-test", title="Unit")
        lane = {
            "lane_id": "deep-amd-vgm",
            "role": "deep",
            "engine": "llama.cpp",
            "backend": "vulkan",
            "endpoint": endpoint,
            "model": {"label": "qwen-27b", "path": "models/qwen.gguf"},
            "device": {"selector": "Vulkan0", "accelerator_ids": ["amd-igpu-0"]},
            "launch": {
                "server_binary": "llama-server",
                "context_tokens": 4096,
                "gpu_layers": 99,
                "threads": 12,
                "environment": {},
            },
        }

        event = run_lane_once(
            blackboard,
            experiment_id="exp-test",
            task_id="task-1",
            prompt="hello",
            lane=lane,
            timeout_seconds=5,
        )

        assert event["payload"]["content"] == ""
        assert event["payload"]["reasoning_content"] == "reasoning-only answer"
    finally:
        server.shutdown()


def test_run_lane_once_records_connection_failure(tmp_path: Path) -> None:
    blackboard = tmp_path / "run.jsonl"
    start_experiment(blackboard, experiment_id="exp-test", title="Unit")
    lane = {
        "lane_id": "deep-amd-vgm",
        "role": "deep",
        "engine": "llama.cpp",
        "backend": "vulkan",
        "endpoint": "http://127.0.0.1:9/v1",
        "model": {"label": "qwen-27b", "path": "models/qwen.gguf"},
        "device": {"selector": "Vulkan0", "accelerator_ids": ["amd-igpu-0"]},
        "launch": {
            "server_binary": "llama-server",
            "context_tokens": 4096,
            "gpu_layers": 99,
            "threads": 12,
            "environment": {},
        },
    }

    event = run_lane_once(
        blackboard,
        experiment_id="exp-test",
        task_id="task-1",
        prompt="hello",
        lane=lane,
        timeout_seconds=1,
    )

    assert event["event_type"] == "failure"
    assert event["payload"]["outcome"] == "connection_error"
