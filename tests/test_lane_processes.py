from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from applens_llm.lane_processes import build_server_command, start_lane, stop_lane


def test_build_server_command_uses_lane_launch_fields() -> None:
    lane = _lane()

    command = build_server_command(lane)

    assert command == [
        "llama-server",
        "-m",
        "models/jan.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "18081",
        "-dev",
        "CUDA0",
        "-ngl",
        "99",
        "-c",
        "2048",
        "-t",
        "12",
        "--alias",
        "jan-v35-4b",
    ]


def test_build_server_command_skips_device_for_cpu_lane() -> None:
    lane = _lane()
    lane["backend"] = "cpu"
    lane["device"]["selector"] = "cpu"
    lane["launch"]["gpu_layers"] = 0

    command = build_server_command(lane)

    assert "-dev" not in command
    assert command[command.index("-ngl") + 1] == "0"


def test_start_lane_writes_process_state_with_injected_popen(tmp_path: Path) -> None:
    state_path = tmp_path / "lane-processes.json"
    logs_dir = tmp_path / "logs"
    lane = _lane()

    record = start_lane(
        lane,
        state_path=state_path,
        logs_dir=logs_dir,
        popen_factory=FakePopen,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert record["pid"] == 4242
    assert record["status"] == "running"
    assert state["processes"]["fast-nvidia"]["pid"] == 4242
    assert state["processes"]["fast-nvidia"]["endpoint"] == "http://127.0.0.1:18081/v1"
    assert Path(state["processes"]["fast-nvidia"]["stdout_log"]).name == "fast-nvidia.out.log"
    assert logs_dir.exists()


def test_stop_lane_terminates_running_process_from_state(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    state_path = tmp_path / "lane-processes.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "processes": {
                    "fast-nvidia": {
                        "lane_id": "fast-nvidia",
                        "pid": process.pid,
                        "status": "running",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        record = stop_lane("fast-nvidia", state_path=state_path, timeout_seconds=10)
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert record["status"] == "stopped"
    assert state["processes"]["fast-nvidia"]["status"] == "stopped"


class FakePopen:
    def __init__(self, command: list[str], **kwargs: object) -> None:
        self.command = command
        self.kwargs = kwargs
        self.pid = 4242


def _lane() -> dict[str, object]:
    return {
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
            "environment": {"GGML_TEST_FLAG": "1"},
        },
    }
