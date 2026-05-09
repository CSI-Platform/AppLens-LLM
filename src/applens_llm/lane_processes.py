from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse


class PopenFactory(Protocol):
    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        ...


def build_server_command(lane: dict[str, Any]) -> list[str]:
    launch = lane["launch"]
    host, port = endpoint_host_port(lane["endpoint"])
    command = [
        launch["server_binary"],
        "-m",
        lane["model"]["path"],
        "--host",
        host,
        "--port",
        str(port),
    ]
    if lane["backend"] != "cpu" and lane["device"]["selector"].lower() != "cpu":
        command.extend(["-dev", lane["device"]["selector"]])
    command.extend(
        [
            "-ngl",
            str(launch["gpu_layers"]),
            "-c",
            str(launch["context_tokens"]),
            "-t",
            str(launch["threads"]),
            "--alias",
            lane["model"]["label"],
        ]
    )
    return command


def endpoint_host_port(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port:
        return host, parsed.port
    return host, 443 if parsed.scheme == "https" else 80


def start_lane(
    lane: dict[str, Any],
    *,
    state_path: Path,
    logs_dir: Path,
    popen_factory: PopenFactory = subprocess.Popen,
) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    command = build_server_command(lane)
    stdout_path = logs_dir / f"{lane['lane_id']}.out.log"
    stderr_path = logs_dir / f"{lane['lane_id']}.err.log"
    env = os.environ.copy()
    env.update(lane["launch"].get("environment", {}))

    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = popen_factory(
            command,
            cwd=_working_directory(command[0]),
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=_creation_flags(),
            start_new_session=platform.system() != "Windows",
        )

    record = {
        "lane_id": lane["lane_id"],
        "pid": process.pid,
        "status": "running",
        "endpoint": lane["endpoint"],
        "engine": lane["engine"],
        "backend": lane["backend"],
        "device_selector": lane["device"]["selector"],
        "accelerator_ids": lane["device"]["accelerator_ids"],
        "model_label": lane["model"]["label"],
        "command": command,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "started_at": _utc_now(),
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }
    state = read_process_state(state_path)
    state["processes"][lane["lane_id"]] = record
    write_process_state(state_path, state)
    return record


def stop_lane(lane_id: str, *, state_path: Path, timeout_seconds: int = 15) -> dict[str, Any]:
    state = read_process_state(state_path)
    record = state["processes"].get(lane_id)
    if not record:
        return {"lane_id": lane_id, "status": "not_found"}

    pid = int(record["pid"])
    if is_process_running(pid):
        _terminate_process(pid)
        deadline = time.monotonic() + timeout_seconds
        while is_process_running(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if is_process_running(pid):
            _kill_process(pid)

    record["status"] = "stopped"
    record["stopped_at"] = _utc_now()
    state["processes"][lane_id] = record
    write_process_state(state_path, state)
    return record


def read_process_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "0.1", "processes": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", "0.1")
    payload.setdefault("processes", {})
    return payload


def write_process_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_process_running(pid: int) -> bool:
    if platform.system() == "Windows":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            text=True,
            capture_output=True,
            check=False,
        )
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_process(pid: int) -> None:
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.kill(pid, signal.SIGTERM)


def _kill_process(pid: int) -> None:
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.kill(pid, signal.SIGKILL)


def _working_directory(binary: str) -> str | None:
    parent = Path(binary).parent
    if str(parent) in ("", "."):
        return None
    return str(parent)


def _creation_flags() -> int:
    if platform.system() != "Windows":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
