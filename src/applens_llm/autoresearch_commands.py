from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from applens_llm.workload_profile import bind_command_parameters


class CommandBlockedError(ValueError):
    pass


def execute_allowed_command(
    command: dict[str, Any] | None,
    *,
    parameters: dict[str, Any],
    workload_root: Path,
    timeout_seconds: int,
    command_id: str | None = None,
    allowed_commands: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = command
    if selected is None:
        if not command_id or not allowed_commands or command_id not in allowed_commands:
            raise CommandBlockedError(f"command is not allowlisted: {command_id or '<unknown>'}")
        selected = allowed_commands[command_id]
    rendered = bind_command_parameters(selected, parameters, workload_root=workload_root)
    argv = _split_command(rendered)
    if not argv:
        raise CommandBlockedError("empty command is blocked")
    _block_shell_operators(argv)
    cwd = _resolve_working_directory(workload_root, str(selected.get("working_directory", ".")))
    timeout = min(int(selected.get("timeout_seconds", timeout_seconds)), timeout_seconds)
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command_id": selected["id"],
            "outcome": "timeout",
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "duration_timeout_seconds": timeout,
        }
    return {
        "command_id": selected["id"],
        "outcome": "success" if completed.returncode == 0 else "failure",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _split_command(command: str) -> list[str]:
    if os.name == "nt":
        return [_strip_wrapping_quotes(part) for part in shlex.split(command, posix=False)]
    return shlex.split(command)


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _block_shell_operators(argv: list[str]) -> None:
    blocked = {"&&", "||", "|", ";", ">", ">>", "<", "`"}
    if any(arg in blocked for arg in argv):
        raise CommandBlockedError("shell operators are blocked in autoresearch commands")


def _resolve_working_directory(root: Path, working_directory: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root / working_directory).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise CommandBlockedError(f"working directory escapes workload root: {working_directory}")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate
