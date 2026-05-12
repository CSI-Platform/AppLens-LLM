from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from applens_llm.autoresearch_commands import CommandBlockedError, execute_allowed_command
from applens_llm.workload_profile import bind_command_parameters, enforce_command_policy, load_allowed_commands


def _write_commands(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "commands": [
                    {
                        "id": "oracle.fake_backtest",
                        "description": "Run safe fake backtest.",
                        "command_template": f"{sys.executable} -c \"print('fake oracle backtest ok')\"",
                        "working_directory": ".",
                        "allowed_parameters": {},
                        "required_inputs": [],
                        "expected_outputs": [],
                        "timeout_seconds": 30,
                        "network_policy": "disabled",
                        "risk_level": "low",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_allowed_commands_and_bind_parameters(tmp_path: Path) -> None:
    commands_path = tmp_path / "commands.json"
    _write_commands(commands_path)

    commands = load_allowed_commands(commands_path)
    bound = bind_command_parameters(commands["oracle.fake_backtest"], {})

    assert "fake oracle backtest ok" in bound


def test_execute_allowed_command_runs_safe_command(tmp_path: Path) -> None:
    commands_path = tmp_path / "commands.json"
    _write_commands(commands_path)
    commands = load_allowed_commands(commands_path)

    result = execute_allowed_command(
        commands["oracle.fake_backtest"],
        parameters={},
        workload_root=tmp_path,
        timeout_seconds=5,
    )

    assert result["outcome"] == "success"
    assert "fake oracle backtest ok" in result["stdout"]
    assert result["returncode"] == 0


def test_execute_allowed_command_blocks_unknown_command(tmp_path: Path) -> None:
    commands_path = tmp_path / "commands.json"
    _write_commands(commands_path)
    commands = load_allowed_commands(commands_path)

    with pytest.raises(CommandBlockedError):
        execute_allowed_command(
            None,
            parameters={},
            workload_root=tmp_path,
            timeout_seconds=5,
            command_id="oracle.live_trade",
            allowed_commands=commands,
        )


def test_enforce_command_policy_blocks_profile_blocked_action() -> None:
    command = {
        "id": "oracle.live_trade",
        "description": "Place a live trade.",
        "command_template": "python -c \"print('trade')\"",
        "working_directory": ".",
        "allowed_parameters": {},
        "required_inputs": [],
        "expected_outputs": [],
        "timeout_seconds": 30,
        "network_policy": "disabled",
        "risk_level": "low",
        "actions": ["live_trade"],
    }

    with pytest.raises(ValueError):
        enforce_command_policy(command, allowed_actions=["run_backtests"], blocked_actions=["live_trade"])


def test_bind_command_parameters_blocks_path_escape(tmp_path: Path) -> None:
    command = {
        "id": "oracle.write_result",
        "description": "Write a result file.",
        "command_template": "python script.py --output {output}",
        "allowed_parameters": {"output": {"type": "path"}},
    }

    with pytest.raises(ValueError):
        bind_command_parameters(command, {"output": "../outside.txt"}, workload_root=tmp_path)

    bound = bind_command_parameters(command, {"output": "artifacts/result.json"}, workload_root=tmp_path)
    assert str((tmp_path / "artifacts" / "result.json").resolve()).replace("\\", "/") in bound
