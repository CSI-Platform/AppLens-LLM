from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "applens_llm.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_cli_autoresearch_init_creates_layout(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"

    result = run_cli(
        "autoresearch",
        "init",
        "--workload-root",
        str(root),
        "--workload-id",
        "oracle",
        "--display-name",
        "Oracle",
    )

    assert result.returncode == 0
    assert "autoresearch layout" in result.stdout
    assert (root / ".applens" / "workload.json").exists()


def test_cli_autoresearch_eval_counts_probes_and_cases(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    init_result = run_cli(
        "autoresearch",
        "init",
        "--workload-root",
        str(root),
        "--workload-id",
        "oracle",
        "--display-name",
        "Oracle",
    )
    assert init_result.returncode == 0

    result = run_cli("autoresearch", "eval", "--workload-root", str(root))

    assert result.returncode == 0
    assert "probes=0 cases=0" in result.stdout


def test_cli_autoresearch_run_executes_allowlisted_fake_command(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    run_cli("autoresearch", "init", "--workload-root", str(root), "--workload-id", "oracle", "--display-name", "Oracle")
    commands = {
        "schema_version": "0.1",
        "commands": [
            {
                "id": "oracle.fake_backtest",
                "description": "Run a fake backtest.",
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
    (root / ".applens" / "commands.json").write_text(json.dumps(commands), encoding="utf-8")
    manifest = {
        "schema_version": "0.1",
        "run_id": "oracle-run-001",
        "workload_id": "oracle",
        "mode": "dry_run",
        "created_at": "2026-05-11T00:00:00Z",
        "workload_root": str(root).replace("\\", "/"),
        "blackboard_path": ".applens/blackboard/oracle-run-001.jsonl",
        "artifact_dir": ".applens/artifacts/oracle-run-001",
        "memory_proposal_dir": ".applens/memory/proposed",
        "self_fit": {"required": True, "skip": True, "freshness_hours": 24, "source": ".applens/runs/self-fit-latest.json"},
        "roles": [{"role": "supervisor", "provider": "manual-review", "lane_id": None}],
        "limits": {"max_iterations": 1, "max_runtime_minutes": 10, "command_timeout_seconds": 60},
        "stop_conditions": ["max_iterations", "command_failure", "blocked_action"],
        "next_step": {"command_id": "oracle.fake_backtest", "parameters": {}},
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }
    manifest_path = root / ".applens" / "runs" / "oracle-run-001.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_cli(
        "autoresearch",
        "run",
        "--workload-root",
        str(root),
        "--manifest",
        str(manifest_path),
        "--skip-self-fit",
    )

    assert result.returncode == 0
    assert "outcome=success" in result.stdout
    blackboard = root / ".applens" / "blackboard" / "oracle-run-001.jsonl"
    assert blackboard.exists()
    events = [json.loads(line) for line in blackboard.read_text(encoding="utf-8").splitlines()]
    assert any(event["event_type"] == "command_result" for event in events)


def test_cli_autoresearch_run_records_blocked_unknown_command(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    run_cli("autoresearch", "init", "--workload-root", str(root), "--workload-id", "oracle", "--display-name", "Oracle")
    manifest = {
        "schema_version": "0.1",
        "run_id": "oracle-run-blocked",
        "workload_id": "oracle",
        "mode": "dry_run",
        "created_at": "2026-05-11T00:00:00Z",
        "workload_root": str(root).replace("\\", "/"),
        "blackboard_path": ".applens/blackboard/oracle-run-blocked.jsonl",
        "artifact_dir": ".applens/artifacts/oracle-run-blocked",
        "memory_proposal_dir": ".applens/memory/proposed",
        "self_fit": {"required": True, "skip": True, "freshness_hours": 24, "source": ".applens/runs/self-fit-latest.json"},
        "roles": [{"role": "supervisor", "provider": "manual-review", "lane_id": None}],
        "limits": {"max_iterations": 1, "max_runtime_minutes": 10, "command_timeout_seconds": 60},
        "stop_conditions": ["max_iterations", "command_failure", "blocked_action"],
        "next_step": {"command_id": "oracle.live_trade", "parameters": {}},
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }
    manifest_path = root / ".applens" / "runs" / "oracle-run-blocked.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_cli(
        "autoresearch",
        "run",
        "--workload-root",
        str(root),
        "--manifest",
        str(manifest_path),
        "--skip-self-fit",
    )

    assert result.returncode == 0
    assert "outcome=blocked" in result.stdout
    events = [
        json.loads(line)
        for line in (root / ".applens" / "blackboard" / "oracle-run-blocked.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event_type"] == "command_blocked" for event in events)


def test_cli_autoresearch_run_blocks_declared_blocked_action(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    run_cli("autoresearch", "init", "--workload-root", str(root), "--workload-id", "oracle", "--display-name", "Oracle")
    commands = {
        "schema_version": "0.1",
        "commands": [
            {
                "id": "oracle.live_trade",
                "description": "Place a live trade.",
                "command_template": f"{sys.executable} -c \"print('should not run')\"",
                "working_directory": ".",
                "allowed_parameters": {},
                "required_inputs": [],
                "expected_outputs": [],
                "timeout_seconds": 30,
                "network_policy": "disabled",
                "risk_level": "low",
                "actions": ["live_trade"],
            }
        ],
    }
    (root / ".applens" / "commands.json").write_text(json.dumps(commands), encoding="utf-8")
    manifest = {
        "schema_version": "0.1",
        "run_id": "oracle-run-blocked-action",
        "workload_id": "oracle",
        "mode": "dry_run",
        "created_at": "2026-05-11T00:00:00Z",
        "workload_root": str(root).replace("\\", "/"),
        "blackboard_path": ".applens/blackboard/oracle-run-blocked-action.jsonl",
        "artifact_dir": ".applens/artifacts/oracle-run-blocked-action",
        "memory_proposal_dir": ".applens/memory/proposed",
        "self_fit": {"required": True, "skip": True, "freshness_hours": 24, "source": ".applens/runs/self-fit-latest.json"},
        "roles": [{"role": "supervisor", "provider": "manual-review", "lane_id": None}],
        "limits": {"max_iterations": 1, "max_runtime_minutes": 10, "command_timeout_seconds": 60},
        "stop_conditions": ["max_iterations", "command_failure", "blocked_action"],
        "next_step": {"command_id": "oracle.live_trade", "parameters": {}},
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }
    manifest_path = root / ".applens" / "runs" / "oracle-run-blocked-action.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_cli(
        "autoresearch",
        "run",
        "--workload-root",
        str(root),
        "--manifest",
        str(manifest_path),
        "--skip-self-fit",
    )

    assert result.returncode == 0
    assert "outcome=blocked" in result.stdout
    events = [
        json.loads(line)
        for line in (root / ".applens" / "blackboard" / "oracle-run-blocked-action.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(event["event_type"] == "command_blocked" for event in events)
    assert not any("should not run" in event["payload"].get("stdout", "") for event in events)


def test_cli_autoresearch_self_fit_writes_latest(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"

    result = run_cli(
        "autoresearch",
        "self-fit",
        "--workload-root",
        str(root),
        "--machine-fingerprint",
        "machine-a",
        "--runtime-fingerprint",
        "runtime-a",
    )

    assert result.returncode == 0
    assert "self-fit" in result.stdout
    assert (root / ".applens" / "runs" / "self-fit-latest.json").exists()
