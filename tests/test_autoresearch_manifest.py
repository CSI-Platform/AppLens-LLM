from __future__ import annotations

import json
from pathlib import Path

import pytest

from applens_llm.autoresearch_manifest import (
    build_run_manifest,
    load_run_manifest,
    self_fit_is_fresh,
    write_self_fit_result,
)
from applens_llm.schemas import SchemaValidationError, validate_payload


def _manifest() -> dict:
    return {
        "schema_version": "0.1",
        "run_id": "oracle-run-001",
        "workload_id": "oracle",
        "mode": "workload",
        "created_at": "2026-05-11T00:00:00Z",
        "workload_root": "C:/work/Oracle",
        "blackboard_path": ".applens/blackboard/oracle-run-001.jsonl",
        "artifact_dir": ".applens/artifacts/oracle-run-001",
        "memory_proposal_dir": ".applens/memory/proposed",
        "self_fit": {
            "required": True,
            "skip": False,
            "freshness_hours": 24,
            "source": ".applens/runs/self-fit-latest.json",
        },
        "roles": [
            {"role": "supervisor", "provider": "manual-review", "lane_id": None},
            {"role": "candidate", "provider": "local-lane", "lane_id": "fast-nvidia"},
        ],
        "limits": {
            "max_iterations": 1,
            "max_runtime_minutes": 10,
            "command_timeout_seconds": 60,
        },
        "stop_conditions": ["max_iterations", "command_failure", "blocked_action"],
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }


def _artifact() -> dict:
    return {
        "schema_version": "0.1",
        "artifact_id": "artifact-backtest-001",
        "workload_id": "oracle",
        "run_id": "oracle-run-001",
        "artifact_type": "backtest_result",
        "path": ".applens/artifacts/oracle-run-001/backtest.json",
        "schema": "oracle-backtest-result",
        "created_by": "workload_executor",
        "summary": "Fake backtest result.",
        "hash": "sha256:abc123",
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }


def test_autoresearch_run_manifest_schema_accepts_v1_manifest() -> None:
    assert validate_payload("autoresearch-run-manifest", _manifest())["run_id"] == "oracle-run-001"


def test_autoresearch_run_manifest_rejects_live_unbounded_loop() -> None:
    payload = _manifest()
    payload["limits"]["max_iterations"] = 0

    with pytest.raises(SchemaValidationError):
        validate_payload("autoresearch-run-manifest", payload)


def test_workload_artifact_schema_accepts_artifact_reference() -> None:
    assert validate_payload("workload-artifact", _artifact())["artifact_type"] == "backtest_result"


def test_build_and_load_run_manifest(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    manifest = build_run_manifest(
        workload_root=root,
        workload_id="oracle",
        run_id="oracle-run-001",
        mode="dry_run",
        skip_self_fit=True,
        created_at="2026-05-11T00:00:00Z",
    )
    path = root / ".applens" / "runs" / "oracle-run-001.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_run_manifest(path)

    assert loaded["mode"] == "dry_run"
    assert loaded["self_fit"]["skip"] is True


def test_self_fit_freshness_uses_machine_and_runtime_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    result = write_self_fit_result(
        root,
        machine_fingerprint="machine-a",
        runtime_fingerprint="runtime-a",
        status="passed",
        created_at="2026-05-11T00:00:00Z",
    )

    assert result["status"] == "passed"
    assert self_fit_is_fresh(
        root,
        machine_fingerprint="machine-a",
        runtime_fingerprint="runtime-a",
        now="2026-05-11T12:00:00Z",
        freshness_hours=24,
    )
    assert not self_fit_is_fresh(
        root,
        machine_fingerprint="machine-b",
        runtime_fingerprint="runtime-a",
        now="2026-05-11T12:00:00Z",
        freshness_hours=24,
    )


def test_autoresearch_manifest_can_include_self_fit_fingerprints() -> None:
    payload = _manifest()
    payload["self_fit"]["machine_fingerprint"] = "machine-a"
    payload["self_fit"]["runtime_fingerprint"] = "runtime-a"

    assert validate_payload("autoresearch-run-manifest", payload)["self_fit"]["machine_fingerprint"] == "machine-a"
