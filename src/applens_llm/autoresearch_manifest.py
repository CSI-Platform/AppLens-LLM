from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.autoresearch_layout import workload_paths
from applens_llm.schemas import validate_payload


def build_run_manifest(
    *,
    workload_root: Path,
    workload_id: str,
    run_id: str,
    mode: str = "workload",
    skip_self_fit: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "0.1",
        "run_id": run_id,
        "workload_id": workload_id,
        "mode": mode,
        "created_at": created_at or _utc_now(),
        "workload_root": str(workload_root).replace("\\", "/"),
        "blackboard_path": f".applens/blackboard/{run_id}.jsonl",
        "artifact_dir": f".applens/artifacts/{run_id}",
        "memory_proposal_dir": ".applens/memory/proposed",
        "self_fit": {
            "required": True,
            "skip": skip_self_fit,
            "freshness_hours": 24,
            "source": ".applens/runs/self-fit-latest.json",
        },
        "roles": [{"role": "supervisor", "provider": "manual-review", "lane_id": None}],
        "limits": {"max_iterations": 1, "max_runtime_minutes": 10, "command_timeout_seconds": 60},
        "stop_conditions": ["max_iterations", "command_failure", "blocked_action"],
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }
    return validate_payload("autoresearch-run-manifest", manifest)


def load_run_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_payload("autoresearch-run-manifest", payload)


def write_self_fit_result(
    root: Path,
    *,
    machine_fingerprint: str,
    runtime_fingerprint: str,
    status: str = "passed",
    created_at: str | None = None,
) -> dict[str, Any]:
    paths = workload_paths(root)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "0.1",
        "mode": "self_fit",
        "status": status,
        "created_at": created_at or _utc_now(),
        "machine_fingerprint": machine_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
    }
    (paths.runs_dir / "self-fit-latest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def self_fit_is_fresh(
    root: Path,
    *,
    machine_fingerprint: str,
    runtime_fingerprint: str,
    now: str | None = None,
    freshness_hours: int = 24,
) -> bool:
    source = workload_paths(root).runs_dir / "self-fit-latest.json"
    if not source.exists():
        return False
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        return False
    if payload.get("machine_fingerprint") != machine_fingerprint:
        return False
    if payload.get("runtime_fingerprint") != runtime_fingerprint:
        return False
    created_at = _parse_utc(payload.get("created_at", ""))
    checked_at = _parse_utc(now or _utc_now())
    age_hours = (checked_at - created_at).total_seconds() / 3600
    return age_hours <= freshness_hours


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

