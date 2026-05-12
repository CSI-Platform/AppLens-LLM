from __future__ import annotations

import json
from pathlib import Path

import pytest

from applens_llm.autoresearch_blackboard import append_workload_event, read_workload_events
from applens_llm.schemas import SchemaValidationError, validate_payload


def test_blackboard_schema_accepts_workload_event_without_legacy_experiment_id() -> None:
    event = {
        "schema_version": "0.1",
        "event_id": "evt-workload-1",
        "workload_id": "oracle",
        "run_id": "oracle-run-001",
        "actor": {"role": "candidate", "lane_id": "fast-nvidia", "provider": "local-lane"},
        "event_type": "proposal",
        "created_at": "2026-05-11T00:00:00Z",
        "payload": {"summary": "Try one fake backtest."},
        "artifacts": [],
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }

    assert validate_payload("blackboard-record", event)["workload_id"] == "oracle"


def test_blackboard_schema_requires_actor_for_workload_event() -> None:
    event = {
        "schema_version": "0.1",
        "event_id": "evt-workload-1",
        "workload_id": "oracle",
        "run_id": "oracle-run-001",
        "event_type": "proposal",
        "created_at": "2026-05-11T00:00:00Z",
        "payload": {"summary": "Try one fake backtest."},
        "artifacts": [],
        "privacy": {"commit_safe": False, "local_paths_included": True},
    }

    with pytest.raises(SchemaValidationError):
        validate_payload("blackboard-record", event)


def test_append_workload_event_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "oracle-run.jsonl"
    event = append_workload_event(
        path,
        workload_id="oracle",
        run_id="oracle-run-001",
        event_type="proposal",
        actor_role="supervisor",
        payload={"summary": "Run safe fake backtest."},
        artifacts=[{"artifact_id": "artifact-1", "artifact_type": "research_note", "path": "artifacts/note.md"}],
        provider="manual-review",
    )

    events = read_workload_events(path)

    assert event["event_type"] == "proposal"
    assert events[0]["actor"]["role"] == "supervisor"
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["run_id"] == "oracle-run-001"
