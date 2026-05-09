from __future__ import annotations

import json
from pathlib import Path

from applens_llm.blackboard import append_event, read_events, start_experiment
from applens_llm.schemas import validate_payload


def test_blackboard_record_schema_accepts_task_and_response() -> None:
    task = {
        "schema_version": "0.1",
        "event_id": "evt-test-task",
        "experiment_id": "exp-test",
        "event_type": "task",
        "created_at": "2026-05-09T00:00:00Z",
        "payload": {"task_id": "task-1", "prompt": "Compare these runtimes.", "metadata": {}},
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }
    response = {
        "schema_version": "0.1",
        "event_id": "evt-test-response",
        "experiment_id": "exp-test",
        "event_type": "model_response",
        "created_at": "2026-05-09T00:00:01Z",
        "payload": {
            "task_id": "task-1",
            "lane_id": "fast-nvidia",
            "model_label": "jan-v35-4b",
            "backend": "cuda",
            "accelerator_ids": ["nvidia-dgpu-0"],
            "latency_ms": 1250,
            "outcome": "success",
            "content": "Short answer.",
        },
        "privacy": {"commit_safe": False, "local_paths_included": False},
    }

    validate_payload("blackboard-record", task)
    validate_payload("blackboard-record", response)


def test_append_and_read_blackboard_events(tmp_path: Path) -> None:
    path = tmp_path / "experiment.jsonl"
    event = start_experiment(path, experiment_id="exp-test", title="Unit test")
    append_event(
        path,
        experiment_id="exp-test",
        event_type="task",
        payload={"task_id": "task-1", "prompt": "Hello", "metadata": {}},
        commit_safe=True,
    )

    events = read_events(path)

    assert events[0]["event_type"] == "experiment_started"
    assert events[0]["event_id"] == event["event_id"]
    assert events[1]["payload"]["prompt"] == "Hello"
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
