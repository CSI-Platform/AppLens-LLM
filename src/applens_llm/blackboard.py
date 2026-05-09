from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


def start_experiment(path: Path, *, experiment_id: str, title: str) -> dict[str, Any]:
    return append_event(
        path,
        experiment_id=experiment_id,
        event_type="experiment_started",
        payload={"title": title},
        commit_safe=False,
    )


def append_event(
    path: Path,
    *,
    experiment_id: str,
    event_type: str,
    payload: dict[str, Any],
    commit_safe: bool = False,
    local_paths_included: bool = False,
) -> dict[str, Any]:
    event = {
        "schema_version": "0.1",
        "event_id": f"evt-{uuid.uuid4().hex}",
        "experiment_id": experiment_id,
        "event_type": event_type,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": payload,
        "privacy": {
            "commit_safe": commit_safe,
            "local_paths_included": local_paths_included,
        },
    }
    validate_payload("blackboard-record", event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
