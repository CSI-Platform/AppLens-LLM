from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


def append_workload_event(
    path: Path,
    *,
    workload_id: str,
    run_id: str,
    event_type: str,
    actor_role: str,
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
    lane_id: str | None = None,
    provider: str | None = None,
    commit_safe: bool = False,
    local_paths_included: bool = True,
) -> dict[str, Any]:
    event = {
        "schema_version": "0.1",
        "event_id": f"evt-{uuid.uuid4().hex}",
        "workload_id": workload_id,
        "run_id": run_id,
        "actor": {"role": actor_role, "lane_id": lane_id, "provider": provider},
        "event_type": event_type,
        "created_at": _utc_now(),
        "payload": payload,
        "artifacts": artifacts or [],
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


def read_workload_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

