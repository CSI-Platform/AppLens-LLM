from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


def load_runtime_lanes(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_payload("runtime-lanes", payload)
    return payload


def get_lane(config: dict[str, Any], lane_id: str) -> dict[str, Any]:
    for lane in config["lanes"]:
        if lane["lane_id"] == lane_id:
            return lane
    raise KeyError(f"runtime lane not found: {lane_id}")
