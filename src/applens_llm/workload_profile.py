from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from applens_llm.autoresearch_layout import workload_paths
from applens_llm.schemas import validate_payload


_PARAM_PATTERN = re.compile(r"^[A-Za-z0-9_./:@+=,-]+$")


def load_workload_profile(root_or_path: Path) -> dict[str, Any]:
    path = root_or_path if root_or_path.name.endswith(".json") else workload_paths(root_or_path).workload_profile
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_payload("workload-profile", payload)


def load_allowed_commands(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    commands = payload.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError(f"{path}: commands must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for command in commands:
        if not isinstance(command, dict):
            raise ValueError(f"{path}: each command must be an object")
        command_id = command.get("id")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError(f"{path}: command id is required")
        if command_id in indexed:
            raise ValueError(f"{path}: duplicate command id {command_id}")
        _validate_command(command, path)
        indexed[command_id] = command
    return indexed


def bind_command_parameters(command: dict[str, Any], parameters: dict[str, Any], *, workload_root: Path | None = None) -> str:
    allowed = command.get("allowed_parameters") or {}
    unexpected = sorted(set(parameters) - set(allowed))
    if unexpected:
        raise ValueError(f"unexpected command parameter(s): {', '.join(unexpected)}")
    safe_parameters = {
        key: _safe_parameter_value(value, _parameter_spec(allowed.get(key)), workload_root=workload_root)
        for key, value in parameters.items()
    }
    try:
        return str(command["command_template"]).format(**safe_parameters)
    except KeyError as exc:
        raise ValueError(f"missing command parameter: {exc.args[0]}") from exc


def enforce_command_policy(
    command: dict[str, Any],
    *,
    allowed_actions: list[str],
    blocked_actions: list[str],
) -> dict[str, Any]:
    declared_actions = [str(action) for action in command.get("actions") or []]
    normalized_allowed = {_normalize_action(action) for action in allowed_actions}
    normalized_blocked = {_normalize_action(action) for action in blocked_actions}
    for action in declared_actions:
        normalized = _normalize_action(action)
        if normalized in normalized_blocked:
            raise ValueError(f"command {command['id']} declares blocked action: {action}")
        if normalized_allowed and normalized not in normalized_allowed:
            raise ValueError(f"command {command['id']} declares unallowed action: {action}")

    surface = " ".join(
        str(command.get(key, "")) for key in ("id", "description", "command_template")
    )
    for action in blocked_actions:
        if _contains_action(surface, action):
            raise ValueError(f"command {command['id']} matches blocked action: {action}")
    return command


def commands_path_for_profile(workload_root: Path, profile: dict[str, Any]) -> Path:
    return _resolve_under_root(workload_root, profile["commands_file"])


def _validate_command(command: dict[str, Any], source: Path) -> None:
    required = {
        "id",
        "description",
        "command_template",
        "working_directory",
        "allowed_parameters",
        "required_inputs",
        "expected_outputs",
        "timeout_seconds",
        "network_policy",
        "risk_level",
    }
    missing = sorted(required - set(command))
    if missing:
        raise ValueError(f"{source}: command {command.get('id', '<unknown>')} missing {', '.join(missing)}")
    if command["network_policy"] != "disabled":
        raise ValueError(f"{source}: command {command['id']} must disable network access in V1")
    if command["risk_level"] not in {"low", "medium"}:
        raise ValueError(f"{source}: command {command['id']} has unsupported risk level")
    if int(command["timeout_seconds"]) < 1:
        raise ValueError(f"{source}: command {command['id']} timeout must be positive")


def _safe_parameter_value(value: Any, spec: str, *, workload_root: Path | None) -> str:
    text = str(value)
    if _contains_parent_path_segment(text):
        raise ValueError(f"unsafe command parameter value: {text}")
    if spec in {"path", "input_path", "output_path"}:
        if workload_root is None:
            raise ValueError("path parameters require a workload root")
        return _resolve_under_root(workload_root, text).as_posix()
    if not _PARAM_PATTERN.match(text):
        raise ValueError(f"unsafe command parameter value: {text}")
    return text


def _parameter_spec(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or value.get("kind") or "")
    return str(value or "")


def _contains_parent_path_segment(value: str) -> bool:
    return any(part == ".." for part in re.split(r"[\\/]+", value))


def _normalize_action(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _contains_action(surface: str, action: str) -> bool:
    normalized_surface = _normalize_action(surface)
    normalized_action = _normalize_action(action)
    compact_surface = normalized_surface.replace("_", "")
    compact_action = normalized_action.replace("_", "")
    return normalized_action in normalized_surface or compact_action in compact_surface


def _resolve_under_root(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve()
    path = Path(relative_path)
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"path escapes workload root: {relative_path}")
    return candidate
