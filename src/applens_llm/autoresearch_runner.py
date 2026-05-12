from __future__ import annotations

from pathlib import Path
from typing import Any

from applens_llm.autoresearch_blackboard import append_workload_event
from applens_llm.autoresearch_commands import CommandBlockedError, execute_allowed_command
from applens_llm.autoresearch_manifest import load_run_manifest, self_fit_is_fresh
from applens_llm.workload_profile import (
    commands_path_for_profile,
    enforce_command_policy,
    load_allowed_commands,
    load_workload_profile,
)


def run_autoresearch_once(
    workload_root: Path,
    *,
    manifest_path: Path,
    skip_self_fit: bool = False,
) -> dict[str, Any]:
    profile = load_workload_profile(workload_root)
    manifest = load_run_manifest(manifest_path)
    blackboard_path = _resolve_under_root(workload_root, manifest["blackboard_path"])
    artifact_dir = _resolve_under_root(workload_root, manifest["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    append_workload_event(
        blackboard_path,
        workload_id=manifest["workload_id"],
        run_id=manifest["run_id"],
        event_type="run_started",
        actor_role="supervisor",
        payload={"mode": manifest["mode"]},
        provider="manual-review",
    )
    if _self_fit_required(manifest, skip_self_fit) and not _manifest_self_fit_fresh(workload_root, manifest):
        payload = {"outcome": "blocked", "reason": "self_fit_required"}
        append_workload_event(
            blackboard_path,
            workload_id=manifest["workload_id"],
            run_id=manifest["run_id"],
            event_type="failure",
            actor_role="supervisor",
            payload=payload,
            provider="manual-review",
        )
        return {"run_id": manifest["run_id"], **payload}

    next_step = manifest.get("next_step")
    if not next_step:
        payload = {"outcome": "manual_review_required", "reason": "no_next_step"}
        append_workload_event(
            blackboard_path,
            workload_id=manifest["workload_id"],
            run_id=manifest["run_id"],
            event_type="proposal",
            actor_role="supervisor",
            payload=payload,
            provider="manual-review",
        )
        return {"run_id": manifest["run_id"], **payload}

    commands = load_allowed_commands(commands_path_for_profile(workload_root, profile))
    command_id = next_step["command_id"]
    selected_command = commands.get(command_id)
    if selected_command is None:
        return _record_blocked_command(blackboard_path, manifest, command_id, f"command is not allowlisted: {command_id}")
    try:
        enforce_command_policy(
            selected_command,
            allowed_actions=profile["allowed_actions"],
            blocked_actions=profile["blocked_actions"],
        )
    except ValueError as exc:
        return _record_blocked_command(blackboard_path, manifest, command_id, str(exc))

    append_workload_event(
        blackboard_path,
        workload_id=manifest["workload_id"],
        run_id=manifest["run_id"],
        event_type="command_started",
        actor_role="workload_executor",
        payload={"command_id": command_id},
        provider="local-lane",
    )
    try:
        result = execute_allowed_command(
            selected_command,
            parameters=next_step.get("parameters", {}),
            workload_root=workload_root,
            timeout_seconds=int(manifest["limits"]["command_timeout_seconds"]),
        )
    except (CommandBlockedError, ValueError) as exc:
        return _record_blocked_command(blackboard_path, manifest, command_id, str(exc))

    append_workload_event(
        blackboard_path,
        workload_id=manifest["workload_id"],
        run_id=manifest["run_id"],
        event_type="command_result",
        actor_role="workload_executor",
        payload=result,
        provider="local-lane",
    )
    append_workload_event(
        blackboard_path,
        workload_id=manifest["workload_id"],
        run_id=manifest["run_id"],
        event_type="run_finished",
        actor_role="supervisor",
        payload={"outcome": result["outcome"]},
        provider="manual-review",
    )
    return {"run_id": manifest["run_id"], "outcome": result["outcome"], "command_id": command_id}


def _record_blocked_command(
    blackboard_path: Path,
    manifest: dict[str, Any],
    command_id: str,
    reason: str,
) -> dict[str, Any]:
    payload = {"outcome": "blocked", "command_id": command_id, "reason": reason}
    append_workload_event(
        blackboard_path,
        workload_id=manifest["workload_id"],
        run_id=manifest["run_id"],
        event_type="command_blocked",
        actor_role="workload_executor",
        payload=payload,
        provider="local-lane",
    )
    return {"run_id": manifest["run_id"], **payload}


def _self_fit_required(manifest: dict[str, Any], skip_self_fit: bool) -> bool:
    self_fit = manifest["self_fit"]
    return bool(self_fit["required"] and not self_fit["skip"] and not skip_self_fit)


def _manifest_self_fit_fresh(workload_root: Path, manifest: dict[str, Any]) -> bool:
    self_fit = manifest["self_fit"]
    machine = self_fit.get("machine_fingerprint")
    runtime = self_fit.get("runtime_fingerprint")
    if not machine or not runtime:
        return False
    return self_fit_is_fresh(
        workload_root,
        machine_fingerprint=machine,
        runtime_fingerprint=runtime,
        freshness_hours=int(self_fit["freshness_hours"]),
    )


def _resolve_under_root(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    candidate = path if path.is_absolute() else root / path
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    if candidate_resolved != root_resolved and root_resolved not in candidate_resolved.parents:
        raise ValueError(f"path escapes workload root: {maybe_relative}")
    return candidate_resolved
