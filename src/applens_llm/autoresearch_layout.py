from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadPaths:
    root: Path
    applens_dir: Path
    workload_profile: Path
    program_file: Path
    commands_file: Path
    metrics_file: Path
    probes_file: Path
    evals_dir: Path
    schemas_dir: Path
    runs_dir: Path
    blackboard_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    memory_proposed_dir: Path
    memory_wiki_dir: Path
    indexes_dir: Path


def workload_paths(root: Path) -> WorkloadPaths:
    applens = root / ".applens"
    return WorkloadPaths(
        root=root,
        applens_dir=applens,
        workload_profile=applens / "workload.json",
        program_file=applens / "program.md",
        commands_file=applens / "commands.json",
        metrics_file=applens / "metrics.json",
        probes_file=applens / "probes.json",
        evals_dir=applens / "evals",
        schemas_dir=applens / "schemas",
        runs_dir=applens / "runs",
        blackboard_dir=applens / "blackboard",
        artifacts_dir=applens / "artifacts",
        logs_dir=applens / "logs",
        memory_proposed_dir=applens / "memory" / "proposed",
        memory_wiki_dir=applens / "memory" / "wiki",
        indexes_dir=applens / "indexes",
    )


def init_workload_layout(root: Path, *, workload_id: str, display_name: str) -> WorkloadPaths:
    paths = workload_paths(root)
    for directory in [
        paths.schemas_dir,
        paths.evals_dir,
        paths.runs_dir,
        paths.blackboard_dir,
        paths.artifacts_dir,
        paths.logs_dir,
        paths.memory_proposed_dir,
        paths.memory_wiki_dir,
        paths.indexes_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not paths.workload_profile.exists():
        _write_json(
            paths.workload_profile,
            {
                "schema_version": "0.1",
                "workload_id": workload_id,
                "display_name": display_name,
                "workload_type": "generic_autoresearch",
                "program_file": ".applens/program.md",
                "commands_file": ".applens/commands.json",
                "metrics_file": ".applens/metrics.json",
                "probes_file": ".applens/probes.json",
                "evals_dir": ".applens/evals",
                "schemas_dir": ".applens/schemas",
                "allowed_actions": ["read_local_data", "write_artifacts"],
                "blocked_actions": ["credential_access", "system_change"],
                "required_artifacts": ["run_summary"],
                "model_role_needs": [{"role": "supervisor", "capabilities": ["reasoning", "safety_review"]}],
                "safety_gates": {
                    "requires_self_fit": True,
                    "allow_network": False,
                    "allow_writes_outside_workload_root": False,
                    "allow_live_trading": False,
                },
            },
        )

    if not paths.program_file.exists():
        paths.program_file.write_text(_default_program(display_name), encoding="utf-8")
    if not paths.commands_file.exists():
        _write_json(paths.commands_file, {"schema_version": "0.1", "commands": []})
    if not paths.metrics_file.exists():
        _write_json(paths.metrics_file, {"schema_version": "0.1", "primary_metric": "manual_review"})
    if not paths.probes_file.exists():
        _write_json(paths.probes_file, {"schema_version": "0.1", "probes": []})
    gitignore = paths.applens_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_default_gitignore(), encoding="utf-8")
    cases = paths.evals_dir / "cases.json"
    if not cases.exists():
        _write_json(cases, {"schema_version": "0.1", "cases": []})

    return paths


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_program(display_name: str) -> str:
    return (
        f"# {display_name} AutoResearch Program\n\n"
        "Goal:\nRun bounded, evidence-backed local research.\n\n"
        "Allowed:\n- propose one small step\n- run approved commands only\n- write artifacts\n\n"
        "Blocked:\n- credential access\n- system changes\n\n"
        "Loop:\n1. Read last result.\n2. Propose one small change.\n3. Run approved command when the selected step requires execution.\n4. Compare result.\n5. Write blackboard event.\n"
    )


def _default_gitignore() -> str:
    return (
        "/blackboard/\n"
        "/runs/*\n"
        "!/runs/*.example.json\n"
        "/artifacts/\n"
        "/logs/\n"
        "/memory/proposed/\n"
        "/indexes/\n"
        "*.local.json\n"
        "*.jsonl\n"
    )
