from __future__ import annotations

from pathlib import Path

from applens_llm.autoresearch_layout import init_workload_layout, workload_paths


def test_init_workload_layout_creates_committed_and_ignored_split(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    paths = init_workload_layout(root, workload_id="oracle", display_name="Oracle")

    assert paths.applens_dir == root / ".applens"
    assert (root / ".applens" / "workload.json").exists()
    assert (root / ".applens" / "program.md").exists()
    assert (root / ".applens" / "commands.json").exists()
    assert (root / ".applens" / "metrics.json").exists()
    assert (root / ".applens" / "probes.json").exists()
    assert (root / ".applens" / "evals").is_dir()
    assert (root / ".applens" / "schemas").is_dir()
    assert (root / ".applens" / "runs").is_dir()
    assert (root / ".applens" / "blackboard").is_dir()
    assert (root / ".applens" / "artifacts").is_dir()
    assert (root / ".applens" / "logs").is_dir()
    assert (root / ".applens" / "memory" / "proposed").is_dir()
    assert (root / ".applens" / "memory" / "wiki").is_dir()
    assert (root / ".applens" / "indexes").is_dir()
    gitignore = (root / ".applens" / ".gitignore").read_text(encoding="utf-8")
    assert "/blackboard/" in gitignore
    assert "/runs/*" in gitignore
    assert "!/runs/*.example.json" in gitignore


def test_workload_paths_are_stable(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    paths = workload_paths(root)

    assert paths.workload_profile == root / ".applens" / "workload.json"
    assert paths.blackboard_dir == root / ".applens" / "blackboard"
