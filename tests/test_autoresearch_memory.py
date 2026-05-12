from __future__ import annotations

from pathlib import Path

import pytest

from applens_llm.autoresearch_memory import promote_memory, write_memory_proposal


def test_write_memory_proposal_does_not_promote(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    proposal = write_memory_proposal(
        root,
        run_id="oracle-run-001",
        title="Use fake backtests first",
        body="Keep Oracle loops on fake/read-only commands until explicit approval.",
    )

    assert proposal.exists()
    assert proposal.parent == root / ".applens" / "memory" / "proposed"
    assert not (root / ".applens" / "memory" / "wiki" / proposal.name).exists()


def test_promote_memory_requires_proposal_under_workload(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    proposal = write_memory_proposal(
        root,
        run_id="oracle-run-001",
        title="Memory rule",
        body="Only promote with an explicit command.",
    )

    promoted = promote_memory(root, proposal)

    assert promoted == root / ".applens" / "memory" / "wiki" / proposal.name
    assert promoted.exists()


def test_promote_memory_blocks_outside_path(tmp_path: Path) -> None:
    root = tmp_path / "Oracle"
    outside = tmp_path / "outside.md"
    outside.write_text("bad", encoding="utf-8")

    with pytest.raises(ValueError):
        promote_memory(root, outside)

