from __future__ import annotations

from pathlib import Path

from applens_llm.schemas import validate_jsonl_file


ROOT = Path(__file__).resolve().parents[1]


def test_machine_profile_seed_matrix_is_valid_and_unique() -> None:
    rows = validate_jsonl_file("machine-profile", ROOT / "data" / "machines.seed.jsonl")
    machine_ids = [row["machine_id"] for row in rows]

    assert len(rows) >= 8
    assert len(machine_ids) == len(set(machine_ids))
    assert any(row["machine_id"] == "gaming-pc" for row in rows)
    assert all(row["collection"]["sanitized"] is False for row in rows if row["capture_status"] == "pending_capture")


def test_machine_profiles_cover_major_target_roles() -> None:
    rows = validate_jsonl_file("machine-profile", ROOT / "data" / "machines.seed.jsonl")
    roles = {role for row in rows for role in row["target_roles"]}

    assert {"training_candidate", "serving_candidate", "cpu_baseline", "mac_baseline"} <= roles
