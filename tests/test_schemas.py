from __future__ import annotations

from pathlib import Path

import pytest

from applens_llm.schemas import SchemaValidationError, validate_document, validate_jsonl_file


ROOT = Path(__file__).resolve().parents[1]


def test_seed_artifacts_match_their_schemas() -> None:
    validate_document("deployment-plan", ROOT / "examples" / "gaming-pc-deployment-plan.json")
    validate_document("benchmark-record", ROOT / "examples" / "gaming-pc-benchmark-record.json")

    rows = validate_jsonl_file("training-example", ROOT / "data" / "examples.seed.jsonl")

    assert len(rows) >= 2
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert all("expected_output" in row for row in rows)


def test_invalid_deployment_plan_is_rejected(tmp_path: Path) -> None:
    bad_plan = tmp_path / "bad-plan.json"
    bad_plan.write_text('{"model": "qwen3.5-2b"}', encoding="utf-8")

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_document("deployment-plan", bad_plan)

    assert "runtime" in str(exc_info.value)
