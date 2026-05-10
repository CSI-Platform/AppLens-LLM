from __future__ import annotations

from pathlib import Path

import pytest

from applens_llm.schemas import SchemaValidationError, validate_document, validate_jsonl_file, validate_payload


ROOT = Path(__file__).resolve().parents[1]


def test_seed_artifacts_match_their_schemas() -> None:
    validate_document("deployment-plan", ROOT / "examples" / "gaming-pc-deployment-plan.json")
    validate_document("benchmark-record", ROOT / "examples" / "gaming-pc-benchmark-record.json")
    validate_document("fit-report", ROOT / "examples" / "asus-px13-fit-report.example.json")

    rows = validate_jsonl_file("training-example", ROOT / "data" / "examples.seed.jsonl")

    assert len(rows) >= 2
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert all("expected_output" in row for row in rows)


def test_capture_record_schema_accepts_minimal_manifest_row() -> None:
    validate_payload(
        "capture-record",
        {
            "schema_version": "0.1",
            "capture_id": "sample-host",
            "source": {"root_name": "raw", "folder": "."},
            "reports": {
                "applens": "AppLens_Results_sample-host.md",
                "applens_tune": None,
                "readme": None,
                "applens_log": None,
                "applens_tune_log": None,
            },
            "metadata": {
                "computer": "sample-host",
                "user": None,
                "scan_date": None,
                "mode": None,
                "machine": None,
                "os": None,
                "ram": None,
                "free_space": None,
            },
            "sections": {"applens": [], "applens_tune": []},
            "inferred": {
                "os_family": "unknown",
                "has_applens_report": True,
                "has_applens_tune_report": False,
                "has_local_llm_profile": False,
                "has_gpu_profile": False,
            },
            "privacy": {
                "sanitized": False,
                "raw_paths_detected": False,
                "serial_or_uuid_terms_detected": False,
            },
        },
    )


def test_invalid_deployment_plan_is_rejected(tmp_path: Path) -> None:
    bad_plan = tmp_path / "bad-plan.json"
    bad_plan.write_text('{"model": "qwen3.5-2b"}', encoding="utf-8")

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_document("deployment-plan", bad_plan)

    assert "runtime" in str(exc_info.value)
