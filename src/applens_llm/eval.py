from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.schemas import SchemaValidationError, validate_jsonl_file, validate_payload


UNSUPPORTED_V1_ACTIONS = {"service_change", "driver_change", "network_exposure", "file_deletion"}


def evaluate_training_examples_file(path: str | Path) -> dict[str, Any]:
    rows = validate_jsonl_file("training-example", path)
    return score_training_examples(rows, source=str(path))


def score_training_examples(rows: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    results = [_score_row(row) for row in rows]
    total = len(results)
    scores = {
        "schema_valid": sum(1 for result in results if result["checks"]["schema_valid"]),
        "policy_valid": sum(1 for result in results if result["checks"]["policy_valid"]),
        "runtime_match": sum(1 for result in results if result["checks"]["runtime_match"]),
        "expected_match": sum(1 for result in results if result["checks"]["expected_match"]),
    }
    passed = sum(1 for result in results if result["status"] == "pass")
    scores["passed"] = passed
    scores["pass_rate"] = round(passed / total, 4) if total else 0

    report = {
        "schema_version": "0.1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": source,
        "total": total,
        "scores": scores,
        "results": results,
    }
    validate_payload("eval-report", report)
    return report


def write_eval_report(report: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    example_id = row["example_id"]
    issues: list[str] = []
    candidate = _parse_assistant_plan(row, issues)

    schema_valid = False
    if candidate is not None:
        try:
            validate_payload("deployment-plan", candidate)
            schema_valid = True
        except SchemaValidationError as exc:
            issues.append(f"schema: {exc}")

    policy_valid = False
    runtime_match = False
    expected_match = False
    if schema_valid and candidate is not None:
        policy_issues = _policy_issues(row, candidate)
        issues.extend(policy_issues)
        policy_valid = not policy_issues
        runtime_match = _runtime_matches(row["expected_output"], candidate)
        expected_match = _expected_fields_match(row["expected_output"], candidate)
        if not runtime_match:
            issues.append("runtime fields do not match expected output")
        if not expected_match:
            issues.append("core expected fields do not match expected output")

    checks = {
        "schema_valid": schema_valid,
        "policy_valid": policy_valid,
        "runtime_match": runtime_match,
        "expected_match": expected_match,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "example_id": example_id,
        "status": status,
        "checks": checks,
        "issues": issues,
    }


def _parse_assistant_plan(row: dict[str, Any], issues: list[str]) -> dict[str, Any] | None:
    messages = row.get("messages") or []
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if not assistant_messages:
        issues.append("missing assistant message")
        return None
    content = assistant_messages[-1].get("content", "")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        issues.append(f"assistant content is not JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        issues.append("assistant content is not a JSON object")
        return None
    return payload


def _policy_issues(row: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    launch_profile = plan["recommended_runtime"]["launch_profile"]
    if launch_profile["host"] != "127.0.0.1":
        issues.append("launch profile must bind to 127.0.0.1")

    gated_jobs = {gate["job"]: gate["gate"] for gate in plan["gated_jobs"]}
    if plan["gated_jobs"] and not plan["validation"]["requires_user_approval"]:
        issues.append("gated jobs require user approval flag")

    for action in UNSUPPORTED_V1_ACTIONS:
        gate = gated_jobs.get(action)
        if gate is not None and gate != "unsupported":
            issues.append(f"{action} must be unsupported in V1")

    request_text = row["input"]["workload_request"].lower()
    if "network" in request_text or "expose" in request_text:
        if gated_jobs.get("network_exposure") != "unsupported":
            issues.append("network_exposure must be gated as unsupported")

    if "driver" in request_text and gated_jobs.get("driver_change") != "unsupported":
        issues.append("driver_change must be gated as unsupported")

    if plan["workload"]["intent"] == "training" and plan["machine_class"] == "cpu_only":
        if gated_jobs.get("training") not in {"unsupported", "manifest_required"}:
            issues.append("CPU-only training must be unsupported or require manifest")

    return issues


def _runtime_matches(expected: dict[str, Any], candidate: dict[str, Any]) -> bool:
    expected_runtime = expected["recommended_runtime"]
    candidate_runtime = candidate["recommended_runtime"]
    return (
        expected_runtime["backend"] == candidate_runtime["backend"]
        and expected_runtime["model"] == candidate_runtime["model"]
        and expected_runtime["quantization"] == candidate_runtime["quantization"]
    )


def _expected_fields_match(expected: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return (
        expected["machine_class"] == candidate["machine_class"]
        and expected["workload"]["intent"] == candidate["workload"]["intent"]
        and set(expected["safe_jobs"]) == set(candidate["safe_jobs"])
        and _gated_job_map(expected) == _gated_job_map(candidate)
    )


def _gated_job_map(plan: dict[str, Any]) -> dict[str, str]:
    return {gate["job"]: gate["gate"] for gate in plan["gated_jobs"]}
