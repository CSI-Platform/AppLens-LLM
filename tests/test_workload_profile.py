from __future__ import annotations

import pytest

from applens_llm.schemas import SchemaValidationError, validate_payload


def _valid_profile() -> dict:
    return {
        "schema_version": "0.1",
        "workload_id": "oracle",
        "display_name": "Oracle",
        "workload_type": "financial_research",
        "program_file": ".applens/program.md",
        "commands_file": ".applens/commands.json",
        "metrics_file": ".applens/metrics.json",
        "probes_file": ".applens/probes.json",
        "evals_dir": ".applens/evals",
        "schemas_dir": ".applens/schemas",
        "allowed_actions": ["read_local_data", "write_artifacts", "run_backtests", "write_reports"],
        "blocked_actions": ["live_trade", "broker_order", "credential_access", "system_change"],
        "required_artifacts": ["strategy_candidate", "backtest_result", "risk_review", "research_report"],
        "model_role_needs": [
            {"role": "supervisor", "capabilities": ["reasoning", "safety_review"]},
            {"role": "candidate", "capabilities": ["small_model_legible_tasks"]},
        ],
        "safety_gates": {
            "requires_self_fit": True,
            "allow_network": False,
            "allow_writes_outside_workload_root": False,
            "allow_live_trading": False,
        },
    }


def test_workload_profile_schema_accepts_oracle_profile() -> None:
    assert validate_payload("workload-profile", _valid_profile())["workload_id"] == "oracle"


def test_workload_profile_schema_rejects_live_trading_enabled() -> None:
    payload = _valid_profile()
    payload["safety_gates"]["allow_live_trading"] = True

    with pytest.raises(SchemaValidationError):
        validate_payload("workload-profile", payload)

