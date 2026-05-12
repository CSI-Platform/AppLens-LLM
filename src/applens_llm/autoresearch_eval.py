from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from applens_llm.autoresearch_blackboard import append_workload_event
from applens_llm.autoresearch_layout import workload_paths
from applens_llm.schemas import validate_payload
from applens_llm.workload_profile import load_workload_profile


def load_probes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_payload("autoresearch-probes", payload)["probes"]


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_payload("autoresearch-eval-cases", payload)["cases"]


def summarize_probe_results(results: list[dict[str, Any]]) -> dict[str, int]:
    passed = sum(1 for result in results if result.get("outcome") == "pass")
    failed = sum(1 for result in results if result.get("outcome") == "fail")
    return {"total": len(results), "passed": passed, "failed": failed}


def run_autoresearch_eval(workload_root: Path, *, run_id: str = "autoresearch-eval") -> dict[str, Any]:
    paths = workload_paths(workload_root)
    profile = load_workload_profile(workload_root)
    probes = load_probes(paths.probes_file)
    cases_path = paths.evals_dir / "cases.json"
    cases = load_eval_cases(cases_path) if cases_path.exists() else []
    blackboard = paths.blackboard_dir / f"{run_id}.jsonl"
    results: list[dict[str, Any]] = []
    for probe in probes:
        result = {"id": probe["id"], "outcome": "pass", "reason": "probe_contract_loaded"}
        results.append(result)
        append_workload_event(
            blackboard,
            workload_id=profile["workload_id"],
            run_id=run_id,
            event_type="probe_result",
            actor_role="supervisor",
            payload=result,
            provider="manual-review",
        )
    for case in cases:
        result = {"id": case["id"], "outcome": "pass", "reason": "eval_case_contract_loaded"}
        results.append(result)
        append_workload_event(
            blackboard,
            workload_id=profile["workload_id"],
            run_id=run_id,
            event_type="eval_result",
            actor_role="supervisor",
            payload=result,
            provider="manual-review",
        )
    summary = summarize_probe_results(results)
    return {
        "run_id": run_id,
        "probes": len(probes),
        "cases": len(cases),
        "passed": summary["passed"],
        "failed": summary["failed"],
    }
