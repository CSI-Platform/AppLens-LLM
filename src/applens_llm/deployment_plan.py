from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


def build_deployment_plan(
    *,
    scorecard: dict[str, Any],
    plan_id: str | None = None,
    workload_name: str = "Local LLM outfitting",
    workload_intent: str = "agent_runtime",
) -> dict[str, Any]:
    rankings = list(scorecard.get("rankings") or [])
    if not rankings:
        raise ValueError("scorecard must include at least one ranking")

    primary = _primary_worker(rankings)
    deep = _deep_worker(rankings)
    supervisor_candidate = _local_supervisor_candidate(rankings)
    machine_label = (scorecard.get("machine") or {}).get("label", "local machine")

    plan = {
        "schema_version": "0.1",
        "plan_id": plan_id or f"{scorecard['scorecard_id']}-deployment-plan",
        "machine_class": _machine_class(scorecard, rankings),
        "workload": {"name": workload_name, "intent": workload_intent},
        "recommended_runtime": _recommended_runtime(primary),
        "safe_jobs": ["read_only_scan", "benchmark", "eval_sweep", "dataset_prep", "local_inference"],
        "gated_jobs": [
            {"job": "training", "gate": "manifest_required"},
            {"job": "large_download", "gate": "user_approval"},
            {"job": "driver_change", "gate": "unsupported"},
            {"job": "service_change", "gate": "unsupported"},
            {"job": "network_exposure", "gate": "unsupported"},
        ],
        "findings": _findings(machine_label, primary, deep, supervisor_candidate),
        "outfitting": {
            "supervisor_baseline": {
                "role": "planner_supervisor",
                "runtime": "cloud_api",
                "relative_score": 100,
                "status": "reference_baseline",
                "replacement_gate_score": 90,
            },
            "local_supervisor_candidate": supervisor_candidate,
            "assignments": _assignments(primary, deep, rankings),
            "runtime_profiles": _runtime_profiles([primary, deep]),
            "context_profiles": _context_profiles([primary, deep]),
            "preflight_actions": _preflight_actions(rankings),
            "tune_recommendations": _tune_recommendations(rankings),
            "promotion_gates": _promotion_gates(supervisor_candidate),
            "next_drilldowns": _next_drilldowns(rankings),
        },
        "validation": {
            "requires_benchmark": True,
            "requires_user_approval": True,
            "policy_notes": [
                "The cloud/API planner-supervisor is the 100-point reference baseline until a local model passes supervisor replacement gates.",
                "Bind local inference endpoints to 127.0.0.1 by default.",
                "Do not treat advertised context or combined graphics memory as usable deployment capacity without benchmark proof.",
                "Downloads, training, driver changes, service changes, and network exposure remain gated actions.",
            ],
        },
    }
    validate_payload("deployment-plan", plan)
    return plan


def write_deployment_plan(
    *,
    scorecard_path: Path,
    output_path: Path,
    plan_id: str | None = None,
    workload_name: str = "Local LLM outfitting",
    workload_intent: str = "agent_runtime",
) -> dict[str, Any]:
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    plan = build_deployment_plan(
        scorecard=scorecard,
        plan_id=plan_id,
        workload_name=workload_name,
        workload_intent=workload_intent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def _primary_worker(rankings: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = [
        row
        for row in rankings
        if row.get("recommended_role") in {"fast_chat", "general_chat", "coding", "summarization"}
        and "observed_failure" not in row.get("blockers", [])
    ]
    return max(preferred or rankings, key=lambda row: (row.get("fit_score", 0), -row.get("rank", 999)))


def _deep_worker(rankings: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = [
        row
        for row in rankings
        if row.get("recommended_role") == "deep_review" and "observed_failure" not in row.get("blockers", [])
    ]
    return max(preferred or rankings, key=lambda row: (row.get("fit_score", 0), -row.get("rank", 999)))


def _local_supervisor_candidate(rankings: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = sorted(
        rankings,
        key=lambda row: (
            _supervisor_relative_score(row),
            row.get("fit_score", 0),
            -row.get("rank", 999),
        ),
        reverse=True,
    )
    candidate = candidates[0]
    relative_score = _supervisor_relative_score(candidate)
    status = "ready" if relative_score >= 90 else "not_ready"
    missing: list[str] = []
    categories = ((candidate.get("evidence") or {}).get("capability_categories") or {})
    for category in ("tool_calling", "coding", "handoff_planning"):
        if _float(categories.get(category)) < 85:
            missing.append(f"{category}_below_supervisor_gate")
    if (candidate.get("evidence") or {}).get("capability_record_count", 0) == 0:
        missing.append("missing_local_capability_record")
    return {
        "model_id": candidate["model_id"],
        "display_name": candidate.get("display_name", candidate["model_id"]),
        "relative_score": relative_score,
        "status": status,
        "reason": "Best local planner-supervisor candidate by capability, fit, and context evidence.",
        "missing_evidence": missing,
    }


def _supervisor_relative_score(row: dict[str, Any]) -> int:
    breakdown = row.get("score_breakdown") or {}
    evidence = row.get("evidence") or {}
    categories = evidence.get("capability_categories") or {}
    capability = _float(evidence.get("capability_score_pct"))
    if not capability:
        capability = _float(breakdown.get("agent_capability")) * 100 / 24
    tool = _float(categories.get("tool_calling")) or capability
    coding = _float(categories.get("coding")) or capability
    handoff = _float(categories.get("handoff_planning")) or capability
    context = _float(evidence.get("context_score_pct"))
    fit = _float(row.get("fit_score"))
    score = round((tool * 0.22) + (coding * 0.22) + (handoff * 0.2) + (context * 0.16) + (fit * 0.2))
    return max(0, min(100, score))


def _machine_class(scorecard: dict[str, Any], rankings: list[dict[str, Any]]) -> str:
    accelerator_ids = {
        accelerator_id
        for row in rankings
        for accelerator_id in ((row.get("best_lane") or {}).get("accelerator_ids") or [])
    }
    ram_gb = ((scorecard.get("machine") or {}).get("platform") or {}).get("ram_gb", 0)
    if not accelerator_ids:
        return "cpu_only"
    if len(accelerator_ids) >= 2 and ram_gb >= 32:
        return "mid_local_ai_workstation"
    return "small_local_ai_worker"


def _recommended_runtime(primary: dict[str, Any]) -> dict[str, Any]:
    evidence = primary.get("evidence") or {}
    lane = primary.get("best_lane") or {}
    return {
        "backend": "llama.cpp",
        "model": primary["model_id"],
        "quantization": primary.get("quantization", "unknown"),
        "launch_profile": {
            "context_tokens": _context_tokens(evidence),
            "threads": 12,
            "gpu_layers": 99 if lane.get("backend") != "cpu" else 0,
            "host": "127.0.0.1",
            "port": _lane_port(lane),
            "extra_flags": ["--parallel", "1"],
        },
    }


def _assignments(primary: dict[str, Any], deep: dict[str, Any], rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assignments = [
        _assignment("primary_local_worker", primary, "Use for fast local execution, chat, short tool loops, and cheap overnight worker steps."),
    ]
    if deep["model_id"] != primary["model_id"]:
        assignments.append(
            _assignment("deep_review_worker", deep, "Use for slower synthesis, critique, and review when quality matters more than latency.")
        )
    avoid = [
        row
        for row in rankings
        if row["model_id"] not in {primary["model_id"], deep["model_id"]}
        and (_float((row.get("evidence") or {}).get("avg_latency_ms")) >= 45000 or row.get("fit_score", 0) < 75)
    ]
    for row in avoid:
        assignments.append(
            _assignment("avoid_primary", row, "Do not use as the default worker until speed, stability, and role quality improve.")
        )
    return assignments


def _assignment(role: str, row: dict[str, Any], rationale: str) -> dict[str, Any]:
    lane = row.get("best_lane") or {}
    return {
        "role": role,
        "model_id": row["model_id"],
        "display_name": row.get("display_name", row["model_id"]),
        "lane_id": lane.get("lane_id", "unknown"),
        "backend": lane.get("backend", "unknown"),
        "accelerator_ids": lane.get("accelerator_ids", []),
        "fit_score": row.get("fit_score", 0),
        "confidence": row.get("confidence", "insufficient"),
        "status": "assigned" if role != "avoid_primary" else "avoid_primary",
        "limits": _limits(row),
        "rationale": rationale,
    }


def _runtime_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row["model_id"] in seen:
            continue
        seen.add(row["model_id"])
        lane = row.get("best_lane") or {}
        profiles.append(
            {
                "profile_id": f"{row['model_id']}-{lane.get('lane_id', 'unknown')}",
                "model_id": row["model_id"],
                "lane_id": lane.get("lane_id", "unknown"),
                "engine": "llama.cpp",
                "backend": lane.get("backend", "unknown"),
                "accelerator_ids": lane.get("accelerator_ids", []),
                "context_tokens": _context_tokens(row.get("evidence") or {}),
                "gpu_layers": 99 if lane.get("backend") != "cpu" else 0,
                "threads": 12,
                "host": "127.0.0.1",
                "port": _lane_port(lane),
                "extra_flags": ["--parallel", "1"],
            }
        )
    return profiles


def _context_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        evidence = row.get("evidence") or {}
        role = row.get("recommended_role", "general_chat")
        key = (row["model_id"], role)
        if key in seen:
            continue
        seen.add(key)
        profiles.append(
            {
                "model_id": row["model_id"],
                "role": role,
                "advertised_context_tokens": int(_float(evidence.get("advertised_context_tokens"))),
                "recommended_context_tokens": _context_tokens(evidence),
                "status": evidence.get("context_evidence_status", "unknown"),
            }
        )
    return profiles


def _preflight_actions(rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = [
        {
            "action": "close_competing_llm_apps",
            "owner": "User",
            "required": True,
            "reason": "Free local memory and ports before measuring or launching llama.cpp lanes.",
            "verification": "Confirm Claude, Jan, Ollama, and unused llama.cpp servers are closed or intentionally excluded.",
        },
        {
            "action": "confirm_localhost_binding",
            "owner": "AppLens-LLM",
            "required": True,
            "reason": "Keep local model endpoints private by default.",
            "verification": "Launch profiles bind to 127.0.0.1.",
        },
    ]
    if any("requires_amd_vgm_vulkan_path" in row.get("blockers", []) for row in rankings):
        actions.append(
            {
                "action": "confirm_vgm_active",
                "owner": "AppLens-Tune",
                "required": True,
                "reason": "AMD/VGM assignments depend on reserved iGPU memory and Vulkan visibility.",
                "verification": "Run llama.cpp device inventory and AMD memory telemetry before deep-lane benchmarks.",
            }
        )
    return actions


def _tune_recommendations(rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations = [
        {
            "owner": "AppLens-Tune",
            "action": "free_memory_before_local_llm",
            "priority": "High",
            "rationale": "Close competing LLM apps and background-heavy processes before fit or context tests.",
            "requires_restart": False,
        },
        {
            "owner": "AppLens-Tune",
            "action": "preserve_localhost_only_runtime",
            "priority": "Medium",
            "rationale": "Local model endpoints should stay private unless a later gated workflow explicitly changes that.",
            "requires_restart": False,
        },
    ]
    if any("requires_amd_vgm_vulkan_path" in row.get("blockers", []) for row in rankings):
        recommendations.append(
            {
                "owner": "AppLens-Tune",
                "action": "verify_amd_vgm_vulkan_readiness",
                "priority": "High",
                "rationale": "Large-model assignments depend on AMD VGM being active and visible to llama.cpp Vulkan.",
                "requires_restart": True,
            }
        )
    return recommendations


def _promotion_gates(supervisor_candidate: dict[str, Any]) -> list[dict[str, Any]]:
    current = supervisor_candidate["relative_score"]
    return [
        {
            "gate": "local_supervisor_replacement",
            "required_score": 90,
            "current_score": current,
            "status": "pass" if current >= 90 else "fail",
        },
        {
            "gate": "tool_calling_and_coding_drilldown",
            "required_score": 85,
            "current_score": min(current, 85),
            "status": "pass" if current >= 85 else "fail",
        },
    ]


def _next_drilldowns(rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(rankings, key=lambda row: row.get("rank", 999))[:3]
    return [
        {
            "benchmark": "applens-local-v1-role-drilldown",
            "target_model_id": row["model_id"],
            "purpose": row.get("next_benchmark", "Collect role-specific local capability evidence."),
            "success_signal": "Schema-valid tool calls, passing coding checks, stable context tier, and no fallback/OOM.",
        }
        for row in rows
    ]


def _findings(
    machine_label: str,
    primary: dict[str, Any],
    deep: dict[str, Any],
    supervisor_candidate: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "category": "Runtime",
            "risk": "Low",
            "evidence": f"{primary['display_name']} is the primary local worker on {primary['best_lane']['backend']}.",
            "guidance": "Use the primary worker for fast local tasks while cloud/API remains planner-supervisor.",
            "verification": "Run a short local capability probe before starting overnight work.",
        },
        {
            "category": "Model",
            "risk": "Medium",
            "evidence": f"{deep['display_name']} is assigned for deeper review, but slower lanes should not run the whole loop.",
            "guidance": "Use deep-review models for critique and synthesis steps, not as the default executor.",
            "verification": "Track latency, context quality, and failure modes in benchmark records.",
        },
        {
            "category": "Benchmark",
            "risk": "Medium",
            "evidence": (
                f"Best local supervisor candidate on {machine_label} scores "
                f"{supervisor_candidate['relative_score']}/100 against the cloud/API supervisor baseline."
            ),
            "guidance": "Keep cloud/API as supervisor until local replacement gates pass.",
            "verification": "Run role drilldowns for tool calling, coding, handoff planning, and long-context retrieval.",
        },
    ]


def _limits(row: dict[str, Any]) -> list[str]:
    limits = list(row.get("blockers") or [])
    evidence = row.get("evidence") or {}
    latency = _float(evidence.get("avg_latency_ms"))
    if latency >= 30000:
        limits.append("high_latency")
    if evidence.get("context_evidence_status") in {"advertised_unproven", "unknown"}:
        limits.append("context_unproven")
    return sorted(set(limits))


def _context_tokens(evidence: dict[str, Any]) -> int:
    recommended = int(_float(evidence.get("recommended_context_tokens")))
    if recommended:
        return recommended
    tested = int(_float(evidence.get("max_tested_context_tokens")))
    if tested:
        return tested
    return 4096


def _lane_port(lane: dict[str, Any]) -> int:
    lane_id = str(lane.get("lane_id", ""))
    backend = str(lane.get("backend", ""))
    accelerator_ids = [str(accelerator_id) for accelerator_id in lane.get("accelerator_ids", [])]
    if "nvidia" in lane_id or any("nvidia" in accelerator_id for accelerator_id in accelerator_ids):
        return 18080
    if "amd" in lane_id or any("amd" in accelerator_id for accelerator_id in accelerator_ids):
        return 18082
    if backend == "cpu":
        return 18084
    return 18080


def _float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
