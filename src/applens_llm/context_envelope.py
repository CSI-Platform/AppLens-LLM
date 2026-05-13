from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.fit_report import load_machine_profile
from applens_llm.model_fit_scorecard import load_model_candidates
from applens_llm.schemas import validate_payload


DEFAULT_CONTEXT_TIERS = [262144, 131072, 65536, 32768, 16384, 8192, 4096]
MIN_USEFUL_LONG_CONTEXT_QUALITY = 60
MIN_USEFUL_GENERATION_TPS = 1.0


def build_context_envelope(
    *,
    machine_profile: dict[str, Any],
    model_candidates: list[dict[str, Any]],
    context_observations: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
    envelope_id: str | None = None,
) -> dict[str, Any]:
    observations = context_observations or []
    observations_by_model = _observations_by_model(observations)
    models = [
        _model_context(candidate, observations_by_model.get(candidate["model_id"], []))
        for candidate in model_candidates
    ]
    envelope = {
        "schema_version": "0.1",
        "envelope_id": envelope_id or f"context-{machine_profile['machine_id']}",
        "created_at": created_at or _utc_now(),
        "machine": {
            "machine_id": machine_profile["machine_id"],
            "label": machine_profile["label"],
        },
        "models": models,
        "comparisons": _comparisons(models),
        "next_actions": _next_actions(models),
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }
    validate_payload("context-envelope", envelope)
    return envelope


def write_context_envelope(
    *,
    machine_profile_path: Path,
    model_candidates_path: Path,
    output_path: Path,
    machine_id: str | None = None,
    context_observation_paths: list[Path] | None = None,
    created_at: str | None = None,
    envelope_id: str | None = None,
) -> dict[str, Any]:
    envelope = build_context_envelope(
        machine_profile=load_machine_profile(machine_profile_path, machine_id=machine_id),
        model_candidates=load_model_candidates(model_candidates_path),
        context_observations=load_context_observations(context_observation_paths or []),
        created_at=created_at,
        envelope_id=envelope_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return envelope


def load_context_observations(paths: list[Path]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if not stripped:
            continue
        lines = [line for line in text.splitlines() if line.strip()]
        if stripped.startswith("["):
            observations.extend(json.loads(text))
            continue
        if stripped.startswith("{") and len(lines) == 1:
            payload = json.loads(text)
            if "observations" in payload:
                observations.extend(payload["observations"])
            else:
                observations.append(payload)
            continue
        for line in lines:
            observations.append(json.loads(line))
    return observations


def _model_context(candidate: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    advertised_tokens = _int(candidate.get("advertised_context_tokens"))
    planned_tiers = _planned_tiers(advertised_tokens)
    normalized_observations = [_normalize_observation(observation) for observation in observations]
    passing = [observation for observation in normalized_observations if observation["status"] == "pass"]
    stable = [
        observation
        for observation in passing
        if observation["quality_score_pct"] >= MIN_USEFUL_LONG_CONTEXT_QUALITY
        and observation["generation_tokens_per_second"] >= MIN_USEFUL_GENERATION_TPS
    ]
    max_tested = max((observation["context_tokens"] for observation in normalized_observations), default=0)
    max_loadable = max((observation["context_tokens"] for observation in passing), default=0)
    max_stable = max((observation["context_tokens"] for observation in stable), default=0)
    workload_recommendations = _workload_recommendations(stable)
    max_recommended = max(
        (recommendation["context_tokens"] for recommendation in workload_recommendations.values()),
        default=0,
    )
    blockers = _blockers(advertised_tokens, normalized_observations, max_recommended)
    context_score = _context_score(advertised_tokens, max_recommended, stable)
    evidence_status = _context_evidence_status(advertised_tokens, normalized_observations, max_recommended)
    return {
        "model_id": candidate["model_id"],
        "display_name": candidate.get("display_name", candidate["model_id"]),
        "advertised_context_tokens": advertised_tokens,
        "advertised_context": {
            "tokens": advertised_tokens,
            "source": candidate.get("advertised_context_source", ""),
            "confidence": "advertised" if advertised_tokens else "unknown",
        },
        "planned_context_tiers": planned_tiers,
        "max_tested_context_tokens": max_tested,
        "max_loadable_context_tokens": max_loadable,
        "max_stable_context_tokens": max_stable,
        "max_useful_context_tokens": max_stable,
        "max_recommended_context_tokens": max_recommended,
        "context_score_pct": context_score,
        "context_evidence_status": evidence_status,
        "context_interpretation": _context_interpretation(evidence_status, advertised_tokens, max_recommended),
        "status": _status(normalized_observations, max_recommended),
        "blockers": blockers,
        "observations": normalized_observations,
        "workload_recommendations": workload_recommendations,
    }


def _observations_by_model(observations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        model_id = observation.get("model_id")
        if model_id:
            grouped[str(model_id)].append(observation)
    return grouped


def _planned_tiers(advertised_tokens: int) -> list[int]:
    if not advertised_tokens:
        return list(DEFAULT_CONTEXT_TIERS)
    tiers = [tier for tier in DEFAULT_CONTEXT_TIERS if tier <= advertised_tokens]
    if advertised_tokens not in tiers:
        tiers.insert(0, advertised_tokens)
    return sorted(set(tiers), reverse=True)


def _normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_tokens": _int(observation.get("context_tokens")),
        "backend": str(observation.get("backend", "unknown")),
        "devices_used": [str(device) for device in observation.get("devices_used", [])],
        "status": str(observation.get("status", "fail")),
        "quality_score_pct": _float(observation.get("quality_score_pct")),
        "generation_tokens_per_second": _float(observation.get("generation_tokens_per_second")),
        "prompt_tokens_per_second": _float(observation.get("prompt_tokens_per_second")),
        "failure_modes": [str(mode) for mode in observation.get("failure_modes", [])],
        "workloads": [str(workload) for workload in observation.get("workloads", [])],
        "notes": str(observation.get("notes", "")),
    }


def _workload_recommendations(stable: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    recommendations: dict[str, dict[str, Any]] = {}
    for workload in sorted({workload for observation in stable for workload in observation["workloads"]}):
        candidates = [observation for observation in stable if workload in observation["workloads"]]
        if workload == "coding":
            chosen = max(candidates, key=lambda item: (item["quality_score_pct"], item["context_tokens"]))
            reason = "highest_quality_at_stable_context"
        else:
            chosen = max(candidates, key=lambda item: (item["context_tokens"], item["quality_score_pct"]))
            reason = "largest_stable_context"
        recommendations[workload] = {
            "context_tokens": chosen["context_tokens"],
            "quality_score_pct": chosen["quality_score_pct"],
            "generation_tokens_per_second": chosen["generation_tokens_per_second"],
            "reason": reason,
        }
    return recommendations


def _comparisons(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    workloads = sorted(
        {
            workload
            for model in models
            for workload in model["workload_recommendations"]
        }
    )
    for workload in workloads:
        candidates = [
            (model, model["workload_recommendations"][workload])
            for model in models
            if workload in model["workload_recommendations"]
        ]
        if len(candidates) < 2:
            continue
        winner, recommendation = max(
            candidates,
            key=lambda item: (
                item[1]["quality_score_pct"],
                item[1]["context_tokens"],
                item[1]["generation_tokens_per_second"],
            ),
        )
        comparisons.append(
            {
                "workload": workload,
                "winner_model_id": winner["model_id"],
                "context_tokens": recommendation["context_tokens"],
                "summary": (
                    f"{winner['display_name']} currently has the strongest observed {workload.replace('_', ' ')} "
                    f"context result at {recommendation['context_tokens']} tokens."
                ),
            }
        )
    return comparisons


def _next_actions(models: list[dict[str, Any]]) -> list[str]:
    actions = ["Run context taper benchmarks before treating advertised context as usable local capacity."]
    if any(model["max_recommended_context_tokens"] == 0 for model in models):
        actions.append("Start at the advertised context tier, then step down until load, stability, and quality pass.")
    if any(model["max_recommended_context_tokens"] < model["advertised_context_tokens"] for model in models):
        actions.append("Show users max recommended context separately from advertised model context.")
    return actions


def _blockers(advertised_tokens: int, observations: list[dict[str, Any]], max_recommended: int) -> list[str]:
    blockers: list[str] = []
    if advertised_tokens and not observations:
        blockers.append("advertised_context_unproven")
    if observations and not max_recommended:
        blockers.append("no_stable_useful_context")
    if max_recommended and advertised_tokens and max_recommended < advertised_tokens:
        blockers.append("recommended_context_below_advertised")
    return blockers


def _context_score(advertised_tokens: int, max_recommended: int, stable: list[dict[str, Any]]) -> float:
    if not max_recommended:
        return 0
    best_quality = max((observation["quality_score_pct"] for observation in stable), default=0)
    if not advertised_tokens:
        coverage = 0.5
    else:
        coverage = min(max_recommended / advertised_tokens, 1.0)
    return round((best_quality * 0.75) + (coverage * 25), 2)


def _context_evidence_status(
    advertised_tokens: int,
    observations: list[dict[str, Any]],
    max_recommended: int,
) -> str:
    if max_recommended:
        return "observed_useful"
    if observations:
        return "observed_limited"
    if advertised_tokens:
        return "advertised_unproven"
    return "not_advertised"


def _context_interpretation(evidence_status: str, advertised_tokens: int, max_recommended: int) -> str:
    if evidence_status == "observed_useful":
        return (
            f"Recommended context is {max_recommended} tokens based on local observations; "
            f"advertised context remains {advertised_tokens} tokens."
        )
    if evidence_status == "observed_limited":
        return "Context was tested locally, but no stable useful context tier has passed yet."
    if evidence_status == "advertised_unproven":
        return "Advertised context is unproven locally; this is not a performance finding."
    return "No advertised context claim is available for this model."


def _status(observations: list[dict[str, Any]], max_recommended: int) -> str:
    if not observations:
        return "needs_context_benchmark"
    if max_recommended:
        return "observed_context"
    if any(observation["status"] in {"oom", "timeout", "crash"} for observation in observations):
        return "context_limited"
    return "not_recommended"


def _int(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
