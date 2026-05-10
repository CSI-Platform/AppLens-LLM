from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.fit_report import load_machine_profile
from applens_llm.schemas import validate_payload


SCORING_WEIGHTS = {
    "capacity_fit": 25,
    "speed_latency": 20,
    "stability": 15,
    "role_fit": 15,
    "quality_size": 10,
    "operational_readiness": 10,
    "evidence_confidence": 5,
}


def build_model_fit_scorecard(
    *,
    machine_profile: dict[str, Any],
    model_candidates: list[dict[str, Any]] | None = None,
    benchmark_records: list[dict[str, Any]] | None = None,
    experiment_summaries: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
    scorecard_id: str | None = None,
    ) -> dict[str, Any]:
    candidates = list(model_candidates or [])
    benchmarks = benchmark_records or []
    summaries = experiment_summaries or []
    observations = _observations_by_model(summaries, benchmarks)
    candidates = _merge_observed_candidates(candidates, observations)
    lane_index = _lane_index(summaries, benchmarks)
    rankings = [
        _score_candidate(candidate, observations.get(_candidate_observed_label(candidate), []), lane_index, machine_profile)
        for candidate in candidates
    ]
    rankings.sort(key=lambda row: (-row["fit_score"], row["model_id"]))
    for index, row in enumerate(rankings, start=1):
        row["rank"] = index

    scorecard = {
        "schema_version": "0.1",
        "scorecard_id": scorecard_id or f"scorecard-{machine_profile['machine_id']}",
        "created_at": created_at or _utc_now(),
        "machine": _machine_summary(machine_profile),
        "scoring_weights": SCORING_WEIGHTS,
        "rankings": rankings,
        "evidence": {
            "experiment_summary_count": len(summaries),
            "benchmark_record_count": len(benchmarks),
            "candidate_model_count": len(candidates),
        },
        "next_actions": _next_actions(rankings),
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }
    validate_payload("model-fit-scorecard", scorecard)
    return scorecard


def write_model_fit_scorecard(
    *,
    machine_profile_path: Path,
    output_path: Path,
    machine_id: str | None = None,
    model_candidates_path: Path | None = None,
    benchmark_record_paths: list[Path] | None = None,
    experiment_summary_paths: list[Path] | None = None,
    created_at: str | None = None,
    scorecard_id: str | None = None,
) -> dict[str, Any]:
    scorecard = build_model_fit_scorecard(
        machine_profile=load_machine_profile(machine_profile_path, machine_id=machine_id),
        model_candidates=load_model_candidates(model_candidates_path) if model_candidates_path else None,
        benchmark_records=[_load_json(path) for path in benchmark_record_paths or []],
        experiment_summaries=[_load_json(path) for path in experiment_summary_paths or []],
        created_at=created_at,
        scorecard_id=scorecard_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return scorecard


def load_model_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("models", []))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"unsupported model candidate file shape: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _machine_summary(machine_profile: dict[str, Any]) -> dict[str, Any]:
    platform = machine_profile["platform"]
    return {
        "machine_id": machine_profile["machine_id"],
        "label": machine_profile["label"],
        "platform": {
            "cpu": platform["cpu"],
            "gpu": platform["gpu"],
            "ram_gb": platform["ram_gb"],
            "os_family": platform["os_family"],
        },
    }


def _lane_index(
    summaries: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lanes: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        lane_roles = {lane_id: role for role, lane_id in (summary.get("lanes") or {}).items()}
        for started in (summary.get("lifecycle") or {}).get("started", []):
            lane_id = started.get("lane_id")
            if lane_id:
                lanes[lane_id] = {
                    "lane_id": lane_id,
                    "backend": started.get("backend", "unknown"),
                    "accelerator_ids": started.get("accelerator_ids", []),
                    "role": lane_roles.get(lane_id, "unknown"),
                    "model_label": started.get("model_label", "unknown"),
                }
    for record in benchmarks:
        runtime = record.get("runtime") or {}
        devices = runtime.get("devices_used") or []
        backend = runtime.get("backend", "unknown")
        lane_id = _benchmark_lane_id(backend, devices)
        lanes[lane_id] = {
            "lane_id": lane_id,
            "backend": backend,
            "accelerator_ids": devices,
            "role": "general_chat",
            "model_label": (record.get("model") or {}).get("name", "unknown"),
        }
    return lanes


def _observations_by_model(
    summaries: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        lane_models = {
            started.get("lane_id"): started.get("model_label")
            for started in (summary.get("lifecycle") or {}).get("started", [])
        }
        lane_roles = {lane_id: role for role, lane_id in (summary.get("lanes") or {}).items()}
        for response in (summary.get("responses") or {}).values():
            lane_id = response.get("lane_id")
            model_label = lane_models.get(lane_id)
            if not model_label:
                continue
            observations[model_label].append(
                {
                    "model_label": model_label,
                    "lane_id": lane_id,
                    "role": lane_roles.get(lane_id, "unknown"),
                    "outcome": response.get("outcome", "unknown"),
                    "latency_ms": _int(response.get("latency_ms")),
                    "total_tokens": _int((response.get("usage") or {}).get("total_tokens")),
                    "source": "experiment_summary",
                }
            )
    for record in benchmarks:
        model_label = (record.get("model") or {}).get("name")
        if not model_label:
            continue
        runtime = record.get("runtime") or {}
        workload = record.get("workload") or {}
        observations[model_label].append(
            {
                "model_label": model_label,
                "lane_id": _benchmark_lane_id(runtime.get("backend", "unknown"), runtime.get("devices_used") or []),
                "role": "general_chat",
                "outcome": "success" if (record.get("outcome") or {}).get("status") == "pass" else "failure",
                "latency_ms": _int((record.get("metrics") or {}).get("latency_ms")),
                "total_tokens": _int(workload.get("prompt_tokens")) + _int(workload.get("completion_tokens")),
                "source": "benchmark_record",
            }
        )
    return observations


def _merge_observed_candidates(
    candidates: list[dict[str, Any]],
    observations: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    seen = {_candidate_observed_label(candidate) for candidate in candidates}
    merged = list(candidates)
    for model_label in observations:
        if model_label in seen:
            continue
        observed_role = _role_from_observation(observations[model_label])
        if observed_role == "unknown":
            observed_role = "general_chat"
        merged.append(
            {
                "model_id": model_label,
                "display_name": model_label,
                "family": _guess_family(model_label),
                "parameter_size_b": _guess_parameter_size(model_label),
                "quantization": _guess_quantization(model_label),
                "file_size_mb": 0,
                "local_status": "local",
                "preferred_roles": [observed_role],
                "quality_prior": "unknown",
                "observed_model_label": model_label,
            }
        )
    return merged


def _score_candidate(
    candidate: dict[str, Any],
    observations: list[dict[str, Any]],
    lane_index: dict[str, dict[str, Any]],
    machine_profile: dict[str, Any],
) -> dict[str, Any]:
    role = _recommended_role(candidate, observations)
    best_observation = _best_observation(observations)
    best_lane = _best_lane(candidate, best_observation, lane_index, machine_profile, role)
    breakdown = {
        "capacity_fit": _capacity_score(candidate, observations, best_lane, machine_profile),
        "speed_latency": _speed_score(best_observation, role, candidate),
        "stability": _stability_score(observations),
        "role_fit": _role_score(candidate, role, best_observation),
        "quality_size": _quality_score(candidate, role),
        "operational_readiness": _readiness_score(candidate, observations, best_lane),
        "evidence_confidence": _confidence_score(observations),
    }
    fit_score = sum(breakdown.values())
    blockers = _blockers(candidate, observations, best_lane, fit_score)
    return {
        "rank": 0,
        "model_id": candidate["model_id"],
        "display_name": candidate.get("display_name", candidate["model_id"]),
        "family": candidate.get("family", "unknown"),
        "parameter_size_b": _float(candidate.get("parameter_size_b")),
        "quantization": candidate.get("quantization", "unknown"),
        "local_status": candidate.get("local_status", "unknown"),
        "recommended_role": role,
        "best_lane": best_lane,
        "fit_score": fit_score,
        "score_band": _score_band(fit_score),
        "score_breakdown": breakdown,
        "confidence": "observed" if observations else "inferred",
        "reasons": _reasons(candidate, role, best_lane, breakdown, observations),
        "blockers": blockers,
        "evidence": _evidence_summary(observations),
        "next_benchmark": _next_benchmark(candidate, role, observations, best_lane),
    }


def _candidate_observed_label(candidate: dict[str, Any]) -> str:
    return str(candidate.get("observed_model_label") or candidate.get("model_id"))


def _recommended_role(candidate: dict[str, Any], observations: list[dict[str, Any]]) -> str:
    valid_roles = {"fast_chat", "deep_review", "coding", "summarization", "general_chat", "not_recommended"}
    roles = candidate.get("preferred_roles") or []
    if roles and roles[0] in valid_roles:
        return roles[0]
    observed_role = _role_from_observation(observations)
    return observed_role if observed_role in valid_roles else "general_chat"


def _role_from_observation(observations: list[dict[str, Any]]) -> str:
    roles = [observation.get("role") for observation in observations]
    if "fast_chat" in roles:
        return "fast_chat"
    if "deep_review" in roles:
        return "deep_review"
    if "general_chat" in roles:
        return "general_chat"
    if "fast" in roles:
        return "fast_chat"
    if "deep" in roles:
        return "deep_review"
    return "unknown"


def _best_observation(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    successes = [observation for observation in observations if observation.get("outcome") == "success"]
    if not successes:
        return observations[-1] if observations else None
    return min(successes, key=lambda observation: observation["latency_ms"])


def _best_lane(
    candidate: dict[str, Any],
    best_observation: dict[str, Any] | None,
    lane_index: dict[str, dict[str, Any]],
    machine_profile: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    if best_observation:
        lane = lane_index.get(best_observation["lane_id"], {})
        return {
            "lane_id": best_observation["lane_id"],
            "backend": lane.get("backend", "unknown"),
            "accelerator_ids": lane.get("accelerator_ids", []),
            "role": role,
            "model_label": candidate.get("observed_model_label", candidate["model_id"]),
        }
    lane = _inferred_lane(candidate, machine_profile, role)
    return {
        "lane_id": lane["lane_id"],
        "backend": lane["backend"],
        "accelerator_ids": lane["accelerator_ids"],
        "role": role,
        "model_label": candidate["model_id"],
    }


def _inferred_lane(candidate: dict[str, Any], machine_profile: dict[str, Any], role: str) -> dict[str, Any]:
    file_size = _float(candidate.get("file_size_mb"))
    accelerators = {
        accelerator["accelerator_id"]: accelerator
        for accelerator in machine_profile["hardware_topology"].get("accelerators", [])
    }
    nvidia = accelerators.get("nvidia-dgpu-0")
    amd = accelerators.get("amd-igpu-0")
    nvidia_capacity = _accelerator_capacity(nvidia)
    amd_capacity = _accelerator_capacity(amd)
    if nvidia and file_size and file_size <= nvidia_capacity * 0.85:
        return {"lane_id": "fast-nvidia", "backend": "cuda", "accelerator_ids": ["nvidia-dgpu-0"]}
    if amd and file_size and file_size <= amd_capacity * 0.9:
        return {"lane_id": "deep-amd-vgm", "backend": "vulkan", "accelerator_ids": ["amd-igpu-0"]}
    if role == "fast_chat" and nvidia:
        return {"lane_id": "fast-nvidia", "backend": "cuda", "accelerator_ids": ["nvidia-dgpu-0"]}
    return {"lane_id": "benchmark-required", "backend": "unknown", "accelerator_ids": []}


def _capacity_score(
    candidate: dict[str, Any],
    observations: list[dict[str, Any]],
    best_lane: dict[str, Any],
    machine_profile: dict[str, Any],
) -> int:
    if any(observation.get("outcome") == "success" for observation in observations):
        return 25
    if observations:
        return 5
    file_size = _float(candidate.get("file_size_mb"))
    if not file_size:
        return 10
    capacity = _lane_capacity(best_lane, machine_profile)
    if not capacity:
        return 8
    ratio = file_size / capacity
    if ratio <= 0.65:
        return 23
    if ratio <= 0.85:
        return 20
    if ratio <= 1.0:
        return 15
    return 5


def _speed_score(best_observation: dict[str, Any] | None, role: str, candidate: dict[str, Any]) -> int:
    if not best_observation:
        params = _float(candidate.get("parameter_size_b"))
        return 13 if params <= 5 else 6
    if best_observation.get("outcome") != "success":
        return 0
    latency = best_observation["latency_ms"]
    if role == "fast_chat":
        if latency <= 3000:
            return 20
        if latency <= 8000:
            return 14
        return 6
    if latency <= 15000:
        return 14
    if latency <= 30000:
        return 9
    return 4


def _stability_score(observations: list[dict[str, Any]]) -> int:
    if not observations:
        return 7
    successes = sum(1 for observation in observations if observation.get("outcome") == "success")
    if successes == len(observations):
        return 15
    if successes:
        return 9
    return 2


def _role_score(candidate: dict[str, Any], role: str, best_observation: dict[str, Any] | None) -> int:
    if role in (candidate.get("preferred_roles") or []):
        return 15
    if best_observation:
        return 12
    return 8


def _quality_score(candidate: dict[str, Any], role: str) -> int:
    prior = candidate.get("quality_prior", "unknown")
    if prior == "high":
        return 10
    if prior == "medium":
        return 8
    if prior == "low":
        return 5
    params = _float(candidate.get("parameter_size_b"))
    if role == "deep_review" and params >= 20:
        return 8
    if params <= 8:
        return 7
    return 6


def _readiness_score(
    candidate: dict[str, Any],
    observations: list[dict[str, Any]],
    best_lane: dict[str, Any],
) -> int:
    if observations:
        if any(observation.get("outcome") == "success" for observation in observations):
            return 10
        return 2
    if candidate.get("local_status") == "local" and best_lane["lane_id"] != "benchmark-required":
        return 8
    if candidate.get("local_status") == "candidate":
        return 6
    return 2


def _confidence_score(observations: list[dict[str, Any]]) -> int:
    successes = sum(1 for observation in observations if observation.get("outcome") == "success")
    if successes >= 3:
        return 5
    if successes:
        return 4
    if observations:
        return 1
    return 2


def _blockers(
    candidate: dict[str, Any],
    observations: list[dict[str, Any]],
    best_lane: dict[str, Any],
    fit_score: int,
) -> list[str]:
    blockers: list[str] = []
    if not observations:
        blockers.append("no_observed_benchmark")
    elif not any(observation.get("outcome") == "success" for observation in observations):
        blockers.append("observed_failure")
    if candidate.get("local_status") in {"candidate", "missing"}:
        blockers.append(f"model_status_{candidate.get('local_status')}")
    if best_lane["lane_id"] == "benchmark-required":
        blockers.append("no_proven_lane")
    if best_lane["backend"] == "vulkan" and "amd-igpu-0" in best_lane["accelerator_ids"]:
        blockers.append("requires_amd_vgm_vulkan_path")
    if fit_score < 60:
        blockers.append("low_fit_score")
    return blockers


def _reasons(
    candidate: dict[str, Any],
    role: str,
    best_lane: dict[str, Any],
    breakdown: dict[str, int],
    observations: list[dict[str, Any]],
) -> list[str]:
    reasons = [
        f"Best current role is {role.replace('_', ' ')}.",
        f"Best lane is {best_lane['lane_id']} on {best_lane['backend']}.",
    ]
    if observations:
        reasons.append(f"{len(observations)} observed run(s) are available for this model label.")
    else:
        reasons.append("Score is inferred until this model has a direct benchmark.")
    if breakdown["capacity_fit"] >= 20:
        reasons.append("Capacity fit is strong for the selected lane.")
    if breakdown["speed_latency"] >= 14:
        reasons.append("Latency evidence is suitable for the recommended role.")
    elif breakdown["speed_latency"] <= 8:
        reasons.append("Latency is the main practical constraint.")
    if candidate.get("quality_prior") == "high":
        reasons.append("Quality prior is high for its size class.")
    return reasons


def _evidence_summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not observations:
        return {"source": "inferred_from_inventory", "observation_count": 0}
    latencies = [observation["latency_ms"] for observation in observations]
    tokens = [observation["total_tokens"] for observation in observations]
    sources = sorted({observation.get("source", "unknown") for observation in observations})
    return {
        "source": "+".join(sources),
        "observation_count": len(observations),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "avg_total_tokens": round(sum(tokens) / len(tokens), 2),
    }


def _next_benchmark(
    candidate: dict[str, Any],
    role: str,
    observations: list[dict[str, Any]],
    best_lane: dict[str, Any],
) -> str:
    if not observations:
        return f"Run a direct {role.replace('_', ' ')} benchmark on {best_lane['lane_id']}."
    if len(observations) < 3:
        return "Run at least three repeat benchmarks with fixed prompt and token caps."
    return "Run task-specific quality checks for the intended role."


def _next_actions(rankings: list[dict[str, Any]]) -> list[str]:
    actions = ["Benchmark the top unobserved candidate before making it the default."]
    if any("requires_amd_vgm_vulkan_path" in row["blockers"] for row in rankings):
        actions.append("Keep AMD VGM/Vulkan checks in the readiness loop for capacity-lane models.")
    actions.append("Use repeated benchmark evidence before changing default model rankings.")
    return actions


def _lane_capacity(best_lane: dict[str, Any], machine_profile: dict[str, Any]) -> float:
    accelerators = {
        accelerator["accelerator_id"]: accelerator
        for accelerator in machine_profile["hardware_topology"].get("accelerators", [])
    }
    capacities = [
        _accelerator_capacity(accelerators.get(accelerator_id))
        for accelerator_id in best_lane.get("accelerator_ids", [])
    ]
    return max(capacities, default=0)


def _accelerator_capacity(accelerator: dict[str, Any] | None) -> float:
    if not accelerator:
        return 0
    memory = accelerator.get("memory", {})
    return float(
        memory.get("estimated_usable_inference_memory_mb")
        or memory.get("physical_dedicated_vram_mb")
        or memory.get("reported_total_graphics_memory_mb")
        or 0
    )


def _score_band(score: int) -> str:
    if score >= 90:
        return "excellent"
    if score >= 80:
        return "good"
    if score >= 65:
        return "usable"
    if score >= 45:
        return "experimental"
    return "not_recommended"


def _guess_family(model_label: str) -> str:
    lowered = model_label.lower()
    if "qwen" in lowered or "jan" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return "unknown"


def _guess_parameter_size(model_label: str) -> float:
    lowered = model_label.lower()
    if "27b" in lowered:
        return 27
    if "31b" in lowered:
        return 31
    if "4b" in lowered:
        return 4
    return 0


def _guess_quantization(model_label: str) -> str:
    lowered = model_label.lower()
    if "q4" in lowered:
        return "Q4"
    if "iq3" in lowered:
        return "IQ3"
    return "unknown"


def _benchmark_lane_id(backend: str, devices: list[str]) -> str:
    device_label = devices[0] if devices else "unknown-device"
    return f"benchmark-{backend}-{device_label}"


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
