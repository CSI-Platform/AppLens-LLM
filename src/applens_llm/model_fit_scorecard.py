from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.fit_report import load_machine_profile
from applens_llm.schemas import validate_payload
from applens_llm.workload_profile import load_workload_profile


SCORING_WEIGHTS = {
    "capacity_fit": 18,
    "speed_latency": 13,
    "stability": 10,
    "role_fit": 10,
    "quality_size": 5,
    "operational_readiness": 10,
    "evidence_confidence": 5,
    "agent_capability": 24,
    "context_evidence": 5,
}


def build_model_fit_scorecard(
    *,
    machine_profile: dict[str, Any],
    model_candidates: list[dict[str, Any]] | None = None,
    benchmark_records: list[dict[str, Any]] | None = None,
    benchmark_suite_results: list[dict[str, Any]] | None = None,
    experiment_summaries: list[dict[str, Any]] | None = None,
    capability_records: list[dict[str, Any]] | None = None,
    context_envelopes: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
    scorecard_id: str | None = None,
    workload_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = list(model_candidates or [])
    benchmarks = benchmark_records or []
    suite_results = benchmark_suite_results or []
    summaries = experiment_summaries or []
    capabilities = capability_records or []
    context_index = _context_by_model(context_envelopes or [])
    suite_index = _suite_results_by_model(suite_results)
    observations = _observations_by_model(summaries, benchmarks)
    capability_index = _capabilities_by_model(capabilities)
    candidates = _merge_observed_candidates(candidates, observations, suite_index)
    lane_index = _lane_index(summaries, benchmarks)
    rankings = [
        _score_candidate(
            candidate,
            observations.get(_candidate_observed_label(candidate), []),
            suite_index.get(_candidate_observed_label(candidate), []),
            capability_index.get(_candidate_observed_label(candidate), []),
            context_index.get(_candidate_observed_label(candidate)),
            lane_index,
            machine_profile,
        )
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
        "benchmark_suites": _benchmark_suite_summaries(suite_results),
        "evidence": {
            "experiment_summary_count": len(summaries),
            "benchmark_record_count": len(benchmarks),
            "benchmark_suite_result_count": len(suite_results),
            "capability_record_count": len(capabilities),
            "candidate_model_count": len(candidates),
        },
        "next_actions": _next_actions(rankings),
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }
    if workload_profile:
        scorecard["workload"] = {
            "workload_id": workload_profile["workload_id"],
            "roles": workload_profile.get("model_role_needs", []),
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
    benchmark_suite_result_paths: list[Path] | None = None,
    experiment_summary_paths: list[Path] | None = None,
    capability_record_paths: list[Path] | None = None,
    context_envelope_paths: list[Path] | None = None,
    workload_profile_path: Path | None = None,
    created_at: str | None = None,
    scorecard_id: str | None = None,
) -> dict[str, Any]:
    scorecard = build_model_fit_scorecard(
        machine_profile=load_machine_profile(machine_profile_path, machine_id=machine_id),
        model_candidates=load_model_candidates(model_candidates_path) if model_candidates_path else None,
        benchmark_records=[_load_json(path) for path in benchmark_record_paths or []],
        benchmark_suite_results=[_load_json(path) for path in benchmark_suite_result_paths or []],
        experiment_summaries=[_load_json(path) for path in experiment_summary_paths or []],
        capability_records=[_load_json(path) for path in capability_record_paths or []],
        context_envelopes=[_load_json(path) for path in context_envelope_paths or []],
        workload_profile=load_workload_profile(workload_profile_path) if workload_profile_path else None,
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


def _capabilities_by_model(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        model = record.get("model") or {}
        model_id = model.get("model_id")
        if model_id:
            grouped[str(model_id)].append(record)
        display_name = model.get("display_name")
        if display_name and display_name != model_id:
            grouped[str(display_name)].append(record)
    return grouped


def _context_by_model(envelopes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for envelope in envelopes:
        for model in envelope.get("models", []):
            model_id = model.get("model_id")
            if model_id:
                index[str(model_id)] = model
    return index


def _suite_results_by_model(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        model = record.get("model") or {}
        model_id = model.get("model_id")
        if model_id:
            grouped[str(model_id)].append(record)
        display_name = model.get("display_name")
        if display_name and display_name != model_id:
            grouped[str(display_name)].append(record)
    return grouped


def _suite_model_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    model = records[0].get("model") or {}
    return {
        "display_name": model.get("display_name"),
        "family": model.get("family"),
        "parameter_size_b": _float(model.get("parameter_size_b")),
        "quantization": model.get("quantization"),
    }


def _suite_primary_model_labels(suite_index: dict[str, list[dict[str, Any]]]) -> set[str]:
    labels = set()
    for records in suite_index.values():
        for record in records:
            model_id = (record.get("model") or {}).get("model_id")
            if model_id:
                labels.add(str(model_id))
    return labels


def _benchmark_suite_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_benchmark_suite_summary(record) for record in records]


def _benchmark_suite_summary(record: dict[str, Any]) -> dict[str, Any]:
    model = record.get("model") or {}
    runtime = record.get("runtime_lane") or {}
    condition = record.get("machine_condition") or {}
    summary = record.get("summary") or {}
    return {
        "suite_run_id": str(record.get("suite_run_id", "unknown-suite")),
        "model_id": str(model.get("model_id") or model.get("display_name") or "unknown-model"),
        "display_name": str(model.get("display_name") or model.get("model_id") or "unknown model"),
        "suite_id": str((record.get("suite") or {}).get("suite_id") or "unknown"),
        "status": str(record.get("status", "pending")),
        "backend": str(runtime.get("backend", "unknown")),
        "accelerator_ids": [str(item) for item in runtime.get("accelerator_ids", [])],
        "condition_id": str(condition.get("condition_id", "unknown")),
        "total": _int(summary.get("total")),
        "passed": _int(summary.get("passed")),
        "failed": _int(summary.get("failed")),
        "unsupported": _int(summary.get("unsupported")),
        "pending": _int(summary.get("pending")),
        "errored": _int(summary.get("errored")),
        "task_statuses": [
            {
                "task_id": str(task.get("task_id", "unknown")),
                "benchmark": str(task.get("benchmark", "unknown")),
                "category": str(task.get("category", "unknown")),
                "status": str(task.get("status", "unknown")),
                "metric_summary": _metric_summary(task.get("metrics") or {}),
                "local_metric_summary": _metric_summary(task.get("local_metrics") or {}),
                "notes": str(task.get("notes", "")),
            }
            for task in record.get("task_results", [])
        ],
    }


def _merge_observed_candidates(
    candidates: list[dict[str, Any]],
    observations: dict[str, list[dict[str, Any]]],
    suite_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    seen = {_candidate_observed_label(candidate) for candidate in candidates}
    merged = list(candidates)
    for model_label in sorted(set(observations) | _suite_primary_model_labels(suite_index)):
        if model_label in seen:
            continue
        observed_role = _role_from_observation(observations[model_label])
        if observed_role == "unknown":
            observed_role = "general_chat"
        suite_model = _suite_model_summary(suite_index.get(model_label, []))
        merged.append(
            {
                "model_id": model_label,
                "display_name": suite_model.get("display_name") or model_label,
                "family": suite_model.get("family") or _guess_family(model_label),
                "parameter_size_b": suite_model.get("parameter_size_b") or _guess_parameter_size(model_label),
                "quantization": suite_model.get("quantization") or _guess_quantization(model_label),
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
    suite_results: list[dict[str, Any]],
    capability_records: list[dict[str, Any]],
    context_model: dict[str, Any] | None,
    lane_index: dict[str, dict[str, Any]],
    machine_profile: dict[str, Any],
) -> dict[str, Any]:
    role = _recommended_role(candidate, observations)
    best_observation = _best_observation(observations)
    best_lane = _best_lane(candidate, best_observation, suite_results, lane_index, machine_profile, role)
    breakdown = {
        "capacity_fit": _capacity_score(candidate, observations, best_lane, machine_profile),
        "speed_latency": _speed_score(best_observation, role, candidate),
        "stability": _stability_score(observations, suite_results),
        "role_fit": _role_score(candidate, role, best_observation),
        "quality_size": _quality_score(candidate, role, suite_results),
        "operational_readiness": _readiness_score(candidate, observations, suite_results, best_lane),
        "evidence_confidence": _confidence_score(observations, suite_results),
        "agent_capability": _agent_capability_score(capability_records),
        "context_evidence": _context_evidence_score(context_model),
    }
    fit_score = sum(breakdown.values())
    blockers = _blockers(candidate, observations, suite_results, best_lane, fit_score)
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
        "confidence": "observed" if observations or suite_results or capability_records else "inferred",
        "reasons": _reasons(candidate, role, best_lane, breakdown, observations, suite_results, capability_records, context_model),
        "blockers": blockers,
        "evidence": _evidence_summary(observations, suite_results, capability_records, context_model),
        "next_benchmark": _next_benchmark(candidate, role, observations, suite_results, best_lane, capability_records, context_model),
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
    suite_results: list[dict[str, Any]],
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
    if suite_results:
        suite = suite_results[-1]
        runtime = suite.get("runtime_lane") or {}
        return {
            "lane_id": _benchmark_lane_id(runtime.get("backend", "unknown"), runtime.get("accelerator_ids") or []),
            "backend": runtime.get("backend", "unknown"),
            "accelerator_ids": runtime.get("accelerator_ids", []),
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
        return 18
    if observations:
        return 5
    file_size = _float(candidate.get("file_size_mb"))
    if not file_size:
        return 10
    capacity = _lane_capacity(best_lane, machine_profile)
    if not capacity:
        return 6
    ratio = file_size / capacity
    if ratio <= 0.65:
        return 16
    if ratio <= 0.85:
        return 14
    if ratio <= 1.0:
        return 10
    return 5


def _speed_score(best_observation: dict[str, Any] | None, role: str, candidate: dict[str, Any]) -> int:
    if not best_observation:
        params = _float(candidate.get("parameter_size_b"))
        return 9 if params <= 5 else 4
    if best_observation.get("outcome") != "success":
        return 0
    latency = best_observation["latency_ms"]
    if role == "fast_chat":
        if latency <= 3000:
            return 13
        if latency <= 8000:
            return 10
        return 5
    if latency <= 15000:
        return 10
    if latency <= 30000:
        return 6
    return 3


def _stability_score(observations: list[dict[str, Any]], suite_results: list[dict[str, Any]]) -> int:
    if not observations and not suite_results:
        return 5
    if suite_results and not observations:
        failed = sum(_suite_count(result, "failed") + _suite_count(result, "errored") for result in suite_results)
        passed = sum(_suite_count(result, "passed") for result in suite_results)
        if failed:
            return 4
        if passed:
            return 8
        return 5
    successes = sum(1 for observation in observations if observation.get("outcome") == "success")
    if successes == len(observations):
        return 10
    if successes:
        return 6
    return 2


def _role_score(candidate: dict[str, Any], role: str, best_observation: dict[str, Any] | None) -> int:
    if role in (candidate.get("preferred_roles") or []):
        return 10
    if best_observation:
        return 8
    return 5


def _quality_score(candidate: dict[str, Any], role: str, suite_results: list[dict[str, Any]]) -> int:
    suite_passed = sum(_suite_count(result, "passed") for result in suite_results)
    if suite_passed >= 3:
        return 5
    if suite_passed:
        return 4
    prior = candidate.get("quality_prior", "unknown")
    if prior == "high":
        return 5
    if prior == "medium":
        return 4
    if prior == "low":
        return 2
    params = _float(candidate.get("parameter_size_b"))
    if role == "deep_review" and params >= 20:
        return 4
    if params <= 8:
        return 3
    return 3


def _readiness_score(
    candidate: dict[str, Any],
    observations: list[dict[str, Any]],
    suite_results: list[dict[str, Any]],
    best_lane: dict[str, Any],
) -> int:
    if suite_results and any(_suite_count(result, "passed") > 0 for result in suite_results):
        return 10
    if observations:
        if any(observation.get("outcome") == "success" for observation in observations):
            return 10
        return 2
    if candidate.get("local_status") == "local" and best_lane["lane_id"] != "benchmark-required":
        return 8
    if candidate.get("local_status") == "candidate":
        return 6
    return 2


def _confidence_score(observations: list[dict[str, Any]], suite_results: list[dict[str, Any]]) -> int:
    successes = sum(1 for observation in observations if observation.get("outcome") == "success")
    suite_successes = sum(_suite_count(result, "passed") for result in suite_results)
    if suite_successes >= 3:
        return 5
    if successes >= 3:
        return 5
    if successes or suite_successes:
        return 4
    if observations or suite_results:
        return 1
    return 2


def _agent_capability_score(capability_records: list[dict[str, Any]]) -> int:
    if not capability_records:
        return 8
    best = max(_capability_score_pct(record) for record in capability_records)
    return min(24, round(best * 24 / 100))


def _context_evidence_score(context_model: dict[str, Any] | None) -> int:
    if not context_model:
        return 0
    if context_model.get("context_evidence_status") == "advertised_unproven":
        return 0
    score_pct = _float(context_model.get("context_score_pct"))
    return min(5, round(score_pct * 5 / 100))


def _blockers(
    candidate: dict[str, Any],
    observations: list[dict[str, Any]],
    suite_results: list[dict[str, Any]],
    best_lane: dict[str, Any],
    fit_score: int,
) -> list[str]:
    blockers: list[str] = []
    if not observations and not suite_results:
        blockers.append("no_observed_benchmark")
    elif not any(observation.get("outcome") == "success" for observation in observations):
        if any((_suite_count(result, "passed") > 0) for result in suite_results):
            pass
        else:
            blockers.append("observed_failure")
    if any(result.get("status") == "partial" for result in suite_results):
        blockers.append("official_suite_partial")
    if any(result.get("status") in {"fail", "blocked"} for result in suite_results):
        blockers.append("official_suite_blocked")
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
    suite_results: list[dict[str, Any]],
    capability_records: list[dict[str, Any]],
    context_model: dict[str, Any] | None,
) -> list[str]:
    reasons = [
        f"Best current role is {role.replace('_', ' ')}.",
        f"Best lane is {best_lane['lane_id']} on {best_lane['backend']}.",
    ]
    if observations:
        reasons.append(f"{len(observations)} observed run(s) are available for this model label.")
    elif suite_results:
        passed = sum(_suite_count(result, "passed") for result in suite_results)
        unsupported = sum(_suite_count(result, "unsupported") for result in suite_results)
        reasons.append(f"Official benchmark-suite evidence is attached: {passed} passed task(s), {unsupported} unsupported task(s).")
    else:
        reasons.append("Score is inferred until this model has a direct benchmark.")
    if breakdown["capacity_fit"] >= 16:
        reasons.append("Capacity fit is strong for the selected lane.")
    if breakdown["speed_latency"] >= 13:
        reasons.append("Latency evidence is suitable for the recommended role.")
    elif breakdown["speed_latency"] <= 8:
        reasons.append("Latency is the main practical constraint.")
    if capability_records:
        best_capability = max(_capability_score_pct(record) for record in capability_records)
        reasons.append(f"AppLens local capability score is {best_capability}/100.")
    else:
        reasons.append("Agent capability is unproven until applens-local-v1 is run.")
    if context_model:
        reasons.append(_context_reason(context_model))
    else:
        reasons.append("Context envelope is unproven until taper benchmarks are run.")
    if candidate.get("quality_prior") == "high":
        reasons.append("Quality prior is high for its size class.")
    return reasons


def _context_reason(context_model: dict[str, Any]) -> str:
    status = context_model.get("context_evidence_status")
    if status == "observed_useful":
        return (
            f"Recommended context is {context_model.get('max_recommended_context_tokens', 0)} tokens "
            f"against advertised {context_model.get('advertised_context_tokens', 0)} tokens."
        )
    if status == "observed_limited":
        return "Context load evidence exists, but useful task quality is not proven yet."
    if status == "advertised_unproven":
        return "Advertised context is unproven locally; this is not a performance finding."
    return "Context evidence is unknown until taper benchmarks are run."


def _evidence_summary(
    observations: list[dict[str, Any]],
    suite_results: list[dict[str, Any]],
    capability_records: list[dict[str, Any]],
    context_model: dict[str, Any] | None,
) -> dict[str, Any]:
    suite_counts = _suite_counts(suite_results)
    suite_unsupported_tasks = _suite_tasks_with_status(suite_results, "unsupported")
    if not observations and not suite_results and not capability_records:
        payload = {
            "source": "inferred_from_inventory",
            "observation_count": 0,
            "benchmark_suite_result_count": 0,
            "benchmark_suite_passed": 0,
            "benchmark_suite_failed": 0,
            "benchmark_suite_unsupported": 0,
            "benchmark_suite_unsupported_tasks": [],
            "capability_record_count": 0,
            "thinking_modes": [],
        }
        _attach_context_evidence(payload, context_model)
        return payload
    latencies = [observation["latency_ms"] for observation in observations]
    tokens = [observation["total_tokens"] for observation in observations]
    sources = sorted({observation.get("source", "unknown") for observation in observations})
    if suite_results:
        sources.append("benchmark_suite_result")
    if capability_records:
        sources.append("local_capability_record")
    payload = {
        "source": "+".join(sources),
        "observation_count": len(observations),
        "benchmark_suite_result_count": len(suite_results),
        "benchmark_suite_passed": suite_counts["passed"],
        "benchmark_suite_failed": suite_counts["failed"] + suite_counts["errored"],
        "benchmark_suite_unsupported": suite_counts["unsupported"],
        "benchmark_suite_unsupported_tasks": suite_unsupported_tasks,
        "capability_record_count": len(capability_records),
        "thinking_modes": sorted(
            {
                str((record.get("model") or {}).get("thinking_mode", "unknown"))
                for record in capability_records
            }
        ),
    }
    if observations:
        payload["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
        payload["avg_total_tokens"] = round(sum(tokens) / len(tokens), 2)
    if capability_records:
        payload["capability_score_pct"] = max(_capability_score_pct(record) for record in capability_records)
        payload["capability_categories"] = _capability_categories(capability_records)
    _attach_context_evidence(payload, context_model)
    return payload


def _next_benchmark(
    candidate: dict[str, Any],
    role: str,
    observations: list[dict[str, Any]],
    suite_results: list[dict[str, Any]],
    best_lane: dict[str, Any],
    capability_records: list[dict[str, Any]],
    context_model: dict[str, Any] | None,
) -> str:
    if suite_results and sum(_suite_count(result, "unsupported") for result in suite_results):
        return "Install or wire unsupported official benchmark runners, then rerun the benchmark suite."
    if not observations and not suite_results:
        return f"Run a direct {role.replace('_', ' ')} benchmark on {best_lane['lane_id']}."
    if not capability_records:
        return "Run applens-local-v1 capability eval for JSON, tool, coding, hardware, safety, and handoff behavior."
    if context_model and context_model.get("context_evidence_status") == "observed_limited":
        return "Run context quality checks against the largest load-tested tier."
    if not context_model or not context_model.get("max_recommended_context_tokens"):
        return "Run context taper benchmarks to prove max useful context for this model and lane."
    if len(observations) < 3:
        return "Run at least three repeat benchmarks with fixed prompt and token caps."
    return "Run task-specific quality checks for the intended role."


def _next_actions(rankings: list[dict[str, Any]]) -> list[str]:
    actions = ["Benchmark the top unobserved candidate before making it the default."]
    if any(row["evidence"].get("benchmark_suite_unsupported", 0) for row in rankings):
        actions.append("Wire unsupported official benchmark runners before treating suite scores as complete.")
    if any(not row["evidence"].get("capability_record_count") for row in rankings):
        actions.append("Run applens-local-v1 on top candidates before using them for agentic local work.")
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


def _capability_score_pct(record: dict[str, Any]) -> float:
    score = ((record.get("scores") or {}).get("score_pct"))
    return _float(score)


def _suite_count(record: dict[str, Any], key: str) -> int:
    return _int((record.get("summary") or {}).get(key))


def _suite_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    keys = ["total", "passed", "failed", "unsupported", "pending", "errored"]
    return {key: sum(_suite_count(record, key) for record in records) for key in keys}


def _suite_tasks_with_status(records: list[dict[str, Any]], status: str) -> list[str]:
    task_ids = {
        str(task.get("task_id"))
        for record in records
        for task in record.get("task_results", [])
        if task.get("status") == status and task.get("task_id")
    }
    return sorted(task_ids)


def _metric_summary(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    parts = []
    for key in sorted(metrics)[:3]:
        value = metrics[key]
        if isinstance(value, float):
            value = round(value, 4)
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def _capability_categories(records: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for record in records:
        for category, score in ((record.get("scores") or {}).get("category_scores") or {}).items():
            best[category] = max(best.get(category, 0), _float(score.get("score_pct")))
    return dict(sorted(best.items()))


def _attach_context_evidence(payload: dict[str, Any], context_model: dict[str, Any] | None) -> None:
    if not context_model:
        payload["advertised_context_tokens"] = 0
        payload["max_tested_context_tokens"] = 0
        payload["recommended_context_tokens"] = 0
        payload["context_score_pct"] = 0
        payload["context_evidence_status"] = "unknown"
        payload["context_interpretation"] = "No context envelope is attached."
        return
    payload["advertised_context_tokens"] = _int(context_model.get("advertised_context_tokens"))
    payload["max_tested_context_tokens"] = _int(context_model.get("max_tested_context_tokens"))
    payload["recommended_context_tokens"] = _int(context_model.get("max_recommended_context_tokens"))
    payload["context_score_pct"] = _float(context_model.get("context_score_pct"))
    payload["context_evidence_status"] = str(context_model.get("context_evidence_status", "unknown"))
    payload["context_interpretation"] = str(context_model.get("context_interpretation", ""))


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
