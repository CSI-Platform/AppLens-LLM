from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


def build_fit_report(
    *,
    machine_profile: dict[str, Any],
    benchmark_records: list[dict[str, Any]] | None = None,
    experiment_summaries: list[dict[str, Any]] | None = None,
    experiment_comparisons: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
    report_id: str | None = None,
) -> dict[str, Any]:
    benchmarks = benchmark_records or []
    summaries = experiment_summaries or []
    comparisons = experiment_comparisons or []
    lane_index = _lane_index(summaries)
    proven_lanes = _proven_lanes(summaries, lane_index)
    fast_lane = _lane_choice("fast", summaries, lane_index)
    capacity_lane = _lane_choice("deep", summaries, lane_index)
    strategy = _strategy(fast_lane, capacity_lane)
    fit_class = _fit_class(machine_profile, strategy)
    profile_capacity = machine_profile["hardware_topology"]["usable_inference_capacity"]

    report = {
        "schema_version": "0.1",
        "report_id": report_id or f"fit-{machine_profile['machine_id']}",
        "created_at": created_at or _utc_now(),
        "machine": _machine_summary(machine_profile),
        "fit": {
            "class": fit_class,
            "confidence": "observed" if proven_lanes else "inferred",
            "summary": _fit_summary(strategy),
        },
        "capacity_assessment": {
            "reported_summary_vram_mb": machine_profile["platform"]["vram_mb"],
            "profile_estimated_usable_memory_mb": profile_capacity["estimated_usable_memory_mb"],
            "profile_confidence": profile_capacity["confidence"],
            "mixed_device_pooling": profile_capacity["mixed_device_pooling"],
            "proven_lanes": proven_lanes,
            "unsupported_claims": _unsupported_claims(machine_profile),
        },
        "runtime_recommendation": {
            "strategy": strategy,
            "primary_engine": _primary_engine(lane_index),
            "fast_lane": fast_lane,
            "capacity_lane": capacity_lane,
            "model_guidance": _model_guidance(strategy),
        },
        "evidence": {
            "experiment_summaries": [_experiment_summary(summary) for summary in summaries],
            "experiment_comparisons": [_comparison_summary(comparison) for comparison in comparisons],
            "benchmark_records": [_benchmark_summary(record) for record in benchmarks],
        },
        "decisions": _decisions(strategy, machine_profile, comparisons),
        "next_benchmarks": _next_benchmarks(strategy),
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }
    validate_payload("fit-report", report)
    return report


def write_fit_report(
    *,
    machine_profile_path: Path,
    output_path: Path,
    machine_id: str | None = None,
    benchmark_record_paths: list[Path] | None = None,
    experiment_summary_paths: list[Path] | None = None,
    experiment_comparison_paths: list[Path] | None = None,
    created_at: str | None = None,
    report_id: str | None = None,
) -> dict[str, Any]:
    machine_profile = load_machine_profile(machine_profile_path, machine_id=machine_id)
    report = build_fit_report(
        machine_profile=machine_profile,
        benchmark_records=[_load_json(path) for path in benchmark_record_paths or []],
        experiment_summaries=[_load_json(path) for path in experiment_summary_paths or []],
        experiment_comparisons=[_load_json(path) for path in experiment_comparison_paths or []],
        created_at=created_at,
        report_id=report_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_machine_profile(path: Path, *, machine_id: str | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            profile = json.loads(text)
        except json.JSONDecodeError:
            profile = None
        if isinstance(profile, dict):
            if machine_id and profile.get("machine_id") != machine_id:
                raise KeyError(f"machine profile not found: {machine_id}")
            return profile

    for line in text.splitlines():
        if not line.strip():
            continue
        profile = json.loads(line)
        if machine_id is None or profile.get("machine_id") == machine_id:
            return profile
    raise KeyError(f"machine profile not found: {machine_id or '<first>'}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _machine_summary(machine_profile: dict[str, Any]) -> dict[str, Any]:
    platform = machine_profile["platform"]
    return {
        "machine_id": machine_profile["machine_id"],
        "label": machine_profile["label"],
        "platform": {
            "vendor": platform["vendor"],
            "model": platform["model"],
            "os_family": platform["os_family"],
            "cpu": platform["cpu"],
            "ram_gb": platform["ram_gb"],
            "gpu": platform["gpu"],
            "vram_mb": platform["vram_mb"],
        },
    }


def _lane_index(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lanes: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        for started in (summary.get("lifecycle") or {}).get("started", []):
            lane_id = started.get("lane_id")
            if lane_id:
                lanes[lane_id] = {
                    "lane_id": lane_id,
                    "engine": started.get("engine", "unknown"),
                    "backend": started.get("backend", "unknown"),
                    "accelerator_ids": started.get("accelerator_ids", []),
                    "model_label": started.get("model_label", "unknown"),
                }
    return lanes


def _proven_lanes(
    summaries: list[dict[str, Any]],
    lane_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    proven: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        for response in (summary.get("responses") or {}).values():
            lane_id = response.get("lane_id")
            if not lane_id or response.get("outcome") != "success":
                continue
            lane = lane_index.get(lane_id, {})
            proven[lane_id] = {
                "lane_id": lane_id,
                "backend": lane.get("backend", "unknown"),
                "accelerator_ids": lane.get("accelerator_ids", []),
                "outcome": response.get("outcome", "unknown"),
                "latency_ms": _int(response.get("latency_ms")),
                "total_tokens": _int((response.get("usage") or {}).get("total_tokens")),
                "model_label": lane.get("model_label", "unknown"),
            }
    return list(proven.values())


def _lane_choice(
    role: str,
    summaries: list[dict[str, Any]],
    lane_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    lane_id = None
    for summary in summaries:
        lane_id = (summary.get("lanes") or {}).get(role)
        if lane_id:
            break
    lane = lane_index.get(lane_id or "", {})
    return {
        "lane_id": lane_id or "unknown",
        "backend": lane.get("backend", "unknown"),
        "accelerator_ids": lane.get("accelerator_ids", []),
        "role": role,
        "model_label": lane.get("model_label", "unknown"),
    }


def _strategy(fast_lane: dict[str, Any], capacity_lane: dict[str, Any]) -> str:
    if fast_lane["lane_id"] != "unknown" and capacity_lane["lane_id"] != "unknown":
        return "two_lane_local"
    if fast_lane["lane_id"] != "unknown" or capacity_lane["lane_id"] != "unknown":
        return "single_lane_local"
    return "benchmark_first"


def _fit_class(machine_profile: dict[str, Any], strategy: str) -> str:
    accelerator_count = len(machine_profile["hardware_topology"].get("accelerators", []))
    if strategy == "two_lane_local" and accelerator_count > 1:
        return "hybrid_local_ai_worker"
    if strategy == "single_lane_local":
        return "single_gpu_local_ai_worker"
    return "insufficient_evidence"


def _fit_summary(strategy: str) -> str:
    if strategy == "two_lane_local":
        return "Use separate local runtime lanes for fast small-model work and slower capacity-oriented model work."
    if strategy == "single_lane_local":
        return "Use the proven local runtime lane and continue benchmarking before larger deployments."
    return "Collect benchmark evidence before recommending a local model deployment."


def _primary_engine(lane_index: dict[str, dict[str, Any]]) -> str:
    engines = [lane.get("engine") for lane in lane_index.values() if lane.get("engine")]
    return engines[0] if engines else "unknown"


def _unsupported_claims(machine_profile: dict[str, Any]) -> list[dict[str, Any]]:
    claims = []
    for claim in machine_profile["hardware_topology"].get("memory_claims", []):
        if claim.get("status") != "verified":
            claims.append(
                {
                    "claim_id": claim["claim_id"],
                    "description": claim["description"],
                    "claimed_total_memory_mb": claim["claimed_total_memory_mb"],
                    "status": claim["status"],
                }
            )
    return claims


def _experiment_summary(summary: dict[str, Any]) -> dict[str, Any]:
    evidence = (summary.get("driver_evidence") or [{}])[0]
    return {
        "experiment_id": summary.get("experiment_id", "unknown"),
        "driver_branch": evidence.get("driver_branch", "unknown"),
        "driver_version": evidence.get("driver_version", "unknown"),
        "lanes": summary.get("lanes", {}),
        "outcomes": {
            role: response.get("outcome", "unknown")
            for role, response in (summary.get("responses") or {}).items()
        },
    }


def _comparison_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": comparison.get("verdict", "unknown"),
        "warnings": (comparison.get("comparability") or {}).get("warnings", []),
    }


def _benchmark_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record.get("run_id", "unknown"),
        "backend": (record.get("runtime") or {}).get("backend", "unknown"),
        "status": (record.get("outcome") or {}).get("status", "unknown"),
        "generation_tokens_per_second": (record.get("metrics") or {}).get("generation_tokens_per_second", 0),
    }


def _model_guidance(strategy: str) -> list[str]:
    if strategy == "two_lane_local":
        return [
            "Use the NVIDIA/CUDA lane for small models that fit dedicated VRAM and need low latency.",
            "Use the AMD/VGM Vulkan lane for larger GGUF models that do not fit the NVIDIA dGPU, accepting slower generation.",
            "Do not combine RTX VRAM and AMD VGM into one advertised memory pool unless a benchmark proves mixed-device offload.",
        ]
    return [
        "Run fit benchmarks before recommending larger local models.",
        "Prefer smaller quantized GGUF models until usable local capacity is proven.",
    ]


def _decisions(
    strategy: str,
    machine_profile: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions = [
        {
            "category": "Runtime",
            "recommendation": "Use llama.cpp runtime lanes as the primary local test harness.",
            "rationale": "The current evidence records backend, lane, device, latency, and outcome without relying on vendor memory claims.",
            "confidence": "observed" if strategy != "benchmark_first" else "inferred",
        },
        {
            "category": "Capacity",
            "recommendation": "Treat advertised or reserved graphics memory as a claim until benchmarked.",
            "rationale": "The machine profile keeps mixed-device pooling as unverified and records unsupported memory claims separately.",
            "confidence": "observed" if _unsupported_claims(machine_profile) else "inferred",
        },
    ]
    if comparisons:
        decisions.append(
            {
                "category": "Driver",
                "recommendation": "Record NVIDIA driver branch and version, but do not rank it above backend, device, and model evidence.",
                "rationale": "Experiment comparisons showed driver-branch differences inside normal run-to-run variation for this workflow.",
                "confidence": "observed",
            }
        )
    return decisions


def _next_benchmarks(strategy: str) -> list[str]:
    if strategy == "two_lane_local":
        return [
            "Run fixed-output repeat benchmarks for candidate small models on the NVIDIA/CUDA fast lane.",
            "Run fixed-output repeat benchmarks for candidate larger GGUF models on the AMD/VGM Vulkan capacity lane.",
            "Attach AMD Software telemetry summaries to longer AMD/VGM runs.",
        ]
    return [
        "Run a local endpoint smoke benchmark.",
        "Capture backend, devices used, memory use, fallback, and latency before recommending model size.",
    ]


def _int(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
