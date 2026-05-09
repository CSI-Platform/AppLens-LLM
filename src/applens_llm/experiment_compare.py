from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LANE_KEYS = ("fast", "deep")


def compare_experiment_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    deltas = {
        lane: _lane_delta(
            baseline.get("responses", {}).get(lane, {}),
            candidate.get("responses", {}).get(lane, {}),
        )
        for lane in LANE_KEYS
    }
    warnings = _warnings(baseline, candidate, deltas)
    return {
        "schema_version": "0.1",
        "baseline": _run_descriptor(baseline),
        "candidate": _run_descriptor(candidate),
        "comparability": {
            "same_driver_version": _driver_version(baseline) == _driver_version(candidate),
            "same_lanes": baseline.get("lanes") == candidate.get("lanes"),
            "warnings": warnings,
        },
        "deltas": deltas,
        "verdict": _verdict(warnings, deltas),
    }


def write_experiment_comparison(
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    comparison = compare_experiment_summaries(baseline, candidate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return comparison


def _run_descriptor(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": summary.get("experiment_id", "unknown"),
        "driver": _driver(summary),
        "lanes": summary.get("lanes", {}),
    }


def _driver(summary: dict[str, Any]) -> dict[str, Any]:
    evidence = (summary.get("driver_evidence") or [{}])[0]
    return {
        "vendor": evidence.get("vendor", "unknown"),
        "device_name": evidence.get("device_name", "unknown"),
        "version": evidence.get("driver_version", "unknown"),
        "branch": evidence.get("driver_branch", "unknown"),
    }


def _driver_version(summary: dict[str, Any]) -> str:
    return _driver(summary)["version"]


def _lane_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_latency = _number_or_zero(baseline.get("latency_ms"))
    candidate_latency = _number_or_zero(candidate.get("latency_ms"))
    baseline_tokens = _number_or_zero((baseline.get("usage") or {}).get("total_tokens"))
    candidate_tokens = _number_or_zero((candidate.get("usage") or {}).get("total_tokens"))
    return {
        "baseline_outcome": baseline.get("outcome", "unknown"),
        "candidate_outcome": candidate.get("outcome", "unknown"),
        "latency_ms_baseline": baseline_latency,
        "latency_ms_candidate": candidate_latency,
        "latency_ms_delta": candidate_latency - baseline_latency,
        "latency_ms_delta_pct": _percent_delta(baseline_latency, candidate_latency),
        "total_tokens_baseline": baseline_tokens,
        "total_tokens_candidate": candidate_tokens,
        "total_tokens_delta": candidate_tokens - baseline_tokens,
        "total_tokens_delta_pct": _percent_delta(baseline_tokens, candidate_tokens),
        "latency_per_token_ms_baseline": _latency_per_token(baseline_latency, baseline_tokens),
        "latency_per_token_ms_candidate": _latency_per_token(candidate_latency, candidate_tokens),
    }


def _warnings(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    deltas: dict[str, dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if _driver_version(baseline) != _driver_version(candidate):
        warnings.append("driver_versions_differ")
    if _driver(baseline)["branch"] != _driver(candidate)["branch"]:
        warnings.append("driver_branches_differ")
    if baseline.get("lanes") != candidate.get("lanes"):
        warnings.append("lanes_differ")
    if any(delta["baseline_outcome"] != delta["candidate_outcome"] for delta in deltas.values()):
        warnings.append("outcomes_differ")
    if any(abs(delta["total_tokens_delta_pct"] or 0) > 5 for delta in deltas.values()):
        warnings.append("token_counts_differ")
    return warnings


def _verdict(warnings: list[str], deltas: dict[str, dict[str, Any]]) -> str:
    if "outcomes_differ" in warnings or "lanes_differ" in warnings:
        return "not_comparable"
    if "token_counts_differ" in warnings:
        return "inconclusive_token_counts_differ"
    if all(abs(delta["latency_ms_delta_pct"] or 0) <= 5 for delta in deltas.values()):
        return "no_material_difference_single_run"
    return "difference_observed_needs_repeats"


def _number_or_zero(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _percent_delta(baseline: int, candidate: int) -> float | None:
    if baseline == 0:
        return None
    return round(((candidate - baseline) / baseline) * 100, 2)


def _latency_per_token(latency_ms: int, total_tokens: int) -> float | None:
    if total_tokens == 0:
        return None
    return round(latency_ms / total_tokens, 3)
