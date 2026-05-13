from __future__ import annotations

import json
from pathlib import Path

from applens_llm.deployment_plan import build_deployment_plan, write_deployment_plan
from applens_llm.schemas import validate_payload


def test_build_deployment_plan_outfits_scorecard_with_cloud_supervisor_baseline() -> None:
    plan = build_deployment_plan(
        scorecard=_scorecard(),
        plan_id="asus-px13-outfit",
        workload_name="Oracle autoresearch",
        workload_intent="agent_runtime",
    )

    validate_payload("deployment-plan", plan)
    assert plan["recommended_runtime"]["model"] == "qwen35-4b-q4km"
    assert plan["recommended_runtime"]["launch_profile"]["host"] == "127.0.0.1"
    assert plan["outfitting"]["supervisor_baseline"] == {
        "role": "planner_supervisor",
        "runtime": "cloud_api",
        "relative_score": 100,
        "status": "reference_baseline",
        "replacement_gate_score": 90,
    }
    assert plan["outfitting"]["local_supervisor_candidate"]["model_id"] == "gemma4-26b-a4b-q3km"
    assert plan["outfitting"]["local_supervisor_candidate"]["status"] == "not_ready"
    assert plan["outfitting"]["assignments"][0]["role"] == "primary_local_worker"
    assert plan["outfitting"]["assignments"][0]["model_id"] == "qwen35-4b-q4km"
    assert any(assignment["role"] == "deep_review_worker" for assignment in plan["outfitting"]["assignments"])
    assert any(action["action"] == "close_competing_llm_apps" for action in plan["outfitting"]["preflight_actions"])
    assert any(rec["owner"] == "AppLens-Tune" for rec in plan["outfitting"]["tune_recommendations"])
    assert "local_inference" in plan["safe_jobs"]
    assert {"job": "large_download", "gate": "user_approval"} in plan["gated_jobs"]


def test_write_deployment_plan_loads_scorecard(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    output = tmp_path / "deployment-plan.json"
    scorecard.write_text(json.dumps(_scorecard()), encoding="utf-8")

    plan = write_deployment_plan(
        scorecard_path=scorecard,
        output_path=output,
        plan_id="asus-px13-outfit",
        workload_name="Oracle autoresearch",
        workload_intent="agent_runtime",
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == plan
    assert plan["outfitting"]["runtime_profiles"][0]["backend"] == "cuda"


def test_vulkan_runtime_profiles_use_distinct_ports_for_nvidia_and_amd() -> None:
    scorecard = _scorecard()
    scorecard["rankings"][0]["best_lane"]["backend"] = "vulkan"
    scorecard["rankings"][0]["best_lane"]["lane_id"] = "benchmark-vulkan-nvidia-dgpu-0"
    scorecard["rankings"][1]["best_lane"]["lane_id"] = "benchmark-vulkan-amd-igpu-0"

    plan = build_deployment_plan(scorecard=scorecard)

    ports = {
        profile["model_id"]: profile["port"]
        for profile in plan["outfitting"]["runtime_profiles"]
    }
    assert ports["qwen35-4b-q4km"] == 18080
    assert ports["gemma4-26b-a4b-q3km"] == 18082


def _scorecard() -> dict:
    return {
        "schema_version": "0.1",
        "scorecard_id": "scorecard-asus-local",
        "created_at": "2026-05-12T19:00:00Z",
        "machine": {
            "machine_id": "asus-laptop",
            "label": "ASUS ProArt PX13",
            "platform": {
                "cpu": "AMD Ryzen AI 9 HX 370",
                "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU + AMD Radeon 890M",
                "ram_gb": 32,
                "os_family": "windows",
            },
        },
        "scoring_weights": {
            "capacity_fit": 18,
            "speed_latency": 13,
            "stability": 10,
            "role_fit": 10,
            "quality_size": 5,
            "operational_readiness": 10,
            "evidence_confidence": 5,
            "agent_capability": 24,
            "context_evidence": 5,
        },
        "rankings": [
            {
                "rank": 1,
                "model_id": "qwen35-4b-q4km",
                "display_name": "Qwen3.5 4B Q4_K_M",
                "family": "qwen",
                "parameter_size_b": 4,
                "quantization": "Q4_K_M",
                "local_status": "local",
                "recommended_role": "fast_chat",
                "best_lane": {
                    "lane_id": "fast-nvidia",
                    "backend": "cuda",
                    "accelerator_ids": ["nvidia-dgpu-0"],
                    "role": "fast_chat",
                    "model_label": "qwen35-4b-q4km",
                },
                "fit_score": 82,
                "score_band": "good",
                "score_breakdown": {
                    "capacity_fit": 18,
                    "speed_latency": 13,
                    "stability": 10,
                    "role_fit": 10,
                    "quality_size": 4,
                    "operational_readiness": 10,
                    "evidence_confidence": 4,
                    "agent_capability": 11,
                    "context_evidence": 2,
                },
                "confidence": "observed",
                "reasons": ["Fast NVIDIA lane is proven."],
                "blockers": [],
                "evidence": {
                    "source": "benchmark_record+local_capability_record",
                    "observation_count": 1,
                    "capability_record_count": 1,
                    "thinking_modes": ["off"],
                    "avg_latency_ms": 3200,
                    "avg_total_tokens": 640,
                    "capability_score_pct": 74,
                    "capability_categories": {"tool_calling": 70, "coding": 78},
                    "advertised_context_tokens": 262144,
                    "max_tested_context_tokens": 16384,
                    "recommended_context_tokens": 16384,
                    "context_score_pct": 72,
                    "context_evidence_status": "observed_useful",
                    "context_interpretation": "Observed useful context at 16384 tokens.",
                },
                "next_benchmark": "Run tool-calling drilldown.",
            },
            {
                "rank": 2,
                "model_id": "gemma4-26b-a4b-q3km",
                "display_name": "Gemma 4 26B A4B Q3_K_M",
                "family": "gemma",
                "parameter_size_b": 26,
                "quantization": "Q3_K_M",
                "local_status": "local",
                "recommended_role": "deep_review",
                "best_lane": {
                    "lane_id": "deep-amd-vgm",
                    "backend": "vulkan",
                    "accelerator_ids": ["amd-igpu-0"],
                    "role": "deep_review",
                    "model_label": "gemma4-26b-a4b-q3km",
                },
                "fit_score": 76,
                "score_band": "good",
                "score_breakdown": {
                    "capacity_fit": 18,
                    "speed_latency": 4,
                    "stability": 10,
                    "role_fit": 10,
                    "quality_size": 5,
                    "operational_readiness": 10,
                    "evidence_confidence": 4,
                    "agent_capability": 12,
                    "context_evidence": 3,
                },
                "confidence": "observed",
                "reasons": ["AMD VGM lane is proven for larger model capacity."],
                "blockers": ["requires_amd_vgm_vulkan_path"],
                "evidence": {
                    "source": "benchmark_record+local_capability_record",
                    "observation_count": 1,
                    "capability_record_count": 1,
                    "thinking_modes": ["unknown"],
                    "avg_latency_ms": 26000,
                    "avg_total_tokens": 640,
                    "capability_score_pct": 86,
                    "capability_categories": {"tool_calling": 82, "coding": 84, "handoff_planning": 90},
                    "advertised_context_tokens": 256000,
                    "max_tested_context_tokens": 16384,
                    "recommended_context_tokens": 16384,
                    "context_score_pct": 84,
                    "context_evidence_status": "observed_useful",
                    "context_interpretation": "Observed useful context at 16384 tokens.",
                },
                "next_benchmark": "Run long-context taper.",
            },
            {
                "rank": 3,
                "model_id": "qwen35-27b-uncensored-iq3m",
                "display_name": "Qwen3.5 27B Uncensored IQ3_M",
                "family": "qwen",
                "parameter_size_b": 27,
                "quantization": "IQ3_M",
                "local_status": "local",
                "recommended_role": "deep_review",
                "best_lane": {
                    "lane_id": "deep-amd-vgm",
                    "backend": "vulkan",
                    "accelerator_ids": ["amd-igpu-0"],
                    "role": "deep_review",
                    "model_label": "qwen35-27b-uncensored-iq3m",
                },
                "fit_score": 72,
                "score_band": "usable",
                "score_breakdown": {
                    "capacity_fit": 18,
                    "speed_latency": 3,
                    "stability": 10,
                    "role_fit": 10,
                    "quality_size": 5,
                    "operational_readiness": 10,
                    "evidence_confidence": 4,
                    "agent_capability": 10,
                    "context_evidence": 2,
                },
                "confidence": "observed",
                "reasons": ["Runnable but slow."],
                "blockers": ["requires_amd_vgm_vulkan_path"],
                "evidence": {
                    "source": "benchmark_record+local_capability_record",
                    "observation_count": 1,
                    "capability_record_count": 1,
                    "thinking_modes": ["unknown"],
                    "avg_latency_ms": 54000,
                    "avg_total_tokens": 640,
                    "capability_score_pct": 78,
                    "capability_categories": {"tool_calling": 75, "coding": 74},
                    "advertised_context_tokens": 262144,
                    "max_tested_context_tokens": 8192,
                    "recommended_context_tokens": 8192,
                    "context_score_pct": 78,
                    "context_evidence_status": "observed_useful",
                    "context_interpretation": "Observed useful context at 8192 tokens.",
                },
                "next_benchmark": "Repeat quality probe.",
            },
        ],
        "evidence": {
            "experiment_summary_count": 0,
            "benchmark_record_count": 3,
            "capability_record_count": 3,
            "candidate_model_count": 3,
        },
        "next_actions": ["Run coding/tool-calling drilldown."],
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }
