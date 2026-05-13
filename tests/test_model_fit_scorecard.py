from __future__ import annotations

import json
from pathlib import Path

from applens_llm.model_fit_scorecard import build_model_fit_scorecard, write_model_fit_scorecard
from applens_llm.schemas import validate_payload


def test_build_model_fit_scorecard_ranks_observed_and_candidate_models() -> None:
    scorecard = build_model_fit_scorecard(
        machine_profile=_machine_profile(),
        model_candidates=_model_candidates(),
        experiment_summaries=[_experiment_summary()],
        created_at="2026-05-10T01:00:00Z",
        scorecard_id="scorecard-asus-px13",
    )

    validate_payload("model-fit-scorecard", scorecard)
    assert "workload" not in scorecard
    rankings = {row["model_id"]: row for row in scorecard["rankings"]}

    assert rankings["jan-v35-4b-q4"]["fit_score"] >= 75
    assert rankings["jan-v35-4b-q4"]["recommended_role"] == "fast_chat"
    assert rankings["jan-v35-4b-q4"]["best_lane"]["backend"] == "cuda"
    assert rankings["qwen-27b-iq3"]["recommended_role"] == "deep_review"
    assert rankings["qwen-27b-iq3"]["best_lane"]["backend"] == "vulkan"
    assert rankings["gemma-e4b-q4"]["confidence"] == "inferred"
    assert "no_observed_benchmark" in rankings["gemma-e4b-q4"]["blockers"]
    assert scorecard["rankings"][0]["fit_score"] >= scorecard["rankings"][-1]["fit_score"]


def test_write_model_fit_scorecard_loads_inputs(tmp_path: Path) -> None:
    machines = tmp_path / "machines.jsonl"
    candidates = tmp_path / "models.json"
    summary = tmp_path / "summary.json"
    output = tmp_path / "scorecard.json"
    machines.write_text(json.dumps(_machine_profile()) + "\n", encoding="utf-8")
    candidates.write_text(json.dumps({"models": _model_candidates()}), encoding="utf-8")
    summary.write_text(json.dumps(_experiment_summary()), encoding="utf-8")

    scorecard = write_model_fit_scorecard(
        machine_profile_path=machines,
        machine_id="asus-laptop",
        model_candidates_path=candidates,
        experiment_summary_paths=[summary],
        output_path=output,
        created_at="2026-05-10T01:00:00Z",
        scorecard_id="scorecard-asus-px13",
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == scorecard
    assert scorecard["machine"]["machine_id"] == "asus-laptop"


def test_build_model_fit_scorecard_uses_benchmark_records_without_experiments() -> None:
    scorecard = build_model_fit_scorecard(
        machine_profile=_machine_profile(),
        benchmark_records=[_benchmark_record()],
        created_at="2026-05-10T01:00:00Z",
        scorecard_id="scorecard-benchmark-only",
    )

    validate_payload("model-fit-scorecard", scorecard)
    top = scorecard["rankings"][0]
    assert top["model_id"] == "qwen2.5:7b"
    assert top["recommended_role"] == "general_chat"
    assert top["best_lane"]["backend"] == "cuda"
    assert top["best_lane"]["accelerator_ids"] == ["nvidia-dgpu-0"]
    assert top["evidence"]["source"] == "benchmark_record"
    assert top["confidence"] == "observed"


def test_model_fit_scorecard_uses_capability_records_to_break_throughput_ties() -> None:
    scorecard = build_model_fit_scorecard(
        machine_profile=_machine_profile(),
        model_candidates=[
            {
                "model_id": "qwen35-2b-q4km",
                "display_name": "Qwen3.5 2B Q4_K_M",
                "family": "qwen",
                "parameter_size_b": 2,
                "quantization": "Q4_K_M",
                "file_size_mb": 1269,
                "local_status": "local",
                "preferred_roles": ["fast_chat", "coding"],
                "quality_prior": "medium",
                "observed_model_label": "qwen35-2b-q4km",
            },
            {
                "model_id": "qwen35-4b-q4km",
                "display_name": "Qwen3.5 4B Q4_K_M",
                "family": "qwen",
                "parameter_size_b": 4,
                "quantization": "Q4_K_M",
                "file_size_mb": 2734,
                "local_status": "local",
                "preferred_roles": ["fast_chat", "coding"],
                "quality_prior": "medium",
                "observed_model_label": "qwen35-4b-q4km",
            },
        ],
        benchmark_records=[
            _benchmark_record(model_name="qwen35-2b-q4km"),
            _benchmark_record(model_name="qwen35-4b-q4km"),
        ],
        capability_records=[
            _capability_record("qwen35-2b-q4km", score_pct=62),
            _capability_record("qwen35-4b-q4km", score_pct=94, thinking_mode="off"),
        ],
        created_at="2026-05-12T17:00:00Z",
        scorecard_id="scorecard-capability",
    )

    validate_payload("model-fit-scorecard", scorecard)
    rankings = {row["model_id"]: row for row in scorecard["rankings"]}
    assert scorecard["rankings"][0]["model_id"] == "qwen35-4b-q4km"
    assert rankings["qwen35-4b-q4km"]["score_breakdown"]["agent_capability"] > rankings["qwen35-2b-q4km"]["score_breakdown"]["agent_capability"]
    assert rankings["qwen35-4b-q4km"]["evidence"]["capability_score_pct"] == 94
    assert rankings["qwen35-4b-q4km"]["evidence"]["thinking_modes"] == ["off"]


def test_model_fit_scorecard_includes_context_envelope_evidence() -> None:
    scorecard = build_model_fit_scorecard(
        machine_profile=_machine_profile(),
        model_candidates=[
            {
                "model_id": "qwen35-27b-iq3",
                "display_name": "Qwen3.5 27B IQ3",
                "family": "qwen",
                "parameter_size_b": 27,
                "quantization": "IQ3_M",
                "file_size_mb": 12001,
                "local_status": "local",
                "preferred_roles": ["deep_review"],
                "quality_prior": "high",
                "observed_model_label": "qwen35-27b-iq3",
            }
        ],
        benchmark_records=[_benchmark_record(model_name="qwen35-27b-iq3")],
        context_envelopes=[_context_envelope()],
        created_at="2026-05-12T18:00:00Z",
        scorecard_id="scorecard-context",
    )

    row = scorecard["rankings"][0]
    assert row["evidence"]["advertised_context_tokens"] == 262144
    assert row["evidence"]["max_tested_context_tokens"] == 65536
    assert row["evidence"]["recommended_context_tokens"] == 65536
    assert row["evidence"]["context_score_pct"] == 76
    assert row["evidence"]["context_evidence_status"] == "observed_useful"
    assert row["score_breakdown"]["context_evidence"] > 0


def test_model_fit_scorecard_marks_untested_context_as_unproven_not_bad_performance() -> None:
    scorecard = build_model_fit_scorecard(
        machine_profile=_machine_profile(),
        model_candidates=[
            {
                "model_id": "qwen35-27b-iq3",
                "display_name": "Qwen3.5 27B IQ3",
                "family": "qwen",
                "parameter_size_b": 27,
                "quantization": "IQ3_M",
                "file_size_mb": 12001,
                "local_status": "local",
                "preferred_roles": ["deep_review"],
                "quality_prior": "high",
                "observed_model_label": "qwen35-27b-iq3",
            }
        ],
        benchmark_records=[_benchmark_record(model_name="qwen35-27b-iq3")],
        context_envelopes=[_unproven_context_envelope()],
        created_at="2026-05-12T18:00:00Z",
        scorecard_id="scorecard-context-unproven",
    )

    row = scorecard["rankings"][0]
    assert "context_fit" not in row["score_breakdown"]
    assert row["score_breakdown"]["context_evidence"] == 0
    assert row["evidence"]["context_evidence_status"] == "advertised_unproven"
    assert row["evidence"]["recommended_context_tokens"] == 0
    assert "not a performance finding" in row["evidence"]["context_interpretation"]
    assert any("Advertised context is unproven locally" in reason for reason in row["reasons"])
    assert all("Recommended context is 0" not in reason for reason in row["reasons"])


def test_failed_benchmark_record_is_not_ranked_as_ready() -> None:
    scorecard = build_model_fit_scorecard(
        machine_profile=_machine_profile(),
        model_candidates=[
            {
                "model_id": "large-local-model",
                "display_name": "Large Local Model",
                "family": "qwen",
                "parameter_size_b": 27,
                "quantization": "IQ3_M",
                "file_size_mb": 11998,
                "local_status": "local",
                "preferred_roles": ["deep_review"],
                "quality_prior": "high",
                "observed_model_label": "large-local-model",
            }
        ],
        benchmark_records=[_benchmark_record(model_name="large-local-model", status="oom")],
        created_at="2026-05-10T01:00:00Z",
        scorecard_id="scorecard-failed-benchmark",
    )

    top = scorecard["rankings"][0]
    assert top["fit_score"] < 60
    assert top["score_band"] in {"experimental", "not_recommended"}
    assert "observed_failure" in top["blockers"]
    assert top["score_breakdown"]["speed_latency"] == 0


def test_model_fit_scorecard_includes_workload_role_guidance() -> None:
    scorecard = build_model_fit_scorecard(
        machine_profile=_machine_profile(),
        model_candidates=_model_candidates(),
        created_at="2026-05-10T01:00:00Z",
        scorecard_id="scorecard-oracle",
        workload_profile={
            "workload_id": "oracle",
            "model_role_needs": [
                {"role": "supervisor", "capabilities": ["reasoning"]},
                {"role": "workload_executor", "capabilities": ["allowlisted_commands"]},
            ],
        },
    )

    validate_payload("model-fit-scorecard", scorecard)
    assert scorecard["workload"]["workload_id"] == "oracle"
    assert any(role["role"] == "supervisor" for role in scorecard["workload"]["roles"])


def _model_candidates() -> list[dict]:
    return [
        {
            "model_id": "jan-v35-4b-q4",
            "display_name": "Jan v3.5 4B Q4",
            "family": "qwen",
            "parameter_size_b": 4,
            "quantization": "Q4_K_XL",
            "file_size_mb": 2860,
            "local_status": "local",
            "preferred_roles": ["fast_chat", "general_chat"],
            "quality_prior": "medium",
            "observed_model_label": "jan-v35-4b-q4",
        },
        {
            "model_id": "qwen-27b-iq3",
            "display_name": "Qwen3.5 27B IQ3",
            "family": "qwen",
            "parameter_size_b": 27,
            "quantization": "IQ3_M",
            "file_size_mb": 11998,
            "local_status": "local",
            "preferred_roles": ["deep_review"],
            "quality_prior": "high",
            "observed_model_label": "qwen-27b-iq3",
        },
        {
            "model_id": "gemma-e4b-q4",
            "display_name": "Gemma E4B Q4",
            "family": "gemma",
            "parameter_size_b": 4,
            "quantization": "Q4",
            "file_size_mb": 3200,
            "local_status": "candidate",
            "preferred_roles": ["fast_chat", "summarization"],
            "quality_prior": "high",
        },
    ]


def _experiment_summary() -> dict:
    return {
        "schema_version": "0.1",
        "experiment_id": "exp-studio",
        "lanes": {"fast": "fast-nvidia", "deep": "deep-amd-vgm"},
        "responses": {
            "fast": {
                "lane_id": "fast-nvidia",
                "outcome": "success",
                "latency_ms": 2174,
                "usage": {"total_tokens": 133},
            },
            "deep": {
                "lane_id": "deep-amd-vgm",
                "outcome": "success",
                "latency_ms": 20801,
                "usage": {"total_tokens": 271},
            },
        },
        "lifecycle": {
            "started": [
                {
                    "lane_id": "fast-nvidia",
                    "engine": "llama.cpp",
                    "backend": "cuda",
                    "accelerator_ids": ["nvidia-dgpu-0"],
                    "model_label": "jan-v35-4b-q4",
                },
                {
                    "lane_id": "deep-amd-vgm",
                    "engine": "llama.cpp",
                    "backend": "vulkan",
                    "accelerator_ids": ["amd-igpu-0"],
                    "model_label": "qwen-27b-iq3",
                },
            ]
        },
    }


def _benchmark_record(model_name: str = "qwen2.5:7b", status: str = "pass") -> dict:
    return {
        "schema_version": "0.1",
        "run_id": "bench-qwen25-7b-cuda",
        "created_at": "2026-05-10T01:00:00Z",
        "host": {
            "name": "asus-laptop",
            "os": "windows",
            "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "vram_mb": 6144,
            "hardware_topology": _machine_profile()["hardware_topology"],
        },
        "runtime": {
            "engine": "llama.cpp",
            "backend": "cuda",
            "build": "sanitized",
            "command": "llama-server -m models/qwen2.5-7b.gguf",
            "devices_used": ["nvidia-dgpu-0"],
            "mixed_device_offload": {
                "attempted": False,
                "worked": False,
                "strategy": "single_device",
                "notes": "Single CUDA device benchmark.",
            },
        },
        "model": {
            "name": model_name,
            "path": "models/qwen2.5-7b.gguf",
            "quantization": "Q4_K_M",
        },
        "workload": {"prompt_tokens": 512, "completion_tokens": 128},
        "metrics": {
            "prompt_tokens_per_second": 500.0,
            "generation_tokens_per_second": 40.0,
            "latency_ms": 3200,
            "vram_used_mb": 5200,
            "device_memory_used_mb": [{"accelerator_id": "nvidia-dgpu-0", "used_mb": 5200}],
            "cpu_spill_mb": 0,
            "thermal_notes": "No thermal throttle observed.",
        },
        "outcome": {
            "status": status,
            "fallback_occurred": status != "pass",
            "failure_modes": ["none"] if status == "pass" else [status],
            "notes": "Sanitized benchmark record.",
        },
    }


def _capability_record(model_id: str, *, score_pct: int, thinking_mode: str = "unknown") -> dict:
    return {
        "schema_version": "0.1",
        "run_id": f"cap-{model_id}",
        "created_at": "2026-05-12T17:00:00Z",
        "benchmark": {
            "id": "applens-local-v1",
            "version": "0.1",
            "inspirations": ["BFCL", "LiveCodeBench", "IFEval", "lm-evaluation-harness"],
        },
        "model": {
            "model_id": model_id,
            "display_name": model_id,
            "family": "qwen",
            "parameter_size_b": 4,
            "quantization": "Q4_K_M",
            "thinking_mode": thinking_mode,
        },
        "runtime": {
            "engine": "llama.cpp",
            "backend": "vulkan",
            "devices_used": ["nvidia-dgpu-0"],
        },
        "thinking": {
            "requested": thinking_mode,
            "observed_thinking_trace": False,
            "notes": "Unit-test capability record.",
        },
        "scores": {
            "total_points": score_pct,
            "max_points": 100,
            "score_pct": score_pct,
            "category_scores": {
                "tool_calling": {"points": score_pct, "max_points": 100, "score_pct": score_pct},
                "coding": {"points": score_pct, "max_points": 100, "score_pct": score_pct},
            },
        },
        "cases": [
            {
                "case_id": "unit",
                "category": "tool_calling",
                "status": "pass",
                "points": score_pct,
                "max_points": 100,
                "issues": [],
                "checks": [{"name": "unit", "passed": True, "points": score_pct, "max_points": 100}],
                "response_excerpt": "{}",
            }
        ],
        "outcome": {
            "status": "pass",
            "band": "agent_ready" if score_pct >= 90 else "limited",
            "notes": "Synthetic capability record.",
        },
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }


def _context_envelope() -> dict:
    return {
        "schema_version": "0.1",
        "envelope_id": "ctx-unit",
        "created_at": "2026-05-12T18:00:00Z",
        "machine": {"machine_id": "asus-laptop", "label": "ASUS ProArt PX13"},
        "models": [
            {
                "model_id": "qwen35-27b-iq3",
                "display_name": "Qwen3.5 27B IQ3",
                "advertised_context_tokens": 262144,
                "advertised_context": {
                    "tokens": 262144,
                    "source": "https://artificialanalysis.ai/leaderboards/models?size=small",
                    "confidence": "advertised",
                },
                "planned_context_tiers": [262144, 131072, 65536, 32768, 16384, 8192, 4096],
                "max_tested_context_tokens": 65536,
                "max_loadable_context_tokens": 65536,
                "max_stable_context_tokens": 65536,
                "max_useful_context_tokens": 65536,
                "max_recommended_context_tokens": 65536,
                "context_score_pct": 76,
                "context_evidence_status": "observed_useful",
                "context_interpretation": "Observed useful context at 65536 tokens.",
                "status": "observed_context",
                "blockers": [],
                "observations": [
                    {
                        "context_tokens": 65536,
                        "backend": "vulkan",
                        "devices_used": ["amd-igpu-0"],
                        "status": "pass",
                        "quality_score_pct": 76,
                        "generation_tokens_per_second": 5.4,
                        "prompt_tokens_per_second": 51,
                        "failure_modes": ["none"],
                        "workloads": ["long_context_retrieval"],
                        "notes": "Unit context observation.",
                    }
                ],
                "workload_recommendations": {
                    "long_context_retrieval": {
                        "context_tokens": 65536,
                        "quality_score_pct": 76,
                        "generation_tokens_per_second": 5.4,
                        "reason": "largest_stable_context",
                    }
                },
            }
        ],
        "comparisons": [],
        "next_actions": ["Run context taper benchmarks."],
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }


def _unproven_context_envelope() -> dict:
    envelope = _context_envelope()
    model = envelope["models"][0]
    model["max_tested_context_tokens"] = 0
    model["max_loadable_context_tokens"] = 0
    model["max_stable_context_tokens"] = 0
    model["max_useful_context_tokens"] = 0
    model["max_recommended_context_tokens"] = 0
    model["context_score_pct"] = 0
    model["context_evidence_status"] = "advertised_unproven"
    model["context_interpretation"] = "Advertised context is unproven locally; this is not a performance finding."
    model["status"] = "needs_context_benchmark"
    model["blockers"] = ["advertised_context_unproven"]
    model["observations"] = []
    model["workload_recommendations"] = {}
    return envelope


def _machine_profile() -> dict:
    return {
        "schema_version": "0.1",
        "machine_id": "asus-laptop",
        "label": "ASUS ProArt PX13",
        "capture_status": "captured_sanitized",
        "capture_priority": 2,
        "platform": {
            "vendor": "asus",
            "model": "ProArt PX13",
            "sku": "asus-proart-px13-sanitized",
            "os_family": "windows",
            "cpu": "AMD Ryzen AI 9 HX 370",
            "ram_gb": 32,
            "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU + AMD Radeon 890M",
            "vram_mb": 6144,
        },
        "hardware_topology": {
            "accelerators": [
                {
                    "accelerator_id": "nvidia-dgpu-0",
                    "kind": "nvidia_dgpu",
                    "vendor": "nvidia",
                    "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                    "present": True,
                    "api_support": ["cuda"],
                    "memory": {
                        "physical_dedicated_vram_mb": 6144,
                        "vgm_reserved_mb": 0,
                        "shared_graphics_memory_mb": 0,
                        "reported_total_graphics_memory_mb": 6144,
                        "estimated_usable_inference_memory_mb": 6144,
                        "confidence": "observed",
                    },
                    "verification": [{"source_type": "inventory", "notes": "Sanitized inventory."}],
                },
                {
                    "accelerator_id": "amd-igpu-0",
                    "kind": "amd_igpu",
                    "vendor": "amd",
                    "name": "AMD Radeon 890M",
                    "present": True,
                    "api_support": ["vulkan"],
                    "memory": {
                        "physical_dedicated_vram_mb": 512,
                        "vgm_reserved_mb": 16384,
                        "shared_graphics_memory_mb": 16384,
                        "reported_total_graphics_memory_mb": 16896,
                        "estimated_usable_inference_memory_mb": 16384,
                        "confidence": "observed",
                    },
                    "verification": [{"source_type": "inventory", "notes": "VGM active."}],
                },
            ],
            "usable_inference_capacity": {
                "estimated_usable_memory_mb": 6144,
                "confidence": "observed",
                "preferred_accelerator_ids": ["nvidia-dgpu-0"],
                "mixed_device_pooling": "unverified",
                "verification": [{"source_type": "inventory", "notes": "Sanitized inventory."}],
            },
            "memory_claims": [],
        },
        "target_roles": ["training_candidate"],
        "collection": {
            "applens_report": "captured",
            "applens_tune_report": "captured",
            "local_ai_profile": "captured",
            "llm_bench": "pending",
            "sanitized": True,
        },
        "notes": "Sanitized test profile.",
    }
