from __future__ import annotations

import json
from pathlib import Path

from applens_llm.fit_report import build_fit_report, write_fit_report
from applens_llm.schemas import validate_payload


def test_build_fit_report_summarizes_hybrid_lane_evidence() -> None:
    report = build_fit_report(
        machine_profile=_machine_profile(),
        experiment_summaries=[_experiment_summary()],
        experiment_comparisons=[_driver_comparison()],
        created_at="2026-05-10T00:00:00Z",
        report_id="fit-asus-px13",
    )

    validate_payload("fit-report", report)
    assert report["machine"]["machine_id"] == "asus-laptop"
    assert report["fit"]["class"] == "hybrid_local_ai_worker"
    assert report["runtime_recommendation"]["strategy"] == "two_lane_local"
    assert report["runtime_recommendation"]["fast_lane"]["backend"] == "cuda"
    assert report["runtime_recommendation"]["capacity_lane"]["backend"] == "vulkan"
    assert report["capacity_assessment"]["mixed_device_pooling"] == "unverified"
    assert report["capacity_assessment"]["unsupported_claims"][0]["claim_id"] == "rtx4050-plus-vgm-22gb"
    assert any(decision["category"] == "Driver" for decision in report["decisions"])


def test_build_fit_report_promotes_standalone_benchmark_records_to_observed_lanes() -> None:
    report = build_fit_report(
        machine_profile=_machine_profile(),
        benchmark_records=[
            _benchmark_record(
                model_name="jan-v35-4b-q4",
                devices=["nvidia-dgpu-0"],
                latency_ms=3200,
                generation_tps=57.3,
            ),
            _benchmark_record(
                model_name="qwen-27b-iq3",
                devices=["amd-igpu-0"],
                latency_ms=32000,
                generation_tps=5.8,
            ),
        ],
        created_at="2026-05-10T00:00:00Z",
        report_id="fit-benchmark-only",
    )

    validate_payload("fit-report", report)
    assert report["fit"]["class"] == "hybrid_local_ai_worker"
    assert report["fit"]["confidence"] == "observed"
    assert report["runtime_recommendation"]["strategy"] == "two_lane_local"
    assert report["runtime_recommendation"]["fast_lane"]["accelerator_ids"] == ["nvidia-dgpu-0"]
    assert report["runtime_recommendation"]["capacity_lane"]["accelerator_ids"] == ["amd-igpu-0"]


def test_write_fit_report_loads_jsonl_machine_profile_and_json_inputs(tmp_path: Path) -> None:
    machine_profiles = tmp_path / "machines.jsonl"
    summary = tmp_path / "summary.json"
    comparison = tmp_path / "comparison.json"
    output = tmp_path / "fit-report.json"
    machine_profiles.write_text(json.dumps(_machine_profile()) + "\n", encoding="utf-8")
    summary.write_text(json.dumps(_experiment_summary()), encoding="utf-8")
    comparison.write_text(json.dumps(_driver_comparison()), encoding="utf-8")

    report = write_fit_report(
        machine_profile_path=machine_profiles,
        machine_id="asus-laptop",
        output_path=output,
        experiment_summary_paths=[summary],
        experiment_comparison_paths=[comparison],
        created_at="2026-05-10T00:00:00Z",
        report_id="fit-asus-px13",
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["evidence"]["experiment_summaries"][0]["experiment_id"] == "exp-studio"


def test_write_fit_report_selects_machine_from_multi_row_jsonl(tmp_path: Path) -> None:
    machine_profiles = tmp_path / "machines.jsonl"
    summary = tmp_path / "summary.json"
    output = tmp_path / "fit-report.json"
    other = _machine_profile()
    other["machine_id"] = "other-machine"
    machine_profiles.write_text(
        json.dumps(other) + "\n" + json.dumps(_machine_profile()) + "\n",
        encoding="utf-8",
    )
    summary.write_text(json.dumps(_experiment_summary()), encoding="utf-8")

    report = write_fit_report(
        machine_profile_path=machine_profiles,
        machine_id="asus-laptop",
        output_path=output,
        experiment_summary_paths=[summary],
        created_at="2026-05-10T00:00:00Z",
        report_id="fit-asus-px13",
    )

    assert report["machine"]["machine_id"] == "asus-laptop"


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
                    "api_support": ["cuda", "vulkan"],
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
                        "vgm_reserved_mb": 0,
                        "shared_graphics_memory_mb": 16384,
                        "reported_total_graphics_memory_mb": 16896,
                        "estimated_usable_inference_memory_mb": 512,
                        "confidence": "observed",
                    },
                    "verification": [{"source_type": "inventory", "notes": "Sanitized inventory."}],
                },
            ],
            "usable_inference_capacity": {
                "estimated_usable_memory_mb": 6144,
                "confidence": "inferred",
                "preferred_accelerator_ids": ["nvidia-dgpu-0"],
                "mixed_device_pooling": "unverified",
                "verification": [{"source_type": "inventory", "notes": "Benchmark required."}],
            },
            "memory_claims": [
                {
                    "claim_id": "rtx4050-plus-vgm-22gb",
                    "description": "Claim that RTX 4050 6 GB plus Radeon 890M 16 GB VGM forms a 22 GB pool.",
                    "claimed_total_memory_mb": 22528,
                    "confidence": "user_claimed",
                    "status": "partially_verified",
                    "verification": [{"source_type": "user_claim", "notes": "Requires benchmarks."}],
                }
            ],
        },
        "target_roles": ["training_candidate", "small_gpu_baseline", "policy_edge_case"],
        "collection": {
            "applens_report": "captured",
            "applens_tune_report": "captured",
            "local_ai_profile": "captured",
            "llm_bench": "pending",
            "sanitized": True,
        },
        "notes": "Sanitized hybrid laptop profile.",
    }


def _experiment_summary() -> dict:
    return {
        "schema_version": "0.1",
        "experiment_id": "exp-studio",
        "driver_evidence": [{"driver_version": "596.36", "driver_branch": "studio"}],
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
                    "device_selector": "CUDA0",
                    "accelerator_ids": ["nvidia-dgpu-0"],
                    "model_label": "jan-v35-4b-q4",
                },
                {
                    "lane_id": "deep-amd-vgm",
                    "engine": "llama.cpp",
                    "backend": "vulkan",
                    "device_selector": "Vulkan0",
                    "accelerator_ids": ["amd-igpu-0"],
                    "model_label": "qwen-27b-iq3",
                },
            ]
        },
    }


def _driver_comparison() -> dict:
    return {
        "schema_version": "0.1",
        "baseline": {"driver": {"version": "596.36", "branch": "game_ready"}},
        "candidate": {"driver": {"version": "596.36", "branch": "studio"}},
        "comparability": {
            "same_driver_version": True,
            "same_lanes": True,
            "warnings": ["driver_branches_differ", "token_counts_differ"],
        },
        "deltas": {
            "fast": {"latency_ms_delta": 36, "total_tokens_delta": 0},
            "deep": {"latency_ms_delta": -3165, "total_tokens_delta": -15},
        },
        "verdict": "inconclusive_token_counts_differ",
    }


def _benchmark_record(
    *,
    model_name: str,
    devices: list[str],
    latency_ms: int,
    generation_tps: float,
    status: str = "pass",
) -> dict:
    return {
        "schema_version": "0.1",
        "run_id": f"bench-{model_name}",
        "created_at": "2026-05-10T00:00:00Z",
        "host": {
            "name": "asus-laptop",
            "os": "windows",
            "gpu": "hybrid",
            "vram_mb": 6144,
            "hardware_topology": _machine_profile()["hardware_topology"],
        },
        "runtime": {
            "engine": "llama.cpp",
            "backend": "vulkan",
            "build": "b8892",
            "command": "llama-bench",
            "devices_used": devices,
            "mixed_device_offload": {
                "attempted": False,
                "worked": False,
                "strategy": "single_device",
                "notes": "unit",
            },
        },
        "model": {"name": model_name, "path": "local", "quantization": "Q4"},
        "workload": {"prompt_tokens": 512, "completion_tokens": 128},
        "metrics": {
            "prompt_tokens_per_second": 100.0,
            "generation_tokens_per_second": generation_tps,
            "latency_ms": latency_ms,
            "vram_used_mb": 0,
            "device_memory_used_mb": [],
            "cpu_spill_mb": 0,
            "thermal_notes": "unit",
            "temperature_c": 0,
        },
        "outcome": {
            "status": status,
            "fallback_occurred": False,
            "failure_modes": ["none"] if status == "pass" else ["oom"],
            "notes": "unit",
        },
    }
