from __future__ import annotations

import pytest

from applens_llm.schemas import SchemaValidationError, validate_jsonl_file, validate_payload


def test_machine_profile_requires_hardware_topology() -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_payload(
            "machine-profile",
            {
                "schema_version": "0.1",
                "machine_id": "hybrid-laptop",
                "label": "Hybrid laptop",
                "capture_status": "captured_sanitized",
                "capture_priority": 1,
                "platform": {
                    "vendor": "asus",
                    "model": "ProArt PX13",
                    "sku": "sanitized-px13",
                    "os_family": "windows",
                    "cpu": "AMD Ryzen AI 9 HX 370",
                    "ram_gb": 32,
                    "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
                    "vram_mb": 6144,
                },
                "target_roles": ["training_candidate"],
                "collection": {
                    "applens_report": "captured",
                    "applens_tune_report": "captured",
                    "local_ai_profile": "captured",
                    "llm_bench": "pending",
                    "sanitized": True,
                },
                "notes": "Topology omitted on purpose.",
            },
        )

    assert "hardware_topology" in str(exc_info.value)


def test_benchmark_record_requires_runtime_backend_and_device_proof() -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_payload(
            "benchmark-record",
            {
                "schema_version": "0.1",
                "run_id": "missing-device-proof",
                "created_at": "2026-05-08T00:00:00Z",
                "host": {
                    "name": "sanitized-host",
                    "os": "Windows",
                    "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
                    "vram_mb": 6144,
                },
                "runtime": {
                    "backend": "llama.cpp",
                    "build": "unknown",
                    "command": "llama-bench -m model.gguf",
                },
                "model": {
                    "name": "qwen-local",
                    "path": "models/qwen-local.gguf",
                    "quantization": "Q4_K_M GGUF",
                },
                "workload": {"prompt_tokens": 16, "completion_tokens": 8},
                "metrics": {
                    "prompt_tokens_per_second": 100.0,
                    "generation_tokens_per_second": 20.0,
                    "latency_ms": 400.0,
                    "vram_used_mb": 2048,
                },
                "outcome": {
                    "status": "pass",
                    "notes": "Missing backend/device proof on purpose.",
                },
            },
        )

    error = str(exc_info.value)
    assert "hardware_topology" in error
    assert "engine" in error
    assert "devices_used" in error


def test_asus_hybrid_profile_keeps_claimed_pool_separate_from_usable_capacity() -> None:
    rows = validate_jsonl_file("machine-profile", "data/machines.seed.jsonl")
    asus = next(row for row in rows if row["machine_id"] == "asus-laptop")
    topology = asus["hardware_topology"]
    accelerators = {item["accelerator_id"]: item for item in topology["accelerators"]}

    assert {"nvidia-dgpu-0", "amd-igpu-0", "amd-npu-0"} <= accelerators.keys()
    assert accelerators["nvidia-dgpu-0"]["memory"]["physical_dedicated_vram_mb"] == 6144
    assert accelerators["amd-igpu-0"]["memory"]["physical_dedicated_vram_mb"] == 512
    assert accelerators["amd-igpu-0"]["memory"]["vgm_reserved_mb"] == 0
    assert topology["usable_inference_capacity"]["mixed_device_pooling"] == "unverified"
    assert topology["usable_inference_capacity"]["estimated_usable_memory_mb"] < 22528

    pool_claim = next(claim for claim in topology["memory_claims"] if claim["claim_id"] == "rtx4050-plus-vgm-22gb")
    assert pool_claim["claimed_total_memory_mb"] == 22528
    assert pool_claim["status"] == "partially_verified"
    assert pool_claim["confidence"] == "user_claimed"
