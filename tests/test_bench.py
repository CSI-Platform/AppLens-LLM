from __future__ import annotations

from applens_llm.bench import build_benchmark_record, build_chat_payload
from applens_llm.schemas import validate_payload


def _unit_test_topology() -> dict:
    return {
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
                "verification": [
                    {
                        "source_type": "command",
                        "command": "nvidia-smi --query-gpu=name,memory.total --format=csv",
                        "notes": "Sanitized unit-test device inventory.",
                    }
                ],
            }
        ],
        "usable_inference_capacity": {
            "estimated_usable_memory_mb": 6144,
            "confidence": "observed",
            "preferred_accelerator_ids": ["nvidia-dgpu-0"],
            "mixed_device_pooling": "not_applicable",
            "verification": [
                {
                    "source_type": "benchmark",
                    "command": "llama-bench -m model.gguf -ngl 99",
                    "notes": "Single CUDA device benchmark proof.",
                }
            ],
        },
        "memory_claims": [],
    }


def test_chat_payload_uses_openai_compatible_shape() -> None:
    payload = build_chat_payload("qwen-local", "Return strict JSON.", max_tokens=64)

    assert payload == {
        "model": "qwen-local",
        "messages": [{"role": "user", "content": "Return strict JSON."}],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }


def test_benchmark_record_builder_matches_schema() -> None:
    response = {
        "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        "choices": [{"message": {"content": "{}"}}],
    }

    record = build_benchmark_record(
        run_id="unit-test",
        created_at="2026-05-03T20:00:00Z",
        host={
            "name": "test-host",
            "os": "Windows",
            "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "vram_mb": 6144,
            "driver_evidence": [
                {
                    "vendor": "nvidia",
                    "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                    "driver_version": "576.88",
                    "driver_branch": "game_ready",
                    "branch_confidence": "user_confirmed",
                    "version_source": "nvidia-smi",
                    "branch_source": "nvidia_app",
                    "benchmark_invalidates_on_change": True,
                }
            ],
            "hardware_topology": _unit_test_topology(),
        },
        runtime={
            "engine": "llama.cpp",
            "backend": "cuda",
            "build": "unknown",
            "command": "POST /v1/chat/completions",
            "devices_used": ["nvidia-dgpu-0"],
            "mixed_device_offload": {
                "attempted": False,
                "worked": False,
                "strategy": "single_device",
                "notes": "Unit test uses one CUDA device.",
            },
        },
        model={"name": "qwen-local", "path": "local", "quantization": "unknown"},
        response_json=response,
        latency_ms=300,
        vram_used_mb=0,
        device_memory_used_mb=[{"accelerator_id": "nvidia-dgpu-0", "used_mb": 0}],
        cpu_spill_mb=0,
        temperature_c=0,
        thermal_notes="No thermal telemetry collected during unit test.",
        notes="Unit test completion.",
    )

    validate_payload("benchmark-record", record)
    assert record["metrics"]["generation_tokens_per_second"] == 20
    assert record["runtime"]["backend"] == "cuda"
    assert record["runtime"]["devices_used"] == ["nvidia-dgpu-0"]
    assert record["outcome"]["fallback_occurred"] is False
    assert record["host"]["driver_evidence"][0]["driver_branch"] == "game_ready"


def test_benchmark_record_accepts_amd_adrenalin_telemetry_source() -> None:
    response = {
        "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        "choices": [{"message": {"content": "{}"}}],
    }

    record = build_benchmark_record(
        run_id="amd-vgm-telemetry-unit-test",
        created_at="2026-05-08T20:00:00Z",
        host={
            "name": "test-host",
            "os": "Windows",
            "gpu": "AMD Radeon 890M",
            "vram_mb": 16384,
            "hardware_topology": _unit_test_topology(),
        },
        runtime={
            "engine": "ollama",
            "backend": "vulkan",
            "build": "0.21.2",
            "command": "ollama run local-test",
            "devices_used": ["nvidia-dgpu-0"],
            "mixed_device_offload": {
                "attempted": False,
                "worked": False,
                "strategy": "single_device",
                "notes": "Unit test for telemetry source shape.",
            },
        },
        model={"name": "local-test", "path": "models/local-test.gguf", "quantization": "Q4_K_M"},
        response_json=response,
        latency_ms=300,
        vram_used_mb=2048,
        device_memory_used_mb=[{"accelerator_id": "nvidia-dgpu-0", "used_mb": 2048}],
        cpu_spill_mb=0,
        temperature_c=58,
        thermal_notes="AMD Software telemetry collected every 2 seconds.",
        telemetry_sources=[
            {
                "source": "amd_adrenalin",
                "sampling_interval_seconds": 2,
                "metrics": [
                    "gpu_utilization",
                    "gpu_clock_speed",
                    "gpu_power_watts",
                    "gpu_temperature_c",
                    "gpu_memory_utilization_mb",
                    "gpu_memory_clock_speed",
                    "cpu_utilization",
                    "system_memory_utilization_gb",
                ],
                "path": "out/vgm/adrenalin-log.csv",
                "notes": "Local-only telemetry path; do not commit raw log.",
            }
        ],
        notes="Unit test completion.",
    )

    validate_payload("benchmark-record", record)
    assert record["telemetry_sources"][0]["source"] == "amd_adrenalin"
