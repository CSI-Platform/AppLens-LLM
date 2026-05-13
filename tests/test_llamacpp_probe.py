from __future__ import annotations

from pathlib import Path

from applens_llm.schemas import validate_payload
from applens_llm.llamacpp_probe import (
    build_benchmark_record_from_llamacpp_record,
    build_llamacpp_bench_command,
    parse_llama_bench_json,
    parse_llamacpp_devices,
    summarize_llama_bench_rows,
)


def test_parse_llamacpp_devices_from_list_devices_output() -> None:
    output = """
Available devices:
  Vulkan0: AMD Radeon(TM) 890M Graphics (24380 MiB, 23161 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4050 Laptop GPU (5920 MiB, 5152 MiB free)
"""

    devices = parse_llamacpp_devices(output)

    assert devices == [
        {
            "llama_device": "Vulkan0",
            "name": "AMD Radeon(TM) 890M Graphics",
            "memory_total_mib": 24380,
            "memory_free_mib": 23161,
            "accelerator_id": "amd-igpu-0",
        },
        {
            "llama_device": "Vulkan1",
            "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "memory_total_mib": 5920,
            "memory_free_mib": 5152,
            "accelerator_id": "nvidia-dgpu-0",
        },
    ]


def test_build_llamacpp_bench_command_targets_one_device() -> None:
    command = build_llamacpp_bench_command(
        binary=Path("C:/llama/llama-bench.exe"),
        model=Path("C:/models/model.gguf"),
        device="Vulkan0",
        prompt_tokens=64,
        generation_tokens=16,
        repetitions=1,
        gpu_layers=99,
        threads=12,
    )

    assert command == [
        "C:\\llama\\llama-bench.exe",
        "-m",
        "C:\\models\\model.gguf",
        "-dev",
        "Vulkan0",
        "-ngl",
        "99",
        "-p",
        "64",
        "-n",
        "16",
        "-r",
        "1",
        "-t",
        "12",
        "-o",
        "json",
    ]


def test_parse_llama_bench_json_extracts_rows_after_logs() -> None:
    output = """
load_backend: loaded Vulkan backend
[
  {"devices":"Vulkan0","n_prompt":64,"n_gen":0,"avg_ts":415.613293},
  {"devices":"Vulkan0","n_prompt":0,"n_gen":16,"avg_ts":27.473348}
]
"""

    rows = parse_llama_bench_json(output)
    summary = summarize_llama_bench_rows(rows)

    assert summary["prompt_tokens_per_second"] == 415.613293
    assert summary["generation_tokens_per_second"] == 27.473348
    assert summary["devices"] == ["Vulkan0"]


def test_build_benchmark_record_from_llamacpp_record_matches_schema() -> None:
    raw_record = {
        "label": "jan4b-vulkan-amd",
        "created_at": "2026-05-12T06:00:00Z",
        "command": ["C:/llama/llama-bench.exe", "-dev", "Vulkan0"],
        "process": {"returncode": 0},
        "model": {
            "path": "C:/models/Jan-v3.5-4B-Q4_K_XL/model.gguf",
            "name": "model.gguf",
            "size_bytes": 2998927360,
        },
        "llamacpp": {
            "binary": "C:/llama/llama-bench.exe",
            "device": "Vulkan0",
            "settings": {
                "prompt_tokens": 512,
                "generation_tokens": 128,
                "repetitions": 3,
                "gpu_layers": 99,
                "threads": 12,
            },
            "summary": {
                "prompt_tokens_per_second": 400.0,
                "generation_tokens_per_second": 25.0,
                "devices": ["Vulkan0"],
                "model_type": "4B",
                "build_number": 8892,
                "build_commit": "abc123",
                "backend": "Vulkan",
            },
            "rows": [{"devices": "Vulkan0"}],
        },
        "raw": {"stdout": "[]", "stderr": ""},
    }
    machine_profile = {
        "machine_id": "unit-host",
        "platform": {
            "os_family": "windows",
            "gpu": "AMD Radeon 890M",
            "vram_mb": 16384,
        },
        "hardware_topology": {
            "accelerators": [
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
                        "reported_total_graphics_memory_mb": 24380,
                        "estimated_usable_inference_memory_mb": 16384,
                        "confidence": "observed",
                    },
                    "verification": [{"source_type": "benchmark", "notes": "unit"}],
                }
            ],
            "usable_inference_capacity": {
                "estimated_usable_memory_mb": 16384,
                "confidence": "observed",
                "preferred_accelerator_ids": ["amd-igpu-0"],
                "mixed_device_pooling": "unverified",
                "verification": [{"source_type": "benchmark", "notes": "unit"}],
            },
            "memory_claims": [],
        },
    }
    devices = {
        "devices": [
            {
                "llama_device": "Vulkan0",
                "name": "AMD Radeon(TM) 890M Graphics",
                "memory_total_mib": 24380,
                "memory_free_mib": 23161,
                "accelerator_id": "amd-igpu-0",
            }
        ]
    }

    record = build_benchmark_record_from_llamacpp_record(
        raw_record=raw_record,
        machine_profile=machine_profile,
        devices_inventory=devices,
        model_name="jan-v35-4b-q4",
        quantization="Q4_K_XL",
    )

    validate_payload("benchmark-record", record)
    assert record["runtime"]["backend"] == "vulkan"
    assert record["runtime"]["devices_used"] == ["amd-igpu-0"]
    assert record["model"]["name"] == "jan-v35-4b-q4"
    assert record["metrics"]["generation_tokens_per_second"] == 25.0
    assert record["metrics"]["latency_ms"] == 6400.0


def test_build_benchmark_record_classifies_vulkan_out_of_device_memory_as_oom() -> None:
    raw_record = {
        "label": "qwen27b-vulkan-nvidia",
        "created_at": "2026-05-12T06:00:00Z",
        "command": ["C:/llama/llama-bench.exe", "-dev", "Vulkan1"],
        "process": {"returncode": 1},
        "model": {"path": "C:/models/qwen/model.gguf", "name": "model.gguf", "size_bytes": 12580873568},
        "llamacpp": {
            "device": "Vulkan1",
            "settings": {"prompt_tokens": 512, "generation_tokens": 128},
            "summary": {
                "prompt_tokens_per_second": 0.0,
                "generation_tokens_per_second": 0.0,
                "devices": [],
                "backend": "Vulkan",
            },
            "rows": [],
        },
        "raw": {"stdout": "", "stderr": "vk::Device::allocateMemory: ErrorOutOfDeviceMemory"},
    }

    record = build_benchmark_record_from_llamacpp_record(
        raw_record=raw_record,
        machine_profile=_machine_profile(),
        devices_inventory={
            "devices": [{"llama_device": "Vulkan1", "accelerator_id": "nvidia-dgpu-0"}]
        },
        model_name="qwen-27b-iq3",
        quantization="IQ3_M",
    )

    assert record["outcome"]["status"] == "oom"
    assert record["outcome"]["failure_modes"] == ["oom"]


def _machine_profile() -> dict:
    return {
        "machine_id": "unit-host",
        "platform": {
            "os_family": "windows",
            "gpu": "AMD Radeon 890M",
            "vram_mb": 16384,
        },
        "hardware_topology": {
            "accelerators": [
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
                        "reported_total_graphics_memory_mb": 24380,
                        "estimated_usable_inference_memory_mb": 16384,
                        "confidence": "observed",
                    },
                    "verification": [{"source_type": "benchmark", "notes": "unit"}],
                }
            ],
            "usable_inference_capacity": {
                "estimated_usable_memory_mb": 16384,
                "confidence": "observed",
                "preferred_accelerator_ids": ["amd-igpu-0"],
                "mixed_device_pooling": "unverified",
                "verification": [{"source_type": "benchmark", "notes": "unit"}],
            },
            "memory_claims": [],
        },
    }
