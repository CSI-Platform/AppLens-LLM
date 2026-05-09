from __future__ import annotations

from pathlib import Path

from applens_llm.llamacpp_probe import (
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
