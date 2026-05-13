from __future__ import annotations

import json
from pathlib import Path

from applens_llm.benchmark_suite import build_benchmark_suite_run, write_benchmark_suite_run
from applens_llm.schemas import validate_payload


def test_builds_tiny_suite_with_real_benchmark_plan() -> None:
    suite = build_benchmark_suite_run(
        suite_run_id="qwen35-4b-vgm16-plan",
        model=_model(parameter_size_b=4, model_id="qwen35-4b-q4km"),
        machine_condition=_machine_condition(vgm_enabled=True, dedicated_vgm_mb=16384, system_ram_available_gb=16),
        runtime_lane=_runtime_lane(backend="vulkan", accelerator_ids=["amd-igpu-0"]),
        created_at="2026-05-13T12:00:00Z",
    )

    validate_payload("benchmark-suite-run", suite)
    assert suite["suite"]["suite_id"] == "tiny-v1"
    assert suite["suite"]["model_size_class"] == "tiny"
    assert suite["suite"]["benchmark_selection_review"]["long_context_primary"] == "LongBench v2"
    assert suite["suite"]["benchmark_selection_review"]["diagnostic_long_context"] == "RULER"
    assert suite["machine_condition"]["vgm_state"]["enabled"] is True
    assert suite["machine_condition"]["vgm_state"]["dedicated_mb"] == 16384

    task_ids = {task["task_id"] for task in suite["benchmark_plan"]["tasks"]}
    assert {
        "ifeval",
        "arc_challenge",
        "hellaswag",
        "gsm8k",
        "bfcl_prompt",
        "bigcodebench_hard_screening",
        "longbench_v2_screening",
        "ruler_context_taper",
    }.issubset(task_ids)
    assert "mmlu_pro" not in task_ids
    assert "generation_tokens_per_second" in suite["benchmark_plan"]["local_metrics"]
    assert "vgm_state" in suite["benchmark_plan"]["comparison_keys"]


def test_builds_small_suite_with_open_leaderboard_v2_family() -> None:
    suite = build_benchmark_suite_run(
        suite_run_id="qwen35-27b-vgm16-plan",
        model=_model(parameter_size_b=27, model_id="qwen35-27b-iq3"),
        machine_condition=_machine_condition(vgm_enabled=True, dedicated_vgm_mb=16384, system_ram_available_gb=16),
        runtime_lane=_runtime_lane(backend="vulkan", accelerator_ids=["amd-igpu-0"]),
        created_at="2026-05-13T12:00:00Z",
    )

    validate_payload("benchmark-suite-run", suite)
    assert suite["suite"]["suite_id"] == "small-v1"
    assert suite["suite"]["model_size_class"] == "small"
    task_ids = {task["task_id"] for task in suite["benchmark_plan"]["tasks"]}
    assert {
        "ifeval",
        "bbh",
        "math_level_5",
        "gpqa",
        "musr",
        "mmlu_pro",
        "bfcl_v4",
        "bigcodebench_hard",
        "longbench_v2_screening",
        "ruler_context_taper",
    }.issubset(task_ids)
    livebench = next(task for task in suite["benchmark_plan"]["tasks"] if task["task_id"] == "livebench_finalist")
    assert livebench["required_for_suite"] is False
    assert livebench["sample_policy"] == "finalists_only"


def test_explicit_suite_id_must_match_model_size_class() -> None:
    suite = build_benchmark_suite_run(
        suite_run_id="tiny-forced",
        suite_id="tiny-v1",
        model=_model(parameter_size_b=4, model_id="qwen35-4b-q4km"),
        machine_condition=_machine_condition(vgm_enabled=False, dedicated_vgm_mb=512, system_ram_available_gb=32),
        runtime_lane=_runtime_lane(backend="cuda", accelerator_ids=["nvidia-dgpu-0"]),
        created_at="2026-05-13T12:00:00Z",
    )

    assert suite["suite"]["model_size_class"] == "tiny"
    assert suite["machine_condition"]["vgm_state"]["enabled"] is False
    assert suite["machine_condition"]["vgm_state"]["system_ram_available_gb"] == 32


def test_write_benchmark_suite_run_writes_schema_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "suite.json"

    suite = write_benchmark_suite_run(
        output_path=output,
        suite_run_id="qwen35-4b-vgm16-plan",
        model=_model(parameter_size_b=4, model_id="qwen35-4b-q4km"),
        machine_condition=_machine_condition(vgm_enabled=True, dedicated_vgm_mb=16384, system_ram_available_gb=16),
        runtime_lane=_runtime_lane(backend="vulkan", accelerator_ids=["amd-igpu-0"]),
        created_at="2026-05-13T12:00:00Z",
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == suite
    validate_payload("benchmark-suite-run", suite)


def _model(*, parameter_size_b: float, model_id: str) -> dict:
    return {
        "model_id": model_id,
        "display_name": model_id.replace("-", " "),
        "family": "qwen",
        "parameter_size_b": parameter_size_b,
        "quantization": "Q4_K_M",
        "model_format": "gguf",
        "path": "sanitized/models/model.gguf",
        "sha256": "unknown",
        "chat_template": "qwen",
        "thinking_mode": "off",
        "reasoning_mode": "off",
    }


def _machine_condition(*, vgm_enabled: bool, dedicated_vgm_mb: int, system_ram_available_gb: int) -> dict:
    return {
        "condition_id": "asus-px13-vgm16-ram16" if vgm_enabled else "asus-px13-vgm0-ram32",
        "label": "ASUS PX13 sanitized condition",
        "os_family": "windows",
        "ram_gb": 32,
        "vgm_state": {
            "enabled": vgm_enabled,
            "dedicated_mb": dedicated_vgm_mb,
            "system_ram_available_gb": system_ram_available_gb,
            "source": "AMD Software: Adrenalin Edition",
        },
        "accelerator_ids": ["amd-igpu-0"] if vgm_enabled else ["nvidia-dgpu-0"],
        "required_preflight": ["close_competing_llm_apps", "record_power_mode"],
        "evidence_paths": [],
    }


def _runtime_lane(*, backend: str, accelerator_ids: list[str]) -> dict:
    return {
        "engine": "llama.cpp",
        "backend": backend,
        "device_selector": "Vulkan0" if backend == "vulkan" else "CUDA0",
        "accelerator_ids": accelerator_ids,
        "endpoint": "http://127.0.0.1:18080/v1",
        "context_tokens": 16384,
        "batch_size": 2048,
        "ubatch_size": 512,
        "threads": 12,
        "gpu_layers": 99,
        "kv_cache_type": "f16",
        "flash_attention": "auto",
        "extra_flags": ["--parallel", "1"],
    }
