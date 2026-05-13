from __future__ import annotations

import json
from pathlib import Path

from applens_llm.benchmark_readiness import build_benchmark_readiness, write_benchmark_readiness
from applens_llm.benchmark_suite import build_benchmark_suite_run
from applens_llm.schemas import validate_payload


def test_build_benchmark_readiness_reports_installed_missing_and_blocked_tools(tmp_path: Path) -> None:
    report = build_benchmark_readiness(
        suite=_suite(),
        lm_eval_binary=Path("lm_eval"),
        runner_paths={"bfcl-eval": None, "bigcodebench": None, "longbench-v2": None, "ruler": None},
        command_runner=_fake_lm_eval_runner,
        endpoint_probe=lambda endpoint, model_name: {
            "endpoint": endpoint,
            "available": True,
            "chat_completions": True,
            "completions": True,
            "remote_tokenizer": False,
            "legacy_loglikelihood": False,
            "notes": "remote tokenizer missing",
        },
        created_at="2026-05-13T20:00:00Z",
    )

    validate_payload("benchmark-readiness", report)
    assert report["status"] == "partial"
    assert report["lm_eval"]["status"] == "ready"
    assert report["endpoint_capabilities"]["chat_completions"] is True
    assert report["endpoint_capabilities"]["legacy_loglikelihood"] is False
    tools = {tool["runner"]: tool for tool in report["runner_readiness"]}
    assert tools["lm-evaluation-harness"]["status"] == "partial"
    assert tools["bfcl-eval"]["status"] == "missing"
    assert tools["bigcodebench"]["status"] == "missing"
    tasks = {task["task_id"]: task for task in report["task_readiness"]}
    assert tasks["ifeval"]["status"] == "ready"
    assert tasks["hellaswag"]["status"] == "blocked"
    assert tasks["hellaswag"]["reason"] == "endpoint_missing_loglikelihood"
    assert any("BFCL" in action for action in report["next_actions"])


def test_write_benchmark_readiness_writes_schema_valid_report(tmp_path: Path) -> None:
    plan = tmp_path / "suite.json"
    output = tmp_path / "readiness.json"
    plan.write_text(json.dumps(_suite()), encoding="utf-8")

    report = write_benchmark_readiness(
        plan_path=plan,
        output_path=output,
        lm_eval_binary=Path("lm_eval"),
        runner_paths={"bfcl-eval": None, "bigcodebench": None, "longbench-v2": None, "ruler": None},
        command_runner=_fake_lm_eval_runner,
        endpoint_probe=lambda endpoint, model_name: {
            "endpoint": endpoint,
            "available": False,
            "chat_completions": False,
            "completions": False,
            "remote_tokenizer": False,
            "legacy_loglikelihood": False,
            "notes": "skipped in unit test",
        },
        created_at="2026-05-13T20:00:00Z",
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    validate_payload("benchmark-readiness", report)


def _fake_lm_eval_runner(args: list[str]) -> tuple[int, str, str]:
    if args[:2] == ["lm_eval", "validate"]:
        return 0, "All tasks found and valid", ""
    return 1, "", "unexpected command"


def _suite() -> dict:
    return build_benchmark_suite_run(
        suite_run_id="qwen35-4b-vgm16-plan",
        model={
            "model_id": "jan-v35-4b-q4-xl",
            "display_name": "Jan v3.5 4B Q4 XL",
            "family": "qwen",
            "parameter_size_b": 4,
            "quantization": "Q4_K_XL",
            "model_format": "gguf",
            "path": "sanitized/models/model.gguf",
            "sha256": "unknown",
            "chat_template": "qwen",
            "thinking_mode": "off",
            "reasoning_mode": "off",
        },
        machine_condition={
            "condition_id": "asus-px13-vgm16-ram16",
            "label": "ASUS PX13 VGM 16GB",
            "os_family": "windows",
            "ram_gb": 32,
            "vgm_state": {
                "enabled": True,
                "dedicated_mb": 16384,
                "system_ram_available_gb": 16,
                "source": "AMD Software: Adrenalin Edition",
            },
            "accelerator_ids": ["amd-igpu-0"],
            "required_preflight": [],
            "evidence_paths": [],
        },
        runtime_lane={
            "engine": "llama.cpp",
            "backend": "vulkan",
            "device_selector": "Vulkan0",
            "accelerator_ids": ["amd-igpu-0"],
            "endpoint": "http://127.0.0.1:18080/v1",
            "context_tokens": 16384,
            "batch_size": 2048,
            "ubatch_size": 512,
            "threads": 12,
            "gpu_layers": 99,
            "kv_cache_type": "auto",
            "flash_attention": "auto",
            "extra_flags": [],
        },
        created_at="2026-05-13T20:00:00Z",
    )
