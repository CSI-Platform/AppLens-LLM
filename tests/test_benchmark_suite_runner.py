from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from applens_llm.benchmark_suite import build_benchmark_suite_run
from applens_llm.benchmark_suite_runner import (
    build_lm_eval_command,
    build_suite_result,
    classify_loglikelihood_probe,
    parse_lm_eval_results,
)
from applens_llm.schemas import validate_payload


def test_build_lm_eval_command_uses_chat_endpoint_for_generation_task(tmp_path: Path) -> None:
    suite = _suite()
    task = next(task for task in suite["benchmark_plan"]["tasks"] if task["task_id"] == "ifeval")

    command = build_lm_eval_command(
        lm_eval_binary=Path("lm_eval"),
        suite=suite,
        task=task,
        output_dir=tmp_path / "ifeval",
        endpoint_base="http://127.0.0.1:18081/v1",
        local_screening_limit=2,
    )

    assert command.args[:4] == ["lm_eval", "run", "--model", "local-chat-completions"]
    assert "--apply_chat_template" in command.args
    assert "--limit" in command.args
    assert any("base_url=http://127.0.0.1:18081/v1/chat/completions" in arg for arg in command.args)
    assert any('"enable_thinking": false' in arg for arg in command.args)


def test_build_lm_eval_command_uses_completions_endpoint_for_loglikelihood_task(tmp_path: Path) -> None:
    suite = _suite()
    task = next(task for task in suite["benchmark_plan"]["tasks"] if task["task_id"] == "hellaswag")

    command = build_lm_eval_command(
        lm_eval_binary=Path("lm_eval"),
        suite=suite,
        task=task,
        output_dir=tmp_path / "hellaswag",
        endpoint_base="http://127.0.0.1:18081/v1",
        local_screening_limit=2,
    )

    assert command.args[:4] == ["lm_eval", "run", "--model", "local-completions"]
    assert "--apply_chat_template" not in command.args
    assert "--limit" in command.args
    assert any("base_url=http://127.0.0.1:18081/v1/completions" in arg for arg in command.args)
    assert any("tokenizer_backend=auto" in arg for arg in command.args)


def test_classify_loglikelihood_probe_rejects_generated_token_only_logprobs() -> None:
    payload = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": ".", "logprob": -0.5, "top_logprobs": []},
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1},
    }

    support = classify_loglikelihood_probe(payload)

    assert support.supported is False
    assert support.reason == "llama_cpp_generated_token_logprobs_only"


def test_classify_loglikelihood_probe_accepts_legacy_prompt_token_logprobs() -> None:
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["Question", " A", "."],
                    "token_logprobs": [None, -0.25, -0.75],
                    "top_logprobs": [None, {" A": -0.25}, {".": -0.75}],
                    "text_offset": [0, 8, 10],
                }
            }
        ]
    }

    support = classify_loglikelihood_probe(payload)

    assert support.supported is True
    assert support.reason == "legacy_prompt_token_logprobs"


def test_parse_lm_eval_results_extracts_primary_metric(tmp_path: Path) -> None:
    result_dir = tmp_path / "lm-eval" / "model"
    result_dir.mkdir(parents=True)
    (result_dir / "results_2026.json").write_text(
        json.dumps(
            {
                "results": {
                    "ifeval": {
                        "prompt_level_strict_acc,none": 0.5,
                        "sample_len": 2,
                    }
                },
                "n-samples": {"ifeval": {"effective": 2}},
            }
        ),
        encoding="utf-8",
    )

    parsed = parse_lm_eval_results(tmp_path / "lm-eval", "ifeval")

    assert parsed.result_path == result_dir / "results_2026.json"
    assert parsed.metrics["prompt_level_strict_acc,none"] == 0.5
    assert parsed.effective_samples == 2


def test_build_suite_result_records_unsupported_loglikelihood_task(tmp_path: Path) -> None:
    suite = _suite()
    task = next(task for task in suite["benchmark_plan"]["tasks"] if task["task_id"] == "hellaswag")

    result = build_suite_result(
        suite=suite,
        plan_path=tmp_path / "plan.json",
        artifact_root=tmp_path,
        task_results=[
            {
                "task_id": task["task_id"],
                "benchmark": task["benchmark"],
                "category": task["category"],
                "runner": task["runner"],
                "required_lm_call": task["required_lm_call"],
                "status": "unsupported",
                "failure_modes": ["unsupported_loglikelihood"],
                "command": [],
                "returncode": None,
                "metrics": {},
                "artifacts": [],
                "notes": "Runtime did not expose prompt token logprobs.",
            }
        ],
    )

    validate_payload("benchmark-suite-result", result)
    assert result["summary"]["unsupported"] == 1
    assert result["status"] == "blocked"


def test_build_suite_result_marks_mixed_pass_and_unsupported_as_partial(tmp_path: Path) -> None:
    suite = _suite()
    tasks = {task["task_id"]: task for task in suite["benchmark_plan"]["tasks"]}
    results = []
    for task_id, status, failure_modes in [
        ("ifeval", "pass", ["none"]),
        ("hellaswag", "unsupported", ["unsupported_loglikelihood"]),
    ]:
        task = tasks[task_id]
        results.append(
            {
                "task_id": task["task_id"],
                "benchmark": task["benchmark"],
                "category": task["category"],
                "runner": task["runner"],
                "required_lm_call": task["required_lm_call"],
                "status": status,
                "failure_modes": failure_modes,
                "command": [],
                "returncode": 0 if status == "pass" else None,
                "metrics": {},
                "effective_samples": 2 if status == "pass" else None,
                "artifacts": [],
                "notes": "",
            }
        )

    result = build_suite_result(
        suite=suite,
        plan_path=tmp_path / "plan.json",
        artifact_root=tmp_path,
        task_results=results,
    )

    validate_payload("benchmark-suite-result", result)
    assert result["summary"]["passed"] == 1
    assert result["summary"]["unsupported"] == 1
    assert result["status"] == "partial"


def test_cli_writes_benchmark_suite_result_dry_run(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    output = tmp_path / "result.json"
    plan.write_text(json.dumps(_suite()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "applens_llm.cli",
            "benchmark-suite-run",
            "--plan",
            str(plan),
            "--output",
            str(output),
            "--dry-run",
            "--task-id",
            "ifeval",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_payload("benchmark-suite-result", payload)
    assert payload["task_results"][0]["status"] == "pending"


def test_cli_dry_run_can_target_managed_llamacpp_proxy(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    output = tmp_path / "result.json"
    plan.write_text(json.dumps(_suite()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "applens_llm.cli",
            "benchmark-suite-run",
            "--plan",
            str(plan),
            "--output",
            str(output),
            "--dry-run",
            "--use-llamacpp-proxy",
            "--proxy-port",
            "19081",
            "--task-id",
            "ifeval",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["runner_context"]["proxy_used"] is True
    assert payload["runner_context"]["endpoint_base"] == "http://127.0.0.1:19081/v1"


def _suite() -> dict:
    return build_benchmark_suite_run(
        suite_run_id="qwen35-4b-vgm16-plan",
        model={
            "model_id": "qwen35-4b-q4km",
            "display_name": "Qwen3.5 4B Q4_K_M",
            "family": "qwen",
            "parameter_size_b": 4,
            "quantization": "Q4_K_M",
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
        created_at="2026-05-13T12:00:00Z",
    )
