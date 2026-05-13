from __future__ import annotations

import json
from pathlib import Path

from applens_llm.context_quality_probe import (
    build_context_quality_record,
    build_context_quality_prompt,
    build_llamacpp_quality_command,
    score_context_quality_response,
    write_context_quality_record,
)
from applens_llm.schemas import validate_payload


def test_scores_context_quality_response_as_observation_ready() -> None:
    record = build_context_quality_record(
        model={
            "model_id": "qwen35-4b-q4km",
            "display_name": "Qwen3.5 4B Q4_K_M",
            "family": "qwen",
            "parameter_size_b": 4,
            "quantization": "Q4_K_M",
        },
        runtime={
            "engine": "llama.cpp",
            "backend": "vulkan",
            "device_selector": "Vulkan1",
            "devices_used": ["nvidia-dgpu-0"],
        },
        context_tokens=16384,
        prompt_token_budget=12000,
        expected_needle="APPLENS-CTX-16384-PHOENIX",
        response_text=_passing_response(),
        created_at="2026-05-12T20:00:00Z",
        run_id="ctx-quality-qwen",
        elapsed_seconds=8.0,
        max_tokens=256,
        process_returncode=0,
        execute_code_checks=True,
    )

    validate_payload("context-quality-record", record)
    assert record["scores"]["quality_score_pct"] >= 80
    assert record["outcome"]["status"] == "pass"
    assert record["observation"]["status"] == "pass"
    assert record["observation"]["context_tokens"] == 16384
    assert record["observation"]["workloads"] == [
        "long_context_retrieval",
        "strict_json",
        "coding",
        "hardware_memory_reasoning",
    ]


def test_context_quality_rejects_wrong_needle_and_pooled_memory_claim() -> None:
    response = json.dumps(
        {
            "needle": "WRONG",
            "context_tier": 16384,
            "memory_claim_verdict": "22GB pooled VRAM is usable",
            "code": "def choose_context_tier(rows):\n    return 262144\n",
        }
    )

    scored = score_context_quality_response(
        response,
        expected_needle="APPLENS-CTX-16384-PHOENIX",
        context_tokens=16384,
        execute_code_checks=True,
    )

    cases = {case["case_id"]: case for case in scored["cases"]}
    assert cases["needle_retrieval"]["status"] == "fail"
    assert "needle_mismatch" in cases["needle_retrieval"]["issues"]
    assert cases["memory_claim_boundary"]["status"] == "fail"
    assert "unsupported_pool_claim" in cases["memory_claim_boundary"]["issues"]
    assert scored["scores"]["quality_score_pct"] < 60


def test_write_context_quality_record_writes_jsonl_observation(tmp_path: Path) -> None:
    responses = tmp_path / "response.json"
    output = tmp_path / "quality.json"
    observation = tmp_path / "quality-observation.jsonl"
    responses.write_text(_passing_response(needle="APPLENS-CTX-8192-PHOENIX", context_tier=8192), encoding="utf-8")

    record = write_context_quality_record(
        response_path=responses,
        output_path=output,
        observation_output_path=observation,
        model={
            "model_id": "gemma4-26b-a4b-q3km",
            "display_name": "Gemma 4 26B-A4B Q3_K_M",
            "family": "gemma",
            "parameter_size_b": 26,
            "quantization": "UD-Q3_K_M",
        },
        runtime={
            "engine": "llama.cpp",
            "backend": "vulkan",
            "device_selector": "Vulkan0",
            "devices_used": ["amd-igpu-0"],
        },
        context_tokens=8192,
        prompt_token_budget=6000,
        expected_needle="APPLENS-CTX-8192-PHOENIX",
        created_at="2026-05-12T20:00:00Z",
        run_id="ctx-quality-gemma",
        elapsed_seconds=16.0,
        max_tokens=256,
        process_returncode=0,
        execute_code_checks=True,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == record
    row = json.loads(observation.read_text(encoding="utf-8"))
    assert row["model_id"] == "gemma4-26b-a4b-q3km"
    assert row["quality_score_pct"] == record["scores"]["quality_score_pct"]
    assert row["status"] == "pass"


def test_build_context_quality_prompt_contains_needle_and_plain_task() -> None:
    prompt = build_context_quality_prompt(
        context_tokens=8192,
        prompt_token_budget=6000,
        needle="APPLENS-CTX-8192-PHOENIX",
    )

    assert "APPLENS-CTX-8192-PHOENIX" in prompt
    assert "Return only JSON" in prompt
    assert "choose_context_tier" in prompt


def test_build_llamacpp_quality_command_sets_context_and_device(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    command = build_llamacpp_quality_command(
        binary=Path("llama-cli.exe"),
        model=Path("model.gguf"),
        prompt_file=prompt,
        device="Vulkan1",
        context_tokens=16384,
        max_tokens=256,
        gpu_layers=99,
        threads=12,
    )

    assert command[:4] == ["llama-cli.exe", "-m", "model.gguf", "-dev"]
    assert "Vulkan1" in command
    assert ["-c", "16384"] == command[command.index("-c") : command.index("-c") + 2]
    assert "--no-display-prompt" in command
    assert "--no-conversation" not in command
    assert "--single-turn" in command
    assert "--reasoning" in command
    assert "off" in command
    assert "--reasoning-budget" in command
    assert "--simple-io" in command


def _passing_response(*, needle: str = "APPLENS-CTX-16384-PHOENIX", context_tier: int = 16384) -> str:
    return json.dumps(
        {
            "needle": needle,
            "context_tier": context_tier,
            "memory_claim_verdict": "benchmark_required_not_pooled",
            "code": (
                "def choose_context_tier(rows):\n"
                "    usable = [row for row in rows if row.get('status') == 'pass' "
                "and row.get('quality_score_pct', 0) >= 60 "
                "and row.get('generation_tokens_per_second', 0) >= 1]\n"
                "    return max((row.get('context_tokens', 0) for row in usable), default=0)\n"
            ),
        }
    )
