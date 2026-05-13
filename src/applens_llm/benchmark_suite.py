from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


TINY_MAX_B = 4.5
SMALL_MAX_B = 30.0

BENCHMARK_REVIEW_SOURCES = [
    {
        "title": "Hugging Face Open LLM Leaderboard v2",
        "url": "https://huggingface.co/docs/leaderboards/main/open_llm_leaderboard/about",
        "reason": "Current open model leaderboard task family: IFEval, BBH, MATH Lvl 5, GPQA, MuSR, and MMLU-Pro.",
    },
    {
        "title": "EleutherAI LM Evaluation Harness",
        "url": "https://lm-evaluation-harness.readthedocs.io/",
        "reason": "Primary reproducible runner for broad local and API model evaluation tasks.",
    },
    {
        "title": "Berkeley Function Calling Leaderboard V4",
        "url": "https://gorilla.cs.berkeley.edu/leaderboard",
        "reason": "Current tool/function-calling benchmark with prompt-mode support for non-native tool models.",
    },
    {
        "title": "BigCodeBench",
        "url": "https://github.com/bigcode-project/bigcodebench",
        "reason": "Practical code-generation benchmark with an official hard split for deeper coding evidence.",
    },
    {
        "title": "LongBench v2",
        "url": "https://longbench2.github.io/",
        "reason": "Primary long-context capability benchmark for realistic reasoning across documents, dialogue, code repos, and structured data.",
    },
    {
        "title": "NVIDIA RULER",
        "url": "https://github.com/NVIDIA/RULER",
        "reason": "Repeatable diagnostic benchmark for effective context length, context tapering, retrieval, tracing, aggregation, and QA.",
    },
]

LOCAL_METRICS = [
    "prompt_tokens_per_second",
    "generation_tokens_per_second",
    "time_to_first_token_ms",
    "wall_time_seconds",
    "peak_device_memory_mb",
    "peak_system_memory_gb",
    "cpu_spill_mb",
    "oom_count",
    "crash_count",
    "fallback_count",
    "thermal_throttle_observed",
    "repeat_variance_pct",
]

COMPARISON_KEYS = [
    "model_id",
    "suite_id",
    "condition_id",
    "vgm_state",
    "backend",
    "accelerator_ids",
    "context_tokens",
    "thinking_mode",
    "reasoning_mode",
    "quantization",
    "llama_cpp_build",
]

INVALIDATION_KEYS = [
    "model_sha256",
    "model_quantization",
    "chat_template",
    "driver_version",
    "runtime_build",
    "backend",
    "accelerator_ids",
    "vgm_state",
    "context_tokens",
    "kv_cache_type",
    "flash_attention",
]


def build_benchmark_suite_run(
    *,
    suite_run_id: str,
    model: dict[str, Any],
    machine_condition: dict[str, Any],
    runtime_lane: dict[str, Any],
    suite_id: str | None = None,
    scoring_mode: str = "local_screening",
    run_intent: str = "plan_only",
    repeats: int = 1,
    timeout_seconds_per_task: int = 900,
    allow_downloads: bool = False,
    allow_network: bool = False,
    expected_duration_minutes: int | None = None,
    artifact_root: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    size_class = _model_size_class(float(model["parameter_size_b"]))
    chosen_suite = suite_id or f"{size_class}-v1"
    if chosen_suite not in {"tiny-v1", "small-v1"}:
        raise ValueError("suite_id must be tiny-v1 or small-v1")
    if chosen_suite == "tiny-v1" and size_class != "tiny":
        raise ValueError("tiny-v1 only supports models at or below 4.5B parameters")
    if chosen_suite == "small-v1" and size_class not in {"tiny", "small"}:
        raise ValueError("small-v1 only supports models at or below 30B parameters")

    timestamp = created_at or _utc_now()
    tasks = _tiny_tasks(runtime_lane) if chosen_suite == "tiny-v1" else _small_tasks(runtime_lane)
    local_paths_included = _has_local_path(model.get("path", "")) or bool(machine_condition.get("evidence_paths"))
    suite = {
        "schema_version": "0.1",
        "suite_run_id": suite_run_id,
        "created_at": timestamp,
        "suite": {
            "suite_id": chosen_suite,
            "model_size_class": "tiny" if chosen_suite == "tiny-v1" else "small",
            "scoring_mode": scoring_mode,
            "benchmark_selection_review": {
                "reviewed_at": timestamp,
                "decision": (
                    "Use official benchmark families for capability scoring. LongBench v2 is the primary "
                    "long-context capability benchmark; RULER is retained as a repeatable diagnostic context taper."
                ),
                "long_context_primary": "LongBench v2",
                "diagnostic_long_context": "RULER",
                "sources": [_source(source, timestamp) for source in BENCHMARK_REVIEW_SOURCES],
            },
        },
        "model": _normalize_model(model),
        "machine_condition": _normalize_machine_condition(machine_condition),
        "runtime_lane": _normalize_runtime_lane(runtime_lane),
        "execution": {
            "run_intent": run_intent,
            "repeats": repeats,
            "timeout_seconds_per_task": timeout_seconds_per_task,
            "allow_downloads": allow_downloads,
            "allow_network": allow_network,
            "expected_duration_minutes": expected_duration_minutes or _expected_duration(chosen_suite, scoring_mode),
            "benchmark_runner": "mixed",
            "command_templates": _command_templates(chosen_suite),
        },
        "benchmark_plan": {
            "tasks": tasks,
            "local_metrics": list(LOCAL_METRICS),
            "comparison_keys": list(COMPARISON_KEYS),
            "invalidation_keys": list(INVALIDATION_KEYS),
            "notes": (
                "local_screening runs may use official subsets for practical local comparison; they are not "
                "leaderboard-comparable certification results until full task counts and official settings are used."
            ),
        },
        "output_contract": {
            "artifact_root": artifact_root or f"out/benchmark-suites/{suite_run_id}",
            "required_artifacts": [
                "benchmark-suite-run",
                "raw_logs",
                "per_task_results",
                "benchmark_suite_summary",
                "scorecard_inputs",
                "html_report",
            ],
            "failure_policy": (
                "Failed, crashed, timed out, OOM, or fallback rows must be retained with invalid throughput and explicit failure modes."
            ),
        },
        "privacy": {
            "commit_safe": not local_paths_included,
            "local_paths_included": local_paths_included,
        },
    }
    validate_payload("benchmark-suite-run", suite)
    return suite


def write_benchmark_suite_run(
    *,
    output_path: Path,
    suite_run_id: str,
    model: dict[str, Any],
    machine_condition: dict[str, Any],
    runtime_lane: dict[str, Any],
    suite_id: str | None = None,
    scoring_mode: str = "local_screening",
    run_intent: str = "plan_only",
    repeats: int = 1,
    timeout_seconds_per_task: int = 900,
    allow_downloads: bool = False,
    allow_network: bool = False,
    expected_duration_minutes: int | None = None,
    artifact_root: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    suite = build_benchmark_suite_run(
        suite_run_id=suite_run_id,
        model=model,
        machine_condition=machine_condition,
        runtime_lane=runtime_lane,
        suite_id=suite_id,
        scoring_mode=scoring_mode,
        run_intent=run_intent,
        repeats=repeats,
        timeout_seconds_per_task=timeout_seconds_per_task,
        allow_downloads=allow_downloads,
        allow_network=allow_network,
        expected_duration_minutes=expected_duration_minutes,
        artifact_root=artifact_root,
        created_at=created_at,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return suite


def _model_size_class(parameter_size_b: float) -> str:
    if parameter_size_b <= TINY_MAX_B:
        return "tiny"
    if parameter_size_b <= SMALL_MAX_B:
        return "small"
    raise ValueError("benchmark-suite-run v1 supports models at or below 30B parameters")


def _tiny_tasks(runtime_lane: dict[str, Any]) -> list[dict[str, Any]]:
    context_tokens = int(runtime_lane["context_tokens"])
    return [
        _task("ifeval", "IFEval", "instruction_following", "lm-evaluation-harness", "ifeval", "zero_shot", "generate_until", ["chat/completions returns non-empty message.content", "thinking output disabled or stripped before scoring"], "mark_failed_if_empty_content", "strict instruction accuracy", "official_subset_for_local_screening", 0, True),
        _task("arc_challenge_chat", "ARC-Challenge Chat", "reasoning", "lm-evaluation-harness", "arc_challenge_chat", "zero_shot", "generate_until", ["chat/completions returns final answer text", "thinking tags stripped before exact-match scoring"], "mark_failed_if_reasoning_tags_remain", "exact-match answer-letter accuracy", "official_subset_for_local_screening", 0, True),
        _task("hellaswag", "HellaSwag", "reasoning", "lm-evaluation-harness", "hellaswag", "few_shot", "loglikelihood", ["prompt token logprobs for continuation scoring", "tokenizer-compatible context length accounting"], "mark_unsupported_do_not_substitute_custom_generation", "normalized accuracy", "official_subset_for_local_screening", 0, True),
        _task("gsm8k", "GSM8K", "math", "lm-evaluation-harness", "gsm8k", "few_shot", "generate_until", ["chat/completions returns extractable final answer", "thinking output disabled or stripped before scoring"], "mark_failed_if_no_extractable_answer", "exact-match extracted answer", "official_subset_for_local_screening", 0, True),
        _task("bfcl_prompt", "BFCL V4", "tool_calling", "bfcl-eval", "prompt-mode", "prompt_tool_calling", "external_execution", ["BFCL prompt-mode runner available", "OpenAI-compatible generation endpoint"], "mark_unsupported_if_runner_unavailable", "tool-call accuracy", "prompt_mode_for_non_native_tool_models", 0, True),
        _task("bigcodebench_hard_screening", "BigCodeBench-Hard", "coding", "bigcodebench", "hard-instruct", "execution", "external_execution", ["BigCodeBench runner available", "sandboxed code execution configured"], "mark_unsupported_if_execution_sandbox_unavailable", "pass@1 with executable tests", "screening_subset", 0, True),
        _task("longbench_v2_screening", "LongBench v2", "long_context", "longbench-v2", "short-medium-mcqa", "zero_shot", "generate_until", ["LongBench v2 data available", "configured context fits runtime lane"], "mark_failed_if_context_truncates_unexpectedly", "multiple-choice accuracy", "screening_subset_under_local_time_budget", context_tokens, True),
        _task("ruler_context_taper", "RULER", "long_context", "ruler", "niah+aggregation+taper", "context_taper", "context_diagnostic", ["RULER repo available", "configured context tiers fit runtime lane"], "mark_failed_if_context_tier_ooms", "effective context score", "diagnostic_taper_at_configured_context_tiers", context_tokens, True),
    ]


def _small_tasks(runtime_lane: dict[str, Any]) -> list[dict[str, Any]]:
    context_tokens = int(runtime_lane["context_tokens"])
    return [
        _task("ifeval", "IFEval", "instruction_following", "lm-evaluation-harness", "ifeval", "zero_shot", "generate_until", ["chat/completions returns non-empty message.content", "thinking output disabled or stripped before scoring"], "mark_failed_if_empty_content", "strict instruction accuracy", "official_settings_for_certification_or_subset_for_screening", 0, True),
        _task("bbh", "BBH", "reasoning", "lm-evaluation-harness", "bbh", "few_shot", "generate_until", ["chat/completions returns extractable final answer", "thinking output disabled or stripped before scoring"], "mark_failed_if_no_extractable_answer", "objective task accuracy", "official_settings_for_certification_or_subset_for_screening", 0, True),
        _task("math_level_5", "MATH Lvl 5", "math", "lm-evaluation-harness", "math_level5", "few_shot", "generate_until", ["chat/completions returns extractable final answer", "thinking output disabled or stripped before scoring"], "mark_failed_if_no_extractable_answer", "exact-match formatted answer", "official_settings_for_certification_or_subset_for_screening", 0, True),
        _task("gpqa", "GPQA", "reasoning", "lm-evaluation-harness", "gpqa", "zero_shot", "loglikelihood", ["prompt token logprobs for multiple-choice scoring", "tokenizer-compatible context length accounting"], "mark_unsupported_do_not_substitute_custom_generation", "multiple-choice accuracy", "official_settings_for_certification_or_subset_for_screening", 0, True),
        _task("musr", "MuSR", "reasoning", "lm-evaluation-harness", "musr", "zero_shot", "generate_until", ["chat/completions returns extractable final answer", "thinking output disabled or stripped before scoring"], "mark_failed_if_no_extractable_answer", "multiple-choice accuracy", "official_settings_for_certification_or_subset_for_screening", 0, True),
        _task("mmlu_pro", "MMLU-Pro", "reasoning", "lm-evaluation-harness", "mmlu_pro", "few_shot", "loglikelihood", ["prompt token logprobs for 10-choice scoring", "tokenizer-compatible context length accounting"], "mark_unsupported_do_not_substitute_custom_generation", "10-choice accuracy", "official_settings_for_certification_or_subset_for_screening", 0, True),
        _task("bfcl_v4", "BFCL V4", "tool_calling", "bfcl-eval", "prompt-or-native", "prompt_tool_calling", "external_execution", ["BFCL runner available", "prompt mode or native tool calling configured"], "mark_unsupported_if_runner_unavailable", "overall function-call accuracy", "prompt_mode_unless_native_tool_calling_is_available", 0, True),
        _task("bigcodebench_hard", "BigCodeBench-Hard", "coding", "bigcodebench", "hard-instruct", "execution", "external_execution", ["BigCodeBench runner available", "sandboxed code execution configured"], "mark_unsupported_if_execution_sandbox_unavailable", "pass@1 with executable tests", "screening_subset_then_full_hard_split_for_finalists", 0, True),
        _task("longbench_v2_screening", "LongBench v2", "long_context", "longbench-v2", "short-medium-long-mcqa", "zero_shot", "generate_until", ["LongBench v2 data available", "configured context fits runtime lane"], "mark_failed_if_context_truncates_unexpectedly", "multiple-choice accuracy", "screening_subset_under_local_time_budget", context_tokens, True),
        _task("ruler_context_taper", "RULER", "long_context", "ruler", "niah+aggregation+taper", "context_taper", "context_diagnostic", ["RULER repo available", "configured context tiers fit runtime lane"], "mark_failed_if_context_tier_ooms", "effective context score", "diagnostic_taper_at_configured_context_tiers", context_tokens, True),
        _task("livebench_finalist", "LiveBench", "reasoning", "livebench", "public-release-subset", "finalist_only", "external_execution", ["LiveBench runner available", "model promoted from local screening"], "skip_unless_finalist", "objective task accuracy", "finalists_only", 0, False),
    ]


def _task(
    task_id: str,
    benchmark: str,
    category: str,
    runner: str,
    task_ref: str,
    mode: str,
    required_lm_call: str,
    runner_requirements: list[str],
    runner_fallback_policy: str,
    scoring: str,
    sample_policy: str,
    context_tokens: int,
    required_for_suite: bool,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "benchmark": benchmark,
        "category": category,
        "runner": runner,
        "task_ref": task_ref,
        "mode": mode,
        "required_lm_call": required_lm_call,
        "runner_requirements": runner_requirements,
        "runner_fallback_policy": runner_fallback_policy,
        "required_for_suite": required_for_suite,
        "scoring": scoring,
        "sample_policy": sample_policy,
        "context_tokens": context_tokens,
        "notes": "Official benchmark task; local screening subsets must be labeled separately from certification runs.",
    }


def _normalize_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": str(model["model_id"]),
        "display_name": str(model.get("display_name") or model["model_id"]),
        "family": str(model.get("family") or "unknown"),
        "parameter_size_b": float(model["parameter_size_b"]),
        "quantization": str(model.get("quantization") or "unknown"),
        "model_format": str(model.get("model_format") or "unknown"),
        "path": str(model.get("path") or "local"),
        "sha256": str(model.get("sha256") or "unknown"),
        "chat_template": str(model.get("chat_template") or "unknown"),
        "thinking_mode": str(model.get("thinking_mode") or "unknown"),
        "reasoning_mode": str(model.get("reasoning_mode") or model.get("thinking_mode") or "unknown"),
    }


def _normalize_machine_condition(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": str(condition["condition_id"]),
        "label": str(condition.get("label") or condition["condition_id"]),
        "os_family": str(condition.get("os_family") or "unknown"),
        "ram_gb": int(condition.get("ram_gb") or 0),
        "vgm_state": {
            "enabled": bool((condition.get("vgm_state") or {}).get("enabled")),
            "dedicated_mb": int((condition.get("vgm_state") or {}).get("dedicated_mb") or 0),
            "system_ram_available_gb": int((condition.get("vgm_state") or {}).get("system_ram_available_gb") or 0),
            "source": str((condition.get("vgm_state") or {}).get("source") or "unknown"),
        },
        "accelerator_ids": [str(item) for item in condition.get("accelerator_ids", [])],
        "required_preflight": [str(item) for item in condition.get("required_preflight", [])],
        "evidence_paths": [str(item) for item in condition.get("evidence_paths", [])],
    }


def _normalize_runtime_lane(lane: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine": str(lane.get("engine") or "unknown"),
        "backend": str(lane.get("backend") or "unknown"),
        "device_selector": str(lane.get("device_selector") or "unknown"),
        "accelerator_ids": [str(item) for item in lane.get("accelerator_ids", [])],
        "endpoint": str(lane.get("endpoint") or "http://127.0.0.1:8080/v1"),
        "context_tokens": int(lane.get("context_tokens") or 4096),
        "batch_size": int(lane.get("batch_size") or 2048),
        "ubatch_size": int(lane.get("ubatch_size") or 512),
        "threads": int(lane.get("threads") or 12),
        "gpu_layers": int(lane.get("gpu_layers") or 99),
        "kv_cache_type": str(lane.get("kv_cache_type") or "auto"),
        "flash_attention": str(lane.get("flash_attention") or "auto"),
        "extra_flags": [str(item) for item in lane.get("extra_flags", [])],
    }


def _source(source: dict[str, str], accessed_at: str) -> dict[str, str]:
    return {
        "title": source["title"],
        "url": source["url"],
        "accessed_at": accessed_at,
        "reason": source["reason"],
    }


def _command_templates(suite_id: str) -> list[str]:
    commands = [
        "start llama-server with runtime_lane settings and bind to 127.0.0.1",
        "lm-eval run --model local-chat-completions --tasks <task_ref> --output_path <artifact_root>/lm-eval",
        "bfcl-eval --model <model_id> --test-category <prompt-or-native> --result-dir <artifact_root>/bfcl",
        "bigcodebench.generate/evaluate for configured screening or hard split",
        "run LongBench v2 configured subset through the local OpenAI-compatible endpoint",
        "run RULER taper at configured context tiers and record effective context score",
    ]
    if suite_id == "small-v1":
        commands.append("run LiveBench public subset only after finalist promotion")
    return commands


def _expected_duration(suite_id: str, scoring_mode: str) -> int:
    if scoring_mode == "certification":
        return 720 if suite_id == "small-v1" else 360
    return 180 if suite_id == "small-v1" else 90


def _has_local_path(path: str) -> bool:
    lowered = path.lower()
    return ":\\" in lowered or lowered.startswith("/") or lowered.startswith("\\\\")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
