from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


BENCHMARK_ID = "applens-context-quality-v1"
BENCHMARK_VERSION = "0.1"
BENCHMARK_INSPIRATIONS = ["Needle-in-a-Haystack", "IFEval", "LiveCodeBench"]
WORKLOADS = ["long_context_retrieval", "strict_json", "coding", "hardware_memory_reasoning"]


def build_context_quality_prompt(*, context_tokens: int, prompt_token_budget: int, needle: str) -> str:
    header = f"""
You are running an AppLens-LLM local context quality probe.
Return only JSON. Do not use markdown.

The JSON object must contain exactly these useful keys:
- needle: copy the buried APPLENS needle value exactly.
- context_tier: the integer context tier being tested.
- memory_claim_verdict: explain that advertised or pooled graphics memory requires benchmark evidence and is not automatically one usable pool.
- code: Python code defining choose_context_tier(rows). It should return the largest row["context_tokens"] where row["status"] == "pass", row["quality_score_pct"] >= 60, and row["generation_tokens_per_second"] >= 1. Return 0 if no rows pass.

Context tier being tested: {context_tokens}
""".strip()
    needle_block = f"""
Buried fact for retrieval:
APPLENS_NEEDLE_VALUE = {needle}
""".strip()
    footer = f"""
Now answer using only JSON:
{{"needle":"...","context_tier":{context_tokens},"memory_claim_verdict":"...","code":"def choose_context_tier(rows):\\n    ..."}}
""".strip()
    filler = _filler_for_budget(prompt_token_budget, len(header) + len(needle_block) + len(footer))
    return "\n\n".join([header, filler[: len(filler) // 2], needle_block, filler[len(filler) // 2 :], footer])


def build_llamacpp_quality_command(
    *,
    binary: Path,
    model: Path,
    prompt_file: Path,
    device: str,
    context_tokens: int,
    max_tokens: int,
    gpu_layers: int,
    threads: int,
) -> list[str]:
    return [
        str(binary),
        "-m",
        str(model),
        "-dev",
        device,
        "-ngl",
        str(gpu_layers),
        "-c",
        str(context_tokens),
        "-n",
        str(max_tokens),
        "-t",
        str(threads),
        "-f",
        str(prompt_file),
        "--temp",
        "0",
        "--no-display-prompt",
        "--single-turn",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--simple-io",
        "--log-disable",
    ]


def score_context_quality_response(
    response_text: str,
    *,
    expected_needle: str,
    context_tokens: int,
    execute_code_checks: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    payload = _parse_json(response_text, issues)
    cases = [
        _score_strict_json(payload, issues, response_text),
        _score_needle(payload, expected_needle, response_text),
        _score_memory_claim(payload, response_text),
        _score_context_tier(payload, context_tokens, response_text),
        _score_coding(payload, response_text, execute_code_checks=execute_code_checks),
    ]
    total_points = round(sum(case["points"] for case in cases), 2)
    max_points = round(sum(case["max_points"] for case in cases), 2)
    quality_score_pct = round((total_points / max_points) * 100, 2) if max_points else 0
    return {
        "scores": {
            "total_points": total_points,
            "max_points": max_points,
            "quality_score_pct": quality_score_pct,
            "category_scores": _category_scores(cases),
        },
        "cases": cases,
    }


def build_context_quality_record(
    *,
    model: dict[str, Any],
    runtime: dict[str, Any],
    context_tokens: int,
    prompt_token_budget: int,
    expected_needle: str,
    response_text: str,
    created_at: str | None = None,
    run_id: str | None = None,
    elapsed_seconds: float = 0,
    max_tokens: int = 256,
    process_returncode: int = 0,
    prompt_tokens_per_second: float = 0,
    generation_tokens_per_second: float | None = None,
    execute_code_checks: bool = False,
) -> dict[str, Any]:
    scored = score_context_quality_response(
        response_text,
        expected_needle=expected_needle,
        context_tokens=context_tokens,
        execute_code_checks=execute_code_checks,
    )
    quality_score = scored["scores"]["quality_score_pct"]
    generation_tps = (
        round(float(generation_tokens_per_second), 4)
        if generation_tokens_per_second is not None
        else _generation_tps(max_tokens=max_tokens, elapsed_seconds=elapsed_seconds)
    )
    status = _status(process_returncode=process_returncode, quality_score=quality_score)
    observation = {
        "model_id": str(model["model_id"]),
        "context_tokens": context_tokens,
        "backend": str(runtime.get("backend", "unknown")),
        "devices_used": [str(device) for device in runtime.get("devices_used", ["unknown-accelerator-0"])],
        "status": status,
        "quality_score_pct": quality_score,
        "generation_tokens_per_second": generation_tps,
        "prompt_tokens_per_second": float(prompt_tokens_per_second),
        "failure_modes": _failure_modes(process_returncode=process_returncode, quality_score=quality_score),
        "workloads": list(WORKLOADS),
        "notes": "Context quality probe scored retrieval, strict JSON, hardware-memory boundary, and coding behavior.",
    }
    record = {
        "schema_version": "0.1",
        "run_id": run_id or f"context-quality-{uuid.uuid4().hex[:12]}",
        "created_at": created_at or _utc_now(),
        "benchmark": {
            "id": BENCHMARK_ID,
            "version": BENCHMARK_VERSION,
            "inspirations": BENCHMARK_INSPIRATIONS,
        },
        "model": _model_payload(model),
        "runtime": _runtime_payload(runtime),
        "probe": {
            "context_tokens": context_tokens,
            "prompt_token_budget": prompt_token_budget,
            "max_tokens": max_tokens,
            "expected_needle": expected_needle,
            "workloads": list(WORKLOADS),
        },
        "scores": scored["scores"],
        "cases": scored["cases"],
        "outcome": {
            "status": "pass" if status == "pass" else ("crash" if process_returncode != 0 else "fail"),
            "band": _band(quality_score),
            "process_returncode": int(process_returncode),
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "notes": "Useful context requires quality_score_pct >= 60 and a clean runtime return code.",
        },
        "observation": observation,
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }
    validate_payload("context-quality-record", record)
    return record


def write_context_quality_record(
    *,
    response_path: Path,
    output_path: Path,
    observation_output_path: Path | None,
    model: dict[str, Any],
    runtime: dict[str, Any],
    context_tokens: int,
    prompt_token_budget: int,
    expected_needle: str,
    created_at: str | None = None,
    run_id: str | None = None,
    elapsed_seconds: float = 0,
    max_tokens: int = 256,
    process_returncode: int = 0,
    prompt_tokens_per_second: float = 0,
    generation_tokens_per_second: float | None = None,
    execute_code_checks: bool = False,
) -> dict[str, Any]:
    record = build_context_quality_record(
        model=model,
        runtime=runtime,
        context_tokens=context_tokens,
        prompt_token_budget=prompt_token_budget,
        expected_needle=expected_needle,
        response_text=response_path.read_text(encoding="utf-8"),
        created_at=created_at,
        run_id=run_id,
        elapsed_seconds=elapsed_seconds,
        max_tokens=max_tokens,
        process_returncode=process_returncode,
        prompt_tokens_per_second=prompt_tokens_per_second,
        generation_tokens_per_second=generation_tokens_per_second,
        execute_code_checks=execute_code_checks,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if observation_output_path is not None:
        observation_output_path.parent.mkdir(parents=True, exist_ok=True)
        observation_output_path.write_text(json.dumps(record["observation"], sort_keys=True) + "\n", encoding="utf-8")
    return record


def run_llamacpp_context_quality_probe(
    *,
    binary: Path,
    gguf_model: Path,
    model: dict[str, Any],
    runtime: dict[str, Any],
    context_tokens: int,
    prompt_token_budget: int,
    output_path: Path,
    observation_output_path: Path | None = None,
    response_output_path: Path | None = None,
    max_tokens: int = 256,
    gpu_layers: int = 99,
    threads: int = 12,
    disable_vulkan_coopmat: bool = False,
    execute_code_checks: bool = False,
) -> dict[str, Any]:
    needle = f"APPLENS-CTX-{context_tokens}-{uuid.uuid4().hex[:8].upper()}"
    prompt = build_context_quality_prompt(
        context_tokens=context_tokens,
        prompt_token_budget=prompt_token_budget,
        needle=needle,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        prompt_file = Path(temp_dir) / "context-quality-prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        command = build_llamacpp_quality_command(
            binary=binary,
            model=gguf_model,
            prompt_file=prompt_file,
            device=str(runtime["device_selector"]),
            context_tokens=context_tokens,
            max_tokens=max_tokens,
            gpu_layers=gpu_layers,
            threads=threads,
        )
        env = os.environ.copy()
        if disable_vulkan_coopmat:
            env["GGML_VK_DISABLE_COOPMAT"] = "1"
        started = time.perf_counter()
        result = subprocess.run(command, text=True, capture_output=True, timeout=1800, check=False, env=env)
        elapsed_seconds = time.perf_counter() - started

    response_text = _clean_llamacpp_output(result.stdout, result.stderr)
    if response_output_path is not None:
        response_output_path.parent.mkdir(parents=True, exist_ok=True)
        response_output_path.write_text(response_text, encoding="utf-8")
    record = build_context_quality_record(
        model=model,
        runtime=runtime,
        context_tokens=context_tokens,
        prompt_token_budget=prompt_token_budget,
        expected_needle=needle,
        response_text=response_text,
        elapsed_seconds=elapsed_seconds,
        max_tokens=max_tokens,
        process_returncode=result.returncode,
        execute_code_checks=execute_code_checks,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if observation_output_path is not None:
        observation_output_path.parent.mkdir(parents=True, exist_ok=True)
        observation_output_path.write_text(json.dumps(record["observation"], sort_keys=True) + "\n", encoding="utf-8")
    return record


def _score_strict_json(payload: dict[str, Any] | None, issues: list[str], response_text: str) -> dict[str, Any]:
    checks = [_check("valid_json_object", isinstance(payload, dict), 20, 20)]
    case_issues = list(issues)
    return _case("strict_json_format", "strict_json", checks, case_issues, response_text)


def _score_needle(payload: dict[str, Any] | None, expected_needle: str, response_text: str) -> dict[str, Any]:
    found = isinstance(payload, dict) and payload.get("needle") == expected_needle
    issues = [] if found else ["needle_mismatch"]
    return _case(
        "needle_retrieval",
        "long_context_retrieval",
        [_check("exact_needle", found, 30, 30)],
        issues,
        response_text,
    )


def _score_memory_claim(payload: dict[str, Any] | None, response_text: str) -> dict[str, Any]:
    value = str(payload.get("memory_claim_verdict", "") if isinstance(payload, dict) else "").lower()
    requires_evidence = any(term in value for term in ("benchmark", "evidence", "not_pooled", "not pooled", "required"))
    unsupported_pool = "pooled vram is usable" in value or "one clean pool" in value or "22gb pooled" in value
    checks = [
        _check("requires_benchmark_evidence", requires_evidence, 10, 10),
        _check("does_not_accept_pooled_memory", not unsupported_pool, 10, 10),
    ]
    issues = ["unsupported_pool_claim"] if unsupported_pool else []
    return _case("memory_claim_boundary", "hardware_memory_reasoning", checks, issues, response_text)


def _score_context_tier(payload: dict[str, Any] | None, context_tokens: int, response_text: str) -> dict[str, Any]:
    value = payload.get("context_tier") if isinstance(payload, dict) else None
    passed = isinstance(value, int) and value == context_tokens
    return _case(
        "context_tier_echo",
        "context_selection",
        [_check("correct_context_tier", passed, 10, 10)],
        [] if passed else ["context_tier_mismatch"],
        response_text,
    )


def _score_coding(payload: dict[str, Any] | None, response_text: str, *, execute_code_checks: bool) -> dict[str, Any]:
    code = str(payload.get("code", "") if isinstance(payload, dict) else "")
    has_function = "def choose_context_tier" in code
    syntax_ok = _python_syntax_ok(code) if code else False
    unit_ok = _run_choose_context_tier_unit_test(code) if execute_code_checks and syntax_ok else False
    checks = [
        _check("has_choose_context_tier", has_function, 5, 5),
        _check("python_syntax", syntax_ok, 5, 5),
    ]
    if execute_code_checks:
        checks.append(_check("unit_test_passed", unit_ok, 20, 20))
    else:
        checks.append(_check("unit_test_not_executed", True, 20, 20, notes="Static scoring only."))
    issues = []
    if execute_code_checks and not unit_ok:
        issues.append("coding_unit_test_failed")
    return _case("coding_context_selector", "coding", checks, issues, response_text)


def _case(
    case_id: str,
    category: str,
    checks: list[dict[str, Any]],
    issues: list[str],
    response_text: str,
) -> dict[str, Any]:
    points = round(sum(check["points"] for check in checks), 2)
    max_points = round(sum(check["max_points"] for check in checks), 2)
    return {
        "case_id": case_id,
        "category": category,
        "status": "pass" if points == max_points and not issues else "fail",
        "points": points,
        "max_points": max_points,
        "issues": sorted(set(issues)),
        "checks": checks,
        "response_excerpt": _response_excerpt(response_text),
    }


def _check(name: str, passed: bool, points: float, max_points: float, *, notes: str | None = None) -> dict[str, Any]:
    payload = {
        "name": name,
        "passed": passed,
        "points": points if passed else 0,
        "max_points": max_points,
    }
    if notes:
        payload["notes"] = notes
    return payload


def _category_scores(cases: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"points": 0.0, "max_points": 0.0})
    for case in cases:
        bucket = grouped[case["category"]]
        bucket["points"] += float(case["points"])
        bucket["max_points"] += float(case["max_points"])
    return {
        category: {
            "points": round(values["points"], 2),
            "max_points": round(values["max_points"], 2),
            "score_pct": round((values["points"] / values["max_points"]) * 100, 2) if values["max_points"] else 0,
        }
        for category, values in sorted(grouped.items())
    }


def _parse_json(response_text: str, issues: list[str]) -> dict[str, Any] | None:
    candidate = response_text.strip()
    for value in (candidate, _extract_fenced_json(candidate), _extract_first_json_object(candidate)):
        if not value:
            continue
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if value != candidate:
                issues.append("non_strict_json_wrapper")
            return payload
    issues.append("invalid_json")
    return None


def _extract_fenced_json(response_text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else None


def _extract_first_json_object(response_text: str) -> str | None:
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start < 0 or end <= start:
        return None
    return response_text[start : end + 1]


def _clean_llamacpp_output(stdout: str, stderr: str) -> str:
    text = stdout.strip()
    if text:
        return text
    return stderr.strip()


def _filler_for_budget(prompt_token_budget: int, fixed_chars: int) -> str:
    target_chars = max((prompt_token_budget * 4) - fixed_chars, 0)
    chunk = (
        "Context filler: AppLens compares reported graphics memory, proven inference memory, "
        "backend behavior, context stability, JSON obedience, and coding reliability. "
    )
    repeats = max(target_chars // len(chunk), 1)
    return (chunk * repeats)[:target_chars]


def _model_payload(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": str(model["model_id"]),
        "display_name": str(model.get("display_name") or model["model_id"]),
        "family": str(model.get("family") or "unknown"),
        "parameter_size_b": float(model.get("parameter_size_b") or 0),
        "quantization": str(model.get("quantization") or "unknown"),
    }


def _runtime_payload(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine": str(runtime.get("engine") or "llama.cpp"),
        "backend": str(runtime.get("backend") or "unknown"),
        "device_selector": str(runtime.get("device_selector") or "unknown"),
        "devices_used": [str(device) for device in runtime.get("devices_used", ["unknown-accelerator-0"])],
    }


def _status(*, process_returncode: int, quality_score: float) -> str:
    if process_returncode != 0:
        return "crash"
    return "pass" if quality_score >= 60 else "fail"


def _failure_modes(*, process_returncode: int, quality_score: float) -> list[str]:
    if process_returncode != 0:
        return ["crash"]
    if quality_score < 60:
        return ["quality_below_threshold"]
    return ["none"]


def _band(score: float) -> str:
    if score >= 80:
        return "useful_context"
    if score >= 60:
        return "limited"
    return "not_recommended"


def _generation_tps(*, max_tokens: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0
    return round(max_tokens / elapsed_seconds, 4)


def _python_syntax_ok(content: str) -> bool:
    try:
        ast.parse(content)
    except SyntaxError:
        return False
    return True


def _run_choose_context_tier_unit_test(content: str) -> bool:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        (temp / "context_selector.py").write_text(content, encoding="utf-8")
        script = (
            "import sys; "
            "sys.path.insert(0, '.'); "
            "import context_selector; "
            "rows = ["
            "{'context_tokens': 8192, 'status': 'pass', 'quality_score_pct': 72, 'generation_tokens_per_second': 9}, "
            "{'context_tokens': 16384, 'status': 'pass', 'quality_score_pct': 59, 'generation_tokens_per_second': 8}, "
            "{'context_tokens': 32768, 'status': 'oom', 'quality_score_pct': 0, 'generation_tokens_per_second': 0}"
            "]; "
            "assert context_selector.choose_context_tier(rows) == 8192; "
            "assert context_selector.choose_context_tier([]) == 0"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd=temp,
            text=True,
            capture_output=True,
            timeout=5,
        )
    return result.returncode == 0


def _response_excerpt(response_text: str) -> str:
    return response_text.replace("\r", " ").replace("\n", "\\n")[:500]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
