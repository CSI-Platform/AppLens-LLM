from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from applens_llm.benchmark_suite_runner import probe_loglikelihood_support
from applens_llm.schemas import validate_payload


CommandRunner = Callable[[list[str]], tuple[int, str, str]]
EndpointProbe = Callable[[str, str], dict[str, Any]]


EXTERNAL_RUNNER_COMMANDS = {
    "bfcl-eval": "bfcl",
    "bigcodebench": "bigcodebench",
    "longbench-v2": "longbench-v2",
    "ruler": "ruler",
}


def build_benchmark_readiness(
    *,
    suite: dict[str, Any],
    lm_eval_binary: Path,
    runner_paths: dict[str, Path | str | None] | None = None,
    command_runner: CommandRunner | None = None,
    endpoint_probe: EndpointProbe | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    command_runner = command_runner or _run_command
    endpoint_probe = endpoint_probe or _default_endpoint_probe
    runner_paths = runner_paths or _default_runner_paths()
    tasks = list(suite["benchmark_plan"]["tasks"])
    lm_eval_tasks = [task for task in tasks if task["runner"] == "lm-evaluation-harness"]
    task_refs = ",".join(task["task_ref"] for task in lm_eval_tasks)
    lm_eval_status = _check_lm_eval(lm_eval_binary, task_refs, command_runner)
    endpoint = endpoint_probe(str(suite["runtime_lane"]["endpoint"]), str(suite["model"]["display_name"]))
    task_readiness = [
        _task_readiness(task, lm_eval_status, endpoint, runner_paths)
        for task in tasks
    ]
    runner_readiness = _runner_readiness(tasks, task_readiness, lm_eval_status, runner_paths)
    report = {
        "schema_version": "0.1",
        "created_at": created_at or _utc_now(),
        "suite_run_id": suite["suite_run_id"],
        "status": _overall_status(task_readiness),
        "lm_eval": lm_eval_status,
        "endpoint_capabilities": endpoint,
        "runner_readiness": runner_readiness,
        "task_readiness": task_readiness,
        "next_actions": _next_actions(task_readiness, runner_readiness, endpoint),
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }
    validate_payload("benchmark-readiness", report)
    return report


def write_benchmark_readiness(
    *,
    plan_path: Path,
    output_path: Path,
    lm_eval_binary: Path,
    runner_paths: dict[str, Path | str | None] | None = None,
    command_runner: CommandRunner | None = None,
    endpoint_probe: EndpointProbe | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    suite = json.loads(plan_path.read_text(encoding="utf-8"))
    report = build_benchmark_readiness(
        suite=suite,
        lm_eval_binary=lm_eval_binary,
        runner_paths=runner_paths,
        command_runner=command_runner,
        endpoint_probe=endpoint_probe,
        created_at=created_at,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def skipped_endpoint_probe(endpoint: str, model_name: str) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "available": False,
        "chat_completions": False,
        "completions": False,
        "remote_tokenizer": False,
        "legacy_loglikelihood": False,
        "notes": f"Endpoint probe skipped for {model_name}.",
    }


def _check_lm_eval(binary: Path, task_refs: str, command_runner: CommandRunner) -> dict[str, str]:
    code, stdout, stderr = command_runner([str(binary), "validate", "--tasks", task_refs])
    if code == 0:
        return {"name": "lm-evaluation-harness", "status": "ready", "path": str(binary), "notes": _first_line(stdout) or "tasks validated"}
    reason = _first_line(stderr) or _first_line(stdout) or "lm-eval validation failed"
    status = "missing" if "not found" in reason.lower() or "no such file" in reason.lower() else "blocked"
    return {"name": "lm-evaluation-harness", "status": status, "path": str(binary), "notes": reason}


def _task_readiness(
    task: dict[str, Any],
    lm_eval_status: dict[str, str],
    endpoint: dict[str, Any],
    runner_paths: dict[str, Path | str | None],
) -> dict[str, str]:
    runner = task["runner"]
    required = task["required_lm_call"]
    status = "ready"
    reason = "ready"
    if runner == "lm-evaluation-harness":
        if lm_eval_status["status"] != "ready":
            status = "missing" if lm_eval_status["status"] == "missing" else "blocked"
            reason = "lm_eval_not_ready"
        elif required == "generate_until" and not endpoint["chat_completions"]:
            status = "blocked"
            reason = "endpoint_missing_chat_completions"
        elif required == "loglikelihood" and not endpoint["legacy_loglikelihood"]:
            status = "blocked"
            reason = "endpoint_missing_loglikelihood"
    else:
        configured = runner_paths.get(runner)
        if not configured:
            status = "missing"
            reason = f"{runner}_runner_missing"
    return {
        "task_id": task["task_id"],
        "benchmark": task["benchmark"],
        "runner": runner,
        "required_lm_call": required,
        "status": status,
        "reason": reason,
    }


def _runner_readiness(
    tasks: list[dict[str, Any]],
    task_readiness: list[dict[str, str]],
    lm_eval_status: dict[str, str],
    runner_paths: dict[str, Path | str | None],
) -> list[dict[str, Any]]:
    task_status_by_id = {task["task_id"]: task for task in task_readiness}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[task["runner"]].append(task)
    rows = []
    for runner, runner_tasks in grouped.items():
        statuses = [task_status_by_id[task["task_id"]]["status"] for task in runner_tasks]
        if runner == "lm-evaluation-harness":
            if lm_eval_status["status"] == "missing":
                status = "missing"
                notes = lm_eval_status["notes"]
            elif all(status == "ready" for status in statuses):
                status = "ready"
                notes = "lm-eval tasks and endpoint capabilities are ready."
            elif any(status == "ready" for status in statuses):
                status = "partial"
                notes = "Some lm-eval tasks are ready; loglikelihood or endpoint support is blocked."
            else:
                status = "blocked"
                notes = "lm-eval tasks are blocked by endpoint or validation readiness."
        else:
            status = "ready" if runner_paths.get(runner) else "missing"
            notes = "runner configured" if status == "ready" else f"{runner} runner is not configured."
        rows.append(
            {
                "runner": runner,
                "status": status,
                "task_ids": [task["task_id"] for task in runner_tasks],
                "notes": notes,
            }
        )
    return sorted(rows, key=lambda row: row["runner"])


def _default_endpoint_probe(endpoint: str, model_name: str) -> dict[str, Any]:
    available = _probe_models(endpoint)
    support = probe_loglikelihood_support(endpoint, model_name) if available else None
    reason = support.reason if support else "endpoint unavailable"
    return {
        "endpoint": endpoint,
        "available": available,
        "chat_completions": available,
        "completions": available,
        "remote_tokenizer": bool(support and "remote_tokenizer" not in support.reason),
        "legacy_loglikelihood": bool(support and support.supported),
        "notes": reason,
    }


def _probe_models(endpoint: str) -> bool:
    request = Request(_join_endpoint(endpoint, "models"), headers={"Content-Type": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=5) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


def _default_runner_paths() -> dict[str, str | None]:
    return {
        runner: shutil.which(command)
        for runner, command in EXTERNAL_RUNNER_COMMANDS.items()
    }


def _run_command(args: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(args, text=True, capture_output=True, check=False)
    except OSError as exc:
        return 127, "", str(exc)
    return completed.returncode, completed.stdout, completed.stderr


def _overall_status(task_readiness: list[dict[str, str]]) -> str:
    statuses = {task["status"] for task in task_readiness}
    if statuses == {"ready"}:
        return "ready"
    if "ready" in statuses:
        return "partial"
    return "blocked"


def _next_actions(
    task_readiness: list[dict[str, str]],
    runner_readiness: list[dict[str, Any]],
    endpoint: dict[str, Any],
) -> list[str]:
    actions = []
    missing_runners = [runner["runner"] for runner in runner_readiness if runner["status"] == "missing"]
    if missing_runners:
        actions.append(f"Install or configure official benchmark runners: {', '.join(_runner_display_name(runner) for runner in missing_runners)}.")
    if any(task["reason"] == "endpoint_missing_loglikelihood" for task in task_readiness):
        actions.append("Use a runtime exposing prompt-token logprobs plus tokenizer endpoints before running loglikelihood tasks.")
    if not endpoint["chat_completions"]:
        actions.append("Start the OpenAI-compatible llama.cpp endpoint before running generation tasks.")
    if not actions:
        actions.append("Run benchmark-suite-run with this plan.")
    return actions


def _join_endpoint(endpoint_base: str, suffix: str) -> str:
    base = endpoint_base.rstrip("/")
    suffix = suffix.lstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{suffix}"
    return f"{base}/v1/{suffix}"


def _runner_display_name(runner: str) -> str:
    return {
        "bfcl-eval": "BFCL",
        "bigcodebench": "BigCodeBench",
        "longbench-v2": "LongBench v2",
        "ruler": "RULER",
    }.get(runner, runner)


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
