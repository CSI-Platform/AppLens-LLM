from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from applens_llm.schemas import validate_payload


DEFAULT_GENERATION_KWARGS = {
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0,
    "top_p": 1,
}

TASK_MAX_GEN_TOKS = {
    "arc_challenge_chat": 128,
    "gsm8k": 512,
    "ifeval": 1280,
    "longbench_v2_screening": 1024,
}


@dataclass(frozen=True)
class CommandSpec:
    args: list[str]
    output_dir: Path


@dataclass(frozen=True)
class LoglikelihoodSupport:
    supported: bool
    reason: str


@dataclass(frozen=True)
class ParsedLmEvalResults:
    result_path: Path | None
    sample_paths: list[Path]
    metrics: dict[str, Any]
    effective_samples: int | None


def run_benchmark_suite(
    *,
    plan_path: Path,
    output_path: Path,
    lm_eval_binary: Path,
    endpoint_base: str | None = None,
    task_ids: list[str] | None = None,
    local_screening_limit: int | None = None,
    dry_run: bool = False,
    allow_unsupported: bool = True,
    timeout_seconds: int = 900,
    use_llamacpp_proxy: bool = False,
    proxy_host: str = "127.0.0.1",
    proxy_port: int = 18081,
) -> dict[str, Any]:
    suite = json.loads(plan_path.read_text(encoding="utf-8"))
    artifact_root = Path(suite["output_contract"]["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = Path.cwd() / artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)

    original_endpoint = endpoint_base or str(suite["runtime_lane"]["endpoint"])
    endpoint = _proxy_endpoint(proxy_host, proxy_port) if use_llamacpp_proxy else original_endpoint
    selected = _select_tasks(suite["benchmark_plan"]["tasks"], task_ids)
    task_results = []
    proxy_process = None
    proxy_owned = False
    try:
        if use_llamacpp_proxy and not dry_run:
            proxy_process, proxy_owned = _ensure_proxy(
                host=proxy_host,
                port=proxy_port,
                upstream_base_url=_server_base_from_endpoint(original_endpoint),
                artifact_root=artifact_root,
            )
        for task in selected:
            task_results.append(
                _run_task(
                    suite=suite,
                    task=task,
                    artifact_root=artifact_root,
                    lm_eval_binary=lm_eval_binary,
                    endpoint_base=endpoint,
                    local_screening_limit=local_screening_limit,
                    dry_run=dry_run,
                    allow_unsupported=allow_unsupported,
                    timeout_seconds=timeout_seconds,
                )
            )
    finally:
        if proxy_process is not None and proxy_owned:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_process.kill()

    result = build_suite_result(
        suite=suite,
        plan_path=plan_path,
        artifact_root=artifact_root,
        task_results=task_results,
        runner_context={
            "lm_eval_binary": str(lm_eval_binary),
            "endpoint_base": endpoint,
            "proxy_used": use_llamacpp_proxy,
            "local_screening_limit": local_screening_limit,
            "allow_unsupported": allow_unsupported,
        },
    )
    validate_payload("benchmark-suite-result", result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_lm_eval_command(
    *,
    lm_eval_binary: Path,
    suite: dict[str, Any],
    task: dict[str, Any],
    output_dir: Path,
    endpoint_base: str,
    local_screening_limit: int | None = None,
) -> CommandSpec:
    required_lm_call = task["required_lm_call"]
    if required_lm_call not in {"generate_until", "loglikelihood"}:
        raise ValueError("build_lm_eval_command only supports lm-eval generation and loglikelihood tasks")

    model_name = Path(str(suite["model"]["path"])).name
    if model_name in {"", ".", "local"} or suite["model"]["path"].startswith("sanitized/"):
        model_name = suite["model"]["display_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if required_lm_call == "loglikelihood":
        model_args = ",".join(
            [
                f"model={model_name}",
                f"base_url={_join_endpoint(endpoint_base, 'completions')}",
                "tokenizer_backend=auto",
                "tokenized_requests=False",
                "num_concurrent=1",
                "max_retries=1",
                f"max_length={suite['runtime_lane']['context_tokens']}",
            ]
        )
        args = [
            str(lm_eval_binary),
            "run",
            "--model",
            "local-completions",
            "--model_args",
            model_args,
            "--tasks",
            str(task["task_ref"]),
            "--output_path",
            str(output_dir),
            "--log_samples",
        ]
        if local_screening_limit is not None:
            args.extend(["--limit", str(local_screening_limit)])
        return CommandSpec(args=args, output_dir=output_dir)

    max_gen_toks = TASK_MAX_GEN_TOKS.get(task["task_id"], 512)
    model_args = ",".join(
        [
            f"model={model_name}",
            f"base_url={_join_endpoint(endpoint_base, 'chat/completions')}",
            "tokenized_requests=False",
            "num_concurrent=1",
            "max_retries=1",
            "eos_string=<|im_end|>",
            f"max_gen_toks={max_gen_toks}",
        ]
    )
    gen_kwargs = dict(DEFAULT_GENERATION_KWARGS)
    gen_kwargs["max_gen_toks"] = max_gen_toks

    args = [
        str(lm_eval_binary),
        "run",
        "--model",
        "local-chat-completions",
        "--model_args",
        model_args,
        "--tasks",
        str(task["task_ref"]),
        "--apply_chat_template",
        "--gen_kwargs",
        json.dumps(gen_kwargs),
        "--output_path",
        str(output_dir),
        "--log_samples",
    ]
    if local_screening_limit is not None:
        args.extend(["--limit", str(local_screening_limit)])
    return CommandSpec(args=args, output_dir=output_dir)


def classify_loglikelihood_probe(payload: dict[str, Any]) -> LoglikelihoodSupport:
    try:
        logprobs = payload["choices"][0]["logprobs"]
    except (KeyError, IndexError, TypeError):
        return LoglikelihoodSupport(False, "missing_logprobs")

    token_logprobs = logprobs.get("token_logprobs")
    text_offsets = logprobs.get("text_offset")
    if isinstance(token_logprobs, list) and isinstance(text_offsets, list) and len(token_logprobs) > 1:
        return LoglikelihoodSupport(True, "legacy_prompt_token_logprobs")

    content = logprobs.get("content")
    prompt_tokens = ((payload.get("usage") or {}).get("prompt_tokens")) or 0
    if isinstance(content, list) and len(content) <= 1 and prompt_tokens > len(content):
        return LoglikelihoodSupport(False, "llama_cpp_generated_token_logprobs_only")
    return LoglikelihoodSupport(False, "unrecognized_logprob_shape")


def probe_loglikelihood_support(endpoint_base: str, model_name: str, timeout_seconds: int = 30) -> LoglikelihoodSupport:
    tokenizer_support = _probe_remote_tokenizer_support(endpoint_base, timeout_seconds=timeout_seconds)
    if not tokenizer_support.supported:
        return tokenizer_support

    payload = {
        "model": model_name,
        "prompt": "Question: Which letter comes after A? Answer: B",
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 5,
        "echo": True,
    }
    request = Request(
        _join_endpoint(endpoint_base, "completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            return classify_loglikelihood_probe(json.loads(response.read().decode("utf-8")))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return LoglikelihoodSupport(False, f"probe_error:{exc.__class__.__name__}")


def parse_lm_eval_results(output_dir: Path, task_ref: str) -> ParsedLmEvalResults:
    result_files = sorted(output_dir.rglob("results_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    sample_paths = sorted(output_dir.rglob(f"samples_{task_ref}_*.jsonl"))
    if not result_files:
        return ParsedLmEvalResults(None, sample_paths, {}, None)
    result_path = result_files[0]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = dict((payload.get("results") or {}).get(task_ref) or {})
    n_samples = ((payload.get("n-samples") or {}).get(task_ref) or {}).get("effective")
    return ParsedLmEvalResults(result_path, sample_paths, metrics, n_samples if isinstance(n_samples, int) else None)


def build_suite_result(
    *,
    suite: dict[str, Any],
    plan_path: Path,
    artifact_root: Path,
    task_results: list[dict[str, Any]],
    runner_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "total": len(task_results),
        "passed": sum(1 for item in task_results if item["status"] == "pass"),
        "failed": sum(1 for item in task_results if item["status"] == "fail"),
        "unsupported": sum(1 for item in task_results if item["status"] == "unsupported"),
        "pending": sum(1 for item in task_results if item["status"] == "pending"),
        "errored": sum(1 for item in task_results if item["status"] in {"error", "timeout"}),
    }
    status = _suite_status(summary)
    return {
        "schema_version": "0.1",
        "suite_run_id": suite["suite_run_id"],
        "created_at": _utc_now(),
        "status": status,
        "plan_path": str(plan_path),
        "artifact_root": str(artifact_root),
        "runner_context": runner_context
        or {
            "lm_eval_binary": "lm_eval",
            "endpoint_base": str((suite.get("runtime_lane") or {}).get("endpoint") or "unknown"),
            "proxy_used": False,
            "local_screening_limit": None,
            "allow_unsupported": True,
        },
        "task_results": task_results,
        "summary": summary,
    }


def _run_task(
    *,
    suite: dict[str, Any],
    task: dict[str, Any],
    artifact_root: Path,
    lm_eval_binary: Path,
    endpoint_base: str,
    local_screening_limit: int | None,
    dry_run: bool,
    allow_unsupported: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    output_dir = artifact_root / "lm-eval" / task["task_id"]
    base = _base_task_result(task)
    if dry_run:
        return base | {"status": "pending", "notes": "Dry run: command was not executed."}

    if task["runner"] != "lm-evaluation-harness":
        return _unsupported(base, "unsupported_runner", f"Runner {task['runner']} is not implemented in this local wrapper yet.")

    if task["required_lm_call"] == "loglikelihood":
        model_name = Path(str(suite["model"]["path"])).name or suite["model"]["display_name"]
        support = probe_loglikelihood_support(endpoint_base, model_name)
        if not support.supported:
            result = _unsupported(base, "unsupported_loglikelihood", f"Runtime loglikelihood support check failed: {support.reason}.")
            if not allow_unsupported:
                result["status"] = "fail"
            return result

    if task["required_lm_call"] not in {"generate_until", "loglikelihood"}:
        return _unsupported(base, "unsupported_runner", f"{task['required_lm_call']} is not executable by this wrapper yet.")

    command = build_lm_eval_command(
        lm_eval_binary=lm_eval_binary,
        suite=suite,
        task=task,
        output_dir=output_dir,
        endpoint_base=endpoint_base,
        local_screening_limit=local_screening_limit,
    )
    logs = artifact_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{task['task_id']}.stdout.log"
    stderr_path = logs / f"{task['task_id']}.stderr.log"
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "dummy")
    env["PYTHONIOENCODING"] = "utf-8"

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command.args,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        return base | {
            "status": "timeout",
            "failure_modes": ["timeout"],
            "command": command.args,
            "returncode": None,
            "local_metrics": _local_metrics(elapsed, None),
            "artifacts": [str(stdout_path), str(stderr_path)],
            "notes": f"Timed out after {timeout_seconds} seconds.",
        }

    elapsed = time.monotonic() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    parsed = parse_lm_eval_results(output_dir, task["task_ref"])
    local_metrics = _local_metrics(elapsed, parsed.effective_samples)
    artifacts = [str(stdout_path), str(stderr_path)]
    artifacts.extend(str(path) for path in parsed.sample_paths)
    if parsed.result_path:
        artifacts.append(str(parsed.result_path))

    if completed.returncode != 0:
        return base | {
            "status": "error",
            "failure_modes": ["command_failed"],
            "command": command.args,
            "returncode": completed.returncode,
            "metrics": parsed.metrics,
            "local_metrics": local_metrics,
            "effective_samples": parsed.effective_samples,
            "artifacts": artifacts,
            "notes": f"lm-eval exited non-zero after {elapsed:.1f}s.",
        }
    if parsed.result_path is None:
        return base | {
            "status": "error",
            "failure_modes": ["result_missing"],
            "command": command.args,
            "returncode": completed.returncode,
            "local_metrics": local_metrics,
            "artifacts": artifacts,
            "notes": "lm-eval exited zero but no results_*.json was found.",
        }
    return base | {
        "status": "pass",
        "failure_modes": ["none"],
        "command": command.args,
        "returncode": completed.returncode,
        "metrics": parsed.metrics,
        "local_metrics": local_metrics,
        "effective_samples": parsed.effective_samples,
        "artifacts": artifacts,
        "notes": f"Completed in {elapsed:.1f}s.",
    }


def _base_task_result(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "benchmark": task["benchmark"],
        "category": task["category"],
        "runner": task["runner"],
        "required_lm_call": task["required_lm_call"],
        "status": "pending",
        "failure_modes": [],
        "command": [],
        "returncode": None,
        "metrics": {},
        "local_metrics": {},
        "effective_samples": None,
        "artifacts": [],
        "notes": "",
    }


def _unsupported(base: dict[str, Any], failure_mode: str, notes: str) -> dict[str, Any]:
    return base | {
        "status": "unsupported",
        "failure_modes": [failure_mode],
        "notes": notes,
    }


def _select_tasks(tasks: list[dict[str, Any]], task_ids: list[str] | None) -> list[dict[str, Any]]:
    if not task_ids:
        return tasks
    wanted = set(task_ids)
    selected = [task for task in tasks if task["task_id"] in wanted]
    missing = wanted - {task["task_id"] for task in selected}
    if missing:
        raise ValueError(f"Unknown suite task_id(s): {', '.join(sorted(missing))}")
    return selected


def _local_metrics(elapsed_seconds: float, effective_samples: int | None) -> dict[str, float]:
    metrics = {"wall_time_seconds": round(elapsed_seconds, 3)}
    if effective_samples and elapsed_seconds > 0:
        metrics["effective_samples_per_second"] = round(effective_samples / elapsed_seconds, 6)
    return metrics


def _join_endpoint(endpoint_base: str, suffix: str) -> str:
    base = endpoint_base.rstrip("/")
    suffix = suffix.lstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{suffix}"
    return f"{base}/v1/{suffix}"


def _proxy_endpoint(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def _server_base_from_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint[:-3] if endpoint.endswith("/v1") else endpoint


def _probe_remote_tokenizer_support(endpoint_base: str, timeout_seconds: int = 30) -> LoglikelihoodSupport:
    server_base = _server_base_from_endpoint(endpoint_base)
    try:
        info_request = Request(
            f"{server_base}/tokenizer_info",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with urlopen(info_request, timeout=timeout_seconds) as response:  # noqa: S310
            info = json.loads(response.read().decode("utf-8"))
        if not isinstance(info, dict) or "eos_token" not in info:
            return LoglikelihoodSupport(False, "missing_remote_tokenizer_info")

        tokenize_request = Request(
            f"{server_base}/tokenize",
            data=json.dumps({"prompt": "test", "add_special_tokens": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(tokenize_request, timeout=timeout_seconds) as response:  # noqa: S310
            tokens = json.loads(response.read().decode("utf-8")).get("tokens")
        if isinstance(tokens, list):
            return LoglikelihoodSupport(True, "remote_tokenizer_supported")
        return LoglikelihoodSupport(False, "missing_remote_tokenizer_tokens")
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return LoglikelihoodSupport(False, f"remote_tokenizer_probe_error:{exc.__class__.__name__}")


def _ensure_proxy(
    *,
    host: str,
    port: int,
    upstream_base_url: str,
    artifact_root: Path,
) -> tuple[subprocess.Popen[str] | None, bool]:
    if _is_port_open(host, port):
        return None, False
    logs = artifact_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / "llamacpp-lmeval-proxy.stdout.log").open("w", encoding="utf-8")
    stderr = (logs / "llamacpp-lmeval-proxy.stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "applens_llm.llamacpp_lmeval",
            "--listen-host",
            host,
            "--listen-port",
            str(port),
            "--upstream-base-url",
            upstream_base_url,
        ],
        text=True,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if _is_port_open(host, port):
            return process, True
        if process.poll() is not None:
            raise RuntimeError("llama.cpp lm-eval proxy exited during startup")
        time.sleep(0.25)
    process.terminate()
    raise TimeoutError("llama.cpp lm-eval proxy did not start within 10 seconds")


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def _suite_status(summary: dict[str, int]) -> str:
    if summary["pending"]:
        return "pending"
    if summary["errored"] or summary["failed"]:
        return "fail"
    if summary["passed"] and summary["unsupported"]:
        return "partial"
    if summary["unsupported"]:
        return "blocked"
    return "pass"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
