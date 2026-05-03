from __future__ import annotations

import json
import platform
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request

from applens_llm.schemas import validate_payload


def build_chat_payload(model: str, prompt: str, *, max_tokens: int = 256) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }


def run_openai_chat_benchmark(
    *,
    endpoint: str,
    model_name: str,
    prompt: str,
    output_path: Path,
    max_tokens: int = 256,
    timeout_seconds: int = 120,
    backend: str = "jan",
    model_path: str = "local",
    quantization: str = "unknown",
) -> dict[str, Any]:
    payload = build_chat_payload(model_name, prompt, max_tokens=max_tokens)
    url = endpoint.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    with request.urlopen(req, timeout=timeout_seconds) as response:
        response_json = json.loads(response.read().decode("utf-8"))
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    record = build_benchmark_record(
        run_id=f"applens-llm-{uuid.uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        host=_host_profile(),
        runtime={
            "backend": backend,
            "build": "unknown",
            "command": f"POST {url}",
        },
        model={
            "name": model_name,
            "path": model_path,
            "quantization": quantization,
        },
        response_json=response_json,
        latency_ms=latency_ms,
        vram_used_mb=0,
        temperature_c=0,
        notes="OpenAI-compatible chat completion benchmark.",
    )
    validate_payload("benchmark-record", record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def build_benchmark_record(
    *,
    run_id: str,
    created_at: str,
    host: dict[str, Any],
    runtime: dict[str, Any],
    model: dict[str, Any],
    response_json: dict[str, Any],
    latency_ms: float,
    vram_used_mb: int,
    temperature_c: int,
    notes: str,
) -> dict[str, Any]:
    usage = response_json.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    seconds = max(latency_ms / 1000, 0.001)

    return {
        "schema_version": "0.1",
        "run_id": run_id,
        "created_at": created_at,
        "host": host,
        "runtime": runtime,
        "model": model,
        "workload": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "metrics": {
            "prompt_tokens_per_second": round(prompt_tokens / seconds, 2),
            "generation_tokens_per_second": round(completion_tokens / seconds, 2),
            "latency_ms": latency_ms,
            "vram_used_mb": vram_used_mb,
            "temperature_c": temperature_c,
        },
        "outcome": {
            "status": "pass",
            "notes": notes,
        },
    }


def _host_profile() -> dict[str, Any]:
    return {
        "name": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "gpu": "unknown",
        "vram_mb": 0,
    }
