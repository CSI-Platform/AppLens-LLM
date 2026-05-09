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
    engine: str = "jan",
    backend: str = "unknown",
    model_path: str = "local",
    quantization: str = "unknown",
    driver_evidence: list[dict[str, Any]] | None = None,
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
    host = _host_profile()
    host["driver_evidence"] = driver_evidence or []
    devices_used = host["hardware_topology"]["usable_inference_capacity"]["preferred_accelerator_ids"]

    record = build_benchmark_record(
        run_id=f"applens-llm-{uuid.uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        host=host,
        runtime={
            "engine": engine,
            "backend": backend,
            "build": "unknown",
            "command": f"POST {url}",
            "devices_used": devices_used,
            "mixed_device_offload": {
                "attempted": False,
                "worked": False,
                "strategy": "unknown",
                "notes": "OpenAI-compatible endpoint benchmark did not inspect backend offload internals.",
            },
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
        device_memory_used_mb=[],
        cpu_spill_mb=0,
        thermal_notes="No thermal telemetry collected by the endpoint benchmark.",
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
    device_memory_used_mb: list[dict[str, Any]] | None = None,
    cpu_spill_mb: int = 0,
    thermal_notes: str = "No thermal telemetry collected.",
    status: str = "pass",
    fallback_occurred: bool = False,
    failure_modes: list[str] | None = None,
    telemetry_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    usage = response_json.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    seconds = max(latency_ms / 1000, 0.001)

    host_payload = dict(host)
    host_payload.setdefault("driver_evidence", [])

    record = {
        "schema_version": "0.1",
        "run_id": run_id,
        "created_at": created_at,
        "host": host_payload,
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
            "device_memory_used_mb": device_memory_used_mb or [],
            "cpu_spill_mb": cpu_spill_mb,
            "thermal_notes": thermal_notes,
            "temperature_c": temperature_c,
        },
        "outcome": {
            "status": status,
            "fallback_occurred": fallback_occurred,
            "failure_modes": failure_modes or ["none"],
            "notes": notes,
        },
    }
    if telemetry_sources is not None:
        record["telemetry_sources"] = telemetry_sources
    return record


def _host_profile() -> dict[str, Any]:
    hardware_topology = {
        "accelerators": [
            {
                "accelerator_id": "unknown-accelerator-0",
                "kind": "unknown",
                "vendor": "unknown",
                "name": "Unverified local endpoint accelerator",
                "present": False,
                "api_support": ["unknown"],
                "memory": {
                    "physical_dedicated_vram_mb": 0,
                    "vgm_reserved_mb": 0,
                    "shared_graphics_memory_mb": 0,
                    "reported_total_graphics_memory_mb": 0,
                    "estimated_usable_inference_memory_mb": 0,
                    "confidence": "inferred",
                },
                "verification": [
                    {
                        "source_type": "inventory",
                        "notes": "Generic endpoint benchmark does not collect local accelerator inventory.",
                    }
                ],
            }
        ],
        "usable_inference_capacity": {
            "estimated_usable_memory_mb": 0,
            "confidence": "inferred",
            "preferred_accelerator_ids": ["unknown-accelerator-0"],
            "mixed_device_pooling": "unverified",
            "verification": [
                {
                    "source_type": "benchmark",
                    "command": "POST /v1/chat/completions",
                    "notes": "Endpoint responded, but device placement and usable memory were not observed.",
                }
            ],
        },
        "memory_claims": [],
    }
    return {
        "name": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "gpu": "unknown",
        "vram_mb": 0,
        "hardware_topology": hardware_topology,
    }
