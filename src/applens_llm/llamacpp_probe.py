from __future__ import annotations

import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


DEVICE_RE = re.compile(
    r"^\s*(?P<device>[A-Za-z]+[0-9]+):\s+"
    r"(?P<name>.+?)\s+"
    r"\((?P<total>[0-9]+)\s+MiB,\s+(?P<free>[0-9]+)\s+MiB free\)"
)


def parse_llamacpp_devices(output: str) -> list[dict[str, Any]]:
    devices = []
    for line in output.splitlines():
        match = DEVICE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        devices.append(
            {
                "llama_device": match.group("device"),
                "name": name,
                "memory_total_mib": int(match.group("total")),
                "memory_free_mib": int(match.group("free")),
                "accelerator_id": infer_accelerator_id(name),
            }
        )
    return devices


def infer_accelerator_id(device_name: str) -> str:
    lowered = device_name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "rtx" in lowered:
        return "nvidia-dgpu-0"
    if "amd" in lowered or "radeon" in lowered or "890m" in lowered:
        return "amd-igpu-0"
    return "unknown-accelerator-0"


def build_llamacpp_bench_command(
    *,
    binary: Path,
    model: Path,
    device: str,
    prompt_tokens: int,
    generation_tokens: int,
    repetitions: int,
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
        "-p",
        str(prompt_tokens),
        "-n",
        str(generation_tokens),
        "-r",
        str(repetitions),
        "-t",
        str(threads),
        "-o",
        "json",
    ]


def parse_llama_bench_json(output: str) -> list[dict[str, Any]]:
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        rows = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def summarize_llama_bench_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_rows = [row for row in rows if int(row.get("n_prompt") or 0) > 0]
    generation_rows = [row for row in rows if int(row.get("n_gen") or 0) > 0]
    devices = sorted({str(row.get("devices")) for row in rows if row.get("devices")})
    return {
        "prompt_tokens_per_second": _first_avg_ts(prompt_rows),
        "generation_tokens_per_second": _first_avg_ts(generation_rows),
        "devices": devices,
        "model_type": rows[0].get("model_type") if rows else None,
        "build_number": rows[0].get("build_number") if rows else None,
        "build_commit": rows[0].get("build_commit") if rows else None,
        "backend": rows[0].get("backends") if rows else None,
    }


def build_benchmark_record_from_llamacpp_record(
    *,
    raw_record: dict[str, Any],
    machine_profile: dict[str, Any],
    devices_inventory: dict[str, Any] | None = None,
    model_name: str | None = None,
    quantization: str = "unknown",
) -> dict[str, Any]:
    llamacpp = raw_record.get("llamacpp") or {}
    settings = llamacpp.get("settings") or {}
    summary = llamacpp.get("summary") or {}
    prompt_tokens = int(settings.get("prompt_tokens") or 0)
    generation_tokens = int(settings.get("generation_tokens") or 0)
    prompt_tps = float(summary.get("prompt_tokens_per_second") or 0)
    generation_tps = float(summary.get("generation_tokens_per_second") or 0)
    latency_ms = _estimated_latency_ms(prompt_tokens, prompt_tps, generation_tokens, generation_tps)
    devices_used = _resolve_accelerator_ids(
        device_selector=str(llamacpp.get("device") or "unknown"),
        observed_devices=[str(device) for device in summary.get("devices") or []],
        devices_inventory=devices_inventory,
    )
    backend = _infer_backend(str(summary.get("backend") or ""), str(llamacpp.get("device") or ""))
    status, failure_modes, notes = _llamacpp_outcome(raw_record)
    build = _llamacpp_build(summary)
    model = raw_record.get("model") or {}
    model_path = str(model.get("path") or "")

    record = {
        "schema_version": "0.1",
        "run_id": f"llamacpp-{_slug(str(raw_record.get('label') or 'bench'))}-{uuid.uuid4().hex[:8]}",
        "created_at": raw_record.get("created_at") or _utc_now(),
        "host": _benchmark_host_from_machine_profile(machine_profile),
        "runtime": {
            "engine": "llama.cpp",
            "backend": backend,
            "build": build,
            "command": " ".join(str(part) for part in raw_record.get("command") or []),
            "devices_used": devices_used,
            "mixed_device_offload": {
                "attempted": len(devices_used) > 1,
                "worked": status == "pass" and len(devices_used) > 1,
                "strategy": "single_device" if len(devices_used) <= 1 else "runtime_default",
                "notes": "llama.cpp benchmark selected one device with -dev unless multiple devices were observed.",
            },
        },
        "model": {
            "name": model_name or _model_name_from_path(model_path),
            "path": model_path,
            "quantization": quantization,
        },
        "workload": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generation_tokens,
        },
        "metrics": {
            "prompt_tokens_per_second": prompt_tps,
            "generation_tokens_per_second": generation_tps,
            "latency_ms": latency_ms,
            "vram_used_mb": 0,
            "device_memory_used_mb": [],
            "cpu_spill_mb": 0,
            "thermal_notes": "No thermal telemetry was attached to this llama.cpp benchmark record.",
            "temperature_c": 0,
        },
        "outcome": {
            "status": status,
            "fallback_occurred": "fallback" in failure_modes,
            "failure_modes": failure_modes,
            "notes": notes,
        },
        "telemetry_sources": [
            {
                "source": "runtime_log",
                "sampling_interval_seconds": 1,
                "metrics": ["runtime_backend", "tokens_per_second"],
                "notes": "Derived from llama.cpp bench JSON output; memory and thermal telemetry require separate vendor logs.",
            }
        ],
    }
    validate_payload("benchmark-record", record)
    return record


def write_llamacpp_devices(
    *,
    binary: Path,
    output_path: Path,
    from_output: str | None = None,
) -> list[dict[str, Any]]:
    raw_output = from_output if from_output is not None else run_llamacpp_list_devices(binary)
    devices = parse_llamacpp_devices(raw_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"devices": devices}, indent=2) + "\n", encoding="utf-8")
    return devices


def run_llamacpp_list_devices(binary: Path) -> str:
    result = subprocess.run(
        [str(binary), "--list-devices"],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def run_llamacpp_bench(
    *,
    binary: Path,
    model: Path,
    device: str,
    label: str,
    output_path: Path,
    prompt_tokens: int = 512,
    generation_tokens: int = 128,
    repetitions: int = 3,
    gpu_layers: int = 99,
    threads: int = 12,
    disable_vulkan_coopmat: bool = False,
    machine_profile: dict[str, Any] | None = None,
    devices_inventory: dict[str, Any] | None = None,
    benchmark_record_path: Path | None = None,
    model_name: str | None = None,
    quantization: str = "unknown",
) -> dict[str, Any]:
    command = build_llamacpp_bench_command(
        binary=binary,
        model=model,
        device=device,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        repetitions=repetitions,
        gpu_layers=gpu_layers,
        threads=threads,
    )
    env = None
    if disable_vulkan_coopmat:
        import os

        env = os.environ.copy()
        env["GGML_VK_DISABLE_COOPMAT"] = "1"

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result = subprocess.run(command, text=True, capture_output=True, timeout=1800, check=False, env=env)
    raw_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    rows = parse_llama_bench_json(raw_output)
    summary = summarize_llama_bench_rows(rows)
    record = {
        "schema_version": "0.1",
        "label": label,
        "created_at": started_at,
        "command": command,
        "environment": {"GGML_VK_DISABLE_COOPMAT": "1" if disable_vulkan_coopmat else None},
        "process": {"returncode": result.returncode},
        "model": {
            "path": str(model),
            "name": model.name,
            "size_bytes": model.stat().st_size if model.exists() else None,
        },
        "llamacpp": {
            "binary": str(binary),
            "device": device,
            "settings": {
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "repetitions": repetitions,
                "gpu_layers": gpu_layers,
                "threads": threads,
            },
            "summary": summary,
            "rows": rows,
        },
        "raw": {
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
        "privacy": {
            "local_paths_included": True,
            "commit_safe": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if benchmark_record_path is not None:
        if machine_profile is None:
            raise ValueError("--machine-profile is required when --benchmark-record is provided")
        benchmark_record = build_benchmark_record_from_llamacpp_record(
            raw_record=record,
            machine_profile=machine_profile,
            devices_inventory=devices_inventory,
            model_name=model_name,
            quantization=quantization,
        )
        benchmark_record_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_record_path.write_text(json.dumps(benchmark_record, indent=2) + "\n", encoding="utf-8")
    return record


def _first_avg_ts(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    value = rows[0].get("avg_ts") or 0
    return float(value)


def _benchmark_host_from_machine_profile(machine_profile: dict[str, Any]) -> dict[str, Any]:
    platform = machine_profile.get("platform") or {}
    return {
        "name": str(machine_profile.get("machine_id") or machine_profile.get("label") or "unknown-host"),
        "os": str(platform.get("os_family") or "unknown"),
        "gpu": str(platform.get("gpu") or "unknown"),
        "vram_mb": int(platform.get("vram_mb") or 0),
        "driver_evidence": machine_profile.get("driver_evidence", []),
        "hardware_topology": machine_profile["hardware_topology"],
    }


def _estimated_latency_ms(
    prompt_tokens: int,
    prompt_tps: float,
    generation_tokens: int,
    generation_tps: float,
) -> float:
    prompt_ms = (prompt_tokens / prompt_tps * 1000) if prompt_tokens and prompt_tps else 0
    generation_ms = (generation_tokens / generation_tps * 1000) if generation_tokens and generation_tps else 0
    return round(prompt_ms + generation_ms, 2)


def _resolve_accelerator_ids(
    *,
    device_selector: str,
    observed_devices: list[str],
    devices_inventory: dict[str, Any] | None,
) -> list[str]:
    labels = _device_labels(observed_devices) or _device_labels([device_selector])
    inventory = {
        str(device.get("llama_device")): str(device.get("accelerator_id"))
        for device in (devices_inventory or {}).get("devices", [])
        if device.get("llama_device") and device.get("accelerator_id")
    }
    accelerator_ids = [inventory.get(label) for label in labels if inventory.get(label)]
    if not accelerator_ids:
        accelerator_ids = ["unknown-accelerator-0"]
    return sorted(set(accelerator_ids))


def _device_labels(values: list[str]) -> list[str]:
    labels: list[str] = []
    for value in values:
        labels.extend(re.findall(r"[A-Za-z]+[0-9]+", value))
    return labels


def _infer_backend(summary_backend: str, device_selector: str) -> str:
    lowered = f"{summary_backend} {device_selector}".lower()
    for backend in ("cuda", "vulkan", "rocm", "hip", "directml", "metal", "openvino"):
        if backend in lowered:
            return backend
    if "cpu" in lowered:
        return "cpu"
    return "unknown"


def _llamacpp_build(summary: dict[str, Any]) -> str:
    number = summary.get("build_number")
    commit = summary.get("build_commit")
    if number and commit:
        return f"b{number} {commit}"
    if number:
        return f"b{number}"
    return "unknown"


def _llamacpp_outcome(raw_record: dict[str, Any]) -> tuple[str, list[str], str]:
    returncode = int((raw_record.get("process") or {}).get("returncode") or 0)
    rows = (raw_record.get("llamacpp") or {}).get("rows") or []
    raw = raw_record.get("raw") or {}
    combined_output = f"{raw.get('stdout') or ''}\n{raw.get('stderr') or ''}".lower()
    if returncode == 0 and rows:
        return "pass", ["none"], "llama.cpp bench completed and emitted parsed benchmark rows."
    if (
        "out of memory" in combined_output
        or "out_of_device_memory" in combined_output
        or "outofdevicememory" in combined_output
    ):
        return "oom", ["oom"], "llama.cpp reported an out-of-memory condition."
    if "no devices" in combined_output or "unsupported" in combined_output:
        return "fail", ["unsupported_device"], "llama.cpp could not use the requested device."
    if returncode != 0:
        return "crash", ["crash"], f"llama.cpp exited nonzero with return code {returncode}."
    return "fail", ["backend_missing"], "llama.cpp did not emit benchmark rows."


def _model_name_from_path(model_path: str) -> str:
    path = Path(model_path)
    if path.name.lower() == "model.gguf" and path.parent.name:
        return path.parent.name
    return path.stem or "unknown-model"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return slug or "bench"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
