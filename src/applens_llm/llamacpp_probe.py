from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    return record


def _first_avg_ts(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    value = rows[0].get("avg_ts") or 0
    return float(value)
