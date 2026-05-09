from __future__ import annotations

import csv
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BYTES_PER_GB = 1024 * 1024 * 1024
RTX_4050_CONTROL_LIMIT_GB = 6
VGM_TARGET_GB = 16


def write_vgm_snapshot(
    output_path: Path,
    *,
    label: str,
    model_roots: list[Path],
    llama_roots: list[Path] | None = None,
    max_models: int = 30,
    video_controllers: list[dict[str, Any]] | None = None,
    registry_video_memory: list[dict[str, Any]] | None = None,
    nvidia_smi_rows: list[dict[str, Any]] | None = None,
    vulkan_devices: list[dict[str, Any]] | None = None,
    llama_binaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    snapshot = build_vgm_snapshot(
        label=label,
        model_roots=model_roots,
        llama_roots=llama_roots,
        max_models=max_models,
        video_controllers=video_controllers,
        registry_video_memory=registry_video_memory,
        nvidia_smi_rows=nvidia_smi_rows,
        vulkan_devices=vulkan_devices,
        llama_binaries=llama_binaries,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


def build_vgm_snapshot(
    *,
    label: str,
    model_roots: list[Path],
    llama_roots: list[Path] | None = None,
    max_models: int = 30,
    video_controllers: list[dict[str, Any]] | None = None,
    registry_video_memory: list[dict[str, Any]] | None = None,
    nvidia_smi_rows: list[dict[str, Any]] | None = None,
    vulkan_devices: list[dict[str, Any]] | None = None,
    llama_binaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    controllers = normalize_video_controllers(
        video_controllers if video_controllers is not None else collect_video_controllers()
    )
    if registry_video_memory is not None:
        registry_rows = registry_video_memory
    elif video_controllers is not None:
        registry_rows = []
    else:
        registry_rows = collect_registry_video_memory()
    registry_memory = normalize_registry_video_memory(registry_rows)
    nvidia_rows = normalize_nvidia_smi_rows(
        nvidia_smi_rows if nvidia_smi_rows is not None else collect_nvidia_smi_rows()
    )
    vulkan = vulkan_devices if vulkan_devices is not None else collect_vulkan_devices()
    llama = llama_binaries if llama_binaries is not None else discover_llama_binaries(llama_roots or [])
    models = []
    for root in model_roots:
        models.extend(find_gguf_models(root, max_items=max_models))
    models = sorted(models, key=lambda item: item["size_gb"], reverse=True)[:max_models]

    vgm_check = build_vgm_check(controllers, registry_memory)
    runtime_readiness = build_runtime_readiness(nvidia_rows, vulkan, llama)

    return {
        "schema_version": "0.1",
        "snapshot_id": f"{label}-{created_at.replace(':', '').replace('-', '')}",
        "label": label,
        "created_at": created_at,
        "host": {
            "name": socket.gethostname(),
            "os": f"{platform.system()} {platform.release()}",
        },
        "video_controllers": controllers,
        "registry_video_memory": registry_memory,
        "nvidia_smi": nvidia_rows,
        "vulkan_devices": vulkan,
        "llama_binaries": llama,
        "local_models": models,
        "vgm_check": vgm_check,
        "runtime_readiness": runtime_readiness,
        "benchmark_plan": build_benchmark_plan(vgm_check, runtime_readiness, models),
        "privacy": {
            "local_paths_included": bool(models or llama),
            "commit_safe": False,
            "notes": "Snapshot output is local evidence under ignored out/ by default. Do not commit raw paths.",
        },
    }


def find_gguf_models(
    root: Path,
    *,
    max_items: int = 30,
    bytes_per_gb: int = DEFAULT_BYTES_PER_GB,
) -> list[dict[str, Any]]:
    if not root.exists():
        return []

    models: list[dict[str, Any]] = []
    for path in root.rglob("*.gguf"):
        if not path.is_file():
            continue
        size_gb = round(path.stat().st_size / bytes_per_gb, 2)
        models.append(
            {
                "path": str(path),
                "name": path.name,
                "size_gb": size_gb,
                "vgm_test_role": classify_model_for_vgm_test(size_gb),
            }
        )

    return sorted(models, key=lambda item: item["size_gb"], reverse=True)[:max_items]


def classify_model_for_vgm_test(size_gb: float) -> str:
    if size_gb <= RTX_4050_CONTROL_LIMIT_GB:
        return "small_control"
    if size_gb <= VGM_TARGET_GB:
        return "vgm_capacity_candidate"
    return "too_large_for_16gb_vgm"


def compare_vgm_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_mb = int(before.get("vgm_check", {}).get("amd_dedicated_memory_mb") or 0)
    after_mb = int(after.get("vgm_check", {}).get("amd_dedicated_memory_mb") or 0)
    vgm_activated = after_mb >= 15_360 and after_mb > before_mb
    has_vulkan_llamacpp = bool(after.get("runtime_readiness", {}).get("has_vulkan_llamacpp"))

    if not vgm_activated:
        next_step = "fix_vgm_setting"
    elif has_vulkan_llamacpp:
        next_step = "run_vulkan_benchmark"
    else:
        next_step = "install_or_build_vulkan_llamacpp"

    return {
        "schema_version": "0.1",
        "before_amd_dedicated_memory_mb": before_mb,
        "after_amd_dedicated_memory_mb": after_mb,
        "amd_dedicated_memory_delta_mb": after_mb - before_mb,
        "vgm_activated": vgm_activated,
        "vulkan_llamacpp_ready": has_vulkan_llamacpp,
        "next_step": next_step,
    }


def build_vgm_check(
    video_controllers: list[dict[str, Any]],
    registry_video_memory: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    amd = next(
        (
            controller
            for controller in video_controllers
            if "amd" in controller.get("name", "").lower()
            and ("radeon" in controller.get("name", "").lower() or "890m" in controller.get("name", "").lower())
        ),
        None,
    )
    amd_mb = int(amd.get("adapter_ram_mb") or 0) if amd else 0
    registry_amd = next(
        (
            item
            for item in (registry_video_memory or [])
            if "amd" in item.get("name", "").lower()
            and ("radeon" in item.get("name", "").lower() or "890m" in item.get("name", "").lower())
        ),
        None,
    )
    if registry_amd:
        amd_mb = max(amd_mb, int(registry_amd.get("hardware_memory_mb") or 0))
    if amd_mb >= 15_360:
        state = "vgm_16gb_or_higher"
    elif 384 <= amd_mb <= 1024:
        state = "default_or_low_vgm"
    elif amd_mb > 0:
        state = "partial_vgm"
    else:
        state = "not_observed"

    return {
        "amd_adapter_name": amd.get("name") if amd else None,
        "amd_dedicated_memory_mb": amd_mb,
        "vgm_16gb_active": amd_mb >= 15_360,
        "state": state,
    }


def build_runtime_readiness(
    nvidia_smi_rows: list[dict[str, Any]],
    vulkan_devices: list[dict[str, Any]],
    llama_binaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "nvidia_cuda_seen": bool(nvidia_smi_rows),
        "vulkan_devices_seen": len(vulkan_devices),
        "has_cuda_llamacpp": any("cuda" in item.get("backends", []) for item in llama_binaries),
        "has_vulkan_llamacpp": any("vulkan" in item.get("backends", []) for item in llama_binaries),
    }


def build_benchmark_plan(
    vgm_check: dict[str, Any],
    runtime_readiness: dict[str, Any],
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    small_controls = [model for model in models if model["vgm_test_role"] == "small_control"]
    capacity_candidates = [model for model in models if model["vgm_test_role"] == "vgm_capacity_candidate"]
    if not vgm_check["vgm_16gb_active"]:
        next_step = "enable_vgm_and_restart"
    elif not runtime_readiness["has_vulkan_llamacpp"]:
        next_step = "install_or_build_vulkan_llamacpp"
    else:
        next_step = "run_small_control_then_capacity_candidate"

    return {
        "next_step": next_step,
        "small_control_model": small_controls[-1] if small_controls else None,
        "capacity_candidate_model": capacity_candidates[0] if capacity_candidates else None,
        "notes": [
            "Use the same prompt and context settings for RTX and Radeon runs.",
            "Treat RTX 4050 as the speed control and Radeon 890M VGM as the capacity test.",
            "A capacity pass means the AMD path loads and generates without CPU spill, OOM, fallback, or crash.",
        ],
    }


def normalize_video_controllers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        adapter_ram_bytes = int(row.get("AdapterRAM") or row.get("adapter_ram_bytes") or 0)
        normalized.append(
            {
                "name": str(row.get("Name") or row.get("name") or "unknown"),
                "adapter_ram_bytes": adapter_ram_bytes,
                "adapter_ram_mb": round(adapter_ram_bytes / (1024 * 1024)),
                "driver_version": str(row.get("DriverVersion") or row.get("driver_version") or "unknown"),
                "video_processor": str(row.get("VideoProcessor") or row.get("video_processor") or "unknown"),
            }
        )
    return normalized


def normalize_nvidia_smi_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "name": str(row.get("name") or "unknown"),
                "memory_total_mib": _parse_mib(row.get("memory.total [MiB]") or row.get("memory_total_mib")),
                "memory_used_mib": _parse_mib(row.get("memory.used [MiB]") or row.get("memory_used_mib")),
                "driver_version": str(row.get("driver_version") or "unknown"),
            }
        )
    return normalized


def normalize_registry_video_memory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        hardware_memory_bytes = int(
            row.get("HardwareInformationMemorySize")
            or row.get("hardware_memory_bytes")
            or row.get("hardware_memory_size")
            or 0
        )
        normalized.append(
            {
                "name": str(row.get("DriverDesc") or row.get("AdapterString") or row.get("name") or "unknown"),
                "hardware_memory_bytes": hardware_memory_bytes,
                "hardware_memory_mb": round(hardware_memory_bytes / (1024 * 1024)),
            }
        )
    return normalized


def collect_video_controllers() -> list[dict[str, Any]]:
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,VideoProcessor | "
        "ConvertTo-Json -Depth 3"
    )
    result = _run_command(["powershell", "-NoProfile", "-Command", script])
    if not result:
        return []
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def collect_nvidia_smi_rows() -> list[dict[str, Any]]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    reader = csv.reader(output.splitlines())
    rows = []
    for row in reader:
        if len(row) != 4:
            continue
        rows.append(
            {
                "name": row[0].strip(),
                "memory_total_mib": row[1].strip(),
                "memory_used_mib": row[2].strip(),
                "driver_version": row[3].strip(),
            }
        )
    return rows


def collect_registry_video_memory() -> list[dict[str, Any]]:
    script = (
        "Get-ChildItem 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}' "
        "-ErrorAction SilentlyContinue | ForEach-Object { "
        "$p = Get-ItemProperty $_.PsPath -ErrorAction SilentlyContinue; "
        "[pscustomobject]@{DriverDesc=$p.DriverDesc;HardwareInformationMemorySize=$p.'HardwareInformation.qwMemorySize'} "
        "} | Where-Object { $_.DriverDesc -and $_.HardwareInformationMemorySize } | ConvertTo-Json -Depth 3"
    )
    result = _run_command(["powershell", "-NoProfile", "-Command", script])
    if not result:
        return []
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def collect_vulkan_devices() -> list[dict[str, Any]]:
    output = _run_command(["vulkaninfo", "--summary"])
    if not output:
        return []

    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("GPU") and line.endswith(":"):
            if current:
                devices.append(current)
            current = {"index": _parse_gpu_index(line)}
        elif current is not None and "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
            if key == "deviceName":
                current["device_name"] = value
            elif key == "deviceType":
                current["device_type"] = value
            elif key == "driverName":
                current["driver_name"] = value
            elif key == "apiVersion":
                current["api_version"] = value
    if current:
        devices.append(current)
    return devices


def discover_llama_binaries(roots: list[Path]) -> list[dict[str, Any]]:
    binaries: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("llama-*.exe"):
            if not path.is_file():
                continue
            binary_dir = path.parent
            backends = []
            if (binary_dir / "ggml-cuda.dll").exists():
                backends.append("cuda")
            if (binary_dir / "ggml-vulkan.dll").exists():
                backends.append("vulkan")
            if (binary_dir / "ggml-hip.dll").exists() or (binary_dir / "ggml-rocm.dll").exists():
                backends.append("rocm")
            binaries.append({"path": str(path), "backends": sorted(set(backends))})
    return sorted(binaries, key=lambda item: item["path"])


def _run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0 and not result.stdout.strip():
        return ""
    return result.stdout.strip()


def _parse_mib(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).replace("MiB", "").strip()
    try:
        return int(float(text))
    except ValueError:
        return 0


def _parse_gpu_index(line: str) -> int:
    text = line.replace("GPU", "").replace(":", "").strip()
    try:
        return int(text)
    except ValueError:
        return -1
