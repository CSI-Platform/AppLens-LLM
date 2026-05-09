from __future__ import annotations

import subprocess
from typing import Any


def collect_nvidia_driver_evidence(
    *,
    driver_branch: str = "unknown",
    nvidia_smi_output: str | None = None,
) -> list[dict[str, Any]]:
    output = nvidia_smi_output if nvidia_smi_output is not None else _run_nvidia_smi_driver_query()
    if not output.strip():
        if driver_branch != "unknown":
            return [_unknown_nvidia_driver_evidence(driver_branch)]
        return []
    return parse_nvidia_smi_driver_csv(output, driver_branch=driver_branch)


def parse_nvidia_smi_driver_csv(output: str, *, driver_branch: str = "unknown") -> list[dict[str, Any]]:
    evidence = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        evidence.append(
            {
                "vendor": "nvidia",
                "device_name": parts[0],
                "driver_version": parts[1],
                "driver_branch": driver_branch,
                "branch_confidence": "user_confirmed" if driver_branch != "unknown" else "unknown",
                "version_source": "nvidia-smi",
                "branch_source": "nvidia_app" if driver_branch != "unknown" else "unknown",
                "benchmark_invalidates_on_change": True,
            }
        )
    return evidence


def _run_nvidia_smi_driver_query() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _unknown_nvidia_driver_evidence(driver_branch: str) -> dict[str, Any]:
    return {
        "vendor": "nvidia",
        "device_name": "unknown NVIDIA GPU",
        "driver_version": "unknown",
        "driver_branch": driver_branch,
        "branch_confidence": "user_confirmed",
        "version_source": "nvidia-smi-unavailable",
        "branch_source": "nvidia_app",
        "benchmark_invalidates_on_change": True,
    }
