from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


HEADER_MAP = {
    "TIME STAMP": "timestamp",
    "GPU UTIL": "gpu_utilization_percent",
    "GPU SCLK": "gpu_clock_mhz",
    "GPU PWR": "gpu_power_watts",
    "GPU TEMP": "gpu_temperature_c",
    "GPU MEM UTIL": "gpu_memory_utilization_mb",
    "GPU MCLK": "gpu_memory_clock_mhz",
    "CPU UTIL": "cpu_utilization_percent",
    "SYSTEM MEM UTIL": "system_memory_utilization_gb",
}


def parse_hardware_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_row in reader:
            timestamp = (source_row.get("TIME STAMP") or "").strip()
            if not timestamp or timestamp.upper() == "N/A":
                continue
            row: dict[str, Any] = {"timestamp": timestamp}
            for source_key, target_key in HEADER_MAP.items():
                if source_key == "TIME STAMP":
                    continue
                row[target_key] = _parse_float(source_row.get(source_key))
            rows.append(row)
    return rows


def summarize_hardware_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "sample_count": len(rows),
        "start_timestamp": rows[0]["timestamp"] if rows else None,
        "end_timestamp": rows[-1]["timestamp"] if rows else None,
    }
    for metric in (
        "gpu_utilization_percent",
        "gpu_clock_mhz",
        "gpu_power_watts",
        "gpu_temperature_c",
        "gpu_memory_utilization_mb",
        "gpu_memory_clock_mhz",
        "cpu_utilization_percent",
        "system_memory_utilization_gb",
    ):
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not values:
            continue
        summary[metric] = {
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "avg": round(sum(values) / len(values), 3),
        }
    return summary


def write_hardware_summary(input_path: Path, output_path: Path) -> dict[str, Any]:
    rows = parse_hardware_csv(input_path)
    summary = {
        "schema_version": "0.1",
        "source": "amd_adrenalin",
        "input_path": str(input_path),
        "summary": summarize_hardware_rows(rows),
        "privacy": {
            "local_paths_included": True,
            "commit_safe": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None
