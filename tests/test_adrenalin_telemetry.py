from __future__ import annotations

from pathlib import Path

from applens_llm.adrenalin_telemetry import parse_hardware_csv, summarize_hardware_rows


def test_parse_hardware_csv_skips_aggregate_row(tmp_path: Path) -> None:
    source = tmp_path / "Hardware.20260508-172749.CSV"
    source.write_text(
        "\n".join(
            [
                "TIME STAMP,GPU UTIL,GPU SCLK,GPU PWR,GPU TEMP,GPU MEM UTIL,GPU MCLK,CPU UTIL,SYSTEM MEM UTIL",
                "N/A,15.51,1438.46,14.40,52.51,3685.72,898.45,11.92,6.29",
                "2026-05-08 17:25:44.708,14.000,2340.000,37,58.000,6883.000,937.000,43.78,7.03",
                "2026-05-08 17:25:46.741,19.000,1444.000,21,55.000,13049.000,937.000,17.95,8.48",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_hardware_csv(source)

    assert len(rows) == 2
    assert rows[0]["timestamp"] == "2026-05-08 17:25:44.708"
    assert rows[1]["gpu_memory_utilization_mb"] == 13049.0


def test_summarize_hardware_rows_reports_peak_memory_and_time_range() -> None:
    rows = [
        {
            "timestamp": "2026-05-08 17:25:44.708",
            "gpu_utilization_percent": 14.0,
            "gpu_power_watts": 37.0,
            "gpu_temperature_c": 58.0,
            "gpu_memory_utilization_mb": 6883.0,
            "cpu_utilization_percent": 43.78,
            "system_memory_utilization_gb": 7.03,
        },
        {
            "timestamp": "2026-05-08 17:25:46.741",
            "gpu_utilization_percent": 19.0,
            "gpu_power_watts": 21.0,
            "gpu_temperature_c": 55.0,
            "gpu_memory_utilization_mb": 13049.0,
            "cpu_utilization_percent": 17.95,
            "system_memory_utilization_gb": 8.48,
        },
    ]

    summary = summarize_hardware_rows(rows)

    assert summary["sample_count"] == 2
    assert summary["start_timestamp"] == "2026-05-08 17:25:44.708"
    assert summary["end_timestamp"] == "2026-05-08 17:25:46.741"
    assert summary["gpu_memory_utilization_mb"]["max"] == 13049.0
    assert summary["gpu_temperature_c"]["max"] == 58.0
