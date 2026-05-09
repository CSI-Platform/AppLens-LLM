from __future__ import annotations

import json
from pathlib import Path

from applens_llm import vgm_probe
from applens_llm.vgm_probe import (
    build_vgm_snapshot,
    compare_vgm_snapshots,
    find_gguf_models,
    write_vgm_snapshot,
)


def test_find_gguf_models_classifies_control_and_capacity_candidates(tmp_path: Path) -> None:
    small = tmp_path / "small.gguf"
    large = tmp_path / "large.gguf"
    huge = tmp_path / "huge.gguf"
    small.write_bytes(b"0" * (3 * 1024))
    large.write_bytes(b"0" * (12 * 1024))
    huge.write_bytes(b"0" * (22 * 1024))

    models = find_gguf_models(tmp_path, max_items=10, bytes_per_gb=1024)
    by_name = {Path(model["path"]).name: model for model in models}

    assert by_name["small.gguf"]["vgm_test_role"] == "small_control"
    assert by_name["large.gguf"]["vgm_test_role"] == "vgm_capacity_candidate"
    assert by_name["huge.gguf"]["vgm_test_role"] == "too_large_for_16gb_vgm"


def test_build_vgm_snapshot_marks_current_amd_vgm_state() -> None:
    snapshot = build_vgm_snapshot(
        label="before-vgm",
        model_roots=[],
        video_controllers=[
            {
                "Name": "AMD Radeon(TM) 890M Graphics",
                "AdapterRAM": 536870912,
                "DriverVersion": "32.0.13058.2",
                "VideoProcessor": "AMD Radeon Graphics Processor",
            },
            {
                "Name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                "AdapterRAM": 4293918720,
                "DriverVersion": "32.0.15.9636",
                "VideoProcessor": "NVIDIA GeForce RTX 4050 Laptop GPU",
            },
        ],
        nvidia_smi_rows=[
            {
                "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                "memory.total [MiB]": "6141 MiB",
                "memory.used [MiB]": "8 MiB",
                "driver_version": "596.36",
            }
        ],
        vulkan_devices=[
            {
                "index": 0,
                "device_name": "AMD Radeon(TM) 890M Graphics",
                "device_type": "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
                "driver_name": "AMD proprietary driver",
            },
            {
                "index": 1,
                "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                "device_type": "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                "driver_name": "NVIDIA",
            },
        ],
        llama_binaries=[
            {
                "path": "C:/llama/build/bin/llama-server.exe",
                "backends": ["cuda"],
            }
        ],
    )

    assert snapshot["vgm_check"]["amd_dedicated_memory_mb"] == 512
    assert snapshot["vgm_check"]["vgm_16gb_active"] is False
    assert snapshot["runtime_readiness"]["vulkan_devices_seen"] == 2
    assert snapshot["runtime_readiness"]["has_vulkan_llamacpp"] is False


def test_build_vgm_snapshot_uses_registry_memory_when_wmi_caps_at_4gb() -> None:
    snapshot = build_vgm_snapshot(
        label="after-vgm",
        model_roots=[],
        video_controllers=[
            {
                "Name": "AMD Radeon(TM) 890M Graphics",
                "AdapterRAM": 4293918720,
                "DriverVersion": "32.0.13058.2",
                "VideoProcessor": "AMD Radeon Graphics Processor",
            }
        ],
        registry_video_memory=[
            {
                "name": "AMD Radeon(TM) 890M Graphics",
                "hardware_memory_bytes": 17179869184,
            }
        ],
        nvidia_smi_rows=[],
        vulkan_devices=[],
        llama_binaries=[],
    )

    assert snapshot["vgm_check"]["amd_dedicated_memory_mb"] == 16384
    assert snapshot["vgm_check"]["vgm_16gb_active"] is True


def test_compare_vgm_snapshots_reports_activation() -> None:
    before = {
        "vgm_check": {"amd_dedicated_memory_mb": 512, "vgm_16gb_active": False},
        "runtime_readiness": {"has_vulkan_llamacpp": False},
    }
    after = {
        "vgm_check": {"amd_dedicated_memory_mb": 16384, "vgm_16gb_active": True},
        "runtime_readiness": {"has_vulkan_llamacpp": True},
    }

    comparison = compare_vgm_snapshots(before, after)

    assert comparison["vgm_activated"] is True
    assert comparison["amd_dedicated_memory_delta_mb"] == 15872
    assert comparison["next_step"] == "run_vulkan_benchmark"


def test_write_vgm_snapshot_creates_json(tmp_path: Path) -> None:
    output = tmp_path / "snapshot.json"

    snapshot = write_vgm_snapshot(
        output,
        label="unit-test",
        model_roots=[],
        video_controllers=[],
        nvidia_smi_rows=[],
        vulkan_devices=[],
        llama_binaries=[],
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["snapshot_id"] == snapshot["snapshot_id"]


def test_run_command_keeps_stdout_when_powershell_returns_nonzero(monkeypatch) -> None:
    class Result:
        returncode = 1
        stdout = '[{"DriverDesc":"AMD Radeon(TM) 890M Graphics"}]'
        stderr = ""

    monkeypatch.setattr(vgm_probe.subprocess, "run", lambda *args, **kwargs: Result())

    assert vgm_probe._run_command(["powershell"]) == Result.stdout
