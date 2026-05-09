from __future__ import annotations

import subprocess

from applens_llm.driver_evidence import collect_nvidia_driver_evidence, parse_nvidia_smi_driver_csv


def test_parse_nvidia_smi_driver_csv() -> None:
    output = "NVIDIA GeForce RTX 4050 Laptop GPU, 576.88\n"

    evidence = parse_nvidia_smi_driver_csv(output, driver_branch="game_ready")

    assert evidence == [
        {
            "vendor": "nvidia",
            "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "driver_version": "576.88",
            "driver_branch": "game_ready",
            "branch_confidence": "user_confirmed",
            "version_source": "nvidia-smi",
            "branch_source": "nvidia_app",
            "benchmark_invalidates_on_change": True,
        }
    ]


def test_collect_nvidia_driver_evidence_accepts_injected_output() -> None:
    evidence = collect_nvidia_driver_evidence(
        driver_branch="studio",
        nvidia_smi_output="NVIDIA GeForce RTX 4050 Laptop GPU, 577.00\n",
    )

    assert evidence[0]["driver_branch"] == "studio"
    assert evidence[0]["driver_version"] == "577.00"


def test_collect_nvidia_driver_evidence_is_empty_when_branch_is_unknown_and_nvidia_smi_is_missing(monkeypatch) -> None:
    def missing_binary(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing_binary)

    assert collect_nvidia_driver_evidence() == []


def test_collect_nvidia_driver_evidence_records_user_branch_when_nvidia_smi_is_missing(monkeypatch) -> None:
    def missing_binary(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing_binary)

    evidence = collect_nvidia_driver_evidence(driver_branch="game_ready")

    assert evidence[0]["driver_branch"] == "game_ready"
    assert evidence[0]["driver_version"] == "unknown"
    assert evidence[0]["version_source"] == "nvidia-smi-unavailable"
