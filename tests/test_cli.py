from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "applens_llm.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_cli_validates_json_document() -> None:
    result = run_cli(
        "validate",
        "--schema",
        "deployment-plan",
        "examples/gaming-pc-deployment-plan.json",
    )

    assert result.returncode == 0
    assert "valid" in result.stdout


def test_cli_validates_jsonl_file() -> None:
    result = run_cli(
        "validate-jsonl",
        "--schema",
        "training-example",
        "data/examples.seed.jsonl",
    )

    assert result.returncode == 0
    assert "rows valid" in result.stdout


def test_cli_ingests_capture_reports(tmp_path: Path) -> None:
    (tmp_path / "AppLens_Results_cli-host.md").write_text(
        "# AppLens Scan Results\n- **Computer:** cli-host\n",
        encoding="utf-8",
    )
    output = tmp_path / "captures.jsonl"

    result = run_cli(
        "ingest-captures",
        "--source",
        str(tmp_path),
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "1 capture records" in result.stdout
    assert output.exists()


def test_cli_writes_vgm_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "vgm-snapshot.json"

    result = run_cli(
        "vgm-snapshot",
        "--label",
        "unit-test",
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "vgm snapshot" in result.stdout
    assert output.exists()


def test_cli_compares_vgm_snapshots(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(
        '{"vgm_check":{"amd_dedicated_memory_mb":512,"vgm_16gb_active":false},'
        '"runtime_readiness":{"has_vulkan_llamacpp":false}}',
        encoding="utf-8",
    )
    after.write_text(
        '{"vgm_check":{"amd_dedicated_memory_mb":16384,"vgm_16gb_active":true},'
        '"runtime_readiness":{"has_vulkan_llamacpp":true}}',
        encoding="utf-8",
    )

    result = run_cli("vgm-compare", "--before", str(before), "--after", str(after))

    assert result.returncode == 0
    assert "VGM activated" in result.stdout
    assert "run_vulkan_benchmark" in result.stdout


def test_cli_prints_llamacpp_devices(tmp_path: Path) -> None:
    fake = tmp_path / "llama-bench.exe"
    fake.write_text("placeholder", encoding="utf-8")
    output = tmp_path / "devices.json"

    result = run_cli(
        "llamacpp-devices",
        "--binary",
        str(fake),
        "--output",
        str(output),
        "--from-output",
        "Available devices:\n  Vulkan0: AMD Radeon(TM) 890M Graphics (24380 MiB, 23161 MiB free)\n",
    )

    assert result.returncode == 0
    assert "1 llama.cpp devices" in result.stdout
    assert output.exists()


def test_cli_summarizes_adrenalin_hardware_log(tmp_path: Path) -> None:
    source = tmp_path / "Hardware.20260508-172749.CSV"
    output = tmp_path / "summary.json"
    source.write_text(
        "\n".join(
            [
                "TIME STAMP,GPU UTIL,GPU SCLK,GPU PWR,GPU TEMP,GPU MEM UTIL,GPU MCLK,CPU UTIL,SYSTEM MEM UTIL",
                "2026-05-08 17:25:46.741,19.000,1444.000,21,55.000,13049.000,937.000,17.95,8.48",
            ]
        ),
        encoding="utf-8",
    )

    result = run_cli("adrenalin-summary", "--input", str(source), "--output", str(output))

    assert result.returncode == 0
    assert "AMD Adrenalin telemetry" in result.stdout
    assert output.exists()


def test_cli_checks_runtime_lanes(tmp_path: Path) -> None:
    config = tmp_path / "lanes.json"
    config.write_text(json.dumps(_runtime_lanes_payload()), encoding="utf-8")

    result = run_cli("lanes-check", "--config", str(config))

    assert result.returncode == 0
    assert "1 runtime lanes valid" in result.stdout


def test_cli_rejects_invalid_runtime_lanes(tmp_path: Path) -> None:
    config = tmp_path / "lanes.json"
    config.write_text('{"schema_version":"0.1","lanes":[]}', encoding="utf-8")

    result = run_cli("lanes-check", "--config", str(config))

    assert result.returncode == 2
    assert "schema error" in result.stdout


def test_cli_initializes_blackboard(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"

    result = run_cli(
        "blackboard-init",
        "--experiment-id",
        "exp-test",
        "--title",
        "Unit",
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "blackboard initialized" in result.stdout
    assert output.exists()


def test_cli_appends_blackboard_task(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"

    result = run_cli(
        "blackboard-task",
        "--experiment-id",
        "exp-test",
        "--task-id",
        "task-1",
        "--prompt",
        "Compare lanes.",
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "task appended" in result.stdout
    event = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert event["event_type"] == "task"
    assert event["payload"]["prompt"] == "Compare lanes."
    assert event["privacy"]["commit_safe"] is False


def test_cli_orchestrate_once_records_lane_failure(tmp_path: Path) -> None:
    config = tmp_path / "lanes.json"
    blackboard = tmp_path / "run.jsonl"
    config.write_text(json.dumps(_runtime_lanes_payload()), encoding="utf-8")

    result = run_cli(
        "orchestrate-once",
        "--config",
        str(config),
        "--lane",
        "fast-nvidia",
        "--experiment-id",
        "exp-test",
        "--task-id",
        "task-1",
        "--prompt",
        "hello",
        "--blackboard",
        str(blackboard),
        "--timeout-seconds",
        "1",
    )

    assert result.returncode == 0
    assert "orchestrated fast-nvidia" in result.stdout
    event = json.loads(blackboard.read_text(encoding="utf-8").splitlines()[0])
    assert event["event_type"] == "failure"
    assert event["payload"]["outcome"] == "connection_error"


def test_cli_lane_start_dry_run_prints_command(tmp_path: Path) -> None:
    config = tmp_path / "lanes.json"
    state = tmp_path / "lane-processes.json"
    config.write_text(json.dumps(_runtime_lanes_payload()), encoding="utf-8")

    result = run_cli(
        "lane-start",
        "--config",
        str(config),
        "--lane",
        "fast-nvidia",
        "--state",
        str(state),
        "--dry-run",
    )

    assert result.returncode == 0
    assert "lane start dry-run fast-nvidia" in result.stdout
    assert "llama-server" in result.stdout
    assert not state.exists()


def test_cli_lane_stop_missing_state_is_clean(tmp_path: Path) -> None:
    state = tmp_path / "lane-processes.json"

    result = run_cli("lane-stop", "--lane", "fast-nvidia", "--state", str(state))

    assert result.returncode == 0
    assert "not_found" in result.stdout


def test_cli_experiment_run_skip_start_writes_summary(tmp_path: Path) -> None:
    config = tmp_path / "lanes.json"
    blackboard = tmp_path / "experiment.jsonl"
    summary = tmp_path / "summary.json"
    payload = _runtime_lanes_payload()
    payload["lanes"].append(
        {
            "lane_id": "deep-amd-vgm",
            "role": "deep",
            "engine": "llama.cpp",
            "backend": "vulkan",
            "endpoint": "http://127.0.0.1:9/v1",
            "model": {"label": "qwen-27b", "path": "models/qwen.gguf"},
            "device": {"selector": "Vulkan0", "accelerator_ids": ["amd-igpu-0"]},
            "launch": {
                "server_binary": "llama-server",
                "context_tokens": 2048,
                "gpu_layers": 99,
                "threads": 12,
                "environment": {},
            },
        }
    )
    config.write_text(json.dumps(payload), encoding="utf-8")

    result = run_cli(
        "experiment-run",
        "--config",
        str(config),
        "--fast-lane",
        "fast-nvidia",
        "--deep-lane",
        "deep-amd-vgm",
        "--experiment-id",
        "exp-test",
        "--prompt",
        "hello",
        "--blackboard",
        str(blackboard),
        "--summary",
        str(summary),
        "--skip-start",
        "--timeout-seconds",
        "1",
        "--nvidia-driver-branch",
        "game_ready",
    )

    assert result.returncode == 0
    assert "experiment run exp-test" in result.stdout
    assert summary.exists()
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload["responses"]["fast"]["outcome"] == "connection_error"
    assert summary_payload["responses"]["deep"]["outcome"] == "connection_error"
    assert summary_payload["driver_evidence"][0]["driver_branch"] == "game_ready"


def test_cli_experiment_compare_writes_comparison(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(_experiment_summary("game_ready", fast_ms=1000)), encoding="utf-8")
    candidate.write_text(json.dumps(_experiment_summary("studio", fast_ms=1100)), encoding="utf-8")

    result = run_cli(
        "experiment-compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "experiment compare" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["deltas"]["fast"]["latency_ms_delta"] == 100
    assert payload["baseline"]["driver"]["branch"] == "game_ready"
    assert payload["candidate"]["driver"]["branch"] == "studio"


def test_cli_fit_report_writes_report(tmp_path: Path) -> None:
    machine_profiles = tmp_path / "machines.jsonl"
    summary = tmp_path / "summary.json"
    output = tmp_path / "fit-report.json"
    machine_profiles.write_text(json.dumps(_fit_machine_profile()) + "\n", encoding="utf-8")
    summary.write_text(json.dumps(_experiment_summary("studio", fast_ms=1100)), encoding="utf-8")

    result = run_cli(
        "fit-report",
        "--machine-profile",
        str(machine_profiles),
        "--machine-id",
        "asus-laptop",
        "--experiment-summary",
        str(summary),
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "fit report" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["machine"]["machine_id"] == "asus-laptop"
    assert payload["runtime_recommendation"]["strategy"] == "two_lane_local"


def _runtime_lanes_payload() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "lanes": [
            {
                "lane_id": "fast-nvidia",
                "role": "fast",
                "engine": "llama.cpp",
                "backend": "cuda",
                "endpoint": "http://127.0.0.1:9/v1",
                "model": {"label": "jan-v35-4b", "path": "models/jan.gguf"},
                "device": {"selector": "cuda:0", "accelerator_ids": ["nvidia-dgpu-0"]},
                "launch": {
                    "server_binary": "llama-server",
                    "context_tokens": 4096,
                    "gpu_layers": 99,
                    "threads": 12,
                    "environment": {},
                },
            }
        ],
    }


def _experiment_summary(branch: str, *, fast_ms: int) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "experiment_id": f"exp-{branch}",
        "driver_evidence": [
            {
                "vendor": "nvidia",
                "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                "driver_version": "596.36",
                "driver_branch": branch,
                "benchmark_invalidates_on_change": True,
            }
        ],
        "lanes": {"fast": "fast-nvidia", "deep": "deep-amd-vgm"},
        "responses": {
            "fast": {
                "lane_id": "fast-nvidia",
                "outcome": "success",
                "latency_ms": fast_ms,
                "usage": {"total_tokens": 100},
            },
            "deep": {
                "lane_id": "deep-amd-vgm",
                "outcome": "success",
                "latency_ms": 2000,
                "usage": {"total_tokens": 200},
            },
        },
        "lifecycle": {
            "started": [
                {
                    "lane_id": "fast-nvidia",
                    "engine": "llama.cpp",
                    "backend": "cuda",
                    "device_selector": "CUDA0",
                    "accelerator_ids": ["nvidia-dgpu-0"],
                    "model_label": "jan-v35-4b-q4",
                },
                {
                    "lane_id": "deep-amd-vgm",
                    "engine": "llama.cpp",
                    "backend": "vulkan",
                    "device_selector": "Vulkan0",
                    "accelerator_ids": ["amd-igpu-0"],
                    "model_label": "qwen-27b-iq3",
                },
            ]
        },
    }


def _fit_machine_profile() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "machine_id": "asus-laptop",
        "label": "ASUS ProArt PX13",
        "capture_status": "captured_sanitized",
        "capture_priority": 2,
        "platform": {
            "vendor": "asus",
            "model": "ProArt PX13",
            "sku": "asus-proart-px13-sanitized",
            "os_family": "windows",
            "cpu": "AMD Ryzen AI 9 HX 370",
            "ram_gb": 32,
            "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU + AMD Radeon 890M",
            "vram_mb": 6144,
        },
        "hardware_topology": {
            "accelerators": [
                {
                    "accelerator_id": "nvidia-dgpu-0",
                    "kind": "nvidia_dgpu",
                    "vendor": "nvidia",
                    "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                    "present": True,
                    "api_support": ["cuda"],
                    "memory": {
                        "physical_dedicated_vram_mb": 6144,
                        "vgm_reserved_mb": 0,
                        "shared_graphics_memory_mb": 0,
                        "reported_total_graphics_memory_mb": 6144,
                        "estimated_usable_inference_memory_mb": 6144,
                        "confidence": "observed",
                    },
                    "verification": [{"source_type": "inventory", "notes": "Sanitized inventory."}],
                }
            ],
            "usable_inference_capacity": {
                "estimated_usable_memory_mb": 6144,
                "confidence": "observed",
                "preferred_accelerator_ids": ["nvidia-dgpu-0"],
                "mixed_device_pooling": "unverified",
                "verification": [{"source_type": "inventory", "notes": "Sanitized inventory."}],
            },
            "memory_claims": [],
        },
        "target_roles": ["training_candidate"],
        "collection": {
            "applens_report": "captured",
            "applens_tune_report": "captured",
            "local_ai_profile": "captured",
            "llm_bench": "pending",
            "sanitized": True,
        },
        "notes": "Sanitized unit-test profile.",
    }
