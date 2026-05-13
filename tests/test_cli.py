from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from applens_llm.benchmark_suite import build_benchmark_suite_run
from applens_llm.schemas import validate_payload


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


def test_cli_writes_benchmark_suite_plan(tmp_path: Path) -> None:
    output = tmp_path / "benchmark-suite.json"

    result = run_cli(
        "benchmark-suite-plan",
        "--suite-run-id",
        "qwen35-4b-vgm16-plan",
        "--model-id",
        "qwen35-4b-q4km",
        "--display-name",
        "Qwen3.5 4B Q4_K_M",
        "--family",
        "qwen",
        "--parameter-size-b",
        "4",
        "--quantization",
        "Q4_K_M",
        "--model-format",
        "gguf",
        "--model-path",
        "sanitized/models/model.gguf",
        "--chat-template",
        "qwen",
        "--thinking-mode",
        "off",
        "--reasoning-mode",
        "off",
        "--condition-id",
        "asus-px13-vgm16-ram16",
        "--condition-label",
        "ASUS PX13 VGM 16GB",
        "--os-family",
        "windows",
        "--ram-gb",
        "32",
        "--vgm-enabled",
        "--vgm-dedicated-mb",
        "16384",
        "--system-ram-available-gb",
        "16",
        "--accelerator-id",
        "amd-igpu-0",
        "--engine",
        "llama.cpp",
        "--backend",
        "vulkan",
        "--device-selector",
        "Vulkan0",
        "--endpoint",
        "http://127.0.0.1:18080/v1",
        "--context-tokens",
        "16384",
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "benchmark suite qwen35-4b-vgm16-plan" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_payload("benchmark-suite-run", payload)
    assert payload["suite"]["suite_id"] == "tiny-v1"
    assert payload["machine_condition"]["vgm_state"]["dedicated_mb"] == 16384


def test_cli_writes_benchmark_readiness_report(tmp_path: Path) -> None:
    plan = tmp_path / "benchmark-suite.json"
    output = tmp_path / "readiness.json"
    plan.write_text(json.dumps(_benchmark_suite_payload()), encoding="utf-8")

    result = run_cli(
        "benchmark-readiness",
        "--plan",
        str(plan),
        "--output",
        str(output),
        "--lm-eval",
        sys.executable,
        "--skip-endpoint",
    )

    assert result.returncode == 0
    assert "benchmark readiness" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_payload("benchmark-readiness", payload)
    assert payload["suite_run_id"] == "qwen35-4b-vgm16-plan"


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


def test_cli_overnight_loop_skip_start_writes_failure_summary(tmp_path: Path) -> None:
    config = tmp_path / "lanes.json"
    blackboard = tmp_path / "overnight.jsonl"
    summary = tmp_path / "overnight-summary.json"
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
        "overnight-loop",
        "--config",
        str(config),
        "--fast-lane",
        "fast-nvidia",
        "--deep-lane",
        "deep-amd-vgm",
        "--experiment-id",
        "overnight-cli",
        "--prompt",
        "test handoff",
        "--blackboard",
        str(blackboard),
        "--summary",
        str(summary),
        "--skip-start",
        "--max-iterations",
        "1",
        "--timeout-seconds",
        "1",
        "--sleep-seconds",
        "0",
    )

    assert result.returncode == 0
    assert "overnight loop overnight-cli" in result.stdout
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["stop_reason"] == "fast_failure"
    assert payload["completed_iterations"] == 0


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


def test_cli_model_fit_scorecard_writes_scorecard(tmp_path: Path) -> None:
    machine_profiles = tmp_path / "machines.jsonl"
    candidates = tmp_path / "models.json"
    summary = tmp_path / "summary.json"
    workload = tmp_path / "workload.json"
    output = tmp_path / "scorecard.json"
    machine_profiles.write_text(json.dumps(_fit_machine_profile()) + "\n", encoding="utf-8")
    candidates.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "jan-v35-4b-q4",
                        "display_name": "Jan v3.5 4B Q4",
                        "family": "qwen",
                        "parameter_size_b": 4,
                        "quantization": "Q4_K_XL",
                        "file_size_mb": 2860,
                        "local_status": "local",
                        "preferred_roles": ["fast_chat"],
                        "quality_prior": "medium",
                        "observed_model_label": "jan-v35-4b-q4",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(json.dumps(_experiment_summary("studio", fast_ms=1100)), encoding="utf-8")
    workload.write_text(json.dumps(_workload_profile()), encoding="utf-8")

    result = run_cli(
        "model-fit-scorecard",
        "--machine-profile",
        str(machine_profiles),
        "--machine-id",
        "asus-laptop",
        "--model-candidates",
        str(candidates),
        "--experiment-summary",
        str(summary),
        "--workload-profile",
        str(workload),
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "model fit scorecard" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rankings"][0]["model_id"] == "jan-v35-4b-q4"
    assert payload["rankings"][0]["fit_score"] > 0
    assert payload["workload"]["workload_id"] == "oracle"


def test_cli_deployment_plan_writes_outfitting_plan(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    output = tmp_path / "deployment-plan.json"
    scorecard.write_text(json.dumps(_scorecard_payload()), encoding="utf-8")

    result = run_cli(
        "deployment-plan",
        "--scorecard",
        str(scorecard),
        "--plan-id",
        "asus-px13-outfit",
        "--workload-name",
        "Oracle autoresearch",
        "--workload-intent",
        "agent_runtime",
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "deployment plan" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_payload("deployment-plan", payload)
    assert payload["outfitting"]["supervisor_baseline"]["runtime"] == "cloud_api"
    assert payload["outfitting"]["assignments"][0]["role"] == "primary_local_worker"


def test_cli_local_capability_eval_scores_response_file(tmp_path: Path) -> None:
    responses = tmp_path / "responses.json"
    output = tmp_path / "capability.json"
    responses.write_text(
        json.dumps(
            {
                "model": {"model_id": "qwen35-4b-q4km", "display_name": "Qwen3.5 4B", "quantization": "Q4_K_M"},
                "runtime": {"engine": "llama.cpp", "backend": "vulkan", "devices_used": ["nvidia-dgpu-0"]},
                "responses": {
                    "strict_json_summary": '{"recommendation":"benchmark_required","confidence":"observed","evidence_required":["benchmark-record"]}',
                    "tool_select_model_lane": '{"tool_call":{"name":"select_model_lane","arguments":{"model_id":"qwen35-4b-q4km","lane_id":"fast-nvidia","reason":"observed fast lane"}}}',
                    "tool_noop_for_memory_claim": '{"tool_call":null,"answer":"No tool yet; benchmark the 22GB pooled VRAM claim first."}',
                    "hardware_memory_reasoning": '{"verdict":"claim_requires_benchmark","unsupported_claims":["22GB pooled VRAM"],"required_evidence":["devices_used"],"notes":"not one clean pooled device"}',
                    "benchmark_interpretation": '{"best_fast_model":"qwen35-4b-q4km","best_deep_model":"gemma4-26b-a4b-q3km","next_test":"capability eval","reason":"role evidence"}',
                    "coding_unit_task": '{"language":"python","files":[{"path":"score_fit.py","content":"def score_fit(rows):\\n    return round(sum(row[\\\"score\\\"] for row in rows) / len(rows)) if rows else 0\\n"}]}',
                    "safety_boundary": '{"allowed_actions":["write_scorecard"],"blocked_actions":["driver_change","firewall_change","delete_user_files"],"requires_user_approval":["downloads"]}',
                    "handoff_planning": '{"planner_model":"gemma4-26b-a4b-q3km","executor_model":"qwen35-2b-q4km","handoff_packet":{"objective":"summarize","steps":["read","write"],"success_check":"schema valid"}}',
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "local-capability-eval",
        "--responses",
        str(responses),
        "--output",
        str(output),
        "--thinking-mode",
        "off",
        "--execute-code-checks",
    )

    assert result.returncode == 0
    assert "local capability eval" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"]["id"] == "applens-local-v1"
    assert payload["model"]["thinking_mode"] == "off"


def test_cli_context_envelope_writes_context_taper_report(tmp_path: Path) -> None:
    machine = tmp_path / "machine.json"
    candidates = tmp_path / "models.json"
    observations = tmp_path / "context-observations.jsonl"
    output = tmp_path / "context-envelope.json"
    machine.write_text(json.dumps(_fit_machine_profile()), encoding="utf-8")
    candidates.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "qwen35-27b-iq3",
                        "display_name": "Qwen3.5 27B IQ3",
                        "family": "qwen",
                        "parameter_size_b": 27,
                        "quantization": "IQ3_M",
                        "local_status": "local",
                        "preferred_roles": ["deep_review"],
                        "quality_prior": "high",
                        "advertised_context_tokens": 262144,
                        "advertised_context_source": "https://artificialanalysis.ai/leaderboards/models?size=small",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    observations.write_text(
        json.dumps(
            {
                "model_id": "qwen35-27b-iq3",
                "context_tokens": 65536,
                "backend": "vulkan",
                "devices_used": ["amd-igpu-0"],
                "status": "pass",
                "quality_score_pct": 78,
                "generation_tokens_per_second": 5.2,
                "prompt_tokens_per_second": 51,
                "failure_modes": ["none"],
                "workloads": ["long_context_retrieval"],
                "notes": "Sanitized CLI observation.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_cli(
        "context-envelope",
        "--machine-profile",
        str(machine),
        "--machine-id",
        "asus-laptop",
        "--model-candidates",
        str(candidates),
        "--context-observation",
        str(observations),
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "context envelope" in result.stdout
    assert "context_evidence_models=1" in result.stdout
    assert "useful_context_models=1" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["models"][0]["max_recommended_context_tokens"] == 65536


def test_cli_context_quality_probe_scores_response_file(tmp_path: Path) -> None:
    response = tmp_path / "response.txt"
    output = tmp_path / "context-quality.json"
    observation = tmp_path / "context-quality.jsonl"
    response.write_text(
        json.dumps(
            {
                "needle": "APPLENS-CTX-8192-TEST",
                "context_tier": 8192,
                "memory_claim_verdict": "benchmark_required_not_pooled",
                "code": (
                    "def choose_context_tier(rows):\n"
                    "    usable = [row for row in rows if row.get('status') == 'pass' "
                    "and row.get('quality_score_pct', 0) >= 60 "
                    "and row.get('generation_tokens_per_second', 0) >= 1]\n"
                    "    return max((row.get('context_tokens', 0) for row in usable), default=0)\n"
                ),
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "context-quality-probe",
        "--response-file",
        str(response),
        "--model-id",
        "qwen35-2b-q4km",
        "--display-name",
        "Qwen3.5 2B Q4_K_M",
        "--family",
        "qwen",
        "--parameter-size-b",
        "2",
        "--quantization",
        "Q4_K_M",
        "--backend",
        "vulkan",
        "--device-selector",
        "Vulkan1",
        "--accelerator-id",
        "nvidia-dgpu-0",
        "--context-tokens",
        "8192",
        "--prompt-token-budget",
        "6000",
        "--expected-needle",
        "APPLENS-CTX-8192-TEST",
        "--elapsed-seconds",
        "8",
        "--prompt-tokens-per-second",
        "100",
        "--generation-tokens-per-second",
        "42",
        "--execute-code-checks",
        "--output",
        str(output),
        "--context-observation-output",
        str(observation),
    )

    assert result.returncode == 0
    assert "context quality" in result.stdout
    assert "score=" in result.stdout
    assert output.exists()
    assert observation.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["outcome"]["status"] == "pass"
    observation_payload = json.loads(observation.read_text(encoding="utf-8"))
    assert observation_payload["status"] == "pass"
    assert observation_payload["generation_tokens_per_second"] == 42


def test_cli_model_fit_html_writes_sortable_report(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    comparison = tmp_path / "comparison.json"
    output = tmp_path / "scorecard.html"
    scorecard.write_text(json.dumps(_scorecard_payload()), encoding="utf-8")
    comparison.write_text(json.dumps(_comparison_payload()), encoding="utf-8")

    result = run_cli(
        "model-fit-html",
        "--scorecard",
        str(scorecard),
        "--experiment-comparison",
        str(comparison),
        "--title",
        "ASUS PX13 Fit",
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "model fit html" in result.stdout
    html = output.read_text(encoding="utf-8")
    assert "ASUS PX13 Fit" in html
    assert "data-sort-table" in html
    assert "Jan v3.5 4B Q4" in html


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


def _benchmark_suite_payload() -> dict[str, object]:
    return build_benchmark_suite_run(
        suite_run_id="qwen35-4b-vgm16-plan",
        model={
            "model_id": "qwen35-4b-q4km",
            "display_name": "Qwen3.5 4B Q4_K_M",
            "family": "qwen",
            "parameter_size_b": 4,
            "quantization": "Q4_K_M",
            "model_format": "gguf",
            "path": "sanitized/models/model.gguf",
            "sha256": "unknown",
            "chat_template": "qwen",
            "thinking_mode": "off",
            "reasoning_mode": "off",
        },
        machine_condition={
            "condition_id": "asus-px13-vgm16-ram16",
            "label": "ASUS PX13 VGM 16GB",
            "os_family": "windows",
            "ram_gb": 32,
            "vgm_state": {
                "enabled": True,
                "dedicated_mb": 16384,
                "system_ram_available_gb": 16,
                "source": "AMD Software: Adrenalin Edition",
            },
            "accelerator_ids": ["amd-igpu-0"],
            "required_preflight": [],
            "evidence_paths": [],
        },
        runtime_lane={
            "engine": "llama.cpp",
            "backend": "vulkan",
            "device_selector": "Vulkan0",
            "accelerator_ids": ["amd-igpu-0"],
            "endpoint": "http://127.0.0.1:18080/v1",
            "context_tokens": 16384,
            "batch_size": 2048,
            "ubatch_size": 512,
            "threads": 12,
            "gpu_layers": 99,
            "kv_cache_type": "auto",
            "flash_attention": "auto",
            "extra_flags": [],
        },
        created_at="2026-05-13T20:00:00Z",
    )


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


def _workload_profile() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "workload_id": "oracle",
        "display_name": "Oracle",
        "workload_type": "financial_research",
        "program_file": ".applens/program.md",
        "commands_file": ".applens/commands.json",
        "metrics_file": ".applens/metrics.json",
        "probes_file": ".applens/probes.json",
        "evals_dir": ".applens/evals",
        "schemas_dir": ".applens/schemas",
        "allowed_actions": ["read_local_data", "write_artifacts"],
        "blocked_actions": ["credential_access", "system_change"],
        "required_artifacts": ["run_summary"],
        "model_role_needs": [{"role": "supervisor", "capabilities": ["reasoning"]}],
        "safety_gates": {
            "requires_self_fit": True,
            "allow_network": False,
            "allow_writes_outside_workload_root": False,
            "allow_live_trading": False,
        },
    }


def _scorecard_payload() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "scorecard_id": "scorecard-test",
        "created_at": "2026-05-10T01:00:00Z",
        "machine": {
            "machine_id": "asus-laptop",
            "label": "ASUS ProArt PX13",
            "platform": {
                "cpu": "AMD Ryzen AI 9 HX 370",
                "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU + AMD Radeon 890M",
                "ram_gb": 32,
                "os_family": "windows",
            },
        },
        "scoring_weights": {
            "capacity_fit": 18,
            "speed_latency": 13,
            "stability": 10,
            "role_fit": 10,
            "quality_size": 5,
            "operational_readiness": 10,
            "evidence_confidence": 5,
            "agent_capability": 24,
            "context_evidence": 5,
        },
        "rankings": [
            {
                "rank": 1,
                "model_id": "jan-v35-4b-q4",
                "display_name": "Jan v3.5 4B Q4",
                "family": "qwen",
                "parameter_size_b": 4.0,
                "quantization": "Q4_K_XL",
                "local_status": "local",
                "recommended_role": "fast_chat",
                "best_lane": {
                    "lane_id": "fast-nvidia",
                    "backend": "cuda",
                    "accelerator_ids": ["nvidia-dgpu-0"],
                    "role": "fast_chat",
                    "model_label": "jan-v35-4b-q4",
                },
                "fit_score": 98,
                "score_band": "excellent",
                "score_breakdown": {
                    "capacity_fit": 18,
                    "speed_latency": 13,
                    "stability": 10,
                    "role_fit": 10,
                    "quality_size": 4,
                    "operational_readiness": 10,
                    "evidence_confidence": 5,
                    "agent_capability": 24,
                    "context_evidence": 0,
                },
                "confidence": "observed",
                "reasons": ["Observed fast CUDA lane."],
                "blockers": [],
                "evidence": {
                    "source": "experiment_summary",
                    "observation_count": 5,
                    "benchmark_suite_result_count": 0,
                    "benchmark_suite_passed": 0,
                    "benchmark_suite_failed": 0,
                    "benchmark_suite_unsupported": 0,
                    "benchmark_suite_unsupported_tasks": [],
                    "capability_record_count": 1,
                    "advertised_context_tokens": 0,
                    "max_tested_context_tokens": 0,
                    "recommended_context_tokens": 0,
                    "context_score_pct": 0,
                    "context_evidence_status": "advertised_unproven",
                    "context_interpretation": "Advertised context is unproven locally; this is not a performance finding.",
                    "capability_score_pct": 96,
                    "capability_categories": {"tool_calling": 100},
                    "thinking_modes": ["off"],
                    "avg_latency_ms": 2166.0,
                    "avg_total_tokens": 133.0,
                },
                "next_benchmark": "Run task-specific quality checks.",
            }
        ],
        "evidence": {
            "experiment_summary_count": 5,
            "benchmark_record_count": 0,
            "benchmark_suite_result_count": 0,
            "capability_record_count": 1,
            "candidate_model_count": 1,
        },
        "benchmark_suites": [],
        "next_actions": ["Benchmark unobserved candidates."],
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }


def _comparison_payload() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "baseline": {
            "experiment_id": "exp-game-ready",
            "driver": {
                "vendor": "nvidia",
                "device_name": "RTX 4050",
                "version": "596.36",
                "branch": "game_ready",
            },
            "lanes": {"fast": "fast-nvidia", "deep": "deep-amd-vgm"},
        },
        "candidate": {
            "experiment_id": "exp-studio",
            "driver": {
                "vendor": "nvidia",
                "device_name": "RTX 4050",
                "version": "596.36",
                "branch": "studio",
            },
            "lanes": {"fast": "fast-nvidia", "deep": "deep-amd-vgm"},
        },
        "comparability": {
            "same_driver_version": True,
            "same_lanes": True,
            "warnings": ["driver_branches_differ"],
        },
        "deltas": {
            "fast": {"latency_ms_delta": 36, "latency_ms_delta_pct": 1.68},
            "deep": {"latency_ms_delta": -3165, "latency_ms_delta_pct": -13.21},
        },
        "verdict": "difference_observed_needs_repeats",
    }
