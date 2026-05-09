from __future__ import annotations

import json
from pathlib import Path

from applens_llm.experiment_compare import compare_experiment_summaries, write_experiment_comparison


def test_compare_experiment_summaries_reports_driver_and_lane_deltas() -> None:
    comparison = compare_experiment_summaries(_summary("game_ready"), _summary("studio"))

    assert comparison["baseline"]["driver"]["branch"] == "game_ready"
    assert comparison["candidate"]["driver"]["branch"] == "studio"
    assert comparison["comparability"]["same_driver_version"] is True
    assert comparison["comparability"]["same_lanes"] is True
    assert comparison["deltas"]["fast"]["latency_ms_delta"] == 36
    assert comparison["deltas"]["fast"]["total_tokens_delta"] == 0
    assert comparison["deltas"]["deep"]["latency_ms_delta"] == -3165
    assert comparison["deltas"]["deep"]["total_tokens_delta"] == -15
    assert "token_counts_differ" in comparison["comparability"]["warnings"]


def test_write_experiment_comparison_writes_json(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(_summary("game_ready")), encoding="utf-8")
    candidate.write_text(json.dumps(_summary("studio")), encoding="utf-8")

    comparison = write_experiment_comparison(baseline, candidate, output)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == comparison
    assert comparison["schema_version"] == "0.1"


def _summary(branch: str) -> dict:
    if branch == "game_ready":
        fast_ms = 2138
        deep_ms = 23966
        deep_tokens = 286
    else:
        fast_ms = 2174
        deep_ms = 20801
        deep_tokens = 271
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
                "usage": {"total_tokens": 133},
            },
            "deep": {
                "lane_id": "deep-amd-vgm",
                "outcome": "success",
                "latency_ms": deep_ms,
                "usage": {"total_tokens": deep_tokens},
            },
        },
    }
