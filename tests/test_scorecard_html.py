from __future__ import annotations

import json
from pathlib import Path

from applens_llm.scorecard_html import build_scorecard_html, write_scorecard_html


def test_build_scorecard_html_contains_sortable_rankings_and_comparisons() -> None:
    html = build_scorecard_html(
        scorecard=_scorecard(),
        experiment_comparisons=[_comparison()],
        title="ASUS PX13 Local AI Fit",
    )

    assert "ASUS PX13 Local AI Fit" in html
    assert "data-sort-table" in html
    assert "Jan v3.5 4B Q4" in html
    assert "Qwen3.5 27B IQ3" in html
    assert "driver_branches_differ" in html
    assert "inconclusive_token_counts_differ" in html
    assert "function sortTable" in html


def test_write_scorecard_html_loads_json_files(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    comparison = tmp_path / "comparison.json"
    output = tmp_path / "scorecard.html"
    scorecard.write_text(json.dumps(_scorecard()), encoding="utf-8")
    comparison.write_text(json.dumps(_comparison()), encoding="utf-8")

    write_scorecard_html(
        scorecard_path=scorecard,
        output_path=output,
        experiment_comparison_paths=[comparison],
        title="Local Fit",
    )

    html = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "<!doctype html>" in html
    assert "Local Fit" in html
    assert "Jan v3.5 4B Q4" in html


def _scorecard() -> dict:
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
            "capacity_fit": 25,
            "speed_latency": 20,
            "stability": 15,
            "role_fit": 15,
            "quality_size": 10,
            "operational_readiness": 10,
            "evidence_confidence": 5,
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
                    "capacity_fit": 25,
                    "speed_latency": 20,
                    "stability": 15,
                    "role_fit": 15,
                    "quality_size": 8,
                    "operational_readiness": 10,
                    "evidence_confidence": 5,
                },
                "confidence": "observed",
                "reasons": ["Observed fast CUDA lane."],
                "blockers": [],
                "evidence": {
                    "source": "experiment_summary",
                    "observation_count": 5,
                    "avg_latency_ms": 2166.0,
                    "avg_total_tokens": 133.0,
                },
                "next_benchmark": "Run task-specific quality checks.",
            },
            {
                "rank": 2,
                "model_id": "qwen-27b-iq3",
                "display_name": "Qwen3.5 27B IQ3",
                "family": "qwen",
                "parameter_size_b": 27.0,
                "quantization": "IQ3_M",
                "local_status": "local",
                "recommended_role": "deep_review",
                "best_lane": {
                    "lane_id": "deep-amd-vgm",
                    "backend": "vulkan",
                    "accelerator_ids": ["amd-igpu-0"],
                    "role": "deep_review",
                    "model_label": "qwen-27b-iq3",
                },
                "fit_score": 89,
                "score_band": "good",
                "score_breakdown": {
                    "capacity_fit": 25,
                    "speed_latency": 9,
                    "stability": 15,
                    "role_fit": 15,
                    "quality_size": 10,
                    "operational_readiness": 10,
                    "evidence_confidence": 5,
                },
                "confidence": "observed",
                "reasons": ["Observed AMD/VGM Vulkan capacity lane."],
                "blockers": ["requires_amd_vgm_vulkan_path"],
                "evidence": {
                    "source": "experiment_summary",
                    "observation_count": 5,
                    "avg_latency_ms": 21566.6,
                    "avg_total_tokens": 273.0,
                },
                "next_benchmark": "Run task-specific quality checks.",
            },
        ],
        "evidence": {
            "experiment_summary_count": 5,
            "benchmark_record_count": 0,
            "candidate_model_count": 2,
        },
        "next_actions": ["Benchmark unobserved candidates."],
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }


def _comparison() -> dict:
    return {
        "schema_version": "0.1",
        "baseline": {
            "experiment_id": "exp-game-ready",
            "driver": {"vendor": "nvidia", "device_name": "RTX 4050", "version": "596.36", "branch": "game_ready"},
            "lanes": {"fast": "fast-nvidia", "deep": "deep-amd-vgm"},
        },
        "candidate": {
            "experiment_id": "exp-studio",
            "driver": {"vendor": "nvidia", "device_name": "RTX 4050", "version": "596.36", "branch": "studio"},
            "lanes": {"fast": "fast-nvidia", "deep": "deep-amd-vgm"},
        },
        "comparability": {
            "same_driver_version": True,
            "same_lanes": True,
            "warnings": ["driver_branches_differ", "token_counts_differ"],
        },
        "deltas": {
            "fast": {"latency_ms_delta": 36, "latency_ms_delta_pct": 1.68},
            "deep": {"latency_ms_delta": -3165, "latency_ms_delta_pct": -13.21},
        },
        "verdict": "inconclusive_token_counts_differ",
    }
