from __future__ import annotations

import json
from pathlib import Path

from applens_llm.context_envelope import (
    DEFAULT_CONTEXT_TIERS,
    build_context_envelope,
    load_context_observations,
    write_context_envelope,
)
from applens_llm.schemas import validate_payload


def test_build_context_envelope_tapers_from_advertised_context() -> None:
    envelope = build_context_envelope(
        machine_profile=_machine_profile(),
        model_candidates=_model_candidates(),
        context_observations=[],
        created_at="2026-05-12T18:00:00Z",
        envelope_id="ctx-asus",
    )

    validate_payload("context-envelope", envelope)
    qwen = _by_model(envelope)["qwen35-27b-iq3"]
    assert qwen["advertised_context_tokens"] == 262144
    assert qwen["advertised_context"]["confidence"] == "advertised"
    assert qwen["planned_context_tiers"] == [262144, 131072, 65536, 32768, 16384, 8192, 4096]
    assert qwen["max_tested_context_tokens"] == 0
    assert qwen["max_recommended_context_tokens"] == 0
    assert qwen["status"] == "needs_context_benchmark"
    assert qwen["context_evidence_status"] == "advertised_unproven"
    assert "not a performance finding" in qwen["context_interpretation"]
    assert "advertised_context_unproven" in qwen["blockers"]


def test_context_envelope_recommends_per_workload_from_observations() -> None:
    envelope = build_context_envelope(
        machine_profile=_machine_profile(),
        model_candidates=_model_candidates(),
        context_observations=[
            _observation("qwen35-27b-iq3", context_tokens=65536, status="pass", quality=76, generation_tps=5.4),
            _observation("qwen35-27b-iq3", context_tokens=131072, status="oom", quality=0, generation_tps=0),
            _observation("gemma4-26b-a4b-q3km", context_tokens=16384, status="pass", quality=89, generation_tps=22.0),
            _observation("gemma4-26b-a4b-q3km", context_tokens=65536, status="oom", quality=0, generation_tps=0),
        ],
        created_at="2026-05-12T18:00:00Z",
        envelope_id="ctx-asus",
    )

    qwen = _by_model(envelope)["qwen35-27b-iq3"]
    gemma = _by_model(envelope)["gemma4-26b-a4b-q3km"]
    assert qwen["max_loadable_context_tokens"] == 65536
    assert qwen["max_recommended_context_tokens"] == 65536
    assert qwen["workload_recommendations"]["long_context_retrieval"]["context_tokens"] == 65536
    assert qwen["context_evidence_status"] == "observed_useful"
    assert gemma["max_loadable_context_tokens"] == 16384
    assert gemma["workload_recommendations"]["coding"]["context_tokens"] == 16384
    assert gemma["workload_recommendations"]["coding"]["reason"] == "highest_quality_at_stable_context"
    assert envelope["comparisons"][0]["summary"].startswith("Gemma 4 26B-A4B")


def test_context_envelope_marks_load_only_context_as_limited_not_bad() -> None:
    envelope = build_context_envelope(
        machine_profile=_machine_profile(),
        model_candidates=_model_candidates(),
        context_observations=[
            _observation(
                "qwen35-27b-iq3",
                context_tokens=16384,
                status="pass",
                quality=0,
                generation_tps=5.2,
            )
        ],
        created_at="2026-05-12T18:00:00Z",
        envelope_id="ctx-asus-load-only",
    )

    qwen = _by_model(envelope)["qwen35-27b-iq3"]
    assert qwen["max_loadable_context_tokens"] == 16384
    assert qwen["max_recommended_context_tokens"] == 0
    assert qwen["context_evidence_status"] == "observed_limited"
    assert "no stable useful context tier" in qwen["context_interpretation"]


def test_write_context_envelope_loads_json_inputs(tmp_path: Path) -> None:
    machine = tmp_path / "machine.json"
    candidates = tmp_path / "models.json"
    observations = tmp_path / "observations.jsonl"
    output = tmp_path / "context-envelope.json"
    machine.write_text(json.dumps(_machine_profile()), encoding="utf-8")
    candidates.write_text(json.dumps({"models": _model_candidates()}), encoding="utf-8")
    observations.write_text(json.dumps(_observation("gemma4-26b-a4b-q3km", context_tokens=16384)) + "\n", encoding="utf-8")

    envelope = write_context_envelope(
        machine_profile_path=machine,
        model_candidates_path=candidates,
        context_observation_paths=[observations],
        output_path=output,
        created_at="2026-05-12T18:00:00Z",
        envelope_id="ctx-file",
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == envelope
    assert load_context_observations([observations])[0]["context_tokens"] == 16384


def test_load_context_observations_reads_multi_line_jsonl(tmp_path: Path) -> None:
    observations = tmp_path / "observations.jsonl"
    observations.write_text(
        "\n".join(
            [
                json.dumps(_observation("qwen35-27b-iq3", context_tokens=8192)),
                json.dumps(_observation("qwen35-27b-iq3", context_tokens=16384)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_context_observations([observations])

    assert [row["context_tokens"] for row in rows] == [8192, 16384]


def test_default_context_tiers_do_not_exceed_advertised_context() -> None:
    envelope = build_context_envelope(
        machine_profile=_machine_profile(),
        model_candidates=[
            {
                "model_id": "tiny-local",
                "display_name": "Tiny Local",
                "family": "qwen",
                "parameter_size_b": 2,
                "quantization": "Q4_K_M",
                "local_status": "local",
                "preferred_roles": ["fast_chat"],
                "quality_prior": "medium",
                "advertised_context_tokens": 32768,
                "advertised_context_source": "unit test",
            }
        ],
        context_observations=[],
        created_at="2026-05-12T18:00:00Z",
        envelope_id="ctx-tiny",
    )

    model = envelope["models"][0]
    assert model["planned_context_tiers"] == [32768, 16384, 8192, 4096]
    assert all(tier <= 32768 for tier in model["planned_context_tiers"])
    assert DEFAULT_CONTEXT_TIERS[0] > 32768


def _by_model(envelope: dict) -> dict[str, dict]:
    return {model["model_id"]: model for model in envelope["models"]}


def _observation(
    model_id: str,
    *,
    context_tokens: int,
    status: str = "pass",
    quality: float = 80,
    generation_tps: float = 10,
) -> dict:
    return {
        "model_id": model_id,
        "context_tokens": context_tokens,
        "backend": "vulkan",
        "devices_used": ["amd-igpu-0" if "gemma" in model_id else "nvidia-dgpu-0"],
        "status": status,
        "quality_score_pct": quality,
        "generation_tokens_per_second": generation_tps,
        "prompt_tokens_per_second": 120,
        "failure_modes": ["none"] if status == "pass" else [status],
        "workloads": ["long_context_retrieval", "coding", "summarization"],
        "notes": "Sanitized unit observation.",
    }


def _model_candidates() -> list[dict]:
    return [
        {
            "model_id": "qwen35-27b-iq3",
            "display_name": "Qwen3.5 27B IQ3",
            "family": "qwen",
            "parameter_size_b": 27,
            "quantization": "IQ3_M",
            "local_status": "local",
            "preferred_roles": ["deep_review", "coding"],
            "quality_prior": "high",
            "advertised_context_tokens": 262144,
            "advertised_context_source": "https://artificialanalysis.ai/leaderboards/models?size=small",
        },
        {
            "model_id": "gemma4-26b-a4b-q3km",
            "display_name": "Gemma 4 26B-A4B Q3_K_M",
            "family": "gemma",
            "parameter_size_b": 26,
            "quantization": "UD-Q3_K_M",
            "local_status": "local",
            "preferred_roles": ["deep_review", "coding"],
            "quality_prior": "high",
            "advertised_context_tokens": 262144,
            "advertised_context_source": "https://artificialanalysis.ai/leaderboards/models?size=small",
        },
    ]


def _machine_profile() -> dict:
    return {
        "schema_version": "0.1",
        "machine_id": "asus-px13-current",
        "label": "ASUS ProArt PX13 current local run",
        "capture_status": "captured_raw",
        "capture_priority": 1,
        "platform": {
            "vendor": "asus",
            "model": "ProArt PX13",
            "sku": "asus-proart-px13-sanitized-current",
            "os_family": "windows",
            "cpu": "AMD Ryzen AI 9 HX 370",
            "ram_gb": 32,
            "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU + AMD Radeon 890M",
            "vram_mb": 6141,
        },
        "hardware_topology": {
            "accelerators": [],
            "usable_inference_capacity": {
                "estimated_usable_memory_mb": 6141,
                "confidence": "observed",
                "preferred_accelerator_ids": ["nvidia-dgpu-0"],
                "mixed_device_pooling": "unverified",
                "verification": [{"source_type": "inventory", "notes": "Unit test."}],
            },
            "memory_claims": [],
        },
        "target_roles": ["training_candidate"],
        "collection": {
            "applens_report": "captured",
            "applens_tune_report": "captured",
            "local_ai_profile": "captured",
            "llm_bench": "captured",
            "sanitized": False,
        },
        "notes": "Unit test profile.",
    }
