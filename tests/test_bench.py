from __future__ import annotations

from applens_llm.bench import build_benchmark_record, build_chat_payload
from applens_llm.schemas import validate_payload


def test_chat_payload_uses_openai_compatible_shape() -> None:
    payload = build_chat_payload("qwen-local", "Return strict JSON.", max_tokens=64)

    assert payload == {
        "model": "qwen-local",
        "messages": [{"role": "user", "content": "Return strict JSON."}],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }


def test_benchmark_record_builder_matches_schema() -> None:
    response = {
        "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        "choices": [{"message": {"content": "{}"}}],
    }

    record = build_benchmark_record(
        run_id="unit-test",
        created_at="2026-05-03T20:00:00Z",
        host={"name": "test-host", "os": "Windows", "gpu": "none", "vram_mb": 0},
        runtime={"backend": "jan", "build": "unknown", "command": "POST /v1/chat/completions"},
        model={"name": "qwen-local", "path": "local", "quantization": "unknown"},
        response_json=response,
        latency_ms=300,
        vram_used_mb=0,
        temperature_c=0,
        notes="Unit test completion.",
    )

    validate_payload("benchmark-record", record)
    assert record["metrics"]["generation_tokens_per_second"] == 20
