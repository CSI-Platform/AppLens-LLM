from __future__ import annotations

from pathlib import Path

from applens_llm.runtime_lanes import get_lane, load_runtime_lanes
from applens_llm.schemas import validate_payload


def test_runtime_lanes_schema_accepts_multiple_machine_classes() -> None:
    payload = {
        "schema_version": "0.1",
        "lanes": [
            {
                "lane_id": "fast-nvidia",
                "role": "fast",
                "engine": "llama.cpp",
                "backend": "cuda",
                "endpoint": "http://127.0.0.1:18081/v1",
                "model": {"label": "jan-v35-4b", "path": "models/jan-v35-4b.gguf"},
                "device": {"selector": "cuda:0", "accelerator_ids": ["nvidia-dgpu-0"]},
                "launch": {
                    "server_binary": "llama-server",
                    "context_tokens": 4096,
                    "gpu_layers": 99,
                    "threads": 12,
                    "environment": {},
                },
            },
            {
                "lane_id": "deep-apple",
                "role": "deep",
                "engine": "llama.cpp",
                "backend": "metal",
                "endpoint": "http://127.0.0.1:18082/v1",
                "model": {"label": "gemma-local", "path": "models/gemma.gguf"},
                "device": {"selector": "metal", "accelerator_ids": ["apple-gpu-0"]},
                "launch": {
                    "server_binary": "llama-server",
                    "context_tokens": 8192,
                    "gpu_layers": 99,
                    "threads": 8,
                    "environment": {},
                },
            },
            {
                "lane_id": "cpu-baseline",
                "role": "baseline",
                "engine": "openai-compatible",
                "backend": "cpu",
                "endpoint": "http://127.0.0.1:18083/v1",
                "model": {"label": "tiny-cpu", "path": "models/tiny.gguf"},
                "device": {"selector": "cpu", "accelerator_ids": ["cpu-0"]},
                "launch": {
                    "server_binary": "llama-server",
                    "context_tokens": 2048,
                    "gpu_layers": 0,
                    "threads": 4,
                    "environment": {},
                },
            },
        ],
    }

    validate_payload("runtime-lanes", payload)


def test_load_runtime_lanes_returns_lane_by_id(tmp_path: Path) -> None:
    config = tmp_path / "lanes.json"
    config.write_text(
        '{"schema_version":"0.1","lanes":[{"lane_id":"deep-amd-vgm","role":"deep","engine":"llama.cpp","backend":"vulkan","endpoint":"http://127.0.0.1:18082/v1","model":{"label":"qwen-27b-iq3","path":"models/qwen.gguf"},"device":{"selector":"Vulkan0","accelerator_ids":["amd-igpu-0"]},"launch":{"server_binary":"llama-server","context_tokens":4096,"gpu_layers":99,"threads":12,"environment":{"GGML_VK_DISABLE_COOPMAT":"1"}}}]}',
        encoding="utf-8",
    )

    lanes = load_runtime_lanes(config)
    lane = get_lane(lanes, "deep-amd-vgm")

    assert lane["backend"] == "vulkan"
    assert lane["device"]["accelerator_ids"] == ["amd-igpu-0"]
