from __future__ import annotations

from applens_llm.handoff_contracts import (
    BLACKBOARD_CONTRACT,
    BLACKBOARD_FIRST_SENTENCE,
    build_deep_review_prompt,
    build_fast_lane_prompt,
)


def test_fast_lane_prompt_defines_blackboard_as_jsonl_ledger_not_gpu_memory() -> None:
    prompt = build_fast_lane_prompt(
        original_prompt="Explain the blackboard.",
        fast_lane_id="fast-nvidia",
        deep_lane_id="deep-amd-vgm",
        fast_backend="cuda",
        deep_backend="vulkan",
        iteration_label="loop-1",
    )

    assert BLACKBOARD_CONTRACT in prompt
    assert "append-only JSONL" in prompt
    assert "not CUDA/Vulkan/ROCm shared memory" in prompt
    assert "not pooled VRAM" in prompt
    assert "fast-nvidia" in prompt
    assert "deep-amd-vgm" in prompt
    assert "Explain the blackboard." in prompt
    assert BLACKBOARD_FIRST_SENTENCE in prompt
    assert "answer yes at the AppLens controller/file-ledger layer" in prompt
    assert "Do not answer with a plain no" in prompt
    assert "Keep the fast-lane answer concise" in prompt
    assert "Use at most 120 words" in prompt
    assert "If and only if" in prompt
    assert "Use exactly one copy" in prompt
    assert "For unrelated tasks, do not restate the blackboard contract" in prompt
    assert "fast-nvidia backend=cuda" in prompt
    assert "deep-amd-vgm backend=vulkan" in prompt


def test_fast_lane_prompt_omits_blackboard_boilerplate_for_unrelated_tasks() -> None:
    prompt = build_fast_lane_prompt(
        original_prompt="Review the current model-fit scorecard and propose the next benchmark.",
        fast_lane_id="fast-nvidia",
        deep_lane_id="deep-amd-vgm",
        fast_backend="cuda",
        deep_backend="vulkan",
        iteration_label="loop-2",
    )

    assert BLACKBOARD_FIRST_SENTENCE not in prompt
    assert BLACKBOARD_CONTRACT not in prompt
    assert "Do not imply pooled VRAM or shared GPU memory" in prompt
    assert "Review the current model-fit scorecard" in prompt


def test_deep_review_prompt_forces_correction_of_shared_memory_drift() -> None:
    prompt = build_deep_review_prompt(
        original_prompt="Explain the blackboard.",
        fast_content="The blackboard is shared GPU memory.",
        fast_lane_id="fast-nvidia",
        deep_lane_id="deep-amd-vgm",
        fast_backend="cuda",
        deep_backend="vulkan",
        iteration_label="loop-1",
    )

    assert BLACKBOARD_CONTRACT in prompt
    assert "Treat any claim that the blackboard is GPU shared memory as an error" in prompt
    assert "plain no answer to a blackboard communication question as an error" in prompt
    assert "The blackboard is shared GPU memory." in prompt
    assert "deep-amd-vgm" in prompt
    assert "fast-nvidia backend=cuda" in prompt
    assert "deep-amd-vgm backend=vulkan" in prompt


def test_deep_review_prompt_does_not_overcorrect_unrelated_tasks() -> None:
    prompt = build_deep_review_prompt(
        original_prompt="Review the current model-fit scorecard and propose the next benchmark.",
        fast_content="Next benchmark: CUDA latency variance under load.",
        fast_lane_id="fast-nvidia",
        deep_lane_id="deep-amd-vgm",
        fast_backend="cuda",
        deep_backend="vulkan",
        iteration_label="loop-2",
    )

    assert BLACKBOARD_CONTRACT not in prompt
    assert "Do not penalize the fast lane for omitting blackboard details" in prompt
    assert "Guard only against pooled-memory or shared-GPU-memory claims" in prompt
    assert "CUDA latency variance under load" in prompt
