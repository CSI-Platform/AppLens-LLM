from __future__ import annotations


BLACKBOARD_CONTRACT = (
    "In AppLens-LLM, the blackboard is an append-only JSONL coordination and evidence ledger "
    "written by blackboard.py. It records tasks, model responses, failures, handoffs, benchmark "
    "references, and verdicts. It is not CUDA/Vulkan/ROCm shared memory, not inter-GPU IPC, "
    "not pooled VRAM, and not a claim that NVIDIA and AMD memory are one device."
)

BLACKBOARD_FIRST_SENTENCE = (
    "The AppLens blackboard is an append-only JSONL evidence ledger, not GPU shared memory or pooled VRAM."
)

BLACKBOARD_COMMUNICATION_RULE = (
    "If the task asks whether lanes or models communicate through the blackboard, answer yes at the "
    "AppLens controller/file-ledger layer and no at the GPU memory/native API layer. Do not answer "
    "with a plain no, because that erases the AppLens handoff mechanism."
)

BLACKBOARD_SCOPE_RULE = (
    "If and only if the user task itself asks about blackboards, handoffs, lane communication, pooled "
    "memory, shared GPU memory, or GPU memory claims: Use exactly one copy of the required first "
    "sentence. For unrelated tasks, do not restate the blackboard contract; answer the task normally "
    "while preserving the contract."
)


def build_fast_lane_prompt(
    *,
    original_prompt: str,
    fast_lane_id: str,
    deep_lane_id: str,
    iteration_label: str,
    fast_backend: str | None = None,
    deep_backend: str | None = None,
) -> str:
    lane_context = _lane_context(
        fast_lane_id=fast_lane_id,
        deep_lane_id=deep_lane_id,
        fast_backend=fast_backend,
        deep_backend=deep_backend,
    )
    if _needs_full_blackboard_rule(original_prompt):
        contract_block = (
            f"Blackboard contract:\n{BLACKBOARD_CONTRACT}\n\n"
            f"Conditional required first sentence:\n"
            f"{BLACKBOARD_FIRST_SENTENCE}\n\n"
            f"Communication rule:\n{BLACKBOARD_COMMUNICATION_RULE}\n\n"
            f"Scope rule:\n{BLACKBOARD_SCOPE_RULE}\n\n"
            "Answer the task using that definition. If the task asks whether lanes communicate through "
            "the blackboard, explain that the models communicate indirectly through the AppLens JSONL "
            "ledger and controller prompts, not through GPU memory or native CUDA/Vulkan/ROCm APIs.\n\n"
            "Complete the controller/file-ledger versus GPU-memory distinction before adding details.\n\n"
        )
    else:
        contract_block = (
            "Handoff constraint:\n"
            "Keep the configured lanes separate. Do not imply pooled VRAM or shared GPU memory, "
            "and do not restate the blackboard contract unless the task asks about handoff mechanics.\n\n"
            "Answer the task directly under that constraint.\n\n"
        )

    return (
        f"{iteration_label}: You are the fast lane ({fast_lane_id}) in an AppLens-LLM handoff to "
        f"{deep_lane_id}.\n\n"
        f"Configured lanes:\n{lane_context}\n\n"
        f"{contract_block}"
        "Keep the fast-lane answer concise. Use at most 120 words unless the task explicitly "
        "requests a longer artifact.\n\n"
        f"Task:\n{original_prompt}"
    )


def build_deep_review_prompt(
    *,
    original_prompt: str,
    fast_content: str,
    fast_lane_id: str,
    deep_lane_id: str,
    iteration_label: str,
    fast_backend: str | None = None,
    deep_backend: str | None = None,
) -> str:
    lane_context = _lane_context(
        fast_lane_id=fast_lane_id,
        deep_lane_id=deep_lane_id,
        fast_backend=fast_backend,
        deep_backend=deep_backend,
    )
    if _needs_full_blackboard_rule(original_prompt) or _needs_full_blackboard_rule(
        fast_content,
        include_backend_terms=False,
    ):
        review_block = (
            f"Blackboard contract:\n{BLACKBOARD_CONTRACT}\n\n"
            "Treat any claim that the blackboard is GPU shared memory as an error. Treat a "
            "plain no answer to a blackboard communication question as an error unless it preserves "
            "the yes-at-ledger/no-at-GPU distinction. Correct confusion between AppLens' JSONL "
            "coordination ledger and native CUDA/Vulkan/ROCm memory behavior. "
            "Also identify concrete runtime evidence gaps such as backend, device, latency, fallback, "
            "OOM, CPU spill, thermal behavior, or model-fit confidence. Keep the answer concise.\n\n"
        )
    else:
        review_block = (
            "Review rule:\n"
            "Review the fast lane's answer for the original task. Do not penalize the fast lane for "
            "omitting blackboard details. Guard only against pooled-memory or shared-GPU-memory claims, "
            "backend mistakes, unsupported benchmark claims, and missing evidence. Keep the answer concise.\n\n"
        )

    return (
        f"{iteration_label}: You are the deep review lane ({deep_lane_id}) reviewing the "
        f"{fast_lane_id} response.\n\n"
        f"Configured lanes:\n{lane_context}\n\n"
        f"{review_block}"
        f"Original task:\n{original_prompt}\n\nFast lane response:\n{fast_content}"
    )


def _lane_context(
    *,
    fast_lane_id: str,
    deep_lane_id: str,
    fast_backend: str | None,
    deep_backend: str | None,
) -> str:
    fast = f"{fast_lane_id} backend={fast_backend}" if fast_backend else fast_lane_id
    deep = f"{deep_lane_id} backend={deep_backend}" if deep_backend else deep_lane_id
    return f"- fast: {fast}\n- deep: {deep}"


def _needs_full_blackboard_rule(prompt: str, *, include_backend_terms: bool = True) -> bool:
    normalized = prompt.lower()
    keywords = (
        "blackboard",
        "handoff",
        "handoffs",
        "lane",
        "lanes",
        "communicate",
        "communication",
        "pooled",
        "pool",
        "vram",
        "shared memory",
        "gpu memory",
    )
    backend_keywords = ("cuda", "vulkan", "rocm")
    if include_backend_terms:
        keywords = keywords + backend_keywords
    return any(keyword in normalized for keyword in keywords)
