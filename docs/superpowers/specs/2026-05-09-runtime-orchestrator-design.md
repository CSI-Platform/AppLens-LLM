# Runtime Orchestrator Design

Date: 2026-05-09

## Decision

Build AppLens-LLM option 2: a blackboard-backed runtime orchestrator that can run and compare multiple local llama.cpp lanes.

The first concrete use case is the ASUS ProArt PX13 with an RTX 4050 lane and an AMD Radeon 890M VGM lane, but the design must not be laptop-specific. Lanes are configuration records that describe runtime endpoint, engine, backend, device selector, model path, model label, and hardware accelerator IDs. Any future machine can add lanes for CUDA, Vulkan, ROCm/HIP, Metal, DirectML, CPU, remote llama.cpp RPC, or other OpenAI-compatible local endpoints.

## External Grounding

llama.cpp is a reasonable base runtime because its current documentation describes broad backend support across CUDA, HIP/ROCm, Vulkan, Metal, SYCL, CPU, and other targets. Its server exposes an HTTP API with OpenAI-compatible chat/completion behavior, and its GPU options include explicit device selection and multi-GPU configuration. The orchestrator should treat those capabilities as runtime facts to probe and record, not as guarantees that any specific backend/device combination will work on every host.

Sources:

- https://www.mintlify.com/ggml-org/llama.cpp/concepts/backends
- https://www.mintlify.com/ggml-org/llama.cpp/inference/server
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

## Architecture

The orchestrator has three small parts:

1. Runtime lane config
2. Append-only blackboard ledger
3. Controller CLI

Runtime lane configs define how to reach or launch a model runtime. The config is generic: it records labels such as `fast_nvidia` or `deep_amd_vgm`, but the behavior comes from fields like endpoint URL, backend, device selector, model path, context size, threads, GPU layers, and environment variables.

The blackboard ledger is append-only JSONL under `out/blackboard/` by default. It records experiment events, tasks, model responses, handoffs, benchmark references, failures, and verdicts. Local paths may appear in ignored `out/` artifacts, but any committed example must sanitize paths, device IDs, serials, UUIDs, product IDs, and host-specific identifiers.

The controller CLI reads a lane config, sends prompts to one or more OpenAI-compatible endpoints, records response metadata, and writes blackboard events. It does not assume one model is smarter. It captures evidence so later evals can compare speed, capacity, quality, fallback, and failure modes.

## Data Flow

1. User creates or selects an experiment.
2. Controller appends `experiment_started`.
3. Controller appends one or more `task` records.
4. Controller sends the task to a configured lane.
5. Controller appends `model_response` with lane, model, endpoint, backend, device IDs, latency, token counts when available, and outcome.
6. Controller can append a `handoff` from one lane to another.
7. Controller can append `benchmark_reference` events that point to existing benchmark artifacts.
8. Controller appends a `verdict` only when there is enough evidence to compare outputs.

## Scalability Rules

The design scales beyond this laptop if these constraints hold:

- No hard-coded ASUS, RTX 4050, Radeon 890M, or VGM behavior in orchestrator logic.
- Lane configs reference accelerator IDs from `hardware_topology` instead of raw OS device IDs.
- Backends are enum/string facts such as `cuda`, `vulkan`, `rocm`, `metal`, `directml`, or `cpu`.
- Device selectors are runtime-specific strings and are recorded exactly as used.
- The orchestrator supports one lane, two lanes, or many lanes.
- The blackboard records failures as first-class outcomes, including timeout, connection failure, OOM, fallback, crash, and invalid response.
- AppLens-LLM never converts advertised memory into usable capacity without benchmark proof.

## Error Handling

Each model call records either a successful response or a failure event. A failed lane does not corrupt the experiment. The controller should keep enough detail to reproduce the failed call while avoiding secrets and private identifiers in commit-safe examples.

Initial failure modes:

- endpoint unavailable
- timeout
- non-JSON response from an expected JSON task
- HTTP error
- model/runtime OOM reported by server logs or benchmark artifacts
- fallback suspected or observed
- response rejected by evaluator

## Testing

Tests should come first.

Minimum test coverage:

- lane config schema accepts NVIDIA, AMD/Vulkan, Apple/Metal, CPU, and generic OpenAI-compatible lanes
- blackboard writer appends valid JSONL events
- controller can call a fake OpenAI-compatible endpoint and record a response
- controller records timeout/HTTP failures without dropping the task
- committed examples contain sanitized identifiers only

## Non-Goals

This phase does not implement training, GRPO, distillation, model downloads, driver changes, service installation, or automatic overnight scheduling. Those can consume blackboard evidence later.

This phase does not depend on Stigmergent. It borrows the blackboard idea but keeps AppLens-LLM self-contained.
