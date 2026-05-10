# Research Synthesis

The local research set converged on the same product shape:

- AppLens measures the machine.
- AppLens-Tune stays read-only for system audit in V1.
- AppLens-LLM becomes the action-taking local AI deployment tailor.
- The first action boundary is model runtime configuration, not system remediation.
- The machine itself is the verifier: launch, benchmark, score, and reject unsafe fits.
- The first product artifact is a model fit scorecard: a ranked local model meter backed by schemas and benchmark evidence.

## Build Implication

The next deliverable is not a large training run. It is an eval harness, AppLens-Bench evidence, and model fit scorecards that convert evidence into ranked local model decisions.

Training starts only after we can measure:

- schema-valid deployment plans
- safe versus gated action agreement
- runtime/model/quantization fit
- benchmark grounding
- model ranking and score explanation
- failure modes such as OOM, bad host binding, missing training manifest, and unsupported network exposure

## Product Critique

An uninterested third party would say AppLens, AppLens-Tune, and AppLens-LLM only become a product when they turn low-level machine evidence into a concrete local model choice. Inventory alone is not enough, and "ready/not ready" is too vague.

The stronger shape is:

- AppLens captures hardware, drivers, memory claims, and local model inventory.
- AppLens-Tune reports readiness gaps such as VGM inactive, missing runtime binaries, driver branch/version changes, thermal constraints, or restart-required state.
- AppLens-LLM ranks local models by fit score, recommended role, backend/device lane, blockers, and confidence.

That scorecard is the meter the user asked for: `Jan v3.5 4B Q4 is 98/100 for fast chat because it is observed, fast, stable, and fits CUDA`; `Qwen3.5 27B IQ3 is 89/100 for deep review because it works on AMD/VGM Vulkan but is slower`; an unbenchmarked Gemma candidate cannot get an observed score until it runs.

## What We Accept

- Qwen3.5-2B as the first AppLens-Tailor LoRA target.
- Qwen3.5-0.8B later for planner or process-reward roles.
- llama.cpp and Jan as first OpenAI-compatible runtime endpoints.
- AppLens-Bench as the proof loop and future dataset generator.
- Symphony-style workflow files and approvals as a later orchestration pattern.

## Reported Memory Versus Usable Capacity

AppLens-LLM must distinguish advertised or allocated graphics memory from benchmark-proven local LLM capacity. AMD VGM is a real mechanism for reallocating system RAM to integrated graphics, but that does not make NVIDIA dGPU VRAM plus AMD iGPU VGM one clean pooled inference device. Shared graphics memory is also not equivalent to dedicated GPU memory for local model placement.

The ASUS ProArt PX13 case is now a policy edge case:

- Inventory may show RTX 4050 Laptop GPU 6 GB, Radeon 890M, Ryzen AI NPU, and 32 GB system RAM.
- A user-facing claim may describe a 22 GB pool from RTX 4050 6 GB plus Radeon 890M VGM 16 GB.
- The checked Windows state showed Radeon 890M dedicated graphics memory at 512 MB, so 16 GB VGM was not active.
- Until benchmark records prove backend, devices used, memory use, mixed offload, CPU spill, and failure behavior, AppLens-LLM must plan against proven or conservatively inferred capacity only.

Training and eval examples should reward language like "claim requires benchmark verification" and penalize recommendations that simply add advertised or reserved memory totals.

For Radeon/VGM runs, AMD Software: Adrenalin Edition is an acceptable telemetry source. AppLens-LLM should request or reference its performance log when the question depends on AMD GPU utilization, dedicated memory use, clocks, power, temperature, CPU load, or system memory pressure. FPS and frame-generation latency are graphics-workload metrics and are normally irrelevant to local LLM inference.

Initial ASUS PX13 llama.cpp evidence:

- VGM activation was confirmed through Windows graphics registry memory: Radeon 890M dedicated memory reached 16 GB.
- llama.cpp b8892 Vulkan saw both devices: `Vulkan0` Radeon 890M and `Vulkan1` RTX 4050.
- AMD Vulkan required `GGML_VK_DISABLE_COOPMAT=1` to avoid a cooperative-matrix extension load failure.
- On a small 4B Q4 model, RTX 4050 Vulkan was faster than Radeon 890M VGM.
- On an 11.7 GB 27B IQ3 model, Radeon 890M VGM loaded and generated slowly; RTX 4050 Vulkan failed with out-of-device-memory.
- This proves VGM can be usable local inference capacity for a Vulkan llama.cpp path. It does not prove a single pooled RTX plus Radeon memory device.

NVIDIA driver branch should be recorded as runtime evidence, but the first Game Ready versus Studio comparison did not show a material local-LLM difference. Both runs used NVIDIA driver `596.36`; the NVIDIA/CUDA fast lane differed by only `36 ms`. A Studio repeatability pass showed a `247 ms` spread on the fast lane and a `3378 ms` spread on the AMD/VGM deep lane, so AppLens-LLM should not over-weight Game Ready versus Studio without repeated benchmark proof.

Reference sources:

- AMD VGM FAQ: https://www.amd.com/en/blogs/2025/faqs-amd-variable-graphics-memory-vram-ai-model-sizes-quantization-mcp-more.html
- AMD Software performance metrics logging: https://www.amd.com/en/resources/support-articles/faqs/DH3-038.html
- AMD Ryzen AI 9 HX 370 specs: https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html
- llama.cpp compute backends: https://www.mintlify.com/ggml-org/llama.cpp/concepts/backends
- Ollama GPU docs: https://docs.ollama.com/gpu
- Ollama FAQ GPU placement check: https://docs.ollama.com/faq
- LM Studio multi-GPU controls: https://lmstudio.ai/blog/lmstudio-v0.3.14

## What We Defer

- GRPO until schema, policy, and benchmark rewards exist.
- Squeez until snapshot/log size becomes a real bottleneck.
- PRM and multi-agent LoRA library until the single Tailor target plateaus.
- Any AppLens-Tune system remediation until rollback, backup, and approval infrastructure exists.

## Rejected Or Low-Confidence Inputs

- Product pivots toward SIEM, PCAP, Zeek, or Azure App Service framing.
- Unverified Qwen architecture claims from secondary blogs.
- Treating HAQA or RooflineBench as direct training datasets.
- Network exposure or service changes as V1 action space.

## Current Target

AppLens-Tailor learns this mapping:

```text
AppLens snapshot + AppLens-Tune local AI profile + benchmark facts + workload request
-> deployment-plan JSON
```

This repo now treats that mapping as an executable contract: training examples, schema validation, eval reports, benchmark records, and model fit scorecards.

The first fit report generated from this PX13 evidence classifies the machine as a `hybrid_local_ai_worker` with a `two_lane_local` strategy. The first scorecard turns that into ranked model choices. That is enough to continue product development without model training yet: the next gap is more benchmark coverage, candidate model inventory, and scorecard quality, not a LoRA.
