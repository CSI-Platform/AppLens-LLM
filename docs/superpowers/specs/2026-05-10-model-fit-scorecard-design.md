# Model Fit Scorecard Design

## Product Point

The user-facing AppLens-LLM artifact should be a local model fit meter, not a generic readiness statement. AppLens captures machine evidence, AppLens-Tune reports readiness gaps, and AppLens-LLM ranks concrete local model choices.

The scorecard answers:

```text
Which local LLMs fit this PC best, for which role, on which backend/device, with what confidence?
```

## Required Inputs

- Sanitized `machine-profile` with `hardware_topology`.
- Optional model candidate inventory with family, parameter size, quantization, file size, local status, preferred roles, and sanitized observed model label.
- Optional direct `benchmark-record` files.
- Optional two-lane experiment summaries from the blackboard orchestrator.

## Output Contract

`schemas/model-fit-scorecard.schema.json` ranks each model with:

- score out of 100
- recommended role
- best lane/backend/accelerator IDs
- score breakdown
- observed or inferred confidence
- reasons
- blockers
- next benchmark

The scoring weights are deliberately explicit: capacity fit 25, speed/latency 20, stability 15, role fit 15, quality/size 10, operational readiness 10, evidence confidence 5.

## Interpretation Rules

- Observed benchmark or experiment evidence beats advertised capacity.
- VGM and shared graphics memory can improve capacity, but they are not automatically one pooled device with NVIDIA VRAM.
- Candidate models without direct evidence must stay `inferred` and include `no_observed_benchmark`.
- AMD/VGM Vulkan recommendations should keep readiness checks because the path can depend on VGM activation, backend support, and runtime flags.
- Driver branch/version is evidence, but repeated benchmarks matter more than branch label alone.

## ASUS PX13 First Artifact

The first scorecard uses sanitized ASUS ProArt PX13 evidence:

- Jan v3.5 4B Q4 on NVIDIA/CUDA: excellent fast-chat fit from observed latency and stability.
- Qwen3.5 27B IQ3 on AMD/VGM Vulkan: good deep-review/capacity fit from observed successful runs, with slower latency.
- Gemma E4B Q4: inferred candidate until directly benchmarked.

This is portable beyond the ASUS laptop because the contract speaks in machine profiles, accelerators, lanes, model candidates, and benchmark evidence rather than hard-coded device paths.
