# Fit Report Design

## Goal

`fit-report` is the first product-facing AppLens-LLM decision artifact. It turns a machine profile plus benchmark and experiment evidence into one schema-valid JSON recommendation.

## Scope

The report answers:

- What class of local AI machine is this?
- Which runtime strategy is currently supported by evidence?
- Which capacity claims remain unsupported?
- Which lanes, backends, and accelerator IDs have actually worked?
- What should be benchmarked next?

The report does not run models, change drivers, install services, expose network endpoints, or include raw local model paths.

## Inputs

- One `machine-profile` JSON object or JSONL row selected by `machine_id`.
- Optional `benchmark-record` JSON files.
- Optional `experiment-run` summary JSON files.
- Optional `experiment-compare` JSON files.

## Output

The output is a `fit-report` JSON document with:

- machine summary
- fit class and confidence
- capacity assessment
- proven runtime lanes
- unsupported memory claims
- runtime recommendation
- evidence summaries
- decisions
- next benchmarks
- privacy flags

## Current ASUS PX13 Interpretation

The current local evidence supports a `hybrid_local_ai_worker` classification with a `two_lane_local` strategy:

- NVIDIA/CUDA lane for fast small-model work.
- AMD/VGM Vulkan lane for slower larger-model capacity work.
- RTX VRAM plus AMD VGM remains an unverified pooled-memory claim.
- NVIDIA Game Ready versus Studio branch should be recorded but not over-weighted without repeated benchmark proof.

## Testing

Tests cover schema validation, multi-row machine-profile JSONL loading, report construction, and CLI writing. A sanitized example report lives under `examples/`.
