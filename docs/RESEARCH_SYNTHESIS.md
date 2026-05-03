# Research Synthesis

The local research set converged on the same product shape:

- AppLens measures the machine.
- AppLens-Tune stays read-only for system audit in V1.
- AppLens-LLM becomes the action-taking local AI deployment tailor.
- The first action boundary is model runtime configuration, not system remediation.
- The machine itself is the verifier: launch, benchmark, score, and reject unsafe fits.

## Build Implication

The next deliverable is not a large training run. It is an eval harness plus AppLens-Bench evidence.

Training starts only after we can measure:

- schema-valid deployment plans
- safe versus gated action agreement
- runtime/model/quantization fit
- benchmark grounding
- failure modes such as OOM, bad host binding, missing training manifest, and unsupported network exposure

## What We Accept

- Qwen3.5-2B as the first AppLens-Tailor LoRA target.
- Qwen3.5-0.8B later for planner or process-reward roles.
- llama.cpp and Jan as first OpenAI-compatible runtime endpoints.
- AppLens-Bench as the proof loop and future dataset generator.
- Symphony-style workflow files and approvals as a later orchestration pattern.

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

This repo now treats that mapping as an executable contract: training examples, schema validation, eval reports, and benchmark records.
