# Architecture

AppLens-LLM is the model-fitting workspace beside AppLens.

AppLens measures the machine. AppLens-Tune describes workstation readiness. AppLens-LLM converts those measurements, workload goals, model inventory, and benchmark records into a model fit scorecard first, then fit reports and deployment plans where needed.

## Boundaries

- AppLens remains the inventory and measurement source.
- AppLens-Tune remains the workstation readiness and tuning advisor.
- AppLens-LLM owns local model fit: runtime selection, model artifact selection, benchmark evidence, training data, and evals.

V1 may start a user-owned local model runtime and write benchmark logs. It may record GPU driver version and user-confirmed driver branch as evidence. It must not change services, startup entries, drivers, firewall rules, firmware, or user data.

## Contracts

- `schemas/hardware-topology.schema.json`: accelerator inventory, memory claims, and usable inference capacity evidence.
- `schemas/runtime-lanes.schema.json`: portable local runtime lane configs for CUDA, Vulkan, ROCm/HIP, Metal, DirectML, CPU, RPC, and OpenAI-compatible endpoints.
- `schemas/blackboard-record.schema.json`: append-only experiment events for tasks, handoffs, responses, failures, benchmark references, and verdicts.
- `schemas/workload-profile.schema.json`: workload-owned `.applens/` adapter contract with actions, artifacts, model role needs, and safety gates.
- `schemas/autoresearch-run-manifest.schema.json`: one bounded self-fit, workload, or dry-run loop.
- `schemas/workload-artifact.schema.json`: evidence references produced by workload loops.
- `schemas/autoresearch-probes.schema.json` and `schemas/autoresearch-eval-cases.schema.json`: committed probes and regression cases for workload behavior.
- `schemas/model-fit-scorecard.schema.json`: primary product artifact for ranking local models by score, role, backend/device lane, blockers, and confidence.
- `schemas/local-capability-record.schema.json`: AppLens-local capability proof for strict JSON, tool-call emulation, coding, hardware reasoning, benchmark interpretation, safety boundaries, handoffs, and thinking-mode variants.
- `schemas/context-envelope.schema.json`: advertised-versus-proven context capacity by model, backend/device lane, and workload.
- `schemas/deployment-plan.schema.json`: scorecard-driven outfitting contract for local model assignments, runtime profiles, preflight actions, AppLens-Tune recommendations, and supervisor replacement gates.
- `schemas/benchmark-record.schema.json`: benchmark proof attached to a recommendation.
- `schemas/training-example.schema.json`: supervised training and eval row shape.
- `schemas/capture-record.schema.json`: ignored raw-intake manifest row for capture folder review.

Every training example keeps both a chat `messages` view and a structured `expected_output`. The chat view is used for SFT. The structured output is used for schema and policy validation.

Machine profiles keep legacy flat `platform.gpu` and `platform.vram_mb` as summary labels, but deployment decisions must read `hardware_topology`. The topology lists accelerators separately, including discrete GPUs, integrated GPUs, NPUs, and CPU fallback entries when useful. Each accelerator records physical dedicated VRAM, VGM or reserved memory, shared graphics memory, reported total graphics memory, estimated usable inference memory, confidence, and verification source.

Benchmark records are the proof layer. They must capture the runtime engine, actual acceleration backend such as CUDA, Vulkan, ROCm, DirectML, Metal, CPU, or unknown, devices used, mixed-device offload status, memory used per device, CPU spill, failure modes, fallback, tokens/sec, and thermal notes.

Local capability records are the agent-behavior proof layer. `llama-bench` can prove that a model loads and how fast it generates, but it cannot prove tool discipline, coding usefulness, hardware reasoning, or whether thinking mode helps enough to justify latency. `applens-local-v1` records those behaviors as model-specific evidence and feeds scorecards through the `agent_capability` score.

Context envelopes are the long-context proof layer. A model card or leaderboard may advertise 256k or 262k context, but AppLens-LLM records that as a claim until this machine proves a tier can load, remain stable, generate at acceptable speed, and pass long-context quality checks. The taper starts from the advertised tier and steps down through 128k, 64k, 32k, 16k, 8k, and 4k as needed.

The contract deliberately separates memory claims from usable capacity. For example, an RTX 4050 6 GB plus Radeon 890M 16 GB VGM "22 GB pool" is recorded as a claim until benchmarks prove a runtime used those devices together without OOM, fallback, or unacceptable CPU spill.

Runtime lanes are the orchestration layer above benchmark records. A lane records an endpoint, engine, backend, model label/path, device selector, accelerator IDs, and launch hints. The first two-lane experiment can use an RTX lane and an AMD/VGM lane, but the schema is portable across future hosts and backend types.

Blackboard records are the experiment ledger. They let AppLens-LLM route one task through one or more lanes, capture successes and failures, and later compare evidence without assuming that advertised capacity or a specific vendor path is inherently better. The blackboard is an AppLens controller/file-ledger mechanism only: lanes communicate through appended JSONL records and follow-up prompts, not through pooled VRAM, CUDA/Vulkan/ROCm shared memory, inter-GPU IPC, or proof that NVIDIA and AMD memory are one device.

Fast-to-deep handoff prompts include that contract directly. If a model is asked whether lanes communicate through the blackboard, the correct distinction is yes through the JSONL ledger and controller prompts, no through GPU memory or native acceleration APIs.

AutoResearch uses the same blackboard idea as a universal workload event protocol. Blackboard is append-only run evidence. Memory/wiki is curated durable knowledge. Workloads such as Oracle own domain code and expose allowlisted commands through `.applens/commands.json`; AppLens-LLM reads those contracts, executes only approved commands, and writes proposed memory without promoting it automatically.

Model fit scorecards are the product surface above those evidence layers. They should not merely say that a machine is ready. They rank concrete local model choices, explain why each score landed where it did, and separate observed fits from inferred candidates. This lets AppLens answer "Qwen 4B is the better fast-chat fit on CUDA" and "Qwen 27B is the better deep-review capacity fit on AMD/VGM Vulkan" without pretending those lanes form one pooled VRAM device.

Scorecards combine hardware fit and local-agent behavior. A small model can win the fast lane on tokens/sec and still lose agentic work if it fails strict JSON, tool selection, coding, or handoff tests. A larger AMD/VGM model can be recommended as a planner or deep-review lane only after capability evidence proves it is better enough to justify slower output.

Scorecards also combine context evidence. A model can advertise 256k context and still be recommended at 16k if larger contexts OOM, fall back to CPU, become too slow, or fail retrieval/summary/codebase tasks. Conversely, a slower model can win the long-context workload if it remains useful at a larger proven tier.

The JSON scorecard is the durable contract. Static HTML reports are generated views for humans: sortable/filterable tables over the scorecard plus optional experiment comparisons. HTML should not become the canonical data format.

Deployment plans sit one layer above scorecards. They are the outfitting artifact: which model runs as the primary local worker, which model runs as deep review, what llama.cpp profile to launch, which context tier to use, what AppLens-Tune should prepare, and which gates block local supervisor promotion. The cloud/API planner-supervisor is recorded as the 100-point reference baseline until a local model passes replacement gates for tool calling, coding, handoff planning, context behavior, and runtime stability.

## Runtime Path

1. Ingest AppLens capture folders into ignored `capture-record` manifests.
2. Promote reviewed captures into sanitized machine profiles or training examples.
3. Read workload intent.
4. Import local model candidates and discovered model inventory.
5. Run or import benchmark records.
6. Run or import `applens-local-v1` capability records for top candidates and thinking-mode variants.
7. Run or import context-envelope observations for advertised context tapering.
8. Optionally run runtime lane orchestration and write ignored blackboard evidence.
9. Produce a model fit scorecard.
10. Generate an HTML view when a user needs sortable inspection.
11. Produce a deployment plan when the workflow needs concrete outfitting instructions.
12. Produce a fit report when the workflow needs a machine-level summary.
13. For AutoResearch, run `self-fit` before workload loops unless explicitly skipped for dry-run/testing.
14. Validate generated artifacts against schema and policy.
15. Gate training, downloads, service changes, and network exposure.

## First Model Target

The first trainable target is AppLens-Tailor on Qwen3.5-2B:

```text
machine evidence + local AI profile + benchmark facts + workload request
-> strict deployment-plan JSON
```

Broad narration, remediation, GRPO, and multi-agent LoRA roles come later, after the schema and benchmark loop is reliable.
