# AppLens-LLM

AppLens-LLM is the local model outfitter workspace for AppLens.

It turns AppLens/AppLens-Tune machine evidence, workload goals, and benchmark results into validated local LLM deployment plans and training examples.

## What This Repo Owns

- Strict schemas for deployment plans, benchmark records, and training examples.
- A hardware topology contract that separates reported graphics memory, reserved/shared memory, and proven usable local inference capacity.
- Seed data for the first AppLens-Tailor training target.
- A validation CLI for JSON and JSONL artifacts.
- A capture ingestion CLI for AppLens `.md` reports and legacy `.txt` reports.
- An OpenAI-compatible benchmark runner for Jan, llama.cpp, and similar local endpoints.
- A blackboard-backed runtime orchestrator for comparing portable local model lanes.
- A `fit-report` artifact that turns machine profiles and benchmark evidence into a deployment-fit recommendation.
- Roadmap and architecture docs for the AppLens-LLM extension.

## Current Training Target

The first target is AppLens-Tailor:

```text
machine evidence + local AI profile + benchmark facts + workload request
-> schema-valid AppLens-LLM deployment-plan JSON
```

The first base model target is Qwen3.5-2B. Training should wait until the baseline eval set proves that the base model misses structure, policy boundaries, or deployment-fit judgment.

## Quick Start

```powershell
uv sync --dev
uv run pytest
uv run applens-llm validate-jsonl --schema training-example data/examples.seed.jsonl
uv run applens-llm validate-jsonl --schema machine-profile data/machines.seed.jsonl
uv run applens-llm validate --schema fit-report examples/asus-px13-fit-report.example.json
uv run applens-llm ingest-captures --source ../AppLens/raw --output data/raw/capture-records.jsonl
uv run applens-llm eval --examples data/examples.seed.jsonl --output out/eval-report.json
uv run applens-llm vgm-snapshot --label before-vgm --output out/vgm/before-vgm.json
uv run applens-llm lanes-check --config examples/runtime-lanes.example.json
```

Jan default endpoint:

```powershell
uv run applens-llm bench --model qwen-local --output out/jan-benchmark.json
```

llama.cpp endpoint:

```powershell
uv run applens-llm bench --endpoint http://127.0.0.1:18080/v1 --engine llama.cpp --backend cuda --model qwen2.5:7b --output out/llamacpp-benchmark.json
```

Generated benchmark output is ignored under `out/`.

## Runtime Lanes

Runtime lanes describe local inference endpoints without hard-coding a specific laptop. A lane can point at CUDA, Vulkan, ROCm/HIP, Metal, DirectML, CPU, RPC, or another OpenAI-compatible endpoint. Validate the sanitized example, then keep a machine-specific copy under ignored `out/runtime/`. The controller records lane responses and failures in an append-only blackboard JSONL file:

```powershell
uv run applens-llm lanes-check --config examples/runtime-lanes.example.json
uv run applens-llm lane-start --config out/runtime/local-lanes.json --lane fast-nvidia
uv run applens-llm blackboard-init --experiment-id exp-local --title "Local lane smoke" --output out/blackboard/exp-local.jsonl
uv run applens-llm orchestrate-once --config out/runtime/local-lanes.json --lane fast-nvidia --experiment-id exp-local --task-id task-1 --prompt "Return a compact status." --blackboard out/blackboard/exp-local.jsonl --timeout-seconds 120
uv run applens-llm lane-stop --lane fast-nvidia
```

For the common fast-to-deep handoff experiment, use one command:

```powershell
uv run applens-llm experiment-run --config out/runtime/local-lanes.json --fast-lane fast-nvidia --deep-lane deep-amd-vgm --experiment-id exp-local --prompt "Explain why AppLens-LLM treats advertised VRAM claims as unproven until benchmarked." --blackboard out/blackboard/exp-local.jsonl --summary out/blackboard/exp-local-summary.json --deep-max-tokens 320 --nvidia-driver-branch game_ready
```

Compare two experiment summaries:

```powershell
uv run applens-llm experiment-compare --baseline out/blackboard/exp-game-ready-summary.json --candidate out/blackboard/exp-studio-summary.json --output out/blackboard/driver-comparison.json
```

Committed lane examples must stay sanitized. Put machine-specific binaries, local model paths, and raw run evidence under ignored `out/`.

Driver branch is runtime evidence. NVIDIA describes Game Ready drivers as game-focused and Studio drivers as reliability-focused for creative workflows; both can run games and creative apps. AppLens records the branch reported by the user plus the version from `nvidia-smi`, and a branch or version change means benchmark comparisons should be rerun instead of treated as equivalent.

## Fit Reports

`fit-report` is the product-facing artifact. It reads a machine profile plus optional benchmark records, experiment summaries, and experiment comparisons, then writes one JSON recommendation:

```powershell
uv run applens-llm fit-report --machine-profile data/machines.seed.jsonl --machine-id asus-laptop --experiment-summary out/blackboard/exp-studio-summary.json --experiment-comparison out/blackboard/driver-comparison.json --output out/fit-reports/asus-px13-local-fit.json
```

The report summarizes local fit class, proven lanes, unsupported memory claims, runtime strategy, model guidance, decisions, and next benchmarks. Generated reports belong under ignored `out/`; committed examples must stay sanitized.

## Hardware Memory Rule

AppLens-LLM must not treat advertised memory, shared graphics memory, VGM reservations, or user-facing "pooled VRAM" claims as proven deployment capacity. Machine profiles and benchmark records now require `hardware_topology` evidence so recommendations can distinguish reported capacity from benchmark-proven backend/device behavior.

## Safety Boundary

V1 can start local model runtimes, run benchmarks, write manifests, and prepare sanitized datasets. It must not change services, startup entries, drivers, firewall rules, firmware, or user data.

See `ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/CAPTURE_GUIDE.md`, and `docs/DEVELOPER_GUIDE.md`.

For AMD Variable Graphics Memory verification, see `docs/VGM_TEST_GUIDE.md`.
