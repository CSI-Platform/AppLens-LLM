# Developer Guide

## Setup

```powershell
uv sync --dev
uv run pytest
```

## Validate Seed Data

```powershell
uv run applens-llm validate --schema deployment-plan examples/gaming-pc-deployment-plan.json
uv run applens-llm validate --schema benchmark-record examples/gaming-pc-benchmark-record.json
uv run applens-llm validate --schema runtime-lanes examples/runtime-lanes.example.json
uv run applens-llm validate --schema model-fit-scorecard examples/asus-px13-model-fit-scorecard.example.json
uv run applens-llm validate --schema fit-report examples/asus-px13-fit-report.example.json
uv run applens-llm validate-jsonl --schema training-example data/examples.seed.jsonl
uv run applens-llm validate-jsonl --schema machine-profile data/machines.seed.jsonl
```

## Run The Eval Harness

```powershell
uv run applens-llm eval --examples data/examples.seed.jsonl --output out/eval-report.json
```

The first eval checks:

- assistant output parses as JSON
- assistant output validates as `deployment-plan`
- V1 policy boundaries are respected
- runtime fields match expected labels
- core expected fields match expected labels

Machine profiles require a broad `model` plus exact `sku`. The SKU is the key for separating hardware variants that share a marketing name.

Machine profiles also require `hardware_topology`. Use it for deployment decisions instead of summing `platform.vram_mb`, shared memory, VGM reservations, or marketing claims.

## Ingest AppLens Captures

Keep raw capture manifests under ignored `data/raw/`:

```powershell
uv run applens-llm ingest-captures --source ../AppLens/raw --output data/raw/capture-records.jsonl
uv run applens-llm validate-jsonl --schema capture-record data/raw/capture-records.jsonl
```

The ingester accepts the current `.md` reports and legacy `.txt` reports. It creates a review manifest only; it does not promote raw reports into tracked training data.

## Run A Local Endpoint Benchmark

Jan defaults to `http://127.0.0.1:1337/v1`:

```powershell
uv run applens-llm bench --model qwen-local --output out/jan-benchmark.json
```

llama.cpp through a local tunnel or direct localhost endpoint:

```powershell
uv run applens-llm bench --endpoint http://127.0.0.1:18080/v1 --engine llama.cpp --backend cuda --model qwen2.5:7b --model-path models/qwen2.5-7b-ollama.gguf --quantization "Q4_K_M GGUF" --output out/llamacpp-benchmark.json
```

Generated benchmark output is ignored under `out/`.

For real capacity proof, the endpoint benchmark record may still need manual enrichment from runtime logs or hardware telemetry. Record the actual backend, devices used, mixed-device offload result, per-device memory use, CPU spill, OOM/fallback/crash status, and thermal notes before promoting a benchmark into tracked examples.

## Run A Runtime Lane Experiment

Runtime lane configs are portable. They can describe CUDA, Vulkan, ROCm/HIP, Metal, DirectML, CPU, RPC, or any OpenAI-compatible local endpoint. Validate the sanitized example:

```powershell
uv run applens-llm lanes-check --config examples/runtime-lanes.example.json
```

Create a machine-specific config under ignored `out/runtime/`, then create an ignored blackboard ledger and run one lane:

```powershell
uv run applens-llm lane-start --config out/runtime/local-lanes.json --lane fast-nvidia
uv run applens-llm blackboard-init --experiment-id exp-local --title "Local lane smoke" --output out/blackboard/exp-local.jsonl
uv run applens-llm blackboard-task --experiment-id exp-local --task-id task-1 --prompt "Return a compact local runtime status." --output out/blackboard/exp-local.jsonl
uv run applens-llm orchestrate-once --config out/runtime/local-lanes.json --lane fast-nvidia --experiment-id exp-local --task-id task-1 --prompt "Return a compact local runtime status." --blackboard out/blackboard/exp-local.jsonl --timeout-seconds 120
uv run applens-llm lane-stop --lane fast-nvidia
```

`lane-start` writes process state to `out/runtime/lane-processes.json` and logs to `out/logs/` by default. Use `--dry-run` to inspect the generated `llama-server` command before starting a model. Do not commit blackboard output, process state, or logs; they may include model responses, runtime errors, and local endpoint details.

For a complete fast-to-deep run, use `experiment-run`:

```powershell
uv run applens-llm experiment-run --config out/runtime/local-lanes.json --fast-lane fast-nvidia --deep-lane deep-amd-vgm --experiment-id exp-local --prompt "Explain why AppLens-LLM treats advertised VRAM claims as unproven until benchmarked." --blackboard out/blackboard/exp-local.jsonl --summary out/blackboard/exp-local-summary.json --deep-max-tokens 320 --nvidia-driver-branch game_ready
```

By default this starts both lanes, waits for their endpoints, routes the task through the fast lane, sends the fast answer to the deep lane for review, writes a blackboard JSONL ledger, writes a summary JSON, and stops started lanes. Use `--skip-start` for already-running endpoints and `--keep-running` when you want to reuse the servers.

For repeated handoffs, use `overnight-loop`:

```powershell
uv run applens-llm overnight-loop --config out/runtime/two-lane.local.json --fast-lane fast-nvidia --deep-lane deep-amd-vgm --experiment-id overnight-local --prompt-file examples/overnight-prompts.example.txt --blackboard out/blackboard/overnight-local.jsonl --summary out/blackboard/overnight-local-summary.json --max-iterations 8 --max-runtime-minutes 480 --sleep-seconds 30 --deep-max-tokens 320 --nvidia-driver-branch studio
```

The loop appends a `task`, fast `model_response` or `failure`, `handoff`, deep `model_response` or `failure`, and `verdict` for each attempted iteration. It stops on the first fast or deep failure unless `--continue-on-failure` is set. Keep the prompt file sanitized if committed; real overnight output belongs under ignored `out/blackboard/`.

`--nvidia-driver-branch` records the driver family that the user sees in NVIDIA App. AppLens collects the installed NVIDIA driver version with `nvidia-smi`; it does not infer Game Ready versus Studio from the version alone. Treat any driver version or branch change as a benchmark invalidator and rerun the same experiment before comparing results.

NVIDIA's public driver guidance says Game Ready prioritizes day-of-launch game support, while Studio prioritizes reliability for creative workflows. AppLens should record this as evidence, not change GPU drivers automatically.

Compare two summaries with `experiment-compare`:

```powershell
uv run applens-llm experiment-compare --baseline out/blackboard/exp-game-ready-summary.json --candidate out/blackboard/exp-studio-summary.json --output out/blackboard/driver-comparison.json
```

The comparison reports driver branch/version, lane equality, per-lane latency deltas, token-count deltas, latency per token, and warnings such as `token_counts_differ`. Treat a single comparison as directional evidence unless repeated runs show the same pattern.

## Write A Model Fit Scorecard

Use `model-fit-scorecard` as the first user-facing local AI artifact. It ranks candidate models by score, role, best backend/device lane, confidence, reasons, blockers, and next benchmark:

```powershell
uv run applens-llm model-fit-scorecard --machine-profile data/machines.seed.jsonl --machine-id asus-laptop --model-candidates examples/asus-px13-model-candidates.example.json --experiment-summary out/blackboard/exp-studio-summary.json --output out/scorecards/asus-px13-model-scorecard.json
```

Inputs are additive. A candidate inventory gives the scorecard models to rank, experiment summaries provide lane-level observed evidence, and benchmark records provide direct backend/device proof. If a candidate has no direct evidence, the scorecard can infer a provisional lane from model size and accelerator capacity, but it must mark confidence as `inferred` and include `no_observed_benchmark`.

For the current ASUS PX13 evidence, the scorecard ranks the observed fast CUDA 4B model as an excellent fast-chat fit, the observed AMD/VGM Vulkan 27B model as a good deep-review/capacity fit, and unbenchmarked candidates as usable or experimental until direct evidence exists.

Create the sortable HTML view after the JSON scorecard exists:

```powershell
uv run applens-llm model-fit-html --scorecard out/scorecards/asus-px13-model-scorecard.json --experiment-comparison out/blackboard/driver-comparison.json --output out/scorecards/asus-px13-model-scorecard.html
```

The HTML report includes ranking and comparison tables with client-side sorting and filtering. Keep generated HTML under ignored `out/`; commit sanitized JSON examples instead.

## Write A Fit Report

Use `fit-report` after a machine profile has at least one benchmark or runtime experiment. It is the supporting machine-level decision summary:

```powershell
uv run applens-llm fit-report --machine-profile data/machines.seed.jsonl --machine-id asus-laptop --experiment-summary out/blackboard/exp-studio-summary.json --experiment-comparison out/blackboard/driver-comparison.json --output out/fit-reports/asus-px13-local-fit.json
```

The report includes the machine summary, local fit class, capacity assessment, proven lanes, unsupported memory claims, runtime strategy, model guidance, decisions, and next benchmarks. It intentionally does not include raw launch commands or local model paths. Store generated reports under ignored `out/fit-reports/` unless they have been sanitized into `examples/`.

For the current ASUS PX13 evidence, the fit report says this is a `hybrid_local_ai_worker` with a `two_lane_local` strategy: NVIDIA/CUDA for fast small-model work, AMD/VGM Vulkan for slower capacity work, and no verified RTX-plus-VGM pooled memory claim.
