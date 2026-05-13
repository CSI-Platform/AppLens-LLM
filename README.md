# AppLens-LLM

AppLens-LLM is the local model outfitter workspace for AppLens.

It turns AppLens/AppLens-Tune machine evidence, workload goals, and benchmark results into local model fit scorecards, fit reports, deployment plans, and training examples.

## What This Repo Owns

- Strict schemas for model fit scorecards, fit reports, deployment plans, benchmark records, and training examples.
- A hardware topology contract that separates reported graphics memory, reserved/shared memory, and proven usable local inference capacity.
- Seed data for the first AppLens-Tailor training target.
- A validation CLI for JSON and JSONL artifacts.
- A capture ingestion CLI for AppLens `.md` reports and legacy `.txt` reports.
- An OpenAI-compatible benchmark runner for Jan, llama.cpp, and similar local endpoints.
- A `benchmark-suite-run` artifact that standardizes model, machine condition, runtime lane, official benchmark tasks, and local metric collection before comparable model tests run.
- An `applens-local-v1` capability eval for strict JSON, tool-call emulation, coding, hardware reasoning, safety, handoff planning, and thinking-mode comparison.
- A `context-envelope` artifact that tapers from advertised context windows down to proven local useful context.
- A blackboard-backed runtime orchestrator for comparing portable local model lanes.
- A `model-fit-scorecard` artifact that ranks local models by role, backend/device lane, score, blockers, and confidence.
- A `fit-report` artifact that summarizes the machine-level deployment posture.
- A scorecard-driven `deployment-plan` artifact that turns rankings into concrete local model assignments, runtime profiles, preflight actions, AppLens-Tune recommendations, and supervisor replacement gates.
- Roadmap and architecture docs for the AppLens-LLM extension.

## Current Product Target

The first product target is the local model fit meter:

```text
machine evidence + local model inventory + benchmark facts + workload goals
-> schema-valid model-fit-scorecard JSON
```

The scorecard answers which local LLMs fit this PC best, for which role, on which backend/device, with what confidence. It is the artifact AppLens should expose first.

## Training Target

The first trainable target is still AppLens-Tailor:

```text
machine evidence + local AI profile + benchmark facts + workload request
-> schema-valid AppLens-LLM deployment-plan JSON
```

The first base model target is Qwen3.5-2B. Training should wait until the baseline eval set proves that the base model misses structure, policy boundaries, or deployment-fit judgment.

`deployment-plan` is also the outfitting contract. The cloud/API planner-supervisor is the 100-point reference baseline until a local model passes replacement gates. Local models can still be assigned as primary workers, deep-review workers, long-context workers, or avoid-primary candidates.

## Quick Start

```powershell
uv sync --dev
uv run pytest
uv run applens-llm validate-jsonl --schema training-example data/examples.seed.jsonl
uv run applens-llm validate-jsonl --schema machine-profile data/machines.seed.jsonl
uv run applens-llm validate --schema model-fit-scorecard examples/asus-px13-model-fit-scorecard.example.json
uv run applens-llm validate --schema fit-report examples/asus-px13-fit-report.example.json
uv run applens-llm ingest-captures --source ../AppLens/raw --output data/raw/capture-records.jsonl
uv run applens-llm eval --examples data/examples.seed.jsonl --output out/eval-report.json
uv run applens-llm local-capability-eval --responses out/local-capability/responses.json --thinking-mode off --output out/local-capability/qwen35-4b-off.json
uv run applens-llm benchmark-suite-plan --suite-run-id qwen35-4b-vgm16-tiny-v1 --model-id qwen35-4b-q4km --display-name "Qwen3.5 4B Q4_K_M" --family qwen --parameter-size-b 4 --quantization Q4_K_M --model-format gguf --model-path sanitized/models/qwen35-4b-q4km.gguf --chat-template qwen --thinking-mode off --reasoning-mode off --condition-id asus-px13-vgm16-ram16 --condition-label "ASUS PX13 VGM 16GB / RAM 16GB" --os-family windows --ram-gb 32 --vgm-enabled --vgm-dedicated-mb 16384 --system-ram-available-gb 16 --accelerator-id amd-igpu-0 --backend vulkan --device-selector Vulkan0 --context-tokens 16384 --output out/benchmark-suites/qwen35-4b-vgm16/benchmark-suite-run.json
uv run applens-llm context-envelope --machine-profile out/pipeline/asus-px13-current-machine-profile.json --model-candidates out/pipeline/current-downloaded-model-candidates.json --output out/context/asus-px13-context-envelope.json
uv run applens-llm deployment-plan --scorecard out/scorecards/asus-px13-model-scorecard.json --workload-name "Oracle autoresearch" --output out/deployment-plans/asus-px13-outfit.json
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

## Benchmark Suite Runs

`benchmark-suite-run` is the standardized pre-test contract. It is the artifact to create before asking AppLens-LLM to compare a model under a condition such as `VGM 16GB / RAM 16GB`, `VGM off / RAM 32GB`, CUDA, Vulkan, ROCm/HIP, CPU, or another runtime lane.

The first two suites are:

- `tiny-v1` for models at or below 4.5B parameters: `IFEval`, official `ARC-Challenge Chat`, `HellaSwag`, `GSM8K`, `BFCL` prompt mode, `BigCodeBench-Hard` screening, `LongBench v2` screening, and `RULER` context taper.
- `small-v1` for models above 4.5B and at or below 30B parameters: Open LLM Leaderboard v2 family (`IFEval`, `BBH`, `MATH Lvl 5`, `GPQA`, `MuSR`, `MMLU-Pro`) plus `BFCL V4`, `BigCodeBench-Hard`, `LongBench v2`, `RULER`, and optional finalist-only `LiveBench`.

As of the May 13, 2026 review, AppLens-LLM treats `LongBench v2` as the primary long-context capability benchmark because it tests realistic long-context reasoning across documents, dialogue, code repositories, and structured data. `RULER` remains the repeatable diagnostic taper for effective context length. Local screening subsets must be labeled as `local_screening`; only full official settings should be treated as certification or leaderboard-comparable.

Each task declares the LM call shape it requires. Generation tasks can run through OpenAI-compatible chat endpoints, but log-likelihood tasks such as `HellaSwag`, `GPQA`, and `MMLU-Pro` require prompt token logprobs and must be marked unsupported if the runtime cannot provide them. For llama.cpp chat runners that leak `<think>` tags into `message.content`, run the lightweight normalizing proxy before `lm-eval`:

```powershell
uv run python -m applens_llm.llamacpp_lmeval --listen-port 18081 --upstream-base-url http://127.0.0.1:18080
```

Use the suite runner to create the result artifact. The runner can start the proxy itself, run supported official `lm-eval` generation tasks, parse `results_*.json`, record local wall-time metrics, and retain unsupported rows instead of substituting custom scores:

```powershell
uv run applens-llm benchmark-suite-run --plan out/benchmark-suites/qwen35-4b-vgm16/benchmark-suite-run.json --output out/benchmark-suites/qwen35-4b-vgm16/benchmark-suite-result.json --lm-eval "$env:LOCALAPPDATA\AppLens-LLM\BenchmarkTools\.venv\Scripts\lm_eval.exe" --use-llamacpp-proxy --local-screening-limit 20
```

```powershell
uv run applens-llm benchmark-suite-plan --suite-run-id qwen35-4b-vgm16-tiny-v1 --model-id qwen35-4b-q4km --display-name "Qwen3.5 4B Q4_K_M" --family qwen --parameter-size-b 4 --quantization Q4_K_M --model-format gguf --model-path sanitized/models/qwen35-4b-q4km.gguf --chat-template qwen --thinking-mode off --reasoning-mode off --condition-id asus-px13-vgm16-ram16 --condition-label "ASUS PX13 VGM 16GB / RAM 16GB" --os-family windows --ram-gb 32 --vgm-enabled --vgm-dedicated-mb 16384 --system-ram-available-gb 16 --accelerator-id amd-igpu-0 --backend vulkan --device-selector Vulkan0 --context-tokens 16384 --output out/benchmark-suites/qwen35-4b-vgm16/benchmark-suite-run.json
```

## Runtime Lanes

Runtime lanes describe local inference endpoints without hard-coding a specific laptop. A lane can point at CUDA, Vulkan, ROCm/HIP, Metal, DirectML, CPU, RPC, or another OpenAI-compatible endpoint. Validate the sanitized example, then keep a machine-specific copy under ignored `out/runtime/`. The controller records lane responses and failures in an append-only blackboard JSONL file:

```powershell
uv run applens-llm lanes-check --config examples/runtime-lanes.example.json
uv run applens-llm lane-start --config out/runtime/local-lanes.json --lane fast-nvidia
uv run applens-llm blackboard-init --experiment-id exp-local --title "Local lane smoke" --output out/blackboard/exp-local.jsonl
uv run applens-llm orchestrate-once --config out/runtime/local-lanes.json --lane fast-nvidia --experiment-id exp-local --task-id task-1 --prompt "Return a compact status." --blackboard out/blackboard/exp-local.jsonl --timeout-seconds 120
uv run applens-llm lane-stop --lane fast-nvidia
```

The blackboard is a controller/file-ledger contract, not a GPU memory contract. Lanes can communicate through appended JSONL tasks, responses, handoffs, and verdicts; they do not thereby gain pooled VRAM, CUDA/Vulkan/ROCm shared memory, or inter-GPU IPC. Fast-to-deep prompts include this rule so local models learn "yes through the ledger, no through pooled GPU memory."

For the common fast-to-deep handoff experiment, use one command:

```powershell
uv run applens-llm experiment-run --config out/runtime/local-lanes.json --fast-lane fast-nvidia --deep-lane deep-amd-vgm --experiment-id exp-local --prompt "Explain why AppLens-LLM treats advertised VRAM claims as unproven until benchmarked." --blackboard out/blackboard/exp-local.jsonl --summary out/blackboard/exp-local-summary.json --deep-max-tokens 320 --nvidia-driver-branch game_ready
```

For a bounded overnight handoff loop, use `overnight-loop`. It repeats fast-to-deep handoffs across inline prompts or a prompt file, writes every task/response/handoff/verdict to the blackboard, and stops on iteration, time, or failure limits:

```powershell
uv run applens-llm overnight-loop --config out/runtime/two-lane.local.json --fast-lane fast-nvidia --deep-lane deep-amd-vgm --experiment-id overnight-local --prompt-file examples/overnight-prompts.example.txt --blackboard out/blackboard/overnight-local.jsonl --summary out/blackboard/overnight-local-summary.json --max-iterations 8 --max-runtime-minutes 480 --sleep-seconds 30 --deep-max-tokens 320 --nvidia-driver-branch studio
```

Use `--skip-start` if both llama.cpp servers are already running. Use `--keep-running` if the command starts the lanes but should leave them up after the loop. By default, the loop stops on the first lane failure; add `--continue-on-failure` only when repeated failure records are useful. For compact fast-lane handoffs, keep `--fast-max-tokens` near 192-256 and give the deep lane more room when reviewing benchmark or scorecard evidence.

Compare two experiment summaries:

```powershell
uv run applens-llm experiment-compare --baseline out/blackboard/exp-game-ready-summary.json --candidate out/blackboard/exp-studio-summary.json --output out/blackboard/driver-comparison.json
```

Committed lane examples must stay sanitized. Put machine-specific binaries, local model paths, and raw run evidence under ignored `out/`.

Driver branch is runtime evidence. NVIDIA describes Game Ready drivers as game-focused and Studio drivers as reliability-focused for creative workflows; both can run games and creative apps. AppLens records the branch reported by the user plus the version from `nvidia-smi`, and a branch or version change means benchmark comparisons should be rerun instead of treated as equivalent.

## AutoResearch

AppLens-LLM has two chronological autoresearch modes:

1. `self-fit`: prove the local model/runtime setup for this machine.
2. `workload`: run a bounded workload loop such as Oracle using explicit allowlists.

Workload repos meet AppLens-LLM through `.applens/` files. Stable contracts are committed; run logs, artifacts, blackboards, proposed memory, and indexes stay local and ignored. V1 records probes, evals, logs, artifacts, and blackboard evidence so a supervisor can diagnose failures, but it does not auto-apply code, prompt, schema, command, or memory patches. It rejects commands that declare network access, but it does not provide an OS network sandbox.

Initialize a workload layout:

```powershell
uv run applens-llm autoresearch init --workload-root ../Oracle --workload-id oracle --display-name Oracle
```

Record a local self-fit result after the model/runtime path has been checked:

```powershell
uv run applens-llm autoresearch self-fit --workload-root ../Oracle --machine-fingerprint machine-a --runtime-fingerprint llama-cpp-cuda
```

Run one bounded workload step from an allowlisted manifest:

```powershell
uv run applens-llm autoresearch run --workload-root examples/oracle --manifest examples/oracle/.applens/runs/oracle-dry-run.example.json --skip-self-fit
```

Read committed probes and eval cases:

```powershell
uv run applens-llm autoresearch eval --workload-root examples/oracle
```

This writes `probe_result` and `eval_result` blackboard events for the contract-level checks it runs.

Promote proposed memory only after review:

```powershell
uv run applens-llm autoresearch promote-memory --workload-root ../Oracle --proposal ../Oracle/.applens/memory/proposed/example.md
```

## Model Fit Scorecards

`model-fit-scorecard` is the primary product artifact. It reads a machine profile, optional model candidate inventory, benchmark records, and experiment summaries, then ranks model choices out of 100:

```powershell
uv run applens-llm model-fit-scorecard --machine-profile data/machines.seed.jsonl --machine-id asus-laptop --model-candidates examples/asus-px13-model-candidates.example.json --workload-profile examples/oracle/.applens/workload.json --experiment-summary out/blackboard/exp-studio-summary.json --output out/scorecards/asus-px13-model-scorecard.json
```

Each ranking includes the recommended role, best lane/backend/device, score breakdown, observed or inferred confidence, reasons, blockers, and next benchmark. Generated scorecards belong under ignored `out/`; committed examples must stay sanitized.

The current sanitized ASUS PX13 example ranks the observed Jan/Qwen 4B fast CUDA lane ahead of the observed Qwen 27B AMD/VGM Vulkan deep lane for fast-chat use, while still recording the 27B model as the better capacity/deep-review lane. Unbenchmarked candidates are scored as inferred until direct evidence exists.

Scorecards now accept `--capability-record` inputs from `applens-local-v1`. This prevents two models from tying on hardware fit alone when one is materially better at strict JSON, tool selection, coding, or handoff planning:

```powershell
uv run applens-llm model-fit-scorecard --machine-profile data/machines.seed.jsonl --machine-id asus-laptop --model-candidates examples/asus-px13-model-candidates.example.json --benchmark-record out/benchmarks/qwen35-4b.json --capability-record out/local-capability/qwen35-4b-off.json --output out/scorecards/asus-px13-model-scorecard.json
```

Scorecards also accept `--context-envelope` inputs. This records advertised context separately from max tested and recommended context:

```powershell
uv run applens-llm model-fit-scorecard --machine-profile out/pipeline/asus-px13-current-machine-profile.json --model-candidates out/pipeline/current-downloaded-model-candidates.json --context-envelope out/context/asus-px13-context-envelope.json --output out/scorecards/asus-px13-model-scorecard.json
```

Generate a sortable local HTML view from the JSON scorecard and any experiment comparison files:

```powershell
uv run applens-llm model-fit-html --scorecard out/scorecards/asus-px13-model-scorecard.json --experiment-comparison out/blackboard/driver-comparison.json --output out/scorecards/asus-px13-model-scorecard.html
```

The HTML file is the human-readable dashboard. The JSON remains the source of truth.

## Deployment Plans

Use `deployment-plan` after a scorecard exists to produce the actual outfitting artifact:

```powershell
uv run applens-llm deployment-plan --scorecard out/scorecards/asus-px13-model-scorecard.json --plan-id asus-px13-oracle-outfit --workload-name "Oracle autoresearch" --workload-intent agent_runtime --output out/deployment-plans/asus-px13-oracle-outfit.json
```

The plan keeps the cloud/API model as planner-supervisor unless a local model clears supervisor replacement gates. It assigns local models to worker roles, emits llama.cpp runtime profiles, records proven context limits, lists preflight actions such as closing competing LLM apps, and tells AppLens-Tune which readiness changes are recommended or restart-gated.

## Local Capability Eval

`llama.cpp` `llama-bench` remains the hardware proof: backend, device placement, tokens/sec, OOM, fallback, and thermal behavior. `benchmark-suite-run` is the official benchmark plan. `applens-local-v1` remains a smoke/local-agent probe and must not be treated as a formal capability score.

The V1 suite scores:

- strict JSON and instruction following
- JSON tool-call emulation, including no-tool cases
- hardware reasoning about reported memory versus proven usable capacity
- benchmark interpretation for fast/deep model lanes
- a small Python coding task, with unit tests only when `--execute-code-checks` is explicitly passed
- safe action boundaries for AppLens-LLM and AppLens-Tune
- planner/executor handoff packets for larger and smaller local models

Thinking mode is recorded as part of the model variant. For Qwen-style models, test direct mode and reasoning mode separately when the runtime supports it:

```powershell
uv run applens-llm local-capability-eval --endpoint http://127.0.0.1:18080/v1 --model qwen35-4b-q4km --display-name "Qwen3.5 4B Q4_K_M" --family qwen --parameter-size-b 4 --quantization Q4_K_M --backend vulkan --device nvidia-dgpu-0 --thinking-mode off --output out/local-capability/qwen35-4b-off.json
uv run applens-llm local-capability-eval --endpoint http://127.0.0.1:18080/v1 --model qwen35-4b-q4km --display-name "Qwen3.5 4B Q4_K_M" --family qwen --parameter-size-b 4 --quantization Q4_K_M --backend vulkan --device nvidia-dgpu-0 --thinking-mode on --output out/local-capability/qwen35-4b-on.json
```

Some runtimes expose thinking controls differently. The record stores requested mode and whether thinking traces leaked into output; the scorecard treats `thinking=on` and `thinking=off` as separate evidence, not as the same model run.

## Context Envelope

Large advertised context windows are claims until this machine proves them. Artificial Analysis may list Qwen and Gemma families around 256k-262k context, but AppLens-LLM must not treat that as usable local deployment capacity by itself.

`context-envelope` tapers from the advertised context tier down through:

```text
262k -> 128k -> 64k -> 32k -> 16k -> 8k -> 4k
```

For each model, backend, and lane, AppLens-LLM should distinguish:

- advertised context
- max tested context
- max loadable context
- max stable context
- max useful context
- recommended context by workload

The intended observations are long-context benchmark rows with context size, backend, devices used, status, quality score, prompt/generation throughput, failure modes, and workload tags such as `long_context_retrieval`, `coding`, or `summarization`.

Example observation row:

```json
{"model_id":"gemma4-26b-a4b-q3km","context_tokens":16384,"backend":"vulkan","devices_used":["amd-igpu-0"],"status":"pass","quality_score_pct":89,"generation_tokens_per_second":22.0,"prompt_tokens_per_second":120.0,"failure_modes":["none"],"workloads":["coding","summarization"],"notes":"Sanitized local context observation."}
```

Generated context envelopes belong under ignored `out/`; committed examples must stay sanitized and must not include raw local paths.

## Fit Reports

`fit-report` is the supporting machine-level artifact. It reads a machine profile plus optional benchmark records, experiment summaries, and experiment comparisons, then writes one JSON recommendation:

```powershell
uv run applens-llm fit-report --machine-profile data/machines.seed.jsonl --machine-id asus-laptop --experiment-summary out/blackboard/exp-studio-summary.json --experiment-comparison out/blackboard/driver-comparison.json --output out/fit-reports/asus-px13-local-fit.json
```

The report summarizes local fit class, proven lanes, unsupported memory claims, runtime strategy, model guidance, decisions, and next benchmarks. AppLens should use it behind or beside the scorecard, not as the main local model ranking surface. Generated reports belong under ignored `out/`; committed examples must stay sanitized.

## Hardware Memory Rule

AppLens-LLM must not treat advertised memory, shared graphics memory, VGM reservations, or user-facing "pooled VRAM" claims as proven deployment capacity. Machine profiles and benchmark records now require `hardware_topology` evidence so recommendations can distinguish reported capacity from benchmark-proven backend/device behavior.

## Safety Boundary

V1 can start local model runtimes, run benchmarks, write manifests, and prepare sanitized datasets. It must not change services, startup entries, drivers, firewall rules, firmware, or user data.

See `ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/CAPTURE_GUIDE.md`, and `docs/DEVELOPER_GUIDE.md`.

For AMD Variable Graphics Memory verification, see `docs/VGM_TEST_GUIDE.md`.
