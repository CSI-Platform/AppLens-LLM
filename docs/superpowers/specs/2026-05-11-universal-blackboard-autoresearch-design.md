# Universal Blackboard AutoResearch Design

Date: 2026-05-11

## Decision

AppLens-LLM will own the universal blackboard, memory/wiki, self-fit, and autoresearch engine inside this repo for V1. Workload apps such as Oracle will meet AppLens-LLM through stable files, schemas, command allowlists, artifacts, and blackboard events.

The product boundary is:

- AppLens scans the machine and exports local evidence.
- AppLens-Tune prepares the machine through explicit, approved actions.
- AppLens-LLM self-fits local model/runtime choices, then runs bounded autoresearch.
- Workload apps own their domain logic and expose approved research commands.

Oracle is the first workload adapter. It must prove the protocol, not become the protocol.

## Core Model

AppLens-LLM supports two chronological autoresearch modes:

1. `self-fit`: AppLens-LLM researches the best local model, runtime, backend, lane, token budget, prompt shape, memory strategy, and stability settings for the current machine.
2. `workload`: AppLens-LLM applies the fitted setup to a workload such as Oracle.

Mode 1 is the default gate before Mode 2. A workload run should first check whether a fresh self-fit result exists for the same machine profile, runtime lanes, model files, driver/backend versions, and stability status. For V1, a self-fit result is fresh for 24 hours before overnight runs and 7 days for planning runs. Users may explicitly bypass the gate with `--skip-self-fit`.

## Supervisor And Candidate Roles

V1 is orchestrator-first and must not assume every user has two local models. A run can map roles onto Codex, Claude Code, API models, local lanes, or manual review.

Core roles:

- `supervisor`: highest-quality available reasoning layer. It proposes next steps, reviews safety, validates structured decisions, and curates memory proposals.
- `candidate`: local model being tested, benchmarked, outfitted, or used for cheap bounded loops.
- `workload_executor`: approved workload command runner, normally deterministic code rather than an LLM.
- `critic`: optional reviewer. It may be the supervisor, another local lane, or manual review.
- `memory_curator`: normally the supervisor in V1.

The run manifest should describe roles, not hard-code `fast_lane` and `deep_lane` as required concepts. On the ASUS PX13, those roles can map to CUDA and AMD/VGM Vulkan lanes. On a normal PC, they may map to one local model plus an API supervisor.

## Supervisor Provider Interface

The supervisor should be a generic provider interface, not a hard-coded vendor integration.

Supported provider types in the V1 contract:

- `codex-current-session`
- `claude-code`
- `openai-api`
- `local-lane`
- `manual-review`

The engine asks the supervisor for a structured next step, validates the response against schema, and executes only workload-profile allowlisted commands. V1 may implement only practical paths first, but the schema must not assume one provider.

## Universal Blackboard Protocol

There will be one blackboard protocol for all workloads. Workload apps adapt to the blackboard; the blackboard does not fork for each app.

Blackboard records are append-only JSONL events. They record what happened, when, by whom, under which workload and run, and which artifacts were produced or consumed.

Required shared fields:

- `event_id`
- `schema_version`
- `created_at`
- `workload_id`
- `run_id`
- `event_type`
- `actor`
- `role`
- `payload`
- `artifact_refs`
- `privacy`

Core event types:

- `run_started`
- `self_fit_checked`
- `task_created`
- `supervisor_decision`
- `candidate_response`
- `command_requested`
- `command_allowed`
- `command_blocked`
- `command_result`
- `artifact_created`
- `critique`
- `memory_proposed`
- `memory_promoted`
- `probe_result`
- `eval_result`
- `verdict`
- `failure`
- `run_stopped`

Domain details stay in `payload` and external artifacts. Oracle-specific strategy and backtest fields must not require Oracle-specific blackboard code.

The blackboard is not memory. It is the chronological evidence ledger.

## Memory And File Layout

Each workload repo uses a local `.applens/` folder by default.

For Oracle:

```text
Oracle/.applens/
  workload.json
  program.md
  commands.json
  metrics.json
  probes.json
  evals/
  schemas/
  runs/
  blackboard/
  artifacts/
  logs/
  memory/
    proposed/
    wiki/
  indexes/
```

Committed:

- `.applens/workload.json`
- `.applens/program.md`
- `.applens/commands.json`
- `.applens/metrics.json`
- `.applens/probes.json`
- `.applens/evals/`
- `.applens/schemas/`
- `.applens/memory/wiki/` only after review/promotion

Ignored:

- `.applens/runs/`
- `.applens/blackboard/`
- `.applens/artifacts/`
- `.applens/logs/`
- `.applens/memory/proposed/`
- `.applens/indexes/`

Meaning:

- `blackboard/` is append-only run history.
- `artifacts/` are produced evidence.
- `logs/` contains command/runtime logs that are useful for diagnosis but not commit-safe by default.
- `memory/proposed/` contains candidate lessons from a run.
- `memory/wiki/` contains reviewed durable knowledge.
- `indexes/` contains rebuildable search/index helpers.

Memory promotion is not automatic in V1. Runs write proposed memory updates. Only an explicit command or user approval promotes them into the wiki.

## Small-Model Legibility

AutoResearch files must be simple enough for a 2B-4B local model to follow overnight.

Design rules:

- short Markdown files
- stable headings
- explicit allowed and blocked actions
- one goal per run
- one primary metric per loop when possible
- one command pattern per strategy
- predictable JSON outputs
- no hidden behavior in prose
- no sprawling agent manifesto prompts

This follows the useful part of the AutoResearch pattern: compact instruction files, a bounded loop, fixed metrics, and a small surface area for the model to modify or reason about.

## Workload Profile

`workload.json` describes how an app participates in AppLens-LLM autoresearch.

Required concepts:

- `workload_id`
- `display_name`
- `workload_type`
- `program_file`
- `commands_file`
- `metrics_file`
- `probes_file`
- `evals_dir`
- `schemas_dir`
- `allowed_actions`
- `blocked_actions`
- `required_artifacts`
- `model_role_needs`
- `safety_gates`

Oracle V1 workload profile:

- workload ID: `oracle`
- type: `financial_research`
- allowed actions: read local data, write strategy artifacts, run approved backtests, write reports
- blocked actions: live trades, broker orders, credential access, system changes, model downloads without approval
- required artifacts: strategy JSON, backtest result JSON, critique markdown, run summary markdown

## Command Allowlist

`autoresearch run` may execute workload commands only through explicit allowlists in the workload profile.

Command records include:

- `id`
- `description`
- `command_template`
- `working_directory`
- `allowed_parameters`
- `required_inputs`
- `expected_outputs`
- `timeout_seconds`
- `network_policy`
- `risk_level`

The model may choose an allowlisted command and fill safe parameters. It may not invent a new executable action. Blocked command attempts are recorded as blackboard events.

## Probes And Evals

V1 should include a small eval/probe layer before any self-improvement behavior. This is the main design adjustment from the auto-improving software article: the loop only becomes useful when actions, data, logs, and pass/fail checks live close enough for an agent to inspect.

`probes.json` contains fast checks derived from the workload contract and `program.md`.

Probe records include:

- `id`
- `description`
- `input`
- `expected_command_id`
- `expected_event_types`
- `expected_artifact_types`
- `rubric`
- `max_runtime_seconds`

`.applens/evals/cases.json` contains regression cases for in-distribution behavior.

Eval case records include:

- `id`
- `input`
- `rubric`
- `expected_command_id`
- `expected_blocked_actions`
- `expected_artifacts`
- `expected_blackboard_events`

V1 may run probes/evals and write `probe_result` or `eval_result` blackboard events. It must not automatically edit workload code, prompts, schemas, or command allowlists based on failures. Failed probes produce evidence and proposed next steps.

## Auto-Improvement Boundary

V1 records enough local files, logs, probes, evals, blackboard events, and artifacts for a supervisor to diagnose failures. V1 does not auto-apply self-improvement patches. If a supervisor finds drift or failed probes, it writes evidence and proposed next steps; code, prompt, schema, command, and memory changes still require explicit approval.

## Program File

Each workload has a small `program.md`.

Oracle V1 shape:

```md
# Oracle AutoResearch Program

Goal:
Find research hypotheses that survive backtesting.

Allowed:
- propose one hypothesis
- write one strategy artifact
- run one approved backtest command
- compare result to baseline
- keep or reject the result

Blocked:
- live trades
- broker orders
- credential access
- changing system settings

Metric:
risk_adjusted_return_after_costs

Loop:
1. Read last result.
2. Propose one small change.
3. Run approved command.
4. Compare metric.
5. Keep if better, reject if worse.
6. Write blackboard event.
```

## Artifact Contract

Artifacts are files produced by the workload or AppLens-LLM and referenced from blackboard events.

Shared artifact envelope:

- `artifact_id`
- `workload_id`
- `run_id`
- `artifact_type`
- `path`
- `schema`
- `created_by`
- `summary`
- `hash`
- `privacy`

Oracle artifacts:

- `strategy_candidate`
- `backtest_result`
- `risk_review`
- `research_report`
- `memory_proposal`

## Model Fit For Workloads

Model fit scorecards must evolve from machine-only fit to machine plus workload plus role fit.

For Oracle, roles include:

- `supervisor`
- `candidate`
- `hypothesis_planner`
- `backtest_coder`
- `evidence_reviewer`
- `market_summarizer`
- `memory_curator`

Scorecard inputs:

- machine profile
- runtime lanes
- local model inventory
- self-fit results
- workload profile
- workload run artifacts
- local benchmarks
- optional external leaderboard metadata

External leaderboards are evidence, not deployment truth. A model is not an observed Oracle fit until it runs local Oracle evals or workload loops on the target machine.

## CLI Shape

Expected commands:

```powershell
uv run applens-llm autoresearch self-fit `
  --workload-root C:\path\to\Oracle `
  --runtime-lanes out\runtime\two-lane.local.json `
  --output .applens\runs\self-fit-latest.json
```

```powershell
uv run applens-llm autoresearch run `
  --workload-root C:\path\to\Oracle `
  --manifest .applens\runs\oracle-run-001.json `
  --blackboard .applens\blackboard\oracle-run-001.jsonl
```

```powershell
uv run applens-llm autoresearch promote-memory `
  --workload-root C:\path\to\Oracle `
  --proposal .applens\memory\proposed\oracle-run-001-lessons.md
```

## Safety Boundary

V1 must not:

- place live trades
- call broker order endpoints
- read secrets or credentials
- change system settings
- install packages without approval
- download models without approval
- modify drivers, services, firewall, firmware, or startup entries
- promote memory automatically

Allowed by explicit profile contract:

- run local read-only workload commands
- write local artifacts
- write blackboard events
- write proposed memory updates
- run local model benchmarks

## Completion Criteria

This design is complete when AppLens-LLM can:

1. Validate a workload profile.
2. Validate an autoresearch run manifest.
3. Initialize a workload `.applens/` structure.
4. Write universal blackboard records with workload/run/artifact references.
5. Run a fake workload command from an allowlist and block an unlisted command.
6. Validate and run simple workload probes/evals, recording pass/fail events.
7. Produce proposed memory updates without promoting them.
8. Run an Oracle example dry run using safe fake or read-only artifacts.
9. Produce workload-aware model-fit output that distinguishes supervisor, candidate, and workload roles.

## Non-Goals

V1 does not implement live trading, broker integrations, automatic model downloads, automatic package installation, automatic memory promotion, automatic self-editing, hill-climb patching, or frontend integration.

## Reference Inputs

- Karpathy AutoResearch: https://github.com/karpathy/autoresearch
- Windows RTX fork: https://github.com/jsegov/autoresearch-win-rtx
- Awesome AutoResearch index: https://github.com/WecoAI/awesome-autoresearch
- LLM Wiki guide: https://www.cognitionus.com/blog/llm-wiki-guide
- Karpathy Wiki explainer: https://karpathy-wiki.lol/en
