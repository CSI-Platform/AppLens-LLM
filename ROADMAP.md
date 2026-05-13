# Roadmap

## Product Thesis

AppLens-LLM should not center on "your model is ready." It should rank local model choices for a specific machine.

The user-facing product is a local model fit meter:

```text
Which local LLMs fit this PC best, for which role, on which backend/device, with what confidence?
```

AppLens captures the raw machine and local AI inventory. AppLens-Tune prepares the machine and reports readiness gaps. AppLens-LLM scores model fit from that evidence and tells the user which models are best suited for fast chat, deep review, coding, summarization, and experimentation.

## Milestone 0: Evidence Foundation

- Create standalone AppLens-LLM repo.
- Track deployment, benchmark, benchmark-suite, training, hardware topology, runtime lane, blackboard, fit report, and model scorecard schemas.
- Track accelerator memory claims separately from proven usable inference capacity.
- Add seed examples from the gaming PC, ASUS hybrid VGM case, and policy contrast cases.
- Add schema validation CLI and tests.
- Add OpenAI-compatible benchmark record generation.

## Milestone 1: Model Fit Scorecard

- Add `model-fit-scorecard` as the primary product artifact.
- Score candidate local models out of 100 by role.
- Include best backend/device lane, confidence, evidence, reasons, blockers, and next benchmark.
- Include advertised, tested, and recommended context windows once context-envelope evidence exists.
- Use concrete weighted scoring:
  - capacity fit
  - speed and latency
  - stability and failure history
  - role fit
  - quality-per-size
  - operational readiness
  - evidence confidence
  - local agent capability
- Generate the first ASUS ProArt PX13 scorecard from local model/runtime evidence.
- Keep generated local scorecards under ignored `out/`; commit only sanitized examples.
- Generate deployment plans from scorecards so rankings become concrete outfitting instructions: local worker assignments, runtime/context profiles, preflight actions, AppLens-Tune recommendations, and local supervisor replacement gates.

## Milestone 2: Readiness And Benchmark Loop

- Let AppLens-Tune feed readiness state into scorecards: VGM active, drivers present, ports available, runtime binaries found, thermal/power notes, and restart-required state.
- Use `benchmark-suite-run` before model comparisons so model, machine condition, runtime lane, official benchmark tasks, local metrics, and output artifacts are fixed before a run starts.
- Use `benchmark-suite-result` after execution so pass/fail/unsupported rows, command lines, artifacts, and runner limitations are preserved for scorecards.
- Add repeat benchmark runs for top-ranked candidate models.
- Standardize `tiny-v1` for models at or below 4.5B using IFEval, official ARC-Challenge Chat, HellaSwag, GSM8K, BFCL prompt mode, BigCodeBench-Hard screening, LongBench v2 screening, and RULER taper, with task-level runner requirements for generation vs. log-likelihood support.
- Standardize `small-v1` for models above 4.5B and at or below 30B using the Open LLM Leaderboard v2 family, BFCL V4, BigCodeBench-Hard, LongBench v2, RULER, and finalist-only LiveBench.
- Treat `applens-local-v1` as a smoke/local-agent probe beside official benchmark evidence, not as a formal model capability score.
- Add `context-envelope` tapering from advertised model context down to proven useful local context.
- Add long-context observations for loadability, stability, throughput, retrieval, summary, coding, and JSON-after-long-context behavior.
- Attach AMD Software and NVIDIA telemetry summaries where available.
- Improve confidence scores as repeated evidence accumulates.
- Keep driver branch/version as evidence, but do not over-weight it without repeated benchmark proof.

## Milestone 3: Deployment Plan And Fit Report

- Keep `deployment-plan` as the scorecard-driven outfitting artifact.
- Treat the cloud/API planner-supervisor as the 100-point reference baseline until local replacement gates pass.
- Assign local models as primary worker, deep-review worker, long-context worker, or avoid-primary based on evidence.
- Emit launch profiles and preflight/Tune guidance without changing drivers, services, downloads, or network exposure automatically.

- Keep `fit-report` as a machine-level decision summary.
- Embed or reference scorecard rankings from fit reports.
- Use fit reports for executive summaries and integration with AppLens.
- Keep runtime lanes, blackboard records, and experiment comparisons as evidence layers under the scorecard.

## Milestone 4: Universal AutoResearch

- Add workload-owned `.applens/` contracts for profiles, programs, commands, metrics, probes, evals, schemas, runs, blackboards, artifacts, logs, proposed memory, wiki, and indexes.
- Require `self-fit` evidence before workload loops unless explicitly skipped for dry-run/testing.
- Execute only allowlisted commands and record blocked actions as evidence.
- Keep blackboard events append-only and workload-neutral so Oracle and future apps can use the same protocol.
- Treat memory/wiki as curated knowledge: proposals can be written by a run, but promotion requires an explicit command.
- Keep V1 focused on evidence capture, probes, evals, and supervised review; do not auto-apply code, prompt, schema, command, or memory patches.

## Milestone 5: Product Integration

- AppLens presents the model fit meter as the main local AI view.
- AppLens-Tune presents readiness gaps and safe improvements.
- AppLens-LLM produces scorecards, fit reports, and benchmark recommendations.
- Unsafe actions remain gated or unsupported until rollback and approval infrastructure exists.

## Milestone 6: Training Only If Needed

- Run Qwen3.5-2B zero-shot and few-shot baselines against scorecard and deployment-plan evals.
- Train a small LoRA only if baseline models fail structure, policy boundaries, or model-fit judgment.
- Require schema validation and policy validation after every generation.
