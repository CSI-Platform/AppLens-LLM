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
- Track deployment, benchmark, training, hardware topology, runtime lane, blackboard, fit report, and model scorecard schemas.
- Track accelerator memory claims separately from proven usable inference capacity.
- Add seed examples from the gaming PC, ASUS hybrid VGM case, and policy contrast cases.
- Add schema validation CLI and tests.
- Add OpenAI-compatible benchmark record generation.

## Milestone 1: Model Fit Scorecard

- Add `model-fit-scorecard` as the primary product artifact.
- Score candidate local models out of 100 by role.
- Include best backend/device lane, confidence, evidence, reasons, blockers, and next benchmark.
- Use concrete weighted scoring:
  - capacity fit
  - speed and latency
  - stability and failure history
  - role fit
  - quality-per-size
  - operational readiness
  - evidence confidence
- Generate the first ASUS ProArt PX13 scorecard from local model/runtime evidence.
- Keep generated local scorecards under ignored `out/`; commit only sanitized examples.

## Milestone 2: Readiness And Benchmark Loop

- Let AppLens-Tune feed readiness state into scorecards: VGM active, drivers present, ports available, runtime binaries found, thermal/power notes, and restart-required state.
- Add repeat benchmark runs for top-ranked candidate models.
- Attach AMD Software and NVIDIA telemetry summaries where available.
- Improve confidence scores as repeated evidence accumulates.
- Keep driver branch/version as evidence, but do not over-weight it without repeated benchmark proof.

## Milestone 3: Fit Report As Supporting Artifact

- Keep `fit-report` as a machine-level decision summary.
- Embed or reference scorecard rankings from fit reports.
- Use fit reports for executive summaries and integration with AppLens.
- Keep runtime lanes, blackboard records, and experiment comparisons as evidence layers under the scorecard.

## Milestone 4: Product Integration

- AppLens presents the model fit meter as the main local AI view.
- AppLens-Tune presents readiness gaps and safe improvements.
- AppLens-LLM produces scorecards, fit reports, and benchmark recommendations.
- Unsafe actions remain gated or unsupported until rollback and approval infrastructure exists.

## Milestone 5: Training Only If Needed

- Run Qwen3.5-2B zero-shot and few-shot baselines against scorecard and deployment-plan evals.
- Train a small LoRA only if baseline models fail structure, policy boundaries, or model-fit judgment.
- Require schema validation and policy validation after every generation.
