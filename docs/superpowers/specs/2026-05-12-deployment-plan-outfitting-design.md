# Deployment Plan Outfitting Design

## Goal

`deployment-plan` converts a model fit scorecard into concrete local LLM outfitting guidance.

The scorecard answers which models fit. The deployment plan answers what to run, where to run it, what to close or verify first, and what evidence is still required before promoting a local model into a higher-responsibility role.

## Product Rule

The cloud/API planner-supervisor is the 100-point reference baseline in V1. Local models can be excellent workers without being ready to run the ship.

A local model can become a planner-supervisor candidate only after role drilldowns prove tool calling, coding, handoff planning, context behavior, and runtime stability. Until then, the plan keeps local models in worker roles and records the failed promotion gates.

## Inputs

- One schema-valid `model-fit-scorecard`.
- Optional workload name and workload intent.
- Existing scorecard evidence: rankings, backend/device lanes, fit score, blockers, capability categories, context profile, and confidence.

## Output

The upgraded `deployment-plan` remains compatible with the existing plan contract and adds an `outfitting` object:

- `supervisor_baseline`: cloud/API baseline with relative score 100.
- `local_supervisor_candidate`: best local candidate and missing evidence.
- `assignments`: primary local worker, deep-review worker, and avoid-primary rows.
- `runtime_profiles`: llama.cpp launch profile details per assigned model.
- `context_profiles`: advertised versus recommended local context per assigned model.
- `preflight_actions`: required checks before launch.
- `tune_recommendations`: actions AppLens-Tune should recommend or gate.
- `promotion_gates`: score thresholds blocking local supervisor promotion.
- `next_drilldowns`: exact follow-up benchmarks.

## Non-Goals

The deployment plan does not download models, modify drivers, start services, expose network endpoints, or change VGM settings. It records those as user-approved or unsupported gates.

## Testing

Tests cover direct plan construction, schema validation, file writing, and CLI generation from a scorecard.
