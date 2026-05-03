# Architecture

AppLens-LLM is the model-fitting workspace beside AppLens.

AppLens measures the machine. AppLens-Tune describes workstation readiness. AppLens-LLM converts those measurements, workload goals, and benchmark records into a deployment plan that a runtime can execute or reject.

## Boundaries

- AppLens remains the inventory and measurement source.
- AppLens-Tune remains the workstation readiness and tuning advisor.
- AppLens-LLM owns local model fit: runtime selection, model artifact selection, benchmark evidence, training data, and evals.

V1 may start a user-owned local model runtime and write benchmark logs. It must not change services, startup entries, drivers, firewall rules, firmware, or user data.

## Contracts

- `schemas/deployment-plan.schema.json`: strict output target for the first AppLens-Tailor model.
- `schemas/benchmark-record.schema.json`: benchmark proof attached to a recommendation.
- `schemas/training-example.schema.json`: supervised training and eval row shape.

Every training example keeps both a chat `messages` view and a structured `expected_output`. The chat view is used for SFT. The structured output is used for schema and policy validation.

## Runtime Path

1. Read a sanitized AppLens/AppLens-Tune snapshot.
2. Read workload intent.
3. Run or import benchmark records.
4. Produce a deployment plan.
5. Validate the plan against schema and policy.
6. Gate training, downloads, service changes, and network exposure.

## First Model Target

The first trainable target is AppLens-Tailor on Qwen3.5-2B:

```text
machine evidence + local AI profile + benchmark facts + workload request
-> strict deployment-plan JSON
```

Broad narration, remediation, GRPO, and multi-agent LoRA roles come later, after the schema and benchmark loop is reliable.
