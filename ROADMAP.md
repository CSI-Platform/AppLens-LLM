# Roadmap

## Milestone 0: Foundation

- Create standalone AppLens-LLM repo.
- Track deployment, benchmark, and training schemas.
- Add seed examples from the gaming PC and policy contrast cases.
- Add schema validation CLI and tests.
- Add OpenAI-compatible benchmark record generation.

## Milestone 1: Eval Harness

- Convert the first 5 real AppLens reports into sanitized eval candidates.
- Build a baseline eval set of 50 to 100 examples.
- Score schema validity, policy agreement, runtime fit, and benchmark grounding.
- Run Qwen3.5-2B zero-shot and few-shot baselines before training.
- Track every eval report under ignored `out/` until promoted.
- Capture AppLens Markdown reports into ignored review manifests before promotion.
- Backlog: add JSON export for schema-backed dataset generation.

## Milestone 2: AppLens-Tailor LoRA

- Train a small LoRA only if baseline misses structure or deployment judgment.
- Keep outputs strict JSON.
- Require schema validation and policy validation after every generation.
- Keep training manifests explicit: data path, base model, adapter output, runtime limit, and stop conditions.

## Milestone 3: Product Integration

- Import sanitized AppLens/AppLens-Tune snapshots.
- Attach AppLens-Bench records to recommendations.
- Surface deployment plans back through AppLens as a second extension.
- Keep unsafe actions gated or unsupported until rollback and approval infrastructure exists.
