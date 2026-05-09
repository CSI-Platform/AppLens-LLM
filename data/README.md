# AppLens-LLM Data

This directory is for sanitized training and evaluation examples.

Tracked data should be small, reviewed, and free of private user paths, tokens, application content, and raw machine dumps. Put local-only material under `data/private/` or `data/raw/`; both are ignored by git.

The first target format is JSONL where each line matches `schemas/training-example.schema.json`.

Machine capture planning lives in `machines.seed.jsonl`, where each line matches `schemas/machine-profile.schema.json`.

Machine rows must include `hardware_topology` so training data can distinguish reported graphics capacity, memory claims, and benchmark-proven usable inference capacity.
