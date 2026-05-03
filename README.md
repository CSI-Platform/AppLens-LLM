# AppLens-LLM

AppLens-LLM is the local model outfitter workspace for AppLens.

It turns AppLens/AppLens-Tune machine evidence, workload goals, and benchmark results into validated local LLM deployment plans and training examples.

## What This Repo Owns

- Strict schemas for deployment plans, benchmark records, and training examples.
- Seed data for the first AppLens-Tailor training target.
- A validation CLI for JSON and JSONL artifacts.
- An OpenAI-compatible benchmark runner for Jan, llama.cpp, and similar local endpoints.
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
```

Jan default endpoint:

```powershell
uv run applens-llm bench --model qwen-local --output out/jan-benchmark.json
```

llama.cpp endpoint:

```powershell
uv run applens-llm bench --endpoint http://127.0.0.1:18080/v1 --backend llama.cpp --model qwen2.5:7b --output out/llamacpp-benchmark.json
```

Generated benchmark output is ignored under `out/`.

## Safety Boundary

V1 can start local model runtimes, run benchmarks, write manifests, and prepare sanitized datasets. It must not change services, startup entries, drivers, firewall rules, firmware, or user data.

See `ROADMAP.md`, `docs/ARCHITECTURE.md`, and `docs/DEVELOPER_GUIDE.md`.
