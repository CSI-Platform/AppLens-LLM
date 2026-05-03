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
uv run applens-llm validate-jsonl --schema training-example data/examples.seed.jsonl
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

## Run A Local Endpoint Benchmark

Jan defaults to `http://127.0.0.1:1337/v1`:

```powershell
uv run applens-llm bench --model qwen-local --output out/jan-benchmark.json
```

llama.cpp through a local tunnel or direct localhost endpoint:

```powershell
uv run applens-llm bench --endpoint http://127.0.0.1:18080/v1 --backend llama.cpp --model qwen2.5:7b --model-path /home/cody/local-llm/models/qwen2.5-7b-ollama.gguf --quantization "Q4_K_M GGUF" --output out/llamacpp-benchmark.json
```

Generated benchmark output is ignored under `out/`.
