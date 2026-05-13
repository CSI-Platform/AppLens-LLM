# Benchmark Suite Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a schema-valid benchmark-suite wrapper that standardizes model, machine condition, runtime lane, benchmark task selection, local metrics, and output artifacts before AppLens-LLM runs local model comparisons.

**Architecture:** Add a `benchmark-suite-run` contract and builder module that emits a plan-only artifact for `tiny-v1` and `small-v1`. The contract records benchmark selection rationale, LongBench v2 as primary long-context capability evidence, RULER as diagnostic context taper, and local machine conditions such as VGM/RAM split.

**Tech Stack:** Python 3.11, argparse CLI, JSON Schema Draft 2020-12, pytest.

---

### Task 1: Contract Tests

**Files:**
- Create: `tests/test_benchmark_suite.py`
- Modify: `tests/test_cli.py`

- [x] **Step 1: Write failing tests**

Tests assert that `build_benchmark_suite_run()` emits schema-valid `tiny-v1` and `small-v1` plans, that tiny models include IFEval, ARC-Challenge, HellaSwag, GSM8K, BFCL prompt mode, BigCodeBench-Hard screening, LongBench v2 screening, and RULER taper, and that small models include the Open LLM Leaderboard v2 family plus coding, tool calling, and long-context tasks.

- [x] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_benchmark_suite.py tests/test_cli.py::test_cli_writes_benchmark_suite_plan -q`

Expected: failure because `applens_llm.benchmark_suite` and `benchmark-suite-run` schema do not exist yet.

### Task 2: Schema and Builder

**Files:**
- Create: `schemas/benchmark-suite-run.schema.json`
- Create: `src/applens_llm/benchmark_suite.py`
- Modify: `src/applens_llm/schemas.py`

- [x] **Step 1: Add schema**

The schema must require `schema_version`, `suite_run_id`, `created_at`, `suite`, `model`, `machine_condition`, `runtime_lane`, `execution`, `benchmark_plan`, `output_contract`, and `privacy`.

- [x] **Step 2: Add builder**

Implement `build_benchmark_suite_run()` and `write_benchmark_suite_run()`. Model size class rules are `tiny` for `parameter_size_b <= 4.5`, `small` for `4.5 < parameter_size_b <= 30`, and `outside_v1_scope` for larger models.

- [x] **Step 3: Register schema**

Add `benchmark-suite-run` to `SCHEMA_NAMES`.

- [x] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_benchmark_suite.py -q`

Expected: pass.

### Task 3: CLI

**Files:**
- Modify: `src/applens_llm/cli.py`

- [x] **Step 1: Add command**

Add `benchmark-suite-plan` with model, condition, runtime, suite, scoring, and output arguments.

- [x] **Step 2: Run CLI test**

Run: `uv run pytest tests/test_cli.py::test_cli_writes_benchmark_suite_plan -q`

Expected: pass.

### Task 4: Example and Docs

**Files:**
- Create: `examples/asus-px13-benchmark-suite-run.example.json`
- Modify: `tests/test_schemas.py`
- Modify: `README.md`
- Modify: `ROADMAP.md`

- [x] **Step 1: Add sanitized example**

The example must not include raw user paths, device UUIDs, serials, or private product IDs.

- [x] **Step 2: Add docs**

Document the plan-only command and explain that local screening subsets are not leaderboard-comparable certification runs.

- [x] **Step 3: Validate schema example**

Run: `uv run pytest tests/test_schemas.py -q`

Expected: pass.

### Task 5: Verification

**Files:**
- No new files.

- [x] **Step 1: Run focused suite**

Run: `uv run pytest tests/test_benchmark_suite.py tests/test_cli.py::test_cli_writes_benchmark_suite_plan tests/test_schemas.py -q`

Expected: pass.

- [x] **Step 2: Run full suite**

Run: `uv run pytest -q`

Expected: all tests pass.
