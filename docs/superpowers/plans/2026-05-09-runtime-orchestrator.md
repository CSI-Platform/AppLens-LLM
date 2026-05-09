# Runtime Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a blackboard-backed AppLens-LLM controller that can call multiple configurable local llama.cpp/OpenAI-compatible runtime lanes and record evidence.

**Architecture:** Add focused modules for lane configuration, append-only blackboard records, and runtime orchestration. Wire them into the existing argparse CLI and keep output under ignored `out/` paths by default.

**Tech Stack:** Python 3.11 standard library, `jsonschema`, `pytest`, existing `applens-llm` CLI patterns.

---

## File Structure

- Create `schemas/runtime-lanes.schema.json`: schema for portable lane config files.
- Create `schemas/blackboard-record.schema.json`: schema for append-only blackboard JSONL events.
- Modify `src/applens_llm/schemas.py`: register the two new schemas.
- Create `src/applens_llm/runtime_lanes.py`: load/validate lane configs and resolve lane metadata.
- Create `src/applens_llm/blackboard.py`: append/read blackboard events.
- Create `src/applens_llm/orchestrator.py`: call OpenAI-compatible endpoints and write events.
- Modify `src/applens_llm/cli.py`: add `lanes-check`, `blackboard-init`, `blackboard-task`, and `orchestrate-once`.
- Create `tests/test_runtime_lanes.py`: schema and loader tests.
- Create `tests/test_blackboard.py`: append/read tests.
- Create `tests/test_orchestrator.py`: fake endpoint success/failure tests.
- Modify `tests/test_cli.py`: CLI command tests.
- Modify `docs/ARCHITECTURE.md`, `docs/DEVELOPER_GUIDE.md`, and `README.md`: document the generic orchestrator path.
- Add `examples/runtime-lanes.example.json`: sanitized portable lane config example.

---

### Task 1: Runtime Lane Schemas And Loader

**Files:**
- Create: `schemas/runtime-lanes.schema.json`
- Modify: `src/applens_llm/schemas.py`
- Create: `src/applens_llm/runtime_lanes.py`
- Create: `tests/test_runtime_lanes.py`
- Create: `examples/runtime-lanes.example.json`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from pathlib import Path

from applens_llm.runtime_lanes import get_lane, load_runtime_lanes
from applens_llm.schemas import validate_payload


def test_runtime_lanes_schema_accepts_multiple_machine_classes() -> None:
    payload = {
        "schema_version": "0.1",
        "lanes": [
            {
                "lane_id": "fast-nvidia",
                "role": "fast",
                "engine": "llama.cpp",
                "backend": "cuda",
                "endpoint": "http://127.0.0.1:18081/v1",
                "model": {"label": "jan-v35-4b", "path": "models/jan-v35-4b.gguf"},
                "device": {"selector": "cuda:0", "accelerator_ids": ["nvidia-dgpu-0"]},
                "launch": {"server_binary": "llama-server", "context_tokens": 4096, "gpu_layers": 99, "threads": 12, "environment": {}},
            },
            {
                "lane_id": "deep-apple",
                "role": "deep",
                "engine": "llama.cpp",
                "backend": "metal",
                "endpoint": "http://127.0.0.1:18082/v1",
                "model": {"label": "gemma-local", "path": "models/gemma.gguf"},
                "device": {"selector": "metal", "accelerator_ids": ["apple-gpu-0"]},
                "launch": {"server_binary": "llama-server", "context_tokens": 8192, "gpu_layers": 99, "threads": 8, "environment": {}},
            },
            {
                "lane_id": "cpu-baseline",
                "role": "baseline",
                "engine": "openai-compatible",
                "backend": "cpu",
                "endpoint": "http://127.0.0.1:18083/v1",
                "model": {"label": "tiny-cpu", "path": "models/tiny.gguf"},
                "device": {"selector": "cpu", "accelerator_ids": ["cpu-0"]},
                "launch": {"server_binary": "llama-server", "context_tokens": 2048, "gpu_layers": 0, "threads": 4, "environment": {}},
            },
        ],
    }

    validate_payload("runtime-lanes", payload)


def test_load_runtime_lanes_returns_lane_by_id(tmp_path: Path) -> None:
    config = tmp_path / "lanes.json"
    config.write_text(
        '{"schema_version":"0.1","lanes":[{"lane_id":"deep-amd-vgm","role":"deep","engine":"llama.cpp","backend":"vulkan","endpoint":"http://127.0.0.1:18082/v1","model":{"label":"qwen-27b-iq3","path":"models/qwen.gguf"},"device":{"selector":"Vulkan0","accelerator_ids":["amd-igpu-0"]},"launch":{"server_binary":"llama-server","context_tokens":4096,"gpu_layers":99,"threads":12,"environment":{"GGML_VK_DISABLE_COOPMAT":"1"}}}]}',
        encoding="utf-8",
    )

    lanes = load_runtime_lanes(config)
    lane = get_lane(lanes, "deep-amd-vgm")

    assert lane["backend"] == "vulkan"
    assert lane["device"]["accelerator_ids"] == ["amd-igpu-0"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runtime_lanes.py -v`

Expected: FAIL because `applens_llm.runtime_lanes` and the schema registration do not exist.

- [ ] **Step 3: Add schema registration**

Add `"runtime-lanes"` to `SCHEMA_NAMES` in `src/applens_llm/schemas.py`.

- [ ] **Step 4: Create `runtime-lanes.schema.json`**

Use a strict object with `schema_version` and a non-empty `lanes` array. Each lane requires `lane_id`, `role`, `engine`, `backend`, `endpoint`, `model`, `device`, and `launch`. Backend enum includes `cuda`, `vulkan`, `rocm`, `hip`, `directml`, `metal`, `openvino`, `cpu`, `rpc`, and `unknown`.

- [ ] **Step 5: Implement `runtime_lanes.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


def load_runtime_lanes(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_payload("runtime-lanes", payload)
    return payload


def get_lane(config: dict[str, Any], lane_id: str) -> dict[str, Any]:
    for lane in config["lanes"]:
        if lane["lane_id"] == lane_id:
            return lane
    raise KeyError(f"runtime lane not found: {lane_id}")
```

- [ ] **Step 6: Add sanitized example config**

Create `examples/runtime-lanes.example.json` with generic lane IDs and sanitized model paths like `models/jan-v35-4b.gguf`, not local absolute paths.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_runtime_lanes.py -v`

Expected: PASS.

---

### Task 2: Blackboard Event Schema And Writer

**Files:**
- Create: `schemas/blackboard-record.schema.json`
- Modify: `src/applens_llm/schemas.py`
- Create: `src/applens_llm/blackboard.py`
- Create: `tests/test_blackboard.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json
from pathlib import Path

from applens_llm.blackboard import append_event, read_events, start_experiment
from applens_llm.schemas import validate_payload


def test_blackboard_record_schema_accepts_task_and_response() -> None:
    task = {
        "schema_version": "0.1",
        "event_id": "evt-test-task",
        "experiment_id": "exp-test",
        "event_type": "task",
        "created_at": "2026-05-09T00:00:00Z",
        "payload": {"task_id": "task-1", "prompt": "Compare these runtimes.", "metadata": {}},
        "privacy": {"commit_safe": True, "local_paths_included": False},
    }
    response = {
        "schema_version": "0.1",
        "event_id": "evt-test-response",
        "experiment_id": "exp-test",
        "event_type": "model_response",
        "created_at": "2026-05-09T00:00:01Z",
        "payload": {
            "task_id": "task-1",
            "lane_id": "fast-nvidia",
            "model_label": "jan-v35-4b",
            "backend": "cuda",
            "accelerator_ids": ["nvidia-dgpu-0"],
            "latency_ms": 1250,
            "outcome": "success",
            "content": "Short answer.",
        },
        "privacy": {"commit_safe": False, "local_paths_included": False},
    }

    validate_payload("blackboard-record", task)
    validate_payload("blackboard-record", response)


def test_append_and_read_blackboard_events(tmp_path: Path) -> None:
    path = tmp_path / "experiment.jsonl"
    event = start_experiment(path, experiment_id="exp-test", title="Unit test")
    append_event(
        path,
        experiment_id="exp-test",
        event_type="task",
        payload={"task_id": "task-1", "prompt": "Hello", "metadata": {}},
        commit_safe=True,
    )

    events = read_events(path)

    assert events[0]["event_type"] == "experiment_started"
    assert events[0]["event_id"] == event["event_id"]
    assert events[1]["payload"]["prompt"] == "Hello"
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_blackboard.py -v`

Expected: FAIL because `applens_llm.blackboard` and the schema registration do not exist.

- [ ] **Step 3: Register schema**

Add `"blackboard-record"` to `SCHEMA_NAMES` in `src/applens_llm/schemas.py`.

- [ ] **Step 4: Create `blackboard-record.schema.json`**

Use strict required fields: `schema_version`, `event_id`, `experiment_id`, `event_type`, `created_at`, `payload`, and `privacy`. Event enum: `experiment_started`, `task`, `model_response`, `handoff`, `benchmark_reference`, `failure`, `verdict`.

- [ ] **Step 5: Implement `blackboard.py`**

```python
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


def start_experiment(path: Path, *, experiment_id: str, title: str) -> dict[str, Any]:
    return append_event(
        path,
        experiment_id=experiment_id,
        event_type="experiment_started",
        payload={"title": title},
        commit_safe=False,
    )


def append_event(
    path: Path,
    *,
    experiment_id: str,
    event_type: str,
    payload: dict[str, Any],
    commit_safe: bool = False,
    local_paths_included: bool = False,
) -> dict[str, Any]:
    event = {
        "schema_version": "0.1",
        "event_id": f"evt-{uuid.uuid4().hex}",
        "experiment_id": experiment_id,
        "event_type": event_type,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": payload,
        "privacy": {"commit_safe": commit_safe, "local_paths_included": local_paths_included},
    }
    validate_payload("blackboard-record", event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_blackboard.py -v`

Expected: PASS.

---

### Task 3: Orchestrator Endpoint Client

**Files:**
- Create: `src/applens_llm/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

from applens_llm.blackboard import read_events, start_experiment
from applens_llm.orchestrator import run_lane_once


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        response = {
            "choices": [{"message": {"content": f"lane answered: {body['messages'][-1]['content']}"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def _server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), ChatHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/v1"


def test_run_lane_once_records_success(tmp_path: Path) -> None:
    server, endpoint = _server()
    try:
        blackboard = tmp_path / "run.jsonl"
        start_experiment(blackboard, experiment_id="exp-test", title="Unit")
        lane = {
            "lane_id": "fast-nvidia",
            "role": "fast",
            "engine": "llama.cpp",
            "backend": "cuda",
            "endpoint": endpoint,
            "model": {"label": "jan-v35-4b", "path": "models/jan.gguf"},
            "device": {"selector": "cuda:0", "accelerator_ids": ["nvidia-dgpu-0"]},
            "launch": {"server_binary": "llama-server", "context_tokens": 4096, "gpu_layers": 99, "threads": 12, "environment": {}},
        }

        event = run_lane_once(
            blackboard,
            experiment_id="exp-test",
            task_id="task-1",
            prompt="hello",
            lane=lane,
            timeout_seconds=5,
        )

        events = read_events(blackboard)
        assert event["payload"]["outcome"] == "success"
        assert event["payload"]["content"] == "lane answered: hello"
        assert event["payload"]["backend"] == "cuda"
        assert events[-1]["event_type"] == "model_response"
    finally:
        server.shutdown()


def test_run_lane_once_records_connection_failure(tmp_path: Path) -> None:
    blackboard = tmp_path / "run.jsonl"
    start_experiment(blackboard, experiment_id="exp-test", title="Unit")
    lane = {
        "lane_id": "deep-amd-vgm",
        "role": "deep",
        "engine": "llama.cpp",
        "backend": "vulkan",
        "endpoint": "http://127.0.0.1:9/v1",
        "model": {"label": "qwen-27b", "path": "models/qwen.gguf"},
        "device": {"selector": "Vulkan0", "accelerator_ids": ["amd-igpu-0"]},
        "launch": {"server_binary": "llama-server", "context_tokens": 4096, "gpu_layers": 99, "threads": 12, "environment": {}},
    }

    event = run_lane_once(
        blackboard,
        experiment_id="exp-test",
        task_id="task-1",
        prompt="hello",
        lane=lane,
        timeout_seconds=1,
    )

    assert event["event_type"] == "failure"
    assert event["payload"]["outcome"] == "connection_error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator.py -v`

Expected: FAIL because `applens_llm.orchestrator` does not exist.

- [ ] **Step 3: Implement `orchestrator.py`**

Use `urllib.request` from the standard library. POST to `{endpoint}/chat/completions` unless the endpoint already ends with `/chat/completions`. Capture elapsed time with `time.perf_counter()`. Write `model_response` on HTTP 200 and `failure` on `URLError`, timeout, invalid JSON, or non-200 response.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_orchestrator.py -v`

Expected: PASS.

---

### Task 4: CLI Wiring

**Files:**
- Modify: `src/applens_llm/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add these tests to `tests/test_cli.py`:

```python
def test_cli_checks_runtime_lanes(tmp_path: Path) -> None:
    config = tmp_path / "lanes.json"
    config.write_text('{"schema_version":"0.1","lanes":[]}', encoding="utf-8")
    result = run_cli("lanes-check", "--config", str(config))
    assert result.returncode == 2
```

and a valid config with one lane that returns `1 runtime lanes valid`.

Add a blackboard init test:

```python
def test_cli_initializes_blackboard(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    result = run_cli("blackboard-init", "--experiment-id", "exp-test", "--title", "Unit", "--output", str(output))
    assert result.returncode == 0
    assert "blackboard initialized" in result.stdout
    assert output.exists()
```

Add an `orchestrate-once` failure-path test that points a valid lane at `http://127.0.0.1:9/v1`, runs the command with `--timeout-seconds 1`, expects return code `0`, and asserts the output blackboard file contains a final `failure` event. This keeps CLI coverage deterministic without duplicating the fake HTTP server from `tests/test_orchestrator.py`.

- [ ] **Step 2: Run CLI tests to verify failure**

Run: `uv run pytest tests/test_cli.py -v`

Expected: FAIL because new commands do not exist.

- [ ] **Step 3: Add CLI commands**

Add imports from `blackboard`, `runtime_lanes`, and `orchestrator`.

Add parsers:

```text
lanes-check --config PATH
blackboard-init --experiment-id ID --title TITLE --output PATH
blackboard-task --experiment-id ID --task-id ID --prompt TEXT --output PATH
orchestrate-once --config PATH --lane ID --experiment-id ID --task-id ID --prompt TEXT --blackboard PATH --timeout-seconds N
```

Return codes:

- `0` for success
- `2` for schema errors, matching existing validation behavior
- `3` for escaped OS/network benchmark style errors, matching the existing benchmark error branch

- [ ] **Step 4: Run CLI tests**

Run: `uv run pytest tests/test_cli.py -v`

Expected: PASS.

---

### Task 5: Documentation And End-to-End Validation

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/DEVELOPER_GUIDE.md`
- Modify: `docs/VGM_TEST_GUIDE.md`

- [ ] **Step 1: Document the generic runtime lane model**

Add a short section explaining that lanes are portable and can represent CUDA, Vulkan, ROCm/HIP, Metal, DirectML, CPU, RPC, or OpenAI-compatible endpoints.

- [ ] **Step 2: Document local ASUS lane example without private paths**

Reference `examples/runtime-lanes.example.json`. Keep committed examples sanitized.

- [ ] **Step 3: Run validation**

Run:

```powershell
uv run pytest
uv run applens-llm validate --schema runtime-lanes examples/runtime-lanes.example.json
```

Expected: all tests pass and the example validates.

- [ ] **Step 4: Inspect for private identifiers**

Run:

```powershell
rg -n "C:\\Users|AppData|UUID|serial|product id|device id" examples schemas docs src tests
```

Expected: no new committed orchestrator examples contain raw local paths or private device identifiers. Existing docs may mention sanitized guidance; inspect any hits before finalizing.

---

## Self-Review Notes

- Spec coverage: lane config, blackboard ledger, controller CLI, error handling, scalability, and non-goals are covered.
- Scope guard: no service installation, model download, driver changes, scheduled overnight automation, training, GRPO, or Stigmergent dependency.
- Type consistency: lane fields are `lane_id`, `role`, `engine`, `backend`, `endpoint`, `model`, `device`, and `launch`; blackboard fields are `schema_version`, `event_id`, `experiment_id`, `event_type`, `created_at`, `payload`, and `privacy`.
