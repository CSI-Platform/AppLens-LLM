from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "applens_llm.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_cli_validates_json_document() -> None:
    result = run_cli(
        "validate",
        "--schema",
        "deployment-plan",
        "examples/gaming-pc-deployment-plan.json",
    )

    assert result.returncode == 0
    assert "valid" in result.stdout


def test_cli_validates_jsonl_file() -> None:
    result = run_cli(
        "validate-jsonl",
        "--schema",
        "training-example",
        "data/examples.seed.jsonl",
    )

    assert result.returncode == 0
    assert "rows valid" in result.stdout
