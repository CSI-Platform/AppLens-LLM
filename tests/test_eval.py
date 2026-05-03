from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from applens_llm.eval import evaluate_training_examples_file, score_training_examples
from applens_llm.schemas import validate_payload


ROOT = Path(__file__).resolve().parents[1]


def test_seed_examples_score_as_passing_eval_set() -> None:
    report = evaluate_training_examples_file(ROOT / "data" / "examples.seed.jsonl")

    validate_payload("eval-report", report)
    assert report["total"] == 2
    assert report["scores"]["schema_valid"] == 2
    assert report["scores"]["policy_valid"] == 2
    assert report["scores"]["expected_match"] == 2
    assert report["scores"]["pass_rate"] == 1


def test_policy_check_catches_missing_network_gate() -> None:
    row = _load_seed_rows()[1]
    bad_plan = deepcopy(row["expected_output"])
    bad_plan["gated_jobs"] = [
        gate for gate in bad_plan["gated_jobs"] if gate["job"] != "network_exposure"
    ]
    row["messages"][-1]["content"] = json.dumps(bad_plan)

    report = score_training_examples([row], source="unit-test")
    result = report["results"][0]

    assert result["checks"]["schema_valid"] is True
    assert result["checks"]["policy_valid"] is False
    assert any("network_exposure" in issue for issue in result["issues"])


def test_eval_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "eval-report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "applens_llm.cli",
            "eval",
            "--examples",
            "data/examples.seed.jsonl",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "2/2 pass" in result.stdout
    validate_payload("eval-report", json.loads(output.read_text(encoding="utf-8")))


def _load_seed_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / "data" / "examples.seed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
