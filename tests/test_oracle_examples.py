from __future__ import annotations

from pathlib import Path

from applens_llm.schemas import validate_document


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_workload_example_validates() -> None:
    validate_document("workload-profile", ROOT / "examples" / "oracle" / ".applens" / "workload.json")


def test_oracle_manifest_example_validates() -> None:
    validate_document(
        "autoresearch-run-manifest",
        ROOT / "examples" / "oracle" / ".applens" / "runs" / "oracle-dry-run.example.json",
    )


def test_oracle_program_is_small_model_legible() -> None:
    text = (ROOT / "examples" / "oracle" / ".applens" / "program.md").read_text(encoding="utf-8")

    assert "Goal:" in text
    assert "Allowed:" in text
    assert "Blocked:" in text
    assert "Loop:" in text
    assert len(text.split()) < 220


def test_oracle_probe_and_eval_examples_validate() -> None:
    validate_document("autoresearch-probes", ROOT / "examples" / "oracle" / ".applens" / "probes.json")
    validate_document("autoresearch-eval-cases", ROOT / "examples" / "oracle" / ".applens" / "evals" / "cases.json")

