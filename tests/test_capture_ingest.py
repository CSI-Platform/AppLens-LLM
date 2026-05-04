from __future__ import annotations

import json
from pathlib import Path

from applens_llm.capture_ingest import discover_capture_records, write_capture_records_jsonl
from applens_llm.schemas import validate_payload


def test_discover_capture_records_prefers_markdown_reports(tmp_path: Path) -> None:
    capture_dir = tmp_path / "AppLens-Capture-TEST-PC"
    capture_dir.mkdir()
    (capture_dir / "AppLens_Results_TEST-PC.md").write_text(
        "\n".join(
            [
                "# AppLens Scan Results",
                "- **Computer:** TEST-PC",
                "- **User:** cody",
                "- **OS:** Microsoft Windows 11 Home (10.0.26200)",
                "- **Scan Date:** 2026-05-03",
                "",
                "## Desktop Applications",
                "Jan",
            ]
        ),
        encoding="utf-8",
    )
    (capture_dir / "AppLens_Tune_Results_TEST-PC.md").write_text(
        "\n".join(
            [
                "# AppLens-Tune Audit Results",
                "- **Computer:** TEST-PC",
                "- **User:** cody",
                "- **Scan Date:** 2026-05-03 20:06:03",
                "- **Mode:** Audit (read-only)",
                "",
                "- **Machine:** Example Vendor Model SKU123",
                "- **OS:** Microsoft Windows 11 Home (10.0.26200)",
                "- **RAM:** 32 GB",
                "- **C: Free:** 100 GB",
                "",
                "## Local LLM Profile",
                "GPU tier             Small NVIDIA GPU profile",
                "",
                "## NVIDIA GPU Profile",
                "Name  Driver  VRAM_MB",
            ]
        ),
        encoding="utf-8",
    )
    (capture_dir / "README-What-To-Send.md").write_text("# AppLens Capture\n", encoding="utf-8")
    (capture_dir / "AppLens_Run_Log.txt").write_text("ok\n", encoding="utf-8")
    (capture_dir / "AppLens_Tune_Run_Log.txt").write_text("ok\n", encoding="utf-8")

    records = discover_capture_records(tmp_path)

    assert len(records) == 1
    record = records[0]
    validate_payload("capture-record", record)
    assert record["capture_id"] == "test-pc"
    assert record["reports"]["applens"] == "AppLens-Capture-TEST-PC/AppLens_Results_TEST-PC.md"
    assert record["reports"]["applens_tune"] == "AppLens-Capture-TEST-PC/AppLens_Tune_Results_TEST-PC.md"
    assert record["metadata"]["computer"] == "TEST-PC"
    assert record["metadata"]["machine"] == "Example Vendor Model SKU123"
    assert record["inferred"]["os_family"] == "windows"
    assert record["inferred"]["has_local_llm_profile"] is True
    assert record["inferred"]["has_gpu_profile"] is True
    assert record["privacy"]["sanitized"] is False


def test_discover_capture_records_accepts_legacy_txt_reports(tmp_path: Path) -> None:
    (tmp_path / "AppLens_Results_legacy-host.txt").write_text(
        "\n".join(
            [
                "=== AppLens Scan Results ===",
                "Computer: legacy-host",
                "User: cody",
                "OS: Linux-6.17.0-14-generic-x86_64-with-glibc2.39",
                "Scan Date: 2026-05-03",
                "",
                "--- Desktop Applications ---",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "AppLens_Tune_Results_legacy-host.txt").write_text(
        "\n".join(
            [
                "=== AppLens-Tune Audit Results ===",
                "Computer: legacy-host",
                "User: cody",
                "Scan Date: 2026-05-03 19:07:35",
                "Mode: Audit (read-only)",
                "",
                "Machine: x86_64",
                "OS: Linux-6.17.0-14-generic-x86_64-with-glibc2.39",
                "RAM: 7.60 GB",
                "Root Free: 72.61 GB",
                "",
                "--- Local LLM Profile ---",
            ]
        ),
        encoding="utf-8",
    )

    records = discover_capture_records(tmp_path)

    assert len(records) == 1
    record = records[0]
    validate_payload("capture-record", record)
    assert record["capture_id"] == "legacy-host"
    assert record["reports"]["applens"].endswith(".txt")
    assert record["reports"]["applens_tune"].endswith(".txt")
    assert record["inferred"]["os_family"] == "linux"
    assert record["inferred"]["has_local_llm_profile"] is True


def test_write_capture_records_jsonl(tmp_path: Path) -> None:
    (tmp_path / "AppLens_Results_one.txt").write_text(
        "=== AppLens Scan Results ===\nComputer: one\n",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "captures.jsonl"

    count = write_capture_records_jsonl(tmp_path, output)

    assert count == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["capture_id"] == "one"
