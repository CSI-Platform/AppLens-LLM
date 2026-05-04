from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from applens_llm.schemas import validate_payload


REPORT_PATTERN = re.compile(r"^(AppLens(?:_Tune)?_Results)_(?P<computer>.+)\.(?P<ext>md|txt)$", re.IGNORECASE)
SECTION_MARKDOWN_PATTERN = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")
SECTION_LEGACY_PATTERN = re.compile(r"^---\s*(?P<title>.+?)\s*---$")
MARKDOWN_METADATA_PATTERN = re.compile(r"^-\s+\*\*(?P<key>[^:]+):\*\*\s*(?P<value>.*)$")
LEGACY_METADATA_PATTERN = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 /_.()-]*):\s*(?P<value>.*)$")
WINDOWS_PATH_PATTERN = re.compile(r"\b[A-Za-z]:\\")
POSIX_HOME_PATH_PATTERN = re.compile(r"(^|\s)/(home|Users)/[^\s]+")
SERIAL_TERMS_PATTERN = re.compile(r"\b(serial|uuid|identifyingnumber)\b", re.IGNORECASE)


def discover_capture_records(source: str | Path) -> list[dict[str, Any]]:
    root = Path(source)
    groups: dict[tuple[Path, str], dict[str, Path]] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        match = REPORT_PATTERN.match(path.name)
        if not match:
            continue
        computer = match.group("computer")
        key = (path.parent, computer)
        group = groups.setdefault(key, {})
        report_key = "applens_tune" if path.name.lower().startswith("applens_tune_results_") else "applens"
        current = group.get(report_key)
        if current is None or _report_priority(path) > _report_priority(current):
            group[report_key] = path

    records = [
        _build_record(root, folder, computer, reports)
        for (folder, computer), reports in sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1].lower()))
    ]
    return records


def write_capture_records_jsonl(source: str | Path, output: str | Path) -> int:
    records = discover_capture_records(source)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return len(records)


def _build_record(root: Path, folder: Path, computer: str, reports: dict[str, Path]) -> dict[str, Any]:
    app_report = reports.get("applens")
    tune_report = reports.get("applens_tune")
    app_text = _read_text(app_report)
    tune_text = _read_text(tune_report)
    app_metadata = _extract_metadata(app_text)
    tune_metadata = _extract_metadata(tune_text)
    metadata = _merge_metadata(app_metadata, tune_metadata, computer)
    app_sections = _extract_sections(app_text)
    tune_sections = _extract_sections(tune_text)
    readme = _find_first(folder, ("README-What-To-Send.md", "README-What-To-Send.txt"))
    app_log = _find_first(folder, ("AppLens_Run_Log.txt",))
    tune_log = _find_first(folder, ("AppLens_Tune_Run_Log.txt",))
    combined_text = "\n".join(text for text in (app_text, tune_text) if text)

    record = {
        "schema_version": "0.1",
        "capture_id": _slug(metadata["computer"] or computer),
        "source": {
            "root_name": root.name,
            "folder": _relative_path(root, folder) or ".",
        },
        "reports": {
            "applens": _relative_path(root, app_report),
            "applens_tune": _relative_path(root, tune_report),
            "readme": _relative_path(root, readme),
            "applens_log": _relative_path(root, app_log),
            "applens_tune_log": _relative_path(root, tune_log),
        },
        "metadata": metadata,
        "sections": {
            "applens": app_sections,
            "applens_tune": tune_sections,
        },
        "inferred": {
            "os_family": _infer_os_family(metadata["os"]),
            "has_applens_report": app_report is not None,
            "has_applens_tune_report": tune_report is not None,
            "has_local_llm_profile": "Local LLM Profile" in tune_sections,
            "has_gpu_profile": "NVIDIA GPU Profile" in tune_sections,
        },
        "privacy": {
            "sanitized": False,
            "raw_paths_detected": bool(WINDOWS_PATH_PATTERN.search(combined_text) or POSIX_HOME_PATH_PATTERN.search(combined_text)),
            "serial_or_uuid_terms_detected": bool(SERIAL_TERMS_PATTERN.search(combined_text)),
        },
    }
    return validate_payload("capture-record", record)


def _merge_metadata(app_metadata: dict[str, str], tune_metadata: dict[str, str], computer: str) -> dict[str, str | None]:
    merged = {**app_metadata, **tune_metadata}
    free_space = merged.get("c: free") or merged.get("root free")
    return {
        "computer": merged.get("computer") or computer,
        "user": merged.get("user"),
        "scan_date": merged.get("scan date"),
        "mode": merged.get("mode"),
        "machine": merged.get("machine"),
        "os": merged.get("os"),
        "ram": merged.get("ram"),
        "free_space": free_space,
    }


def _extract_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = MARKDOWN_METADATA_PATTERN.match(line) or LEGACY_METADATA_PATTERN.match(line)
        if not match:
            continue
        key = _normalize_key(match.group("key"))
        value = match.group("value").strip()
        if value:
            metadata[key] = value
    return metadata


def _extract_sections(text: str) -> list[str]:
    sections: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = SECTION_MARKDOWN_PATTERN.match(line) or SECTION_LEGACY_PATTERN.match(line)
        if not match:
            continue
        title = match.group("title").strip()
        if title.startswith("AppLens"):
            continue
        if title and title not in sections:
            sections.append(title)
    return sections


def _infer_os_family(os_text: str | None) -> str:
    value = (os_text or "").lower()
    if "windows" in value or "microsoft" in value:
        return "windows"
    if "darwin" in value or "macos" in value or "mac os" in value:
        return "macos"
    if "linux" in value or "glibc" in value or "ubuntu" in value:
        return "linux"
    return "unknown"


def _read_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _find_first(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = folder / name
        if path.exists():
            return path
    return None


def _relative_path(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _report_priority(path: Path) -> int:
    return 2 if path.suffix.lower() == ".md" else 1


def _normalize_key(key: str) -> str:
    return re.sub(r"\s+", " ", key.strip().lower())


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "unknown-capture"
