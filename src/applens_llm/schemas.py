from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from applens_llm.paths import repo_root


SCHEMA_NAMES = ("deployment-plan", "benchmark-record", "training-example")


class SchemaValidationError(ValueError):
    pass


def schema_path(name: str) -> Path:
    if name not in SCHEMA_NAMES:
        valid = ", ".join(SCHEMA_NAMES)
        raise SchemaValidationError(f"Unknown schema '{name}'. Expected one of: {valid}")
    return repo_root() / "schemas" / f"{name}.schema.json"


def load_schema(name: str) -> dict[str, Any]:
    try:
        return json.loads(schema_path(name).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SchemaValidationError(f"Unable to read schema '{name}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"Schema '{name}' is invalid JSON: {exc}") from exc


def _registry() -> Registry:
    resources = []
    for name in SCHEMA_NAMES:
        schema = load_schema(name)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    return Draft202012Validator(
        schema,
        registry=_registry(),
        format_checker=FormatChecker(),
    )


def validate_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    validator = _validator(name)
    errors = sorted(validator.iter_errors(payload), key=_error_sort_key)
    if errors:
        message = "; ".join(_format_error(error) for error in errors)
        raise SchemaValidationError(message)
    return payload


def validate_document(name: str, path: str | Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"{path}: expected top-level JSON object")
    return validate_payload(name, payload)


def validate_jsonl_file(name: str, path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SchemaValidationError(f"{source}: unable to read file: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SchemaValidationError(f"{source}:{line_number}: expected JSON object")
        try:
            rows.append(validate_payload(name, payload))
        except SchemaValidationError as exc:
            raise SchemaValidationError(f"{source}:{line_number}: {exc}") from exc
    return rows


def _load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SchemaValidationError(f"{source}: unable to read file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"{source}: invalid JSON: {exc}") from exc


def _error_sort_key(error: ValidationError) -> tuple[str, str]:
    return ("/".join(str(part) for part in error.absolute_path), error.message)


def _format_error(error: ValidationError) -> str:
    path = "/".join(str(part) for part in error.absolute_path) or "<root>"
    return f"{path}: {error.message}"
