"""Validate prompt-facing RAG2 evidence artifacts for raw-data leaks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PUBLIC_EVIDENCE_KEYS = (
    "answer",
    "third_party_profiles",
    "memory_evidence",
    "recall_evidence",
    "conversation_evidence",
    "external_evidence",
)

FORBIDDEN_KEYS = {
    "_id",
    "base64_data",
    "binary",
    "conversation_row_id",
    "embedding",
    "global_user_id",
    "platform_message_id",
    "raw_wire_text",
    "source_global_user_id",
    "storage_object_id",
    "storage_url",
}

RAW_ROW_KEYS = {
    "candidates",
    "projection_payload",
    "raw_result",
    "resolved_refs",
    "rows",
    "source_refs",
    "worker_payloads",
}

FORBIDDEN_TEXT_MARKERS = (
    "[CQ:",
    "raw_wire_text",
    "base64_data",
    "conversation_row_id",
    "global_user_id",
    "platform_message_id",
    ";base64,",
    "source_global_user_id",
    "data:image/",
)
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{12}\b",
    flags=re.IGNORECASE,
)

RAW_URL_ASSIGNMENT_RE = re.compile(r"\burl\s*=", flags=re.IGNORECASE)
RAW_UTC_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:[\.,]\d{1,6})?"
    r"(?:Z|\+00:00)\b",
    flags=re.IGNORECASE,
)
EMPTY_IMAGE_RE = re.compile(r"<image>\s*</image>", flags=re.IGNORECASE)


def main() -> int:
    """Run artifact validation from CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Validate public projected_rag_result evidence artifacts.",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--case-prefix", default="")
    parser.add_argument("--expected-count", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    artifact_paths = _artifact_paths(input_dir, args.case_prefix)
    errors: list[str] = []

    if args.expected_count and len(artifact_paths) != args.expected_count:
        errors.append(
            f"expected {args.expected_count} artifacts, found {len(artifact_paths)}"
        )

    for artifact_path in artifact_paths:
        _validate_artifact(artifact_path, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"validated {len(artifact_paths)} public RAG2 evidence artifacts")
    return 0


def _artifact_paths(input_dir: Path, case_prefix: str) -> list[Path]:
    """Return sorted JSON artifact paths matching the requested prefix."""

    if not input_dir.exists():
        return []

    paths = [
        path
        for path in input_dir.glob("*.json")
        if not case_prefix or path.name.startswith(case_prefix)
    ]
    paths.sort(key=lambda path: path.name)
    return paths


def _validate_artifact(path: Path, errors: list[str]) -> None:
    """Validate one JSON artifact."""

    try:
        with path.open("r", encoding="utf-8") as file:
            artifact = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: cannot read JSON artifact: {exc}")
        return

    if not isinstance(artifact, dict):
        errors.append(f"{path}: artifact root is not an object")
        return

    projected = artifact.get("projected_rag_result")
    if not isinstance(projected, dict):
        errors.append(f"{path}: missing projected_rag_result object")
        return

    for key in PUBLIC_EVIDENCE_KEYS:
        if key not in projected:
            continue
        _collect_violations(
            projected[key],
            path=f"{path.name}.projected_rag_result.{key}",
            errors=errors,
        )


def _collect_violations(value: object, *, path: str, errors: list[str]) -> None:
    """Collect public evidence leak violations from a value subtree."""

    if isinstance(value, dict):
        if _looks_like_raw_source_row(value):
            errors.append(f"{path}: raw source row dictionary leaked")
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}"
            if key_text in FORBIDDEN_KEYS:
                errors.append(f"{item_path}: forbidden raw key")
            if key_text == "url" and not _allowed_external_url_path(item_path):
                errors.append(f"{item_path}: raw url key outside external evidence")
            _collect_violations(item, path=item_path, errors=errors)
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if isinstance(item, str) and not item.strip():
                errors.append(f"{item_path}: blank evidence string")
            _collect_violations(item, path=item_path, errors=errors)
        return

    if isinstance(value, str):
        _collect_text_violations(value, path=path, errors=errors)


def _looks_like_raw_source_row(value: dict[Any, Any]) -> bool:
    """Return whether a dict contains raw worker/source-row shape."""

    keys = {str(key) for key in value}
    has_raw_row_key = bool(keys & RAW_ROW_KEYS)
    has_storage_identity = bool(keys & FORBIDDEN_KEYS)
    return has_raw_row_key or has_storage_identity


def _collect_text_violations(
    value: str,
    *,
    path: str,
    errors: list[str],
) -> None:
    """Collect prompt-unsafe text markers from a string."""

    for marker in FORBIDDEN_TEXT_MARKERS:
        if marker in value:
            errors.append(f"{path}: forbidden text marker {marker!r}")
            return

    if RAW_URL_ASSIGNMENT_RE.search(value):
        errors.append(f"{path}: raw url assignment text")
        return

    if RAW_UTC_RE.search(value):
        errors.append(f"{path}: raw UTC timestamp")
        return

    if UUID_RE.search(value) and not _allowed_uuid_path(path):
        errors.append(f"{path}: raw UUID text")
        return

    if EMPTY_IMAGE_RE.search(value):
        errors.append(f"{path}: empty image block")


def _allowed_external_url_path(path: str) -> bool:
    """Return whether a public URL key belongs to external evidence."""

    allowed = (
        ".projected_rag_result.external_evidence[" in path
        and path.endswith(".url")
    )
    return allowed


def _allowed_uuid_path(path: str) -> bool:
    """Return whether a UUID-like string is allowed as compatibility metadata."""

    allowed = path.endswith(".scope_global_user_id")
    return allowed


if __name__ == "__main__":
    raise SystemExit(main())
