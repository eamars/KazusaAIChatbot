"""Shared helpers for read-only MongoDB export scripts."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bson import json_util
from dotenv import load_dotenv

DEFAULT_EXCLUDED_FIELDS = ("_id", "embedding")


def configure_stdout() -> None:
    """Prefer UTF-8 terminal output when the active stream supports it.

    Returns:
        None.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def configure_logging(verbose: bool) -> None:
    """Keep export scripts quiet unless verbose output is requested.

    Args:
        verbose: Whether project INFO logs should be shown.

    Returns:
        None.
    """
    if not verbose:
        logging.getLogger("kazusa_ai_chatbot").setLevel(logging.WARNING)


def load_project_env() -> Path | None:
    """Load repository ``.env`` settings before importing DB modules.

    Returns:
        The loaded ``.env`` path when found, otherwise ``None``.
    """
    for parent in (Path.cwd(), *Path.cwd().parents):
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return env_path
    load_dotenv(override=False)
    return None


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp.

    Returns:
        Current time in UTC.
    """
    return datetime.now(timezone.utc)


def timestamp_hours_ago(hours: float, *, now: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp ``hours`` before ``now``.

    Args:
        hours: Number of hours to subtract.
        now: Optional reference time for deterministic callers.

    Returns:
        ISO-8601 timestamp with a UTC offset.
    """
    reference = now or utc_now()
    return (reference - timedelta(hours=hours)).isoformat()


def parse_json_object(raw: str, label: str) -> dict[str, Any]:
    """Parse a command-line JSON object.

    Args:
        raw: JSON text supplied by the caller.
        label: Human-readable argument name for error messages.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the value is not valid JSON or does not decode to an object.
    """
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return value


def projection_from_exclusions(exclude_fields: list[str]) -> dict[str, int]:
    """Build a MongoDB projection that excludes selected fields.

    Args:
        exclude_fields: Field names to omit from exported documents.

    Returns:
        MongoDB projection document.
    """
    return {field: 0 for field in exclude_fields if field.strip()}


def scrub_document(value: Any, exclude_fields: set[str]) -> Any:
    """Remove excluded field names recursively from exported data.

    Args:
        value: Raw MongoDB-derived value.
        exclude_fields: Field names to remove wherever they appear.

    Returns:
        A recursively scrubbed value.
    """
    if isinstance(value, dict):
        return {
            key: scrub_document(item, exclude_fields)
            for key, item in value.items()
            if key not in exclude_fields
        }
    if isinstance(value, list):
        return [scrub_document(item, exclude_fields) for item in value]
    return value


def write_json_export(
    *,
    output_path: Path,
    query: dict[str, Any],
    records_key: str,
    records: list[dict[str, Any]] | dict[str, Any],
    exclude_fields: list[str],
) -> None:
    """Write an export payload as readable relaxed Extended JSON.

    Args:
        output_path: Destination path.
        query: Query metadata to include in the export.
        records_key: Payload key for the exported data.
        records: Exported records or single document.
        exclude_fields: Field names omitted from the exported data.

    Returns:
        None.
    """
    scrubbed_records = scrub_document(records, set(exclude_fields))
    record_count = len(scrubbed_records) if isinstance(scrubbed_records, list) else int(bool(scrubbed_records))
    payload = {
        "query": {
            **query,
            "excluded_fields": exclude_fields,
            "record_count": record_count,
        },
        records_key: scrubbed_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json_util.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def default_output_path(prefix: str, identifier: str, suffix: str = "json") -> Path:
    """Build a stable default output path in ``test_artifacts``.

    Args:
        prefix: File prefix describing the export domain.
        identifier: User, channel, or collection identifier.
        suffix: File extension without dot.

    Returns:
        Path under ``test_artifacts``.
    """
    clean_identifier = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in identifier.strip()
    )
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return Path("test_artifacts") / f"{prefix}_{clean_identifier}_{timestamp}.{suffix}"


PROJECT_ENV_PATH = load_project_env()
