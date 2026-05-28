"""Target and source extraction helpers for live context."""

from __future__ import annotations

import re
from typing import Any

from kazusa_ai_chatbot.utils import text_or_empty

_URL_PATTERN = re.compile(r"https?://[^\s)>\]}\"']+")

def _clean_target(target: str) -> str:
    """Normalize a target phrase extracted from structured slot text."""
    cleaned = target.strip(" .,:;\"'")
    return cleaned

def _target_after_marker(task_body: str, marker: str) -> str:
    """Extract text following a structured target marker."""
    pattern = re.compile(rf"{re.escape(marker)}\s+(.+)$", re.IGNORECASE)
    match = pattern.search(task_body)
    if match is None:
        return_value = ""
        return return_value
    return_value = _clean_target(match.group(1))
    return return_value

def _extract_text_from_rows(rows: object) -> str:
    """Extract a target/scope phrase from a worker result payload.

    Args:
        rows: Raw worker ``result`` value.

    Returns:
        First available human-readable target text.
    """

    if isinstance(rows, str):
        return rows.strip()

    if isinstance(rows, dict):
        for field in (
            "content",
            "body_text",
            "text",
            "description",
            "summary",
            "selected_summary",
            "fact",
        ):
            text = text_or_empty(rows.get(field))
            if text:
                return text
        return_value = ""
        return return_value

    if isinstance(rows, list):
        for item in rows:
            text = _extract_text_from_rows(item)
            if text:
                return text
        return_value = ""
        return return_value

    return_value = text_or_empty(rows)
    return return_value

def _extract_worker_text(worker_result: dict[str, Any]) -> str:
    """Extract target/scope text from a helper-agent result."""
    rows = worker_result.get("result")
    text = _extract_text_from_rows(rows)
    return text

def _url_from_text(text: str) -> str:
    """Return the first URL embedded in a web evidence string."""
    match = _URL_PATTERN.search(text)
    if match is None:
        return_value = ""
        return return_value
    return_value = match.group(0).rstrip(".,")
    return return_value

def _location_ref(*, role: str, text: str) -> dict[str, str]:
    """Build a structured location reference for downstream slots."""
    ref = {
        "ref_type": "location",
        "role": role,
        "text": text,
    }
    return ref

def _web_task(*, fact_type: str, target: str, original_task: str) -> str:
    """Build the delegated web worker task for a resolved live target."""
    task = (
        "Web-evidence: retrieve current live external evidence "
        f"for fact_type={fact_type}; target={target}; "
        f"original_task={original_task}"
    )
    return task
