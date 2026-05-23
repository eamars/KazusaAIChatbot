"""Prompt-facing RAG evidence formatting and safety checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.utils import text_or_empty

_PUBLIC_RAG_EVIDENCE_KEYS = (
    "answer",
    "third_party_profiles",
    "memory_evidence",
    "recall_evidence",
    "conversation_evidence",
    "external_evidence",
)
_UUID_PATTERN = (
    r"[0-9a-f]{8}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{12}"
)
_FORBIDDEN_PUBLIC_KEYS = {
    "_id",
    "base64_data",
    "conversation_row_id",
    "embedding",
    "global_user_id",
    "platform_message_id",
    "raw_wire_text",
    "source_global_user_id",
    "storage_object_id",
    "storage_url",
}
_FORBIDDEN_TEXT_MARKERS = (
    "[CQ:",
    "raw_wire_text",
    "base64_data",
    "conversation_row_id",
    "global_user_id",
    "platform_message_id",
    "source_global_user_id",
    ";base64,",
    "data:image/",
)
_UUID_RE = re.compile(rf"\b{_UUID_PATTERN}\b", flags=re.IGNORECASE)
_GLOBAL_USER_ID_TEXT_RE = re.compile(
    r"\s*\(?\bglobal_user_id\s*[:=]\s*"
    rf"{_UUID_PATTERN}\)?",
    flags=re.IGNORECASE,
)
_SEPARATOR_SOURCE_ID_RE = re.compile(
    rf"\s*(?:\|\s*|\(\s*|\[\s*){_UUID_PATTERN}(?:\s*\)|\s*\])?",
    flags=re.IGNORECASE,
)
_RAW_URL_ASSIGNMENT_RE = re.compile(r"\burl\s*=", flags=re.IGNORECASE)
_RAW_STORAGE_UTC_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:[\.,]\d{1,6})?"
    r"(?:Z|\+00:00)\b",
    flags=re.IGNORECASE,
)


def sanitize_public_rag_evidence_text(value: object) -> str:
    """Remove source-id text from prompt-facing evidence prose.

    Args:
        value: Evidence prose from a helper summary, result row, or finalizer.

    Returns:
        Text with raw source-id fragments removed while preserving readable
        surrounding prose.
    """

    text = text_or_empty(value)
    if not text:
        return ""

    text = _GLOBAL_USER_ID_TEXT_RE.sub("", text)
    text = _SEPARATOR_SOURCE_ID_RE.sub("", text)
    text = _UUID_RE.sub("[source id omitted]", text)
    return text


def format_evidence_block(
    *,
    conclusion: str,
    evidence_items: list[str],
    uncertainty: str = "none",
) -> str:
    """Build the standard cognition-ready evidence block.

    Args:
        conclusion: Direct answer or no-evidence conclusion.
        evidence_items: Ordered prompt-facing support lines.
        uncertainty: Remaining uncertainty or ``"none"`` when clear.

    Returns:
        A compact evidence block with conclusion first, evidence when present,
        and uncertainty last.
    """

    conclusion_text = sanitize_public_rag_evidence_text(conclusion)
    uncertainty_text = sanitize_public_rag_evidence_text(uncertainty) or "none"
    clean_items = [
        text
        for item in evidence_items
        if (text := sanitize_public_rag_evidence_text(item))
    ]

    lines = [f"Conclusion: {conclusion_text}"]
    if clean_items:
        lines.append("Evidence summary:")
        for item in clean_items:
            lines.append(f"- {item}")
    lines.append(f"Uncertainty: {uncertainty_text}")

    block = "\n".join(lines)
    return block


def ensure_public_rag_evidence_prompt_safe(rag_result: Mapping[str, Any]) -> None:
    """Validate prompt-facing RAG evidence for raw-storage leak markers.

    Args:
        rag_result: Projected RAG result or evidence-like mapping.

    Raises:
        ValueError: If public evidence fields contain raw adapter wire syntax,
            storage ids, binary payloads, embeddings, raw attachment URLs, or
            storage UTC timestamps. ``supervisor_trace`` is intentionally not
            scanned because it is trace-only material.
    """

    violations: list[str] = []
    for key in _PUBLIC_RAG_EVIDENCE_KEYS:
        if key not in rag_result:
            continue
        path = f"rag_result.{key}"
        _collect_prompt_safety_violations(
            rag_result[key],
            path=path,
            violations=violations,
        )

    if violations:
        first_violation = violations[0]
        raise ValueError(
            "prompt-facing RAG evidence contains unsafe material at "
            f"{first_violation}"
        )


def _collect_prompt_safety_violations(
    value: object,
    *,
    path: str,
    violations: list[str],
) -> None:
    """Collect raw-storage markers from a public evidence subtree."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}"
            if key_text in _FORBIDDEN_PUBLIC_KEYS:
                violations.append(item_path)
            if key_text == "url" and not _is_external_evidence_url_path(item_path):
                violations.append(item_path)
            _collect_prompt_safety_violations(
                item,
                path=item_path,
                violations=violations,
            )
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            _collect_prompt_safety_violations(
                item,
                path=item_path,
                violations=violations,
            )
        return

    if isinstance(value, tuple):
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            _collect_prompt_safety_violations(
                item,
                path=item_path,
                violations=violations,
            )
        return

    if isinstance(value, str):
        _collect_text_violations(value, path=path, violations=violations)


def _collect_text_violations(
    value: str,
    *,
    path: str,
    violations: list[str],
) -> None:
    """Collect prompt-unsafe markers from one public evidence string."""

    for marker in _FORBIDDEN_TEXT_MARKERS:
        if marker in value:
            violations.append(path)
            return

    if _RAW_URL_ASSIGNMENT_RE.search(value):
        violations.append(path)
        return

    if _RAW_STORAGE_UTC_RE.search(value):
        violations.append(path)
        return

    if _UUID_RE.search(value) and not _is_allowed_public_uuid_path(path):
        violations.append(path)


def _is_external_evidence_url_path(path: str) -> bool:
    """Return whether a URL key belongs to public external evidence."""

    is_allowed = (
        path.startswith("rag_result.external_evidence[")
        and path.endswith(".url")
    )
    return is_allowed


def _is_allowed_public_uuid_path(path: str) -> bool:
    """Return whether a UUID-like value is allowed as compatibility metadata."""

    is_allowed = path.endswith(".scope_global_user_id")
    return is_allowed
