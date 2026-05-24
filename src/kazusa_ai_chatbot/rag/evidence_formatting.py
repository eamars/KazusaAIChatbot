"""Prompt-facing RAG evidence formatting and safety checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm
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
    r"\s*\(?\bglobal_user_id\s*(?::|=)?\s*"
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
_SOURCE_ID_PREFIX_RE = re.compile(
    r"\b(?:"
    r"platform_message_id|conversation_row_id|source_global_user_id|"
    r"global_user_id"
    r")\s*:\s*[^:;\n]+:\s*",
    flags=re.IGNORECASE,
)
_SOURCE_ID_TEXT_RE = re.compile(
    r"\b(?:"
    r"platform_message_id|conversation_row_id|source_global_user_id|"
    r"global_user_id"
    r")\b\s*(?:\u4e3a|is|=|:)?\s*[0-9a-f][0-9a-f-]{5,40}",
    flags=re.IGNORECASE,
)
_SOURCE_ID_LABEL_RE = re.compile(
    r"\b(?:"
    r"platform_message_id|conversation_row_id|source_global_user_id|"
    r"global_user_id"
    r")\b",
    flags=re.IGNORECASE,
)
_INTERNAL_PUBLIC_LABEL_REPLACEMENTS = (
    ("user_memory_evidence_agent", "user memory evidence"),
    ("persistent_memory_search_agent", "durable memory evidence"),
    ("conversation_evidence_agent", "conversation evidence"),
    ("memory_evidence_agent", "memory evidence"),
    ("person_context_agent", "person context"),
    ("user_lookup_agent", "user lookup"),
    ("user_profile_agent", "user profile"),
    ("user_list_agent", "user list"),
    ("recall_agent", "recall evidence"),
    ("user_memory_units", "user memory units"),
    ("conversation_evidence", "conversation evidence"),
    ("memory_evidence", "memory evidence"),
    ("recall_evidence", "recall evidence"),
    ("durable_commitment", "durable commitment"),
    ("active_episode_agreement", "active episode agreement"),
    ("exact_agreement_history", "exact agreement history"),
    ("episode_position", "episode position"),
)
_INTERNAL_PREFIX_REPLACEMENTS = (
    (
        re.compile(r"\brecall\s*:\s*", flags=re.IGNORECASE),
        "Recall candidate: ",
    ),
    (
        re.compile(r"\bmemory\s*:\s*", flags=re.IGNORECASE),
        "Memory candidate: ",
    ),
    (
        re.compile(r"\bconversation\s*:\s*", flags=re.IGNORECASE),
        "Conversation candidate: ",
    ),
)


def _format_storage_utc_match(match: re.Match[str]) -> str:
    """Render an embedded storage UTC timestamp for public evidence text.

    Args:
        match: Regex match containing one storage UTC timestamp.

    Returns:
        Configured-local wall-clock text, or the original timestamp if the
        timestamp cannot be projected.
    """

    raw_timestamp = match.group(0)
    formatted_timestamp = format_storage_utc_for_llm(raw_timestamp)
    if formatted_timestamp:
        return formatted_timestamp
    return raw_timestamp


def _replace_internal_public_labels(text: str) -> str:
    """Replace storage-facing labels with reader-facing evidence labels.

    Args:
        text: Public evidence prose that may contain internal source labels.

    Returns:
        Evidence prose with known internal labels rendered as readable source
        descriptions.
    """

    cleaned_text = text
    for prefix_re, replacement in _INTERNAL_PREFIX_REPLACEMENTS:
        cleaned_text = prefix_re.sub(replacement, cleaned_text)
    for marker, replacement in _INTERNAL_PUBLIC_LABEL_REPLACEMENTS:
        marker_re = re.compile(
            rf"\b{re.escape(marker)}\b",
            flags=re.IGNORECASE,
        )
        cleaned_text = marker_re.sub(replacement, cleaned_text)
    return cleaned_text


def sanitize_public_rag_evidence_text(value: object) -> str:
    """Remove source internals from prompt-facing evidence prose.

    Args:
        value: Evidence prose from a helper summary, result row, or finalizer.

    Returns:
        Text with raw source-id fragments, storage timestamps, and
        implementation labels removed or rendered for downstream prompts.
    """

    text = text_or_empty(value)
    if not text:
        return ""

    text = _RAW_STORAGE_UTC_RE.sub(_format_storage_utc_match, text)
    text = _replace_internal_public_labels(text)
    text = _SOURCE_ID_PREFIX_RE.sub("", text)
    text = _SOURCE_ID_TEXT_RE.sub("[source id omitted]", text)
    text = _GLOBAL_USER_ID_TEXT_RE.sub("", text)
    text = _SEPARATOR_SOURCE_ID_RE.sub("", text)
    text = _UUID_RE.sub("[source id omitted]", text)
    text = _SOURCE_ID_LABEL_RE.sub("source id", text)
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
