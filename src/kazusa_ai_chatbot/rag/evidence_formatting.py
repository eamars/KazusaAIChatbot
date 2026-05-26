"""Prompt-facing RAG evidence formatting and safety checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

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
    "evidence_time=",
    "global_user_id",
    "platform_message_id",
    "source_global_user_id",
    ";base64,",
    "data:image/",
)
_MAX_PUBLIC_EXTERNAL_URL_CHARS = 2048
_CONTROL_TEXT_RE = re.compile(r"[\x00-\x1f\x7f]")
_DROP_PUBLIC_VALUE = object()
_HTTP_URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
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
_READABLE_MESSAGE_ID_TEXT_RE = re.compile(
    r"(?:\bmessage\s*id\s*(?:is|=|:)?\s*\d{5,}\b)"
    r"|(?:消息\s*(?:ID|id|编号)\s*(?:为|是|=|:)?\s*\d{5,})",
    flags=re.IGNORECASE,
)
_INTERNAL_PUBLIC_LABEL_REPLACEMENTS = (
    ("user_memory_evidence_agent", "用户记忆证据"),
    ("persistent_memory_search_agent", "持久记忆证据"),
    ("conversation_evidence_agent", "对话证据"),
    ("memory_evidence_agent", "记忆证据"),
    ("person_context_agent", "人物上下文"),
    ("user_lookup_agent", "用户识别"),
    ("user_profile_agent", "用户画像"),
    ("user_list_agent", "用户列表"),
    ("recall_agent", "召回证据"),
    ("user_memory_units", "用户记忆"),
    ("conversation_evidence", "对话证据"),
    ("memory_evidence", "记忆证据"),
    ("recall_evidence", "召回证据"),
    ("durable_commitment", "持续承诺"),
    ("active_episode_agreement", "当前对话约定"),
    ("exact_agreement_history", "历史约定"),
    ("episode_position", "当前对话进展"),
)
_INTERNAL_PREFIX_REPLACEMENTS = (
    (
        re.compile(r"\brecall\s*:\s*", flags=re.IGNORECASE),
        "召回候选：",
    ),
    (
        re.compile(r"\bmemory\s*:\s*", flags=re.IGNORECASE),
        "记忆候选：",
    ),
    (
        re.compile(r"\bconversation\s*:\s*", flags=re.IGNORECASE),
        "对话候选：",
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
    text = _GLOBAL_USER_ID_TEXT_RE.sub("", text)
    text = _SOURCE_ID_TEXT_RE.sub("[来源标识已省略]", text)
    text = _READABLE_MESSAGE_ID_TEXT_RE.sub("消息记录", text)
    text = _SEPARATOR_SOURCE_ID_RE.sub("", text)
    text = _UUID_RE.sub("[来源标识已省略]", text)
    text = _SOURCE_ID_LABEL_RE.sub("来源标识", text)
    return text


def format_evidence_block(
    *,
    conclusion: str,
    evidence_items: list[str],
    uncertainty: str = "无",
) -> str:
    """Build the standard cognition-ready evidence block.

    Args:
        conclusion: Direct answer or no-evidence conclusion.
        evidence_items: Ordered prompt-facing support lines.
        uncertainty: Remaining uncertainty or ``"无"`` when clear.

    Returns:
        A compact evidence block with conclusion first, evidence when present,
        and uncertainty last.
    """

    conclusion_text = sanitize_public_rag_evidence_text(conclusion)
    uncertainty_text = sanitize_public_rag_evidence_text(uncertainty) or "无"
    clean_items = [
        text
        for item in evidence_items
        if (text := sanitize_public_rag_evidence_text(item))
    ]

    lines = [f"结论：{conclusion_text}"]
    if clean_items:
        lines.append("上下文：")
        for item in clean_items:
            lines.append(f"- {item}")
    lines.append(f"不确定性：{uncertainty_text}")

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

    violations = public_rag_evidence_prompt_safety_violations(rag_result)

    if violations:
        first_violation = violations[0]
        raise ValueError(
            "prompt-facing RAG evidence contains unsafe material at "
            f"{first_violation}"
        )


def public_rag_evidence_prompt_safety_violations(
    rag_result: Mapping[str, Any],
) -> list[str]:
    """Return unsafe public evidence paths without raising.

    Args:
        rag_result: Projected RAG result or evidence-like mapping.

    Returns:
        Prompt-facing evidence paths that still contain raw storage, adapter,
        binary, or malformed external URL material.
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
    return violations


def recover_public_rag_evidence_prompt_safe(
    rag_result: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Recover prompt-facing RAG evidence without raising on unsafe leaves.

    Args:
        rag_result: Projected RAG result or evidence-like mapping.

    Returns:
        A recovered RAG result and compact incident paths. Recoverable unsafe
        prose is stripped or dropped, malformed optional URLs are blanked, and
        unrecoverable public evidence falls back to empty public RAG fields.
    """

    recovered = dict(rag_result)
    incidents: list[str] = []
    for key in _PUBLIC_RAG_EVIDENCE_KEYS:
        if key not in rag_result:
            continue
        path = f"rag_result.{key}"
        recovered_value = _recover_prompt_safe_value(
            rag_result[key],
            path=path,
            incidents=incidents,
        )
        if recovered_value is _DROP_PUBLIC_VALUE:
            recovered[key] = _empty_public_rag_evidence_value(key)
            continue
        recovered[key] = recovered_value

    remaining_violations = public_rag_evidence_prompt_safety_violations(
        recovered,
    )
    if remaining_violations:
        incidents.extend(
            f"emptied_unrecoverable:{path}"
            for path in remaining_violations
        )
        for key in _PUBLIC_RAG_EVIDENCE_KEYS:
            if key in recovered:
                recovered[key] = _empty_public_rag_evidence_value(key)

    return_value = (recovered, incidents)
    return return_value


def _empty_public_rag_evidence_value(key: str) -> str | list[Any]:
    """Return the empty value for one public RAG evidence field."""

    if key == "answer":
        return_value: str | list[Any] = ""
        return return_value
    return_value = []
    return return_value


def _recover_prompt_safe_value(
    value: object,
    *,
    path: str,
    incidents: list[str],
) -> object:
    """Return a best-effort prompt-safe value for one public evidence leaf."""

    if isinstance(value, Mapping):
        recovered_mapping: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}"
            if key_text in _FORBIDDEN_PUBLIC_KEYS:
                incidents.append(f"dropped_key:{item_path}")
                continue
            if key_text == "url":
                if _is_external_evidence_url_path(item_path):
                    recovered_url = _recover_external_evidence_url(
                        item,
                        path=item_path,
                        incidents=incidents,
                    )
                    recovered_mapping[key_text] = recovered_url
                    continue
                incidents.append(f"dropped_url_key:{item_path}")
                continue
            recovered_item = _recover_prompt_safe_value(
                item,
                path=item_path,
                incidents=incidents,
            )
            if recovered_item is not _DROP_PUBLIC_VALUE:
                recovered_mapping[key_text] = recovered_item
        return recovered_mapping

    if isinstance(value, list):
        recovered_items = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            recovered_item = _recover_prompt_safe_value(
                item,
                path=item_path,
                incidents=incidents,
            )
            if _recovered_value_has_content(recovered_item):
                recovered_items.append(recovered_item)
        return recovered_items

    if isinstance(value, tuple):
        recovered_items = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            recovered_item = _recover_prompt_safe_value(
                item,
                path=item_path,
                incidents=incidents,
            )
            if _recovered_value_has_content(recovered_item):
                recovered_items.append(recovered_item)
        return recovered_items

    if isinstance(value, str):
        recovered_text = _recover_prompt_safe_text(
            value,
            path=path,
            incidents=incidents,
        )
        return recovered_text

    return value


def _recovered_value_has_content(value: object) -> bool:
    """Return whether a recovered evidence value should remain public."""

    if value is _DROP_PUBLIC_VALUE:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, Mapping):
        return any(
            _recovered_value_has_content(item)
            for item in value.values()
        )
    if isinstance(value, list):
        return any(
            _recovered_value_has_content(item)
            for item in value
        )
    return True


def _recover_external_evidence_url(
    value: object,
    *,
    path: str,
    incidents: list[str],
) -> str:
    """Return a validated public external URL, or blank it when malformed."""

    url = text_or_empty(value).strip()
    if not url:
        return ""
    if _external_evidence_url_violation(url):
        incidents.append(f"blanked_url:{path}")
        return ""
    return url


def _recover_prompt_safe_text(
    value: str,
    *,
    path: str,
    incidents: list[str],
) -> object:
    """Sanitize prompt-facing prose and drop unsafe residual lines."""

    violations: list[str] = []
    _collect_text_violations(value, path=path, violations=violations)
    if not violations:
        return value

    text = sanitize_public_rag_evidence_text(value)
    sanitized_violations: list[str] = []
    _collect_text_violations(
        text,
        path=path,
        violations=sanitized_violations,
    )
    if not sanitized_violations:
        if text != value:
            incidents.append(f"sanitized_text:{path}")
        return text

    recovered_lines = []
    for line in text.splitlines():
        line_violations: list[str] = []
        _collect_text_violations(
            line,
            path=path,
            violations=line_violations,
        )
        if line_violations:
            incidents.append(f"dropped_text_line:{path}")
            continue
        recovered_lines.append(line)
    recovered_text = "\n".join(recovered_lines).strip()

    remaining_violations: list[str] = []
    _collect_text_violations(
        recovered_text,
        path=path,
        violations=remaining_violations,
    )
    if remaining_violations:
        incidents.append(f"dropped_text:{path}")
        return _DROP_PUBLIC_VALUE
    if not recovered_text:
        incidents.append(f"dropped_text:{path}")
        return _DROP_PUBLIC_VALUE
    return recovered_text


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
        if _is_external_evidence_url_path(path):
            if _external_evidence_url_violation(value):
                violations.append(path)
            return
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

    value_without_public_urls = _remove_valid_public_urls(value)

    if _RAW_URL_ASSIGNMENT_RE.search(value_without_public_urls):
        violations.append(path)
        return

    if _RAW_STORAGE_UTC_RE.search(value):
        violations.append(path)
        return

    if _READABLE_MESSAGE_ID_TEXT_RE.search(value):
        violations.append(path)
        return

    if (
        _UUID_RE.search(value_without_public_urls)
        and not _is_allowed_public_uuid_path(path)
    ):
        violations.append(path)


def _remove_valid_public_urls(value: str) -> str:
    """Remove valid public URLs before scanning prose-only unsafe markers."""

    def replace_match(match: re.Match[str]) -> str:
        candidate = match.group(0)
        url = candidate.rstrip(".,;:!?)]}")
        suffix = candidate[len(url):]
        if url and not _external_evidence_url_violation(url):
            return suffix
        return candidate

    cleaned_text = _HTTP_URL_RE.sub(replace_match, value)
    return cleaned_text


def _is_external_evidence_url_path(path: str) -> bool:
    """Return whether a URL key belongs to public external evidence."""

    is_allowed = (
        path.startswith("rag_result.external_evidence[")
        and path.endswith(".url")
    )
    return is_allowed


def _external_evidence_url_violation(value: str) -> bool:
    """Return whether a public external URL field is malformed."""

    url = text_or_empty(value).strip()
    if not url:
        return False
    if len(url) > _MAX_PUBLIC_EXTERNAL_URL_CHARS:
        return True
    if _CONTROL_TEXT_RE.search(url):
        return True
    if any(character.isspace() for character in url):
        return True
    for marker in _FORBIDDEN_TEXT_MARKERS:
        if marker in url:
            return True

    try:
        parsed = urlsplit(url)
    except ValueError:
        return True
    if parsed.scheme not in {"http", "https"}:
        return True
    if not parsed.netloc:
        return True
    return False


def _is_allowed_public_uuid_path(path: str) -> bool:
    """Return whether a UUID-like value is allowed as compatibility metadata."""

    is_allowed = path.endswith(".scope_global_user_id")
    return is_allowed
