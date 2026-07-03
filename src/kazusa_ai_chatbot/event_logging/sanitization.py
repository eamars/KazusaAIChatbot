"""Sanitizers for prompt-safe event logging documents."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

from kazusa_ai_chatbot.event_logging.models import EventScopeInput
from kazusa_ai_chatbot.event_logging.schemas import EventScopeRecord

_CHANNEL_REF_SALT = "kazusa-event-log-scope-v1"
_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_MAX_SHORT_TEXT_CHARS = 300
_MAX_LIST_ITEMS = 25

_DENIED_FIELD_NAMES = frozenset(
    {
        "human_" + "prompt",
        "system_" + "prompt",
        "raw_" + "output",
        "base64_" + "data",
        "embed" + "ding",
        "api_" + "key",
        "shared_" + "secret",
        "message_" + "envelope",
    }
)


def sanitize_short_text(value: object, *, limit: int = _MAX_SHORT_TEXT_CHARS) -> str:
    """Return a compact single-field string safe for event-log storage.

    Args:
        value: Input value from a runtime caller or exception.
        limit: Maximum returned character count.

    Returns:
        Sanitized string with control bytes removed and length capped.
    """

    text = str(value or "")
    normalized_text = _CONTROL_PATTERN.sub(" ", text).strip()
    if len(normalized_text) > limit:
        clipped_text = normalized_text[:limit].rstrip()
        normalized_text = f"{clipped_text}..."
    return normalized_text


def sanitize_string_list(values: Sequence[object]) -> list[str]:
    """Return capped sanitized strings from a sequence-like caller value."""

    sanitized_values = [
        sanitize_short_text(value)
        for value in list(values)[:_MAX_LIST_ITEMS]
    ]
    return sanitized_values


def build_scope_record(scope: EventScopeInput | None) -> EventScopeRecord:
    """Project caller scope into a persisted scope without raw channel IDs.

    Args:
        scope: Optional caller-provided platform scope.

    Returns:
        Persisted event scope with a stable private channel reference.
    """

    input_scope = scope or {}
    platform = sanitize_short_text(input_scope.get("platform", ""), limit=80)
    channel_value = sanitize_short_text(
        input_scope.get("platform_channel_id", ""),
        limit=160,
    )
    channel_type = sanitize_short_text(input_scope.get("channel_type", ""), limit=40)
    channel_ref = ""
    if channel_value:
        digest_input = f"{_CHANNEL_REF_SALT}:{platform}:{channel_value}"
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        channel_ref = f"ch_{digest[:32]}"
    scope_record = EventScopeRecord(
        platform=platform,
        platform_channel_ref=channel_ref,
        channel_type=channel_type,
    )
    return scope_record


def unsafe_field_paths(value: object, *, prefix: str = "") -> list[str]:
    """Return denied field paths found in an event document candidate.

    Args:
        value: Nested mapping/list/scalar value to inspect.
        prefix: Internal recursion prefix.

    Returns:
        List of denied key paths. Empty means no denied keys were found.
    """

    if isinstance(value, Mapping):
        paths: list[str] = []
        for raw_key, raw_child in value.items():
            key = str(raw_key)
            child_prefix = key if not prefix else f"{prefix}.{key}"
            if key in _DENIED_FIELD_NAMES:
                paths.append(child_prefix)
                continue
            paths.extend(unsafe_field_paths(raw_child, prefix=child_prefix))
        return paths

    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value[:_MAX_LIST_ITEMS]):
            child_prefix = f"{prefix}[{index}]"
            paths.extend(unsafe_field_paths(child, prefix=child_prefix))
        return paths

    return_value: list[str] = []
    return return_value


def sanitized_failure_reason(exc: BaseException) -> str:
    """Return sanitized exception text for a failed telemetry write."""

    reason = sanitize_short_text(f"{type(exc).__name__}: {exc}")
    return reason


def sanitized_rejection_reason(paths: Sequence[str]) -> str:
    """Return a compact rejection reason for denied field paths."""

    preview_paths = ", ".join(paths[:5])
    reason = sanitize_short_text(f"unsafe fields: {preview_paths}")
    return reason
