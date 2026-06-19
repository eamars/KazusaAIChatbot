"""Deterministic redaction for console API responses and log views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


REDACTED = "[redacted]"
MAX_SAFE_TEXT_CHARS = 800
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "token",
    "secret",
    "password",
    "prompt",
    "embedding",
    "env",
    "raw_message",
    "raw_output",
    "message_envelope",
    "message_text",
    "body_text",
    "raw_wire_text",
    "base64",
)
SECRET_TEXT_PATTERNS = (
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"(api[_-]?key|token|secret|password)=\S+", re.IGNORECASE),
    re.compile(r"raw_message=\S+", re.IGNORECASE),
)


def redact_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively redacted copy of a mapping."""

    redacted: dict[str, Any] = {}
    for key, value in source.items():
        if _is_sensitive_key(key):
            continue
        else:
            redacted[key] = redact_value(value)
    return redacted


def redact_value(value: Any) -> Any:
    """Redact one JSON-like value while preserving safe structure."""

    if isinstance(value, Mapping):
        redacted = redact_mapping(value)
        return redacted
    if isinstance(value, str):
        redacted_text = redact_text(value)
        return redacted_text
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        redacted_items = [redact_value(item) for item in value[:50]]
        return redacted_items
    return value


def redact_text(text: str) -> str:
    """Remove secret-bearing or unbounded text from a log/event string."""

    if _contains_prompt_or_raw_message(text):
        return REDACTED

    redacted_text = text
    for pattern in SECRET_TEXT_PATTERNS:
        redacted_text = pattern.sub(REDACTED, redacted_text)

    if len(redacted_text) > MAX_SAFE_TEXT_CHARS:
        redacted_text = f"{redacted_text[:MAX_SAFE_TEXT_CHARS]}..."
    return redacted_text


def _is_sensitive_key(key: str) -> bool:
    """Return whether a field name is denied in operator responses."""

    normalized_key = key.lower().replace("-", "_")
    is_sensitive = any(part in normalized_key for part in SENSITIVE_KEY_PARTS)
    return is_sensitive


def _contains_prompt_or_raw_message(text: str) -> bool:
    """Return whether text carries prompt or raw-message content."""

    normalized_text = text.lower()
    contains_sensitive_text = "prompt" in normalized_text or "raw_message" in normalized_text
    return contains_sensitive_text
