"""Deterministic redaction for console API responses and log views."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
    CognitionRunObservationV1,
)

REDACTED = "[redacted]"
MAX_SAFE_TEXT_CHARS = 800
CHARACTER_OPERATIONAL_STATE_VIEW_SCHEMA = (
    "character_operational_state_view.v1"
)
CHARACTER_OPERATIONAL_CONTEXT_SCHEMA = "character_operational_context.v1"
RELATIONSHIP_OPERATIONAL_CONTEXT_SCHEMA = (
    "relationship_operational_context.v1"
)
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
_CHARACTER_AFFECT_FIELDS = (
    "emotion_id",
    "intensity",
    "phase",
    "trend",
    "root_kind",
    "cause_class",
    "freshness",
)
_CHARACTER_PRESSURE_FIELDS = (
    "kind",
    "salience",
    "lifecycle",
    "cause_class",
    "freshness",
)
_RELATIONSHIP_CAUSAL_FIELDS = (
    "entity_kind",
    "semantic_summary",
    "salience",
    "lifecycle",
    "freshness",
)
_RELATIONSHIP_AFFECT_FIELDS = (
    "emotion_id",
    "intensity",
    "phase",
    "trend",
    "freshness",
)
_RELATIONSHIP_AXIS_FIELDS = (
    "familiarity",
    "positive_regard",
    "trust",
    "attachment",
    "desired_closeness",
    "perceived_closeness",
    "care",
    "boundary_safety",
    "exclusivity",
    "unresolved_injury",
    "salience",
)
_STYLE_SOURCE_NAMES = ("user", "group_channel")
_STYLE_ROLES = ("relevance", "cognition", "surface")
_STYLE_STATUSES = frozenset({"active", "empty", "missing", "failed"})


def redact_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively redacted copy of a mapping."""

    if source.get("schema_version") == "cognition_run_observation.v1":
        try:
            observation = CognitionRunObservationV1.model_validate(source)
        except ValidationError:
            pass
        else:
            return observation.model_dump(mode="json")

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


def redact_character_operational_state_view(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Allowlist the public persisted or effective character posture view."""

    if source.get("schema_version") != CHARACTER_OPERATIONAL_STATE_VIEW_SCHEMA:
        return {}
    result = _public_text_fields(
        source,
        (
            "schema_version",
            "source_updated_at",
            "effective_at",
            "source_digest",
            "view_digest",
        ),
        maximum=160,
    )
    result["affect"] = _public_rows(
        source.get("affect"),
        fields=_CHARACTER_AFFECT_FIELDS,
        limit=21,
    )
    result["pressures"] = _public_rows(
        source.get("pressures"),
        fields=_CHARACTER_PRESSURE_FIELDS,
        limit=8,
    )
    return result


def redact_character_operational_context(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Allowlist one exact public character-posture consumer selection."""

    if source.get("schema_version") != CHARACTER_OPERATIONAL_CONTEXT_SCHEMA:
        return {}
    result = _public_text_fields(
        source,
        (
            "schema_version",
            "source_updated_at",
            "effective_at",
            "view_digest",
            "consumer_role",
            "context_digest",
        ),
        maximum=160,
    )
    result["affect"] = _public_rows(
        source.get("affect"),
        fields=_CHARACTER_AFFECT_FIELDS,
        limit=3,
    )
    result["pressures"] = _public_rows(
        source.get("pressures"),
        fields=_CHARACTER_PRESSURE_FIELDS,
        limit=4,
    )
    return result


def redact_operational_relationship_context(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Allowlist current-user relationship context without durable identity."""

    if (
        source.get("schema_version") not in {
            RELATIONSHIP_OPERATIONAL_CONTEXT_SCHEMA,
            None,
        }
        and "handle" not in source
    ):
        return {}
    raw_axes = source.get("axes")
    axes: dict[str, str] = {}
    if isinstance(raw_axes, Mapping):
        for field_name in _RELATIONSHIP_AXIS_FIELDS:
            value = _public_text(raw_axes.get(field_name), maximum=80)
            if value:
                axes[field_name] = value
    result: dict[str, Any] = {"axes": axes}
    result["causal_context"] = _public_rows(
        source.get("causal_context"),
        fields=_RELATIONSHIP_CAUSAL_FIELDS,
        limit=2,
        field_maximums={"semantic_summary": 160},
    )
    result["affect"] = _public_rows(
        source.get("affect"),
        fields=_RELATIONSHIP_AFFECT_FIELDS,
        limit=2,
    )
    result.update(_public_text_fields(
        source,
        ("relationship_freshness", "evidence_freshness"),
        maximum=80,
    ))
    return result


def redact_interaction_style_projections(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Allowlist source-labelled relevance, cognition, and surface style."""

    projections: dict[str, Any] = {}
    for consumer_role in _STYLE_ROLES:
        role_source = source.get(consumer_role)
        if not isinstance(role_source, Mapping):
            continue
        projected_sources: dict[str, Any] = {}
        for source_name in _STYLE_SOURCE_NAMES:
            raw_projection = role_source.get(source_name)
            if not isinstance(raw_projection, Mapping):
                continue
            projection = _redact_style_source(
                raw_projection,
                consumer_role=consumer_role,
            )
            if projection:
                projected_sources[source_name] = projection
        if projected_sources:
            projections[consumer_role] = projected_sources
    return projections


def _public_operational_receipt(source: Mapping[str, Any]) -> dict[str, Any]:
    """Expose receipt health while excluding episode and lease identifiers."""

    result = _public_text_fields(
        source,
        (
            "status",
            "base_updated_at",
            "committed_updated_at",
            "registered_at",
            "completed_at",
            "error_code",
        ),
        maximum=160,
    )
    durable = source.get("durable")
    if isinstance(durable, bool):
        result["durable"] = durable
    attempt_count = _public_non_negative_int(source.get("attempt_count"))
    if attempt_count is not None:
        result["attempt_count"] = attempt_count
    return result


def _redact_style_source(
    source: Mapping[str, Any],
    *,
    consumer_role: str,
) -> dict[str, Any]:
    """Project one source-role style view without document provenance."""

    status = _public_text(source.get("status"), maximum=40)
    if status not in _STYLE_STATUSES:
        return {}
    result: dict[str, Any] = {"status": status}
    revision = _public_non_negative_int(source.get("revision"))
    if revision is not None:
        result["revision"] = revision
    if consumer_role == "surface":
        overlay = source.get("overlay")
        if not isinstance(overlay, Mapping):
            return result
        projected_overlay = {
            field_name: _public_style_guidance(overlay.get(field_name), limit=8)
            for field_name in (
                "speech_guidelines",
                "social_guidelines",
                "pacing_guidelines",
                "engagement_guidelines",
            )
        }
        confidence = _public_text(overlay.get("confidence"), maximum=80)
        if confidence:
            projected_overlay["confidence"] = confidence
        result["overlay"] = projected_overlay
        return result

    guideline_fields = (
        ("engagement_guidelines",)
        if consumer_role == "relevance"
        else ("social_guidelines", "engagement_guidelines")
    )
    for field_name in guideline_fields:
        result[field_name] = _public_style_guidance(
            source.get(field_name),
            limit=3,
        )
    confidence = _public_text(source.get("confidence"), maximum=80)
    if confidence:
        result["confidence"] = confidence
    return result


def _public_rows(
    value: Any,
    *,
    fields: tuple[str, ...],
    limit: int,
    field_maximums: Mapping[str, int] | None = None,
) -> list[dict[str, str]]:
    """Copy only bounded text fields from a public semantic row collection."""

    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    rows: list[dict[str, str]] = []
    for raw_row in value[:limit]:
        if not isinstance(raw_row, Mapping):
            continue
        row: dict[str, str] = {}
        for field_name in fields:
            maximum = (
                field_maximums.get(field_name, 80)
                if field_maximums is not None
                else 80
            )
            text = _public_text(raw_row.get(field_name), maximum=maximum)
            if text:
                row[field_name] = text
        if row:
            rows.append(row)
    return rows


def _public_text_fields(
    source: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    maximum: int,
) -> dict[str, str]:
    """Copy declared bounded text fields from one public projection."""

    result: dict[str, str] = {}
    for field_name in fields:
        text = _public_text(source.get(field_name), maximum=maximum)
        if text:
            result[field_name] = text
    return result


def _public_style_guidance(value: Any, *, limit: int) -> list[str]:
    """Return bounded redacted learned-guidance strings."""

    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    guidance: list[str] = []
    for item in value[:limit]:
        text = _public_text(item, maximum=240)
        if text:
            guidance.append(text)
    return guidance


def _public_text(value: Any, *, maximum: int) -> str:
    """Return bounded text only; public projections never coerce objects."""

    if not isinstance(value, str):
        return ""
    text = redact_text(value.strip())
    return text[:maximum]


def _public_non_negative_int(value: Any) -> int | None:
    """Return one bounded non-negative counter without boolean coercion."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return min(value, 1_000_000)


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
