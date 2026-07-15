"""Internal typed event document contracts for event logging."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from kazusa_ai_chatbot.event_logging.models import (
    CognitionV2EventFields,
    CognitionV2SnapshotSummary,
    EventRefRecord,
    EventSeverity,
)

EventPayloadRecord = dict[str, Any]


class EventScopeRecord(TypedDict):
    """Persisted event scope with a private channel reference."""

    platform: str
    platform_channel_ref: str
    channel_type: str


class EventErrorRecord(TypedDict):
    """Sanitized error metadata stored in the event stream."""

    error_class: str
    error_preview: str
    stack_fingerprint: str
    recovered: bool


class EventLogEventDoc(TypedDict):
    """Canonical append-only event stream document."""

    event_id: str
    event_family: str
    event_type: str
    component: str
    severity: EventSeverity
    status: str
    correlation_id: str
    run_id: str
    trigger_id: str
    attempt_id: str
    occurred_at: str
    created_at: str
    expires_at: str
    duration_ms: int | None
    scope: EventScopeRecord
    metrics: dict[str, int | float | bool | str]
    labels: dict[str, str]
    refs: list[EventRefRecord]
    warning_codes: list[str]
    error: EventErrorRecord
    payload: EventPayloadRecord
    cognition_v2: NotRequired[CognitionV2EventFields]


class EventLogSnapshotDoc(TypedDict):
    """Deterministic aggregate snapshot for later operator review."""

    snapshot_id: str
    snapshot_kind: str
    window_start: str
    window_end: str
    generated_at: str
    expires_at: str
    source_counts: dict[str, int]
    semantic_descriptors: dict[str, str]
    findings: list[dict[str, str]]
    source_event_refs: list[str]
    cognition_v2_summary: NotRequired[CognitionV2SnapshotSummary]
