"""Typed contracts for permissioned proactive output records."""

from __future__ import annotations

from typing import Literal, TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    OutputMode,
    TriggerSource,
    Visibility,
)

__all__ = [
    "QuietHoursPolicy",
    "ProactiveOutboxRecord",
    "ProactiveOutboxStateError",
    "ProactiveOutboxStatus",
    "ProactivePermissionRecord",
    "ProactivePolicyDecision",
    "ProactivePreviewRecord",
    "ProactiveSendAuditRecord",
]


class QuietHoursPolicy(TypedDict):
    """Local quiet-hour window applied before proactive delivery."""

    enabled: bool
    start_local_time: str
    end_local_time: str


class ProactivePermissionRecord(TypedDict):
    """Explicit permission required before a proactive preview can be sent."""

    permission_id: str
    platform: str
    platform_channel_id: str
    channel_type: str
    target_global_user_id: str
    target_platform_user_id: str
    allowed_trigger_sources: list[TriggerSource]
    allowed_output_modes: list[OutputMode]
    quiet_hours: QuietHoursPolicy
    expires_at: str
    enabled: bool
    created_at: str
    audit_reason: str


class ProactivePreviewRecord(TypedDict):
    """Approved candidate text before it is accepted into an outbox."""

    preview_id: str
    episode_id: str
    trigger_source: TriggerSource
    output_mode: OutputMode
    visibility: Visibility
    platform: str
    platform_channel_id: str
    channel_type: str
    target_global_user_id: str
    target_platform_user_id: str
    preview_text: str
    idempotency_key: str
    created_at: str
    audit_reason: str


class ProactivePolicyDecision(TypedDict):
    """Deterministic allow or deny result for one proactive preview."""

    allowed: bool
    reason: str


ProactiveOutboxStatus = Literal[
    "dry_run",
    "ready",
    "sent",
    "denied",
    "failed",
    "cancelled",
]


class ProactiveOutboxRecord(TypedDict):
    """Auditable outbox row for a proactive preview send attempt."""

    outbox_id: str
    preview_id: str
    permission_id: str
    idempotency_key: str
    platform: str
    platform_channel_id: str
    channel_type: str
    target_global_user_id: str
    target_platform_user_id: str
    preview_text: str
    status: ProactiveOutboxStatus
    created_at: str
    updated_at: str
    transport_attempt_count: int
    last_failure_reason: str
    sent_at: str
    platform_message_id: str
    delivery_adapter: str
    origin_kind: str


class ProactiveSendAuditRecord(TypedDict):
    """Audit event emitted for proactive outbox state changes."""

    audit_id: str
    outbox_id: str
    event_type: str
    created_at: str
    reason: str
    platform_message_id: str
    delivery_adapter: str


class ProactiveOutboxStateError(ValueError):
    """Raised when an outbox state transition is not allowed."""
