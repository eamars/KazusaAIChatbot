"""Public type contracts and semantic labels for event logging."""

from __future__ import annotations

from typing import Literal, TypedDict

EventSeverity = Literal["debug", "info", "warning", "error", "critical"]

EVENT_SEVERITIES: frozenset[str] = frozenset(
    {"debug", "info", "warning", "error", "critical"}
)


class EventScopeInput(TypedDict, total=False):
    """Caller-supplied runtime scope before private channel ref projection."""

    platform: str
    platform_channel_id: str
    channel_type: str


class SelfCognitionBudget(TypedDict):
    """LLM-call budget counters for one self-cognition case."""

    rag_calls: int
    cognition_calls: int
    dialog_calls: int
    topic_limit: int


class EventRefRecord(TypedDict):
    """Reference to an existing durable runtime artifact."""

    ref_type: str
    ref_id: str


class EventLogWriteResult(TypedDict):
    """Best-effort event write result returned to runtime callers."""

    accepted: bool
    event_id: str
    status: Literal["recorded", "rejected", "failed"]
    reason: str


def reflection_health_label(*, failed_count: int, succeeded_count: int) -> str:
    """Return a compact reflection health label for operator summaries.

    Args:
        failed_count: Number of failed reflection events in the window.
        succeeded_count: Number of successful reflection events in the window.

    Returns:
        A stable health label suitable for prompt-safe aggregate snapshots.
    """

    if failed_count == 0 and succeeded_count > 0:
        label = "healthy"
    elif failed_count > 0 and succeeded_count > 0:
        label = "mixed"
    elif failed_count > 0:
        label = "failing"
    else:
        label = "inactive"
    return label


def self_cognition_liveness_label(*, run_count: int, dispatch_count: int) -> str:
    """Return a compact liveness label for self-cognition activity.

    Args:
        run_count: Number of self-cognition events in the window.
        dispatch_count: Number of accepted dispatch results in the window.

    Returns:
        A stable liveness label for aggregate operator payloads.
    """

    if run_count == 0:
        label = "inactive"
    elif dispatch_count > 0:
        label = "active_with_handoff"
    else:
        label = "active_internal_only"
    return label


def llm_parse_stability_label(*, failed_count: int, repaired_count: int) -> str:
    """Return a compact parse-stability label for model-stage telemetry.

    Args:
        failed_count: Number of failed parse or contract events.
        repaired_count: Number of events that needed deterministic repair.

    Returns:
        A stable stability label for aggregate status and snapshots.
    """

    if failed_count > 0:
        label = "degraded"
    elif repaired_count > 0:
        label = "watch"
    else:
        label = "stable"
    return label


def worker_error_level_label(*, error_count: int) -> str:
    """Return a compact worker error level for a summary window.

    Args:
        error_count: Number of runtime-error events for worker components.

    Returns:
        A stable severity label for aggregate status and snapshots.
    """

    if error_count == 0:
        label = "none"
    elif error_count < 3:
        label = "low"
    elif error_count < 10:
        label = "elevated"
    else:
        label = "high"
    return label
