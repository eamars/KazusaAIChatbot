"""Best-effort public event recorder implementations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal
from uuid import uuid4

from kazusa_ai_chatbot.character_identity_growth import models as identity_models
from kazusa_ai_chatbot.config import AUDIT_LOG_TTL_DAYS
from kazusa_ai_chatbot.event_logging import repository
from kazusa_ai_chatbot.event_logging.models import (
    CONTINUITY_AGE_LABEL_VALUES,
    CONTINUITY_BARRIER_DISPOSITION_VALUES,
    CONTINUITY_BOUNDARY_STATUS_VALUES,
    CONTINUITY_BOUNDARY_VALUES,
    CONTINUITY_CACHE_DISPOSITION_VALUES,
    CONTINUITY_RECORDER_DISPOSITION_VALUES,
    CONTINUITY_SCOPE_KIND_VALUES,
    CONTINUITY_WRITE_DISPOSITION_VALUES,
    EVENT_SEVERITIES,
    SELF_COGNITION_EXECUTION_DISPOSITION_VALUES,
    SELF_COGNITION_POLICY_DISPOSITION_VALUES,
    SELF_COGNITION_POLICY_REASON_VALUES,
    SELF_COGNITION_RESPONSE_GATE_CODE_LIMIT,
    SELF_COGNITION_RESPONSE_GATE_CODE_VALUES,
    SELF_COGNITION_SEMANTIC_DISPOSITION_VALUES,
    EventLogWriteResult,
    EventScopeInput,
    EventSeverity,
    SelfCognitionBudget,
)
from kazusa_ai_chatbot.event_logging.sanitization import (
    build_scope_record,
    sanitize_cognition_v2_event_fields,
    sanitize_short_text,
    sanitize_string_list,
    sanitized_failure_reason,
    sanitized_rejection_reason,
    unsafe_field_paths,
)
from kazusa_ai_chatbot.event_logging.schemas import EventLogEventDoc
from kazusa_ai_chatbot.logging_retention import expiry_from_storage_iso
from kazusa_ai_chatbot.time_boundary import (
    normalize_storage_utc_iso,
    storage_utc_now_iso,
)

logger = logging.getLogger(__name__)

EVENT_LOG_WRITE_TIMEOUT_SECONDS = 0.25
_IDENTITY_GROWTH_EVENT_TYPES = frozenset({
    "routing",
    "proposal",
    "review",
    "policy",
    "promotion",
    "consumption",
})


def _iso_from_optional(value: datetime | None) -> str:
    """Return a canonical storage UTC timestamp from an optional datetime."""

    if value is None:
        timestamp = storage_utc_now_iso()
    else:
        timestamp = normalize_storage_utc_iso(value.isoformat())
    return timestamp


def _result(
    *,
    accepted: bool,
    event_id: str,
    status: Literal["recorded", "rejected", "failed"],
    reason: str,
) -> EventLogWriteResult:
    """Build the public write result shape."""

    result = EventLogWriteResult(
        accepted=accepted,
        event_id=event_id,
        status=status,
        reason=reason,
    )
    return result


def _failure_result(event_id: str, reason: str) -> EventLogWriteResult:
    """Build a failed result for a write that could not be persisted."""

    result = _result(
        accepted=False,
        event_id=event_id,
        status="failed",
        reason=reason,
    )
    return result


def _rejection_result(event_id: str, reason: str) -> EventLogWriteResult:
    """Build a rejected result for an unsafe or invalid event."""

    result = _result(
        accepted=False,
        event_id=event_id,
        status="rejected",
        reason=reason,
    )
    return result


async def _record_event(
    *,
    event_family: str,
    event_type: str,
    component: str,
    status: str,
    severity: EventSeverity,
    payload: Mapping[str, object],
    correlation_id: str = "",
    run_id: str = "",
    trigger_id: str = "",
    attempt_id: str = "",
    duration_ms: int | None = None,
    scope: EventScopeInput | None = None,
    metrics: dict[str, int | float | bool | str] | None = None,
    labels: dict[str, str] | None = None,
    refs: list[dict[str, str]] | None = None,
    warning_codes: Sequence[str] = (),
    error_class: str = "",
    error_preview: str = "",
    stack_fingerprint: str = "",
    recovered: bool = False,
    occurred_at: datetime | None = None,
    cognition_v2: Mapping[str, object] | None = None,
) -> EventLogWriteResult:
    """Build, sanitize, and persist one event document.

    Args:
        event_family: Canonical event family name.
        event_type: Family-specific event type.
        component: Runtime component emitting the event.
        status: Component-specific status value.
        severity: Event severity.
        payload: Family-specific typed metadata built by public recorders.
        correlation_id: Optional cross-stage correlation identifier.
        run_id: Optional durable run identifier.
        trigger_id: Optional self-cognition trigger identifier.
        attempt_id: Optional action-attempt identifier.
        duration_ms: Optional elapsed duration.
        scope: Optional caller scope that will be privately projected.
        metrics: Internal numeric or boolean event metrics.
        labels: Internal string labels.
        refs: Internal durable references.
        warning_codes: Sanitized warning code values.
        error_class: Sanitized error class name.
        error_preview: Sanitized short error text.
        stack_fingerprint: Sanitized stack fingerprint.
        recovered: Whether the caller recovered from the error.
        occurred_at: Optional event time.

    Returns:
        Best-effort write result. Persistence failures are contained here.
    """

    event_id = uuid4().hex
    if severity not in EVENT_SEVERITIES:
        result = _rejection_result(event_id, "invalid severity")
        return result

    occurred_at_utc = _iso_from_optional(occurred_at)
    if occurred_at is None:
        created_at_utc = occurred_at_utc
    else:
        created_at_utc = storage_utc_now_iso()
    event_doc = EventLogEventDoc(
        event_id=event_id,
        event_family=sanitize_short_text(event_family, limit=80),
        event_type=sanitize_short_text(event_type, limit=80),
        component=sanitize_short_text(component, limit=120),
        severity=severity,
        status=sanitize_short_text(status, limit=80),
        correlation_id=sanitize_short_text(correlation_id, limit=120),
        run_id=sanitize_short_text(run_id, limit=160),
        trigger_id=sanitize_short_text(trigger_id, limit=160),
        attempt_id=sanitize_short_text(attempt_id, limit=160),
        occurred_at=occurred_at_utc,
        created_at=created_at_utc,
        expires_at=expiry_from_storage_iso(
            occurred_at_utc,
            ttl_days=AUDIT_LOG_TTL_DAYS,
        ).isoformat(),
        duration_ms=duration_ms,
        scope=build_scope_record(scope),
        metrics=metrics or {},
        labels=labels or {},
        refs=refs or [],
        warning_codes=sanitize_string_list(warning_codes),
        error={
            "error_class": sanitize_short_text(error_class, limit=120),
            "error_preview": sanitize_short_text(error_preview),
            "stack_fingerprint": sanitize_short_text(
                stack_fingerprint,
                limit=160,
            ),
            "recovered": recovered,
        },
        payload=dict(payload),
    )
    if cognition_v2 is not None:
        event_doc["cognition_v2"] = sanitize_cognition_v2_event_fields(
            cognition_v2,
        )
    unsafe_paths = unsafe_field_paths(event_doc)
    if unsafe_paths:
        reason = sanitized_rejection_reason(unsafe_paths)
        result = _rejection_result(event_id, reason)
        return result

    try:
        await asyncio.wait_for(
            repository.write_event(event_doc),
            timeout=EVENT_LOG_WRITE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        reason = sanitized_failure_reason(exc)
        logger.warning(f"Event-log write timed out: {reason}")
        result = _failure_result(event_id, reason)
        return result
    except asyncio.CancelledError as exc:
        reason = sanitized_failure_reason(exc)
        logger.warning(f"Event-log write cancelled: {reason}")
        result = _failure_result(event_id, reason)
        return result
    except Exception as exc:
        reason = sanitized_failure_reason(exc)
        logger.warning(f"Event-log write failed: {reason}")
        result = _failure_result(event_id, reason)
        return result

    result = _result(
        accepted=True,
        event_id=event_id,
        status="recorded",
        reason="",
    )
    return result


async def record_continuity_boundary_event(
    *,
    component: str,
    boundary: str,
    status: str,
    scope_kind: str,
    candidate_count: int = 0,
    selected_count: int = 0,
    packet_turn_count: int = 0,
    protected_anchor_count: int = 0,
    rendered_chars: int = 0,
    packet_age: str = "unknown",
    source_age: str = "unknown",
    recorder_disposition: str = "unknown",
    write_disposition: str = "unknown",
    cache_disposition: str = "unknown",
    barrier_disposition: str = "unknown",
    trace_ref: str = "",
    correlation_ref: str = "",
    operation_ref: str = "",
    run_id: str = "",
    correlation_id: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record text-free continuity metadata at an approved boundary."""

    event_id = uuid4().hex
    if boundary not in CONTINUITY_BOUNDARY_VALUES:
        return _rejection_result(event_id, "invalid continuity boundary")
    if status not in CONTINUITY_BOUNDARY_STATUS_VALUES:
        return _rejection_result(event_id, "invalid continuity status")
    if scope_kind not in CONTINUITY_SCOPE_KIND_VALUES:
        return _rejection_result(event_id, "invalid continuity scope")
    if packet_age not in CONTINUITY_AGE_LABEL_VALUES:
        return _rejection_result(event_id, "invalid continuity packet age")
    if source_age not in CONTINUITY_AGE_LABEL_VALUES:
        return _rejection_result(event_id, "invalid continuity source age")
    if recorder_disposition not in CONTINUITY_RECORDER_DISPOSITION_VALUES:
        return _rejection_result(event_id, "invalid continuity recorder disposition")
    if write_disposition not in CONTINUITY_WRITE_DISPOSITION_VALUES:
        return _rejection_result(event_id, "invalid continuity write disposition")
    if cache_disposition not in CONTINUITY_CACHE_DISPOSITION_VALUES:
        return _rejection_result(event_id, "invalid continuity cache disposition")
    if barrier_disposition not in CONTINUITY_BARRIER_DISPOSITION_VALUES:
        return _rejection_result(event_id, "invalid continuity barrier disposition")

    bounded_counts = {
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "packet_turn_count": packet_turn_count,
        "protected_anchor_count": protected_anchor_count,
    }
    for field_name, value in bounded_counts.items():
        if not _valid_continuity_count(value, maximum=1000):
            return _rejection_result(event_id, f"invalid {field_name}")
    if not _valid_continuity_count(rendered_chars, maximum=100000):
        return _rejection_result(event_id, "invalid rendered_chars")

    refs: list[dict[str, str]] = []
    for ref_type, ref_id in (
        ("trace", trace_ref),
        ("correlation", correlation_ref),
        ("operation", operation_ref),
    ):
        sanitized_ref = sanitize_short_text(ref_id, limit=160)
        if sanitized_ref:
            refs.append({"ref_type": ref_type, "ref_id": sanitized_ref})

    payload = {
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "packet_turn_count": packet_turn_count,
        "protected_anchor_count": protected_anchor_count,
        "rendered_chars": rendered_chars,
    }
    labels = {
        "packet_age": packet_age,
        "source_age": source_age,
        "recorder_disposition": recorder_disposition,
        "write_disposition": write_disposition,
        "cache_disposition": cache_disposition,
        "barrier_disposition": barrier_disposition,
    }
    result = await _record_event(
        event_family="continuity_boundary",
        event_type=boundary,
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        run_id=run_id,
        correlation_id=correlation_id,
        refs=refs,
        labels=labels,
        metrics={
            "candidate_count": candidate_count,
            "selected_count": selected_count,
            "packet_turn_count": packet_turn_count,
            "protected_anchor_count": protected_anchor_count,
            "rendered_chars": rendered_chars,
        },
        occurred_at=occurred_at,
    )
    return result


def _valid_continuity_count(value: object, *, maximum: int) -> bool:
    """Validate one non-negative bounded continuity metric."""

    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= maximum
    )


async def record_cognition_v2_event(
    *,
    component: str,
    cognition_component: str,
    status: str,
    stage_status: Literal["started", "completed", "failed", "skipped"],
    selected_branch_id: str = "",
    state_scope: Literal["", "user", "character"] = "",
    state_commit_status: Literal[
        "not_started",
        "committed",
        "failed",
        "skipped",
    ] = "not_started",
    severity: EventSeverity = "info",
    warning_codes: Sequence[str] = (),
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record one bounded V2 stage or state-commit diagnostic event."""

    fields = {
        "cognition_component": cognition_component,
        "selected_branch_id": selected_branch_id,
        "state_scope": state_scope,
        "state_commit_status": state_commit_status,
        "stage_status": stage_status,
    }
    result = await _record_event(
        event_family="cognition_v2",
        event_type="component_status",
        component=component,
        status=status,
        severity=severity,
        payload={},
        warning_codes=warning_codes,
        occurred_at=occurred_at,
        cognition_v2=fields,
    )
    return result


async def record_character_identity_growth_event(
    *,
    event_type: Literal[
        "routing",
        "proposal",
        "review",
        "policy",
        "promotion",
        "consumption",
    ],
    stage: str,
    reason_code: str,
    status: str,
    correlation_id: str = "",
    run_id: str = "",
    revision_number: int | None = None,
    consumer_count: int = 0,
    projection_digest: str = "",
    severity: EventSeverity = "info",
    warning_codes: Sequence[str] = (),
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record sanitized identity funnel, promotion, or consumption metadata."""

    if event_type not in _IDENTITY_GROWTH_EVENT_TYPES:
        return _rejection_result(uuid4().hex, "invalid identity event type")
    if reason_code not in identity_models.IDENTITY_GROWTH_REASON_CODES:
        return _rejection_result(uuid4().hex, "invalid identity reason code")
    if (
        revision_number is not None
        and (
            not isinstance(revision_number, int)
            or isinstance(revision_number, bool)
            or revision_number < 0
        )
    ):
        return _rejection_result(
            uuid4().hex,
            "invalid identity revision number",
        )
    if (
        not isinstance(consumer_count, int)
        or isinstance(consumer_count, bool)
        or consumer_count < 0
        or consumer_count > len(identity_models.IDENTITY_CONSUMER_KINDS)
    ):
        return _rejection_result(
            uuid4().hex,
            "invalid identity consumer count",
        )
    normalized_digest = sanitize_short_text(
        projection_digest,
        limit=64,
    )
    if normalized_digest and (
        len(normalized_digest) != 64
        or normalized_digest != normalized_digest.lower()
        or any(
            character not in "0123456789abcdef"
            for character in normalized_digest
        )
    ):
        return _rejection_result(
            uuid4().hex,
            "invalid identity projection digest",
        )
    payload: dict[str, object] = {
        "stage": sanitize_short_text(stage, limit=80),
        "reason_code": reason_code,
        "consumer_count": consumer_count,
        "projection_digest": normalized_digest,
    }
    if revision_number is not None:
        payload["revision_number"] = revision_number
    return await _record_event(
        event_family="character_identity_growth",
        event_type=event_type,
        component="character_identity_growth",
        status=status,
        severity=severity,
        payload=payload,
        correlation_id=correlation_id,
        run_id=run_id,
        warning_codes=warning_codes,
        occurred_at=occurred_at,
    )


async def record_process_event(
    *,
    event_type: Literal["startup", "shutdown", "lifespan_error"],
    phase: str,
    component: str,
    status: str,
    pid: int,
    host_label: str,
    config_snapshot_id: str = "",
    git_commit: str = "",
    severity: EventSeverity = "info",
    correlation_id: str = "",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record process lifecycle metadata."""

    payload = {
        "phase": sanitize_short_text(phase, limit=80),
        "pid": pid,
        "host_label": sanitize_short_text(host_label, limit=120),
        "config_snapshot_id": sanitize_short_text(config_snapshot_id, limit=160),
        "git_commit": sanitize_short_text(git_commit, limit=80),
    }
    result = await _record_event(
        event_family="process",
        event_type=event_type,
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        correlation_id=correlation_id,
        occurred_at=occurred_at,
    )
    return result


async def record_worker_event(
    *,
    event_type: Literal["started", "stopped", "tick", "disabled", "cancelled"],
    component: str,
    worker_name: str,
    enabled: bool,
    dry_run: bool,
    run_kind: str,
    status: str,
    processed_count: int = 0,
    succeeded_count: int = 0,
    failed_count: int = 0,
    skipped_count: int = 0,
    deferred: bool = False,
    defer_reason: str = "",
    run_id: str = "",
    duration_ms: int | None = None,
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record worker lifecycle or tick metadata."""

    payload = {
        "worker_name": sanitize_short_text(worker_name, limit=120),
        "enabled": enabled,
        "dry_run": dry_run,
        "run_kind": sanitize_short_text(run_kind, limit=100),
        "processed_count": processed_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "deferred": deferred,
        "defer_reason": sanitize_short_text(defer_reason),
    }
    result = await _record_event(
        event_family="worker",
        event_type=event_type,
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        run_id=run_id,
        duration_ms=duration_ms,
        occurred_at=occurred_at,
    )
    return result


async def record_llm_stage_event(
    *,
    component: str,
    stage_name: str,
    route_name: str,
    model_name: str,
    status: str,
    prompt_chars: int,
    output_chars: int,
    parse_status: str,
    retry_count: int,
    json_repair_used: bool,
    run_id: str = "",
    correlation_id: str = "",
    duration_ms: int | None = None,
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record metadata for one model-backed stage."""

    payload = {
        "stage_name": sanitize_short_text(stage_name, limit=120),
        "route_name": sanitize_short_text(route_name, limit=120),
        "model_name": sanitize_short_text(model_name, limit=160),
        "prompt_chars": prompt_chars,
        "output_chars": output_chars,
        "parse_status": sanitize_short_text(parse_status, limit=80),
        "retry_count": retry_count,
        "json_repair_used": json_repair_used,
    }
    result = await _record_event(
        event_family="llm_stage",
        event_type=stage_name,
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        run_id=run_id,
        correlation_id=correlation_id,
        duration_ms=duration_ms,
        occurred_at=occurred_at,
    )
    return result


async def record_runtime_error_event(
    *,
    component: str,
    error_class: str,
    error_preview: str,
    stack_fingerprint: str,
    top_frame_module: str,
    recovered: bool,
    status: str = "failed",
    run_id: str = "",
    correlation_id: str = "",
    severity: EventSeverity = "error",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record a recoverable runtime error boundary."""

    payload = {
        "error_class": sanitize_short_text(error_class, limit=120),
        "error_preview": sanitize_short_text(error_preview),
        "stack_fingerprint": sanitize_short_text(stack_fingerprint, limit=160),
        "top_frame_module": sanitize_short_text(top_frame_module, limit=160),
        "recovered": recovered,
    }
    result = await _record_event(
        event_family="runtime_error",
        event_type="runtime_error",
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        run_id=run_id,
        correlation_id=correlation_id,
        error_class=error_class,
        error_preview=error_preview,
        stack_fingerprint=stack_fingerprint,
        recovered=recovered,
        occurred_at=occurred_at,
    )
    return result


async def record_pipeline_turn_event(
    *,
    component: str,
    correlation_id: str,
    status: str,
    queue_wait_ms: int,
    stages_reached: Sequence[str],
    final_outcome: str,
    scheduled_followups: int,
    debug_modes: Sequence[str] = (),
    scope: EventScopeInput | None = None,
    duration_ms: int | None = None,
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record sanitized live-turn pipeline outcome metadata."""

    payload = {
        "queue_wait_ms": queue_wait_ms,
        "stages_reached": sanitize_string_list(stages_reached),
        "final_outcome": sanitize_short_text(final_outcome, limit=120),
        "scheduled_followups": scheduled_followups,
        "debug_modes": sanitize_string_list(debug_modes),
    }
    result = await _record_event(
        event_family="pipeline_turn",
        event_type="turn",
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        correlation_id=correlation_id,
        duration_ms=duration_ms,
        scope=scope,
        occurred_at=occurred_at,
    )
    return result


async def record_queue_intake_event(
    *,
    component: str,
    correlation_id: str,
    status: str,
    queue_depth: int,
    coalesced_count: int,
    dropped_count: int,
    protected_by_reply: bool,
    listen_only: bool,
    scope: EventScopeInput | None = None,
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record queue intake or pruning metadata."""

    payload = {
        "queue_depth": queue_depth,
        "coalesced_count": coalesced_count,
        "dropped_count": dropped_count,
        "protected_by_reply": protected_by_reply,
        "listen_only": listen_only,
    }
    result = await _record_event(
        event_family="queue_intake",
        event_type="queue_intake",
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        correlation_id=correlation_id,
        scope=scope,
        occurred_at=occurred_at,
    )
    return result


async def record_rag_stage_event(
    *,
    component: str,
    correlation_id: str,
    agent_name: str,
    status: str,
    slot_count: int,
    retrieval_count: int,
    cache_hit: bool,
    no_evidence: bool,
    latency_ms: int,
    safety_recovery_count: int = 0,
    safety_recovery_first: str = "",
    run_id: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record retrieval-stage metadata without evidence text."""

    payload = {
        "agent_name": sanitize_short_text(agent_name, limit=120),
        "slot_count": slot_count,
        "retrieval_count": retrieval_count,
        "cache_hit": cache_hit,
        "no_evidence": no_evidence,
        "latency_ms": latency_ms,
        "safety_recovery_count": max(0, safety_recovery_count),
        "safety_recovery_first": sanitize_short_text(
            safety_recovery_first,
            limit=160,
        ),
    }
    result = await _record_event(
        event_family="rag_stage",
        event_type=agent_name,
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        correlation_id=correlation_id,
        run_id=run_id,
        metrics={"latency_ms": latency_ms},
        occurred_at=occurred_at,
    )
    return result


async def record_dialog_quality_event(
    *,
    component: str,
    correlation_id: str,
    usage_mode: str,
    quality_status: str,
    retry_count: int,
    failure_codes: Sequence[str],
    content_plan_entry_count: int,
    status: str,
    run_id: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record dialog output quality metadata."""

    payload = {
        "usage_mode": sanitize_short_text(usage_mode, limit=120),
        "quality_status": sanitize_short_text(quality_status, limit=100),
        "retry_count": retry_count,
        "failure_codes": sanitize_string_list(failure_codes),
        "content_plan_entry_count": content_plan_entry_count,
    }
    result = await _record_event(
        event_family="dialog_quality",
        event_type="dialog_quality",
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        correlation_id=correlation_id,
        run_id=run_id,
        warning_codes=failure_codes,
        occurred_at=occurred_at,
    )
    return result


async def record_dispatcher_event(
    *,
    component: str,
    action_kind: str,
    validation_status: str,
    adapter_available: bool,
    status: str,
    scheduled_event_ids: Sequence[str] = (),
    rejection_codes: Sequence[str] = (),
    attempt_id: str = "",
    correlation_id: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record scheduler or adapter handoff outcome metadata."""

    sanitized_event_ids = sanitize_string_list(scheduled_event_ids)
    payload = {
        "action_kind": sanitize_short_text(action_kind, limit=100),
        "validation_status": sanitize_short_text(validation_status, limit=100),
        "adapter_available": adapter_available,
        "scheduled_event_ids": sanitized_event_ids,
        "rejection_codes": sanitize_string_list(rejection_codes),
    }
    refs = [
        {"ref_type": "scheduled_event", "ref_id": event_id}
        for event_id in sanitized_event_ids
    ]
    result = await _record_event(
        event_family="dispatcher",
        event_type=action_kind,
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        attempt_id=attempt_id,
        correlation_id=correlation_id,
        refs=refs,
        warning_codes=rejection_codes,
        occurred_at=occurred_at,
    )
    return result


async def record_database_operation_event(
    *,
    component: str,
    collection: str,
    operation_kind: str,
    status: str,
    idempotency_result: str,
    latency_ms: int,
    document_ref: str = "",
    run_id: str = "",
    correlation_id: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record one approved production database operation outcome."""

    payload = {
        "collection": sanitize_short_text(collection, limit=120),
        "operation_kind": sanitize_short_text(operation_kind, limit=120),
        "idempotency_result": sanitize_short_text(idempotency_result, limit=120),
        "latency_ms": latency_ms,
        "document_ref": sanitize_short_text(document_ref, limit=160),
    }
    refs = []
    if document_ref:
        refs.append({
            "ref_type": sanitize_short_text(collection, limit=120),
            "ref_id": sanitize_short_text(document_ref),
        })
    result = await _record_event(
        event_family="database_operation",
        event_type=operation_kind,
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        run_id=run_id,
        correlation_id=correlation_id,
        refs=refs,
        metrics={"latency_ms": latency_ms},
        occurred_at=occurred_at,
    )
    return result


async def record_self_cognition_event(
    *,
    component: str,
    case_id: str,
    trigger_kind: str,
    selected_route: str,
    output_mode: str,
    budget: SelfCognitionBudget,
    dispatch_status: str,
    status: str,
    trigger_id: str = "",
    run_id: str = "",
    attempt_id: str = "",
    consolidation_outcome: Mapping[str, object] | None = None,
    target_binding_failure: Mapping[str, object] | None = None,
    semantic_disposition: str | None = None,
    policy_disposition: str | None = None,
    execution_disposition: str | None = None,
    policy_reason: str = "",
    response_gate_codes: Sequence[str] = (),
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record production self-cognition case metadata."""

    budget_payload = {
        "rag_calls": int(budget["rag_calls"]),
        "cognition_calls": int(budget["cognition_calls"]),
        "dialog_calls": int(budget["dialog_calls"]),
        "topic_limit": int(budget["topic_limit"]),
    }
    payload = {
        "case_id": sanitize_short_text(case_id, limit=160),
        "trigger_kind": sanitize_short_text(trigger_kind, limit=120),
        "selected_route": sanitize_short_text(selected_route, limit=120),
        "output_mode": sanitize_short_text(output_mode, limit=120),
        "budget": budget_payload,
        "dispatch_status": sanitize_short_text(dispatch_status, limit=100),
    }
    if semantic_disposition is not None:
        payload.update({
            "semantic_disposition": _closed_self_cognition_value(
                semantic_disposition,
                SELF_COGNITION_SEMANTIC_DISPOSITION_VALUES,
                "cognition_contract_failed",
            ),
            "policy_disposition": _closed_self_cognition_value(
                policy_disposition,
                SELF_COGNITION_POLICY_DISPOSITION_VALUES,
                "not_evaluated",
            ),
            "execution_disposition": _closed_self_cognition_value(
                execution_disposition,
                SELF_COGNITION_EXECUTION_DISPOSITION_VALUES,
                "not_requested",
            ),
            "policy_reason": _closed_self_cognition_value(
                policy_reason,
                SELF_COGNITION_POLICY_REASON_VALUES,
                "",
            ),
            "response_gate_codes": [
                code
                for code in response_gate_codes
                if isinstance(code, str)
                and code in SELF_COGNITION_RESPONSE_GATE_CODE_VALUES
            ][:SELF_COGNITION_RESPONSE_GATE_CODE_LIMIT],
        })
    sanitized_consolidation_outcome = (
        _sanitize_self_cognition_consolidation_outcome(
            consolidation_outcome,
        )
    )
    if sanitized_consolidation_outcome:
        payload["consolidation_outcome"] = sanitized_consolidation_outcome
    sanitized_target_binding_failure = (
        _sanitize_self_cognition_target_binding_failure(
            target_binding_failure,
        )
    )
    if sanitized_target_binding_failure:
        payload["target_binding_failure"] = sanitized_target_binding_failure
    result = await _record_event(
        event_family="self_cognition",
        event_type=trigger_kind,
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        trigger_id=trigger_id,
        run_id=run_id,
        attempt_id=attempt_id,
        occurred_at=occurred_at,
    )
    return result


def _closed_self_cognition_value(
    value: object,
    allowed_values: frozenset[str],
    fallback: str,
) -> str:
    """Keep one self-cognition event field inside its closed value set."""

    if isinstance(value, str) and value in allowed_values:
        return value
    return fallback


def _sanitize_self_cognition_target_binding_failure(
    target_binding_failure: Mapping[str, object] | None,
) -> dict[str, object]:
    """Project target-binding failure metadata into the event-log shape."""

    if target_binding_failure is None:
        empty_failure: dict[str, object] = {}
        return empty_failure

    target_global_user_id = sanitize_short_text(
        target_binding_failure.get("target_global_user_id", ""),
        limit=160,
    )
    target_platform_user_id = sanitize_short_text(
        target_binding_failure.get("target_platform_user_id", ""),
        limit=160,
    )
    sanitized_failure: dict[str, object] = {
        "reason": sanitize_short_text(
            target_binding_failure.get("reason", ""),
            limit=120,
        ),
        "platform": sanitize_short_text(
            target_binding_failure.get("platform", ""),
            limit=80,
        ),
        "source_ref": sanitize_short_text(
            target_binding_failure.get("source_ref", ""),
            limit=160,
        ),
        "source_platform_channel_id": sanitize_short_text(
            target_binding_failure.get("source_platform_channel_id", ""),
            limit=160,
        ),
        "source_channel_type": sanitize_short_text(
            target_binding_failure.get("source_channel_type", ""),
            limit=80,
        ),
        "has_target_global_user_id": bool(target_global_user_id),
        "has_target_platform_user_id": bool(target_platform_user_id),
    }
    return sanitized_failure


def _sanitize_self_cognition_consolidation_outcome(
    consolidation_outcome: Mapping[str, object] | None,
) -> dict[str, object]:
    """Project consolidation metadata into the approved event-log shape.

    Args:
        consolidation_outcome: Sanitized runner artifact metadata, if the case
            applied same-path consolidation.

    Returns:
        Event-log payload subset without source text, private finalization,
        generated candidate text, or raw database documents.
    """

    if consolidation_outcome is None:
        empty_outcome: dict[str, object] = {}
        return empty_outcome

    raw_write_success = consolidation_outcome.get("write_success")
    write_success: dict[str, bool] = {}
    if isinstance(raw_write_success, Mapping):
        for raw_key, raw_value in raw_write_success.items():
            key = sanitize_short_text(raw_key, limit=80)
            if not key or unsafe_field_paths({key: ""}):
                continue
            write_success[key] = bool(raw_value)

    raw_scheduled_event_count = consolidation_outcome.get(
        "scheduled_event_count"
    )
    scheduled_event_count = 0
    if (
        isinstance(raw_scheduled_event_count, int)
        and not isinstance(raw_scheduled_event_count, bool)
    ):
        scheduled_event_count = raw_scheduled_event_count

    raw_cache_evicted_count = consolidation_outcome.get("cache_evicted_count")
    cache_evicted_count = 0
    if (
        isinstance(raw_cache_evicted_count, int)
        and not isinstance(raw_cache_evicted_count, bool)
    ):
        cache_evicted_count = raw_cache_evicted_count

    sanitized_outcome = {
        "consolidation_called": (
            consolidation_outcome.get("consolidation_called") is True
        ),
        "write_success": write_success,
        "scheduled_event_count": scheduled_event_count,
        "cache_evicted_count": cache_evicted_count,
        "origin_trigger_source": sanitize_short_text(
            consolidation_outcome.get("origin_trigger_source", ""),
            limit=80,
        ),
        "origin_episode_id": sanitize_short_text(
            consolidation_outcome.get("origin_episode_id", ""),
            limit=160,
        ),
    }
    return sanitized_outcome


async def record_model_contract_event(
    *,
    component: str,
    stage_name: str,
    violation_kind: str,
    missing_fields: Sequence[str],
    invalid_fields: Sequence[str],
    repair_used: bool,
    status: str,
    run_id: str = "",
    correlation_id: str = "",
    severity: EventSeverity = "warning",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record sanitized model contract drift metadata."""

    payload = {
        "stage_name": sanitize_short_text(stage_name, limit=120),
        "violation_kind": sanitize_short_text(violation_kind, limit=120),
        "missing_fields": sanitize_string_list(missing_fields),
        "invalid_fields": sanitize_string_list(invalid_fields),
        "repair_used": repair_used,
    }
    result = await _record_event(
        event_family="model_contract",
        event_type=violation_kind,
        component=component,
        status=status,
        severity=severity,
        payload=payload,
        run_id=run_id,
        correlation_id=correlation_id,
        warning_codes=[violation_kind],
        occurred_at=occurred_at,
    )
    return result


async def record_resource_health_event(
    *,
    component: str,
    resource_name: str,
    resource_kind: str,
    availability: str,
    latency_ms: int,
    failure_class: str = "",
    status: str = "",
    severity: EventSeverity = "info",
    occurred_at: datetime | None = None,
) -> EventLogWriteResult:
    """Record one external resource health sample."""

    effective_status = status or availability
    payload = {
        "resource_name": sanitize_short_text(resource_name, limit=120),
        "resource_kind": sanitize_short_text(resource_kind, limit=120),
        "availability": sanitize_short_text(availability, limit=80),
        "latency_ms": latency_ms,
        "failure_class": sanitize_short_text(failure_class, limit=120),
    }
    result = await _record_event(
        event_family="resource_health",
        event_type=resource_kind,
        component=component,
        status=effective_status,
        severity=severity,
        payload=payload,
        metrics={"latency_ms": latency_ms},
        occurred_at=occurred_at,
    )
    return result
