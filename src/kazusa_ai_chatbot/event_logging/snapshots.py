"""Deterministic event-log snapshot writer."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Literal
from uuid import uuid4

from kazusa_ai_chatbot.event_logging import repository
from kazusa_ai_chatbot.config import AUDIT_LOG_TTL_DAYS
from kazusa_ai_chatbot.event_logging.models import EventLogWriteResult
from kazusa_ai_chatbot.event_logging.recording import (
    EVENT_LOG_WRITE_TIMEOUT_SECONDS,
)
from kazusa_ai_chatbot.event_logging.sanitization import sanitized_failure_reason
from kazusa_ai_chatbot.event_logging.schemas import EventLogSnapshotDoc
from kazusa_ai_chatbot.logging_retention import expiry_from_datetime
from kazusa_ai_chatbot.event_logging.status import (
    build_semantic_descriptors,
    build_snapshot_source_counts,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now

logger = logging.getLogger(__name__)


def _snapshot_result(
    *,
    accepted: bool,
    snapshot_id: str,
    status: Literal["recorded", "rejected", "failed"],
    reason: str,
) -> EventLogWriteResult:
    """Build a write-result shape for snapshot writes."""

    result = EventLogWriteResult(
        accepted=accepted,
        event_id=snapshot_id,
        status=status,
        reason=reason,
    )
    return result


async def write_analysis_snapshot(
    *,
    window_hours: int = 24,
    snapshot_kind: Literal["event_log_snapshot"] = "event_log_snapshot",
) -> EventLogWriteResult:
    """Write one deterministic aggregate snapshot for later analysis."""

    snapshot_id = uuid4().hex
    generated_at_utc = storage_utc_now()
    seconds = max(0, int(window_hours)) * 3600
    window_start_utc = generated_at_utc - timedelta(seconds=seconds)
    source_counts = await build_snapshot_source_counts(window_hours=window_hours)
    descriptors = build_semantic_descriptors(source_counts)
    snapshot_doc = EventLogSnapshotDoc(
        snapshot_id=snapshot_id,
        snapshot_kind=snapshot_kind,
        window_start=window_start_utc.isoformat(),
        window_end=generated_at_utc.isoformat(),
        generated_at=generated_at_utc.isoformat(),
        expires_at=expiry_from_datetime(
            generated_at_utc,
            ttl_days=AUDIT_LOG_TTL_DAYS,
        ).isoformat(),
        source_counts=source_counts,
        semantic_descriptors=descriptors,
        findings=[],
        source_event_refs=[],
    )
    try:
        await asyncio.wait_for(
            repository.write_snapshot(snapshot_doc),
            timeout=EVENT_LOG_WRITE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        reason = sanitized_failure_reason(exc)
        logger.warning(f"Event-log snapshot write timed out: {reason}")
        result = _snapshot_result(
            accepted=False,
            snapshot_id=snapshot_id,
            status="failed",
            reason=reason,
        )
        return result
    except asyncio.CancelledError as exc:
        reason = sanitized_failure_reason(exc)
        logger.warning(f"Event-log snapshot write cancelled: {reason}")
        result = _snapshot_result(
            accepted=False,
            snapshot_id=snapshot_id,
            status="failed",
            reason=reason,
        )
        return result
    except Exception as exc:
        reason = sanitized_failure_reason(exc)
        logger.warning(f"Event-log snapshot write failed: {reason}")
        result = _snapshot_result(
            accepted=False,
            snapshot_id=snapshot_id,
            status="failed",
            reason=reason,
        )
        return result

    result = _snapshot_result(
        accepted=True,
        snapshot_id=snapshot_id,
        status="recorded",
        reason="",
    )
    return result
