"""Operator status builders backed by event-log aggregates."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from kazusa_ai_chatbot.event_logging import repository
from kazusa_ai_chatbot.event_logging.models import (
    llm_parse_stability_label,
    reflection_health_label,
    self_cognition_liveness_label,
    worker_error_level_label,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now


def _window_filter(
    *,
    window_hours: int,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build a Mongo-style lower-bound filter for recent events."""

    seconds = max(0, int(window_hours)) * 3600
    reference_utc = now_utc or storage_utc_now()
    window_start_utc = reference_utc - timedelta(seconds=seconds)
    filter_doc = {"occurred_at": {"$gte": window_start_utc.isoformat()}}
    return filter_doc


async def _count(filter_doc: dict[str, Any]) -> int:
    """Return an event count through the private repository adapter."""

    count = await repository.count_events(filter_doc)
    return count


async def _latest_event(filter_doc: dict[str, Any]) -> dict[str, Any]:
    """Return the latest matching event or an empty mapping."""

    rows = await repository.find_events(
        filter_doc,
        sort=[("occurred_at", -1)],
        limit=1,
    )
    latest = rows[0] if rows else {}
    return latest


async def build_runtime_status(*, window_hours: int = 24) -> dict[str, object]:
    """Build bounded runtime status for `/ops/runtime-status`."""

    generated_at_utc = storage_utc_now()
    base_filter = _window_filter(
        window_hours=window_hours,
        now_utc=generated_at_utc,
    )
    process_latest = await _latest_event({
        **base_filter,
        "event_family": "process",
    })
    reflection_latest = await _latest_event({
        **base_filter,
        "component": "reflection_cycle.worker",
    })
    self_cognition_latest = await _latest_event({
        **base_filter,
        "component": "self_cognition.worker",
    })
    worker_error_count = await _count({
        **base_filter,
        "event_family": "runtime_error",
        "component": {"$in": ["reflection_cycle.worker", "self_cognition.worker"]},
    })
    status = {
        "status": "ok",
        "generated_at": generated_at_utc.isoformat(),
        "window_hours": window_hours,
        "process": {
            "last_event_at": str(process_latest.get("occurred_at", "")),
            "last_status": str(process_latest.get("status", "")),
        },
        "workers": {
            "reflection_cycle": {
                "last_event_at": str(reflection_latest.get("occurred_at", "")),
                "last_status": str(reflection_latest.get("status", "unknown")),
            },
            "self_cognition": {
                "last_event_at": str(self_cognition_latest.get("occurred_at", "")),
                "last_status": str(self_cognition_latest.get("status", "unknown")),
            },
        },
        "semantic_descriptors": {
            "worker_error_level": worker_error_level_label(
                error_count=worker_error_count,
            ),
        },
    }
    return status


async def build_reflection_stats(*, window_hours: int = 24) -> dict[str, object]:
    """Build bounded reflection stats for `/ops/reflection/stats`."""

    generated_at_utc = storage_utc_now()
    base_filter = _window_filter(
        window_hours=window_hours,
        now_utc=generated_at_utc,
    )
    event_filter = {
        **base_filter,
        "component": "reflection_cycle.worker",
    }
    succeeded_count = await _count({**event_filter, "status": "succeeded"})
    failed_count = await _count({**event_filter, "status": "failed"})
    skipped_count = await _count({**event_filter, "status": "skipped"})
    latest = await _latest_event(event_filter)
    stats = {
        "status": "ok",
        "generated_at": generated_at_utc.isoformat(),
        "window_hours": window_hours,
        "counts": {
            "succeeded": succeeded_count,
            "failed": failed_count,
            "skipped": skipped_count,
        },
        "latest": {
            "event_id": str(latest.get("event_id", "")),
            "run_id": str(latest.get("run_id", "")),
            "occurred_at": str(latest.get("occurred_at", "")),
            "status": str(latest.get("status", "")),
        },
        "semantic_descriptors": {
            "reflection_health": reflection_health_label(
                failed_count=failed_count,
                succeeded_count=succeeded_count,
            ),
        },
    }
    return stats


async def build_self_cognition_stats(*, window_hours: int = 24) -> dict[str, object]:
    """Build bounded self-cognition stats for `/ops/self-cognition/stats`."""

    generated_at_utc = storage_utc_now()
    base_filter = _window_filter(
        window_hours=window_hours,
        now_utc=generated_at_utc,
    )
    event_filter = {
        **base_filter,
        "event_family": "self_cognition",
    }
    run_count = await _count(event_filter)
    dispatch_count = await _count({
        **event_filter,
        "payload.dispatch_status": "accepted",
    })
    latest = await _latest_event(event_filter)
    stats = {
        "status": "ok",
        "generated_at": generated_at_utc.isoformat(),
        "window_hours": window_hours,
        "counts": {
            "runs": run_count,
            "dispatch_accepted": dispatch_count,
        },
        "latest": {
            "event_id": str(latest.get("event_id", "")),
            "run_id": str(latest.get("run_id", "")),
            "trigger_id": str(latest.get("trigger_id", "")),
            "attempt_id": str(latest.get("attempt_id", "")),
            "occurred_at": str(latest.get("occurred_at", "")),
            "status": str(latest.get("status", "")),
        },
        "semantic_descriptors": {
            "self_cognition_liveness": self_cognition_liveness_label(
                run_count=run_count,
                dispatch_count=dispatch_count,
            ),
        },
    }
    return stats


async def build_snapshot_source_counts(
    *,
    window_hours: int,
) -> dict[str, int]:
    """Build aggregate counts used by snapshot writers."""

    base_filter = _window_filter(window_hours=window_hours)
    source_counts = {
        "reflection_succeeded": await _count({
            **base_filter,
            "component": "reflection_cycle.worker",
            "status": "succeeded",
        }),
        "reflection_failed": await _count({
            **base_filter,
            "component": "reflection_cycle.worker",
            "status": "failed",
        }),
        "self_cognition_runs": await _count({
            **base_filter,
            "event_family": "self_cognition",
        }),
        "self_cognition_dispatch_accepted": await _count({
            **base_filter,
            "event_family": "self_cognition",
            "payload.dispatch_status": "accepted",
        }),
        "llm_failed": await _count({
            **base_filter,
            "event_family": "llm_stage",
            "status": "failed",
        }),
        "llm_repaired": await _count({
            **base_filter,
            "event_family": "llm_stage",
            "payload.json_repair_used": True,
        }),
        "worker_errors": await _count({
            **base_filter,
            "event_family": "runtime_error",
            "component": {
                "$in": ["reflection_cycle.worker", "self_cognition.worker"],
            },
        }),
    }
    return source_counts


def build_semantic_descriptors(
    source_counts: dict[str, int],
) -> dict[str, str]:
    """Build prompt-safe semantic labels from aggregate counts."""

    descriptors = {
        "reflection_health": reflection_health_label(
            failed_count=source_counts["reflection_failed"],
            succeeded_count=source_counts["reflection_succeeded"],
        ),
        "self_cognition_liveness": self_cognition_liveness_label(
            run_count=source_counts["self_cognition_runs"],
            dispatch_count=source_counts["self_cognition_dispatch_accepted"],
        ),
        "llm_parse_stability": llm_parse_stability_label(
            failed_count=source_counts["llm_failed"],
            repaired_count=source_counts["llm_repaired"],
        ),
        "worker_error_level": worker_error_level_label(
            error_count=source_counts["worker_errors"],
        ),
    }
    return descriptors
