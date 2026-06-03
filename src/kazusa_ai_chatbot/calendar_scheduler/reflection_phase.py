"""Mechanical mapping between reflection phase intents and calendar runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from kazusa_ai_chatbot.calendar_scheduler import models
from kazusa_ai_chatbot.reflection_cycle import phase_scheduler


def build_reflection_phase_calendar_runs(
    intents: list[phase_scheduler.ReflectionPhaseRunIntent],
    *,
    storage_timestamp_utc: str,
) -> list[dict[str, Any]]:
    """Project reflection phase intents into pending calendar run documents."""

    runs: list[dict[str, Any]] = []
    for intent in intents:
        run = {
            "schema_version": models.CALENDAR_RUN_SCHEMA_VERSION,
            "owner": models.CALENDAR_OWNER,
            "run_id": intent["run_id"],
            "schedule_id": intent["idempotency_key"],
            "trigger_kind": models.TRIGGER_REFLECTION_PHASE_SLOT,
            "status": models.RUN_STATUS_PENDING,
            "due_at": intent["due_at"],
            "payload": {"reflection_phase_intent": deepcopy(intent)},
            "source_scope": deepcopy(intent["source_scope"]),
            "idempotency_key": intent["idempotency_key"],
            "attempt_count": 0,
            "max_attempts": models.DEFAULT_RUN_MAX_ATTEMPTS,
            "claimed_at": None,
            "completed_at": None,
            "failed_at": None,
            "skipped_at": None,
            "lease_owner": None,
            "lease_expires_at": None,
            "result_summary": None,
            "failure_summary": None,
            "legacy_source": None,
            "period_start_utc": intent["period_start_utc"],
            "slot_index": intent["slot_index"],
            "offset_seconds": intent["offset_seconds"],
            "created_at": storage_timestamp_utc,
            "updated_at": storage_timestamp_utc,
        }
        runs.append(run)

    return runs


def calendar_run_to_reflection_phase_intent(
    run: dict[str, Any],
) -> phase_scheduler.ReflectionPhaseRunIntent:
    """Restore the original reflection phase intent from a calendar run."""

    intent = deepcopy(run["payload"]["reflection_phase_intent"])
    return intent
