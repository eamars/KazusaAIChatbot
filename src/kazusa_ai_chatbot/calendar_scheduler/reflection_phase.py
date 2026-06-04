"""Mechanical mapping between reflection phase intents and calendar runs."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.config import (
    REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD,
    REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS,
    REFLECTION_WORKER_INTERVAL_SECONDS,
)
from kazusa_ai_chatbot.calendar_scheduler import models
from kazusa_ai_chatbot.calendar_scheduler import repository as calendar_repository
from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
from kazusa_ai_chatbot.reflection_cycle import phase_scheduler
from kazusa_ai_chatbot.reflection_cycle.models import (
    READONLY_REFLECTION_MONITOR_ELIGIBILITY_HOURS,
    READONLY_REFLECTION_PROMPT_VERSION,
    ReflectionWorkerResult,
)
from kazusa_ai_chatbot.reflection_cycle.selector import collect_reflection_inputs
from kazusa_ai_chatbot.reflection_cycle.worker import (
    ExpectedDailyChannelHourlyRuns,
    collect_phase_scope_input_for_intent,
    execute_reflection_phase_intent,
    expected_hourly_run_ids_for_scope,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime


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


async def materialize_reflection_phase_period(
    *,
    period_start_utc: datetime,
    storage_timestamp_utc: str,
    repository: Any = calendar_repository,
) -> dict[str, Any]:
    """Snapshot eligible scopes and upsert phase slot calendar runs."""

    input_set = await collect_reflection_inputs(
        lookback_hours=READONLY_REFLECTION_MONITOR_ELIGIBILITY_HOURS,
        now=period_start_utc,
        allow_fallback=False,
    )
    intents = phase_scheduler.build_phase_run_intents(
        period_start_utc=period_start_utc,
        eligible_scopes=input_set.selected_scopes,
        phase_period_seconds=REFLECTION_WORKER_INTERVAL_SECONDS,
        max_slots_per_period=REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD,
        min_slot_spacing_seconds=REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS,
        prompt_version=READONLY_REFLECTION_PROMPT_VERSION,
    )
    runs = build_reflection_phase_calendar_runs(
        intents,
        storage_timestamp_utc=storage_timestamp_utc,
    )
    for run in runs:
        await repository.upsert_calendar_run(run)

    summary = {
        "materialized_count": len(runs),
        "run_ids": [
            run["run_id"]
            for run in runs
        ],
    }
    return summary


async def handle_reflection_phase_calendar_run(
    run: dict[str, Any],
    *,
    now: datetime,
    dry_run: bool,
    is_primary_interaction_busy: Callable[[], bool],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None,
    execute_phase_intent_func: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute a claimed phase slot through the reflection worker seam."""

    intent = calendar_run_to_reflection_phase_intent(run)
    phase_executor = execute_phase_intent_func
    if phase_executor is None:
        phase_executor = execute_reflection_phase_intent
    phase_results = await phase_executor(
        intent=intent,
        now=now,
        dry_run=dry_run,
        is_primary_interaction_busy=is_primary_interaction_busy,
        adapter_registry_provider=adapter_registry_provider,
    )
    summary = _reflection_phase_result_summary(phase_results)
    return summary


class CalendarReflectionPhaseRunProvider:
    """Daily-readiness provider backed by durable phase slot calendar runs."""

    def __init__(
        self,
        *,
        repository: Any = calendar_repository,
        collect_phase_scope_input_func: Callable[..., Any] = (
            collect_phase_scope_input_for_intent
        ),
        expected_hourly_run_ids_func: Callable[..., list[str]] = (
            expected_hourly_run_ids_for_scope
        ),
    ) -> None:
        self.repository = repository
        self.collect_phase_scope_input_func = collect_phase_scope_input_func
        self.expected_hourly_run_ids_func = expected_hourly_run_ids_func

    async def expected_hourly_runs_for_character_local_date(
        self,
        *,
        character_local_date: str,
    ) -> list[ExpectedDailyChannelHourlyRuns]:
        """Return expected hourly run ids from stored phase slot rows."""

        runs = await (
            self.repository
            .list_reflection_phase_slot_calendar_runs_for_character_local_date(
                character_local_date=character_local_date,
            )
        )
        expected_by_scope: dict[str, ExpectedDailyChannelHourlyRuns] = {}
        for run in runs:
            intent = calendar_run_to_reflection_phase_intent(run)
            due_at = _intent_due_at(intent)
            channel_scope = await self.collect_phase_scope_input_func(
                intent=intent,
                now=due_at,
            )
            run_ids = self.expected_hourly_run_ids_func(
                channel_scope=channel_scope,
                character_local_date=character_local_date,
                now=due_at,
            )
            if not run_ids:
                continue
            expected = expected_by_scope.get(channel_scope.scope_ref)
            if expected is None:
                expected = ExpectedDailyChannelHourlyRuns(
                    channel_scope=channel_scope,
                    expected_run_ids=[],
                )
                expected_by_scope[channel_scope.scope_ref] = expected
            expected.expected_run_ids.extend(run_ids)

        expected_rows = list(expected_by_scope.values())
        for expected in expected_rows:
            expected.expected_run_ids = sorted(set(expected.expected_run_ids))
        return expected_rows


def _intent_due_at(
    intent: phase_scheduler.ReflectionPhaseRunIntent,
) -> datetime:
    """Parse the due timestamp from a reflection phase intent."""

    due_at = parse_storage_utc_datetime(str(intent["due_at"]))
    return due_at


def _reflection_phase_result_summary(
    results: list[ReflectionWorkerResult],
) -> dict[str, Any]:
    """Build the bounded calendar completion summary for phase execution."""

    processed_count = 0
    succeeded_count = 0
    failed_count = 0
    skipped_count = 0
    run_ids: list[str] = []
    for result in results:
        processed_count += result.processed_count
        succeeded_count += result.succeeded_count
        failed_count += result.failed_count
        skipped_count += result.skipped_count
        run_ids.extend(result.run_ids)

    summary = {
        "status": "completed",
        "run_kind": models.TRIGGER_REFLECTION_PHASE_SLOT,
        "processed_count": processed_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "run_ids": run_ids,
    }
    return summary
