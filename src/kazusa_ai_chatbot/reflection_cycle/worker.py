"""Production reflection-cycle worker orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from kazusa_ai_chatbot.config import (
    GLOBAL_CHARACTER_GROWTH_PASS_ENABLED,
    CHARACTER_GLOBAL_USER_ID,
    REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME,
    REFLECTION_HOURLY_SLOTS_PER_TICK,
    REFLECTION_LORE_PROMOTION_ENABLED,
    REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD,
    REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS,
    REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
    REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED,
    REFLECTION_WORKER_INTERVAL_SECONDS,
)
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.global_character_growth import (
    run_global_character_growth_pass,
)
from kazusa_ai_chatbot.db import (
    find_self_cognition_group_review_window,
    get_character_profile,
    upsert_self_cognition_group_review_window,
)
from kazusa_ai_chatbot.db.schemas import (
    CharacterReflectionRunDoc,
    SelfCognitionGroupReviewWindowDoc,
)
from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
from kazusa_ai_chatbot.runtime_coordination import (
    PipelineCancelled,
    PipelineCoordinator,
    PipelineRunHandle,
    PipelineScope,
)
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    GroupActivityWindow,
    build_group_activity_windows,
)
from kazusa_ai_chatbot.reflection_cycle import repository
from kazusa_ai_chatbot.reflection_cycle.affect_settling import (
    run_daily_affect_settling as _run_daily_affect_settling,
    settling_local_date_for_due_affect_settling,
)
from kazusa_ai_chatbot.reflection_cycle.models import (
    DailySynthesisResult,
    PromptBuildResult,
    READONLY_REFLECTION_PROMPT_VERSION,
    REFLECTION_RUN_KIND_DAILY_CHANNEL,
    REFLECTION_RUN_KIND_HOURLY,
    REFLECTION_STATUS_FAILED,
    REFLECTION_STATUS_SKIPPED,
    ReflectionInputSet,
    ReflectionLLMResult,
    ReflectionPromotionResult,
    ReflectionScopeInput,
    ReflectionWorkerHandle,
    ReflectionWorkerResult,
)
from kazusa_ai_chatbot.reflection_cycle.phase_scheduler import (
    REFLECTION_PHASE_GROUPS_PER_SLOT,
    ReflectionPhaseRunIntent,
    build_phase_run_intents,
)
from kazusa_ai_chatbot.reflection_cycle.prompts import (
    build_daily_synthesis_prompt,
    build_hourly_reflection_prompt,
    build_skipped_daily_result,
    build_skipped_hourly_result,
    run_daily_synthesis_llm,
    run_hourly_reflection_llm,
)
from kazusa_ai_chatbot.reflection_cycle.runtime import (
    _channel_input_set,
    _split_scope_into_hourly_scopes,
)
from kazusa_ai_chatbot.reflection_cycle.promotion import (
    _run_global_reflection_promotion,
)
from kazusa_ai_chatbot.reflection_cycle.interaction_style import (
    run_daily_interaction_style_update as _run_daily_interaction_style_update,
)
from kazusa_ai_chatbot.reflection_cycle.selector import (
    collect_reflection_inputs,
    collect_reflection_scope_input,
)
from kazusa_ai_chatbot.self_cognition import sources as self_cognition_sources
from kazusa_ai_chatbot.self_cognition.sleep_period import (
    is_self_cognition_sleep_period,
)
from kazusa_ai_chatbot.self_cognition import worker as self_cognition_worker
from kazusa_ai_chatbot.time_boundary import (
    local_date_bounds_to_storage_utc_iso,
    local_time_context_from_storage_utc,
    normalize_storage_utc_iso,
    parse_storage_utc_datetime,
    storage_utc_now,
)


logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
REFLECTION_PHASE_LOOKBACK_HOURS = 24
GROUP_REVIEW_COALESCED_SKIP_REASON = "coalesced_by_newer_group_phase_window"
GROUP_REVIEW_STALE_SKIP_REASON = "no_group_review_case_built"
GROUP_REVIEW_FAILED_SKIP_REASON = "self_cognition_worker_failed"
_DAILY_STYLE_MAINTENANCE_GATE = "daily_channel_and_style"
_GLOBAL_PROMOTION_MAINTENANCE_GATE = "global_reflection_promotion"
_DAILY_AFFECT_SETTLING_MAINTENANCE_GATE = "daily_affect_settling"


@dataclass
class ExpectedDailyChannelHourlyRuns:
    """Expected hourly materialization for one daily channel synthesis."""

    channel_scope: ReflectionScopeInput
    expected_run_ids: list[str]


class LocalReflectionPhaseRunProvider:
    """Materialize local phase run intents from monitored channel snapshots."""

    async def period_run_intents(
        self,
        *,
        period_start_utc: datetime,
    ) -> list[ReflectionPhaseRunIntent]:
        """Return local calendar-compatible intents for one phase period."""

        input_set = await collect_reflection_inputs(
            lookback_hours=REFLECTION_PHASE_LOOKBACK_HOURS,
            now=period_start_utc,
            allow_fallback=False,
        )
        intents = build_phase_run_intents(
            period_start_utc=period_start_utc,
            eligible_scopes=input_set.selected_scopes,
            phase_period_seconds=REFLECTION_WORKER_INTERVAL_SECONDS,
            max_slots_per_period=REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD,
            min_slot_spacing_seconds=REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS,
            prompt_version=READONLY_REFLECTION_PROMPT_VERSION,
        )
        return intents

    async def expected_hourly_runs_for_character_local_date(
        self,
        *,
        character_local_date: str,
    ) -> list[ExpectedDailyChannelHourlyRuns]:
        """Return expected hourly run ids from local phase materialization."""

        start_utc_iso, end_utc_iso = local_date_bounds_to_storage_utc_iso(
            character_local_date,
        )
        start_utc = parse_storage_utc_datetime(start_utc_iso)
        end_utc = parse_storage_utc_datetime(end_utc_iso)
        period_start = _reflection_phase_period_start(start_utc)
        expected_by_scope: dict[str, ExpectedDailyChannelHourlyRuns] = {}
        while period_start < end_utc:
            intents = await self.period_run_intents(
                period_start_utc=period_start,
            )
            for intent in intents:
                due_at = _intent_datetime(intent, "due_at")
                if due_at < start_utc or due_at >= end_utc:
                    continue
                channel_scope = await _collect_phase_scope_input(
                    intent=intent,
                    now=due_at,
                )
                run_ids = _expected_hourly_run_ids_for_scope(
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
            period_start += timedelta(seconds=REFLECTION_WORKER_INTERVAL_SECONDS)

        expected_rows = list(expected_by_scope.values())
        for expected in expected_rows:
            expected.expected_run_ids = sorted(set(expected.expected_run_ids))
        return expected_rows


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _reflection_phase_period_start(value: datetime) -> datetime:
    """Return the UTC phase-period boundary for a storage timestamp."""

    if value.tzinfo is None:
        raise ValueError("reflection phase timestamp must be timezone-aware")
    value_utc = value.astimezone(timezone.utc)
    timestamp_seconds = int(value_utc.timestamp())
    period_offset = timestamp_seconds % REFLECTION_WORKER_INTERVAL_SECONDS
    period_start = value_utc - timedelta(seconds=period_offset)
    period_start = period_start.replace(microsecond=0)
    return period_start


def _intent_datetime(intent: ReflectionPhaseRunIntent, field_name: str) -> datetime:
    """Parse a UTC datetime field from a phase run intent."""

    timestamp = parse_storage_utc_datetime(str(intent[field_name]))
    return timestamp


def _due_phase_intents(
    *,
    now: datetime,
    intents: list[ReflectionPhaseRunIntent],
    executed_run_ids: set[str],
) -> list[ReflectionPhaseRunIntent]:
    """Return due, unexecuted phase intents ordered by due time."""

    due_intents = [
        intent
        for intent in intents
        if str(intent["run_id"]) not in executed_run_ids
        if _intent_datetime(intent, "due_at") <= now
    ]
    due_intents.sort(key=lambda intent: _intent_datetime(intent, "due_at"))
    return due_intents


def _next_reflection_phase_wait_seconds(
    *,
    now: datetime,
    period_start_utc: datetime,
    intents: list[ReflectionPhaseRunIntent],
    executed_run_ids: set[str],
) -> int:
    """Return seconds until the next phase intent or period boundary."""

    unexecuted_due_times = [
        _intent_datetime(intent, "due_at")
        for intent in intents
        if str(intent["run_id"]) not in executed_run_ids
    ]
    if any(due_at <= now for due_at in unexecuted_due_times):
        wait_seconds = 0
        return wait_seconds
    next_due_times = [
        due_at
        for due_at in unexecuted_due_times
        if due_at > now
    ]
    if next_due_times:
        next_due_at = min(next_due_times)
    else:
        next_due_at = period_start_utc + timedelta(
            seconds=REFLECTION_WORKER_INTERVAL_SECONDS,
        )
    wait_seconds = max(0, int((next_due_at - now).total_seconds()))
    return wait_seconds


async def run_hourly_reflection_cycle(
    *,
    now: datetime | None = None,
    dry_run: bool,
) -> ReflectionWorkerResult:
    """Run the public hourly reflection production pass."""

    result = await _run_hourly_reflection_cycle(
        now=now,
        dry_run=dry_run,
        is_primary_interaction_busy=lambda: False,
    )
    return result


async def run_daily_channel_reflection_cycle(
    *,
    character_local_date: str,
    dry_run: bool,
) -> ReflectionWorkerResult:
    """Run the public per-channel daily reflection production pass."""

    result = await _run_daily_channel_reflection_cycle(
        character_local_date=character_local_date,
        dry_run=dry_run,
        is_primary_interaction_busy=lambda: False,
    )
    return result


async def run_daily_interaction_style_update(
    *,
    character_local_date: str,
    dry_run: bool,
) -> ReflectionWorkerResult:
    """Run the public daily interaction-style update pass."""

    result = await _run_daily_interaction_style_update(
        character_local_date=character_local_date,
        dry_run=dry_run,
        is_primary_interaction_busy=lambda: False,
    )
    return result


async def execute_reflection_phase_intent(
    *,
    intent: ReflectionPhaseRunIntent,
    now: datetime,
    dry_run: bool,
    is_primary_interaction_busy: Callable[[], bool],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None,
    pipeline_coordinator: PipelineCoordinator | None = None,
) -> list[ReflectionWorkerResult]:
    """Execute one reflection phase intent through the production seam."""

    results = await _run_reflection_phase_intent(
        intent=intent,
        now=now,
        dry_run=dry_run,
        is_primary_interaction_busy=is_primary_interaction_busy,
        adapter_registry_provider=adapter_registry_provider,
        pipeline_coordinator=pipeline_coordinator,
    )
    return results


async def collect_phase_scope_input_for_intent(
    *,
    intent: ReflectionPhaseRunIntent,
    now: datetime,
) -> ReflectionScopeInput:
    """Fetch the fresh scope input for one reflection phase intent."""

    channel_scope = await _collect_phase_scope_input(intent=intent, now=now)
    return channel_scope


def expected_hourly_run_ids_for_scope(
    *,
    channel_scope: ReflectionScopeInput,
    character_local_date: str,
    now: datetime,
) -> list[str]:
    """Build expected hourly run ids for one phase-selected scope."""

    run_ids = _expected_hourly_run_ids_for_scope(
        channel_scope=channel_scope,
        character_local_date=character_local_date,
        now=now,
    )
    return run_ids


def start_reflection_cycle_worker(
    *,
    is_primary_interaction_busy: Callable[[], bool],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None,
    phase_run_provider: Any | None = None,
    character_state_refresh_callback: Callable[[], Any] | None = None,
) -> ReflectionWorkerHandle:
    """Start the process-local reflection worker loop."""

    stop_event = asyncio.Event()
    task = asyncio.create_task(
        _reflection_worker_loop(
            stop_event=stop_event,
            is_primary_interaction_busy=is_primary_interaction_busy,
            adapter_registry_provider=adapter_registry_provider,
            phase_run_provider=phase_run_provider,
            character_state_refresh_callback=character_state_refresh_callback,
        )
    )
    handle = ReflectionWorkerHandle(task=task, stop_event=stop_event)
    logger.info("Reflection cycle worker started")
    return handle


async def stop_reflection_cycle_worker(handle: ReflectionWorkerHandle) -> None:
    """Stop a reflection worker handle created by the public starter."""

    handle.stop_event.set()
    task = handle.task
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except TimeoutError:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    logger.info("Reflection cycle worker stopped")
    await event_logging.record_worker_event(
        event_type="stopped",
        component="reflection_cycle.worker",
        worker_name="reflection_cycle",
        enabled=True,
        dry_run=False,
        run_kind="reflection_worker",
        status="stopped",
    )


async def _reflection_worker_loop(
    *,
    stop_event: asyncio.Event,
    is_primary_interaction_busy: Callable[[], bool],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None,
    phase_run_provider: Any | None = None,
    character_state_refresh_callback: Callable[[], Any] | None = None,
) -> None:
    """Run reflection scheduling ticks until stopped."""

    effective_phase_provider = phase_run_provider or LocalReflectionPhaseRunProvider()
    current_period_start: datetime | None = None
    executed_phase_run_ids: set[str] = set()
    executed_period_maintenance_keys: set[tuple[datetime, str, str]] = set()
    await event_logging.record_worker_event(
        event_type="started",
        component="reflection_cycle.worker",
        worker_name="reflection_cycle",
        enabled=True,
        dry_run=False,
        run_kind="reflection_worker",
        status="started",
    )
    while not stop_event.is_set():
        now = storage_utc_now()
        period_start = _reflection_phase_period_start(now)
        if current_period_start != period_start:
            current_period_start = period_start
            executed_phase_run_ids = set()
            executed_period_maintenance_keys = set()
        try:
            await _run_worker_tick(
                now=now,
                is_primary_interaction_busy=is_primary_interaction_busy,
                adapter_registry_provider=adapter_registry_provider,
                phase_run_provider=effective_phase_provider,
                executed_phase_run_ids=executed_phase_run_ids,
                executed_period_maintenance_keys=(
                    executed_period_maintenance_keys
                ),
                character_state_refresh_callback=(
                    character_state_refresh_callback
                ),
            )
        except Exception as exc:
            logger.exception(f"Reflection worker tick failed: {exc}")
            await event_logging.record_runtime_error_event(
                component="reflection_cycle.worker",
                error_class=type(exc).__name__,
                error_preview=str(exc),
                stack_fingerprint="reflection_worker_tick",
                top_frame_module=__name__,
                recovered=True,
            )
        now = storage_utc_now()
        period_start = _reflection_phase_period_start(now)
        if current_period_start != period_start:
            current_period_start = period_start
            executed_phase_run_ids = set()
            executed_period_maintenance_keys = set()
        try:
            intents = await effective_phase_provider.period_run_intents(
                period_start_utc=period_start,
            )
        except Exception as exc:
            logger.exception(f"Reflection phase wait planning failed: {exc}")
            await event_logging.record_runtime_error_event(
                component="reflection_cycle.worker",
                error_class=type(exc).__name__,
                error_preview=str(exc),
                stack_fingerprint="reflection_phase_wait_planning",
                top_frame_module=__name__,
                recovered=True,
            )
            intents = []
        timeout_seconds = _next_reflection_phase_wait_seconds(
            now=now,
            period_start_utc=period_start,
            intents=intents,
            executed_run_ids=executed_phase_run_ids,
        )
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            continue


async def _run_worker_tick(
    *,
    now: datetime,
    is_primary_interaction_busy: Callable[[], bool],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None,
    phase_run_provider: Any | None = None,
    executed_phase_run_ids: set[str] | None = None,
    executed_period_maintenance_keys: set[tuple[datetime, str, str]] | None = None,
    character_state_refresh_callback: Callable[[], Any] | None = None,
) -> list[Any]:
    """Run one scheduled worker tick in approved priority order."""

    if is_primary_interaction_busy():
        logger.info("Reflection cycle tick skipped: primary interaction busy")
        return_value: list[Any] = [
            ReflectionWorkerResult(
                run_kind="reflection_tick",
                dry_run=False,
                deferred=True,
                defer_reason="primary interaction busy",
            )
        ]
        await _record_reflection_worker_result(return_value[0])
        return return_value

    results: list[Any] = []
    period_start = _reflection_phase_period_start(now)
    if phase_run_provider is None:
        hourly_result = await _run_hourly_reflection_cycle(
            now=now,
            dry_run=False,
            is_primary_interaction_busy=is_primary_interaction_busy,
        )
        results.append(hourly_result)
        await _record_reflection_worker_result(hourly_result)
    else:
        intents = await phase_run_provider.period_run_intents(
            period_start_utc=period_start,
        )
        executed_run_ids = executed_phase_run_ids
        if executed_run_ids is None:
            executed_run_ids = set()
        due_intents = _due_phase_intents(
            now=now,
            intents=intents,
            executed_run_ids=executed_run_ids,
        )
        for intent in due_intents:
            try:
                phase_results = await _run_reflection_phase_intent(
                    intent=intent,
                    now=now,
                    dry_run=False,
                    is_primary_interaction_busy=is_primary_interaction_busy,
                    adapter_registry_provider=adapter_registry_provider,
                )
            except Exception as exc:
                logger.exception(f"Reflection phase intent failed: {exc}")
                await event_logging.record_runtime_error_event(
                    component="reflection_cycle.worker",
                    error_class=type(exc).__name__,
                    error_preview=str(exc),
                    stack_fingerprint="reflection_phase_intent",
                    top_frame_module=__name__,
                    recovered=True,
                )
                phase_results = [
                    ReflectionWorkerResult(
                        run_kind="reflection_phase_slot",
                        dry_run=False,
                        failed_count=1,
                        defer_reason=f"{type(exc).__name__}: {exc}",
                    )
                ]
            finally:
                executed_run_ids.add(str(intent["run_id"]))
            results.extend(phase_results)
            for phase_result in phase_results:
                await _record_reflection_worker_result(phase_result)
            if is_primary_interaction_busy():
                break

    now_utc_iso = normalize_storage_utc_iso(now.isoformat())
    local_time_context = local_time_context_from_storage_utc(now_utc_iso)
    local_datetime = local_time_context["current_local_datetime"]
    current_local_date = date.fromisoformat(local_datetime[:10])
    previous_local_date = (current_local_date - timedelta(days=1)).isoformat()
    daily_maintenance_key = (
        period_start,
        previous_local_date,
        _DAILY_STYLE_MAINTENANCE_GATE,
    )
    daily_maintenance_done = (
        executed_period_maintenance_keys is not None
        and daily_maintenance_key in executed_period_maintenance_keys
    )
    if (
        _local_time_is_after(local_datetime, REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME)
        and not daily_maintenance_done
    ):
        if is_primary_interaction_busy():
            return results
        if executed_period_maintenance_keys is not None:
            executed_period_maintenance_keys.add(daily_maintenance_key)
        daily_result = await _run_daily_channel_reflection_cycle(
            character_local_date=previous_local_date,
            dry_run=False,
            is_primary_interaction_busy=is_primary_interaction_busy,
            phase_run_provider=phase_run_provider,
        )
        results.append(daily_result)
        await _record_reflection_worker_result(daily_result)
        if is_primary_interaction_busy():
            return results
        style_result = await _run_daily_interaction_style_update(
            character_local_date=previous_local_date,
            dry_run=False,
            is_primary_interaction_busy=is_primary_interaction_busy,
        )
        results.append(style_result)
        await _record_reflection_worker_result(style_result)

    promotion_maintenance_key = (
        period_start,
        previous_local_date,
        _GLOBAL_PROMOTION_MAINTENANCE_GATE,
    )
    promotion_maintenance_done = (
        executed_period_maintenance_keys is not None
        and promotion_maintenance_key in executed_period_maintenance_keys
    )
    if (
        _local_time_is_after(
            local_datetime,
            REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
        )
        and not promotion_maintenance_done
    ):
        if is_primary_interaction_busy():
            return results
        if executed_period_maintenance_keys is not None:
            executed_period_maintenance_keys.add(promotion_maintenance_key)
        promotion_result = await _run_global_reflection_promotion(
            character_local_date=previous_local_date,
            dry_run=False,
            enable_memory_writes=(
                REFLECTION_LORE_PROMOTION_ENABLED
                or REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED
            ),
            is_primary_interaction_busy=is_primary_interaction_busy,
        )
        results.append(promotion_result)
        await _record_reflection_worker_result(promotion_result)
        if is_primary_interaction_busy():
            return results
        if (
            GLOBAL_CHARACTER_GROWTH_PASS_ENABLED
            and promotion_result.succeeded_count > 0
        ):
            growth_result = await run_global_character_growth_pass(
                character_local_date=previous_local_date,
                dry_run=False,
                enable_trait_writes=True,
            )
            results.append(growth_result)

    settling_local_date = settling_local_date_for_due_affect_settling(
        local_datetime,
    )
    affect_maintenance_key = (
        period_start,
        settling_local_date,
        _DAILY_AFFECT_SETTLING_MAINTENANCE_GATE,
    )
    affect_maintenance_done = (
        executed_period_maintenance_keys is not None
        and affect_maintenance_key in executed_period_maintenance_keys
    )
    if settling_local_date and not affect_maintenance_done:
        if is_primary_interaction_busy():
            return results
        if executed_period_maintenance_keys is not None:
            executed_period_maintenance_keys.add(affect_maintenance_key)
        affect_result = await _run_daily_affect_settling(
            settling_local_date=settling_local_date,
            dry_run=False,
            enable_character_state_write=True,
            character_state_refresh_callback=character_state_refresh_callback,
        )
        results.append(affect_result)
        await _record_reflection_worker_result(affect_result)
    return results


async def _run_hourly_reflection_cycle(
    *,
    now: datetime | None,
    dry_run: bool,
    is_primary_interaction_busy: Callable[[], bool],
) -> ReflectionWorkerResult:
    """Collect monitored channels and persist due hourly reflection runs."""

    input_set = await collect_reflection_inputs(
        lookback_hours=24,
        now=now,
        allow_fallback=False,
    )
    result = ReflectionWorkerResult(
        run_kind=REFLECTION_RUN_KIND_HOURLY,
        dry_run=dry_run,
    )
    hourly_scopes: list[ReflectionScopeInput] = []
    for channel_scope in input_set.selected_scopes:
        hourly_scopes.extend(_split_scope_into_hourly_scopes(channel_scope))

    if not hourly_scopes:
        result.skipped_count = 1
        result.defer_reason = "no monitored message-bearing hourly slots"
        logger.info(
            "Reflection hourly pass skipped: no monitored message-bearing slots"
        )
        return result

    candidate_pairs = [
        (
            hourly_scope,
            repository.hourly_run_id(
                scope_ref=repository.channel_scope_ref_for_hourly(
                    hourly_scope.scope_ref,
                ),
                hour_start=repository.hour_start_for_scope(
                    hourly_scope,
                ).isoformat(),
            ),
        )
        for hourly_scope in hourly_scopes
    ]
    existing_ids = await repository.existing_run_ids([
        run_id
        for _, run_id in candidate_pairs
    ])
    due_pairs = [
        (scope, run_id)
        for scope, run_id in candidate_pairs
        if run_id not in existing_ids
    ][:REFLECTION_HOURLY_SLOTS_PER_TICK]

    if not due_pairs:
        result.skipped_count = 1
        result.defer_reason = "no due hourly slots"
        logger.info("Reflection hourly pass skipped: no due hourly slots")
        return result

    for hourly_scope, _ in due_pairs:
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break
        run_doc = await _run_one_hourly_slot(
            hourly_scope=hourly_scope,
            dry_run=dry_run,
        )
        _apply_doc_status(result, run_doc)
        result.run_ids.append(str(run_doc["run_id"]))
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break

    logger.info(
        "Reflection hourly pass complete: "
        f"processed={result.processed_count} succeeded={result.succeeded_count} "
        f"failed={result.failed_count} skipped={result.skipped_count} "
        f"deferred={result.deferred} reason={result.defer_reason}"
    )
    return result


_DEFAULT_HOURLY_REFLECTION_CYCLE = _run_hourly_reflection_cycle


async def _run_reflection_phase_intent(
    *,
    intent: ReflectionPhaseRunIntent,
    now: datetime,
    dry_run: bool,
    is_primary_interaction_busy: Callable[[], bool],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None,
    pipeline_coordinator: PipelineCoordinator | None = None,
) -> list[ReflectionWorkerResult]:
    """Execute one calendar-shaped reflection phase intent."""

    channel_scope = await _collect_phase_scope_input(intent=intent, now=now)
    results: list[ReflectionWorkerResult] = []
    hourly_result = await _run_hourly_reflection_for_scope(
        now=now,
        channel_scope=channel_scope,
        dry_run=dry_run,
        is_primary_interaction_busy=is_primary_interaction_busy,
    )
    results.append(hourly_result)
    if channel_scope.channel_type == "group":
        group_result = await _run_group_self_cognition_review_for_scope(
            now=now,
            channel_scope=channel_scope,
            is_primary_interaction_busy=is_primary_interaction_busy,
            adapter_registry_provider=adapter_registry_provider,
            pipeline_coordinator=pipeline_coordinator,
        )
        results.append(group_result)
    return results


async def _collect_phase_scope_input(
    *,
    intent: ReflectionPhaseRunIntent,
    now: datetime,
) -> ReflectionScopeInput:
    """Fetch fresh bounded messages for the source scope in a phase intent."""

    source_scope = intent["source_scope"]
    channel_scope = await collect_reflection_scope_input(
        platform=str(source_scope["platform"]),
        platform_channel_id=str(source_scope["platform_channel_id"]),
        channel_type=str(source_scope["channel_type"]),
        scope_ref=str(source_scope["scope_ref"]),
        lookback_hours=REFLECTION_PHASE_LOOKBACK_HOURS,
        now=now,
    )
    return channel_scope


async def _run_hourly_reflection_for_scope(
    *,
    now: datetime,
    channel_scope: ReflectionScopeInput,
    dry_run: bool,
    is_primary_interaction_busy: Callable[[], bool],
) -> ReflectionWorkerResult:
    """Run at most one due closed-hour reflection slot for one channel."""

    result = ReflectionWorkerResult(
        run_kind=REFLECTION_RUN_KIND_HOURLY,
        dry_run=dry_run,
    )
    due_hourly_scopes = _closed_hourly_scopes_for_scope(
        channel_scope=channel_scope,
        now=now,
    )
    if not due_hourly_scopes:
        result.skipped_count = 1
        result.defer_reason = "no due hourly slots"
        return result

    candidate_pairs = [
        (
            hourly_scope,
            repository.hourly_run_id(
                scope_ref=repository.channel_scope_ref_for_hourly(
                    hourly_scope.scope_ref,
                ),
                hour_start=repository.hour_start_for_scope(
                    hourly_scope,
                ).isoformat(),
            ),
        )
        for hourly_scope in due_hourly_scopes
    ]
    existing_ids = await repository.existing_run_ids([
        run_id
        for _, run_id in candidate_pairs
    ])
    due_pairs = [
        (scope, run_id)
        for scope, run_id in candidate_pairs
        if run_id not in existing_ids
    ][:1]

    if not due_pairs:
        result.skipped_count = 1
        result.defer_reason = "no due hourly slots"
        return result

    if is_primary_interaction_busy():
        result.deferred = True
        result.defer_reason = "primary interaction busy"
        return result

    hourly_scope, _ = due_pairs[0]
    run_doc = await _run_one_hourly_slot(
        hourly_scope=hourly_scope,
        dry_run=dry_run,
    )
    _apply_doc_status(result, run_doc)
    result.run_ids.append(str(run_doc["run_id"]))
    return result


async def _run_group_self_cognition_review_for_scope(
    *,
    now: datetime,
    channel_scope: ReflectionScopeInput,
    is_primary_interaction_busy: Callable[[], bool],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None,
    pipeline_coordinator: PipelineCoordinator | None = None,
) -> ReflectionWorkerResult:
    """Run one reviewed-window group self-cognition case for a channel."""

    result = ReflectionWorkerResult(
        run_kind="group_self_cognition_review",
        dry_run=False,
    )

    if is_primary_interaction_busy():
        result.deferred = True
        result.defer_reason = "primary interaction busy"
        return result

    if channel_scope.channel_type != "group":
        result.skipped_count = 1
        result.defer_reason = "not a group scope"
        return result

    pipeline_run_handle: PipelineRunHandle | None = None
    if pipeline_coordinator is not None:
        pipeline_scope = PipelineScope(
            platform=channel_scope.platform,
            platform_channel_id=channel_scope.platform_channel_id,
            channel_type=channel_scope.channel_type,
        )
        admission = await pipeline_coordinator.start_run(
            scope=pipeline_scope,
            owner="reflection_cycle.group_review",
            precedence="background",
            run_kind="group_self_cognition_review",
        )
        if not admission.admitted:
            result.deferred = True
            result.defer_reason = admission.defer_reason or ""
            return result
        pipeline_run_handle = admission.handle

    try:
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled(
                "before_group_review_context",
            )

        if is_self_cognition_sleep_period(now):
            result.skipped_count = 1
            result.defer_reason = "self-cognition sleep period"
            return result

        character_profile = await get_character_profile()
        windows = _group_activity_windows_for_scope(
            channel_scope=channel_scope,
            now=now,
            character_profile=character_profile,
        )
        unreviewed_windows = await _unreviewed_group_windows(windows)
        if not unreviewed_windows:
            result.skipped_count = 1
            result.defer_reason = "no group review cases"
            return result

        selected_window = unreviewed_windows[0]
        older_windows = unreviewed_windows[1:]
        for window in older_windows:
            await _record_group_review_window(
                window=window,
                now=now,
                status="coalesced_skipped",
                case_id=None,
                selected_route=None,
                dispatch_status=None,
                skip_reason=GROUP_REVIEW_COALESCED_SKIP_REASON,
            )

        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled(
                "before_group_review_cases",
            )
        cases = await self_cognition_sources.collect_group_review_cases(
            now=now,
            character_profile=character_profile,
            windows=[selected_window],
            max_cases=REFLECTION_PHASE_GROUPS_PER_SLOT,
        )
        if not cases:
            await _record_group_review_window(
                window=selected_window,
                now=now,
                status="stale_skipped",
                case_id=None,
                selected_route=None,
                dispatch_status=None,
                skip_reason=GROUP_REVIEW_STALE_SKIP_REASON,
            )
            result.skipped_count = 1
            result.defer_reason = "no group review cases"
            return result

        selected_cases = cases[:REFLECTION_PHASE_GROUPS_PER_SLOT]

        async def _collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
            max_cases = int(kwargs["max_cases"])
            limited_cases = selected_cases[:max_cases]
            return limited_cases

        self_result = await self_cognition_worker.run_self_cognition_worker_tick(
            now=now,
            is_primary_interaction_busy=is_primary_interaction_busy,
            character_profile=character_profile,
            collect_cases_func=_collect_cases,
            adapter_registry_provider=adapter_registry_provider,
            pipeline_run_handle=pipeline_run_handle,
            max_cases=REFLECTION_PHASE_GROUPS_PER_SLOT,
        )
        result.processed_count = self_result.processed_count
        result.succeeded_count = max(
            0,
            self_result.processed_count - self_result.failed_count,
        )
        result.failed_count = self_result.failed_count
        result.skipped_count = self_result.skipped_count
        result.deferred = self_result.deferred
        result.defer_reason = self_result.defer_reason

        selected_case = selected_cases[0]
        if self_result.deferred:
            return result

        if _group_review_case_target_binding_failed(selected_case):
            await _record_group_review_window(
                window=selected_window,
                now=now,
                status="target_binding_failed",
                case_id=str(selected_case["case_id"]),
                selected_route=None,
                dispatch_status="target_binding_failed",
                skip_reason=_group_review_target_binding_skip_reason(
                    selected_case
                ),
            )
        elif self_result.failed_count > 0:
            await _record_group_review_window(
                window=selected_window,
                now=now,
                status="review_failed",
                case_id=str(selected_case["case_id"]),
                selected_route=None,
                dispatch_status=None,
                skip_reason=GROUP_REVIEW_FAILED_SKIP_REASON,
            )
        else:
            await _record_group_review_window(
                window=selected_window,
                now=now,
                status="reviewed",
                case_id=str(selected_case["case_id"]),
                selected_route=None,
                dispatch_status=None,
                skip_reason=None,
            )
    except PipelineCancelled as exc:
        result.deferred = True
        result.defer_reason = exc.cancellation.reason
        return result
    finally:
        if pipeline_run_handle is not None:
            await pipeline_run_handle.__aexit__(None, None, None)
    return result


def _group_review_case_target_binding_failed(case: dict[str, Any]) -> bool:
    """Return whether a selected group case failed deterministic binding."""

    if case.get("target_binding_status") == "failed":
        return_value = True
        return return_value
    if not isinstance(case.get("delivery_target"), dict):
        return_value = True
        return return_value
    return_value = False
    return return_value


def _group_review_target_binding_skip_reason(case: dict[str, Any]) -> str:
    """Return a stable skip reason for target-binding-failed ledger rows."""

    failure = case.get("target_binding_failure")
    if isinstance(failure, dict):
        reason = str(failure.get("reason", "") or "")
        if reason:
            return_value = reason
            return return_value
    return_value = "target_binding_failed"
    return return_value


def _closed_hourly_scopes_for_scope(
    *,
    channel_scope: ReflectionScopeInput,
    now: datetime,
) -> list[ReflectionScopeInput]:
    """Return message-bearing hourly scopes whose hour has fully closed."""

    hourly_scopes = _split_scope_into_hourly_scopes(channel_scope)
    closed_scopes = [
        hourly_scope
        for hourly_scope in hourly_scopes
        if repository.hour_start_for_scope(hourly_scope) + timedelta(hours=1) <= now
    ]
    return closed_scopes


def _expected_hourly_run_ids_for_scope(
    *,
    channel_scope: ReflectionScopeInput,
    character_local_date: str,
    now: datetime,
) -> list[str]:
    """Build expected hourly run ids for one channel and local day."""

    run_ids: list[str] = []
    hourly_scopes = _closed_hourly_scopes_for_scope(
        channel_scope=channel_scope,
        now=now,
    )
    for hourly_scope in hourly_scopes:
        hour_start = repository.hour_start_for_scope(hourly_scope)
        if (
            repository.character_local_date_for_utc(hour_start)
            != character_local_date
        ):
            continue
        run_id = repository.hourly_run_id(
            scope_ref=repository.channel_scope_ref_for_hourly(
                hourly_scope.scope_ref,
            ),
            hour_start=hour_start.isoformat(),
        )
        run_ids.append(run_id)
    return run_ids


def _group_activity_windows_for_scope(
    *,
    channel_scope: ReflectionScopeInput,
    now: datetime,
    character_profile: dict[str, Any],
) -> list[GroupActivityWindow]:
    """Build newest-first group activity windows for one selected scope."""

    if not channel_scope.messages:
        return []
    window_start = parse_storage_utc_datetime(channel_scope.first_timestamp)
    windows = build_group_activity_windows(
        scope=channel_scope,
        window_start=window_start,
        window_end=now,
        now=now,
        character_global_user_id=(
            str(character_profile.get("global_user_id", "") or "")
            or CHARACTER_GLOBAL_USER_ID
        ),
        platform_bot_id=str(character_profile.get("platform_bot_id", "") or ""),
    )
    windows.sort(key=lambda item: item.window_start, reverse=True)
    return windows


def _daily_input_set_from_expected(
    *,
    character_local_date: str,
    channel_scopes: list[ReflectionScopeInput],
) -> ReflectionInputSet:
    """Build a compact daily input set from provider materialized scopes."""

    if channel_scopes:
        timestamps = [
            timestamp
            for scope in channel_scopes
            for timestamp in (scope.first_timestamp, scope.last_timestamp)
            if timestamp
        ]
    else:
        timestamps = []
    if timestamps:
        effective_start = min(timestamps)
        effective_end = max(timestamps)
    else:
        effective_start = f"{character_local_date}T00:00:00+00:00"
        effective_end = f"{character_local_date}T23:59:59+00:00"
    input_set = ReflectionInputSet(
        lookback_hours=REFLECTION_PHASE_LOOKBACK_HOURS,
        requested_start=effective_start,
        requested_end=effective_end,
        effective_start=effective_start,
        effective_end=effective_end,
        fallback_used=False,
        fallback_reason="",
        selected_scopes=channel_scopes,
        query_diagnostics={},
    )
    return input_set


async def _unreviewed_group_windows(
    windows: list[GroupActivityWindow],
) -> list[GroupActivityWindow]:
    """Suppress windows that already have terminal reviewed-window rows."""

    unreviewed: list[GroupActivityWindow] = []
    for window in windows:
        existing = await find_self_cognition_group_review_window(
            window.source_id,
        )
        if existing is not None:
            continue
        unreviewed.append(window)
    return unreviewed


async def _record_group_review_window(
    *,
    window: GroupActivityWindow,
    now: datetime,
    status: str,
    case_id: str | None,
    selected_route: str | None,
    dispatch_status: str | None,
    skip_reason: str | None,
) -> None:
    """Record one terminal group-review source-window ledger row."""

    row: SelfCognitionGroupReviewWindowDoc = {
        "source_id": window.source_id,
        "case_id": case_id,
        "scope_ref": window.scope_ref,
        "platform": window.platform,
        "platform_channel_id": window.platform_channel_id,
        "channel_type": "group",
        "window_start": normalize_storage_utc_iso(
            window.window_start.isoformat(),
        ),
        "window_end": normalize_storage_utc_iso(window.window_end.isoformat()),
        "status": status,
        "reviewed_at": normalize_storage_utc_iso(now.isoformat()),
        "selected_route": selected_route,
        "dispatch_status": dispatch_status,
        "skip_reason": skip_reason,
    }
    await upsert_self_cognition_group_review_window(row)


async def _run_daily_channel_reflection_cycle(
    *,
    character_local_date: str,
    dry_run: bool,
    is_primary_interaction_busy: Callable[[], bool],
    phase_run_provider: Any | None = None,
) -> ReflectionWorkerResult:
    """Persist due daily-channel reflection runs for monitored scopes."""

    result = ReflectionWorkerResult(
        run_kind=REFLECTION_RUN_KIND_DAILY_CHANNEL,
        dry_run=dry_run,
    )
    if phase_run_provider is None:
        input_set = await collect_reflection_inputs(
            lookback_hours=24,
            now=None,
            allow_fallback=False,
        )
        expected_rows = [
            ExpectedDailyChannelHourlyRuns(
                channel_scope=channel_scope,
                expected_run_ids=[],
            )
            for channel_scope in input_set.selected_scopes
        ]
    else:
        expected_rows = (
            await phase_run_provider.expected_hourly_runs_for_character_local_date(
                character_local_date=character_local_date,
            )
        )
        input_set = None

    for expected in expected_rows:
        channel_scope = expected.channel_scope
        run_id = repository.daily_channel_run_id(
            scope_ref=channel_scope.scope_ref,
            character_local_date=character_local_date,
        )
        existing_ids = await repository.existing_run_ids([run_id])
        if run_id in existing_ids:
            result.skipped_count += 1
            continue
        hourly_docs = await repository.hourly_runs_for_channel_day(
            scope_ref=channel_scope.scope_ref,
            character_local_date=character_local_date,
        )
        terminal_docs = [
            document
            for document in hourly_docs
            if repository.is_terminal_run(document)
        ]
        expected_run_ids = list(expected.expected_run_ids)
        if expected_run_ids:
            terminal_run_ids = {
                str(document["run_id"])
                for document in terminal_docs
            }
            missing_run_ids = [
                run_id
                for run_id in expected_run_ids
                if run_id not in terminal_run_ids
            ]
            if missing_run_ids:
                result.skipped_count += 1
                result.defer_reason = "missing expected hourly reflection runs"
                result.validation_warnings.append(
                    "missing_expected_hourly_runs "
                    f"scope_ref={channel_scope.scope_ref} "
                    f"missing_count={len(missing_run_ids)}"
                )
                continue
        if not terminal_docs:
            result.skipped_count += 1
            continue
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break
        validation_warnings: list[str] = []
        if len(terminal_docs) < len(hourly_docs):
            validation_warnings.append(
                "partial_hourly_input "
                f"terminal_count={len(terminal_docs)} "
                f"total_count={len(hourly_docs)}"
            )
        if input_set is None:
            input_set = _daily_input_set_from_expected(
                character_local_date=character_local_date,
                channel_scopes=[
                    row.channel_scope
                    for row in expected_rows
                ],
            )
        run_doc = await _run_one_daily_channel(
            channel_scope=channel_scope,
            input_set=input_set,
            hourly_docs=terminal_docs,
            character_local_date=character_local_date,
            dry_run=dry_run,
            additional_validation_warnings=validation_warnings,
        )
        _apply_doc_status(result, run_doc)
        result.run_ids.append(str(run_doc["run_id"]))
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break

    if not expected_rows:
        result.skipped_count = 1
        result.defer_reason = "no monitored channels"
    logger.info(
        "Reflection daily-channel pass complete: "
        f"date={character_local_date} processed={result.processed_count} "
        f"succeeded={result.succeeded_count} failed={result.failed_count} "
        f"skipped={result.skipped_count} deferred={result.deferred} "
        f"reason={result.defer_reason}"
    )
    return result


async def _run_one_hourly_slot(
    *,
    hourly_scope: ReflectionScopeInput,
    dry_run: bool,
) -> CharacterReflectionRunDoc:
    """Run or skip one hourly LLM call and persist its run document."""

    if dry_run:
        llm_result = build_skipped_hourly_result(hourly_scope)
        status = repository.status_for_result(dry_run=True)
        run_doc = repository.build_hourly_run_document(
            scope=hourly_scope,
            result=llm_result,
            status=status,
            attempt_count=0,
        )
        await _upsert_reflection_run_with_event(run_doc)
        return run_doc

    attempt_count = 0
    try:
        llm_result, attempt_count = await _run_hourly_with_retry(hourly_scope)
    except Exception as exc:
        logger.exception(f"Hourly reflection slot failed: {exc}")
        run_doc = repository.build_hourly_run_document(
            scope=hourly_scope,
            result=None,
            status=REFLECTION_STATUS_FAILED,
            attempt_count=max(1, attempt_count),
            error=f"{type(exc).__name__}: {exc}",
        )
        await _upsert_reflection_run_with_event(run_doc)
        return run_doc

    run_doc = repository.build_hourly_run_document(
        scope=hourly_scope,
        result=llm_result,
        status=repository.status_for_result(dry_run=False),
        attempt_count=attempt_count,
    )
    await _upsert_reflection_run_with_event(run_doc)
    return run_doc


async def _run_one_daily_channel(
    *,
    channel_scope: ReflectionScopeInput,
    input_set: Any,
    hourly_docs: list[CharacterReflectionRunDoc],
    character_local_date: str,
    dry_run: bool,
    additional_validation_warnings: list[str] | None = None,
) -> CharacterReflectionRunDoc:
    """Run or skip one per-channel daily LLM call and persist its document."""

    extra_warnings = additional_validation_warnings or []
    hourly_scopes = [
        _hourly_scope_from_doc(document)
        for document in hourly_docs
    ]
    hourly_results = [
        _hourly_result_from_doc(document)
        for document in hourly_docs
    ]
    channel_input_set = _channel_input_set(
        input_set=input_set,
        hourly_scopes=hourly_scopes,
        channel_scope=channel_scope,
    )
    if dry_run:
        daily_result = build_skipped_daily_result(
            input_set=channel_input_set,
            channel_scope=channel_scope,
            hourly_results=hourly_results,
        )
        status = repository.status_for_result(dry_run=True)
        run_doc = repository.build_daily_channel_run_document(
            channel_scope=channel_scope,
            hourly_docs=hourly_docs,
            result=daily_result,
            character_local_date=character_local_date,
            status=status,
            attempt_count=0,
        )
        _extend_validation_warnings(run_doc, extra_warnings)
        await _upsert_reflection_run_with_event(run_doc)
        return run_doc

    attempt_count = 0
    try:
        daily_result, attempt_count = await _run_daily_with_retry(
            input_set=channel_input_set,
            channel_scope=channel_scope,
            hourly_results=hourly_results,
        )
    except Exception as exc:
        logger.exception(f"Daily reflection channel failed: {exc}")
        run_doc = repository.build_daily_channel_run_document(
            channel_scope=channel_scope,
            hourly_docs=hourly_docs,
            result=None,
            character_local_date=character_local_date,
            status=REFLECTION_STATUS_FAILED,
            attempt_count=max(1, attempt_count),
            error=f"{type(exc).__name__}: {exc}",
        )
        _extend_validation_warnings(run_doc, extra_warnings)
        await _upsert_reflection_run_with_event(run_doc)
        return run_doc

    run_doc = repository.build_daily_channel_run_document(
        channel_scope=channel_scope,
        hourly_docs=hourly_docs,
        result=daily_result,
        character_local_date=character_local_date,
        status=repository.status_for_result(dry_run=False),
        attempt_count=attempt_count,
    )
    _extend_validation_warnings(run_doc, extra_warnings)
    await _upsert_reflection_run_with_event(run_doc)
    return run_doc


async def _run_hourly_with_retry(
    hourly_scope: ReflectionScopeInput,
) -> tuple[ReflectionLLMResult, int]:
    """Run one hourly LLM call with one external retry."""

    prompt = build_hourly_reflection_prompt(hourly_scope)
    attempt_count = 0
    last_error: BaseException | None = None
    for attempt in (1, 2):
        attempt_count = attempt
        try:
            llm_result = await run_hourly_reflection_llm(
                scope_ref=hourly_scope.scope_ref,
                prompt=prompt,
            )
            await _record_reflection_llm_stage(
                stage_name="hourly_reflection",
                run_kind=REFLECTION_RUN_KIND_HOURLY,
                prompt_chars=prompt.prompt_chars,
                output_chars=len(llm_result.raw_output),
                status="succeeded",
                retry_count=attempt - 1,
                validation_warnings=llm_result.validation_warnings,
            )
            return_value = (llm_result, attempt_count)
            return return_value
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                await _record_reflection_llm_stage(
                    stage_name="hourly_reflection",
                    run_kind=REFLECTION_RUN_KIND_HOURLY,
                    prompt_chars=prompt.prompt_chars,
                    output_chars=0,
                    status="failed",
                    retry_count=attempt - 1,
                    validation_warnings=[],
                )
                break
    if last_error is None:
        raise RuntimeError("hourly reflection failed without exception")
    raise last_error


def _extend_validation_warnings(
    document: CharacterReflectionRunDoc,
    warnings: list[str],
) -> None:
    """Append runtime validation warnings to a built run document."""

    if not warnings:
        return
    validation_warnings = list(document["validation_warnings"])
    validation_warnings.extend(warnings)
    document["validation_warnings"] = validation_warnings


async def _run_daily_with_retry(
    *,
    input_set: Any,
    channel_scope: ReflectionScopeInput,
    hourly_results: list[ReflectionLLMResult],
) -> tuple[DailySynthesisResult, int]:
    """Run one daily LLM call with one external retry."""

    prompt = build_daily_synthesis_prompt(
        input_set=input_set,
        channel_scope=channel_scope,
        hourly_results=hourly_results,
    )
    attempt_count = 0
    last_error: BaseException | None = None
    for attempt in (1, 2):
        attempt_count = attempt
        try:
            daily_result = await run_daily_synthesis_llm(prompt=prompt)
            await _record_reflection_llm_stage(
                stage_name="daily_synthesis",
                run_kind=REFLECTION_RUN_KIND_DAILY_CHANNEL,
                prompt_chars=prompt.prompt_chars,
                output_chars=len(daily_result.raw_output),
                status="succeeded",
                retry_count=attempt - 1,
                validation_warnings=daily_result.validation_warnings,
            )
            return_value = (daily_result, attempt_count)
            return return_value
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                await _record_reflection_llm_stage(
                    stage_name="daily_synthesis",
                    run_kind=REFLECTION_RUN_KIND_DAILY_CHANNEL,
                    prompt_chars=prompt.prompt_chars,
                    output_chars=0,
                    status="failed",
                    retry_count=attempt - 1,
                    validation_warnings=[],
                )
                break
    if last_error is None:
        raise RuntimeError("daily reflection failed without exception")
    raise last_error


def _hourly_scope_from_doc(
    document: CharacterReflectionRunDoc,
) -> ReflectionScopeInput:
    """Rebuild compact hourly scope metadata for daily prompt projection."""

    scope = document["scope"]
    hour_start = str(document.get("hour_start", "") or "")
    raw_output = document.get("output")
    if isinstance(raw_output, dict):
        hourly_scope_ref = str(raw_output.get("hourly_scope_ref", "") or "")
    else:
        hourly_scope_ref = ""
    if not hourly_scope_ref:
        hourly_scope_ref = str(document["run_id"])
    scope_input = ReflectionScopeInput(
        scope_ref=hourly_scope_ref,
        platform=str(scope["platform"]),
        platform_channel_id=str(scope["platform_channel_id"]),
        channel_type=str(scope["channel_type"]),
        assistant_message_count=0,
        user_message_count=0,
        total_message_count=0,
        first_timestamp=hour_start,
        last_timestamp=str(document.get("hour_end", "") or hour_start),
        messages=[],
    )
    return scope_input


def _hourly_result_from_doc(
    document: CharacterReflectionRunDoc,
) -> ReflectionLLMResult:
    """Rebuild compact hourly LLM result metadata for daily synthesis."""

    output = document.get("output", {})
    if not isinstance(output, dict):
        output = {}
    scope = _hourly_scope_from_doc(document)
    prompt = PromptBuildResult(
        system_prompt="",
        human_payload={},
        human_prompt="",
        prompt_chars=0,
        prompt_preview="",
        validation_warnings=[],
    )
    result = ReflectionLLMResult(
        scope_ref=scope.scope_ref,
        prompt=prompt,
        raw_output="",
        parsed_output=output,
        validation_warnings=list(document.get("validation_warnings", [])),
        llm_skipped=document.get("status") in {
            REFLECTION_STATUS_SKIPPED,
            REFLECTION_STATUS_FAILED,
        },
    )
    return result


def _apply_doc_status(
    result: ReflectionWorkerResult,
    document: CharacterReflectionRunDoc,
) -> None:
    """Update result counters from one persisted run document."""

    result.processed_count += 1
    status = str(document["status"])
    if status == "succeeded":
        result.succeeded_count += 1
    elif status == "failed":
        result.failed_count += 1
    else:
        result.skipped_count += 1


async def _record_reflection_worker_result(
    result: ReflectionWorkerResult,
) -> None:
    """Record one sanitized reflection worker pass summary."""

    if result.deferred:
        status = "deferred"
    elif result.failed_count > 0:
        status = "failed"
    elif result.succeeded_count > 0:
        status = "succeeded"
    else:
        status = "skipped"
    await event_logging.record_worker_event(
        event_type="tick",
        component="reflection_cycle.worker",
        worker_name="reflection_cycle",
        enabled=True,
        dry_run=result.dry_run,
        run_kind=result.run_kind,
        status=status,
        processed_count=result.processed_count,
        succeeded_count=result.succeeded_count,
        failed_count=result.failed_count,
        skipped_count=result.skipped_count,
        deferred=result.deferred,
        defer_reason=result.defer_reason,
        run_id=result.run_ids[0] if result.run_ids else "",
    )


async def _upsert_reflection_run_with_event(
    document: CharacterReflectionRunDoc,
) -> None:
    """Persist one reflection run and record the approved DB operation."""

    started_at = time.perf_counter()
    await repository.upsert_run(document)
    latency_ms = _elapsed_ms(started_at)
    await _record_reflection_db_operation(document, latency_ms=latency_ms)


async def _record_reflection_db_operation(
    document: CharacterReflectionRunDoc,
    *,
    latency_ms: int,
) -> None:
    """Record one approved reflection-run upsert outcome."""

    await event_logging.record_database_operation_event(
        component="reflection_cycle.worker",
        collection="character_reflection_runs",
        operation_kind="upsert_reflection_run",
        status=str(document["status"]),
        idempotency_result="upserted",
        latency_ms=latency_ms,
        document_ref=str(document["run_id"]),
        run_id=str(document["run_id"]),
    )


async def _record_reflection_llm_stage(
    *,
    stage_name: str,
    run_kind: str,
    prompt_chars: int,
    output_chars: int,
    status: str,
    retry_count: int,
    validation_warnings: list[str],
) -> None:
    """Record model-stage metadata and contract warnings for reflection."""

    parse_status = "valid"
    if status == "failed":
        parse_status = "failed"
    elif validation_warnings:
        parse_status = "warning"
    await event_logging.record_llm_stage_event(
        component="reflection_cycle.worker",
        stage_name=stage_name,
        route_name=run_kind,
        model_name="consolidation_route",
        status=status,
        prompt_chars=prompt_chars,
        output_chars=output_chars,
        parse_status=parse_status,
        retry_count=retry_count,
        json_repair_used=False,
    )
    if validation_warnings:
        await event_logging.record_model_contract_event(
            component="reflection_cycle.worker",
            stage_name=stage_name,
            violation_kind="validation_warning",
            missing_fields=[],
            invalid_fields=validation_warnings,
            repair_used=False,
            status="warning",
        )


def _local_time_is_after(local_datetime: str, value: str) -> bool:
    """Return whether the local clock is at or after a HH:MM gate."""

    hour_text, minute_text = value.split(":", maxsplit=1)
    gate_minutes = int(hour_text) * 60 + int(minute_text)
    local_hour_text, local_minute_text = local_datetime[11:16].split(
        ":",
        maxsplit=1,
    )
    current_minutes = int(local_hour_text) * 60 + int(local_minute_text)
    return_value = current_minutes >= gate_minutes
    return return_value
