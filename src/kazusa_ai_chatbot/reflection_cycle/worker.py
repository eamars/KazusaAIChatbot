"""Production reflection-cycle worker orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from kazusa_ai_chatbot.config import (
    CHARACTER_TIME_ZONE,
    REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME,
    REFLECTION_HOURLY_SLOTS_PER_TICK,
    REFLECTION_LORE_PROMOTION_ENABLED,
    REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
    REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED,
    REFLECTION_WORKER_INTERVAL_SECONDS,
)
from kazusa_ai_chatbot.db.schemas import CharacterReflectionRunDoc
from kazusa_ai_chatbot.reflection_cycle import repository
from kazusa_ai_chatbot.reflection_cycle.models import (
    DailySynthesisResult,
    PromptBuildResult,
    REFLECTION_RUN_KIND_DAILY_CHANNEL,
    REFLECTION_RUN_KIND_HOURLY,
    REFLECTION_STATUS_FAILED,
    REFLECTION_STATUS_SKIPPED,
    ReflectionLLMResult,
    ReflectionPromotionResult,
    ReflectionScopeInput,
    ReflectionWorkerHandle,
    ReflectionWorkerResult,
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
from kazusa_ai_chatbot.reflection_cycle.selector import collect_reflection_inputs
from kazusa_ai_chatbot.reflection_cycle.promotion import (
    _run_global_reflection_promotion,
)
from kazusa_ai_chatbot.reflection_cycle.interaction_style import (
    run_daily_interaction_style_update as _run_daily_interaction_style_update,
)


logger = logging.getLogger(__name__)


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


def start_reflection_cycle_worker(
    *,
    is_primary_interaction_busy: Callable[[], bool],
) -> ReflectionWorkerHandle:
    """Start the process-local reflection worker loop."""

    stop_event = asyncio.Event()
    task = asyncio.create_task(
        _reflection_worker_loop(
            stop_event=stop_event,
            is_primary_interaction_busy=is_primary_interaction_busy,
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


async def _reflection_worker_loop(
    *,
    stop_event: asyncio.Event,
    is_primary_interaction_busy: Callable[[], bool],
) -> None:
    """Run reflection scheduling ticks until stopped."""

    while not stop_event.is_set():
        await _run_worker_tick(
            now=datetime.now(timezone.utc),
            is_primary_interaction_busy=is_primary_interaction_busy,
        )
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=REFLECTION_WORKER_INTERVAL_SECONDS,
            )
        except TimeoutError:
            continue


async def _run_worker_tick(
    *,
    now: datetime,
    is_primary_interaction_busy: Callable[[], bool],
) -> list[ReflectionWorkerResult]:
    """Run one scheduled worker tick in approved priority order."""

    if is_primary_interaction_busy():
        logger.info("Reflection cycle tick skipped: primary interaction busy")
        return_value: list[ReflectionWorkerResult] = [
            ReflectionWorkerResult(
                run_kind="reflection_tick",
                dry_run=False,
                deferred=True,
                defer_reason="primary interaction busy",
            )
        ]
        return return_value

    results: list[ReflectionWorkerResult] = []
    hourly_result = await _run_hourly_reflection_cycle(
        now=now,
        dry_run=False,
        is_primary_interaction_busy=is_primary_interaction_busy,
    )
    results.append(hourly_result)

    local_now = now.astimezone(ZoneInfo(CHARACTER_TIME_ZONE))
    previous_local_date = (local_now.date() - timedelta(days=1)).isoformat()
    if _local_time_is_after(local_now, REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME):
        if is_primary_interaction_busy():
            return results
        daily_result = await _run_daily_channel_reflection_cycle(
            character_local_date=previous_local_date,
            dry_run=False,
            is_primary_interaction_busy=is_primary_interaction_busy,
        )
        results.append(daily_result)
        if is_primary_interaction_busy():
            return results
        style_result = await _run_daily_interaction_style_update(
            character_local_date=previous_local_date,
            dry_run=False,
            is_primary_interaction_busy=is_primary_interaction_busy,
        )
        results.append(style_result)

    if _local_time_is_after(local_now, REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME):
        if is_primary_interaction_busy():
            return results
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


async def _run_daily_channel_reflection_cycle(
    *,
    character_local_date: str,
    dry_run: bool,
    is_primary_interaction_busy: Callable[[], bool],
) -> ReflectionWorkerResult:
    """Persist due daily-channel reflection runs for monitored scopes."""

    input_set = await collect_reflection_inputs(
        lookback_hours=24,
        now=None,
        allow_fallback=False,
    )
    result = ReflectionWorkerResult(
        run_kind=REFLECTION_RUN_KIND_DAILY_CHANNEL,
        dry_run=dry_run,
    )
    for channel_scope in input_set.selected_scopes:
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

    if not input_set.selected_scopes:
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
        await repository.upsert_run(run_doc)
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
        await repository.upsert_run(run_doc)
        return run_doc

    run_doc = repository.build_hourly_run_document(
        scope=hourly_scope,
        result=llm_result,
        status=repository.status_for_result(dry_run=False),
        attempt_count=attempt_count,
    )
    await repository.upsert_run(run_doc)
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
        await repository.upsert_run(run_doc)
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
        await repository.upsert_run(run_doc)
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
    await repository.upsert_run(run_doc)
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
            return_value = (llm_result, attempt_count)
            return return_value
        except Exception as exc:
            last_error = exc
            if attempt == 2:
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
            return_value = (daily_result, attempt_count)
            return return_value
        except Exception as exc:
            last_error = exc
            if attempt == 2:
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


def _local_time_is_after(local_now: datetime, value: str) -> bool:
    """Return whether the local clock is at or after a HH:MM gate."""

    hour_text, minute_text = value.split(":", maxsplit=1)
    gate_minutes = int(hour_text) * 60 + int(minute_text)
    current_minutes = local_now.hour * 60 + local_now.minute
    return_value = current_minutes >= gate_minutes
    return return_value
