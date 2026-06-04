"""Calendar worker orchestration for claimed typed runs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.calendar_scheduler import models
from kazusa_ai_chatbot.calendar_scheduler import repository as calendar_repository
from kazusa_ai_chatbot.calendar_scheduler.reflection_phase import (
    materialize_reflection_phase_period,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now


CalendarRunHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
CalendarWorkerTickFunc = Callable[..., Awaitable[dict[str, int]]]
ReflectionPhaseMaterializer = Callable[..., Awaitable[dict[str, Any]]]
NowFunc = Callable[[], datetime]

logger = logging.getLogger(__name__)

CALENDAR_SCHEDULER_LEASE_OWNER = "calendar_scheduler_worker"
CALENDAR_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


@dataclass
class CalendarSchedulerWorkerHandle:
    """Owned process task and stop signal for the calendar worker."""

    task: asyncio.Task
    stop_event: asyncio.Event


class CalendarRunHandlerRegistry:
    """Closed registry from calendar trigger kind to async run handler."""

    def __init__(self) -> None:
        self._handlers: dict[str, CalendarRunHandler] = {}

    def register(
        self,
        trigger_kind: str,
        handler: CalendarRunHandler,
    ) -> None:
        """Register one handler for a supported trigger kind."""

        if trigger_kind not in models.CALENDAR_TRIGGER_KINDS:
            raise ValueError(f"unsupported calendar trigger kind: {trigger_kind}")
        self._handlers[trigger_kind] = handler

    def get(self, trigger_kind: str) -> CalendarRunHandler | None:
        """Return the handler for a trigger kind, if registered."""

        handler = self._handlers.get(trigger_kind)
        return handler

    def trigger_kinds(self) -> list[str]:
        """Return registered trigger kinds or the closed roster for claiming."""

        if self._handlers:
            trigger_kinds = sorted(self._handlers)
            return trigger_kinds

        trigger_kinds = sorted(models.CALENDAR_TRIGGER_KINDS)
        return trigger_kinds


def start_calendar_scheduler_worker(
    *,
    repository: Any = calendar_repository,
    handler_registry: CalendarRunHandlerRegistry,
    poll_interval_seconds: int,
    lease_owner: str = CALENDAR_SCHEDULER_LEASE_OWNER,
    lease_duration_seconds: int,
    claim_limit: int,
    max_attempts: int,
    now_func: NowFunc = storage_utc_now,
    materialize_reflection_phase_period_func: (
        ReflectionPhaseMaterializer | None
    ) = None,
    run_worker_tick_func: CalendarWorkerTickFunc | None = None,
) -> CalendarSchedulerWorkerHandle:
    """Start the durable calendar worker loop for registered trigger kinds."""

    phase_materializer = (
        materialize_reflection_phase_period_func
        or materialize_reflection_phase_period
    )
    worker_tick = run_worker_tick_func or run_calendar_worker_tick
    stop_event = asyncio.Event()
    task = asyncio.create_task(
        _calendar_scheduler_worker_loop(
            stop_event=stop_event,
            repository=repository,
            handler_registry=handler_registry,
            poll_interval_seconds=poll_interval_seconds,
            lease_owner=lease_owner,
            lease_duration_seconds=lease_duration_seconds,
            claim_limit=claim_limit,
            max_attempts=max_attempts,
            now_func=now_func,
            materialize_reflection_phase_period_func=(
                phase_materializer
            ),
            run_worker_tick_func=worker_tick,
        )
    )
    handle = CalendarSchedulerWorkerHandle(task=task, stop_event=stop_event)
    logger.info("Calendar scheduler worker started")
    return handle


async def stop_calendar_scheduler_worker(
    handle: CalendarSchedulerWorkerHandle,
) -> None:
    """Stop a calendar worker handle created by the public starter."""

    handle.stop_event.set()
    try:
        await asyncio.wait_for(
            handle.task,
            timeout=CALENDAR_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        handle.task.cancel()
        with suppress(asyncio.CancelledError):
            await handle.task
    logger.info("Calendar scheduler worker stopped")


async def run_calendar_worker_tick(
    *,
    current_timestamp_utc: str,
    repository: Any,
    handler_registry: CalendarRunHandlerRegistry,
    lease_owner: str,
    lease_duration_seconds: int,
    claim_limit: int,
    max_attempts: int,
) -> dict[str, int]:
    """Claim due runs and dispatch each through its typed handler."""

    claimed_runs = await repository.claim_due_calendar_runs(
        current_timestamp_utc=current_timestamp_utc,
        lease_owner=lease_owner,
        lease_duration_seconds=lease_duration_seconds,
        limit=claim_limit,
        max_attempts=max_attempts,
        trigger_kinds=handler_registry.trigger_kinds(),
    )

    completed_count = 0
    failed_count = 0
    skipped_count = 0
    for run in claimed_runs:
        trigger_kind = run["trigger_kind"]
        handler = handler_registry.get(trigger_kind)
        if handler is None:
            error = f"unsupported calendar trigger kind: {trigger_kind}"
            await repository.mark_calendar_run_failed(
                run["run_id"],
                lease_owner=lease_owner,
                storage_timestamp_utc=current_timestamp_utc,
                error=error,
                retryable=False,
            )
            failed_count += 1
            continue

        try:
            result = await handler(run)
        except Exception as exc:
            logger.exception(
                f"Calendar run handler failed for {run['run_id']}: {exc}"
            )
            await repository.mark_calendar_run_failed(
                run["run_id"],
                lease_owner=lease_owner,
                storage_timestamp_utc=current_timestamp_utc,
                error=str(exc),
                retryable=False,
            )
            failed_count += 1
            continue
        if result.get("status") == "skipped":
            await repository.mark_calendar_run_skipped(
                run["run_id"],
                lease_owner=lease_owner,
                storage_timestamp_utc=current_timestamp_utc,
                reason=result["reason"],
            )
            skipped_count += 1
            continue

        await repository.mark_calendar_run_completed(
            run["run_id"],
            lease_owner=lease_owner,
            storage_timestamp_utc=current_timestamp_utc,
            result=result,
        )
        completed_count += 1

    tick_result = {
        "claimed_count": len(claimed_runs),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
    }
    return tick_result


async def _calendar_scheduler_worker_loop(
    *,
    stop_event: asyncio.Event,
    repository: Any,
    handler_registry: CalendarRunHandlerRegistry,
    poll_interval_seconds: int,
    lease_owner: str,
    lease_duration_seconds: int,
    claim_limit: int,
    max_attempts: int,
    now_func: NowFunc,
    materialize_reflection_phase_period_func: ReflectionPhaseMaterializer,
    run_worker_tick_func: CalendarWorkerTickFunc,
) -> None:
    """Run materialization and due-run claiming until a stop signal arrives."""

    while not stop_event.is_set():
        now = now_func()
        current_timestamp_utc = now.isoformat()
        try:
            await materialize_reflection_phase_period_func(
                period_start_utc=now,
                storage_timestamp_utc=current_timestamp_utc,
                repository=repository,
            )
            await run_worker_tick_func(
                current_timestamp_utc=current_timestamp_utc,
                repository=repository,
                handler_registry=handler_registry,
                lease_owner=lease_owner,
                lease_duration_seconds=lease_duration_seconds,
                claim_limit=claim_limit,
                max_attempts=max_attempts,
            )
        except Exception as exc:
            logger.exception(f"Calendar scheduler worker tick failed: {exc}")

        if stop_event.is_set():
            break
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=poll_interval_seconds,
            )
        except TimeoutError:
            continue
