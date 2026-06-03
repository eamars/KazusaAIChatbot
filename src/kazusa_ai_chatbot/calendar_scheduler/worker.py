"""Calendar worker orchestration for claimed typed runs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from kazusa_ai_chatbot.calendar_scheduler import models


CalendarRunHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


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

        result = await handler(run)
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
