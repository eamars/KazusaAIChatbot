"""Runtime loop for background-work worker and delivery ticks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.background_work.delivery import (
    BackgroundWorkCognitionDeliveryFunc,
    run_background_work_delivery_tick,
)
from kazusa_ai_chatbot.background_work.worker import (
    BACKGROUND_WORK_WORKER_COMPONENT,
    run_background_work_worker_tick,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_WORKER_CLAIM_LIMIT,
    BACKGROUND_WORK_WORKER_ENABLED,
    BACKGROUND_WORK_WORKER_INTERVAL_SECONDS,
    BACKGROUND_WORK_WORKER_LEASE_SECONDS,
    BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
)

logger = logging.getLogger(__name__)

BusyCheck = Callable[[], bool]


@dataclass
class BackgroundWorkRuntimeHandle:
    """Owned task handle for the background-work runtime."""

    task: asyncio.Task | None
    enabled: bool


async def run_background_work_runtime_tick(
    *,
    is_primary_interaction_busy: BusyCheck | None = None,
    deliver_result_episode_func: BackgroundWorkCognitionDeliveryFunc
    | None = None,
) -> dict[str, Any]:
    """Run one worker tick plus one result-delivery tick when not busy."""

    if is_primary_interaction_busy is not None and is_primary_interaction_busy():
        result = {
            "status": "skipped",
            "defer_reason": "primary_interaction_busy",
            "processed_count": 0,
            "succeeded_count": 0,
            "failed_count": 0,
        }
        return result

    worker_result = await run_background_work_worker_tick(
        claim_limit=BACKGROUND_WORK_WORKER_CLAIM_LIMIT,
        lease_seconds=BACKGROUND_WORK_WORKER_LEASE_SECONDS,
        max_attempts=BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
    )
    delivery_result = await run_background_work_delivery_tick(
        deliver_result_episode_func=deliver_result_episode_func,
        limit=BACKGROUND_WORK_WORKER_CLAIM_LIMIT,
    )
    result = {
        "status": "succeeded",
        **worker_result,
        "delivery_processed_count": delivery_result["processed_count"],
        "delivery_delivered_count": delivery_result["delivered_count"],
        "delivery_failed_count": delivery_result["failed_count"],
        "delivery_recovered_count": delivery_result.get("recovered_count", 0),
    }
    return result


def start_background_work_runtime(
    *,
    is_primary_interaction_busy: BusyCheck,
    deliver_result_episode_func: BackgroundWorkCognitionDeliveryFunc | None = None,
) -> BackgroundWorkRuntimeHandle:
    """Start the owned background-work runtime loop."""

    if not BACKGROUND_WORK_WORKER_ENABLED:
        handle = BackgroundWorkRuntimeHandle(task=None, enabled=False)
        return handle
    task = asyncio.create_task(
        _background_work_runtime_loop(
            is_primary_interaction_busy=is_primary_interaction_busy,
            deliver_result_episode_func=deliver_result_episode_func,
        ),
        name="background_work_runtime",
    )
    handle = BackgroundWorkRuntimeHandle(task=task, enabled=True)
    return handle


async def stop_background_work_runtime(
    handle: BackgroundWorkRuntimeHandle,
) -> None:
    """Stop a started background-work runtime task."""

    if handle.task is None:
        return
    handle.task.cancel()
    try:
        await handle.task
    except asyncio.CancelledError:
        logger.info("Background-work runtime stopped")


async def _background_work_runtime_loop(
    *,
    is_primary_interaction_busy: BusyCheck,
    deliver_result_episode_func: BackgroundWorkCognitionDeliveryFunc | None,
) -> None:
    """Run the periodic runtime loop until shutdown."""

    while True:
        try:
            tick_result = await run_background_work_runtime_tick(
                is_primary_interaction_busy=is_primary_interaction_busy,
                deliver_result_episode_func=deliver_result_episode_func,
            )
            await event_logging.record_worker_event(
                component=BACKGROUND_WORK_WORKER_COMPONENT,
                worker_name="background_work",
                event_type="tick",
                enabled=True,
                dry_run=False,
                run_kind="background_work",
                processed_count=int(tick_result.get("processed_count", 0)),
                succeeded_count=int(tick_result.get("succeeded_count", 0)),
                failed_count=int(tick_result.get("failed_count", 0)),
                skipped_count=1 if tick_result.get("status") == "skipped" else 0,
                deferred=tick_result.get("status") == "skipped",
                defer_reason=str(tick_result.get("defer_reason", "")),
                status=str(tick_result.get("status", "succeeded")),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"Background-work runtime tick failed: {exc}")
            await event_logging.record_runtime_error_event(
                component=BACKGROUND_WORK_WORKER_COMPONENT,
                error_class=exc.__class__.__name__,
                error_preview=str(exc)[:500],
                stack_fingerprint="background_work_runtime_tick",
                top_frame_module=__name__,
                recovered=True,
                status="failed",
                severity="warning",
            )
        await asyncio.sleep(BACKGROUND_WORK_WORKER_INTERVAL_SECONDS)
