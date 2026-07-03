"""Runtime loop for background artifact worker and delivery ticks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.background_artifact.delivery import (
    BackgroundArtifactCognitionDeliveryFunc,
    run_background_artifact_delivery_tick,
)
from kazusa_ai_chatbot.background_artifact.worker import (
    BACKGROUND_ARTIFACT_WORKER_COMPONENT,
    run_background_artifact_worker_tick,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT,
    BACKGROUND_ARTIFACT_WORKER_ENABLED,
    BACKGROUND_ARTIFACT_WORKER_INTERVAL_SECONDS,
    BACKGROUND_ARTIFACT_WORKER_LEASE_SECONDS,
    BACKGROUND_ARTIFACT_WORKER_MAX_ATTEMPTS,
)

logger = logging.getLogger(__name__)

BusyCheck = Callable[[], bool]


@dataclass
class BackgroundArtifactRuntimeHandle:
    """Owned task handle for the background artifact runtime."""

    task: asyncio.Task | None
    enabled: bool


async def run_background_artifact_runtime_tick(
    *,
    is_primary_interaction_busy: BusyCheck | None = None,
    deliver_result_episode_func: BackgroundArtifactCognitionDeliveryFunc
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

    worker_result = await run_background_artifact_worker_tick(
        claim_limit=BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT,
        lease_seconds=BACKGROUND_ARTIFACT_WORKER_LEASE_SECONDS,
        max_attempts=BACKGROUND_ARTIFACT_WORKER_MAX_ATTEMPTS,
    )
    delivery_result = await run_background_artifact_delivery_tick(
        deliver_result_episode_func=deliver_result_episode_func,
        limit=BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT,
    )
    result = {
        "status": "succeeded",
        **worker_result,
        "delivery_processed_count": delivery_result["processed_count"],
        "delivery_delivered_count": delivery_result["delivered_count"],
        "delivery_failed_count": delivery_result["failed_count"],
    }
    return result


def start_background_artifact_runtime(
    *,
    is_primary_interaction_busy: BusyCheck,
    deliver_result_episode_func: BackgroundArtifactCognitionDeliveryFunc
    | None = None,
) -> BackgroundArtifactRuntimeHandle:
    """Start the owned background artifact runtime loop."""

    if not BACKGROUND_ARTIFACT_WORKER_ENABLED:
        handle = BackgroundArtifactRuntimeHandle(task=None, enabled=False)
        return handle
    task = asyncio.create_task(
        _background_artifact_runtime_loop(
            is_primary_interaction_busy=is_primary_interaction_busy,
            deliver_result_episode_func=deliver_result_episode_func,
        ),
        name="background_artifact_runtime",
    )
    handle = BackgroundArtifactRuntimeHandle(task=task, enabled=True)
    return handle


async def stop_background_artifact_runtime(
    handle: BackgroundArtifactRuntimeHandle,
) -> None:
    """Stop a started background artifact runtime task."""

    if handle.task is None:
        return
    handle.task.cancel()
    try:
        await handle.task
    except asyncio.CancelledError:
        logger.info("Background artifact runtime stopped")


async def _background_artifact_runtime_loop(
    *,
    is_primary_interaction_busy: BusyCheck,
    deliver_result_episode_func: BackgroundArtifactCognitionDeliveryFunc
    | None,
) -> None:
    """Run the periodic runtime loop until shutdown."""

    while True:
        try:
            tick_result = await run_background_artifact_runtime_tick(
                is_primary_interaction_busy=is_primary_interaction_busy,
                deliver_result_episode_func=deliver_result_episode_func,
            )
            await event_logging.record_worker_event(
                component=BACKGROUND_ARTIFACT_WORKER_COMPONENT,
                worker_name="background_artifact",
                event_type="tick",
                enabled=True,
                dry_run=False,
                run_kind="background_artifact",
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
            logger.exception(
                f"Background artifact runtime tick failed: {exc}"
            )
            await event_logging.record_runtime_error_event(
                component=BACKGROUND_ARTIFACT_WORKER_COMPONENT,
                error_class=exc.__class__.__name__,
                error_preview=str(exc)[:500],
                stack_fingerprint="background_artifact_runtime_tick",
                top_frame_module=__name__,
                recovered=True,
                status="failed",
                severity="warning",
            )
        await asyncio.sleep(BACKGROUND_ARTIFACT_WORKER_INTERVAL_SECONDS)
