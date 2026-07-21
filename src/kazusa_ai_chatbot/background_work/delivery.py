"""Delivery boundary for completed background-work jobs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any
from uuid import uuid4

from kazusa_ai_chatbot.accepted_task import (
    mark_accepted_task_delivered,
    mark_accepted_task_delivery_failed,
    mark_accepted_task_delivery_in_progress,
    recover_stale_delivery_in_progress_tasks,
)
from kazusa_ai_chatbot.background_work.result_source import (
    build_result_ready_episode_from_job,
)
from kazusa_ai_chatbot.cognition_episode import CognitiveEpisodeV1
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_WORKER_CLAIM_LIMIT,
    BACKGROUND_WORK_WORKER_LEASE_SECONDS,
)
from kazusa_ai_chatbot.db.background_work_jobs import (
    find_deliverable_background_work_jobs,
    mark_background_work_delivered,
    mark_background_work_delivery_failed,
    mark_background_work_delivery_in_progress,
    recover_stale_background_work_delivery_in_progress,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now, storage_utc_now_iso

BackgroundWorkCognitionDeliveryFunc = Callable[
    [CognitiveEpisodeV1],
    Awaitable[dict[str, Any]],
]


async def run_background_work_delivery_tick(
    *,
    deliver_result_episode_func: BackgroundWorkCognitionDeliveryFunc
    | None = None,
    limit: int = BACKGROUND_WORK_WORKER_CLAIM_LIMIT,
) -> dict[str, int]:
    """Convert ready jobs to result-ready cognition and mark delivery outcome."""

    recovered_count = await _recover_stale_delivery_attempts()
    jobs = await find_deliverable_background_work_jobs(limit=limit)
    processed_count = 0
    delivered_count = 0
    failed_count = 0
    for job in jobs:
        processed_count += 1
        delivery_tracking_id = f"background-work-delivery-{uuid4().hex}"
        started_at = storage_utc_now_iso()
        marked_job = await mark_background_work_delivery_in_progress(
            job_id=job["job_id"],
            delivery_tracking_id=delivery_tracking_id,
            started_at=started_at,
        )
        if marked_job is None:
            failed_count += 1
            continue
        accepted_task_id = _accepted_task_id_from_job(marked_job)
        if accepted_task_id:
            await mark_accepted_task_delivery_in_progress(
                accepted_task_id=accepted_task_id,
                delivery_tracking_id=delivery_tracking_id,
                updated_at=started_at,
            )
        episode = build_result_ready_episode_from_job(marked_job)
        if deliver_result_episode_func is None:
            await mark_background_work_delivery_failed(
                job_id=job["job_id"],
                failure_summary="Result-ready cognition delivery is unavailable.",
                failed_at=storage_utc_now_iso(),
            )
            if accepted_task_id:
                await mark_accepted_task_delivery_failed(
                    accepted_task_id=accepted_task_id,
                    failure_summary=(
                        "Result-ready cognition delivery is unavailable."
                    ),
                    failed_at=storage_utc_now_iso(),
                )
            failed_count += 1
            continue
        delivery_result = await deliver_result_episode_func(episode)
        if delivery_result.get("status") == "delivered":
            delivered_at = storage_utc_now_iso()
            await mark_background_work_delivered(
                job_id=job["job_id"],
                delivered_conversation_message_id=str(
                    delivery_result.get("conversation_message_id", ""),
                ),
                delivered_at=delivered_at,
            )
            if accepted_task_id:
                await mark_accepted_task_delivered(
                    accepted_task_id=accepted_task_id,
                    delivered_conversation_message_id=str(
                        delivery_result.get("conversation_message_id", ""),
                    ),
                    delivered_at=delivered_at,
                )
            delivered_count += 1
        else:
            failed_at = storage_utc_now_iso()
            failure_summary = str(
                delivery_result.get("reason", "delivery failed"),
            )
            await mark_background_work_delivery_failed(
                job_id=job["job_id"],
                failure_summary=failure_summary,
                failed_at=failed_at,
            )
            if accepted_task_id:
                await mark_accepted_task_delivery_failed(
                    accepted_task_id=accepted_task_id,
                    failure_summary=failure_summary,
                    failed_at=failed_at,
                )
            failed_count += 1
    result = {
        "processed_count": processed_count,
        "delivered_count": delivered_count,
        "failed_count": failed_count,
        "recovered_count": recovered_count,
    }
    return result


async def _recover_stale_delivery_attempts() -> int:
    """Recover interrupted delivery claims before finding retryable jobs."""

    now_utc = storage_utc_now()
    stale_before_utc = (
        now_utc - timedelta(seconds=BACKGROUND_WORK_WORKER_LEASE_SECONDS)
    ).isoformat()
    recovered_at = now_utc.isoformat()
    background_recovered_count = (
        await recover_stale_background_work_delivery_in_progress(
            stale_before_utc=stale_before_utc,
            recovered_at=recovered_at,
        )
    )
    accepted_task_recovered_count = (
        await recover_stale_delivery_in_progress_tasks(
            stale_before_utc=stale_before_utc,
            recovered_at=recovered_at,
        )
    )
    recovered_count = (
        background_recovered_count + accepted_task_recovered_count
    )
    return recovered_count


def _accepted_task_id_from_job(job: dict[str, Any]) -> str:
    """Return accepted-task id from a new-background-work job row."""

    value = job.get("accepted_task_id")
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value
