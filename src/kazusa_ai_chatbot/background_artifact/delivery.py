"""Delivery boundary for completed background artifact jobs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from kazusa_ai_chatbot.background_artifact.result_source import (
    build_result_ready_episode_from_job,
)
from kazusa_ai_chatbot.cognition_episode import CognitiveEpisode
from kazusa_ai_chatbot.config import BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT
from kazusa_ai_chatbot.db import (
    find_deliverable_background_artifact_jobs,
    mark_background_artifact_delivered,
    mark_background_artifact_delivery_failed,
    mark_background_artifact_delivery_in_progress,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso

BackgroundArtifactCognitionDeliveryFunc = Callable[
    [CognitiveEpisode],
    Awaitable[dict[str, Any]],
]


async def run_background_artifact_delivery_tick(
    *,
    deliver_result_episode_func: BackgroundArtifactCognitionDeliveryFunc
    | None = None,
    limit: int = BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT,
) -> dict[str, int]:
    """Convert ready jobs to result-ready cognition and mark delivery outcome."""

    jobs = await find_deliverable_background_artifact_jobs(limit=limit)
    processed_count = 0
    delivered_count = 0
    failed_count = 0
    for job in jobs:
        processed_count += 1
        delivery_tracking_id = f"background-artifact-delivery-{uuid4().hex}"
        started_at = storage_utc_now_iso()
        marked_job = await mark_background_artifact_delivery_in_progress(
            job_id=job["job_id"],
            delivery_tracking_id=delivery_tracking_id,
            started_at=started_at,
        )
        if marked_job is None:
            failed_count += 1
            continue
        episode = build_result_ready_episode_from_job(marked_job)
        if deliver_result_episode_func is None:
            await mark_background_artifact_delivery_failed(
                job_id=job["job_id"],
                failure_summary="Result-ready cognition delivery is unavailable.",
                failed_at=storage_utc_now_iso(),
            )
            failed_count += 1
            continue
        delivery_result = await deliver_result_episode_func(episode)
        if delivery_result.get("status") == "delivered":
            await mark_background_artifact_delivered(
                job_id=job["job_id"],
                delivered_conversation_message_id=str(
                    delivery_result.get("conversation_message_id", ""),
                ),
                delivered_at=storage_utc_now_iso(),
            )
            delivered_count += 1
        else:
            await mark_background_artifact_delivery_failed(
                job_id=job["job_id"],
                failure_summary=str(
                    delivery_result.get("reason", "delivery failed"),
                ),
                failed_at=storage_utc_now_iso(),
            )
            failed_count += 1
    result = {
        "processed_count": processed_count,
        "delivered_count": delivered_count,
        "failed_count": failed_count,
    }
    return result
