"""Build source-bound cognition inputs from completed background-work jobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict

from kazusa_ai_chatbot.background_work.models import BackgroundWorkJobDoc
from kazusa_ai_chatbot.coding_agent.coding_run.ledger import (
    sanitize_coding_run_context,
)
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    build_accepted_task_result_ready_cognitive_episode,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


class AcceptedTaskCognitionSourceV2(TypedDict):
    """Typed accepted-task outcome admitted to a later cognition episode."""

    source_kind: Literal["accepted_task_result"]
    source_id: str
    occurred_at: str
    semantic_summary: str


def build_result_ready_episode_from_job(
    job: BackgroundWorkJobDoc,
) -> CognitiveEpisode:
    """Project one completed generic job into a result-ready episode."""

    completed_at = job.get("completed_at") or job.get("updated_at") or job["created_at"]
    turn_clock = build_turn_clock_from_storage_utc(completed_at)
    accepted_task_id = job.get("accepted_task_id", "").strip()
    if not accepted_task_id:
        raise ValueError("accepted_task_id is required for result delivery")

    worker_metadata = job.get("worker_metadata")
    coding_run_context = None
    if isinstance(worker_metadata, Mapping):
        metadata_context = worker_metadata.get("coding_run_context")
        coding_run_context = sanitize_coding_run_context(metadata_context)
    episode = build_accepted_task_result_ready_cognitive_episode(
        episode_id=f"accepted_task_result_ready:{accepted_task_id}",
        percept_id=f"accepted_task_result_ready:{accepted_task_id}:result:0",
        storage_timestamp_utc=completed_at,
        local_time_context=turn_clock["local_time_context"],
        accepted_task_id=accepted_task_id,
        accepted_task_summary=job["task_brief"],
        artifact_text=job.get("artifact_text", ""),
        failure_summary=job.get("failure_summary", ""),
        result_summary=job.get("result_summary", ""),
        platform=job.get("source_platform", ""),
        platform_channel_id=job.get("source_channel_id", ""),
        channel_type=job.get("source_channel_type", ""),
        platform_message_id=job.get("source_message_id", ""),
        requester_platform_user_id=job.get(
            "requester_platform_user_id",
            "",
        ),
        requester_global_user_id=job.get("requester_global_user_id", ""),
        requester_display_name=job.get("requester_display_name", ""),
        source_platform_bot_id=job.get("source_platform_bot_id", ""),
        source_character_name=job.get("source_character_name", ""),
        coding_run_context=coding_run_context,
    )
    outcome_summary = (
        job.get("result_summary")
        or job.get("failure_summary")
        or "accepted task completed without a result summary"
    )
    cognition_source = AcceptedTaskCognitionSourceV2(
        source_kind="accepted_task_result",
        source_id=accepted_task_id,
        occurred_at=completed_at,
        semantic_summary=outcome_summary,
    )
    episode["percepts"][0]["metadata"]["cognition_source"] = cognition_source
    return episode
