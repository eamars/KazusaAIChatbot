"""Build source-bound cognition inputs from completed artifact jobs."""

from __future__ import annotations

from kazusa_ai_chatbot.background_artifact.models import BackgroundArtifactJobDoc
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    build_background_artifact_result_ready_cognitive_episode,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


def build_result_ready_episode_from_job(
    job: BackgroundArtifactJobDoc,
) -> CognitiveEpisode:
    """Project one completed job into a result-ready cognitive episode."""

    completed_at = job.get("completed_at") or job.get("updated_at") or job["created_at"]
    turn_clock = build_turn_clock_from_storage_utc(completed_at)
    episode = build_background_artifact_result_ready_cognitive_episode(
        episode_id=f"background_artifact_result_ready:{job['job_id']}",
        percept_id=(
            f"background_artifact_result_ready:{job['job_id']}:result:0"
        ),
        storage_timestamp_utc=completed_at,
        local_time_context=turn_clock["local_time_context"],
        job_id=job["job_id"],
        work_kind=job["work_kind"],
        objective_summary=job["objective"],
        artifact_text=job.get("artifact_text", ""),
        failure_summary=job.get("failure_summary", ""),
        platform=job.get("source_platform", ""),
        platform_channel_id=job.get("source_channel_id", ""),
        channel_type=job.get("source_channel_type", ""),
        platform_message_id=job.get("source_message_id", ""),
        requester_platform_user_id=job.get("requester_platform_user_id", ""),
        requester_global_user_id=job.get("requester_global_user_id", ""),
        requester_display_name=job.get("requester_display_name", ""),
        source_platform_bot_id=job.get("source_platform_bot_id", ""),
        source_character_name=job.get("source_character_name", ""),
    )
    return episode
