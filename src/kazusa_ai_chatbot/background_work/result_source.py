"""Build source-bound cognition inputs from completed background-work jobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from kazusa_ai_chatbot.background_work.models import BackgroundWorkJobDoc
from kazusa_ai_chatbot.coding_agent.coding_run.ledger import (
    sanitize_coding_run_context,
)
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    EvidenceRefV1,
    TargetScopeV1,
    ToolResultReadyV1,
    build_tool_result_episode,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


class ToolResultCognitionSourceV1(TypedDict):
    """Typed tool outcome admitted to a later cognition episode."""

    source_kind: str
    source_id: str
    occurred_at: str
    semantic_summary: str


def build_result_ready_episode_from_job(
    job: BackgroundWorkJobDoc,
) -> CognitiveEpisodeV1:
    """Project one completed generic job into a canonical tool-result episode."""

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
    outcome_summary = (
        job.get("result_summary")
        or job.get("failure_summary")
        or "tool result completed without a result summary"
    )
    target_scope: TargetScopeV1 = {
        "platform": job.get("source_platform", ""),
        "platform_channel_id": job.get("source_channel_id", ""),
        "channel_type": job.get("source_channel_type", ""),
        "current_platform_user_id": job.get(
            "requester_platform_user_id",
            "",
        ),
        "current_global_user_id": job.get("requester_global_user_id", ""),
        "current_display_name": job.get("requester_display_name", ""),
        "target_addressed_user_ids": [
            job.get("requester_global_user_id", "")
        ] if job.get("requester_global_user_id") else [],
        "target_broadcast": False,
    }
    evidence_ref: EvidenceRefV1 = {
        "schema_version": "evidence_ref.v1",
        "evidence_kind": "tool_result",
        "evidence_id": accepted_task_id,
        "owner": "background_work",
        "excerpt": outcome_summary[:800],
        "observed_at": completed_at,
    }
    result: ToolResultReadyV1 = {
        "schema_version": "tool_result_ready.v1",
        "task_id": accepted_task_id,
        "task_kind": "accepted_task",
        "semantic_summary": outcome_summary,
        "artifact_text": job.get("artifact_text", ""),
        "failure_text": job.get("failure_summary", ""),
        "completed_at": completed_at,
        "target_scope": target_scope,
        "evidence_refs": [evidence_ref],
        "result_ref": accepted_task_id,
        "source_platform_bot_id": job.get("source_platform_bot_id", ""),
        "source_character_name": job.get("source_character_name", ""),
    }
    if coding_run_context is not None:
        result["coding_run_context"] = dict(coding_run_context)
    episode = build_tool_result_episode(
        result=result,
        evidence_refs=[evidence_ref],
        local_time_context=turn_clock["local_time_context"],
        created_at=completed_at,
    )
    cognition_source = ToolResultCognitionSourceV1(
        source_kind="tool_result",
        source_id=accepted_task_id,
        occurred_at=completed_at,
        semantic_summary=outcome_summary,
    )
    episode["percepts"][0]["content"]["cognition_source"] = cognition_source
    return episode
