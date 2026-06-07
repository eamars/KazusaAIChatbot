"""Typed contracts for background artifact jobs."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

BACKGROUND_ARTIFACT_JOBS_COLLECTION = "background_artifact_jobs"
BACKGROUND_ARTIFACT_JOB_REF_OWNER = "background_artifact_job"
BACKGROUND_ARTIFACT_REQUESTED_DELIVERY = "send_result_when_done"

BackgroundArtifactWorkKind = Literal[
    "coding_snippet",
    "text_rewrite",
    "summary",
]
BackgroundArtifactJobStatus = Literal[
    "queued",
    "in_progress",
    "completed",
    "failed",
    "delivery_in_progress",
    "delivered",
    "delivery_failed",
]
BackgroundArtifactDeliveryState = Literal[
    "queued",
    "ready",
    "in_progress",
    "delivered",
    "failed",
]

BACKGROUND_ARTIFACT_WORK_KINDS: tuple[BackgroundArtifactWorkKind, ...] = (
    "coding_snippet",
    "text_rewrite",
    "summary",
)


class BackgroundArtifactQueueRequest(TypedDict):
    """Request to create one durable background artifact job."""

    action_attempt_id: str
    idempotency_key: str
    work_kind: BackgroundArtifactWorkKind
    objective: str
    input_summary: str
    requested_delivery: Literal["send_result_when_done"]
    max_output_chars: int
    source_platform: str
    source_channel_id: str
    source_channel_type: str
    source_message_id: str
    source_platform_bot_id: str
    source_character_name: str
    requester_global_user_id: str
    requester_platform_user_id: str
    requester_display_name: str
    storage_timestamp_utc: str


class BackgroundArtifactQueueResult(TypedDict):
    """Prompt-safe queue result exposed to action traces and L3."""

    status: Literal["pending", "rejected", "failed"]
    queue_state: str
    job_id: str
    job_ref: str
    work_kind: str
    objective_summary: str
    result_summary: str
    operational_owner: Literal["background_artifact_job"]
    acknowledgement_constraint: Literal[
        "promise_allowed",
        "promise_forbidden_explain_failure",
    ]
    evidence_ref: NotRequired[dict[str, str]]


class BackgroundArtifactJobDoc(TypedDict, total=False):
    """MongoDB document for one background artifact job."""

    schema_version: Literal["background_artifact_job.v1"]
    job_id: str
    idempotency_key: str
    source_action_attempt_id: str
    status: BackgroundArtifactJobStatus
    delivery_state: BackgroundArtifactDeliveryState
    work_kind: BackgroundArtifactWorkKind
    objective: str
    input_summary: str
    requested_delivery: Literal["send_result_when_done"]
    max_output_chars: int
    source_platform: str
    source_channel_id: str
    source_channel_type: str
    source_message_id: str
    source_platform_bot_id: str
    source_character_name: str
    requester_global_user_id: str
    requester_platform_user_id: str
    requester_display_name: str
    created_at: str
    updated_at: str
    lease_owner: str | None
    lease_expires_at: str | None
    attempt_count: int
    max_attempts: int
    artifact_text: str
    artifact_char_count: int
    failure_summary: str
    completed_at: str
    delivery_attempt_count: int
    delivery_failure_summary: str
    delivery_tracking_id: str
    delivered_conversation_message_id: str
    delivered_at: str


def background_artifact_job_ref(job_id: str) -> str:
    """Return the prompt-safe evidence id for one job."""

    clean_job_id = job_id.strip()
    if clean_job_id.startswith("background_artifact_job:"):
        return clean_job_id
    return_value = f"background_artifact_job:{clean_job_id}"
    return return_value
