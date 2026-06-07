"""Typed contracts for generic background-work jobs."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

BACKGROUND_WORK_JOBS_COLLECTION = "background_work_jobs"
BACKGROUND_WORK_JOB_REF_OWNER = "background_work_job"
BACKGROUND_WORK_REQUESTED_DELIVERY = "send_result_when_done"

BackgroundWorkJobStatus = Literal[
    "queued",
    "in_progress",
    "completed",
    "failed",
    "delivery_in_progress",
    "delivered",
    "delivery_failed",
]
BackgroundWorkDeliveryState = Literal[
    "queued",
    "ready",
    "in_progress",
    "delivered",
    "failed",
]
BackgroundWorkRouterAction = Literal[
    "execute",
    "reject",
    "needs_user_input",
    "stop",
]
BackgroundWorkWorkerStatus = Literal[
    "succeeded",
    "failed",
    "needs_user_input",
    "rejected",
]


class BackgroundWorkQueueRequest(TypedDict):
    """Request to create one durable generic background-work job."""

    action_attempt_id: str
    idempotency_key: str
    task_brief: str
    source_context: NotRequired[str]
    source_platform: str
    source_channel_id: str
    source_channel_type: str
    source_message_id: str
    source_platform_bot_id: str
    source_character_name: str
    requester_global_user_id: str
    requester_platform_user_id: str
    requester_display_name: str
    requested_delivery: Literal["send_result_when_done"]
    max_output_chars: int
    storage_timestamp_utc: str


class BackgroundWorkQueueResult(TypedDict):
    """Prompt-safe queue result exposed to action traces and L3."""

    status: Literal["pending", "rejected", "failed"]
    queue_state: str
    job_id: str
    job_ref: str
    task_summary: str
    result_summary: str
    operational_owner: Literal["background_work_job"]
    acknowledgement_constraint: Literal[
        "promise_allowed",
        "promise_forbidden_explain_failure",
    ]
    evidence_ref: NotRequired[dict[str, str]]


class BackgroundWorkRouterDecision(TypedDict):
    """Route-only decision for one claimed background-work job."""

    action: BackgroundWorkRouterAction
    worker: str
    reason: str


class BackgroundWorkWorkerDecision(BackgroundWorkRouterDecision, total=False):
    """Route decision plus deterministic context passed to a worker."""

    source_summary: str


class BackgroundWorkResult(TypedDict):
    """Prompt-safe worker result recorded on the durable job."""

    status: BackgroundWorkWorkerStatus
    worker: str
    artifact_text: str
    failure_summary: str
    result_summary: str
    worker_metadata: dict[str, object]


class BackgroundWorkJobRef(TypedDict):
    """Stable prompt-safe reference for one background-work job."""

    job_id: str
    job_ref: str


class BackgroundWorkJobDoc(TypedDict, total=False):
    """MongoDB document for one generic background-work job."""

    schema_version: Literal["background_work_job.v1"]
    job_id: str
    idempotency_key: str
    source_action_attempt_id: str
    status: BackgroundWorkJobStatus
    delivery_state: BackgroundWorkDeliveryState
    task_brief: str
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
    router_action: str
    worker: str
    routed_task: str
    router_reason: str
    source_context: str
    artifact_text: str
    artifact_char_count: int
    failure_summary: str
    result_summary: str
    worker_metadata: dict[str, object]
    completed_at: str
    delivery_attempt_count: int
    delivery_failure_summary: str
    delivery_tracking_id: str
    delivered_conversation_message_id: str
    delivered_at: str


def background_work_job_ref(job_id: str) -> str:
    """Return the prompt-safe evidence id for one generic job."""

    clean_job_id = job_id.strip()
    if clean_job_id.startswith("background_work_job:"):
        return clean_job_id
    return_value = f"background_work_job:{clean_job_id}"
    return return_value
