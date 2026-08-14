"""Typed v2 persistence contracts for task-orchestrator background work."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.cognition_episode import GoalContinuationRefV1


BACKGROUND_WORK_JOBS_COLLECTION = "background_work_jobs"
BACKGROUND_WORK_JOB_SCHEMA_VERSION = "background_work_job.v2"
BACKGROUND_WORK_JOB_REF_OWNER = "background_work_job"
BACKGROUND_WORK_REQUESTED_DELIVERY = "send_result_when_done"
TASK_ORCHESTRATOR_WORKER = "task_orchestrator"
FUTURE_SPEAK_WORKER = "future_speak"
TASK_ORCHESTRATOR_WORKER_PAYLOAD_VERSION = "task_orchestrator_worker_payload.v1"

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
TaskOrchestratorOperation = Literal[
    "resume_task_resolution",
    "continue_bound_coding_run",
]
BackgroundWorkRequestedWorker = Literal["task_orchestrator", "future_speak"]


class TaskOrchestratorWorkerPayloadV1(TypedDict):
    """Reviewed durable payload for the single generic task worker."""

    schema_version: Literal["task_orchestrator_worker_payload.v1"]
    operation: TaskOrchestratorOperation
    checkpoint: dict[str, object] | None
    coding_request: dict[str, object] | None


class FutureSpeakWorkerPayloadV1(TypedDict):
    """Deterministic future-speak scheduling payload kept outside task routing."""

    trigger_at: str
    continuation_objective: str


class BackgroundWorkQueueRequest(TypedDict):
    """Request to persist one reviewed v2 background-work job."""

    job_id: str
    source_action_attempt_id: str
    source_llm_trace_id: NotRequired[str]
    idempotency_key: str
    accepted_task_id: str
    task_identity_key: str
    semantic_objective: str
    goal_continuation_ref: GoalContinuationRefV1 | None
    requested_worker: BackgroundWorkRequestedWorker
    worker_payload: TaskOrchestratorWorkerPayloadV1 | FutureSpeakWorkerPayloadV1
    task_execution_context: NotRequired[dict[str, object]]
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
    """Prompt-safe confirmation that durable work exists."""

    status: Literal["pending", "failed"]
    job_id: str
    job_ref: str
    accepted_task_id: str
    task_identity_key: str
    accepted_task_summary: str
    acknowledgement_constraint: Literal[
        "promise_allowed",
        "promise_forbidden_explain_failure",
    ]
    wait_guidance: Literal["non_numeric_wait", "unavailable"]
    result_summary: str
    accepted_task_state: NotRequired[str]


class BackgroundWorkJobRef(TypedDict):
    """Stable prompt-safe reference for one durable job."""

    job_id: str
    job_ref: str


class BackgroundWorkJobDoc(TypedDict, total=False):
    """MongoDB document for one v2 task-orchestrator or future-speak job."""

    schema_version: Literal["background_work_job.v2"]
    job_id: str
    idempotency_key: str
    source_action_attempt_id: str
    source_llm_trace_id: str
    correlation_write_status: str
    correlation_conflict_source_llm_trace_id: str
    accepted_task_id: str
    task_identity_key: str
    semantic_objective: str
    goal_continuation_ref: GoalContinuationRefV1 | None
    status: BackgroundWorkJobStatus
    delivery_state: BackgroundWorkDeliveryState
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
    requested_worker: BackgroundWorkRequestedWorker
    worker_payload: dict[str, object]
    task_execution_context: dict[str, object]
    task_resolution_result: dict[str, object]
    artifact_text: str
    failure_summary: str
    result_summary: str
    completed_at: str
    delivery_attempt_count: int
    delivery_failure_summary: str
    delivery_tracking_id: str
    delivered_conversation_message_id: str
    delivered_at: str


def background_work_job_ref(job_id: str) -> str:
    """Return the prompt-safe evidence id for one v2 background-work job."""

    clean_job_id = job_id.strip()
    if clean_job_id.startswith("background_work_job:"):
        return clean_job_id
    return f"background_work_job:{clean_job_id}"
