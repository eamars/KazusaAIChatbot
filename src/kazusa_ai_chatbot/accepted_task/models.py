"""Typed contracts for accepted delayed-task lifecycle state."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

ACCEPTED_TASKS_COLLECTION = "accepted_tasks"
ACCEPTED_TASK_SCHEMA_VERSION = "accepted_task.v1"
ACCEPTED_TASK_REQUESTED_DELIVERY = "send_result_when_done"

AcceptedTaskState = Literal[
    "enqueueing",
    "pending",
    "running",
    "result_ready",
    "failure_ready",
    "delivery_in_progress",
    "delivery_retryable",
    "delivered",
    "enqueue_failed",
    "delivery_exhausted",
    "cancelled",
    "superseded",
]
AcceptedTaskResultKind = Literal["none", "artifact", "failure"]
AcceptedTaskCreateStatus = Literal["created", "already_active"]
AcceptedTaskStatusCheckStatus = Literal["active", "none"]

ACTIVE_ACCEPTED_TASK_STATES = (
    "enqueueing",
    "pending",
    "running",
    "result_ready",
    "failure_ready",
    "delivery_in_progress",
    "delivery_retryable",
)
TERMINAL_ACCEPTED_TASK_STATES = (
    "delivered",
    "enqueue_failed",
    "delivery_exhausted",
    "cancelled",
    "superseded",
)


class AcceptedTaskIdentityMaterial(TypedDict):
    """Stable material used to build an active accepted-task identity."""

    source_platform: str
    source_channel_id: str
    source_channel_type: str
    requester_global_user_id: str
    requester_platform_user_id: str
    action_kind: str
    accepted_task_seed: str
    accepted_task_detail: str


class AcceptedTaskCreateRequest(TypedDict):
    """Request to create or resolve one accepted delayed user task."""

    action_kind: str
    accepted_task_seed: str
    accepted_task_detail: str
    accepted_task_summary: str
    source_context: str
    requested_delivery: Literal["send_result_when_done"]
    max_output_chars: int
    source_trigger_source: str
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


class AcceptedTaskStatusCheckRequest(TypedDict, total=False):
    """Trusted scope used to find an active accepted task."""

    source_platform: str
    source_channel_id: str
    source_channel_type: str
    requester_global_user_id: str
    requester_platform_user_id: str


class AcceptedTaskDoc(TypedDict, total=False):
    """MongoDB document for a user-facing accepted delayed task."""

    schema_version: Literal["accepted_task.v1"]
    accepted_task_id: str
    task_identity_key: str
    active_identity_key: str
    task_identity_material: AcceptedTaskIdentityMaterial
    action_kind: str
    first_source_message_id: str
    related_source_message_ids: list[str]
    source_trigger_source: str
    state: AcceptedTaskState
    result_kind: AcceptedTaskResultKind
    executor_kind: Literal["background_work"]
    executor_ref: str
    accepted_task_summary: str
    source_context: str
    requested_delivery: Literal["send_result_when_done"]
    max_output_chars: int
    source_platform: str
    source_channel_id: str
    source_channel_type: str
    source_platform_bot_id: str
    source_character_name: str
    requester_global_user_id: str
    requester_platform_user_id: str
    requester_display_name: str
    created_at: str
    updated_at: str
    started_at: str
    completed_at: str
    delivered_at: str
    result_summary: str
    artifact_text: str
    failure_summary: str
    delivery_failure_summary: str
    delivery_tracking_id: str
    delivered_conversation_message_id: str
    last_progress_reported_at: str


class AcceptedTaskCreateResult(TypedDict):
    """Result of resolving an accepted task against active duplicates."""

    status: AcceptedTaskCreateStatus
    task: AcceptedTaskDoc


class AcceptedTaskStatusResult(TypedDict):
    """Result of checking active accepted-task state for a trusted scope."""

    status: AcceptedTaskStatusCheckStatus
    task: NotRequired[AcceptedTaskDoc]
