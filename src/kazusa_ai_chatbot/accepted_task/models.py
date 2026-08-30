"""Typed contracts for the v2 accepted-task lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.cognition_episode import GoalContinuationRefV1

ACCEPTED_TASKS_COLLECTION = "accepted_tasks"
ACCEPTED_TASK_SCHEMA_VERSION = "accepted_task.v2"
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
AcceptedTaskKind = Literal[
    "task_resolution",
    "future_speak",
]
AcceptedTaskCompletionStatus = Literal["none", "resolved", "partial", "failed"]
AcceptedTaskResultKind = Literal[
    "none",
    "resolved",
    "partial",
    "needs_user_input",
    "approval_required",
    "unavailable",
    "failed",
]
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
    """Stable task-kind-specific identity from trusted scope and lineage."""

    task_kind: AcceptedTaskKind
    source_platform: str
    source_channel_id: str
    source_channel_type: str
    requester_global_user_id: str
    requester_platform_user_id: str
    goal_continuation_ref: NotRequired[GoalContinuationRefV1]
    semantic_objective: NotRequired[str]


class AcceptedTaskCreateRequest(TypedDict):
    """Trusted request to create one task-resolution-backed lifecycle row."""

    task_kind: AcceptedTaskKind
    semantic_objective: str
    accepted_task_summary: str
    goal_continuation_ref: GoalContinuationRefV1 | None
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
    scheduled_future_speech_authority: NotRequired[dict[str, Any]]


class AcceptedTaskStatusCheckRequest(TypedDict, total=False):
    """Trusted scope used to find one active accepted task."""

    source_platform: str
    source_channel_id: str
    source_channel_type: str
    requester_global_user_id: str
    requester_platform_user_id: str


class AcceptedTaskDoc(TypedDict, total=False):
    """MongoDB document for one v2 user-facing accepted task."""

    schema_version: Literal["accepted_task.v2"]
    accepted_task_id: str
    task_identity_key: str
    active_identity_key: str
    task_identity_material: AcceptedTaskIdentityMaterial
    task_kind: AcceptedTaskKind
    semantic_objective: str
    goal_continuation_ref: GoalContinuationRefV1 | None
    first_source_message_id: str
    related_source_message_ids: list[str]
    source_trigger_source: str
    state: AcceptedTaskState
    completion_status: AcceptedTaskCompletionStatus
    result_kind: AcceptedTaskResultKind
    executor_kind: Literal["background_work"]
    executor_ref: str
    accepted_task_summary: str
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
    remaining_needs: list[str]
    failure_summary: str
    delivery_failure_summary: str
    delivery_tracking_id: str
    delivered_conversation_message_id: str
    last_progress_reported_at: str
    revision: int
    dsh_task_session_id: str
    dsh_operation_generation: int
    dsh_followup_open: bool
    dsh_followup_claim_action_attempt_id: str | None
    scheduled_future_speech_authority: NotRequired[dict[str, Any]]


class AcceptedTaskCreateResult(TypedDict):
    """Result of active-duplicate resolution for one accepted task."""

    status: AcceptedTaskCreateStatus
    task: AcceptedTaskDoc


class AcceptedTaskStatusResult(TypedDict):
    """Result of checking one trusted scope for active work."""

    status: AcceptedTaskStatusCheckStatus
    task: NotRequired[AcceptedTaskDoc]


class DshAcceptedTaskAffordanceV1(TypedDict):
    """Prompt-safe affordance for one delivered DSH-backed task."""

    schema_version: Literal["dsh_accepted_task_affordance.v1"]
    accepted_task_ref: str
    task_state: str
    objective_summary: str
    latest_summary: str
    allowed_next_actions: list[Literal["continue", "summarize", "cancel"]]
    followup_open: bool
    updated_at: str


def project_dsh_task_affordance(
    task: Mapping[str, object],
    binding: Mapping[str, object],
) -> DshAcceptedTaskAffordanceV1:
    """Project only safe task controls; omit durable identity and authority."""

    accepted_task_id = _required_text(task, "accepted_task_id")
    state = _required_text(task, "state")
    followup_open = bool(task.get("dsh_followup_open", False))
    binding_state = binding.get("state")
    if (
        followup_open
        and state == "delivered"
        and binding_state == "terminal"
    ):
        actions: list[Literal["continue", "summarize", "cancel"]] = [
            "continue",
            "summarize",
            "cancel",
        ]
    elif state in {"pending", "running", "enqueueing"}:
        actions = ["cancel"]
    else:
        actions = []
    latest = task.get("result_summary") or task.get("accepted_task_summary")
    objective = task.get("semantic_objective") or task.get("accepted_task_summary")
    updated_at = task.get("updated_at")
    return {
        "schema_version": "dsh_accepted_task_affordance.v1",
        "accepted_task_ref": f"accepted_task:{accepted_task_id}",
        "task_state": state,
        "objective_summary": str(objective or ""),
        "latest_summary": str(latest or ""),
        "allowed_next_actions": actions,
        "followup_open": followup_open,
        "updated_at": str(updated_at or ""),
    }


def _required_text(value: Mapping[str, object], field_name: str) -> str:
    field_value = value.get(field_name)
    if not isinstance(field_value, str) or not field_value.strip():
        raise ValueError(f"{field_name} is required")
    return field_value.strip()
