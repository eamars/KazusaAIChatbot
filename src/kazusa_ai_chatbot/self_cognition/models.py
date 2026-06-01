"""Typed constants for self-cognition tracking records."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.time_boundary import LocalTimeContextDoc


CASE_COMMITMENT_BEFORE_DUE = "commitment_before_due"
CASE_COMMITMENT_PAST_DUE = "commitment_past_due"
CASE_COMMITMENT_DUPLICATE_TICK = "commitment_duplicate_tick"
CASE_PRIVATE_NO_ACTION = "private_no_action"
CASE_GROUP_NOISE_REJECTED = "group_noise_rejected"
CASE_GROUP_CHAT_REVIEW = "group_chat_review"
CASE_TOPIC_RAG_FOLLOWUP = "topic_rag_followup"
CASE_SCHEDULED_FUTURE_COGNITION = "scheduled_future_cognition"
SUPPORTED_CASE_NAMES = frozenset(
    (
        CASE_COMMITMENT_BEFORE_DUE,
        CASE_COMMITMENT_PAST_DUE,
        CASE_COMMITMENT_DUPLICATE_TICK,
        CASE_PRIVATE_NO_ACTION,
        CASE_GROUP_NOISE_REJECTED,
        CASE_GROUP_CHAT_REVIEW,
        CASE_TOPIC_RAG_FOLLOWUP,
        CASE_SCHEDULED_FUTURE_COGNITION,
    )
)

TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK = "active_commitment_due_check"
TRIGGER_CONVERSATION_PROGRESS_REVIEW = "conversation_progress_review"
TRIGGER_RECENT_DIRECT_DIALOG_REVIEW = "recent_direct_dialog_review"
TRIGGER_PENDING_OUTBOX_REVIEW = "pending_outbox_review"
TRIGGER_BOUNDED_FOLLOWUP_TOPIC = "bounded_followup_topic"
TRIGGER_GROUP_CHAT_REVIEW = "group_chat_trigger_review"
TRIGGER_SCHEDULED_FUTURE_COGNITION = "scheduled_future_cognition_due"

DUE_STATE_FUTURE_DUE = "future_due"
DUE_STATE_DUE_NOW = "due_now"
DUE_STATE_PAST_DUE = "past_due"
CONTACT_DUE_STATES = frozenset((DUE_STATE_DUE_NOW, DUE_STATE_PAST_DUE))

ROUTE_ACTION_CANDIDATE = "action_candidate"
ROUTE_PROGRESS_MAINTENANCE = "progress_maintenance"
ROUTE_AUDIT_ONLY = "audit_only"
ROUTE_SILENT_NO_WRITE = "silent_no_write"
SUPPORTED_ROUTES = frozenset(
    (
        ROUTE_ACTION_CANDIDATE,
        ROUTE_PROGRESS_MAINTENANCE,
        ROUTE_AUDIT_ONLY,
        ROUTE_SILENT_NO_WRITE,
    )
)
ACTION_KIND_SEND_MESSAGE = "send_message"
ACTION_ATTEMPT_STATUS_CANDIDATE = "candidate"
ACTION_ATTEMPT_STATUS_HELD = "held"
ACTION_ATTEMPT_STATUS_PENDING_HANDOFF = "pending_handoff"
ACTION_ATTEMPT_STATUS_HANDOFF_ACCEPTED = "handoff_accepted"
ACTION_ATTEMPT_STATUS_SCHEDULED = "scheduled"
ACTION_ATTEMPT_STATUS_SENT = "sent"
ACTION_ATTEMPT_STATUS_DELIVERY_FAILED = "delivery_failed"
ACTION_ATTEMPT_STATUS_DUPLICATE = "duplicate_suppressed"
ACTION_ATTEMPT_STATUS_CLOSED_NO_ACTION = "closed_no_action"
ACTION_ATTEMPT_SUPPRESSING_STATUSES = frozenset(
    (
        ACTION_ATTEMPT_STATUS_CANDIDATE,
        ACTION_ATTEMPT_STATUS_HELD,
        ACTION_ATTEMPT_STATUS_PENDING_HANDOFF,
        ACTION_ATTEMPT_STATUS_HANDOFF_ACCEPTED,
        ACTION_ATTEMPT_STATUS_SCHEDULED,
        ACTION_ATTEMPT_STATUS_SENT,
        ACTION_ATTEMPT_STATUS_DUPLICATE,
    )
)

ARTIFACT_TRIGGER_RECORD = "self_cognition_trigger_record.json"
ARTIFACT_RUN_RECORD = "self_cognition_run_record.json"
ARTIFACT_COGNITION_INPUT = "self_cognition_cognition_input.json"
ARTIFACT_COGNITION_OUTPUT = "self_cognition_cognition_output.json"
ARTIFACT_ROUTE_EFFECT = "self_cognition_route_effect.json"
ARTIFACT_ACTION_ATTEMPT = "self_cognition_action_attempt.json"
ARTIFACT_ACTION_CANDIDATE = "self_cognition_action_candidate.json"
ARTIFACT_DISPATCH_RESULT = "self_cognition_dispatch_result.json"
ARTIFACT_CONSOLIDATION_OUTCOME = "self_cognition_consolidation_outcome.json"
ARTIFACT_LOOP_TRACE = "self_cognition_loop_trace.md"
TRACKING_ARTIFACT_NAMES = frozenset(
    (
        ARTIFACT_TRIGGER_RECORD,
        ARTIFACT_RUN_RECORD,
        ARTIFACT_COGNITION_INPUT,
        ARTIFACT_COGNITION_OUTPUT,
        ARTIFACT_ROUTE_EFFECT,
        ARTIFACT_ACTION_ATTEMPT,
        ARTIFACT_ACTION_CANDIDATE,
        ARTIFACT_DISPATCH_RESULT,
        ARTIFACT_CONSOLIDATION_OUTCOME,
        ARTIFACT_LOOP_TRACE,
    )
)

PROGRESS_MAINTENANCE_MARKER = "[PROGRESS_MAINTENANCE]"
AUDIT_ONLY_MARKER = "[AUDIT_ONLY]"
SILENT_NO_WRITE_MARKER = "[SILENT_NO_WRITE]"

RAG_SUPERVISOR_INVOCATION_LIMIT = 1
COGNITION_CALL_LIMIT = 1
DIALOG_RENDER_CALL_LIMIT = 1
TOPIC_LIMIT = 1
DEFAULT_SELF_COGNITION_AFFINITY = 500
STABLE_ID_DIGEST_PREFIX_LENGTH = 24
SOURCE_VISIBLE_CONTEXT_LIMIT = 6
DEFAULT_SELF_COGNITION_ASSISTANT_GLOBAL_USER_ID = (
    "00000000-0000-4000-8000-000000000001"
)
EMPTY_ROUTE_EFFECT_NEXT_TOPIC = None

SELF_COGNITION_INPUT_TEXT = (
    '我所在聊天窗口的最近可见内容。'
)


class SelfCognitionTargetScope(TypedDict):
    """Target surface that an idle self-cognition run may consider."""

    platform: str
    platform_channel_id: str
    channel_type: str
    user_id: str | None
    platform_user_id: NotRequired[str | None]
    display_name: NotRequired[str]


class DeliveryMention(TypedDict):
    """Platform-neutral request for an adapter-rendered outbound mention."""

    entity_kind: str
    placement: str
    platform_user_id: str | None
    global_user_id: str | None
    display_name: str
    requested_by: str


class SelfCognitionDeliveryTarget(TypedDict):
    """Deterministic send destination bound before cognition starts."""

    schema_version: Literal["self_cognition_delivery_target.v1"]
    platform: str
    platform_channel_id: str
    channel_type: Literal["private", "group"]
    target_global_user_id: str | None
    target_platform_user_id: str | None
    source_kind: Literal[
        "target_private_channel",
        "self_cognition_source_channel",
    ]
    source_ref: str
    source_platform_channel_id: str
    source_channel_type: Literal["private", "group"]
    source_message_id: str
    source_global_user_id: str | None
    source_platform_bot_id: str
    source_character_name: str
    guild_id: str | None
    bot_permission_role: str
    fallback_reason: Literal["", "private_channel_unavailable"]


class SelfCognitionTargetBindingFailure(TypedDict):
    """Auditable reason a production case cannot bind a send target."""

    status: Literal["target_binding_failed"]
    reason: Literal[
        "missing_platform",
        "missing_target_user",
        "missing_delivery_target",
        "private_channel_unavailable_and_source_invalid",
        "private_channel_unavailable_and_source_missing",
    ]
    platform: str
    source_ref: str
    source_platform_channel_id: str
    source_channel_type: str
    target_global_user_id: str | None
    target_platform_user_id: str | None


class SelfCognitionSourceRef(TypedDict, total=False):
    """Reference to visible or actionable evidence for one source case."""

    source_kind: str
    source_id: str
    due_at: str | None
    summary: str


class SelfCognitionBudget(TypedDict):
    """Local budget counters recorded in self-cognition run records."""

    rag_calls: int
    cognition_calls: int
    dialog_calls: int
    topic_limit: int


class SelfCognitionCase(TypedDict, total=False):
    """Source-case shape accepted by the self-cognition runner."""

    case_name: str
    case_id: str
    idle_timestamp_utc: str
    last_evidence_timestamp_utc: str
    trigger_kind: str
    target_scope: SelfCognitionTargetScope
    source_refs: list[SelfCognitionSourceRef]
    semantic_due_state: str | None
    actionability: str
    visible_context: list[dict[str, Any]]
    conversation_progress: dict[str, Any]
    group_activity_window: dict[str, Any]
    current_mood: str
    global_vibe: str
    reflection_modifier: dict[str, Any]
    existing_attempts: list[dict[str, Any]]
    character_profile: dict[str, Any]
    user_profile: dict[str, Any]
    platform_bot_id: str
    channel_topic: str
    promoted_reflection_context: dict[str, Any]
    budget: SelfCognitionBudget
    source_scheduled_event_id: str
    source_action_attempt_id: str
    delivery_target: SelfCognitionDeliveryTarget
    target_binding_status: Literal["bound", "failed"]
    target_binding_failure: SelfCognitionTargetBindingFailure


class SourcePacket(TypedDict):
    """Model-facing packet passed into shared cognition."""

    instruction: str
    case_name: str
    idle_local_datetime: str
    last_evidence_local_datetime: str
    local_time_context: LocalTimeContextDoc
    trigger_kind: str
    semantic_due_state: str | None
    actionability: str
    target_scope: SelfCognitionTargetScope
    source_refs: list[SelfCognitionSourceRef]
    visible_context: list[dict[str, Any]]
    conversation_progress: NotRequired[dict[str, Any]]
    group_activity_window: NotRequired[dict[str, Any]]
    current_mood: NotRequired[str]
    global_vibe: NotRequired[str]
    reflection_modifier: NotRequired[dict[str, Any]]
