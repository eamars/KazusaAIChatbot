"""Closed contracts for the DSH task-resolution boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from kazusa_ai_chatbot.cognition_episode import (
    GoalContinuationRefV1,
    validate_goal_continuation_ref,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    ResolverCapabilityRequestV2,
    _validate_scene_context,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

if TYPE_CHECKING:
    from kazusa_ai_chatbot.cognition_shared.contracts import SceneContextV2


TASK_RESOLUTION_EXECUTION_CONTEXT_V2_VERSION = (
    "task_resolution_execution_context.v2"
)
DSH_RESOLUTION_REF_V1_VERSION = "dsh_resolution_ref.v1"
TASK_RESOLUTION_ADMISSION_V1_VERSION = "task_resolution_admission.v1"
ACCEPTED_TASK_CONTROL_V1_VERSION = "accepted_task_control.v1"
DSH_TASK_SOURCE_SCOPE_V1_VERSION = "dsh_task_source_scope.v1"
DSH_TASK_START_SPEC_V1_VERSION = "dsh_task_start_spec.v1"
TASK_RESOLUTION_EVIDENCE_VERSION = "task_resolution_evidence.v1"
TASK_RESOLUTION_RESULT_VERSION = "task_resolution_result.v1"

TASK_RESOLUTION_STATUSES = frozenset({
    "resolved",
    "partial",
    "needs_user_input",
    "approval_required",
    "unavailable",
    "failed",
    "deferred",
})
TASK_RESOLUTION_EVIDENCE_STATES = frozenset({
    "complete",
    "partial",
    "pending",
    "missing",
    "blocked",
})
# Kept as the explicit outward name used by the result-source owner.
TASK_RESOLUTION_RESULT_EVIDENCE_STATES = TASK_RESOLUTION_EVIDENCE_STATES
MAX_TASK_RESOLUTION_TEXT_CHARS = 1200
MAX_TASK_RESOLUTION_REASON_CHARS = 600
MAX_TASK_RESOLUTION_LIST_ITEMS = 8
# The cognition projection names this cap in terms of text items.
MAX_TASK_RESOLUTION_TEXT_ITEMS = MAX_TASK_RESOLUTION_LIST_ITEMS
MAX_TASK_RESOLUTION_FACTS = 10


class TaskResolutionContractError(ValueError):
    """Raised when a task-resolution carrier is structurally invalid."""


class TaskResolutionExecutionContextV2(TypedDict):
    """Trusted prompt-safe context shared by Brain and one DSH session."""

    schema_version: Literal["task_resolution_execution_context.v2"]
    character_name: str
    platform: str
    channel_id: str
    channel_type: str
    requester_global_user_id: str
    requester_platform_user_id: str
    requester_display_name: str
    source_message_id: str
    source_platform_bot_id: str
    source_trigger_source: str
    source_llm_trace_id: str
    brain_conversation_ref: str
    scene_context: SceneContextV2
    goal_continuation_ref: GoalContinuationRefV1
    local_time_context: dict[str, object]
    prompt_message_context: dict[str, object]
    chat_history_recent: list[dict[str, object]]
    chat_history_wide: list[dict[str, object]]
    conversation_progress: dict[str, object]
    persona_summary: str
    conversation_summary: str
    current_timestamp_utc: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    session_media_refs: list[dict[str, object]]
    max_output_chars: int


class DshResolutionRefV1(TypedDict):
    """Opaque durable identity for one DSH thread and segment."""

    schema_version: Literal["dsh_resolution_ref.v1"]
    resolution_thread_id: str
    segment_id: str
    dsh_session_id: str
    activation_id: str
    lease_epoch: int
    document_revision: int
    last_committed_seq: int


class TaskResolutionAdmissionV1(TypedDict):
    """Transient model-hidden identity for a queued DSH task admission."""

    schema_version: Literal["task_resolution_admission.v1"]
    accepted_task_id: str
    background_work_job_id: str
    task_session_id: str


class AcceptedTaskControlV1(TypedDict):
    """Typed prompt-safe operation selected for one accepted task."""

    schema_version: Literal["accepted_task_control.v1"]
    accepted_task_ref: str
    operation: Literal["continue", "summarize", "cancel"]
    instruction: str | None


class DshTaskSourceScopeV1(TypedDict):
    """Trusted source scope copied from the execution context."""

    schema_version: Literal["dsh_task_source_scope.v1"]
    platform: str
    channel_id: str
    channel_type: str
    requester_global_user_id: str
    requester_platform_user_id: str
    source_message_id: str
    source_platform_bot_id: str


class DshTaskStartSpecV1(TypedDict):
    """Model-hidden, authority-free DSH opening carrier."""

    schema_version: Literal["dsh_task_start_spec.v1"]
    resolver_request: ResolverCapabilityRequestV2
    execution_context: TaskResolutionExecutionContextV2
    model_facts: list[str]
    model_facts_digest: str
    objective_ref: str


class TaskResolutionEvidenceV1(TypedDict):
    """Prompt-safe DSH evidence receipt projected into the V1 carrier."""

    schema_version: Literal["task_resolution_evidence.v1"]
    evidence_id: str
    task_node_id: str
    specialist: Literal["dsh"]
    summary: str
    provenance_refs: list[str]
    limitations: list[str]


class TaskResolutionResultV1(TypedDict):
    """Stable outward carrier consumed by cognition and background delivery."""

    schema_version: Literal["task_resolution_result.v1"]
    semantic_objective: str
    status: Literal[
        "resolved",
        "partial",
        "needs_user_input",
        "approval_required",
        "unavailable",
        "failed",
        "deferred",
    ]
    scene_context: SceneContextV2
    goal_continuation_ref: GoalContinuationRefV1
    evidence_state: Literal[
        "complete",
        "partial",
        "pending",
        "missing",
        "blocked",
    ]
    evidence_excerpts: list[str]
    evidence_handles: list[str]
    prompt_safe_summary: str
    evidence: list[TaskResolutionEvidenceV1]
    completed_subgoals: list[str]
    remaining_needs: list[str]
    checkpoint: dict[str, object]
    coding_run_context: dict[str, object]


def validate_task_resolution_execution_context(
    value: object,
) -> TaskResolutionExecutionContextV2:
    """Validate the exact V2 context without interpreting user text."""

    data = _mapping(value, "task_resolution_execution_context")
    _exact_keys(data, {
        "schema_version",
        "character_name",
        "platform",
        "channel_id",
        "channel_type",
        "requester_global_user_id",
        "requester_platform_user_id",
        "requester_display_name",
        "source_message_id",
        "source_platform_bot_id",
        "source_trigger_source",
        "source_llm_trace_id",
        "brain_conversation_ref",
        "scene_context",
        "goal_continuation_ref",
        "local_time_context",
        "prompt_message_context",
        "chat_history_recent",
        "chat_history_wide",
        "conversation_progress",
        "persona_summary",
        "conversation_summary",
        "current_timestamp_utc",
        "active_turn_platform_message_ids",
        "active_turn_conversation_row_ids",
        "session_media_refs",
        "max_output_chars",
    }, "task_resolution_execution_context")
    _version(data, TASK_RESOLUTION_EXECUTION_CONTEXT_V2_VERSION, "context")
    for field in (
        "character_name",
        "platform",
        "channel_id",
        "channel_type",
        "requester_global_user_id",
        "requester_platform_user_id",
        "requester_display_name",
        "source_message_id",
        "source_platform_bot_id",
        "source_trigger_source",
        "brain_conversation_ref",
        "current_timestamp_utc",
    ):
        _text(data[field], f"context.{field}")
    _text(data["source_llm_trace_id"], "context.source_llm_trace_id", empty=True)
    scene = data["scene_context"]
    try:
        _validate_scene_context(scene)
    except (TypeError, ValueError) as exc:
        raise TaskResolutionContractError(
            f"context.scene_context is invalid: {exc}",
        ) from exc
    goal = _goal_ref(data["goal_continuation_ref"], "context.goal_continuation_ref")
    local_time = _dict(data["local_time_context"], "context.local_time_context")
    message_context = _dict(
        data["prompt_message_context"],
        "context.prompt_message_context",
    )
    progress = _dict(data["conversation_progress"], "context.conversation_progress")
    recent = _mapping_list(data["chat_history_recent"], "context.chat_history_recent")
    wide = _mapping_list(data["chat_history_wide"], "context.chat_history_wide")
    media = _mapping_list(data["session_media_refs"], "context.session_media_refs")
    platform_ids = _text_list(
        data["active_turn_platform_message_ids"],
        "context.active_turn_platform_message_ids",
    )
    row_ids = _text_list(
        data["active_turn_conversation_row_ids"],
        "context.active_turn_conversation_row_ids",
    )
    max_output = _integer(data["max_output_chars"], "context.max_output_chars")
    if max_output <= 0:
        raise TaskResolutionContractError("context.max_output_chars must be positive")
    return {
        "schema_version": TASK_RESOLUTION_EXECUTION_CONTEXT_V2_VERSION,
        **{
            field: _text(data[field], f"context.{field}")
            for field in (
                "character_name",
                "platform",
                "channel_id",
                "channel_type",
                "requester_global_user_id",
                "requester_platform_user_id",
                "requester_display_name",
                "source_message_id",
                "source_platform_bot_id",
                "source_trigger_source",
                "brain_conversation_ref",
                "current_timestamp_utc",
            )
        },
        "source_llm_trace_id": _text(
            data["source_llm_trace_id"],
            "context.source_llm_trace_id",
            empty=True,
        ),
        "scene_context": cast("SceneContextV2", dict(scene)),
        "goal_continuation_ref": goal,
        "local_time_context": local_time,
        "prompt_message_context": message_context,
        "chat_history_recent": recent,
        "chat_history_wide": wide,
        "conversation_progress": progress,
        "persona_summary": _bounded_text(
            data["persona_summary"], "context.persona_summary",
        ),
        "conversation_summary": _bounded_text(
            data["conversation_summary"], "context.conversation_summary",
        ),
        "active_turn_platform_message_ids": platform_ids,
        "active_turn_conversation_row_ids": row_ids,
        "session_media_refs": media,
        "max_output_chars": max_output,
    }  # type: ignore[return-value]


def validate_dsh_resolution_ref(value: object) -> DshResolutionRefV1:
    """Validate one complete DSH thread/segment reference."""

    data = _mapping(value, "dsh_resolution_ref")
    _exact_keys(data, {
        "schema_version",
        "resolution_thread_id",
        "segment_id",
        "dsh_session_id",
        "activation_id",
        "lease_epoch",
        "document_revision",
        "last_committed_seq",
    }, "dsh_resolution_ref")
    _version(data, DSH_RESOLUTION_REF_V1_VERSION, "dsh_resolution_ref")
    for field in (
        "resolution_thread_id",
        "segment_id",
        "dsh_session_id",
        "activation_id",
    ):
        _text(data[field], f"dsh_resolution_ref.{field}")
    lease_epoch = _integer(data["lease_epoch"], "dsh_resolution_ref.lease_epoch")
    document_revision = _integer(
        data["document_revision"],
        "dsh_resolution_ref.document_revision",
    )
    sequence = _integer(
        data["last_committed_seq"],
        "dsh_resolution_ref.last_committed_seq",
    )
    if lease_epoch < 0 or document_revision < 0 or sequence < 0:
        raise TaskResolutionContractError(
            "dsh_resolution_ref counters must be non-negative",
        )
    return {
        "schema_version": DSH_RESOLUTION_REF_V1_VERSION,
        "resolution_thread_id": str(data["resolution_thread_id"]),
        "segment_id": str(data["segment_id"]),
        "dsh_session_id": str(data["dsh_session_id"]),
        "activation_id": str(data["activation_id"]),
        "lease_epoch": lease_epoch,
        "document_revision": document_revision,
        "last_committed_seq": sequence,
    }


def validate_task_resolution_admission(
    value: object,
) -> TaskResolutionAdmissionV1:
    """Validate the identity returned by background admission."""

    data = _mapping(value, "task_resolution_admission")
    _exact_keys(data, {
        "schema_version",
        "accepted_task_id",
        "background_work_job_id",
        "task_session_id",
    }, "task_resolution_admission")
    _version(
        data,
        TASK_RESOLUTION_ADMISSION_V1_VERSION,
        "task_resolution_admission",
    )
    accepted_task_id = _text(
        data["accepted_task_id"],
        "task_resolution_admission.accepted_task_id",
    )
    background_work_job_id = _text(
        data["background_work_job_id"],
        "task_resolution_admission.background_work_job_id",
    )
    task_session_id = _text(
        data["task_session_id"],
        "task_resolution_admission.task_session_id",
    )
    return {
        "schema_version": TASK_RESOLUTION_ADMISSION_V1_VERSION,
        "accepted_task_id": accepted_task_id,
        "background_work_job_id": background_work_job_id,
        "task_session_id": task_session_id,
    }


def validate_accepted_task_control(value: object) -> AcceptedTaskControlV1:
    """Validate one typed accepted-task operation and its instruction rule."""

    data = _mapping(value, "accepted_task_control")
    _exact_keys(data, {
        "schema_version", "accepted_task_ref", "operation", "instruction",
    }, "accepted_task_control")
    _version(data, ACCEPTED_TASK_CONTROL_V1_VERSION, "accepted_task_control")
    accepted_ref = _text(
        data["accepted_task_ref"],
        "accepted_task_control.accepted_task_ref",
    )
    operation = data["operation"]
    if operation not in {"continue", "summarize", "cancel"}:
        raise TaskResolutionContractError(
            "accepted_task_control.operation is unsupported",
        )
    instruction = data["instruction"]
    if operation == "continue":
        if not isinstance(instruction, str) or not instruction.strip():
            raise TaskResolutionContractError(
                "accepted_task_control.instruction is required",
            )
        normalized_instruction: str | None = _bounded_text(
            instruction,
            "accepted_task_control.instruction",
        )
    else:
        if instruction is not None:
            raise TaskResolutionContractError(
                "accepted_task_control.instruction is operation-specific",
            )
        normalized_instruction = None
    return {
        "schema_version": ACCEPTED_TASK_CONTROL_V1_VERSION,
        "accepted_task_ref": accepted_ref,
        "operation": cast(Literal["continue", "summarize", "cancel"], operation),
        "instruction": normalized_instruction,
    }


def validate_dsh_task_source_scope(value: object) -> DshTaskSourceScopeV1:
    """Validate the exact trusted scope projection used by a binding."""

    data = _mapping(value, "dsh_task_source_scope")
    _exact_keys(data, {
        "schema_version",
        "platform",
        "channel_id",
        "channel_type",
        "requester_global_user_id",
        "requester_platform_user_id",
        "source_message_id",
        "source_platform_bot_id",
    }, "dsh_task_source_scope")
    _version(data, DSH_TASK_SOURCE_SCOPE_V1_VERSION, "dsh_task_source_scope")
    return {
        "schema_version": DSH_TASK_SOURCE_SCOPE_V1_VERSION,
        **{
            field: _text(data[field], f"dsh_task_source_scope.{field}")
            for field in (
                "platform",
                "channel_id",
                "channel_type",
                "requester_global_user_id",
                "requester_platform_user_id",
                "source_message_id",
                "source_platform_bot_id",
            )
        },
    }  # type: ignore[return-value]


def validate_dsh_task_start_spec(value: object) -> DshTaskStartSpecV1:
    """Validate an authority-free DSH start carrier and its digests."""

    data = _mapping(value, "dsh_task_start_spec")
    _exact_keys(data, {
        "schema_version",
        "resolver_request",
        "execution_context",
        "model_facts",
        "model_facts_digest",
        "objective_ref",
    }, "dsh_task_start_spec")
    _version(data, DSH_TASK_START_SPEC_V1_VERSION, "dsh_task_start_spec")
    request = _resolver_request(data["resolver_request"])
    context = validate_task_resolution_execution_context(data["execution_context"])
    facts = _text_list(data["model_facts"], "dsh_task_start_spec.model_facts")
    if len(facts) != MAX_TASK_RESOLUTION_FACTS:
        raise TaskResolutionContractError(
            "dsh_task_start_spec.model_facts must contain exactly ten facts",
        )
    expected_digest = content_digest(facts)
    if data["model_facts_digest"] != expected_digest:
        raise TaskResolutionContractError(
            "dsh_task_start_spec.model_facts_digest does not match facts",
        )
    objective_ref = content_digest(context["goal_continuation_ref"])
    if data["objective_ref"] != objective_ref:
        raise TaskResolutionContractError(
            "dsh_task_start_spec.objective_ref does not match continuation",
        )
    if request["semantic_goal"] != context["prompt_message_context"].get(
        "semantic_goal",
        request["semantic_goal"],
    ):
        # The prompt context is not required to repeat the objective; this
        # branch only prevents an explicitly conflicting trusted projection.
        raise TaskResolutionContractError(
            "dsh_task_start_spec objective projection conflicts",
        )
    if request["goal_continuation_ref"] != context["goal_continuation_ref"]:
        raise TaskResolutionContractError(
            "dsh_task_start_spec continuation projection conflicts",
        )
    return {
        "schema_version": DSH_TASK_START_SPEC_V1_VERSION,
        "resolver_request": request,
        "execution_context": context,
        "model_facts": facts,
        "model_facts_digest": expected_digest,
        "objective_ref": objective_ref,
    }


def validate_task_resolution_evidence(
    value: object,
) -> TaskResolutionEvidenceV1:
    """Validate one DSH-owned prompt-safe evidence row."""

    data = _mapping(value, "task_resolution_evidence")
    _exact_keys(data, {
        "schema_version",
        "evidence_id",
        "task_node_id",
        "specialist",
        "summary",
        "provenance_refs",
        "limitations",
    }, "task_resolution_evidence")
    _version(data, TASK_RESOLUTION_EVIDENCE_VERSION, "task_resolution_evidence")
    specialist = data["specialist"]
    if specialist != "dsh":
        raise TaskResolutionContractError(
            "task_resolution_evidence.specialist must be dsh",
        )
    return {
        "schema_version": TASK_RESOLUTION_EVIDENCE_VERSION,
        "evidence_id": _text(data["evidence_id"], "evidence.evidence_id"),
        "task_node_id": _text(data["task_node_id"], "evidence.task_node_id"),
        "specialist": "dsh",
        "summary": _bounded_text(data["summary"], "evidence.summary"),
        "provenance_refs": _text_list(
            data["provenance_refs"], "evidence.provenance_refs",
        ),
        "limitations": _bounded_text_list(
            data["limitations"], "evidence.limitations",
        ),
    }


def validate_task_resolution_result(value: object) -> TaskResolutionResultV1:
    """Validate the closed result mapping from DSH into Brain."""

    data = _mapping(value, "task_resolution_result")
    _exact_keys(data, {
        "schema_version",
        "semantic_objective",
        "status",
        "scene_context",
        "goal_continuation_ref",
        "evidence_state",
        "evidence_excerpts",
        "evidence_handles",
        "prompt_safe_summary",
        "evidence",
        "completed_subgoals",
        "remaining_needs",
        "checkpoint",
        "coding_run_context",
    }, "task_resolution_result")
    _version(data, TASK_RESOLUTION_RESULT_VERSION, "task_resolution_result")
    objective = _bounded_text(
        data["semantic_objective"],
        "task_resolution_result.semantic_objective",
    )
    status = data["status"]
    if status not in TASK_RESOLUTION_STATUSES:
        raise TaskResolutionContractError("task_resolution_result.status is invalid")
    evidence_state = data["evidence_state"]
    if evidence_state not in TASK_RESOLUTION_EVIDENCE_STATES:
        raise TaskResolutionContractError(
            "task_resolution_result.evidence_state is invalid",
        )
    expected_state = {
        "resolved": "complete",
        "partial": "partial",
        "needs_user_input": "pending",
        "approval_required": "pending",
        "unavailable": "missing",
        "failed": "blocked",
        "deferred": "pending",
    }[str(status)]
    if evidence_state != expected_state:
        raise TaskResolutionContractError(
            "task_resolution_result status/evidence_state mismatch",
        )
    scene = data["scene_context"]
    try:
        _validate_scene_context(scene)
    except (TypeError, ValueError) as exc:
        raise TaskResolutionContractError(
            f"task result scene_context is invalid: {exc}",
        ) from exc
    goal = _goal_ref(
        data["goal_continuation_ref"],
        "task_resolution_result.goal_continuation_ref",
    )
    excerpts = _bounded_text_list(
        data["evidence_excerpts"],
        "task_resolution_result.evidence_excerpts",
    )
    handles = _bounded_text_list(
        data["evidence_handles"],
        "task_resolution_result.evidence_handles",
    )
    if len(set(handles)) != len(handles):
        raise TaskResolutionContractError("task result evidence handles must be unique")
    summary = _bounded_text(
        data["prompt_safe_summary"],
        "task_resolution_result.prompt_safe_summary",
    )
    evidence_value = data["evidence"]
    if not isinstance(evidence_value, list) or len(evidence_value) > MAX_TASK_RESOLUTION_LIST_ITEMS:
        raise TaskResolutionContractError("task result evidence is out of bounds")
    evidence = [validate_task_resolution_evidence(item) for item in evidence_value]
    completed = _bounded_text_list(
        data["completed_subgoals"],
        "task_resolution_result.completed_subgoals",
    )
    remaining = _bounded_text_list(
        data["remaining_needs"],
        "task_resolution_result.remaining_needs",
    )
    checkpoint = _dict(data["checkpoint"], "task_resolution_result.checkpoint")
    if status == "deferred":
        validate_dsh_resolution_ref(checkpoint)
    elif checkpoint:
        raise TaskResolutionContractError(
            "task result checkpoint is only valid for deferred results",
        )
    coding_context = _dict(
        data["coding_run_context"],
        "task_resolution_result.coding_run_context",
    )
    if coding_context:
        raise TaskResolutionContractError(
            "task result coding_run_context must remain empty",
        )
    if status == "partial" and not evidence:
        raise TaskResolutionContractError(
            "partial task result requires DSH evidence",
        )
    return {
        "schema_version": TASK_RESOLUTION_RESULT_VERSION,
        "semantic_objective": objective,
        "status": cast(Any, status),
        "scene_context": cast("SceneContextV2", dict(scene)),
        "goal_continuation_ref": goal,
        "evidence_state": cast(Any, evidence_state),
        "evidence_excerpts": excerpts,
        "evidence_handles": handles,
        "prompt_safe_summary": summary,
        "evidence": evidence,
        "completed_subgoals": completed,
        "remaining_needs": remaining,
        "checkpoint": checkpoint,
        "coding_run_context": {},
    }


def _resolver_request(value: object) -> ResolverCapabilityRequestV2:
    """Validate the task-resolution request embedded by the host."""

    data = _mapping(value, "dsh_task_start_spec.resolver_request")
    _exact_keys(data, {
        "capability",
        "semantic_goal",
        "reason",
        "evidence_handles",
        "start_in_background",
        "goal_continuation_ref",
    }, "resolver_request")
    if data["capability"] != "task_resolution_request":
        raise TaskResolutionContractError(
            "resolver_request.capability must be task_resolution_request",
        )
    if not isinstance(data["start_in_background"], bool):
        raise TaskResolutionContractError(
            "resolver_request.start_in_background must be boolean",
        )
    continuation = _goal_ref(
        data["goal_continuation_ref"],
        "resolver_request.goal_continuation_ref",
    )
    return {
        "capability": "task_resolution_request",
        "semantic_goal": _bounded_text(
            data["semantic_goal"], "resolver_request.semantic_goal",
        ),
        "reason": _bounded_text(data["reason"], "resolver_request.reason"),
        "evidence_handles": _bounded_text_list(
            data["evidence_handles"], "resolver_request.evidence_handles",
        ),
        "start_in_background": data["start_in_background"],
        "goal_continuation_ref": continuation,
    }


def _goal_ref(value: object, field: str) -> GoalContinuationRefV1:
    """Validate a continuation reference and translate its errors."""

    try:
        return validate_goal_continuation_ref(value)
    except (TypeError, ValueError) as exc:
        raise TaskResolutionContractError(f"{field} is invalid: {exc}") from exc


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TaskResolutionContractError(f"{field} must be an object")
    return dict(value)


def _dict(value: object, field: str) -> dict[str, object]:
    return _mapping(value, field)


def _exact_keys(
    data: Mapping[str, object],
    expected: set[str],
    field: str,
) -> None:
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        detail = f"missing={missing}" if missing else f"unknown={extra}"
        raise TaskResolutionContractError(f"{field} fields are not exact: {detail}")


def _version(data: Mapping[str, object], expected: str, field: str) -> None:
    if data.get("schema_version") != expected:
        raise TaskResolutionContractError(f"{field}.schema_version is invalid")


def _text(value: object, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise TaskResolutionContractError(f"{field} must be a non-empty string")
    return value


def _bounded_text(value: object, field: str) -> str:
    text = _text(value, field)
    if len(text) > MAX_TASK_RESOLUTION_TEXT_CHARS:
        raise TaskResolutionContractError(f"{field} exceeds its bound")
    return text


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TaskResolutionContractError(f"{field} must be an integer")
    return value


def _text_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise TaskResolutionContractError(f"{field} must be a list")
    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(_text(item, f"{field}[{index}]"))
    return normalized


def _bounded_text_list(value: object, field: str) -> list[str]:
    normalized = _text_list(value, field)
    if len(normalized) > MAX_TASK_RESOLUTION_LIST_ITEMS:
        raise TaskResolutionContractError(f"{field} exceeds its item bound")
    return [
        _bounded_text(item, f"{field}[{index}]")
        for index, item in enumerate(normalized)
    ]


def _mapping_list(value: object, field: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TaskResolutionContractError(f"{field} must be a list")
    return [_mapping(item, f"{field}[{index}]") for index, item in enumerate(value)]


__all__ = [
    "AcceptedTaskControlV1",
    "DshResolutionRefV1",
    "DshTaskSourceScopeV1",
    "DshTaskStartSpecV1",
    "TaskResolutionAdmissionV1",
    "TaskResolutionContractError",
    "TaskResolutionEvidenceV1",
    "TaskResolutionExecutionContextV2",
    "TaskResolutionResultV1",
    "validate_accepted_task_control",
    "validate_dsh_resolution_ref",
    "validate_dsh_task_source_scope",
    "validate_dsh_task_start_spec",
    "validate_task_resolution_admission",
    "validate_task_resolution_evidence",
    "validate_task_resolution_execution_context",
    "validate_task_resolution_result",
]
