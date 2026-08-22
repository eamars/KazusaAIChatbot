"""Typed public contracts for bounded task-resolution sessions.

The task-resolution package owns one resumable semantic task session.  The
contracts in this module keep specialist selection, evidence, and durable
checkpoint state separate from adapter, worker, database, and coding-agent
internals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    GoalContinuationRefV1,
    validate_goal_continuation_ref,
)

if TYPE_CHECKING:
    from kazusa_ai_chatbot.cognition_shared.contracts import SceneContextV2


TASK_RESOLUTION_EXECUTION_CONTEXT_VERSION = "task_resolution_execution_context.v1"
TASK_RESOLUTION_CHECKPOINT_VERSION = "task_resolution_checkpoint.v1"
TASK_RESOLUTION_NODE_VERSION = "task_resolution_node.v1"
TASK_RESOLUTION_EVIDENCE_VERSION = "task_resolution_evidence.v1"
TASK_PENDING_DISPATCH_VERSION = "task_pending_dispatch.v1"
TASK_SPECIALIST_REQUEST_VERSION = "task_specialist_request.v1"
TASK_SPECIALIST_RESULT_VERSION = "task_specialist_result.v1"
TASK_RESOLUTION_RESULT_VERSION = "task_resolution_result.v1"
CODING_RUN_CONTEXT_VERSION = "coding_run_context.v1"

MAX_TASK_RESOLUTION_DISPATCHES = 4
MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS = 4
MAX_TASK_RESOLUTION_ROUTE_CORRECTIONS = 2
MAX_TASK_RESOLUTION_SPECIALIST_INVOCATIONS = 2
MAX_TASK_RESOLUTION_NODES = 8
MAX_TASK_RESOLUTION_EVIDENCE = 8
MAX_TASK_RESOLUTION_TEXT_ITEMS = 8
MAX_TASK_RESOLUTION_TRACE_ENTRIES = 4
MAX_TASK_RESOLUTION_TEXT_CHARS = 1200
MAX_TASK_RESOLUTION_REASON_CHARS = 600

TASK_SPECIALISTS = frozenset((
    "local_context",
    "public_research",
    "coding",
    "text_computation",
))
TASK_CODING_OBJECTIVE_MODES = frozenset((
    "none",
    "read_only",
    "propose_patch",
))
TASK_CODING_SPECIALIST_OBJECTIVE_MODES = frozenset((
    "read_only",
    "propose_patch",
))
TASK_PENDING_DISPATCH_PHASES = frozenset((
    "selected",
    "started",
))
TASK_SPECIALIST_STATUSES = frozenset((
    "resolved",
    "partial",
    "incompatible",
    "temporarily_unavailable",
    "needs_user_input",
    "approval_required",
    "failed",
))
TASK_RESOLUTION_STATUSES = frozenset((
    "resolved",
    "partial",
    "needs_user_input",
    "approval_required",
    "unavailable",
    "failed",
    "deferred",
))
TASK_RESOLUTION_RESULT_EVIDENCE_STATES = frozenset((
    "complete",
    "partial",
    "pending",
    "missing",
    "blocked",
))
TASK_NODE_STATUSES = frozenset((
    "pending",
    "resolving",
    "resolved",
    "blocked",
    "incompatible",
))


class TaskResolutionContractError(ValueError):
    """Raised when a task-resolution payload violates its public contract."""


class TaskResolutionExecutionContextV1(TypedDict):
    """Trusted prompt-safe context available to specialist adapters."""

    schema_version: Literal["task_resolution_execution_context.v1"]
    character_name: str
    platform: str
    channel_id: str
    channel_type: str
    requester_global_user_id: str
    requester_platform_user_id: str
    source_message_id: str
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
    coding_workspace_root: str
    max_output_chars: int


class TaskResolutionNodeV1(TypedDict):
    """One bounded semantic objective inside a task-resolution session."""

    schema_version: Literal["task_resolution_node.v1"]
    node_id: str
    objective: str
    status: str
    depends_on: list[str]


class TaskResolutionEvidenceV1(TypedDict):
    """Prompt-safe evidence returned by one registered specialist."""

    schema_version: Literal["task_resolution_evidence.v1"]
    evidence_id: str
    task_node_id: str
    specialist: str
    summary: str
    provenance_refs: list[str]
    limitations: list[str]


class TaskSpecialistAttemptV1(TypedDict):
    """Durable record of one incompatible node/specialist combination."""

    task_node_id: str
    specialist: str


class TaskSpecialistInvocationCountV1(TypedDict):
    """Persisted invocation count for one node and specialist."""

    task_node_id: str
    specialist: str
    count: int


class TaskResolutionTraceEntryV1(TypedDict):
    """Bounded prompt-safe dispatch trace row."""

    dispatch_index: int
    task_node_id: str
    specialist: str
    result_status: str
    reason: str


class TaskPendingDispatchV1(TypedDict):
    """One durable specialist selection awaiting or undergoing execution."""

    schema_version: Literal["task_pending_dispatch.v1"]
    task_node_id: str
    specialist: str
    subgoal: str
    coding_objective_mode: str
    phase: str


class TaskResolutionCheckpointV1(TypedDict):
    """Resumable deterministic state for one semantic task session."""

    schema_version: Literal["task_resolution_checkpoint.v1"]
    session_id: str
    semantic_objective: str
    scene_context: SceneContextV2
    goal_continuation_ref: GoalContinuationRefV1
    source_scope: dict[str, str]
    nodes: list[TaskResolutionNodeV1]
    active_node_id: str
    evidence: list[TaskResolutionEvidenceV1]
    remaining_needs: list[str]
    attempted_specialists: list[TaskSpecialistAttemptV1]
    dispatch_count: int
    orchestrator_call_count: int
    route_correction_count: int
    specialist_invocation_counts: list[TaskSpecialistInvocationCountV1]
    pending_dispatch: TaskPendingDispatchV1 | None
    terminal_status: str
    trace_summary: list[TaskResolutionTraceEntryV1]


class TaskSpecialistRequestV1(TypedDict):
    """Canonical request from the orchestrator to one specialist handler."""

    schema_version: Literal["task_specialist_request.v1"]
    task_node_id: str
    objective: str
    available_evidence: list[TaskResolutionEvidenceV1]
    remaining_needs: list[str]
    trusted_scope: dict[str, str]
    coding_objective_mode: str


class TaskSpecialistResultV1(TypedDict, total=False):
    """Typed result returned by exactly one specialist handler."""

    schema_version: Literal["task_specialist_result.v1"]
    specialist: str
    status: str
    evidence: list[TaskResolutionEvidenceV1]
    completed_subgoals: list[str]
    remaining_needs: list[str]
    reason: str
    retryable: bool
    coding_run_context: dict[str, object]


class TaskResolutionResultV1(TypedDict):
    """Prompt-safe terminal or deferred session result."""

    schema_version: Literal["task_resolution_result.v1"]
    semantic_objective: str
    status: str
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
) -> TaskResolutionExecutionContextV1:
    """Validate trusted prompt-safe context before a specialist receives it."""

    data = _require_mapping(value, "task_resolution_execution_context")
    _require_exact_keys(
        data,
        {
            "schema_version",
            "character_name",
            "platform",
            "channel_id",
            "channel_type",
            "requester_global_user_id",
            "requester_platform_user_id",
            "source_message_id",
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
            "coding_workspace_root",
            "max_output_chars",
        },
        "task_resolution_execution_context",
    )
    _require_version(data, TASK_RESOLUTION_EXECUTION_CONTEXT_VERSION)
    normalized: TaskResolutionExecutionContextV1 = {
        "schema_version": TASK_RESOLUTION_EXECUTION_CONTEXT_VERSION,
        "character_name": _require_text(data, "character_name"),
        "platform": _require_text(data, "platform"),
        "channel_id": _require_text(data, "channel_id"),
        "channel_type": _require_text(data, "channel_type"),
        "requester_global_user_id": _require_text(
            data,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _require_text(
            data,
            "requester_platform_user_id",
        ),
        "source_message_id": _require_text(data, "source_message_id"),
        "scene_context": _validate_scene_context_value(
            data["scene_context"],
            "scene_context",
        ),
        "goal_continuation_ref": _validate_goal_continuation_ref_value(
            data["goal_continuation_ref"],
            "goal_continuation_ref",
        ),
        "local_time_context": _require_dict(data, "local_time_context"),
        "prompt_message_context": _require_dict(
            data,
            "prompt_message_context",
        ),
        "chat_history_recent": _require_mapping_list(
            data,
            "chat_history_recent",
        ),
        "chat_history_wide": _require_mapping_list(data, "chat_history_wide"),
        "conversation_progress": _require_dict(data, "conversation_progress"),
        "persona_summary": _require_text(data, "persona_summary", allow_empty=True),
        "conversation_summary": _require_text(
            data,
            "conversation_summary",
            allow_empty=True,
        ),
        "current_timestamp_utc": _require_text(data, "current_timestamp_utc"),
        "active_turn_platform_message_ids": _require_text_list(
            data,
            "active_turn_platform_message_ids",
        ),
        "active_turn_conversation_row_ids": _require_text_list(
            data,
            "active_turn_conversation_row_ids",
        ),
        "session_media_refs": _require_mapping_list(data, "session_media_refs"),
        "coding_workspace_root": _require_text(
            data,
            "coding_workspace_root",
            allow_empty=True,
        ),
        "max_output_chars": _require_positive_int(data, "max_output_chars"),
    }
    return normalized


def validate_task_resolution_checkpoint(
    value: object,
) -> TaskResolutionCheckpointV1:
    """Validate bounded checkpoint state before inline or queued execution.

    Args:
        value: Candidate persistent state supplied by the inline service or
            background-worker resume path.

    Returns:
        A normalized checkpoint containing only validated semantic state.
    """

    data = _require_mapping(value, "task_resolution_checkpoint")
    _require_exact_keys(
        data,
        {
            "schema_version",
            "session_id",
            "semantic_objective",
            "scene_context",
            "goal_continuation_ref",
            "source_scope",
            "nodes",
            "active_node_id",
            "evidence",
            "remaining_needs",
            "attempted_specialists",
            "dispatch_count",
            "orchestrator_call_count",
            "route_correction_count",
            "specialist_invocation_counts",
            "pending_dispatch",
            "terminal_status",
            "trace_summary",
        },
        "task_resolution_checkpoint",
    )
    _require_version(data, TASK_RESOLUTION_CHECKPOINT_VERSION)
    nodes = _validate_nodes(data)
    active_node_id = _require_text(data, "active_node_id")
    node_ids = {node["node_id"] for node in nodes}
    if active_node_id not in node_ids:
        raise TaskResolutionContractError(
            "active_node_id: expected an existing task node"
        )
    evidence = _validate_evidence_list(data, "evidence")
    attempted_specialists = _validate_attempted_specialists(data)
    invocation_counts = _validate_invocation_counts(data)
    dispatch_count = _require_nonnegative_int(data, "dispatch_count")
    if dispatch_count > MAX_TASK_RESOLUTION_DISPATCHES:
        raise TaskResolutionContractError(
            "dispatch_count: exceeds task-resolution dispatch cap"
        )
    orchestrator_call_count = _require_nonnegative_int(
        data,
        "orchestrator_call_count",
    )
    if orchestrator_call_count > MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS:
        raise TaskResolutionContractError(
            "orchestrator_call_count: exceeds task-resolution call cap"
        )
    route_correction_count = _require_nonnegative_int(
        data,
        "route_correction_count",
    )
    if route_correction_count > MAX_TASK_RESOLUTION_ROUTE_CORRECTIONS:
        raise TaskResolutionContractError(
            "route_correction_count: exceeds task-resolution correction cap"
    )
    trace_summary = _validate_trace_entries(data)
    pending_dispatch = _validate_pending_dispatch(data)
    _validate_checkpoint_node_references(
        node_ids=node_ids,
        evidence=evidence,
        attempted_specialists=attempted_specialists,
        invocation_counts=invocation_counts,
        trace_summary=trace_summary,
        pending_dispatch=pending_dispatch,
    )
    _validate_checkpoint_counters(
        dispatch_count=dispatch_count,
        route_correction_count=route_correction_count,
        attempted_specialists=attempted_specialists,
        invocation_counts=invocation_counts,
        trace_summary=trace_summary,
        pending_dispatch=pending_dispatch,
    )
    terminal_status = _require_text(data, "terminal_status", allow_empty=True)
    if terminal_status and terminal_status not in TASK_RESOLUTION_STATUSES - {
        "deferred"
    }:
        raise TaskResolutionContractError("terminal_status: unsupported value")
    if terminal_status and pending_dispatch is not None:
        raise TaskResolutionContractError(
            "pending_dispatch: terminal checkpoints cannot retain a dispatch"
        )
    normalized: TaskResolutionCheckpointV1 = {
        "schema_version": TASK_RESOLUTION_CHECKPOINT_VERSION,
        "session_id": _require_text(data, "session_id"),
        "semantic_objective": _require_text(data, "semantic_objective"),
        "scene_context": _validate_scene_context_value(
            data["scene_context"],
            "scene_context",
        ),
        "goal_continuation_ref": _validate_goal_continuation_ref_value(
            data["goal_continuation_ref"],
            "goal_continuation_ref",
        ),
        "source_scope": _validate_source_scope(data, "source_scope"),
        "nodes": nodes,
        "active_node_id": active_node_id,
        "evidence": evidence,
        "remaining_needs": _require_bounded_text_list(
            data,
            "remaining_needs",
        ),
        "attempted_specialists": attempted_specialists,
        "dispatch_count": dispatch_count,
        "orchestrator_call_count": orchestrator_call_count,
        "route_correction_count": route_correction_count,
        "specialist_invocation_counts": invocation_counts,
        "pending_dispatch": pending_dispatch,
        "terminal_status": terminal_status,
        "trace_summary": trace_summary,
    }
    return normalized


def validate_task_specialist_request(value: object) -> TaskSpecialistRequestV1:
    """Validate one canonical specialist request before adapter mapping."""

    data = _require_mapping(value, "task_specialist_request")
    _require_exact_keys(
        data,
        {
            "schema_version",
            "task_node_id",
            "objective",
            "available_evidence",
            "remaining_needs",
            "trusted_scope",
            "coding_objective_mode",
        },
        "task_specialist_request",
    )
    _require_version(data, TASK_SPECIALIST_REQUEST_VERSION)
    normalized: TaskSpecialistRequestV1 = {
        "schema_version": TASK_SPECIALIST_REQUEST_VERSION,
        "task_node_id": _require_text(data, "task_node_id"),
        "objective": _require_text(data, "objective"),
        "available_evidence": _validate_evidence_list(
            data,
            "available_evidence",
        ),
        "remaining_needs": _require_bounded_text_list(
            data,
            "remaining_needs",
        ),
        "trusted_scope": _validate_source_scope(data, "trusted_scope"),
        "coding_objective_mode": _require_enum(
            data,
            "coding_objective_mode",
            TASK_CODING_OBJECTIVE_MODES,
        ),
    }
    return normalized


def validate_task_specialist_result(value: object) -> TaskSpecialistResultV1:
    """Validate one specialist outcome before it changes session state."""

    data = _require_mapping(value, "task_specialist_result")
    allowed_keys = {
        "schema_version",
        "specialist",
        "status",
        "evidence",
        "completed_subgoals",
        "remaining_needs",
        "reason",
        "retryable",
        "coding_run_context",
    }
    _require_allowed_keys(data, allowed_keys, "task_specialist_result")
    _require_required_keys(
        data,
        allowed_keys - {"coding_run_context"},
        "task_specialist_result",
    )
    _require_version(data, TASK_SPECIALIST_RESULT_VERSION)
    specialist = _require_enum(data, "specialist", TASK_SPECIALISTS)
    status = _require_enum(data, "status", TASK_SPECIALIST_STATUSES)
    evidence = _validate_evidence_list(data, "evidence")
    if status == "partial" and not evidence:
        raise TaskResolutionContractError(
            "partial specialist result requires validated evidence"
        )
    retryable = _require_bool(data, "retryable")
    if status == "temporarily_unavailable" and not retryable:
        raise TaskResolutionContractError(
            "retryable: temporarily_unavailable requires a retryable result"
        )
    if retryable and status != "temporarily_unavailable":
        raise TaskResolutionContractError(
            "retryable: only temporarily_unavailable may be retryable"
        )
    coding_run_context = _validate_specialist_coding_context(
        data,
        specialist=specialist,
    )
    for evidence_row in evidence:
        if evidence_row["specialist"] != specialist:
            raise TaskResolutionContractError(
                "evidence.specialist: expected result specialist"
            )
    normalized: TaskSpecialistResultV1 = {
        "schema_version": TASK_SPECIALIST_RESULT_VERSION,
        "specialist": specialist,
        "status": status,
        "evidence": evidence,
        "completed_subgoals": _require_bounded_text_list(
            data,
            "completed_subgoals",
        ),
        "remaining_needs": _require_bounded_text_list(
            data,
            "remaining_needs",
        ),
        "reason": _require_text(
            data,
            "reason",
            maximum=MAX_TASK_RESOLUTION_REASON_CHARS,
        ),
        "retryable": retryable,
    }
    if coding_run_context:
        normalized["coding_run_context"] = coding_run_context
    return normalized


def validate_task_resolution_result(value: object) -> TaskResolutionResultV1:
    """Validate a terminal or deferred task-resolution result.

    A partial result is successful only when it retains at least one evidence
    record with provenance.  This prevents completed-subgoal claims from
    becoming an ungrounded visible outcome.
    """

    data = _require_mapping(value, "task_resolution_result")
    _require_exact_keys(
        data,
        {
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
        },
        "task_resolution_result",
    )
    _require_version(data, TASK_RESOLUTION_RESULT_VERSION)
    semantic_objective = _require_text(data, "semantic_objective")
    status = _require_enum(data, "status", TASK_RESOLUTION_STATUSES)
    scene_context = _validate_scene_context_value(
        data["scene_context"],
        "scene_context",
    )
    continuation_ref = _validate_goal_continuation_ref_value(
        data["goal_continuation_ref"],
        "goal_continuation_ref",
    )
    evidence_state = _require_enum(
        data,
        "evidence_state",
        TASK_RESOLUTION_RESULT_EVIDENCE_STATES,
    )
    evidence = _validate_evidence_list(data, "evidence")
    evidence_excerpts = _require_bounded_text_list(
        data,
        "evidence_excerpts",
    )
    evidence_handles = _require_text_list(data, "evidence_handles")
    completed_subgoals = _require_bounded_text_list(
        data,
        "completed_subgoals",
    )
    remaining_needs = _require_bounded_text_list(
        data,
        "remaining_needs",
    )
    _validate_result_evidence_projection(
        status=status,
        evidence_state=evidence_state,
        evidence=evidence,
        evidence_excerpts=evidence_excerpts,
        evidence_handles=evidence_handles,
    )
    if status == "resolved":
        if not evidence:
            raise TaskResolutionContractError(
                "resolved task-resolution result requires validated evidence"
            )
        if remaining_needs:
            raise TaskResolutionContractError(
                "resolved task-resolution result cannot retain remaining needs"
            )
    if status == "partial":
        if not evidence:
            raise TaskResolutionContractError(
                "partial task-resolution result requires validated evidence"
            )
        if not remaining_needs:
            raise TaskResolutionContractError(
                "partial task-resolution result requires remaining needs"
            )
    checkpoint = _validate_result_checkpoint(data)
    if checkpoint:
        _validate_result_checkpoint_binding(
            checkpoint,
            semantic_objective=semantic_objective,
            scene_context=scene_context,
            continuation_ref=continuation_ref,
        )
    if status == "deferred":
        if not checkpoint:
            raise TaskResolutionContractError(
                "deferred task-resolution result requires a checkpoint"
            )
        dispatch_count = checkpoint["dispatch_count"]
        if dispatch_count >= MAX_TASK_RESOLUTION_DISPATCHES:
            raise TaskResolutionContractError(
                "deferred checkpoint has exhausted dispatch budget"
            )
        if checkpoint["terminal_status"]:
            raise TaskResolutionContractError(
                "deferred checkpoint must remain nonterminal"
            )
        if (
            checkpoint["orchestrator_call_count"]
            >= MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS
            and checkpoint["pending_dispatch"] is None
        ):
            raise TaskResolutionContractError(
                "deferred checkpoint has exhausted orchestrator-call budget"
            )
    if status in {
        "needs_user_input",
        "approval_required",
        "unavailable",
        "failed",
    } and not remaining_needs:
        raise TaskResolutionContractError(
            "objective-scoped task status requires remaining needs"
        )
    coding_run_context = _validate_result_coding_context(data)
    normalized: TaskResolutionResultV1 = {
        "schema_version": TASK_RESOLUTION_RESULT_VERSION,
        "semantic_objective": semantic_objective,
        "status": status,
        "scene_context": scene_context,
        "goal_continuation_ref": continuation_ref,
        "evidence_state": evidence_state,
        "evidence_excerpts": evidence_excerpts,
        "evidence_handles": evidence_handles,
        "prompt_safe_summary": _require_text(
            data,
            "prompt_safe_summary",
            maximum=MAX_TASK_RESOLUTION_TEXT_CHARS,
        ),
        "evidence": evidence,
        "completed_subgoals": completed_subgoals,
        "remaining_needs": remaining_needs,
        "checkpoint": checkpoint,
        "coding_run_context": coding_run_context,
    }
    return normalized


def validate_task_resolution_evidence(value: object) -> TaskResolutionEvidenceV1:
    """Validate one provenance-bearing specialist evidence row."""

    data = _require_mapping(value, "task_resolution_evidence")
    _require_exact_keys(
        data,
        {
            "schema_version",
            "evidence_id",
            "task_node_id",
            "specialist",
            "summary",
            "provenance_refs",
            "limitations",
        },
        "task_resolution_evidence",
    )
    _require_version(data, TASK_RESOLUTION_EVIDENCE_VERSION)
    provenance_refs = _require_text_list(data, "provenance_refs")
    if not provenance_refs:
        raise TaskResolutionContractError(
            "provenance_refs: evidence requires at least one provenance ref"
        )
    normalized: TaskResolutionEvidenceV1 = {
        "schema_version": TASK_RESOLUTION_EVIDENCE_VERSION,
        "evidence_id": _require_text(data, "evidence_id"),
        "task_node_id": _require_text(data, "task_node_id"),
        "specialist": _require_enum(data, "specialist", TASK_SPECIALISTS),
        "summary": _require_text(
            data,
            "summary",
            maximum=MAX_TASK_RESOLUTION_TEXT_CHARS,
        ),
        "provenance_refs": provenance_refs,
        "limitations": _require_bounded_text_list(data, "limitations"),
    }
    return normalized


def _validate_nodes(data: Mapping[str, object]) -> list[TaskResolutionNodeV1]:
    """Validate the bounded node set stored in a checkpoint."""

    raw_nodes = _require_list(data, "nodes")
    if not raw_nodes or len(raw_nodes) > MAX_TASK_RESOLUTION_NODES:
        raise TaskResolutionContractError("nodes: exceeds task-resolution cap")
    nodes: list[TaskResolutionNodeV1] = []
    node_ids: set[str] = set()
    for raw_node in raw_nodes:
        node = _validate_node(raw_node)
        if node["node_id"] in node_ids:
            raise TaskResolutionContractError("nodes: duplicate node_id")
        node_ids.add(node["node_id"])
        nodes.append(node)
    for node in nodes:
        for dependency_id in node["depends_on"]:
            if dependency_id not in node_ids:
                raise TaskResolutionContractError(
                    "nodes.depends_on: expected an existing task node"
                )
    return nodes


def _validate_node(value: object) -> TaskResolutionNodeV1:
    """Validate one task node without cross-node reference checks."""

    data = _require_mapping(value, "task_resolution_node")
    _require_exact_keys(
        data,
        {"schema_version", "node_id", "objective", "status", "depends_on"},
        "task_resolution_node",
    )
    _require_version(data, TASK_RESOLUTION_NODE_VERSION)
    normalized: TaskResolutionNodeV1 = {
        "schema_version": TASK_RESOLUTION_NODE_VERSION,
        "node_id": _require_text(data, "node_id"),
        "objective": _require_text(data, "objective"),
        "status": _require_enum(data, "status", TASK_NODE_STATUSES),
        "depends_on": _require_text_list(data, "depends_on"),
    }
    return normalized


def _validate_evidence_list(
    data: Mapping[str, object],
    field_name: str,
) -> list[TaskResolutionEvidenceV1]:
    """Validate one bounded list of provenance-bearing evidence rows."""

    raw_rows = _require_list(data, field_name)
    if len(raw_rows) > MAX_TASK_RESOLUTION_EVIDENCE:
        raise TaskResolutionContractError(
            f"{field_name}: exceeds task-resolution evidence cap"
        )
    evidence: list[TaskResolutionEvidenceV1] = []
    evidence_ids: set[str] = set()
    for raw_row in raw_rows:
        row = validate_task_resolution_evidence(raw_row)
        if row["evidence_id"] in evidence_ids:
            raise TaskResolutionContractError(f"{field_name}: duplicate evidence_id")
        evidence_ids.add(row["evidence_id"])
        evidence.append(row)
    return evidence


def _validate_pending_dispatch(
    data: Mapping[str, object],
) -> TaskPendingDispatchV1 | None:
    """Validate the one durable selection that may cross a lease boundary."""

    raw_pending_dispatch = data.get("pending_dispatch")
    if raw_pending_dispatch is None:
        return None
    pending_dispatch = _require_mapping(
        raw_pending_dispatch,
        "pending_dispatch",
    )
    _require_exact_keys(
        pending_dispatch,
        {
            "schema_version",
            "task_node_id",
            "specialist",
            "subgoal",
            "coding_objective_mode",
            "phase",
        },
        "pending_dispatch",
    )
    _require_version(pending_dispatch, TASK_PENDING_DISPATCH_VERSION)
    specialist = _require_enum(
        pending_dispatch,
        "specialist",
        TASK_SPECIALISTS,
    )
    coding_objective_mode = _require_enum(
        pending_dispatch,
        "coding_objective_mode",
        TASK_CODING_OBJECTIVE_MODES,
    )
    if specialist == "coding":
        if coding_objective_mode not in TASK_CODING_SPECIALIST_OBJECTIVE_MODES:
            raise TaskResolutionContractError(
                "coding_objective_mode: coding requires read_only or propose_patch"
            )
    elif coding_objective_mode != "none":
        raise TaskResolutionContractError(
            "coding_objective_mode: non-coding specialists require none"
        )
    normalized: TaskPendingDispatchV1 = {
        "schema_version": TASK_PENDING_DISPATCH_VERSION,
        "task_node_id": _require_text(pending_dispatch, "task_node_id"),
        "specialist": specialist,
        "subgoal": _require_text(pending_dispatch, "subgoal"),
        "coding_objective_mode": coding_objective_mode,
        "phase": _require_enum(
            pending_dispatch,
            "phase",
            TASK_PENDING_DISPATCH_PHASES,
        ),
    }
    return normalized


def _validate_attempted_specialists(
    data: Mapping[str, object],
) -> list[TaskSpecialistAttemptV1]:
    """Validate the durable non-repeat ledger for incompatible dispatches."""

    raw_rows = _require_list(data, "attempted_specialists")
    if len(raw_rows) > MAX_TASK_RESOLUTION_DISPATCHES:
        raise TaskResolutionContractError(
            "attempted_specialists: exceeds dispatch cap"
        )
    attempts: list[TaskSpecialistAttemptV1] = []
    pairs: set[tuple[str, str]] = set()
    for raw_row in raw_rows:
        row = _require_mapping(raw_row, "attempted_specialists")
        _require_exact_keys(
            row,
            {"task_node_id", "specialist"},
            "attempted_specialists",
        )
        task_node_id = _require_text(row, "task_node_id")
        specialist = _require_enum(row, "specialist", TASK_SPECIALISTS)
        pair = (task_node_id, specialist)
        if pair in pairs:
            raise TaskResolutionContractError(
                "attempted_specialists: duplicate task node and specialist pair"
            )
        pairs.add(pair)
        attempts.append({
            "task_node_id": task_node_id,
            "specialist": specialist,
        })
    return attempts


def _validate_invocation_counts(
    data: Mapping[str, object],
) -> list[TaskSpecialistInvocationCountV1]:
    """Validate same-specialist invocation counts for every task node."""

    raw_rows = _require_list(data, "specialist_invocation_counts")
    if len(raw_rows) > MAX_TASK_RESOLUTION_NODES:
        raise TaskResolutionContractError(
            "specialist_invocation_counts: exceeds node cap"
        )
    counts: list[TaskSpecialistInvocationCountV1] = []
    pairs: set[tuple[str, str]] = set()
    for raw_row in raw_rows:
        row = _require_mapping(raw_row, "specialist_invocation_counts")
        _require_exact_keys(
            row,
            {"task_node_id", "specialist", "count"},
            "specialist_invocation_counts",
        )
        task_node_id = _require_text(row, "task_node_id")
        specialist = _require_enum(row, "specialist", TASK_SPECIALISTS)
        count = _require_positive_int(row, "count")
        if count > MAX_TASK_RESOLUTION_SPECIALIST_INVOCATIONS:
            raise TaskResolutionContractError(
                "specialist_invocation_counts.count: exceeds specialist cap"
            )
        pair = (task_node_id, specialist)
        if pair in pairs:
            raise TaskResolutionContractError(
                "specialist_invocation_counts: duplicate task node and specialist pair"
            )
        pairs.add(pair)
        counts.append({
            "task_node_id": task_node_id,
            "specialist": specialist,
            "count": count,
        })
    return counts


def _validate_trace_entries(
    data: Mapping[str, object],
) -> list[TaskResolutionTraceEntryV1]:
    """Validate prompt-safe trace rows without raw worker diagnostics."""

    raw_rows = _require_list(data, "trace_summary")
    if len(raw_rows) > MAX_TASK_RESOLUTION_TRACE_ENTRIES:
        raise TaskResolutionContractError("trace_summary: exceeds dispatch cap")
    traces: list[TaskResolutionTraceEntryV1] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        row = _require_mapping(raw_row, "trace_summary")
        _require_exact_keys(
            row,
            {
                "dispatch_index",
                "task_node_id",
                "specialist",
                "result_status",
                "reason",
            },
            "trace_summary",
        )
        dispatch_index = _require_positive_int(row, "dispatch_index")
        if dispatch_index != index:
            raise TaskResolutionContractError(
                "trace_summary.dispatch_index: expected contiguous values"
            )
        traces.append({
            "dispatch_index": dispatch_index,
            "task_node_id": _require_text(row, "task_node_id"),
            "specialist": _require_enum(row, "specialist", TASK_SPECIALISTS),
            "result_status": _require_enum(
                row,
                "result_status",
                TASK_SPECIALIST_STATUSES,
            ),
            "reason": _require_text(
                row,
                "reason",
                maximum=MAX_TASK_RESOLUTION_REASON_CHARS,
            ),
        })
    return traces


def _validate_checkpoint_node_references(
    *,
    node_ids: set[str],
    evidence: Sequence[TaskResolutionEvidenceV1],
    attempted_specialists: Sequence[TaskSpecialistAttemptV1],
    invocation_counts: Sequence[TaskSpecialistInvocationCountV1],
    trace_summary: Sequence[TaskResolutionTraceEntryV1],
    pending_dispatch: TaskPendingDispatchV1 | None,
) -> None:
    """Require every persisted specialist row to target a declared task node."""

    rows_by_label: tuple[tuple[str, Sequence[Mapping[str, object]]], ...] = (
        ("evidence", evidence),
        ("attempted_specialists", attempted_specialists),
        ("specialist_invocation_counts", invocation_counts),
        ("trace_summary", trace_summary),
    )
    for label, rows in rows_by_label:
        for row in rows:
            task_node_id = row["task_node_id"]
            if task_node_id not in node_ids:
                raise TaskResolutionContractError(
                    f"{label}.task_node_id: expected an existing task node"
                )
    if (
        pending_dispatch is not None
        and pending_dispatch["task_node_id"] not in node_ids
    ):
        raise TaskResolutionContractError(
            "pending_dispatch.task_node_id: expected an existing task node"
        )


def _validate_checkpoint_counters(
    *,
    dispatch_count: int,
    route_correction_count: int,
    attempted_specialists: Sequence[TaskSpecialistAttemptV1],
    invocation_counts: Sequence[TaskSpecialistInvocationCountV1],
    trace_summary: Sequence[TaskResolutionTraceEntryV1],
    pending_dispatch: TaskPendingDispatchV1 | None,
) -> None:
    """Require counters and durable ledgers to represent the same dispatches."""

    started_dispatch = (
        pending_dispatch
        if pending_dispatch is not None and pending_dispatch["phase"] == "started"
        else None
    )
    expected_trace_total = len(trace_summary) + (1 if started_dispatch else 0)
    if dispatch_count != expected_trace_total:
        raise TaskResolutionContractError(
            "dispatch_count: expected the completed and started dispatch total"
        )
    invocation_total = sum(row["count"] for row in invocation_counts)
    if dispatch_count != invocation_total:
        raise TaskResolutionContractError(
            "dispatch_count: expected the specialist invocation total"
        )
    if route_correction_count != len(attempted_specialists):
        raise TaskResolutionContractError(
            "route_correction_count: expected attempted_specialists total"
        )
    _validate_trace_invocation_pairs(
        invocation_counts,
        trace_summary,
        started_dispatch=started_dispatch,
    )
    _validate_attempt_trace_pairs(attempted_specialists, trace_summary)


def _validate_trace_invocation_pairs(
    invocation_counts: Sequence[TaskSpecialistInvocationCountV1],
    trace_summary: Sequence[TaskResolutionTraceEntryV1],
    *,
    started_dispatch: TaskPendingDispatchV1 | None,
) -> None:
    """Require persisted invocation rows to equal the trace pair counts."""

    trace_counts: dict[tuple[str, str], int] = {}
    for trace_row in trace_summary:
        pair = (trace_row["task_node_id"], trace_row["specialist"])
        trace_counts[pair] = trace_counts.get(pair, 0) + 1
    if started_dispatch is not None:
        started_pair = (
            started_dispatch["task_node_id"],
            started_dispatch["specialist"],
        )
        trace_counts[started_pair] = trace_counts.get(started_pair, 0) + 1
    persisted_counts = {
        (row["task_node_id"], row["specialist"]): row["count"]
        for row in invocation_counts
    }
    if persisted_counts != trace_counts:
        raise TaskResolutionContractError(
            "specialist_invocation_counts: expected trace_summary pair counts"
        )


def _validate_attempt_trace_pairs(
    attempted_specialists: Sequence[TaskSpecialistAttemptV1],
    trace_summary: Sequence[TaskResolutionTraceEntryV1],
) -> None:
    """Require each retained incompatible pair to have an incompatible trace."""

    incompatible_pairs = {
        (trace_row["task_node_id"], trace_row["specialist"])
        for trace_row in trace_summary
        if trace_row["result_status"] == "incompatible"
    }
    for attempt in attempted_specialists:
        pair = (attempt["task_node_id"], attempt["specialist"])
        if pair not in incompatible_pairs:
            raise TaskResolutionContractError(
                "attempted_specialists: expected an incompatible trace"
            )


def _validate_source_scope(
    data: Mapping[str, object],
    field_name: str,
) -> dict[str, str]:
    """Validate the trusted requester and conversation scope for one task."""

    scope = _require_mapping(data.get(field_name), field_name)
    _require_exact_keys(
        scope,
        {
            "trigger_source",
            "platform",
            "channel_id",
            "channel_type",
            "source_message_id",
            "requester_global_user_id",
            "requester_platform_user_id",
        },
        field_name,
    )
    normalized = {
        "trigger_source": _require_text(scope, "trigger_source"),
        "platform": _require_text(scope, "platform"),
        "channel_id": _require_text(scope, "channel_id"),
        "channel_type": _require_text(scope, "channel_type"),
        "source_message_id": _require_text(scope, "source_message_id"),
        "requester_global_user_id": _require_text(
            scope,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _require_text(
            scope,
            "requester_platform_user_id",
        ),
    }
    return normalized


def _validate_specialist_coding_context(
    data: Mapping[str, object],
    *,
    specialist: str,
) -> dict[str, object]:
    """Accept coding-run context only from the frozen coding adapter."""

    raw_context = data.get("coding_run_context", {})
    context = _require_mapping(raw_context, "coding_run_context")
    if specialist != "coding" and context:
        raise TaskResolutionContractError(
            "coding_run_context: only the coding specialist may emit it"
        )
    if not context:
        return {}
    return _validate_coding_run_context(context)


def _validate_result_coding_context(
    data: Mapping[str, object],
) -> dict[str, object]:
    """Validate a prompt-safe coding-run projection in a final result."""

    raw_context = data["coding_run_context"]
    context = _require_mapping(raw_context, "coding_run_context")
    if not context:
        return {}
    normalized = _validate_coding_run_context(context)
    return normalized


def _validate_result_evidence_projection(
    *,
    status: str,
    evidence_state: str,
    evidence: list[TaskResolutionEvidenceV1],
    evidence_excerpts: list[str],
    evidence_handles: list[str],
) -> None:
    """Keep result-visible evidence tied to validated task evidence only."""

    expected_state = {
        "resolved": "complete",
        "partial": "partial",
        "deferred": "pending",
        "needs_user_input": "blocked",
        "approval_required": "blocked",
        "unavailable": "blocked",
        "failed": "blocked",
    }[status]
    if evidence_state != expected_state:
        raise TaskResolutionContractError(
            "evidence_state: does not match task-resolution status"
        )
    if evidence_state not in {"complete", "partial"}:
        if evidence_excerpts or evidence_handles:
            raise TaskResolutionContractError(
                "non-factual task state cannot expose evidence excerpts or handles"
            )
        return
    expected_excerpts = [row["summary"] for row in evidence]
    expected_handles = [row["evidence_id"] for row in evidence]
    if evidence_excerpts != expected_excerpts:
        raise TaskResolutionContractError(
            "evidence_excerpts: expected validated evidence summaries"
        )
    if evidence_handles != expected_handles:
        raise TaskResolutionContractError(
            "evidence_handles: expected validated evidence identifiers"
        )


def _validate_result_checkpoint_binding(
    checkpoint: dict[str, object],
    *,
    semantic_objective: str,
    scene_context: SceneContextV2,
    continuation_ref: GoalContinuationRefV1,
) -> None:
    """Require a result checkpoint to retain the same task context lineage."""

    if checkpoint["semantic_objective"] != semantic_objective:
        raise TaskResolutionContractError(
            "checkpoint.semantic_objective: conflicts with result"
        )
    if checkpoint["scene_context"] != scene_context:
        raise TaskResolutionContractError(
            "checkpoint.scene_context: conflicts with result"
        )
    if checkpoint["goal_continuation_ref"] != continuation_ref:
        raise TaskResolutionContractError(
            "checkpoint.goal_continuation_ref: conflicts with result"
        )


def _validate_coding_run_context(
    context: Mapping[str, object],
) -> dict[str, object]:
    """Validate the stable public coding-run projection used by this package."""

    _require_exact_keys(
        context,
        {
            "schema_version",
            "coding_run_ref",
            "status",
            "summary",
            "limitations",
            "allowed_next_actions",
            "followup_open",
        },
        "coding_run_context",
    )
    _require_version(context, CODING_RUN_CONTEXT_VERSION)
    allowed_actions = _require_text_list(context, "allowed_next_actions")
    limitations = _require_bounded_text_list(context, "limitations")
    normalized = {
        "schema_version": CODING_RUN_CONTEXT_VERSION,
        "coding_run_ref": _require_text(context, "coding_run_ref"),
        "status": _require_text(context, "status"),
        "summary": _require_text(
            context,
            "summary",
            maximum=MAX_TASK_RESOLUTION_TEXT_CHARS,
        ),
        "limitations": limitations,
        "allowed_next_actions": allowed_actions,
        "followup_open": _require_bool(context, "followup_open"),
    }
    return normalized


def _validate_result_checkpoint(
    data: Mapping[str, object],
) -> dict[str, object]:
    """Validate an optional checkpoint attached to a result envelope."""

    raw_checkpoint = data["checkpoint"]
    checkpoint = _require_mapping(raw_checkpoint, "checkpoint")
    if not checkpoint:
        return {}
    validated = validate_task_resolution_checkpoint(checkpoint)
    normalized = cast(dict[str, object], dict(validated))
    return normalized


def _validate_scene_context_value(
    value: object,
    field_name: str,
) -> SceneContextV2:
    """Reuse the canonical bounded scene validator at this public boundary."""

    if not isinstance(value, Mapping):
        raise TaskResolutionContractError(f"{field_name}: expected object")

    # Local import only: a module-scope import would cycle through
    # cognition_core_v3.facade -> llm_tracing -> db.background_work_jobs,
    # which imports this module.
    from kazusa_ai_chatbot.cognition_shared.contracts import (
        CognitionContractError,
        _validate_scene_context,
    )

    try:
        _validate_scene_context(value)
    except CognitionContractError as exc:
        raise TaskResolutionContractError(
            f"{field_name}: invalid canonical scene context: {exc}"
        ) from exc
    return cast("SceneContextV2", dict(value))


def _validate_goal_continuation_ref_value(
    value: object,
    field_name: str,
) -> GoalContinuationRefV1:
    """Require one exact deterministic continuation reference."""

    try:
        continuation_ref = validate_goal_continuation_ref(value)
    except CognitiveEpisodeValidationError as exc:
        raise TaskResolutionContractError(
            f"{field_name}: invalid goal continuation reference: {exc}"
        ) from exc
    return continuation_ref


def _require_mapping(value: object, label: str) -> dict[str, object]:
    """Return a mutable mapping or raise a precise contract error."""

    if not isinstance(value, Mapping):
        raise TaskResolutionContractError(f"{label}: expected object")
    normalized = dict(value)
    return normalized


def _require_list(data: Mapping[str, object], field_name: str) -> list[object]:
    """Return a list field or raise a precise contract error."""

    value = data.get(field_name)
    if not isinstance(value, list):
        raise TaskResolutionContractError(f"{field_name}: expected list")
    normalized = list(value)
    return normalized


def _require_dict(data: Mapping[str, object], field_name: str) -> dict[str, object]:
    """Return one required mapping field."""

    value = data.get(field_name)
    normalized = _require_mapping(value, field_name)
    return normalized


def _require_text(
    data: Mapping[str, object],
    field_name: str,
    *,
    allow_empty: bool = False,
    maximum: int = MAX_TASK_RESOLUTION_TEXT_CHARS,
) -> str:
    """Return one bounded semantic string from a typed payload."""

    value = data.get(field_name)
    if not isinstance(value, str):
        raise TaskResolutionContractError(f"{field_name}: expected string")
    normalized = value.strip()
    if (not allow_empty and not normalized) or len(normalized) > maximum:
        raise TaskResolutionContractError(f"{field_name}: invalid text")
    return normalized


def _require_text_list(
    data: Mapping[str, object],
    field_name: str,
) -> list[str]:
    """Return one bounded list of non-empty semantic strings."""

    return _normalize_text_list(_require_list(data, field_name), field_name)


def _require_bounded_text_list(
    data: Mapping[str, object],
    field_name: str,
) -> list[str]:
    """Return one bounded list of short prompt-safe semantic strings."""

    return _normalize_text_list(_require_list(data, field_name), field_name)


def _normalize_text_list(values: Sequence[object], field_name: str) -> list[str]:
    """Normalize one short semantic text list with strict structural bounds."""

    if len(values) > MAX_TASK_RESOLUTION_TEXT_ITEMS:
        raise TaskResolutionContractError(
            f"{field_name}: exceeds task-resolution text-item cap"
        )
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TaskResolutionContractError(
                f"{field_name}: expected strings"
            )
        item = value.strip()
        if not item or len(item) > MAX_TASK_RESOLUTION_TEXT_CHARS:
            raise TaskResolutionContractError(f"{field_name}: invalid text item")
        normalized.append(item)
    return normalized


def _require_mapping_list(
    data: Mapping[str, object],
    field_name: str,
) -> list[dict[str, object]]:
    """Return one bounded mapping list used by trusted context adapters."""

    values = _require_list(data, field_name)
    if len(values) > MAX_TASK_RESOLUTION_TEXT_ITEMS:
        raise TaskResolutionContractError(
            f"{field_name}: exceeds task-resolution context cap"
        )
    normalized: list[dict[str, object]] = []
    for value in values:
        row = _require_mapping(value, field_name)
        normalized.append(row)
    return normalized


def _require_positive_int(data: Mapping[str, object], field_name: str) -> int:
    """Return one positive integer field."""

    value = data.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TaskResolutionContractError(
            f"{field_name}: expected positive integer"
        )
    return value


def _require_nonnegative_int(data: Mapping[str, object], field_name: str) -> int:
    """Return one non-negative integer field."""

    value = data.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TaskResolutionContractError(
            f"{field_name}: expected non-negative integer"
        )
    return value


def _require_bool(data: Mapping[str, object], field_name: str) -> bool:
    """Return one exact boolean field."""

    value = data.get(field_name)
    if not isinstance(value, bool):
        raise TaskResolutionContractError(f"{field_name}: expected boolean")
    return value


def _require_enum(
    data: Mapping[str, object],
    field_name: str,
    allowed_values: frozenset[str],
) -> str:
    """Return one closed semantic enum field."""

    value = data.get(field_name)
    if not isinstance(value, str) or value not in allowed_values:
        raise TaskResolutionContractError(f"{field_name}: unsupported value")
    return value


def _require_version(data: Mapping[str, object], version: str) -> None:
    """Require one exact version marker on a typed public envelope."""

    schema_version = data.get("schema_version")
    if schema_version != version:
        raise TaskResolutionContractError(
            f"schema_version: expected {version}"
        )


def _require_exact_keys(
    data: Mapping[str, object],
    required_keys: set[str],
    label: str,
) -> None:
    """Require one payload to contain exactly its declared public fields."""

    _require_required_keys(data, required_keys, label)
    _require_allowed_keys(data, required_keys, label)


def _require_required_keys(
    data: Mapping[str, object],
    required_keys: set[str],
    label: str,
) -> None:
    """Require every declared mandatory field on one payload."""

    missing = sorted(required_keys - set(data))
    if missing:
        raise TaskResolutionContractError(
            f"{label}: missing required fields: {', '.join(missing)}"
        )


def _require_allowed_keys(
    data: Mapping[str, object],
    allowed_keys: set[str],
    label: str,
) -> None:
    """Reject undeclared fields that could leak runtime implementation data."""

    unsupported = sorted(set(data) - allowed_keys)
    if unsupported:
        raise TaskResolutionContractError(
            f"{unsupported[0]}: unsupported {label} field"
        )
