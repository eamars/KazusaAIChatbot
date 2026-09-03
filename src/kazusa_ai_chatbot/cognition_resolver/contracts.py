"""Structural contracts for cognition resolver recurrence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    GoalContinuationRefV1,
    validate_goal_continuation_ref,
)

if TYPE_CHECKING:
    from kazusa_ai_chatbot.action_spec.models import (
        ActionSpecV1,
        EvidenceRefV1,
    )

RESOLVER_CYCLE_STATE_VERSION = "resolver_cycle_state.v1"
RESOLVER_CAPABILITY_REQUEST_VERSION = "resolver_capability_request.v1"
RESOLVER_OBSERVATION_VERSION = "resolver_observation.v1"
RESOLVER_CYCLE_TRACE_VERSION = "resolver_cycle_trace.v1"
RESOLVER_PENDING_RESUME_VERSION = "resolver_pending_resume.v3"
RESOLVER_PENDING_CONTINUATION_VERSION = "resolver_pending_continuation.v3"
PENDING_TASK_CONTINUATION_VERSION = "pending_task_continuation.v1"
RESOLVER_PENDING_RESOLUTION_VERSION = "resolver_pending_resolution.v1"
RESOLVER_GOAL_PROGRESS_VERSION = "resolver_goal_progress.v1"
RESOLVER_EVIDENCE_STATE_VERSION = "resolver_evidence_state.v1"
REQUIRED_RESOLVER_EVIDENCE_DEPENDENCY_VERSION = (
    "required_resolver_evidence_dependency.v1"
)
TERMINAL_RESOLVER_SURFACE_DECISION = "explain terminal evidence blocker"
SHARED_MEMORY_PREWARM_OUTCOME_VERSION = "shared_memory_prewarm_outcome.v1"
MAX_SHARED_MEMORY_PREWARM_ENTRIES = 24
MAX_SHARED_MEMORY_PREWARM_LATENCY_MS = 120000
MAX_SHARED_MEMORY_PREWARM_JSON_CHARS = 65536
MAX_RESOLVER_SUMMARY_CHARS = 600
MAX_RESOLVER_OBJECTIVE_CHARS = 400
MAX_RESOLVER_REASON_CHARS = 400
MAX_RESOLVER_TRACE_CHARS = 600
MAX_RESOLVER_GOAL_FIELD_CHARS = 500
MAX_RESOLVER_GOAL_ITEM_CHARS = 240
MAX_RESOLVER_GOAL_ITEMS = 8
MAX_RESOLVER_RAG_EVIDENCE_SUMMARY_CHARS = 320
MAX_RESOLVER_RAG_EVIDENCE_ITEMS = 4
MAX_RESOLVER_KNOWLEDGE_ITEMS = 8
MAX_RESOLVER_EVIDENCE_EXCERPTS = 4
MAX_RESOLVER_EVIDENCE_EXCERPT_CHARS = 500

SHARED_MEMORY_PREWARM_RAG_FIELDS = frozenset({
    "answer",
    "user_image",
    "user_memory_unit_candidates",
    "character_image",
    "third_party_profiles",
    "memory_evidence",
    "recall_evidence",
    "conversation_evidence",
    "external_evidence",
    "supervisor_trace",
})
SHARED_MEMORY_PREWARM_STATUSES = frozenset({
    "completed",
    "empty",
    "skipped",
    "failed",
})
SHARED_MEMORY_PREWARM_REASON_CODES = frozenset({
    "not_first_cycle",
    "unsupported_episode",
    "empty_query_after_character_mention",
    "worker_error",
    "worker_contract_invalid",
    "worker_unresolved",
    "no_shared_memory",
    "projection_failed",
    "shared_memory_ready",
    "shared_memory_merged",
})
SHARED_MEMORY_PREWARM_REASON_STATUS = {
    "not_first_cycle": "skipped",
    "unsupported_episode": "skipped",
    "empty_query_after_character_mention": "skipped",
    "worker_error": "failed",
    "worker_contract_invalid": "failed",
    "worker_unresolved": "empty",
    "no_shared_memory": "empty",
    "projection_failed": "failed",
    "shared_memory_ready": "completed",
    "shared_memory_merged": "completed",
}

_RAW_MARKER_RE = re.compile(r"\braw-[A-Za-z0-9_-]+")

ALLOWED_RESOLVER_CAPABILITIES = frozenset((
    "task_resolution_request",
    "human_clarification",
    "approval_preparation",
    "self_goal_resolution",
))
RESOLVER_CAPABILITY_SEMANTICS = {
    "approval_preparation": (
        "Prepare one minimal approval question before an allowed side effect."
    ),
    "human_clarification": (
        "Ask for one missing user-controlled fact only when the task objective "
        "itself is not yet bounded and cognition cannot formulate or authorize "
        "it. A known prerequisite keeps task admission closed until the user "
        "supplies it."
    ),
    "task_resolution_request": (
        "Select this only when the current request supplies a bounded executable "
        "semantic task objective and the character judges that required evidence "
        "must be obtained. A missing known user-controlled fact or choice belongs "
        "to human_clarification and blocks admission; do not admit the task first "
        "or use this capability to solicit that prerequisite."
    ),
    "self_goal_resolution": (
        "Resolve or prioritize one internal self-cognition goal for an "
        "eligible private source."
    ),
}
ALLOWED_RESOLVER_PRIORITIES = frozenset(("now", "background"))
ALLOWED_OBSERVATION_STATUSES = frozenset(("succeeded", "blocked", "failed"))
ALLOWED_OBSERVATION_BLOCKER_KINDS = frozenset(("requires_user_input",))
ALLOWED_RESOLVER_STATES = frozenset((
    "running",
    "terminal",
    "blocked",
    "max_cycles",
    "waiting_for_user",
    "waiting_for_approval",
))
ALLOWED_PENDING_CAPABILITIES = frozenset((
    "human_clarification",
    "approval_preparation",
))
ALLOWED_PENDING_STATUSES = frozenset((
    "waiting_for_user",
    "waiting_for_approval",
    "closed",
    "expired",
    "superseded",
))
ALLOWED_PENDING_DECISIONS = frozenset((
    "continue_waiting",
    "answered",
    "approved",
    "rejected",
    "superseded",
))
ALLOWED_PENDING_DISPOSITIONS = frozenset((
    "continue_waiting",
    "answered",
    "rejected",
    "superseded",
))
ALLOWED_PENDING_TASK_CONTINUATIONS = frozenset((
    "no_task_admission",
    "foreground_task_admission",
    "background_task_admission",
))
ALLOWED_GOAL_DELIVERABLE_STATUSES = frozenset((
    "pending",
    "partial",
    "satisfied",
    "blocked",
))
ALLOWED_RESOLVER_EVIDENCE_STATES = frozenset((
    "complete",
    "partial",
    "pending",
    "missing",
    "blocked",
))


class ResolverValidationError(ValueError):
    """Raised when a resolver contract payload is structurally invalid."""


SharedMemoryPrewarmStatus = Literal[
    "completed",
    "empty",
    "skipped",
    "failed",
]
SharedMemoryPrewarmReasonCode = Literal[
    "not_first_cycle",
    "unsupported_episode",
    "empty_query_after_character_mention",
    "worker_error",
    "worker_contract_invalid",
    "worker_unresolved",
    "no_shared_memory",
    "projection_failed",
    "shared_memory_ready",
    "shared_memory_merged",
]


class SharedMemoryPrewarmOutcomeV1(TypedDict):
    """Bounded diagnostic outcome for one shared-memory prewarm attempt."""

    schema_version: Literal["shared_memory_prewarm_outcome.v1"]
    status: SharedMemoryPrewarmStatus
    reason_code: SharedMemoryPrewarmReasonCode
    attempted: bool
    latency_ms: int
    retrieved_shared_count: int
    merged_shared_count: int
    rag_result: dict[str, object]


class ResolverCapabilityRequestV1(TypedDict):
    """A cognition-selected request for one bounded resolver capability."""

    schema_version: Literal["resolver_capability_request.v1"]
    capability_kind: Literal[
        "task_resolution_request",
        "human_clarification",
        "approval_preparation",
        "self_goal_resolution",
    ]
    objective: str
    reason: str
    priority: Literal["now", "background"]
    goal_continuation_ref: GoalContinuationRefV1 | None


ResolverObservationBlockerV1 = Literal["requires_user_input"]


class ResolverObservationV1(TypedDict):
    """Prompt-safe result returned by one deterministic capability handler."""

    schema_version: Literal["resolver_observation.v1"]
    observation_id: str
    capability_kind: str
    request_objective: str
    request_reason: str
    status: Literal["succeeded", "blocked", "failed"]
    prompt_safe_summary: str
    rag_result: NotRequired[dict]
    knowledge_projection: NotRequired[dict[str, object]]
    pending_resume_id: NotRequired[str]
    blocker_kind: NotRequired[ResolverObservationBlockerV1]
    task_resolution_evidence_state: NotRequired[
        ResolverEvidenceStateV1
    ]
    goal_continuation_ref: NotRequired[GoalContinuationRefV1]
    evidence_refs: list[EvidenceRefV1]
    created_at_utc: str


class ResolverEvidenceStateV1(TypedDict):
    """Typed answer-evidence disposition for a task-resolution observation."""

    schema_version: Literal["resolver_evidence_state.v1"]
    state: Literal["complete", "partial", "pending", "missing", "blocked"]
    remaining_needs: list[str]


class RequiredResolverEvidenceDependencyV1(TypedDict):
    """Exact resolver dependency bound to one accepted answer-evidence request."""

    schema_version: Literal["required_resolver_evidence_dependency.v1"]
    accepted_request_handle: str
    observation_id: str
    prompt_safe_observation_handle: str
    capability_kind: Literal["task_resolution_request"]
    state: Literal["complete", "partial", "pending", "missing", "blocked"]
    evidence_handles: list[str]
    remaining_needs: list[str]
    goal_continuation_ref: GoalContinuationRefV1


class ResolverCycleTraceV1(TypedDict):
    """Prompt-safe review row for one full cognition resolver cycle."""

    schema_version: Literal["resolver_cycle_trace.v1"]
    cycle_index: int
    status_before_cycle: str
    l1_emotional_appraisal: str
    l1_interaction_subtext: str
    l2_internal_monologue_summary: str
    l2_logical_stance: str
    l2_character_intent: str
    l2_judgment_note: str
    l2d_resolver_capability_requests: list[ResolverCapabilityRequestV1]
    l2d_action_specs_summary: list[str]
    selected_capability_kind: str
    observation_ids: list[str]
    final_surface_decision: str
    terminal_reason: str
    created_at_utc: str


class PendingTaskContinuationV1(TypedDict):
    """P-authored answer-conditioned admission state for a clarification."""

    schema_version: Literal["pending_task_continuation.v1"]
    on_answered_clarification: Literal[
        "no_task_admission",
        "foreground_task_admission",
        "background_task_admission",
    ]


class ResolverPendingResumeV3(TypedDict):
    """Durable pending HIL or approval row projected into later cognition."""

    schema_version: Literal["resolver_pending_resume.v3"]
    resume_id: str
    capability_kind: Literal["human_clarification", "approval_preparation"]
    status: Literal[
        "waiting_for_user",
        "waiting_for_approval",
        "closed",
        "expired",
        "superseded",
    ]
    platform: str
    platform_channel_id: str
    global_user_id: str
    source_message_id: str
    prompt_safe_original_goal: str
    prompt_safe_question: str
    prompt_safe_approval_summary: str
    pending_task_continuation: NotRequired[PendingTaskContinuationV1]
    prompt_safe_goal_progress: NotRequired[ResolverGoalProgressV1]
    created_at_utc: str
    expires_at_utc: str


class ResolverPendingContinuationV3(TypedDict):
    """Prompt-safe human-clarification context without durable row identity."""

    schema_version: Literal["resolver_pending_continuation.v3"]
    capability_kind: Literal["human_clarification"]
    status: Literal["waiting_for_user"]
    original_goal: str
    question: str
    pending_task_continuation: PendingTaskContinuationV1


class ResolverPendingDispositionV1(TypedDict):
    """Semantic P-stage disposition for one selected pending continuation."""

    decision: Literal[
        "continue_waiting",
        "answered",
        "rejected",
        "superseded",
    ]
    reason: str


class ResolverPendingResolutionV1(TypedDict):
    """L2d decision describing how to update a pending resolver row."""

    schema_version: Literal["resolver_pending_resolution.v1"]
    resume_id: str
    decision: Literal[
        "continue_waiting",
        "answered",
        "approved",
        "rejected",
        "superseded",
    ]
    reason: str


class ResolverGoalDeliverableV1(TypedDict):
    """One cognition-maintained deliverable inside a resolver goal."""

    description: str
    status: Literal["pending", "partial", "satisfied", "blocked"]
    note: str


class ResolverGoalProgressV1(TypedDict):
    """Cognition-maintained goal checklist carried across resolver cycles."""

    schema_version: Literal["resolver_goal_progress.v1"]
    original_goal: str
    current_focus: str
    deliverables: list[ResolverGoalDeliverableV1]
    missing_user_inputs: list[str]
    evidence_dependencies: list[str]
    attempted_paths: list[str]
    source_backed_facts: list[str]
    assumptions_or_inferences: list[str]
    blockers: list[str]
    final_response_requirements: list[str]


class ResolverCycleStateV1(TypedDict):
    """State accumulated by the deterministic resolver recurrence controller."""

    schema_version: Literal["resolver_cycle_state.v1"]
    cycle_index: int
    max_cycles: int
    status: str
    original_decontextualized_input: str
    observations: list[ResolverObservationV1]
    cycle_traces: list[ResolverCycleTraceV1]
    held_action_specs: list[ActionSpecV1]
    pending_resume: NotRequired[ResolverPendingResumeV3]
    goal_progress: NotRequired[ResolverGoalProgressV1]
    required_resolver_evidence_dependency: NotRequired[
        RequiredResolverEvidenceDependencyV1
    ]
    terminal_reason: str


def validate_shared_memory_prewarm_outcome(
    value: object,
) -> SharedMemoryPrewarmOutcomeV1:
    """Validate and deep-copy one shared-memory prewarm disposition."""

    if not isinstance(value, Mapping):
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome: expected object"
        )
    data = dict(value)
    _require_exact_keys(
        data,
        {
            "schema_version",
            "status",
            "reason_code",
            "attempted",
            "latency_ms",
            "retrieved_shared_count",
            "merged_shared_count",
            "rag_result",
        },
        "shared_memory_prewarm_outcome",
    )
    _require_version(data, SHARED_MEMORY_PREWARM_OUTCOME_VERSION)
    status = _require_enum(
        data,
        "status",
        SHARED_MEMORY_PREWARM_STATUSES,
    )
    reason_code = _require_enum(
        data,
        "reason_code",
        SHARED_MEMORY_PREWARM_REASON_CODES,
    )
    expected_status = SHARED_MEMORY_PREWARM_REASON_STATUS[reason_code]
    if status != expected_status:
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome: status and reason conflict"
        )

    attempted = data["attempted"]
    if not isinstance(attempted, bool):
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.attempted: expected boolean"
        )
    latency_ms = _require_bounded_int(
        data["latency_ms"],
        "shared_memory_prewarm_outcome.latency_ms",
        minimum=0,
        maximum=MAX_SHARED_MEMORY_PREWARM_LATENCY_MS,
    )
    retrieved_shared_count = _require_bounded_int(
        data["retrieved_shared_count"],
        "shared_memory_prewarm_outcome.retrieved_shared_count",
        minimum=0,
        maximum=MAX_SHARED_MEMORY_PREWARM_ENTRIES,
    )
    merged_shared_count = _require_bounded_int(
        data["merged_shared_count"],
        "shared_memory_prewarm_outcome.merged_shared_count",
        minimum=0,
        maximum=MAX_SHARED_MEMORY_PREWARM_ENTRIES,
    )
    normalized_rag_result = _validate_shared_memory_prewarm_rag_result(
        data["rag_result"]
    )
    actual_retrieved_count = len(normalized_rag_result["memory_evidence"])
    if retrieved_shared_count != actual_retrieved_count:
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome: retrieved count is not truthful"
        )

    if status == "skipped":
        if (
            attempted
            or latency_ms
            or retrieved_shared_count
            or merged_shared_count
        ):
            raise ResolverValidationError(
                "skipped prewarm outcome must not be attempted"
            )
        if not _prewarm_rag_result_is_empty(normalized_rag_result):
            raise ResolverValidationError(
                "skipped prewarm outcome must contain empty RAG"
            )
    elif status in {"empty", "failed"}:
        if not attempted or merged_shared_count:
            raise ResolverValidationError(
                "empty or failed prewarm outcome has invalid attempt state"
            )
        if not _prewarm_rag_result_is_empty(normalized_rag_result):
            raise ResolverValidationError(
                "empty or failed prewarm outcome must contain empty RAG"
            )
    elif reason_code == "shared_memory_ready":
        if (
            not attempted
            or retrieved_shared_count <= 0
            or merged_shared_count
        ):
            raise ResolverValidationError(
                "shared_memory_ready outcome has invalid counts"
            )
    elif (
        not attempted
        or retrieved_shared_count <= 0
        or merged_shared_count != retrieved_shared_count
    ):
        raise ResolverValidationError(
            "shared_memory_merged outcome has invalid counts"
        )

    normalized: SharedMemoryPrewarmOutcomeV1 = {
        "schema_version": SHARED_MEMORY_PREWARM_OUTCOME_VERSION,
        "status": status,
        "reason_code": reason_code,
        "attempted": attempted,
        "latency_ms": latency_ms,
        "retrieved_shared_count": retrieved_shared_count,
        "merged_shared_count": merged_shared_count,
        "rag_result": normalized_rag_result,
    }
    return_value = deepcopy(normalized)
    return return_value


def _validate_shared_memory_prewarm_rag_result(
    value: object,
) -> dict[str, object]:
    """Validate the strict prewarm-only RAG shape and project safe values."""

    if not isinstance(value, Mapping):
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.rag_result: expected object"
        )
    data = dict(value)
    if set(data) != SHARED_MEMORY_PREWARM_RAG_FIELDS:
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.rag_result: fields are not exact"
        )
    if not isinstance(data["answer"], str):
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.rag_result.answer: expected string"
        )
    for field_name in (
        "user_image",
        "character_image",
        "supervisor_trace",
    ):
        if not isinstance(data[field_name], Mapping):
            raise ResolverValidationError(
                f"shared_memory_prewarm_outcome.rag_result.{field_name}: "
                "expected object"
            )
    for field_name in (
        "user_memory_unit_candidates",
        "third_party_profiles",
        "memory_evidence",
        "recall_evidence",
        "conversation_evidence",
        "external_evidence",
    ):
        if not isinstance(data[field_name], list):
            raise ResolverValidationError(
                f"shared_memory_prewarm_outcome.rag_result.{field_name}: "
                "expected list"
            )
    for field_name in (
        "user_memory_unit_candidates",
        "third_party_profiles",
        "recall_evidence",
        "conversation_evidence",
        "external_evidence",
    ):
        if data[field_name]:
            raise ResolverValidationError(
                "shared-memory prewarm RAG may contain only memory evidence"
            )
    memory_evidence = data["memory_evidence"]
    if len(memory_evidence) > MAX_SHARED_MEMORY_PREWARM_ENTRIES:
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.rag_result.memory_evidence: "
            "too many entries"
        )
    if any(not isinstance(item, Mapping) for item in memory_evidence):
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.rag_result.memory_evidence: "
            "expected objects"
        )

    normalized = normalize_projected_rag_result(data)
    if set(normalized) != SHARED_MEMORY_PREWARM_RAG_FIELDS:
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.rag_result: projection is incomplete"
        )
    if normalized["answer"] != "":
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.rag_result.answer: expected empty"
        )
    try:
        serialized = json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.rag_result: payload is not "
            "serializable"
        ) from exc
    if len(serialized) > MAX_SHARED_MEMORY_PREWARM_JSON_CHARS:
        raise ResolverValidationError(
            "shared_memory_prewarm_outcome.rag_result: payload is over budget"
        )
    return_value = deepcopy(normalized)
    return return_value


def _prewarm_rag_result_is_empty(value: Mapping[str, object]) -> bool:
    """Return whether a prewarm RAG payload contains no projected evidence."""

    return (
        value["answer"] == ""
        and all(
            not value[field_name]
            for field_name in (
                "user_memory_unit_candidates",
                "third_party_profiles",
                "memory_evidence",
                "recall_evidence",
                "conversation_evidence",
                "external_evidence",
            )
        )
    )


def _require_bounded_int(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Require a non-Boolean integer within one declared contract bound."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ResolverValidationError(
            f"{field_name}: expected integer in {minimum}..{maximum}"
        )
    return_value = value
    return return_value


def validate_resolver_evidence_state(
    value: object,
) -> ResolverEvidenceStateV1:
    """Validate one task-resolution evidence disposition."""

    data = _require_mapping(value, "resolver_evidence_state")
    _require_exact_keys(
        data,
        {"schema_version", "state", "remaining_needs"},
        "resolver_evidence_state",
    )
    _require_version(data, RESOLVER_EVIDENCE_STATE_VERSION)
    state = _require_enum(data, "state", ALLOWED_RESOLVER_EVIDENCE_STATES)
    remaining_needs = _normalize_goal_text_list(data, "remaining_needs")
    return_value: ResolverEvidenceStateV1 = {
        "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
        "state": state,
        "remaining_needs": remaining_needs,
    }
    return return_value


def validate_required_resolver_evidence_dependency(
    value: object,
) -> RequiredResolverEvidenceDependencyV1:
    """Validate one accepted resolver request's answer-evidence dependency."""

    data = _require_mapping(value, "required_resolver_evidence_dependency")
    _require_exact_keys(
        data,
        {
            "schema_version",
            "accepted_request_handle",
            "observation_id",
            "prompt_safe_observation_handle",
            "capability_kind",
            "state",
            "evidence_handles",
            "remaining_needs",
            "goal_continuation_ref",
        },
        "required_resolver_evidence_dependency",
    )
    _require_version(data, REQUIRED_RESOLVER_EVIDENCE_DEPENDENCY_VERSION)
    accepted_request_handle = _require_non_empty_string(
        data,
        "accepted_request_handle",
    )
    observation_id = _require_non_empty_string(data, "observation_id")
    prompt_safe_observation_handle = _require_non_empty_string(
        data,
        "prompt_safe_observation_handle",
    )
    if data.get("capability_kind") != "task_resolution_request":
        raise ResolverValidationError(
            "required_resolver_evidence_dependency: capability_kind is invalid"
        )
    state = _require_enum(data, "state", ALLOWED_RESOLVER_EVIDENCE_STATES)
    evidence_handles = _normalize_string_handle_list(
        data,
        "evidence_handles",
    )
    remaining_needs = _normalize_goal_text_list(data, "remaining_needs")
    continuation_ref = _validate_continuation_ref(
        data.get("goal_continuation_ref")
    )
    if continuation_ref is None:
        raise ResolverValidationError(
            "required_resolver_evidence_dependency: goal_continuation_ref is required"
        )
    if state == "complete" and (not evidence_handles or remaining_needs):
        raise ResolverValidationError(
            "complete dependency requires evidence handles and no remaining needs"
        )
    if state == "partial" and (not evidence_handles or not remaining_needs):
        raise ResolverValidationError(
            "partial dependency requires evidence handles and remaining needs"
        )
    if state == "missing" and evidence_handles:
        raise ResolverValidationError(
            "missing dependency cannot contain evidence handles"
        )
    return_value: RequiredResolverEvidenceDependencyV1 = {
        "schema_version": REQUIRED_RESOLVER_EVIDENCE_DEPENDENCY_VERSION,
        "accepted_request_handle": accepted_request_handle,
        "observation_id": observation_id,
        "prompt_safe_observation_handle": prompt_safe_observation_handle,
        "capability_kind": "task_resolution_request",
        "state": state,
        "evidence_handles": evidence_handles,
        "remaining_needs": remaining_needs,
        "goal_continuation_ref": continuation_ref,
    }
    return return_value


def validate_resolver_capability_request(
    value: object,
) -> ResolverCapabilityRequestV1:
    """Validate a cognition-selected resolver capability request."""

    data = _require_mapping(value, "resolver_capability_request")
    _require_version(data, RESOLVER_CAPABILITY_REQUEST_VERSION)
    capability_kind = _require_enum(
        data,
        "capability_kind",
        ALLOWED_RESOLVER_CAPABILITIES,
    )
    objective = _require_non_empty_string(data, "objective")
    reason = _require_non_empty_string(data, "reason")
    priority = _require_enum(data, "priority", ALLOWED_RESOLVER_PRIORITIES)
    if "goal_continuation_ref" not in data:
        raise ResolverValidationError(
            "resolver_capability_request: goal_continuation_ref is required"
        )
    continuation_ref = _validate_continuation_ref(
        data.get("goal_continuation_ref")
    )
    if capability_kind == "task_resolution_request" and continuation_ref is None:
        raise ResolverValidationError(
            "resolver_capability_request: task resolution requires "
            "goal_continuation_ref"
        )
    return_value = {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": capability_kind,
        "objective": _clip_text(objective, MAX_RESOLVER_OBJECTIVE_CHARS),
        "reason": _clip_text(reason, MAX_RESOLVER_REASON_CHARS),
        "priority": priority,
        "goal_continuation_ref": continuation_ref,
    }
    return return_value


def validate_resolver_observation(value: object) -> ResolverObservationV1:
    """Validate one capability observation before storing or projecting it."""

    data = _require_mapping(value, "resolver_observation")
    _require_version(data, RESOLVER_OBSERVATION_VERSION)
    observation_id = _require_non_empty_string(data, "observation_id")
    capability_kind = _require_enum(
        data,
        "capability_kind",
        ALLOWED_RESOLVER_CAPABILITIES,
    )
    request_objective = _require_non_empty_string(data, "request_objective")
    request_reason = _require_non_empty_string(data, "request_reason")
    status = _require_enum(data, "status", ALLOWED_OBSERVATION_STATUSES)
    prompt_safe_summary = _require_non_empty_string(data, "prompt_safe_summary")
    evidence_refs = _require_list(data, "evidence_refs")
    normalized_evidence_refs = _normalize_evidence_refs(evidence_refs)
    created_at_utc = _require_non_empty_string(data, "created_at_utc")
    continuation_ref = None
    if capability_kind == "task_resolution_request":
        if "goal_continuation_ref" not in data:
            raise ResolverValidationError(
                "goal_continuation_ref: required for task resolution"
            )
        continuation_ref = _validate_continuation_ref(
            data.get("goal_continuation_ref")
        )
        if continuation_ref is None:
            raise ResolverValidationError(
                "goal_continuation_ref: required for task resolution"
            )
    elif "goal_continuation_ref" in data:
        raise ResolverValidationError(
            "goal_continuation_ref: only valid for task resolution"
        )
    task_evidence_state = None
    if capability_kind == "task_resolution_request":
        if "task_resolution_evidence_state" not in data:
            raise ResolverValidationError(
                "task_resolution_evidence_state: required for task resolution"
            )
        task_evidence_state = validate_resolver_evidence_state(
            data["task_resolution_evidence_state"],
        )
        _validate_task_evidence_state_consistency(
            task_evidence_state,
            status=status,
            evidence_refs=normalized_evidence_refs,
        )
    elif "task_resolution_evidence_state" in data:
        raise ResolverValidationError(
            "task_resolution_evidence_state: only valid for task resolution"
        )
    pending_resume_id = data.get("pending_resume_id")
    if pending_resume_id is not None and not isinstance(pending_resume_id, str):
        raise ResolverValidationError("pending_resume_id: expected string")
    blocker_kind = None
    if "blocker_kind" in data:
        blocker_kind = _require_enum(
            data,
            "blocker_kind",
            ALLOWED_OBSERVATION_BLOCKER_KINDS,
        )
        if status != "blocked":
            raise ResolverValidationError(
                "blocker_kind: expected blocked observation status",
            )

    normalized: ResolverObservationV1 = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": observation_id,
        "capability_kind": capability_kind,
        "request_objective": _clip_text(
            request_objective,
            MAX_RESOLVER_OBJECTIVE_CHARS,
        ),
        "request_reason": _clip_text(
            request_reason,
            MAX_RESOLVER_REASON_CHARS,
        ),
        "status": status,
        "prompt_safe_summary": _clip_text(
            prompt_safe_summary,
            MAX_RESOLVER_SUMMARY_CHARS,
        ),
        "evidence_refs": normalized_evidence_refs,
        "created_at_utc": created_at_utc,
    }
    if "rag_result" in data:
        normalized["rag_result"] = normalize_projected_rag_result(
            data["rag_result"]
        )
    if "knowledge_projection" in data:
        normalized["knowledge_projection"] = _normalize_knowledge_projection(
            data["knowledge_projection"],
        )
    if pending_resume_id is not None:
        normalized["pending_resume_id"] = pending_resume_id
    if blocker_kind is not None:
        normalized["blocker_kind"] = blocker_kind
    if task_evidence_state is not None:
        normalized["task_resolution_evidence_state"] = task_evidence_state
    if continuation_ref is not None:
        normalized["goal_continuation_ref"] = continuation_ref
    return_value = normalized
    return return_value


def validate_resolver_cycle_trace(value: object) -> ResolverCycleTraceV1:
    """Validate one prompt-safe resolver cycle trace row."""

    data = _require_mapping(value, "resolver_cycle_trace")
    _require_version(data, RESOLVER_CYCLE_TRACE_VERSION)
    cycle_index = data.get("cycle_index")
    if not isinstance(cycle_index, int) or cycle_index < 0:
        raise ResolverValidationError("cycle_index: expected non-negative integer")
    status_before_cycle = _require_enum(
        data,
        "status_before_cycle",
        ALLOWED_RESOLVER_STATES,
    )
    l1_emotional_appraisal = _require_string(data, "l1_emotional_appraisal")
    l1_interaction_subtext = _require_string(data, "l1_interaction_subtext")
    l2_internal_monologue_summary = _require_string(
        data,
        "l2_internal_monologue_summary",
    )
    l2_logical_stance = _require_string(data, "l2_logical_stance")
    l2_character_intent = _require_string(data, "l2_character_intent")
    l2_judgment_note = _require_string(data, "l2_judgment_note")
    selected_capability_kind = _require_string(data, "selected_capability_kind")
    final_surface_decision = _require_string(data, "final_surface_decision")
    terminal_reason = _require_string(data, "terminal_reason")
    created_at_utc = _require_string(data, "created_at_utc")
    requests = _require_list(data, "l2d_resolver_capability_requests")
    normalized_requests = []
    for request in requests:
        normalized_request = validate_resolver_capability_request(request)
        normalized_requests.append(normalized_request)
    summaries = _require_list(data, "l2d_action_specs_summary")
    normalized_summaries = []
    for summary in summaries:
        if not isinstance(summary, str):
            raise ResolverValidationError("l2d_action_specs_summary: expected strings")
        normalized_summary = _clip_text(summary, MAX_RESOLVER_SUMMARY_CHARS)
        normalized_summaries.append(normalized_summary)
    observation_ids = _require_list(data, "observation_ids")
    normalized_observation_ids = []
    for observation_id in observation_ids:
        if not isinstance(observation_id, str):
            raise ResolverValidationError("observation_ids: expected strings")
        normalized_observation_ids.append(observation_id)
    normalized: ResolverCycleTraceV1 = {
        "schema_version": RESOLVER_CYCLE_TRACE_VERSION,
        "cycle_index": cycle_index,
        "status_before_cycle": status_before_cycle,
        "l1_emotional_appraisal": _clip_text(
            l1_emotional_appraisal,
            MAX_RESOLVER_TRACE_CHARS,
        ),
        "l1_interaction_subtext": _clip_text(
            l1_interaction_subtext,
            MAX_RESOLVER_TRACE_CHARS,
        ),
        "l2_internal_monologue_summary": _clip_text(
            l2_internal_monologue_summary,
            MAX_RESOLVER_TRACE_CHARS,
        ),
        "l2_logical_stance": _clip_text(
            l2_logical_stance,
            MAX_RESOLVER_TRACE_CHARS,
        ),
        "l2_character_intent": _clip_text(
            l2_character_intent,
            MAX_RESOLVER_TRACE_CHARS,
        ),
        "l2_judgment_note": _clip_text(
            l2_judgment_note,
            MAX_RESOLVER_TRACE_CHARS,
        ),
        "l2d_resolver_capability_requests": normalized_requests,
        "l2d_action_specs_summary": normalized_summaries,
        "selected_capability_kind": _clip_text(
            selected_capability_kind,
            MAX_RESOLVER_TRACE_CHARS,
        ),
        "observation_ids": normalized_observation_ids,
        "final_surface_decision": _clip_text(
            final_surface_decision,
            MAX_RESOLVER_TRACE_CHARS,
        ),
        "terminal_reason": _clip_text(terminal_reason, MAX_RESOLVER_TRACE_CHARS),
        "created_at_utc": created_at_utc,
    }
    return_value = normalized
    return return_value


def validate_pending_task_continuation(
    value: object,
) -> PendingTaskContinuationV1:
    """Validate P's closed answer-conditioned task-admission decision."""

    data = _require_mapping(value, "pending_task_continuation")
    _require_exact_keys(
        data,
        {"schema_version", "on_answered_clarification"},
        "pending_task_continuation",
    )
    _require_version(data, PENDING_TASK_CONTINUATION_VERSION)
    continuation_value = _require_enum(
        data,
        "on_answered_clarification",
        ALLOWED_PENDING_TASK_CONTINUATIONS,
    )
    normalized: PendingTaskContinuationV1 = {
        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
        "on_answered_clarification": continuation_value,
    }
    return_value = normalized
    return return_value


def validate_resolver_pending_resume(value: object) -> ResolverPendingResumeV3:
    """Validate one durable pending HIL or approval row."""

    data = _require_mapping(value, "resolver_pending_resume")
    _require_version(data, RESOLVER_PENDING_RESUME_VERSION)
    resume_id = _require_non_empty_string(data, "resume_id")
    capability_kind = _require_enum(
        data,
        "capability_kind",
        ALLOWED_PENDING_CAPABILITIES,
    )
    status = _require_enum(data, "status", ALLOWED_PENDING_STATUSES)
    platform = _require_non_empty_string(data, "platform")
    platform_channel_id = _require_string(data, "platform_channel_id")
    global_user_id = _require_non_empty_string(data, "global_user_id")
    source_message_id = _require_non_empty_string(data, "source_message_id")
    raw_original_goal = data.get("prompt_safe_original_goal")
    if isinstance(raw_original_goal, str):
        original_goal = raw_original_goal
    else:
        original_goal = ""
    question = _require_string(data, "prompt_safe_question")
    approval_summary = _require_string(data, "prompt_safe_approval_summary")
    created_at_utc = _require_non_empty_string(data, "created_at_utc")
    expires_at_utc = _require_non_empty_string(data, "expires_at_utc")
    normalized: ResolverPendingResumeV3 = {
        "schema_version": RESOLVER_PENDING_RESUME_VERSION,
        "resume_id": resume_id,
        "capability_kind": capability_kind,
        "status": status,
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "global_user_id": global_user_id,
        "source_message_id": source_message_id,
        "prompt_safe_original_goal": _clip_text(
            original_goal,
            MAX_RESOLVER_SUMMARY_CHARS,
        ),
        "prompt_safe_question": _clip_text(
            question,
            MAX_RESOLVER_SUMMARY_CHARS,
        ),
        "prompt_safe_approval_summary": _clip_text(
            approval_summary,
            MAX_RESOLVER_SUMMARY_CHARS,
        ),
        "created_at_utc": created_at_utc,
        "expires_at_utc": expires_at_utc,
    }
    raw_goal_progress = data.get("prompt_safe_goal_progress")
    if raw_goal_progress is not None:
        normalized["prompt_safe_goal_progress"] = validate_resolver_goal_progress(
            raw_goal_progress,
        )
    raw_pending_task_continuation = data.get("pending_task_continuation")
    if capability_kind == "human_clarification":
        if raw_pending_task_continuation is None:
            raise ResolverValidationError(
                "pending_task_continuation is required for human clarification"
            )
        normalized["pending_task_continuation"] = (
            validate_pending_task_continuation(
                raw_pending_task_continuation,
            )
        )
    elif raw_pending_task_continuation is not None:
        raise ResolverValidationError(
            "pending_task_continuation is limited to human clarification"
        )
    return_value = normalized
    return return_value


def validate_resolver_pending_continuation(
    value: object,
) -> ResolverPendingContinuationV3:
    """Validate the model-facing human-clarification projection."""

    data = _require_mapping(value, "pending_resolver_continuation")
    expected_keys = {
        "schema_version",
        "capability_kind",
        "status",
        "original_goal",
        "question",
        "pending_task_continuation",
    }
    if set(data) != expected_keys:
        raise ResolverValidationError(
            "pending_resolver_continuation: fields are not exact"
        )
    capability_kind = _require_enum(
        data,
        "capability_kind",
        {"human_clarification"},
    )
    _require_version(data, RESOLVER_PENDING_CONTINUATION_VERSION)
    status = _require_enum(
        data,
        "status",
        {"waiting_for_user"},
    )
    original_goal = _require_non_empty_string(data, "original_goal")
    question = _require_string(data, "question")
    pending_task_continuation = validate_pending_task_continuation(
        data["pending_task_continuation"],
    )
    normalized: ResolverPendingContinuationV3 = {
        "schema_version": RESOLVER_PENDING_CONTINUATION_VERSION,
        "capability_kind": capability_kind,
        "status": status,
        "original_goal": _clip_text(
            original_goal,
            MAX_RESOLVER_SUMMARY_CHARS,
        ),
        "question": _clip_text(question, MAX_RESOLVER_SUMMARY_CHARS),
        "pending_task_continuation": pending_task_continuation,
    }
    return_value = normalized
    return return_value


def validate_resolver_pending_disposition(
    value: object,
) -> ResolverPendingDispositionV1:
    """Validate the model-facing pending disposition without a row id."""

    data = _require_mapping(value, "pending_resolution")
    _require_exact_keys(data, {"decision", "reason"}, "pending_resolution")
    decision = _require_enum(
        data,
        "decision",
        ALLOWED_PENDING_DISPOSITIONS,
    )
    reason = _require_non_empty_string(data, "reason")
    normalized: ResolverPendingDispositionV1 = {
        "decision": decision,
        "reason": _clip_text(reason, MAX_RESOLVER_REASON_CHARS),
    }
    return_value = normalized
    return return_value


def validate_resolver_pending_resolution(
    value: object,
) -> ResolverPendingResolutionV1:
    """Validate L2d's structural decision for one pending resolver row."""

    data = _require_mapping(value, "resolver_pending_resolution")
    _require_version(data, RESOLVER_PENDING_RESOLUTION_VERSION)
    resume_id = _require_non_empty_string(data, "resume_id")
    decision = _require_enum(data, "decision", ALLOWED_PENDING_DECISIONS)
    reason = _require_non_empty_string(data, "reason")
    normalized: ResolverPendingResolutionV1 = {
        "schema_version": RESOLVER_PENDING_RESOLUTION_VERSION,
        "resume_id": resume_id,
        "decision": decision,
        "reason": _clip_text(reason, MAX_RESOLVER_REASON_CHARS),
    }
    return_value = normalized
    return return_value


def new_empty_goal_progress(*, original_goal: str) -> ResolverGoalProgressV1:
    """Build the empty goal-progress shell before L2d adds semantics."""

    if not isinstance(original_goal, str) or not original_goal.strip():
        raise ResolverValidationError("original_goal: expected non-empty string")
    progress = {
        "schema_version": RESOLVER_GOAL_PROGRESS_VERSION,
        "original_goal": _clip_text(
            original_goal.strip(),
            MAX_RESOLVER_GOAL_FIELD_CHARS,
        ),
        "current_focus": "",
        "deliverables": [],
        "missing_user_inputs": [],
        "evidence_dependencies": [],
        "attempted_paths": [],
        "source_backed_facts": [],
        "assumptions_or_inferences": [],
        "blockers": [],
        "final_response_requirements": [],
    }
    return_value = validate_resolver_goal_progress(progress)
    return return_value


def validate_resolver_goal_progress(value: object) -> ResolverGoalProgressV1:
    """Validate L2d's goal-progress checklist before storing or projecting it."""

    data = _require_mapping(value, "resolver_goal_progress")
    _require_version(data, RESOLVER_GOAL_PROGRESS_VERSION)
    original_goal = _require_non_empty_string(data, "original_goal")
    current_focus = _require_string(data, "current_focus")
    deliverables = _normalize_goal_deliverables(
        _require_list(data, "deliverables"),
    )
    normalized: ResolverGoalProgressV1 = {
        "schema_version": RESOLVER_GOAL_PROGRESS_VERSION,
        "original_goal": _clip_text(
            original_goal,
            MAX_RESOLVER_GOAL_FIELD_CHARS,
        ),
        "current_focus": _clip_text(
            current_focus,
            MAX_RESOLVER_GOAL_FIELD_CHARS,
        ),
        "deliverables": deliverables,
        "missing_user_inputs": _normalize_goal_text_list(
            data,
            "missing_user_inputs",
        ),
        "evidence_dependencies": _normalize_goal_text_list(
            data,
            "evidence_dependencies",
        ),
        "attempted_paths": _normalize_goal_text_list(data, "attempted_paths"),
        "source_backed_facts": _normalize_goal_text_list(
            data,
            "source_backed_facts",
        ),
        "assumptions_or_inferences": _normalize_goal_text_list(
            data,
            "assumptions_or_inferences",
        ),
        "blockers": _normalize_goal_text_list(data, "blockers"),
        "final_response_requirements": _normalize_goal_text_list(
            data,
            "final_response_requirements",
        ),
    }
    return_value = normalized
    return return_value


def project_observations_for_cognition(
    observations: list[ResolverObservationV1],
) -> str:
    """Project resolver observations without raw tool or evidence identifiers."""

    lines: list[str] = []
    for index, observation in enumerate(observations, start=1):
        validated = validate_resolver_observation(observation)
        alias = f"resolver_obs_{index}"
        capability_kind = validated["capability_kind"]
        status = validated["status"]
        summary = validated["prompt_safe_summary"]
        blocker_kind = validated.get("blocker_kind")
        task_evidence_state = validated.get("task_resolution_evidence_state")
        evidence_excerpts = resolver_evidence_excerpts_for_cognition(
            validated,
        )
        knowledge_context = _project_knowledge_projection(validated)
        if knowledge_context:
            line_parts = [
                f"{alias}: capability={capability_kind}",
                (
                    "objective="
                    f"{_prompt_safe_projection_text(validated['request_objective'])}"
                ),
                f"summary={_prompt_safe_projection_text(summary)}",
                knowledge_context,
            ]
            _append_task_evidence_projection(
                line_parts,
                task_evidence_state=task_evidence_state,
                evidence_excerpts=evidence_excerpts,
            )
            if blocker_kind is not None:
                line_parts.append(f"blocker_kind={blocker_kind}")
            line = "; ".join(line_parts)
            lines.append(line)
            continue
        line_parts = [
            f"{alias}: capability={capability_kind}",
            f"status={status}",
            (
                "objective="
                f"{_prompt_safe_projection_text(validated['request_objective'])}"
            ),
            f"summary={_prompt_safe_projection_text(summary)}",
        ]
        if blocker_kind is not None:
            line_parts.append(f"blocker_kind={blocker_kind}")
        _append_task_evidence_projection(
            line_parts,
            task_evidence_state=task_evidence_state,
            evidence_excerpts=evidence_excerpts,
        )
        rag_summary = _project_rag_result_summary(validated)
        if rag_summary:
            line_parts.append(
                f"rag_answer={_prompt_safe_projection_text(rag_summary)}"
            )
        line = "; ".join(line_parts)
        lines.append(line)
    projection = "\n".join(lines)
    return_value = projection
    return return_value


def resolver_evidence_excerpts_for_cognition(
    observation: ResolverObservationV1,
) -> list[str]:
    """Derive bounded source-owned excerpts in their stored evidence order."""

    excerpts: list[str] = []
    for evidence_ref in observation["evidence_refs"]:
        excerpt = evidence_ref.get("excerpt")
        if not isinstance(excerpt, str) or not excerpt.strip():
            continue
        excerpts.append(_clip_text(excerpt.strip(), MAX_RESOLVER_EVIDENCE_EXCERPT_CHARS))
        if len(excerpts) >= MAX_RESOLVER_EVIDENCE_EXCERPTS:
            break
    return_value = excerpts
    return return_value


def _append_task_evidence_projection(
    line_parts: list[str],
    *,
    task_evidence_state: ResolverEvidenceStateV1 | None,
    evidence_excerpts: list[str],
) -> None:
    """Append typed task evidence state without exposing source identifiers."""

    if task_evidence_state is None:
        return
    line_parts.append(f"evidence_state={task_evidence_state['state']}")
    if evidence_excerpts:
        rendered_excerpts = " | ".join(
            _prompt_safe_projection_text(excerpt)
            for excerpt in evidence_excerpts
        )
        line_parts.append(f"evidence_excerpts={rendered_excerpts}")
    remaining_needs = task_evidence_state["remaining_needs"]
    if remaining_needs:
        rendered_needs = " | ".join(
            _prompt_safe_projection_text(item)
            for item in remaining_needs
        )
        line_parts.append(f"remaining_needs={rendered_needs}")


def _validate_task_evidence_state_consistency(
    evidence_state: ResolverEvidenceStateV1,
    *,
    status: str,
    evidence_refs: list[EvidenceRefV1],
) -> None:
    """Keep task evidence state aligned with the observation disposition."""

    evidence_available = any(
        isinstance(evidence_ref.get("excerpt"), str)
        and bool(evidence_ref["excerpt"].strip())
        for evidence_ref in evidence_refs
    )
    state = evidence_state["state"]
    remaining_needs = evidence_state["remaining_needs"]
    if state == "complete":
        if status != "succeeded" or not evidence_available or remaining_needs:
            raise ResolverValidationError(
                "complete task evidence requires succeeded status, evidence, "
                "and no remaining needs"
            )
        return
    if state == "partial":
        if status != "succeeded" or not evidence_available or not remaining_needs:
            raise ResolverValidationError(
                "partial task evidence requires succeeded status, evidence, "
                "and remaining needs"
            )
        return
    if state == "pending":
        if status != "succeeded":
            raise ResolverValidationError(
                "pending task evidence requires succeeded observation status"
            )
        return
    if state == "missing":
        if status != "succeeded" or evidence_available:
            raise ResolverValidationError(
                "missing task evidence requires succeeded status without evidence"
            )
        return
    if status not in {"blocked", "failed"}:
        raise ResolverValidationError(
            "blocked task evidence requires blocked or failed observation status"
        )


def _normalize_knowledge_projection(value: object) -> dict[str, object]:
    """Validate semantic knowledge returned by an evidence capability."""

    data = _require_mapping(value, "knowledge_projection")
    normalized = {
        "investigation_summary": _clip_text(
            _optional_string(data, "investigation_summary"),
            MAX_RESOLVER_SUMMARY_CHARS,
        ),
        "knowledge_we_know_so_far": _normalize_knowledge_list(
            data,
            "knowledge_we_know_so_far",
        ),
        "knowledge_still_lacking": _normalize_knowledge_list(
            data,
            "knowledge_still_lacking",
        ),
        "recommended_next_iteration": _normalize_knowledge_list(
            data,
            "recommended_next_iteration",
        ),
        "evidence_boundary_notes": _normalize_knowledge_list(
            data,
            "evidence_boundary_notes",
        ),
    }
    return_value = normalized
    return return_value


def _normalize_knowledge_list(data: dict, field_name: str) -> list[str]:
    """Return bounded semantic knowledge rows from a projection field."""

    raw_items = data.get(field_name, [])
    if not isinstance(raw_items, list):
        raise ResolverValidationError(f"{field_name}: expected list")
    normalized_items: list[str] = []
    for raw_item in raw_items[:MAX_RESOLVER_KNOWLEDGE_ITEMS]:
        if not isinstance(raw_item, str):
            raise ResolverValidationError(f"{field_name}: expected strings")
        item = raw_item.strip()
        if not item:
            continue
        normalized_items.append(_clip_text(item, MAX_RESOLVER_GOAL_ITEM_CHARS))
    return_value = normalized_items
    return return_value


def _project_knowledge_projection(observation: ResolverObservationV1) -> str:
    """Render semantic knowledge sections for the next cognition pass."""

    projection = observation.get("knowledge_projection")
    if not isinstance(projection, dict):
        return_value = ""
        return return_value
    lines: list[str] = []
    summary = projection["investigation_summary"]
    if isinstance(summary, str) and summary:
        lines.append(
            "investigation_summary="
            f"{_prompt_safe_projection_text(summary)}"
        )
    for field_name, label in (
        ("knowledge_we_know_so_far", "knowledge_we_know_so_far"),
        ("knowledge_still_lacking", "knowledge_still_lacking"),
        ("recommended_next_iteration", "recommended_next_iteration"),
        ("evidence_boundary_notes", "evidence_boundary_notes"),
    ):
        raw_items = projection[field_name]
        if not isinstance(raw_items, list) or not raw_items:
            continue
        items = [
            _prompt_safe_projection_text(str(item))
            for item in raw_items
        ]
        lines.append(f"{label}: " + "；".join(items))
    rendered_projection = "; ".join(lines)
    return_value = rendered_projection
    return return_value


def project_goal_progress_for_cognition(
    goal_progress: ResolverGoalProgressV1 | None,
) -> str:
    """Project the cognition-maintained goal checklist into compact text."""

    if goal_progress is None:
        return_value = ""
        return return_value
    validated = validate_resolver_goal_progress(goal_progress)
    lines = [
        (
            "resolver_goal_progress: "
            f"original_goal={validated['original_goal']}; "
            f"current_focus={validated['current_focus']}"
        ),
    ]
    if validated["deliverables"]:
        lines.append("deliverables:")
        for index, deliverable in enumerate(validated["deliverables"], start=1):
            lines.append(
                f"{index}. status={deliverable['status']}; "
                f"description={deliverable['description']}; "
                f"note={deliverable['note']}"
            )
    for field_name, label in (
        ("missing_user_inputs", "missing_user_inputs"),
        ("evidence_dependencies", "evidence_dependencies"),
        ("attempted_paths", "attempted_paths"),
        ("source_backed_facts", "source_backed_facts"),
        ("assumptions_or_inferences", "assumptions_or_inferences"),
        ("blockers", "blockers"),
        ("final_response_requirements", "final_response_requirements"),
    ):
        items = validated[field_name]
        if items:
            lines.append(f"{label}: " + "；".join(items))
    projection = "\n".join(lines)
    return_value = projection
    return return_value


def project_pending_resume_for_cognition(
    pending: ResolverPendingResumeV3 | None,
) -> str:
    """Project legacy resolver pending context without durable identifiers."""

    if pending is None:
        return_value = ""
        return return_value

    validated = validate_resolver_pending_resume(pending)
    if validated["status"] not in {
        "waiting_for_user",
        "waiting_for_approval",
    }:
        return_value = ""
        return return_value
    pending_context = (
        "pending_resolver_resume: "
        f"capability={validated['capability_kind']}; "
        f"status={validated['status']}; "
        "original_goal="
        f"{validated['prompt_safe_original_goal']}; "
        "question="
        f"{validated['prompt_safe_question']}; "
        "approval_summary="
        f"{validated['prompt_safe_approval_summary']}"
    )
    lines = [pending_context]
    projection = "\n".join(lines)
    return_value = projection
    return return_value


def project_pending_resume_for_prompt(
    pending: ResolverPendingResumeV3 | None,
) -> ResolverPendingContinuationV3 | None:
    """Project one open human clarification without durable identifiers."""

    if pending is None:
        return_value = None
        return return_value
    validated = validate_resolver_pending_resume(pending)
    if (
        validated["capability_kind"] != "human_clarification"
        or validated["status"] != "waiting_for_user"
    ):
        return_value = None
        return return_value
    projected: ResolverPendingContinuationV3 = {
        "schema_version": RESOLVER_PENDING_CONTINUATION_VERSION,
        "capability_kind": validated["capability_kind"],
        "status": validated["status"],
        "original_goal": validated["prompt_safe_original_goal"],
        "question": validated["prompt_safe_question"],
        "pending_task_continuation": validated["pending_task_continuation"],
    }
    return_value = validate_resolver_pending_continuation(projected)
    return return_value


def _validate_continuation_ref(
    value: object,
) -> GoalContinuationRefV1 | None:
    """Validate one nullable canonical goal-continuation reference."""

    if value is None:
        return None
    try:
        validated_ref = validate_goal_continuation_ref(value)
    except CognitiveEpisodeValidationError as exc:
        raise ResolverValidationError(
            f"goal_continuation_ref: invalid reference: {exc}"
        ) from exc
    return validated_ref


def _require_mapping(value: object, label: str) -> dict:
    """Return a dictionary payload or raise a contract error."""

    if not isinstance(value, dict):
        raise ResolverValidationError(f"{label}: expected object")
    return_value = value
    return return_value


def _require_exact_keys(
    data: dict,
    expected_keys: set[str],
    label: str,
) -> None:
    """Require one nested contract mapping to contain exactly its keys."""

    if set(data) != expected_keys:
        raise ResolverValidationError(f"{label}: fields are not exact")


def _normalize_string_handle_list(
    data: dict,
    field_name: str,
) -> list[str]:
    """Normalize a bounded list of non-empty prompt-safe handles."""

    raw_items = _require_list(data, field_name)
    normalized: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, str) or not raw_item.strip():
            raise ResolverValidationError(f"{field_name}: expected strings")
        normalized.append(_clip_text(raw_item.strip(), MAX_RESOLVER_GOAL_ITEM_CHARS))
        if len(normalized) >= MAX_RESOLVER_KNOWLEDGE_ITEMS:
            break
    return_value = normalized
    return return_value


def _require_version(data: dict, expected: str) -> None:
    """Require a specific schema version on one resolver contract object."""

    actual = data.get("schema_version")
    if actual != expected:
        raise ResolverValidationError(f"schema_version: expected {expected}")


def _require_non_empty_string(data: dict, field_name: str) -> str:
    """Require one non-empty string field."""

    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ResolverValidationError(f"{field_name}: expected non-empty string")
    return_value = value
    return return_value


def _optional_string(data: dict, field_name: str) -> str:
    """Read an optional string field from external structured output."""

    value = data.get(field_name, "")
    if not isinstance(value, str):
        raise ResolverValidationError(f"{field_name}: expected string")
    return_value = value
    return return_value


def _require_string(data: dict, field_name: str) -> str:
    """Require one string field, allowing an empty string."""

    value = data.get(field_name)
    if not isinstance(value, str):
        raise ResolverValidationError(f"{field_name}: expected string")
    return_value = value
    return return_value


def _require_enum(data: dict, field_name: str, allowed: frozenset[str]) -> str:
    """Require one string field to belong to an allowed vocabulary."""

    value = data.get(field_name)
    if not isinstance(value, str) or value not in allowed:
        expected = sorted(allowed)
        raise ResolverValidationError(f"{field_name}: expected one of {expected}")
    return_value = value
    return return_value


def _require_list(data: dict, field_name: str) -> list:
    """Require one list field."""

    value = data.get(field_name)
    if not isinstance(value, list):
        raise ResolverValidationError(f"{field_name}: expected list")
    return_value = value
    return return_value


def _normalize_goal_deliverables(
    deliverables: list,
) -> list[ResolverGoalDeliverableV1]:
    """Normalize nested deliverable rows from L2d's semantic checklist."""

    normalized: list[ResolverGoalDeliverableV1] = []
    for raw_deliverable in deliverables:
        if not isinstance(raw_deliverable, dict):
            raise ResolverValidationError("deliverables: expected objects")
        description = _require_non_empty_string(
            raw_deliverable,
            "description",
        )
        status = _require_enum(
            raw_deliverable,
            "status",
            ALLOWED_GOAL_DELIVERABLE_STATUSES,
        )
        note = _require_string(raw_deliverable, "note")
        normalized.append({
            "description": _clip_text(
                description,
                MAX_RESOLVER_GOAL_ITEM_CHARS,
            ),
            "status": status,
            "note": _clip_text(note, MAX_RESOLVER_GOAL_ITEM_CHARS),
        })
        if len(normalized) >= MAX_RESOLVER_GOAL_ITEMS:
            break
    return_value = normalized
    return return_value


def _normalize_goal_text_list(data: dict, field_name: str) -> list[str]:
    """Normalize a bounded list of prompt-safe goal-progress strings."""

    raw_items = _require_list(data, field_name)
    normalized: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, str):
            raise ResolverValidationError(f"{field_name}: expected strings")
        item = raw_item.strip()
        if not item:
            continue
        normalized.append(_clip_text(item, MAX_RESOLVER_GOAL_ITEM_CHARS))
        if len(normalized) >= MAX_RESOLVER_GOAL_ITEMS:
            break
    return_value = normalized
    return return_value


def _prompt_safe_projection_text(value: str) -> str:
    """Redact raw-looking local identifiers before prompt projection."""

    redacted = _RAW_MARKER_RE.sub("<redacted>", value)
    return_value = redacted
    return return_value


def _clip_text(value: str, max_chars: int) -> str:
    """Return text clipped to a prompt-safe character budget."""

    if len(value) <= max_chars:
        return_value = value
        return return_value
    clipped = value[:max_chars]
    return_value = clipped
    return return_value


def _normalize_evidence_refs(evidence_refs: list) -> list[EvidenceRefV1]:
    """Validate evidence refs and strip fields outside the public contract."""

    from kazusa_ai_chatbot.action_spec.models import (
        ActionValidationError,
        validate_evidence_ref,
    )

    normalized_refs: list[EvidenceRefV1] = []
    for evidence_ref in evidence_refs:
        try:
            validated = validate_evidence_ref(evidence_ref)
        except ActionValidationError as exc:
            raise ResolverValidationError(f"evidence_refs: {exc}") from exc
        normalized_ref: EvidenceRefV1 = {
            "schema_version": validated["schema_version"],
            "evidence_kind": validated["evidence_kind"],
            "evidence_id": validated["evidence_id"],
            "owner": validated["owner"],
            "excerpt": validated["excerpt"],
            "observed_at": validated["observed_at"],
        }
        normalized_refs.append(normalized_ref)
    return_value = normalized_refs
    return return_value


def normalize_projected_rag_result(value: object) -> dict:
    """Keep the prompt-safe projected RAG payload for later cognition."""

    if not isinstance(value, Mapping):
        raise ResolverValidationError("rag_result: expected object")
    value = dict(value)
    normalized: dict[str, object] = {}
    answer = value.get("answer")
    if isinstance(answer, str) and answer.strip():
        normalized["answer"] = _clip_text(answer, MAX_RESOLVER_SUMMARY_CHARS)
    else:
        normalized["answer"] = ""

    for field_name in (
        "user_image",
        "character_image",
        "supervisor_trace",
    ):
        field_value = value.get(field_name)
        if isinstance(field_value, Mapping):
            normalized[field_name] = _normalize_rag_mapping(field_value)

    for field_name in (
        "user_memory_unit_candidates",
        "third_party_profiles",
        "memory_evidence",
        "recall_evidence",
        "conversation_evidence",
        "external_evidence",
    ):
        field_value = value.get(field_name)
        if isinstance(field_value, list):
            normalized[field_name] = _normalize_rag_list(field_value)

    return_value = normalized
    return return_value


def _normalize_rag_mapping(value: Mapping) -> dict:
    """Recursively copy prompt-safe RAG mapping values."""

    normalized: dict[str, object] = {}
    for field_name, field_value in value.items():
        if field_name in {"raw_id", "raw_payload", "raw_result"}:
            continue
        if isinstance(field_value, str):
            normalized[field_name] = _clip_text(
                field_value,
                MAX_RESOLVER_SUMMARY_CHARS,
            )
            continue
        if isinstance(field_value, Mapping):
            normalized[field_name] = _normalize_rag_mapping(field_value)
            continue
        if isinstance(field_value, list):
            normalized[field_name] = _normalize_rag_list(field_value)
            continue
        if field_value is None or isinstance(field_value, bool | int | float):
            normalized[field_name] = field_value
    return_value = normalized
    return return_value


def _normalize_rag_list(value: list) -> list[object]:
    """Recursively copy prompt-safe RAG list values."""

    normalized: list[object] = []
    for item in value:
        if isinstance(item, str):
            normalized.append(_clip_text(item, MAX_RESOLVER_SUMMARY_CHARS))
            continue
        if isinstance(item, Mapping):
            normalized.append(_normalize_rag_mapping(item))
            continue
        if isinstance(item, list):
            normalized.append(_normalize_rag_list(item))
            continue
        if item is None or isinstance(item, bool | int | float):
            normalized.append(item)
    return_value = normalized
    return return_value


def _project_rag_result_summary(observation: ResolverObservationV1) -> str:
    """Project bounded RAG answer and evidence summaries for cognition."""

    rag_result = observation.get("rag_result")
    if not isinstance(rag_result, dict):
        return_value = ""
        return return_value
    summary_segments: list[str] = []
    answer = rag_result.get("answer")
    if isinstance(answer, str) and answer.strip():
        summary_segments.append(
            "answer="
            + _clip_text(answer, MAX_RESOLVER_SUMMARY_CHARS)
        )
    external_summaries = _project_rag_evidence_summaries(
        rag_result.get("external_evidence"),
    )
    if external_summaries:
        summary_segments.append(
            "external_evidence="
            + " | ".join(external_summaries)
        )
    if summary_segments:
        return_value = "; ".join(summary_segments)
        return return_value
    memory_evidence = rag_result.get("memory_evidence")
    if not isinstance(memory_evidence, list):
        return_value = ""
        return return_value
    projected_facts = []
    for memory_item in memory_evidence:
        if isinstance(memory_item, dict):
            fact_summary = memory_item.get("summary")
        else:
            fact_summary = memory_item
        if not isinstance(fact_summary, str) or not fact_summary.strip():
            continue
        clipped_fact = _clip_text(fact_summary, MAX_RESOLVER_SUMMARY_CHARS)
        projected_facts.append(clipped_fact)
    return_value = "; ".join(projected_facts)
    return return_value


def _project_rag_evidence_summaries(value: object) -> list[str]:
    """Return bounded prompt-safe summaries from RAG evidence rows."""

    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value

    projected_summaries: list[str] = []
    for evidence in value[:MAX_RESOLVER_RAG_EVIDENCE_ITEMS]:
        if not isinstance(evidence, dict):
            continue
        summary = evidence.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            continue
        clipped_summary = _clip_text(
            summary,
            MAX_RESOLVER_RAG_EVIDENCE_SUMMARY_CHARS,
        )
        projected_summaries.append(clipped_summary)

    return_value = projected_summaries
    return return_value
