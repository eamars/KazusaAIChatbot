"""Public type contracts and semantic labels for event logging."""

from __future__ import annotations

from typing import Literal, TypedDict

EventSeverity = Literal["debug", "info", "warning", "error", "critical"]

EVENT_SEVERITIES: frozenset[str] = frozenset(
    {"debug", "info", "warning", "error", "critical"}
)

ContinuityBoundary = Literal[
    "progress_load",
    "progress_record",
    "residue_load",
    "residue_record",
    "reflection_projection",
    "post_turn",
]
ContinuityBoundaryStatus = Literal[
    "started",
    "succeeded",
    "empty",
    "skipped",
    "contract_failed",
    "provider_failed",
    "persistence_failed",
    "interrupted",
    "guarded_write_lost",
    "cache_not_published",
    "reconciled",
    "unknown",
]
ContinuityScopeKind = Literal[
    "user_thread",
    "group_scene",
    "private",
    "targetless",
]
ContinuityAgeLabel = Literal["unknown", "fresh", "recent", "stale"]
ContinuityRecorderDisposition = Literal[
    "unknown",
    "not_called",
    "append",
    "replace_scope",
    "clear_scope",
]
ContinuityWriteDisposition = Literal[
    "unknown",
    "not_attempted",
    "written",
    "duplicate_same_payload",
    "conflict",
    "write_failed",
    "lost_guarded_write",
    "interrupted",
    "reconciled_written",
    "reconciled_absent",
]
ContinuityCacheDisposition = Literal[
    "unknown",
    "not_attempted",
    "cache_hit",
    "published",
    "invalidated",
    "not_published",
]
ContinuityBarrierDisposition = Literal[
    "unknown",
    "none",
    "append",
    "replace_scope",
    "clear_scope",
]

CONTINUITY_BOUNDARY_VALUES = frozenset({
    "progress_load",
    "progress_record",
    "residue_load",
    "residue_record",
    "reflection_projection",
    "post_turn",
})
CONTINUITY_BOUNDARY_STATUS_VALUES = frozenset({
    "started",
    "succeeded",
    "empty",
    "skipped",
    "contract_failed",
    "provider_failed",
    "persistence_failed",
    "interrupted",
    "guarded_write_lost",
    "cache_not_published",
    "reconciled",
    "unknown",
})
CONTINUITY_SCOPE_KIND_VALUES = frozenset({
    "user_thread",
    "group_scene",
    "private",
    "targetless",
})
CONTINUITY_AGE_LABEL_VALUES = frozenset({
    "unknown",
    "fresh",
    "recent",
    "stale",
})
CONTINUITY_RECORDER_DISPOSITION_VALUES = frozenset({
    "unknown",
    "not_called",
    "append",
    "replace_scope",
    "clear_scope",
})
CONTINUITY_WRITE_DISPOSITION_VALUES = frozenset({
    "unknown",
    "not_attempted",
    "written",
    "duplicate_same_payload",
    "conflict",
    "write_failed",
    "lost_guarded_write",
    "interrupted",
    "reconciled_written",
    "reconciled_absent",
})
CONTINUITY_CACHE_DISPOSITION_VALUES = frozenset({
    "unknown",
    "not_attempted",
    "cache_hit",
    "published",
    "invalidated",
    "not_published",
})
CONTINUITY_BARRIER_DISPOSITION_VALUES = frozenset({
    "unknown",
    "none",
    "append",
    "replace_scope",
    "clear_scope",
})


class ContinuityBoundaryEventFields(TypedDict):
    """Text-free bounded metrics for one continuity boundary."""

    boundary: ContinuityBoundary
    status: ContinuityBoundaryStatus
    scope_kind: ContinuityScopeKind
    candidate_count: int
    selected_count: int
    packet_turn_count: int
    protected_anchor_count: int
    rendered_chars: int
    packet_age: ContinuityAgeLabel
    source_age: ContinuityAgeLabel
    recorder_disposition: ContinuityRecorderDisposition
    write_disposition: ContinuityWriteDisposition
    cache_disposition: ContinuityCacheDisposition
    barrier_disposition: ContinuityBarrierDisposition


class EventScopeInput(TypedDict, total=False):
    """Caller-supplied runtime scope before private channel ref projection."""

    platform: str
    platform_channel_id: str
    channel_type: str


class SelfCognitionBudget(TypedDict):
    """LLM-call budget counters for one self-cognition case."""

    rag_calls: int
    cognition_calls: int
    dialog_calls: int
    topic_limit: int


SelfCognitionSemanticDisposition = Literal[
    "cognition_declined",
    "reply_proposed",
    "cognition_contract_failed",
]
SelfCognitionPolicyDisposition = Literal[
    "not_evaluated",
    "approved",
    "rejected",
]
SelfCognitionExecutionDisposition = Literal[
    "not_requested",
    "dialog_failed",
    "dispatch_failed",
    "delivered",
]
SelfCognitionPolicyReason = Literal[
    "",
    "stale_source",
    "invalid_provenance",
    "unresolved_target",
    "permission_denied",
    "duplicate",
    "cooldown",
    "policy_risk",
]

SELF_COGNITION_SEMANTIC_DISPOSITION_VALUES = frozenset({
    "cognition_declined",
    "reply_proposed",
    "cognition_contract_failed",
})
SELF_COGNITION_POLICY_DISPOSITION_VALUES = frozenset({
    "not_evaluated",
    "approved",
    "rejected",
})
SELF_COGNITION_EXECUTION_DISPOSITION_VALUES = frozenset({
    "not_requested",
    "dialog_failed",
    "dispatch_failed",
    "delivered",
})
SELF_COGNITION_POLICY_REASON_VALUES = frozenset({
    "",
    "stale_source",
    "invalid_provenance",
    "unresolved_target",
    "permission_denied",
    "duplicate",
    "cooldown",
    "policy_risk",
})
SELF_COGNITION_RESPONSE_GATE_CODE_VALUES = frozenset({
    "response_contract",
    "group_source_provenance",
    "recent_source",
    "participation_grounding",
    "bound_group_target",
    "duplicate_reservation",
    "approved_for_dialog",
    "no_admitted_bid",
    "semantic_declined",
    "dialog_failed",
    "dispatch_failed",
})
SELF_COGNITION_RESPONSE_GATE_CODE_LIMIT = 8


class SelfCognitionResponseTelemetry(TypedDict):
    """Sanitized self-cognition outcome fields mirrored into event logs."""

    semantic_disposition: SelfCognitionSemanticDisposition
    policy_disposition: SelfCognitionPolicyDisposition
    execution_disposition: SelfCognitionExecutionDisposition
    policy_reason: SelfCognitionPolicyReason
    response_gate_codes: list[str]


class EventRefRecord(TypedDict):
    """Reference to an existing durable runtime artifact."""

    ref_type: str
    ref_id: str


class EventLogWriteResult(TypedDict):
    """Best-effort event write result returned to runtime callers."""

    accepted: bool
    event_id: str
    status: Literal["recorded", "rejected", "failed"]
    reason: str


class CognitionV2EventFields(TypedDict):
    """Bounded cognition diagnostics with no raw state or private bids."""

    cognition_component: str
    selected_branch_id: str
    state_scope: Literal["", "user", "character"]
    state_commit_status: Literal[
        "not_started",
        "committed",
        "failed",
        "skipped",
    ]
    stage_status: Literal["started", "completed", "failed", "skipped"]


class CognitionChainEventFields(TypedDict):
    """Bounded Cognition V3 chain-run diagnostics without raw evidence."""

    run_id: str
    cognition_invocation_id: str
    terminal_disposition: str
    chain_model_name: str
    sidecar_model_name: str
    step_count: int
    repair_count: int
    cold_start_count: int
    prompt_chars_total: int
    new_suffix_chars_total: int
    prefix_share_ratio: float
    max_estimated_prompt_tokens: int
    max_reserved_completion_tokens: int
    max_estimated_total_context_tokens: int
    active_total_ceiling_tokens: int
    extension_available: bool
    extension_used: bool
    reanchor_used: bool
    session_disposition: str
    duration_ms: int
    deadline_ms: int
    deadline_consumption_ratio: float
    l1_stream_count: int
    json_repair_call_count: int
    action_auth_attempt_count: int
    resolver_auth_attempt_count: int
    sidecar_queue_wait_ms_total: int
    sidecar_max_in_flight: int
    l1_preempted_by_repair: bool
    sidecar_cancellation_count: int
    warning_codes: list[str]


class CognitionV2SnapshotSummary(TypedDict):
    """Bounded aggregate consumed by later diagnostics."""

    event_count: int
    component_counts: dict[str, int]
    branch_counts: dict[str, int]
    commit_status_counts: dict[str, int]
    failed_stage_count: int


def reflection_health_label(*, failed_count: int, succeeded_count: int) -> str:
    """Return a compact reflection health label for operator summaries.

    Args:
        failed_count: Number of failed reflection events in the window.
        succeeded_count: Number of successful reflection events in the window.

    Returns:
        A stable health label suitable for prompt-safe aggregate snapshots.
    """

    if failed_count == 0 and succeeded_count > 0:
        label = "healthy"
    elif failed_count > 0 and succeeded_count > 0:
        label = "mixed"
    elif failed_count > 0:
        label = "failing"
    else:
        label = "inactive"
    return label


def self_cognition_liveness_label(*, run_count: int, dispatch_count: int) -> str:
    """Return a compact liveness label for self-cognition activity.

    Args:
        run_count: Number of self-cognition events in the window.
        dispatch_count: Number of accepted dispatch results in the window.

    Returns:
        A stable liveness label for aggregate operator payloads.
    """

    if run_count == 0:
        label = "inactive"
    elif dispatch_count > 0:
        label = "active_with_handoff"
    else:
        label = "active_internal_only"
    return label


def llm_parse_stability_label(*, failed_count: int, repaired_count: int) -> str:
    """Return a compact parse-stability label for model-stage telemetry.

    Args:
        failed_count: Number of failed parse or contract events.
        repaired_count: Number of events that needed deterministic repair.

    Returns:
        A stable stability label for aggregate status and snapshots.
    """

    if failed_count > 0:
        label = "degraded"
    elif repaired_count > 0:
        label = "watch"
    else:
        label = "stable"
    return label


def worker_error_level_label(*, error_count: int) -> str:
    """Return a compact worker error level for a summary window.

    Args:
        error_count: Number of runtime-error events for worker components.

    Returns:
        A stable severity label for aggregate status and snapshots.
    """

    if error_count == 0:
        label = "none"
    elif error_count < 3:
        label = "low"
    elif error_count < 10:
        label = "elevated"
    else:
        label = "high"
    return label
