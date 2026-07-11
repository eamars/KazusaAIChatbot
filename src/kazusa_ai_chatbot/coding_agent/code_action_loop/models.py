"""Typed contracts shared by the coding action-loop controller and executor."""

from typing import Literal, TypedDict


ActionName = Literal["read", "search", "edit", "run", "note", "finish", "block"]


class CodingAction(TypedDict, total=False):
    """One strict controller request for a deterministic action executor."""

    schema_version: Literal["coding_action.v1"]
    action_id: str
    action: ActionName
    reason: str
    args: dict[str, object]
    working_note: str


class CodingObservation(TypedDict, total=False):
    """Prompt-safe record of one deterministic action outcome."""

    schema_version: Literal["coding_observation.v1"]
    sequence: int
    action_sequence: int
    outcome: Literal["ok", "rejected", "failed", "unavailable", "stale"]
    kind: str
    summary: str
    evidence: list[dict[str, object]]
    candidate_revision: int
    index_snapshot_id: str
    overlay_revision: int
    budget_remaining: dict[str, int]
    created_at: str


class ActionDispatchRecord(TypedDict):
    """Durable identity written before one validated action effect."""

    sequence: int
    raw_output_hash: str
    parsed_action: CodingAction
    operation_id: str
    validation: Literal["ok"]


class CurrentFailure(TypedDict):
    """Current unresolved failure bound to candidate and effect identity."""

    kind: str
    summary: str
    candidate_revision: int
    effect_id: str
    execution_index: int | None


class ApprovalBinding(TypedDict):
    """Internal approval identity for one exact reviewed candidate."""

    schema_version: Literal["coding_action_loop_approval_binding.v1"]
    proposal_digest: str
    candidate_revision: int
    candidate_tree_digest: str
    approval_evidence_digest: str
    source_message_id: str


class PendingVerificationEffect(TypedDict, total=False):
    """Durable apply and verification transition owned by one approval."""

    schema_version: Literal["coding_action_loop_effect.v1"]
    effect_id: str
    state: Literal[
        "applying",
        "verifying",
        "completed",
        "failed",
        "rejected",
    ]
    proposal_digest: str
    candidate_revision: int
    candidate_tree_digest: str
    approval: dict[str, object]
    approval_binding: ApprovalBinding
    execution_specs: list[dict[str, object]]
    execution_spec_digest: str
    apply_package_id: str
    apply_result: dict[str, object]
    next_execution_index: int
    active_execution_index: int
    execution_results: list[dict[str, object]]
