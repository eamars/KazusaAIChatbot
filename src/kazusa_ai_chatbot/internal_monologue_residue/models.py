"""Typed contracts for internal monologue residue runtime state."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, NotRequired, TypedDict

ResidueScopeKind = Literal[
    "group_scene",
    "user_thread",
]
ResidueSourceKind = Literal["chat", "self_cognition"]
ResidueDisposition = Literal[
    "append",
    "replace_scope",
    "clear_scope",
]
ResidueDispositionResult = ResidueDisposition | Literal["none"]
RESIDUE_DISPOSITION_VALUES = frozenset({
    "append",
    "replace_scope",
    "clear_scope",
})


class ResidueTriggerScope(TypedDict):
    """Current trigger scope used to select prompt-facing residue."""

    character_id: str
    platform: str
    platform_channel_id: str
    channel_type: str
    global_user_id: str


class ResidueScopeCandidate(TypedDict):
    """One deterministic storage scope eligible for a trigger."""

    scope_kind: ResidueScopeKind
    scope_key: str
    rank: int


class InternalMonologueResidueSourceRef(TypedDict):
    """Sanitized source reference stored with a residue row."""

    ref_kind: str
    ref_id: str


class InternalMonologueResidueRow(TypedDict):
    """Canonical stored private residue row."""

    residue_id: str
    character_id: str
    scope_key: str
    scope_kind: ResidueScopeKind
    platform: str
    platform_channel_id: str
    channel_type: str
    global_user_id: str
    residue_text: str
    source_kind: ResidueSourceKind
    source_refs: list[InternalMonologueResidueSourceRef]
    created_at: str
    schema_version: Literal["internal_monologue_residue.v2"]
    operation_id: str
    disposition: ResidueDisposition
    purge_at: datetime


class ResidueLoadResult(TypedDict):
    """Prompt-facing residue load result for cognition callers."""

    internal_monologue_residue_context: str
    selected_count: int
    candidate_count: int
    scope_order: list[ResidueScopeKind]
    status: str
    barrier_disposition: ResidueDispositionResult


class ResidueRecordResult(TypedDict):
    """Sanitized post-episode recorder outcome."""

    status: str
    source_kind: str
    scope_kind: str
    written: bool
    retry_count: int
    validation_errors: list[str]
    disposition: ResidueDispositionResult
    operation_id: str
    idempotency_result: Literal[
        "not_attempted",
        "written",
        "duplicate_same_payload",
        "conflict",
    ]
    residue_id: NotRequired[str]


class ResidueWriteResult(TypedDict):
    """Idempotent database disposition for one v2 residue operation."""

    status: Literal[
        "written",
        "duplicate_same_payload",
        "conflict",
    ]
    residue_id: str


class RecorderInput(TypedDict):
    """Minimal model-facing input for the residue recorder."""

    character_name: str
    ambient_condition: str
    source_kind: ResidueSourceKind
    internal_monologue: str
    current_speaker_display_name: str
    exact_name_candidates: list[str]
    ambient_evidence_summary: str
    incoming_residue_context: str
    source_reliability_notes: list[str]
    visible_outcome_summary: str
    surface_content_plan: str
    visible_boundaries: list[str]


class RecorderValidationResult(TypedDict):
    """Structural validation outcome for one recorder candidate."""

    accepted: bool
    status: str
    failure_reason: str
    disposition: NotRequired[ResidueDisposition]
