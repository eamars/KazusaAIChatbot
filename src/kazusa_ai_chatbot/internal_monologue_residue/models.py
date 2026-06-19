"""Typed contracts for internal monologue residue runtime state."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

ResidueScopeKind = Literal[
    "character_global",
    "group_scene",
    "user_thread",
]
ResidueSourceKind = Literal["chat", "self_cognition"]


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


class InternalMonologueResidueSourceRef(TypedDict, total=False):
    """Sanitized source reference stored with a residue row."""

    ref_kind: str
    ref_id: str


class InternalMonologueResidueRow(TypedDict, total=False):
    """Stored private residue row without MongoDB storage internals."""

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


class ResidueLoadResult(TypedDict):
    """Prompt-facing residue load result for cognition callers."""

    internal_monologue_residue_context: str
    selected_count: int
    candidate_count: int
    scope_order: list[ResidueScopeKind]
    status: str


class ResidueRecordResult(TypedDict):
    """Sanitized post-episode recorder outcome."""

    status: str
    source_kind: str
    scope_kind: str
    written: bool
    retry_count: int
    validation_errors: list[str]
    residue_id: NotRequired[str]


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


class RecorderValidationResult(TypedDict):
    """Structural validation outcome for one recorder candidate."""

    accepted: bool
    status: str
    failure_reason: str
