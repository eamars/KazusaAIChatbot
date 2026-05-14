"""Typed contracts and constants for global character growth."""

from __future__ import annotations

from typing import Literal, TypedDict


EVALUATION_MODE = "global_character_growth_v1"
PROMPT_VERSION = "global_character_growth_candidate_v1"
RUN_KIND = "global_character_growth"

MAX_MEMORY_CARDS = 80
MAX_CARD_CONTENT_CHARS = 420
MAX_CARD_CONFIDENCE_NOTE_CHARS = 120
MAX_CARD_DATES = 8
MAX_CARD_REFLECTION_RUN_IDS = 8
MAX_CURRENT_TRAITS = 12
MAX_CURRENT_TRAIT_GUIDANCE_CHARS = 220
MAX_ACCEPTED_CANDIDATES = 4
MAX_SOURCE_CARDS_PER_CANDIDATE = 8
MAX_GUIDANCE_CHARS = 240
MAX_TRAIT_NAME_CHARS = 80
SHADOW_PROJECTION_LIMIT = 5
RUNTIME_CONTEXT_LIMIT = 3

PREVIOUS_STRENGTH_WEIGHT = 0.85
EVIDENCE_STRENGTH_WEIGHT = 0.15
MAX_DAILY_STRENGTH_DELTA = 0.18
OBSERVED_STRENGTH_CEILING = 0.25
EMERGING_STRENGTH_CEILING = 0.50
STABILIZING_STRENGTH_CEILING = 0.75
DUPLICATE_OVERLAP_THRESHOLD = 0.65

FULL_EVIDENCE_STRENGTH = 0.94
EMERGING_EVIDENCE_STRENGTH = 0.55
LOW_EVIDENCE_STRENGTH = 0.25

ALLOWED_GROWTH_AXES = (
    "boundary_timing",
    "guarded_care",
    "playful_challenge",
    "recovery_style",
    "clarity",
    "emotional_exposure",
    "trust_calibration",
    "other_communication",
)

GrowthAxis = Literal[
    "boundary_timing",
    "guarded_care",
    "playful_challenge",
    "recovery_style",
    "clarity",
    "emotional_exposure",
    "trust_calibration",
    "other_communication",
]
TraitStatus = Literal["active", "superseded", "rejected"]
MaturityBand = Literal["observed", "emerging", "stabilizing", "promoted"]
RunStatus = Literal["dry_run", "applied", "skipped", "failed"]
PromptBudgetStatus = Literal[
    "within_budget",
    "trimmed_to_budget",
    "empty_after_budget",
]


class MemoryCard(TypedDict):
    """Prompt-safe projection of one reflection-promoted memory row."""

    source_card_id: str
    memory_unit_id: str
    memory_name: str
    memory_type: str
    content: str
    character_local_dates: list[str]
    source_reflection_run_ids: list[str]
    confidence_note: str


class CurrentTraitSummary(TypedDict):
    """Compact current trait summary used by candidate generation."""

    trait_id: str
    growth_axis: str
    guidance: str
    maturity_band: str


class CandidateLimits(TypedDict):
    """Candidate-generation hard caps exposed to the LLM."""

    max_candidates: int
    max_source_cards_per_candidate: int


class CandidatePromptPayload(TypedDict):
    """JSON payload consumed by the candidate-generation prompt."""

    evaluation_mode: str
    prompt_version: str
    memory_cards: list[MemoryCard]
    current_global_growth_traits: list[CurrentTraitSummary]
    candidate_limits: CandidateLimits
    allowed_growth_axes: list[str]


class InputQualityDiagnostics(TypedDict):
    """Auditable input-density diagnostics for run documents."""

    raw_memory_rows: int
    eligible_memory_cards: int
    unique_source_dates: int
    source_date_span_days: int
    promotion_density: str
    dropped_rows: dict[str, int]
    quality_notes: list[str]


class PromptBudgetDiagnostics(TypedDict):
    """Auditable final prompt-size diagnostics for run documents."""

    prompt_char_budget: int
    rendered_prompt_chars_before_budget: int
    rendered_prompt_chars_after_budget: int
    memory_cards_before_prompt_budget: int
    memory_cards_after_prompt_budget: int
    dropped_memory_cards_for_prompt_budget: int
    prompt_budget_status: PromptBudgetStatus


class AcceptedCandidate(TypedDict, total=False):
    """Validated candidate ready for stable drift planning."""

    candidate_id: str
    growth_axis: str
    trait_name: str
    guidance: str
    source_card_ids: list[str]
    supporting_dates: list[str]
    source_memory_unit_ids: list[str]
    source_reflection_run_ids: list[str]
    support_level: str
    confidence: str
    evidence_strength: float
    novelty_reason: str
    stability_reason: str


class RejectedCandidate(TypedDict, total=False):
    """Rejected candidate with deterministic reason for audit."""

    growth_axis: str
    trait_name: str
    guidance: str
    reason: str
    source_card_ids: list[str]


class ValidatedCandidateSet(TypedDict):
    """Validation output split into accepted and rejected rows."""

    accepted_candidates: list[AcceptedCandidate]
    rejected_candidates: list[RejectedCandidate]
    validation_warnings: list[str]


class TraitUpdate(TypedDict, total=False):
    """Planned mutation for one trait row."""

    action: Literal["insert", "update"]
    trait: dict


class GlobalCharacterGrowthContext(TypedDict, total=False):
    """Prompt-safe runtime context for promoted global growth."""

    promoted_global_growth: list[dict]
    retrieval_notes: list[str]


class GlobalCharacterGrowthRunResult(TypedDict, total=False):
    """Public runner summary returned to worker and CLI callers."""

    run_id: str
    run_kind: str
    status: str
    dry_run: bool
    eligible_memory_cards: int
    accepted_candidate_count: int
    rejected_candidate_count: int
    trait_update_count: int
    promoted_trait_count: int
    shadow_projection_count: int
    input_quality_density: str
    dropped_memory_cards_for_prompt_budget: int
    rendered_prompt_chars_after_budget: int
    warning_count: int
