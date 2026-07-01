"""TypedDict document schemas for every MongoDB collection.

Each TypedDict mirrors exactly one document shape and is referenced by
function signatures across the ``db.*`` submodules. Schemas use
``total=False`` so optional fields don't trip type checkers.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from kazusa_ai_chatbot.message_envelope.types import (
    ConversationAuthorRole,
    MentionEntityKind,
)


class UserMemoryUnitType:
    """String constants for ``user_memory_units.unit_type``."""

    STABLE_PATTERN = "stable_pattern"
    RECENT_SHIFT = "recent_shift"
    OBJECTIVE_FACT = "objective_fact"
    MILESTONE = "milestone"
    ACTIVE_COMMITMENT = "active_commitment"


class UserMemoryUnitStatus:
    """String constants for ``user_memory_units.status``."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AttachmentDoc(TypedDict, total=False):
    """Multimedia attachment embedded in a conversation message."""
    media_type: str       # MIME type: "image/png", "audio/ogg", etc.
    url: str              # External URL (CDN, S3, etc.) — preferred for large files
    base64_data: str      # Inline base64 — optional, config-gated
    description: str      # Alt-text / transcription / OCR summary
    size_bytes: int       # File size
    storage_shape: str    # "inline" | "url_only" | "drop"


class ReplyAttachmentSummaryDoc(TypedDict, total=False):
    """Prompt-safe attachment summary copied from a replied-to message."""
    media_kind: str
    description: str
    summary_status: Literal["available", "unavailable"]


class ReplyContextDoc(TypedDict, total=False):
    """Structured reply metadata for a conversation message."""
    reply_to_message_id: str
    reply_to_platform_user_id: str
    reply_to_display_name: str
    reply_excerpt: str
    reply_attachments: list[ReplyAttachmentSummaryDoc]


class MentionDoc(TypedDict, total=False):
    """Structured mention metadata for a conversation message."""
    platform_user_id: str
    global_user_id: str
    display_name: str
    entity_kind: MentionEntityKind
    raw_text: str


class ConversationMessageDoc(TypedDict, total=False):
    """A single chat message in the ``conversation_history`` collection.

    Indexed by ``(platform, platform_channel_id, timestamp)`` (descending)
    for efficient retrieval of the most recent messages in a channel.
    """

    platform: str              # "discord" | "qq" | "wechat" | "whatsapp" | "telegram" | "system"
    platform_channel_id: str   # Original channel/group ID from the platform
    channel_type: str          # "group" | "private" | "system"
    channel_name: str          # Optional sanitized human-readable group label
    role: ConversationAuthorRole  # "user" | "assistant"
    platform_message_id: str   # Original platform message ID when available
    platform_user_id: str      # Original user/bot ID from the platform
    global_user_id: str        # Our internal UUID key
    display_name: str          # Display name at time of message
    body_text: str             # Content-only text without platform wire markers
    raw_wire_text: str         # Original on-the-wire text for audit/replay
    content_type: str          # "text" | "image" | "voice" | "mixed"
    addressed_to_global_user_ids: list[str]  # Typed addressees for user/assistant rows
    mentions: list[MentionDoc]  # Typed mentions extracted by adapter normalizers
    broadcast: bool            # True only for assistant-authored channel replies
    attachments: list[AttachmentDoc]  # Images, voice, files
    reply_context: ReplyContextDoc     # Structured reply-to metadata when available
    delivery_tracking_id: str  # Brain-generated id for adapter delivery receipts
    logical_message_index: int  # Zero-based logical message index in one response
    delivery_status: str       # "pending" | "delivered"
    delivered_at: str          # ISO timestamp reported by the adapter
    delivery_adapter: str      # Adapter that reported the delivery receipt
    llm_trace_id: str          # Turn-scoped LLM trace id, when available
    timestamp: str             # ISO-8601 UTC timestamp
    embedding: list[float]     # Dense vector (on text content only)


class ConversationEpisodeEntryDoc(TypedDict, total=False):
    """One short-term episode entry with first-seen metadata."""

    text: str
    first_seen_at: str


class ConversationEpisodeStateDoc(TypedDict, total=False):
    """Short-lived operational progress state for one user/channel episode."""

    episode_state_id: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    status: str
    episode_label: str
    continuity: str
    conversation_mode: str
    episode_phase: str
    topic_momentum: str
    current_thread: str
    user_goal: str
    current_blocker: str
    user_state_updates: list[ConversationEpisodeEntryDoc]
    assistant_moves: list[str]
    overused_moves: list[str]
    open_loops: list[ConversationEpisodeEntryDoc]
    resolved_threads: list[ConversationEpisodeEntryDoc]
    avoid_reopening: list[ConversationEpisodeEntryDoc]
    emotional_trajectory: str
    next_affordances: list[str]
    progression_guidance: str
    turn_count: int
    last_user_input: str
    created_at: str
    updated_at: str
    expires_at: str


class InternalMonologueResidueSourceRefDoc(TypedDict, total=False):
    """Sanitized source identifier for an internal residue row."""

    ref_kind: str
    ref_id: str


class InternalMonologueResidueDoc(TypedDict, total=False):
    """Compact private residue row in ``internal_monologue_residue_state``."""

    residue_id: str
    character_id: str
    scope_key: str
    scope_kind: str
    platform: str
    platform_channel_id: str
    channel_type: str
    global_user_id: str
    residue_text: str
    source_kind: str
    source_refs: list[InternalMonologueResidueSourceRefDoc]
    created_at: str


class PlatformAccountDoc(TypedDict, total=False):
    """A linked platform account within a UserProfileDoc."""
    platform: str             # "discord" | "qq" | ...
    platform_user_id: str     # Original ID on that platform
    display_name: str         # Last known display name
    linked_at: str            # ISO-8601 when this account was linked


class BoundaryProfileDoc(TypedDict, total=False):
    """Character's psychological boundary parameters.
    
    Controls how the character handles relationships, control, and emotional vulnerability.
    """
    self_integrity: float              # 0.0–1.0: how firmly character maintains their sense of self
    control_sensitivity: float         # 0.0–1.0: how strongly character notices/reacts to control
    compliance_strategy: str           # "resist" | "evade" | "comply"
    relational_override: float         # 0.0–1.0: how much relationship importance overrides boundaries
    control_intimacy_misread: float    # 0.0–1.0: risk of mistaking control for affection
    boundary_recovery: str             # "rebound" | "delayed_rebound" | "decay" | "detach"
    authority_skepticism: float        # 0.0–1.0: distrust of authority & power structures


class LinguisticTextureProfileDoc(TypedDict, total=False):
    """Character's linguistic and speech pattern parameters.
    
    Controls how the character sounds: verbal patterns, hesitations, assertiveness, emotional presence.
    All parameters are floats from 0.0–1.0.
    """
    fragmentation: float               # 0.0–1.0: choppy vs fluent speech
    hesitation_density: float          # 0.0–1.0: filler words and pauses
    counter_questioning: float         # 0.0–1.0: responds with questions back
    softener_density: float            # 0.0–1.0: hedging language like "maybe", "I think"
    formalism_avoidance: float         # 0.0–1.0: casual vs polite language
    abstraction_reframing: float       # 0.0–1.0: intellectualizing vs concrete speech
    direct_assertion: float            # 0.0–1.0: confident statements vs hedging
    emotional_leakage: float           # 0.0–1.0: emotion visible in speech
    rhythmic_bounce: float             # 0.0–1.0: playful vs flat cadence
    self_deprecation: float            # 0.0–1.0: self-critical humor and language


class UserProfileDoc(TypedDict, total=False):
    """Long-term memory about a single user in the ``user_profiles`` collection.

    Keyed by ``global_user_id`` (UUID4). Cognition-facing user memory lives in
    ``user_memory_units`` and is projected by the RAG layer.
    """

    global_user_id: str                          # UUID4 — our internal unique key
    platform_accounts: list[PlatformAccountDoc]  # All linked accounts
    suspected_aliases: list[str]                 # Other global_user_ids suspected to be same person

    # ── Relationship metrics ───────────────────────────────────
    affinity: int                                # 0–1000 affinity score (default 500)
    last_relationship_insight: str               # Character's instantaneous impression of the user


class UserMemoryUnitSourceRef(TypedDict, total=False):
    """Source evidence reference attached to a user memory unit."""

    source: str
    timestamp: str
    message_id: str


class UserMemoryUnitMergeHistoryEntry(TypedDict, total=False):
    """One merge/evolve event in a user memory unit's lifecycle."""

    timestamp: str
    decision: str
    candidate_id: str
    reason: str


class UserMemoryUnitDoc(TypedDict, total=False):
    """A durable fact-anchored user memory unit.

    Documents live in ``user_memory_units`` and replace prompt-facing
    historical summary, recent-window, and character-diary user memory.
    """

    unit_id: str
    global_user_id: str
    unit_type: str
    fact: str
    subjective_appraisal: str
    relationship_signal: str
    status: str
    count: int
    first_seen_at: str
    last_seen_at: str
    updated_at: str
    source_refs: list[UserMemoryUnitSourceRef]
    embedding: list[float]
    merge_history: list[UserMemoryUnitMergeHistoryEntry]
    due_at: str | None
    completed_at: str | None
    cancelled_at: str | None
    archived_at: str | None


class UserMemoryContextEntry(TypedDict, total=False):
    """Prompt-facing projection of one user memory unit."""

    fact: str
    subjective_appraisal: str
    relationship_signal: str
    updated_at: str
    due_at: str
    due_state: str


class UserMemoryContextDoc(TypedDict, total=False):
    """Prompt-facing user memory context consumed by cognition."""

    stable_patterns: list[UserMemoryContextEntry]
    recent_shifts: list[UserMemoryContextEntry]
    objective_facts: list[UserMemoryContextEntry]
    milestones: list[UserMemoryContextEntry]
    active_commitments: list[UserMemoryContextEntry]


class InteractionStyleScopeType:
    """String constants for ``interaction_style_images.scope_type``."""

    USER = "user"
    GROUP_CHANNEL = "group_channel"


class InteractionStyleStatus:
    """String constants for ``interaction_style_images.status``."""

    ACTIVE = "active"
    EMPTY = "empty"
    DISABLED = "disabled"


class InteractionStyleOverlayDoc(TypedDict, total=False):
    """Prompt-facing abstract interaction guidance for L3 style stages."""

    speech_guidelines: list[str]
    social_guidelines: list[str]
    pacing_guidelines: list[str]
    engagement_guidelines: list[str]
    confidence: str


class InteractionStyleImageDoc(TypedDict, total=False):
    """Durable current interaction-style image for a user or group channel."""

    style_image_id: str
    scope_type: str
    global_user_id: str
    platform: str
    platform_channel_id: str
    status: str
    overlay: InteractionStyleOverlayDoc
    source_reflection_run_ids: list[str]
    revision: int
    created_at: str
    updated_at: str


class CharacterProfileDoc(TypedDict, total=False):
    """All fields of the singleton ``_id: "global"`` document in
    the ``character_state`` collection.

    Both personality profile fields **and** runtime state fields live
    at the top level. The schema is intentionally open-ended
    (``total=False``) so new fields can be added without migration.
    """

    # ── personality profile ────────────────────────────────────────
    name: str
    description: str
    gender: str
    age: int
    birthday: str
    tone: str
    speech_patterns: str
    backstory: str
    personality_brief: dict
    boundary_profile: BoundaryProfileDoc
    linguistic_texture_profile: LinguisticTextureProfileDoc

    # ── runtime state ─────────────────────────────────────────────
    mood: str               # e.g. "melancholic", "playful", "irritated"
    global_vibe: str        # See Cognition Layer
    reflection_summary: str # See Cognition Layer
    updated_at: str         # ISO-8601 UTC timestamp of last update

    # ── Three-tier character self-image (NEW) ─────────────────
    self_image: dict        # {milestones, recent_window, historical_summary, meta}


class MemoryDoc(TypedDict, total=False):
    """Evolving shared-memory unit in the ``memory`` collection."""
    memory_unit_id: str             # Stable id for this memory unit
    lineage_id: str                 # Stable lineage id across superseding versions
    version: int                    # Monotonic version within a lineage
    memory_name: str                # Name of the memory
    content: str                    # memory content
    source_global_user_id: str      # UUID4 of the user who triggered this memory (empty for non-user-specific)
    timestamp: str                  # ISO-8601 UTC timestamp of when memory was created/updated
    updated_at: str                 # ISO-8601 UTC timestamp of last lifecycle update
    embedding: list[float]          # dense vector for similarity search

    # --- Structured metadata ---
    memory_type: str                # "fact" | "promise" | "impression" | "narrative" | "defense_rule"
    source_kind: str                # "conversation_extracted" | "relationship_inferred" | "reflection_inferred" | "seeded_manual" | "external_imported"
    authority: str                  # "seed" | "reflection_promoted" | "manual"
    confidence_note: str            # free-form note on how downstream should treat this memory
    status: str                     # "active" | "fulfilled" | "expired" | "superseded" | "rejected"
    expiry_timestamp: str | None    # ISO-8601 or None (never expires)
    supersedes_memory_unit_ids: list[str]
    merged_from_memory_unit_ids: list[str]
    evidence_refs: list[dict]
    privacy_review: dict


class ReflectionMessageRefDoc(TypedDict, total=False):
    """Persistence-only source-message reference for reflection runs."""

    conversation_history_id: str
    platform: str
    platform_channel_id: str
    channel_type: str
    role: Literal["user", "assistant"]
    timestamp: str


class ReflectionScopeDoc(TypedDict):
    """Raw monitored-scope metadata stored on reflection run documents."""

    scope_ref: str
    platform: str
    platform_channel_id: str
    channel_type: str


class CharacterReflectionRunDoc(TypedDict, total=False):
    """A production reflection-run audit document.

    Documents live in ``character_reflection_runs`` and use ``run_id`` as both
    the MongoDB ``_id`` and readable logical id.
    """

    _id: str
    run_id: str
    run_kind: Literal[
        "hourly_slot",
        "daily_channel",
        "daily_global_promotion",
    ]
    status: Literal["succeeded", "failed", "skipped", "dry_run"]
    prompt_version: str
    attempt_count: int
    scope: ReflectionScopeDoc
    window_start: str
    window_end: str
    hour_start: str
    hour_end: str
    character_local_date: str
    source_message_refs: list[ReflectionMessageRefDoc]
    source_reflection_run_ids: list[str]
    output: dict
    promotion_decisions: list[dict]
    validation_warnings: list[str]
    error: str
    created_at: str
    updated_at: str


class GlobalCharacterGrowthTraitDoc(TypedDict, total=False):
    """Durable global character-growth trait row."""

    _id: str
    trait_id: str
    lineage_id: str
    status: Literal["active", "superseded", "rejected"]
    growth_axis: str
    trait_name: str
    guidance: str
    strength: float
    maturity_band: Literal["observed", "emerging", "stabilizing", "promoted"]
    first_observed_date: str
    last_observed_date: str
    supporting_dates: list[str]
    source_memory_unit_ids: list[str]
    source_reflection_run_ids: list[str]
    source_candidate_ids: list[str]
    evidence_count: int
    version: int
    supersedes_trait_ids: list[str]
    merged_from_trait_ids: list[str]
    created_at: str
    updated_at: str


class GlobalCharacterGrowthRunDoc(TypedDict, total=False):
    """Audit document for one global character-growth run."""

    _id: str
    run_id: str
    run_kind: Literal["global_character_growth"]
    status: Literal["dry_run", "applied", "skipped", "failed"]
    dry_run: bool
    prompt_version: str
    created_at: str
    updated_at: str
    character_local_date: str
    input_counts: dict
    input_quality: dict
    source_memory_unit_ids: list[str]
    source_reflection_run_ids: list[str]
    accepted_candidates: list[dict]
    rejected_candidates: list[dict]
    trait_updates: list[dict]
    shadow_projection: list[dict]
    validation_warnings: list[str]
    raw_llm_output: str
    summary: str
    error: str


class RAGCache2PersistentEntryDoc(TypedDict, total=False):
    """A durable backing row for selected Cache2 entries.

    Rows are keyed by the stable cache key in ``_id``. The initial allowlisted
    cache is only ``rag2_initializer``.
    """

    _id: str
    cache_name: str
    version_key: str
    result: dict
    metadata: dict
    created_at: str
    updated_at: str
    hit_count: int


def build_memory_doc(
    memory_name: str,
    content: str,
    source_global_user_id: str,
    memory_type: str,
    source_kind: str,
    confidence_note: str,
    status: str = "active",
    expiry_timestamp: str | None = None,
) -> dict:
    """Build a memory document dict ready for ``save_memory``.

    Single place to construct a well-formed memory payload so every caller
    produces consistent documents.
    """
    return_value = {
        "memory_name": memory_name,
        "content": content,
        "source_global_user_id": source_global_user_id,
        "memory_type": memory_type,
        "source_kind": source_kind,
        "confidence_note": confidence_note,
        "status": status,
        "expiry_timestamp": expiry_timestamp,
    }
    return return_value


class ScheduledEventDoc(TypedDict, total=False):
    """Historical ``scheduled_events`` document retained for migration audit."""

    event_id: str
    tool: str
    args: dict
    execute_at: str
    created_at: str
    status: str
    cancelled_at: str
    source_platform: str
    source_channel_id: str
    source_channel_type: str
    source_user_id: str
    source_message_id: str
    source_platform_bot_id: str
    source_character_name: str
    guild_id: str | None
    bot_role: str


class CalendarScheduleDoc(TypedDict, total=False):
    """Durable schedule definition in ``calendar_schedules``."""

    schema_version: str
    owner: str
    schedule_id: str
    trigger_kind: str
    status: str
    start_at: str
    next_run_at: str
    recurrence: dict
    payload: dict
    source_scope: dict
    idempotency_key: str
    timezone: str
    legacy_source: dict | None
    created_at: str
    updated_at: str
    cancelled_at: str
    cancel_reason: str


class CalendarRunDoc(TypedDict, total=False):
    """Durable due-run document in ``calendar_runs``."""

    schema_version: str
    owner: str
    run_id: str
    schedule_id: str
    trigger_kind: str
    status: str
    due_at: str
    payload: dict
    source_scope: dict
    idempotency_key: str
    attempt_count: int
    max_attempts: int
    claimed_at: str | None
    completed_at: str | None
    failed_at: str | None
    skipped_at: str | None
    lease_owner: str | None
    lease_expires_at: str | None
    period_start_utc: str | None
    slot_index: int | None
    offset_seconds: int | None
    result_summary: dict | None
    failure_summary: dict | None
    legacy_source: dict | None
    created_at: str
    updated_at: str
    skip_reason: str


class SelfCognitionActionAttemptDoc(TypedDict, total=False):
    """Durable action-attempt state for idle self-cognition deduplication."""

    attempt_id: str
    run_id: str
    trigger_id: str
    source_kind: str
    source_id: str
    target_scope: dict
    action_kind: str
    due_at: str | None
    idempotency_key: str
    status: str
    dispatch_status: str
    scheduled_event_ids: list[str]
    recorded_at: str
    action_spec_schema_version: str
    cognition_mode: str | None
    validation_status: str
    handler_owner: str | None
    continuation_status: str
    execution_result: dict | None
    errors: list[str]


class SelfCognitionGroupReviewWindowDoc(TypedDict, total=False):
    """Terminal reviewed-window ledger row for group self-cognition review."""

    source_id: str
    case_id: str | None
    scope_ref: str
    platform: str
    platform_channel_id: str
    channel_type: Literal["group"]
    window_start: str
    window_end: str
    status: Literal[
        "reviewed",
        "target_binding_failed",
        "review_failed",
        "coalesced_skipped",
        "stale_skipped",
    ]
    reviewed_at: str
    selected_route: str | None
    dispatch_status: str | None
    skip_reason: str | None
