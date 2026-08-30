"""TypedDict document schemas for every MongoDB collection.

Each TypedDict mirrors exactly one document shape and is referenced by
function signatures across the ``db.*`` submodules. Optional fields are
declared explicitly only on document contracts that actually permit them.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, TypedDict

from kazusa_ai_chatbot.character_identity_growth.models import (
    CharacterEffectiveIdentityV1,
)
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
    source_episode_id: str     # Settled cognitive episode root for reflection
    timestamp: str             # ISO-8601 UTC timestamp
    received_at: str           # Server-generated UTC arrival instant for inbound user rows
    embedding: list[float]     # Dense vector (on text content only)


class ConversationProgressSourceRefDoc(TypedDict):
    """Source-lineage alias stored inside one progress event."""

    ref_kind: Literal['conversation_row', 'llm_trace']
    ref_id: str
    occurred_at: str


class ConversationProgressEventDoc(TypedDict):
    """Exact stored semantic event snapshot."""

    event_id: str
    semantic_summary: str
    is_obligation: bool
    actor: str
    action: str
    object: str
    beneficiary: str
    precondition: str
    state: Literal[
        'open',
        'in_progress',
        'completed',
        'rejected',
        'superseded',
    ]
    outcome: str
    retention: Literal[
        'decision_critical',
        'active_scene',
        'background',
    ]
    source_refs: list[ConversationProgressSourceRefDoc]
    first_seen_at: str
    updated_at: str


class ConversationEpisodeStateDoc(TypedDict):
    """Exact V2 document in ``conversation_episode_state``."""

    schema_version: Literal['conversation_progress.v2']
    episode_state_id: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    status: Literal['active', 'suspended', 'closed']
    continuity: Literal[
        'same_episode',
        'related_shift',
        'sharp_transition',
    ]
    turn_count: int
    episode_narrative: str
    current_thread: str
    character_stance: str
    user_goal: str
    current_blocker: str
    emotional_trajectory: str
    events: list[ConversationProgressEventDoc]
    overused_moves: list[str]
    recent_turn_refs: list[str]
    compacted_block_refs: list[str]
    created_at: str
    updated_at: str
    expires_at: str
    purge_after: datetime


class ConversationEpisodeBlockDoc(TypedDict):
    """Exact compacted document in ``conversation_episode_blocks``."""

    schema_version: Literal['conversation_progress_block.v1']
    block_id: str
    episode_state_id: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    level: int
    source_turn_count: int
    covered_turn_refs: list[str]
    source_block_ids: list[str]
    narrative: str
    events: list[ConversationProgressEventDoc]
    semantic_keys: list[str]
    source_started_at: str
    source_ended_at: str
    content_hash: str
    superseded_by_block_id: str
    embedding: list[float]
    created_at: str
    expires_at: str
    purge_after: datetime


class InternalMonologueResidueSourceRefDoc(TypedDict):
    """Sanitized source identifier for an internal residue row."""

    ref_kind: str
    ref_id: str


class InternalMonologueResidueV2Doc(TypedDict):
    """Canonical private residue row in ``internal_monologue_residue_state``."""

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
    schema_version: Literal["internal_monologue_residue.v2"]
    operation_id: str
    disposition: Literal["append", "replace_scope", "clear_scope"]
    purge_at: datetime


INTERNAL_MONOLOGUE_RESIDUE_V2_REQUIRED_FIELDS = frozenset({
    "residue_id",
    "character_id",
    "scope_key",
    "scope_kind",
    "platform",
    "platform_channel_id",
    "channel_type",
    "global_user_id",
    "residue_text",
    "source_kind",
    "source_refs",
    "created_at",
    "schema_version",
    "operation_id",
    "disposition",
    "purge_at",
})


def validate_internal_monologue_residue_v2_doc(
    value: Mapping[str, object],
) -> None:
    """Validate required v2 fields before a residue row reaches MongoDB."""

    missing = INTERNAL_MONOLOGUE_RESIDUE_V2_REQUIRED_FIELDS - set(value)
    if missing:
        raise ValueError("v2 residue document is missing required fields")
    if value["schema_version"] != "internal_monologue_residue.v2":
        raise ValueError("v2 residue document schema_version is invalid")
    if value["disposition"] not in {
        "append",
        "replace_scope",
        "clear_scope",
    }:
        raise ValueError("v2 residue document disposition is invalid")
    for field_name in (
        "residue_id",
        "character_id",
        "scope_key",
        "scope_kind",
        "platform",
        "platform_channel_id",
        "channel_type",
        "created_at",
        "operation_id",
    ):
        if not isinstance(value[field_name], str):
            raise ValueError(
                f"v2 residue document {field_name} is invalid"
            )
        if field_name != "global_user_id" and not value[field_name]:
            raise ValueError(
                f"v2 residue document {field_name} is empty"
            )
    if value["scope_kind"] == "user_thread" and not value["global_user_id"]:
        raise ValueError(
            "v2 user-thread residue document global_user_id is empty"
        )
    if not isinstance(value["global_user_id"], str):
        raise ValueError(
            "v2 residue document global_user_id is invalid"
        )
    if not isinstance(value["residue_text"], str):
        raise ValueError("v2 residue document residue_text is invalid")
    source_refs = value["source_refs"]
    if not isinstance(source_refs, list):
        raise ValueError("v2 residue document source_refs is invalid")
    for source_ref in source_refs:
        if not isinstance(source_ref, Mapping):
            raise ValueError("v2 residue document source_ref is invalid")
        if not isinstance(source_ref.get("ref_kind"), str):
            raise ValueError("v2 residue document source_ref kind is invalid")
        if not isinstance(source_ref.get("ref_id"), str):
            raise ValueError("v2 residue document source_ref id is invalid")
        if not source_ref["ref_kind"] or not source_ref["ref_id"]:
            raise ValueError("v2 residue document source_ref is empty")
    purge_at = value["purge_at"]
    if not isinstance(purge_at, datetime) or purge_at.tzinfo is None:
        raise ValueError("v2 residue document purge_at is invalid")


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

    cognition_state: dict                         # Validated cognition_state.v2 user state


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


class CharacterProfileDoc(CharacterEffectiveIdentityV1, total=False):
    """Graph-facing composition of latest identity and operational state."""

    global_user_id: str
    cognition_state: dict
    updated_at: str


class InternalActionLatchV1(TypedDict, total=False):
    """Durable one-shot continuation request emitted by settled cognition."""

    schema_version: Literal["internal_action_latch.v1"]
    latch_id: str
    idempotency_key: str
    source_episode_id: str
    source_action_attempt_id: str
    continuation_objective: str
    evidence_refs: list[dict]
    target_scope: dict
    privacy_scope: str
    continuation_depth: int
    status: Literal["pending", "claimed", "consumed", "expired", "failed"]
    not_before: str
    expires_at: str
    claimed_by: str
    claim_token: str
    claim_expires_at: str
    attempt_count: int
    max_attempts: Literal[3]
    last_error_code: str
    consumed_episode_id: str
    created_at: str
    updated_at: str
    purge_after: str


class InternalActionLatchClaimV1(TypedDict, total=False):
    """Claim result returned to the internal-thought producer."""

    latch: InternalActionLatchV1
    claim_token: str


class PostTurnLifecycleRecordV1(TypedDict, total=False):
    """Durable post-turn action/consolidation lifecycle projection."""

    schema_version: Literal["post_turn_lifecycle_record.v1"]
    lifecycle_record_id: str
    source_episode_id: str
    delivery_tracking_id: str
    action_projections: list[dict]
    status: Literal["skipped", "completed", "partial", "failed"]
    error_codes: list[str]
    created_at: str
    purge_after: str


class CharacterOperationalReceiptV1(TypedDict, total=False):
    """Durable terminal state for one character carry-over episode."""

    schema_version: Literal["character_operational_receipt.v1"]
    source_episode_id: str
    status: Literal[
        "pending",
        "no_change",
        "committed",
        "failed",
        "timed_out",
    ]
    sequence: int
    durable: bool
    base_updated_at: str
    committed_updated_at: str
    registered_at: str
    completed_at: str
    lease_owner: str
    lease_expires_at: str
    attempt_count: int
    error_code: str | None


class CharacterOperationalClaimV1(TypedDict, total=False):
    """Result of atomically claiming one lifecycle receipt."""

    claim_status: Literal["claimed", "in_progress", "terminal"]
    receipt: CharacterOperationalReceiptV1


class PostTurnLifecycleRecordV2(TypedDict, total=False):
    """Mutable post-turn audit record with an operational receipt."""

    schema_version: Literal["post_turn_lifecycle_record.v2"]
    lifecycle_record_id: str
    source_episode_id: str
    delivery_tracking_id: str
    action_projections: list[dict]
    status: Literal["skipped", "completed", "partial", "failed"]
    error_codes: list[str]
    character_operational_receipt: CharacterOperationalReceiptV1
    created_at: str
    purge_after: str


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
    source_episode_id: str
    timestamp: str


class ReflectionEpisodeRefDoc(TypedDict):
    """Recursive settled-episode root carried across reflection levels."""

    root_episode_id: str
    correlation_id: str
    character_local_date: str
    scope_kind: Literal["private", "group", "self_cognition"]
    captured_at: str


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
        "daily_affect_settling",
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
    source_episode_refs: list[ReflectionEpisodeRefDoc]
    source_reflection_run_ids: list[str]
    output: dict
    promotion_decisions: list[dict]
    validation_warnings: list[str]
    error: str
    created_at: str
    updated_at: str




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
    source_llm_trace_id: str
    correlation_write_status: str
    correlation_conflict_source_llm_trace_id: str
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
    source_llm_trace_id: str
    correlation_write_status: str
    correlation_conflict_source_llm_trace_id: str
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
    source_llm_trace_id: str
    correlation_write_status: str
    correlation_conflict_source_llm_trace_id: str
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


class ResolutionLeaseDoc(TypedDict):
    """Current fenced activation lease embedded in a resolution thread."""

    activation_id: str
    lease_epoch: int
    owner_id: str
    expires_at: str


class ResolutionOperationDoc(TypedDict):
    """Bounded semantic operation ledger row without model-visible content."""

    operation_id: str
    operation_payload_digest: str
    method: Literal[
        "resolution.open",
        "resolution.continue",
        "resolution.amend",
        "resolution.request_checkpoint",
        "resolution.cancel",
        "resolution.dispose_activation",
    ]
    resolution_thread_id: str
    segment_id: str
    activation_id: str | None
    lease_epoch: int | None
    dsh_message_source_id: str | None
    disposition: Literal[
        "prepared",
        "admitted_active",
        "checkpointed",
        "terminal",
        "canceled",
        "faulted",
    ]
    last_committed_seq: int | None
    outcome_digest: str | None
    fault_code: str | None


class ResolverSessionSegmentDoc(TypedDict):
    """One exact-compatible DSH session segment in a thread lineage."""

    schema_version: Literal["resolver_session_segment.v1"]
    segment_id: str
    resolution_thread_id: str
    dsh_session_id: str
    resolver_profile_version: Literal["kazusa-resolver-v1"]
    dsh_release: Literal["0.1.1-rc.2"]
    session_store_epoch: Literal["dsh-sqlite-0.1.1-rc.2-v1"]
    tool_catalog_digest: str
    policy_epoch: str
    scope_fingerprint: str
    audience_fingerprint: str
    model_route: str
    state: Literal["live", "checkpointed", "terminal", "canceled", "faulted"]
    last_committed_seq: int
    parent_segment_id: str | None
    rotation_reason: str | None
    created_at: str
    last_used_at: str


class ResolutionThreadDoc(TypedDict):
    """Strict standalone resolution thread lifecycle document."""

    schema_version: Literal["resolution_thread_store.v1"]
    resolution_thread_id: str
    brain_conversation_ref: str
    root_goal_ref: str
    current_segment_id: str
    state: Literal["active", "checkpointed", "terminal", "canceled", "faulted"]
    priority: Literal["now", "background"]
    audience_fingerprint: str
    scope_fingerprint: str
    created_at: str
    updated_at: str
    last_terminal_status: str | None
    continuation_eligible_until: str
    document_revision: int
    lease_epoch: int
    current_lease: ResolutionLeaseDoc | None
    segments: list[ResolverSessionSegmentDoc]
    operations: list[ResolutionOperationDoc]


class DshTaskBindingDoc(TypedDict):
    """Durable identity binding between Brain task delivery and DSH state."""

    schema_version: Literal["dsh_task_binding.v1"]
    task_session_id: str
    semantic_objective: str
    goal_continuation_ref: dict[str, object]
    source_scope: dict[str, object]
    state: str
    start_spec: dict[str, object]
    resolution_thread_id: str | None
    segment_id: str | None
    resolution_ref: dict[str, object] | None
    operation_generation: int
    current_accepted_task_id: str | None
    current_background_work_job_id: str | None
    latest_task_resolution_result: dict[str, object] | None
    revision: int
    created_at: str
    updated_at: str
