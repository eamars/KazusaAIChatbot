"""TypedDict document schemas for every MongoDB collection.

Each TypedDict mirrors exactly one document shape and is referenced by
function signatures across the ``db.*`` submodules. Schemas use
``total=False`` so optional fields don't trip type checkers.
"""

from __future__ import annotations

from typing import TypedDict


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
    base64_data: str      # Inline base64 — for small attachments only
    description: str      # Alt-text / transcription / OCR summary
    size_bytes: int       # File size


class ReplyContextDoc(TypedDict, total=False):
    """Structured reply metadata for a conversation message."""
    reply_to_message_id: str
    reply_to_platform_user_id: str
    reply_to_display_name: str
    reply_to_current_bot: bool
    reply_excerpt: str


class ConversationMessageDoc(TypedDict, total=False):
    """A single chat message in the ``conversation_history`` collection.

    Indexed by ``(platform, platform_channel_id, timestamp)`` (descending)
    for efficient retrieval of the most recent messages in a channel.
    """

    platform: str              # "discord" | "qq" | "wechat" | "whatsapp" | "telegram" | "system"
    platform_channel_id: str   # Original channel/group ID from the platform
    channel_type: str          # "group" | "private" | "system"
    role: str                  # "user" | "assistant"
    platform_message_id: str   # Original platform message ID when available
    platform_user_id: str      # Original user/bot ID from the platform
    global_user_id: str        # Our internal UUID key
    display_name: str          # Display name at time of message
    content: str               # Text content
    content_type: str          # "text" | "image" | "voice" | "mixed"
    mentioned_bot: bool        # Whether the platform structurally mentioned the bot
    attachments: list[AttachmentDoc]  # Images, voice, files
    reply_context: ReplyContextDoc     # Structured reply-to metadata when available
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


class UserMemoryContextEntry(TypedDict, total=False):
    """Prompt-facing projection of one user memory unit."""

    fact: str
    subjective_appraisal: str
    relationship_signal: str
    updated_at: str


class UserMemoryContextDoc(TypedDict, total=False):
    """Prompt-facing user memory context consumed by cognition."""

    stable_patterns: list[UserMemoryContextEntry]
    recent_shifts: list[UserMemoryContextEntry]
    objective_facts: list[UserMemoryContextEntry]
    milestones: list[UserMemoryContextEntry]
    active_commitments: list[UserMemoryContextEntry]


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
    """Memory base in the ``memory`` collection."""
    memory_name: str                # Name of the memory
    content: str                    # memory content
    source_global_user_id: str      # UUID4 of the user who triggered this memory (empty for non-user-specific)
    timestamp: str                  # ISO-8601 UTC timestamp of when memory was created/updated
    embedding: list[float]          # dense vector for similarity search

    # --- Structured metadata ---
    memory_type: str                # "fact" | "promise" | "impression" | "narrative" | "defense_rule"
    source_kind: str                # "conversation_extracted" | "relationship_inferred" | "reflection_inferred" | "seeded_manual" | "external_imported"
    confidence_note: str            # free-form note on how downstream should treat this memory
    status: str                     # "active" | "fulfilled" | "expired" | "superseded"
    expiry_timestamp: str | None    # ISO-8601 or None (never expires)


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
    return {
        "memory_name": memory_name,
        "content": content,
        "source_global_user_id": source_global_user_id,
        "memory_type": memory_type,
        "source_kind": source_kind,
        "confidence_note": confidence_note,
        "status": status,
        "expiry_timestamp": expiry_timestamp,
    }


class ScheduledEventDoc(TypedDict, total=False):
    """A scheduled future event in the ``scheduled_events`` collection.

    Used by the scheduler to persist pending tool calls across restarts.
    """
    event_id: str
    tool: str
    args: dict
    execute_at: str
    created_at: str
    status: str
    cancelled_at: str
    source_platform: str
    source_channel_id: str
    source_user_id: str
    source_message_id: str
    guild_id: str | None
    bot_role: str
