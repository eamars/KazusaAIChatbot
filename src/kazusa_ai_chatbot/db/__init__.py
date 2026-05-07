"""``kazusa_ai_chatbot.db`` — MongoDB layer.

Submodule map:

* ``_client``      — MongoDB connection, embedding client, vector index helper
* ``schemas``      — TypedDict document shapes
* ``bootstrap``    — startup: collections, indices, seeded documents
* ``conversation`` — ``conversation_history`` operations
* ``users``        — ``user_profiles`` operations (identity, profile, affinity)
* ``character``    — ``character_state`` operations
* ``memory``       — ``memory`` operations
* ``interaction_style_images`` — L3-only interaction style overlays
* ``scheduled_events`` — scheduled-event persistence helpers
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ── Re-export config constants that old callers imported from here ──
from kazusa_ai_chatbot.config import AFFINITY_DEFAULT, AFFINITY_MAX, AFFINITY_MIN

# ── Client + embedding ─────────────────────────────────────────────
from kazusa_ai_chatbot.db._client import (
    close_db,
    enable_vector_index,
    get_text_embedding,
    get_text_embeddings_batch,
)
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.health import check_database_connection

# ── Schemas ────────────────────────────────────────────────────────
from kazusa_ai_chatbot.db.schemas import (
    AttachmentDoc,
    CharacterProfileDoc,
    CharacterReflectionRunDoc,
    ConversationEpisodeEntryDoc,
    ConversationEpisodeStateDoc,
    ConversationMessageDoc,
    InteractionStyleImageDoc,
    InteractionStyleOverlayDoc,
    InteractionStyleScopeType,
    InteractionStyleStatus,
    MemoryDoc,
    MentionDoc,
    PlatformAccountDoc,
    RAGCache2PersistentEntryDoc,
    ReflectionMessageRefDoc,
    ReflectionScopeDoc,
    ScheduledEventDoc,
    UserMemoryContextDoc,
    UserMemoryContextEntry,
    UserMemoryUnitDoc,
    UserMemoryUnitMergeHistoryEntry,
    UserMemoryUnitSourceRef,
    UserMemoryUnitStatus,
    UserMemoryUnitType,
    UserProfileDoc,
    build_memory_doc,
)

# ── Bootstrap ──────────────────────────────────────────────────────
from kazusa_ai_chatbot.db.bootstrap import db_bootstrap

# ── Conversation history ──────────────────────────────────────────
from kazusa_ai_chatbot.db.conversation import (
    aggregate_conversation_by_user,
    get_conversation_history,
    save_conversation,
    search_conversation_history,
    update_conversation_attachment_descriptions,
)
from kazusa_ai_chatbot.db.conversation_reflection import (
    explain_monitored_channel_query,
    list_recent_character_message_channels,
    list_reflection_scope_messages,
    resolve_single_private_scope_user_id,
)
from kazusa_ai_chatbot.db.reflection_cycle import (
    ensure_reflection_run_indexes,
    find_reflection_run_by_id,
    list_daily_channel_runs,
    list_existing_run_ids,
    list_hourly_runs_for_channel_day,
    upsert_reflection_run,
)
from kazusa_ai_chatbot.db.interaction_style_images import (
    build_interaction_style_context,
    empty_interaction_style_overlay,
    ensure_interaction_style_image_indexes,
    get_group_channel_style_image,
    get_user_style_image,
    upsert_group_channel_style_image,
    upsert_user_style_image,
    validate_interaction_style_overlay,
)

# ── Users (identity + profile + affinity) ─────────────────────────
from kazusa_ai_chatbot.db.users import (
    add_suspected_alias,
    backfill_character_conversation_identity,
    create_user_profile,
    ensure_character_identity,
    find_user_profile_by_identifier,
    get_affinity,
    get_user_profile,
    link_platform_account,
    list_users_by_affinity,
    list_users_by_display_name,
    resolve_global_user_id,
    search_users_by_display_name,
    update_affinity,
    update_last_relationship_insight,
)

from kazusa_ai_chatbot.db.user_memory_units import (
    build_user_memory_unit_doc,
    insert_user_memory_units,
    query_user_memory_units,
    search_user_memory_units_by_keyword,
    search_user_memory_units_by_vector,
    update_user_memory_unit_semantics,
    update_user_memory_unit_window,
    validate_user_memory_unit_semantics,
)

# ── Character state ───────────────────────────────────────────────
from kazusa_ai_chatbot.db.character import (
    get_character_profile,
    get_character_state,
    save_character_profile,
    upsert_character_self_image,
    upsert_character_state,
)

from kazusa_ai_chatbot.db.scheduled_events import (
    cancel_pending_scheduled_event,
    insert_scheduled_event,
    list_pending_scheduler_events,
    mark_scheduled_event_completed,
    mark_scheduled_event_failed,
    mark_scheduled_event_running,
    query_pending_scheduled_events,
)

from kazusa_ai_chatbot.db.rag_cache2_persistent import (
    build_initializer_version_key,
    load_initializer_entries,
    prune_persistent_entries,
    purge_stale_initializer_entries,
    record_initializer_hit,
    upsert_initializer_entry,
)

_LAZY_MEMORY_EXPORTS = {
    "enable_memory_vector_index",
    "get_active_promises",
    "save_memory",
    "search_memory",
}


def __getattr__(name: str) -> Any:
    """Resolve legacy memory helpers without creating import-time cycles.

    The evolving-memory repository imports its DB submodule during package
    import. Loading the legacy memory facade eagerly from here would pull that
    repository back in before it finishes initialising.
    """
    if name in _LAZY_MEMORY_EXPORTS:
        memory_module = import_module("kazusa_ai_chatbot.db.memory")
        resolved_value = getattr(memory_module, name)
        return resolved_value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config
    "AFFINITY_DEFAULT", "AFFINITY_MAX", "AFFINITY_MIN",
    # Client
    "check_database_connection", "close_db", "DatabaseOperationError",
    "enable_vector_index",
    "get_text_embedding", "get_text_embeddings_batch",
    # Schemas
    "AttachmentDoc", "CharacterProfileDoc", "CharacterReflectionRunDoc",
    "ConversationEpisodeEntryDoc", "ConversationEpisodeStateDoc",
    "ConversationMessageDoc", "InteractionStyleImageDoc",
    "InteractionStyleOverlayDoc", "InteractionStyleScopeType",
    "InteractionStyleStatus", "MemoryDoc", "MentionDoc",
    "PlatformAccountDoc", "RAGCache2PersistentEntryDoc",
    "ReflectionMessageRefDoc", "ReflectionScopeDoc",
    "ScheduledEventDoc", "UserMemoryContextDoc", "UserMemoryContextEntry",
    "UserMemoryUnitDoc", "UserMemoryUnitMergeHistoryEntry", "UserMemoryUnitSourceRef",
    "UserMemoryUnitStatus", "UserMemoryUnitType",
    "UserProfileDoc", "build_memory_doc",
    # Bootstrap
    "db_bootstrap",
    # Conversation
    "aggregate_conversation_by_user", "get_conversation_history", "save_conversation",
    "search_conversation_history", "update_conversation_attachment_descriptions",
    "explain_monitored_channel_query",
    "list_recent_character_message_channels",
    "list_reflection_scope_messages",
    "resolve_single_private_scope_user_id",
    "ensure_reflection_run_indexes", "find_reflection_run_by_id",
    "list_daily_channel_runs", "list_existing_run_ids",
    "list_hourly_runs_for_channel_day", "upsert_reflection_run",
    "build_interaction_style_context", "empty_interaction_style_overlay",
    "ensure_interaction_style_image_indexes", "get_group_channel_style_image",
    "get_user_style_image", "upsert_group_channel_style_image",
    "upsert_user_style_image", "validate_interaction_style_overlay",
    # Users
    "add_suspected_alias", "backfill_character_conversation_identity",
    "create_user_profile",
    "ensure_character_identity",
    "find_user_profile_by_identifier",
    "get_affinity",
    "get_user_profile", "link_platform_account",
    "list_users_by_affinity", "list_users_by_display_name",
    "resolve_global_user_id", "search_users_by_display_name", "update_affinity",
    "update_last_relationship_insight",
    "build_user_memory_unit_doc", "insert_user_memory_units", "query_user_memory_units",
    "search_user_memory_units_by_keyword",
    "search_user_memory_units_by_vector",
    "update_user_memory_unit_semantics", "update_user_memory_unit_window",
    "validate_user_memory_unit_semantics",
    # Character
    "get_character_profile", "get_character_state", "save_character_profile",
    "upsert_character_self_image", "upsert_character_state",
    # Memory
    "enable_memory_vector_index", "get_active_promises", "save_memory", "search_memory",
    # Scheduled events
    "cancel_pending_scheduled_event", "insert_scheduled_event",
    "list_pending_scheduler_events", "mark_scheduled_event_completed",
    "mark_scheduled_event_failed", "mark_scheduled_event_running",
    "query_pending_scheduled_events",
    # Persistent Cache2
    "build_initializer_version_key", "load_initializer_entries",
    "prune_persistent_entries", "purge_stale_initializer_entries",
    "record_initializer_hit", "upsert_initializer_entry",
]
