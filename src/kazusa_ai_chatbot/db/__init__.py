"""``kazusa_ai_chatbot.db`` — MongoDB layer.

Submodule map:

* ``_client``      — MongoDB connection, embedding client, vector index helper
* ``schemas``      — TypedDict document shapes
* ``bootstrap``    — startup: collections, indices, seeded documents
* ``conversation`` — ``conversation_history`` operations
* ``users``        — ``user_profiles`` operations (identity, profile, affinity)
* ``character``    — ``character_state`` operations
* ``memory``       — ``memory`` operations
"""

from __future__ import annotations

# ── Re-export config constants that old callers imported from here ──
from kazusa_ai_chatbot.config import AFFINITY_DEFAULT, AFFINITY_MAX, AFFINITY_MIN

# ── Client + embedding ─────────────────────────────────────────────
from kazusa_ai_chatbot.db._client import (
    close_db,
    enable_vector_index,
    get_db,
    get_text_embedding,
    get_text_embeddings_batch,
)

# ── Schemas ────────────────────────────────────────────────────────
from kazusa_ai_chatbot.db.schemas import (
    AttachmentDoc,
    CharacterProfileDoc,
    ConversationEpisodeEntryDoc,
    ConversationEpisodeStateDoc,
    ConversationMessageDoc,
    MemoryDoc,
    MentionDoc,
    PlatformAccountDoc,
    RAGCache2PersistentEntryDoc,
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
)

# ── Users (identity + profile + affinity) ─────────────────────────
from kazusa_ai_chatbot.db.users import (
    add_suspected_alias,
    backfill_character_conversation_identity,
    create_user_profile,
    ensure_character_identity,
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

# ── Memory ────────────────────────────────────────────────────────
from kazusa_ai_chatbot.db.memory import (
    enable_memory_vector_index,
    get_active_promises,
    save_memory,
    search_memory,
)

from kazusa_ai_chatbot.db.rag_cache2_persistent import (
    build_initializer_version_key,
    load_initializer_entries,
    prune_persistent_entries,
    purge_stale_initializer_entries,
    record_initializer_hit,
    upsert_initializer_entry,
)

__all__ = [
    # Config
    "AFFINITY_DEFAULT", "AFFINITY_MAX", "AFFINITY_MIN",
    # Client
    "close_db", "enable_vector_index", "get_db", "get_text_embedding", "get_text_embeddings_batch",
    # Schemas
    "AttachmentDoc", "CharacterProfileDoc",
    "ConversationEpisodeEntryDoc", "ConversationEpisodeStateDoc",
    "ConversationMessageDoc", "MemoryDoc", "MentionDoc",
    "PlatformAccountDoc", "RAGCache2PersistentEntryDoc",
    "ScheduledEventDoc", "UserMemoryContextDoc", "UserMemoryContextEntry",
    "UserMemoryUnitDoc", "UserMemoryUnitMergeHistoryEntry", "UserMemoryUnitSourceRef",
    "UserMemoryUnitStatus", "UserMemoryUnitType",
    "UserProfileDoc", "build_memory_doc",
    # Bootstrap
    "db_bootstrap",
    # Conversation
    "aggregate_conversation_by_user", "get_conversation_history", "save_conversation",
    "search_conversation_history",
    # Users
    "add_suspected_alias", "backfill_character_conversation_identity",
    "create_user_profile",
    "ensure_character_identity",
    "get_affinity",
    "get_user_profile", "link_platform_account",
    "list_users_by_affinity", "list_users_by_display_name",
    "resolve_global_user_id", "search_users_by_display_name", "update_affinity",
    "update_last_relationship_insight",
    "build_user_memory_unit_doc", "insert_user_memory_units", "query_user_memory_units",
    "search_user_memory_units_by_vector",
    "update_user_memory_unit_semantics", "update_user_memory_unit_window",
    "validate_user_memory_unit_semantics",
    # Character
    "get_character_profile", "get_character_state", "save_character_profile",
    "upsert_character_self_image", "upsert_character_state",
    # Memory
    "enable_memory_vector_index", "get_active_promises", "save_memory", "search_memory",
    # Persistent Cache2
    "build_initializer_version_key", "load_initializer_entries",
    "prune_persistent_entries", "purge_stale_initializer_entries",
    "record_initializer_hit", "upsert_initializer_entry",
]
