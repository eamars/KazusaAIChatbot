"""``kazusa_ai_chatbot.db`` — MongoDB layer.

This package replaces the former monolithic ``db.py`` module. All previously
public names are re-exported here so ``from kazusa_ai_chatbot.db import X``
continues to work unchanged.

Submodule map:

* ``_client``      — MongoDB connection, embedding client, vector index helper
* ``schemas``      — TypedDict document shapes
* ``bootstrap``    — startup: collections, indices, schema migration
* ``conversation`` — ``conversation_history`` operations
* ``users``        — ``user_profiles`` operations (identity, profile, diary, facts, affinity)
* ``character``    — ``character_state`` operations
* ``memory``       — ``memory`` operations
* ``rag_cache``    — ``rag_cache_index`` + ``rag_metadata_index`` operations
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
)

# ── Schemas ────────────────────────────────────────────────────────
from kazusa_ai_chatbot.db.schemas import (
    AttachmentDoc,
    CharacterDiaryEntry,
    CharacterProfileDoc,
    ConversationMessageDoc,
    MemoryDoc,
    ObjectiveFactEntry,
    PlatformAccountDoc,
    RagCacheIndexDoc,
    RagMetadataIndexDoc,
    ScheduledEventDoc,
    UserProfileDoc,
    build_memory_doc,
)

# ── Bootstrap ──────────────────────────────────────────────────────
from kazusa_ai_chatbot.db.bootstrap import db_bootstrap

# ── Conversation history ──────────────────────────────────────────
from kazusa_ai_chatbot.db.conversation import (
    get_conversation_history,
    save_conversation,
    search_conversation_history,
)

# ── Users (identity + profile + diary + facts + affinity) ─────────
from kazusa_ai_chatbot.db.users import (
    add_suspected_alias,
    create_user_profile,
    enable_user_facts_vector_index,
    get_affinity,
    get_character_diary,
    get_objective_facts,
    get_user_facts,
    get_user_profile,
    link_platform_account,
    overwrite_user_facts,
    resolve_global_user_id,
    search_users_by_facts,
    update_affinity,
    update_last_relationship_insight,
    upsert_character_diary,
    upsert_objective_facts,
    upsert_user_facts,
)

# ── Character state ───────────────────────────────────────────────
from kazusa_ai_chatbot.db.character import (
    get_character_profile,
    get_character_state,
    save_character_profile,
    upsert_character_state,
)

# ── Memory ────────────────────────────────────────────────────────
from kazusa_ai_chatbot.db.memory import (
    enable_memory_vector_index,
    save_memory,
    search_memory,
)

# ── RAG cache (NEW) ───────────────────────────────────────────────
from kazusa_ai_chatbot.db.rag_cache import (
    clear_all_cache_for_user,
    find_cache_entries,
    get_rag_version,
    increment_rag_version,
    insert_cache_entry,
    soft_delete_cache_entries,
)

__all__ = [
    # Config
    "AFFINITY_DEFAULT", "AFFINITY_MAX", "AFFINITY_MIN",
    # Client
    "close_db", "enable_vector_index", "get_db", "get_text_embedding",
    # Schemas
    "AttachmentDoc", "CharacterDiaryEntry", "CharacterProfileDoc",
    "ConversationMessageDoc", "MemoryDoc", "ObjectiveFactEntry",
    "PlatformAccountDoc", "RagCacheIndexDoc", "RagMetadataIndexDoc",
    "ScheduledEventDoc", "UserProfileDoc", "build_memory_doc",
    # Bootstrap
    "db_bootstrap",
    # Conversation
    "get_conversation_history", "save_conversation", "search_conversation_history",
    # Users
    "add_suspected_alias", "create_user_profile", "enable_user_facts_vector_index",
    "get_affinity", "get_character_diary", "get_objective_facts", "get_user_facts",
    "get_user_profile", "link_platform_account", "overwrite_user_facts",
    "resolve_global_user_id", "search_users_by_facts", "update_affinity",
    "update_last_relationship_insight", "upsert_character_diary",
    "upsert_objective_facts", "upsert_user_facts",
    # Character
    "get_character_profile", "get_character_state", "save_character_profile",
    "upsert_character_state",
    # Memory
    "enable_memory_vector_index", "save_memory", "search_memory",
    # RAG cache
    "clear_all_cache_for_user", "find_cache_entries", "get_rag_version",
    "increment_rag_version", "insert_cache_entry", "soft_delete_cache_entries",
]
