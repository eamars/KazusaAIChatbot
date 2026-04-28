"""One-shot startup routine that ensures collections, indices, and seeded
documents exist.

All operations are idempotent — safe to run on every service start.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from kazusa_ai_chatbot.db._client import enable_vector_index, get_db

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def db_bootstrap() -> None:
    """Create all required collections, indices, and seeded documents.

    Called once at service startup. Safe to call repeatedly.
    """
    db = await get_db()
    existing = set(await db.list_collection_names())

    for legacy in ("rag_cache_index", "rag_metadata_index"):
        if legacy in existing:
            await db.drop_collection(legacy)
            logger.info("Dropped legacy collection '%s'", legacy)
            existing.discard(legacy)

    required_collections = [
        "conversation_history",
        "user_profiles",
        "character_state",
        "memory",
        "user_profile_memories",
        "scheduled_events",
        "conversation_episode_state",
    ]
    for name in required_collections:
        if name not in existing:
            await db.create_collection(name)
            logger.info("Created collection '%s'", name)
        else:
            logger.debug("Collection '%s' already exists", name)

    # ── Seed singleton character_state ─────────────────────────────
    existing_state = await db.character_state.find_one({"_id": "global"})
    if existing_state is None:
        await db.character_state.insert_one({
            "_id": "global",
            "mood": "neutral",
            "global_vibe": "",
            "reflection_summary": "",
            "updated_at": _now_iso(),
        })
        logger.info("Seeded default character_state document")

    # ── Standard regular indexes (idempotent) ──────────────────────
    await db.conversation_history.create_index(
        [("platform", 1), ("platform_channel_id", 1), ("timestamp", -1)],
        name="conv_platform_channel_ts",
    )
    await db.user_profiles.create_index(
        "global_user_id", unique=True, name="user_global_id_unique",
    )
    await db.scheduled_events.create_index(
        "event_id", unique=True, name="event_id_unique",
    )
    await db.scheduled_events.create_index(
        [("status", 1), ("execute_at", 1)], name="event_status_execute_at",
    )
    await db.scheduled_events.create_index(
        "source_user_id", name="event_source_user",
    )
    await db.conversation_episode_state.create_index(
        [("platform", 1), ("platform_channel_id", 1), ("global_user_id", 1)],
        unique=True,
        name="conversation_episode_scope_unique",
    )
    await db.conversation_episode_state.create_index(
        "expires_at",
        expireAfterSeconds=0,
        name="conversation_episode_expires_at_ttl",
    )
    await db.memory.create_index(
        "memory_name", name="memory_name_idx",
    )
    await db.memory.create_index(
        "source_global_user_id", name="memory_source_user_idx",
    )

    # ── user_profile_memories indexes ─────────────────────────────
    await db.user_profile_memories.create_index(
        "memory_id", unique=True, name="user_profile_memory_id_unique",
    )
    await db.user_profile_memories.create_index(
        [("global_user_id", 1), ("memory_type", 1), ("created_at", -1)],
        name="user_profile_memory_owner_type_recent",
    )
    await db.user_profile_memories.create_index(
        [("global_user_id", 1), ("deleted", 1), ("expires_at", 1)],
        name="user_profile_memory_live_lookup",
    )
    await db.user_profile_memories.create_index(
        [("global_user_id", 1), ("commitment_id", 1)],
        name="user_profile_memory_commitment_id",
    )
    await db.user_profile_memories.create_index(
        [("global_user_id", 1), ("memory_type", 1), ("dedup_key", 1)],
        name="user_profile_memory_dedup",
    )
    await db.user_profile_memories.create_index(
        [("memory_type", 1), ("status", 1), ("expires_at", 1)],
        name="user_profile_memory_expiry_sweep",
    )
    await db.user_profile_memories.create_index(
        [("global_user_id", 1), ("memory_type", 1), ("status", 1), ("created_at", -1)],
        name="user_profile_memory_status_recent",
    )
    await db.user_profile_memories.create_index(
        [
            ("global_user_id", 1),
            ("memory_type", 1),
            ("status", 1),
            ("deleted", 1),
            ("created_at", -1),
            ("expires_at", 1),
        ],
        name="user_profile_memory_active_commitments",
    )

    # ── Vector search indexes (best-effort — requires Atlas) ──────
    for collection, index_name, path in (
        ("conversation_history", "conversation_history_vector_index", "embedding"),
        ("memory",               "memory_vector_index",              "embedding"),
        ("user_profile_memories", "user_profile_memories_vector",     "embedding"),
    ):
        try:
            filter_paths = None
            if collection == "user_profile_memories":
                filter_paths = ["global_user_id", "memory_type", "deleted", "status"]
            await enable_vector_index(collection, index_name, path=path, filter_paths=filter_paths)
        except Exception:
            logger.warning(
                "Could not create vector index '%s' on %s.%s (requires Atlas)",
                index_name, collection, path,
            )

    logger.info("Database bootstrap complete")
