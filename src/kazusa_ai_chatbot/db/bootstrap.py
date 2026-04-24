"""One-shot startup routine that ensures collections, indices, and seeded
documents exist. Also runs the user_profiles legacy → diary/facts migration.

All operations are idempotent — safe to run on every service start.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from kazusa_ai_chatbot.db._client import enable_vector_index, get_db

logger = logging.getLogger(__name__)


# Heuristic markers used by the legacy → new schema migration. A fact whose
# text contains any of these phrases is more likely a subjective observation
# (diary) than an objective verifiable fact.
_DIARY_MARKERS = (
    "think", "feel", "seem", "feels like",
    "我觉得", "我想", "似乎", "好像", "感觉",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_legacy_fact(text: str) -> str:
    """Return ``"diary"`` if the text reads like a subjective note, else ``"fact"``."""
    lowered = text.lower()
    for marker in _DIARY_MARKERS:
        if marker in lowered:
            return "diary"
    return "fact"


async def db_bootstrap() -> None:
    """Create all required collections, indices, and seeded documents.

    Called once at service startup. Safe to call repeatedly.
    """
    db = await get_db()
    existing = set(await db.list_collection_names())

    required_collections = [
        "conversation_history",
        "user_profiles",
        "character_state",
        "memory",
        "scheduled_events",
        "rag_cache_index",        # NEW (Stage 2)
        "rag_metadata_index",     # NEW (Stage 2)
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
        [("status", 1), ("scheduled_at", 1)], name="event_status_scheduled",
    )
    await db.scheduled_events.create_index(
        "target_global_user_id", name="event_target_user",
    )
    await db.memory.create_index(
        "memory_name", name="memory_name_idx",
    )
    await db.memory.create_index(
        "source_global_user_id", name="memory_source_user_idx",
    )

    # ── NEW: user_profiles structured-field indexes ───────────────
    await db.user_profiles.create_index(
        [("character_diary.timestamp", -1)], name="user_diary_ts",
    )
    await db.user_profiles.create_index(
        [("objective_facts.timestamp", -1)], name="user_facts_ts",
    )
    await db.user_profiles.create_index(
        [("objective_facts.category", 1)], name="user_facts_category",
    )

    # ── NEW: rag_cache_index indexes ──────────────────────────────
    # TTL — auto-delete expired entries. expireAfterSeconds=0 means
    # "expire when ttl_expires_at is in the past".
    await db.rag_cache_index.create_index(
        "ttl_expires_at",
        expireAfterSeconds=0,
        name="rag_cache_ttl",
    )
    await db.rag_cache_index.create_index(
        [("cache_type", 1), ("global_user_id", 1), ("deleted", 1)],
        name="rag_cache_lookup",
    )
    await db.rag_cache_index.create_index(
        "cache_id", unique=True, name="rag_cache_id_unique",
    )

    # ── NEW: rag_metadata_index indexes ───────────────────────────
    await db.rag_metadata_index.create_index(
        "global_user_id", unique=True, name="rag_meta_user_unique",
    )

    # ── Vector search indexes (best-effort — requires Atlas) ──────
    for collection, index_name, path in (
        ("conversation_history", "conversation_history_vector_index", "embedding"),
        ("memory",               "memory_vector_index",              "embedding"),
        ("rag_cache_index",      "rag_cache_vector_index",           "embedding"),          # NEW
    ):
        try:
            await enable_vector_index(collection, index_name, path=path)
        except Exception:
            logger.warning(
                "Could not create vector index '%s' on %s.%s (requires Atlas)",
                index_name, collection, path,
            )

    # ── Schema migration: legacy facts → character_diary + objective_facts ──
    await _migrate_user_profiles_legacy_facts()

    logger.info("Database bootstrap complete")


async def _migrate_user_profiles_legacy_facts() -> None:
    """Split each legacy ``facts: list[str]`` into ``character_diary`` and ``objective_facts``.

    Runs once per profile — only touches docs where ``facts`` exists and
    ``character_diary`` is absent. Safe to run repeatedly.
    """
    db = await get_db()
    cursor = db.user_profiles.find({
        "facts": {"$exists": True, "$ne": []},
        "character_diary": {"$exists": False},
    })

    migrated = 0
    async for doc in cursor:
        legacy_facts = doc.get("facts") or []
        if not isinstance(legacy_facts, list) or not legacy_facts:
            continue

        diary_entries: list[dict] = []
        fact_entries: list[dict] = []
        ts = _now_iso()

        for text in legacy_facts:
            if not isinstance(text, str) or not text.strip():
                continue
            kind = _classify_legacy_fact(text)
            if kind == "diary":
                diary_entries.append({
                    "entry": text,
                    "timestamp": ts,
                    "confidence": 0.7,
                    "context": "migrated_from_legacy_facts",
                })
            else:
                fact_entries.append({
                    "fact": text,
                    "category": "general",
                    "timestamp": ts,
                    "source": "migrated_from_legacy_facts",
                    "confidence": 0.8,
                })

        await db.user_profiles.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "character_diary": diary_entries,
                "diary_updated_at": ts,
                "objective_facts": fact_entries,
                "facts_updated_at": ts,
            }},
        )
        migrated += 1

    if migrated:
        logger.info(
            "Migrated %d user_profiles document(s): legacy facts → character_diary + objective_facts",
            migrated,
        )
