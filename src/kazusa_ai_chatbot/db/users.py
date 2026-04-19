"""Operations against the ``user_profiles`` collection.

Covers:

* Identity resolution (``resolve_global_user_id``, ``link_platform_account``)
* Profile read/create (``get_user_profile``, ``create_user_profile``)
* Affinity (``get_affinity``, ``update_affinity``)
* Relationship insight (``update_last_relationship_insight``)
* Character diary (NEW): ``get_character_diary``, ``upsert_character_diary``
* Objective facts (NEW): ``get_objective_facts``, ``upsert_objective_facts``
* Legacy facts shim: ``get_user_facts``, ``upsert_user_facts``,
  ``overwrite_user_facts`` — kept for backward compat; new code should
  use the diary/facts pair above.
* Vector search: ``enable_user_facts_vector_index``, ``search_users_by_facts``
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.config import AFFINITY_DEFAULT, AFFINITY_MAX, AFFINITY_MIN
from kazusa_ai_chatbot.db._client import enable_vector_index, get_db, get_text_embedding
from kazusa_ai_chatbot.db.schemas import (
    CharacterDiaryEntry,
    ObjectiveFactEntry,
    UserProfileDoc,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── Identity resolution ────────────────────────────────────────────


async def resolve_global_user_id(
    platform: str,
    platform_user_id: str,
    display_name: str = "",
) -> str:
    """Look up or auto-create a global_user_id for a platform account.

    If no matching ``(platform, platform_user_id)`` exists, a new
    ``UserProfileDoc`` is created with a fresh UUID4 and returned.
    """
    db = await get_db()
    doc = await db.user_profiles.find_one(
        {"platform_accounts": {"$elemMatch": {"platform": platform, "platform_user_id": platform_user_id}}}
    )
    if doc is not None:
        return doc["global_user_id"]

    new_id = str(uuid.uuid4())
    new_profile: UserProfileDoc = {
        "global_user_id": new_id,
        "platform_accounts": [{
            "platform": platform,
            "platform_user_id": platform_user_id,
            "display_name": display_name,
            "linked_at": _now_iso(),
        }],
        "suspected_aliases": [],
        "facts": [],
        "affinity": AFFINITY_DEFAULT,
        "last_relationship_insight": "",
        "embedding": [],
    }
    await db.user_profiles.insert_one(new_profile)
    logger.info("Created new user profile %s for %s/%s", new_id, platform, platform_user_id)
    return new_id


async def link_platform_account(
    global_user_id: str,
    platform: str,
    platform_user_id: str,
    display_name: str = "",
) -> None:
    """Add a platform account to an existing user profile."""
    db = await get_db()
    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$addToSet": {"platform_accounts": {
            "platform": platform,
            "platform_user_id": platform_user_id,
            "display_name": display_name,
            "linked_at": _now_iso(),
        }}},
    )


async def add_suspected_alias(
    global_user_id: str,
    other_global_user_id: str,
) -> None:
    """Record a suspected cross-platform alias between two users."""
    db = await get_db()
    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$addToSet": {"suspected_aliases": other_global_user_id}},
    )
    await db.user_profiles.update_one(
        {"global_user_id": other_global_user_id},
        {"$addToSet": {"suspected_aliases": global_user_id}},
    )


# ── Profile read/create ───────────────────────────────────────────


async def get_user_profile(global_user_id: str) -> UserProfileDoc:
    """Retrieve a user profile, stripping ``_id`` and bulky embedding fields."""
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return {}
    doc.pop("_id", None)
    doc.pop("embedding", None)
    doc.pop("diary_embedding", None)
    doc.pop("facts_embedding", None)
    return doc


async def create_user_profile(user_profile: UserProfileDoc) -> None:
    """Create a user profile document.

    Computes the legacy ``embedding`` from ``facts`` if provided, so old
    callers continue to work. Diary/facts embeddings should be set via
    ``upsert_character_diary``/``upsert_objective_facts``.
    """
    db = await get_db()
    facts = user_profile.get("facts", [])
    if facts:
        user_profile["embedding"] = await get_text_embedding("\n".join(facts))
    else:
        user_profile["embedding"] = []
    user_profile.setdefault("global_user_id", str(uuid.uuid4()))
    user_profile.setdefault("platform_accounts", [])
    user_profile.setdefault("suspected_aliases", [])
    user_profile.setdefault("affinity", AFFINITY_DEFAULT)
    user_profile.setdefault("last_relationship_insight", "")
    await db.user_profiles.insert_one(user_profile)


# ── Character diary (NEW — subjective observations) ────────────────


async def get_character_diary(global_user_id: str) -> list[CharacterDiaryEntry]:
    """Retrieve the character's subjective diary entries about ``global_user_id``."""
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return []
    return doc.get("character_diary", [])


async def upsert_character_diary(
    global_user_id: str,
    new_entries: list[CharacterDiaryEntry],
) -> None:
    """Append diary entries and recompute ``diary_embedding``.

    Existing entries are preserved in chronological order; new entries are
    appended at the end. The combined ``entry`` text of every diary item is
    embedded as a single vector for semantic search.

    Args:
        global_user_id: Owner of the diary.
        new_entries: List of ``CharacterDiaryEntry`` dicts to append.
    """
    if not new_entries:
        return
    db = await get_db()
    existing = await get_character_diary(global_user_id)
    merged = list(existing) + list(new_entries)

    diary_text = "\n".join(e.get("entry", "") for e in merged if e.get("entry"))
    diary_embedding = await get_text_embedding(diary_text) if diary_text else []

    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {
            "global_user_id": global_user_id,
            "character_diary": merged,
            "diary_embedding": diary_embedding,
            "diary_updated_at": _now_iso(),
        }},
        upsert=True,
    )


# ── Objective facts (NEW — verified facts) ─────────────────────────


async def get_objective_facts(global_user_id: str) -> list[ObjectiveFactEntry]:
    """Retrieve the verified objective facts about ``global_user_id``."""
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return []
    return doc.get("objective_facts", [])


async def upsert_objective_facts(
    global_user_id: str,
    new_facts: list[ObjectiveFactEntry],
) -> None:
    """Add objective facts (case-insensitive dedup on ``fact``) and recompute embedding.

    A new entry with the same fact text (case-insensitive) overwrites any
    existing entry. The combined ``fact`` text of every entry is embedded
    as a single vector.

    Args:
        global_user_id: Owner of the facts.
        new_facts: List of ``ObjectiveFactEntry`` dicts to merge in.
    """
    if not new_facts:
        return
    db = await get_db()
    existing = await get_objective_facts(global_user_id)

    by_text: dict[str, ObjectiveFactEntry] = {f.get("fact", "").lower(): f for f in existing if f.get("fact")}
    for nf in new_facts:
        text = nf.get("fact", "")
        if text:
            by_text[text.lower()] = nf

    merged = list(by_text.values())
    facts_text = "\n".join(f.get("fact", "") for f in merged if f.get("fact"))
    facts_embedding = await get_text_embedding(facts_text) if facts_text else []

    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {
            "global_user_id": global_user_id,
            "objective_facts": merged,
            "facts_embedding": facts_embedding,
            "facts_updated_at": _now_iso(),
        }},
        upsert=True,
    )


# ── Legacy flat-list facts (DEPRECATED — kept for backward compat) ──


async def get_user_facts(global_user_id: str) -> list[str]:
    """Return a flat list of all known text about the user (DEPRECATED).

    Combines diary entries and objective facts into a single list of strings.
    Falls back to the legacy ``facts`` field when the new schema fields are
    absent. New code should call :func:`get_character_diary` and
    :func:`get_objective_facts` directly.
    """
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return []
    diary = [e.get("entry", "") for e in doc.get("character_diary", []) if e.get("entry")]
    facts = [f.get("fact", "") for f in doc.get("objective_facts", []) if f.get("fact")]
    if diary or facts:
        return diary + facts
    return doc.get("facts", [])


async def upsert_user_facts(global_user_id: str, new_facts: list[str]) -> None:
    """Append text-only facts to the legacy ``facts`` array (DEPRECATED).

    Order is preserved and duplicates are filtered. Embedding is recomputed
    over the full merged list. New code should use
    :func:`upsert_character_diary` or :func:`upsert_objective_facts`.
    """
    db = await get_db()
    existing_doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    existing = existing_doc.get("facts", []) if existing_doc else []
    merged = list(dict.fromkeys(existing + new_facts))

    if merged:
        embedding = await get_text_embedding("\n".join(merged))
    else:
        embedding = []

    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"global_user_id": global_user_id, "facts": merged, "embedding": embedding}},
        upsert=True,
    )


async def overwrite_user_facts(global_user_id: str, facts: list[str]) -> None:
    """Replace the legacy ``facts`` array entirely (DEPRECATED)."""
    db = await get_db()

    if facts:
        embedding = await get_text_embedding("\n".join(facts))
    else:
        embedding = []

    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"global_user_id": global_user_id, "facts": facts, "embedding": embedding}},
        upsert=True,
    )


# ── Affinity & relationship insight ────────────────────────────────


async def get_affinity(global_user_id: str) -> int:
    """Return the affinity score for a user (0–1000, default ``AFFINITY_DEFAULT``)."""
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return AFFINITY_DEFAULT
    return doc.get("affinity", AFFINITY_DEFAULT)


async def update_affinity(global_user_id: str, delta: int) -> int:
    """Apply a delta to the user's affinity score, clamped to ``[AFFINITY_MIN, AFFINITY_MAX]``.

    Creates the ``user_profiles`` doc if it doesn't exist yet.

    Returns:
        The new (clamped) affinity value.
    """
    current = await get_affinity(global_user_id)
    new_value = max(AFFINITY_MIN, min(AFFINITY_MAX, current + delta))
    db = await get_db()
    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"affinity": new_value}},
        upsert=True,
    )
    return new_value


async def update_last_relationship_insight(global_user_id: str, insight: str) -> None:
    """Update the last relationship insight for a user."""
    db = await get_db()
    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"last_relationship_insight": insight}},
        upsert=True,
    )


# ── Vector search on legacy facts embedding ────────────────────────


async def enable_user_facts_vector_index() -> None:
    """Create the legacy vector index on ``user_profiles.embedding``."""
    await enable_vector_index("user_profiles", "user_facts_vector_index")


async def search_users_by_facts(
    query: str,
    limit: int = 5,
) -> list[tuple[float, UserProfileDoc]]:
    """Search users by semantic similarity over the legacy ``embedding`` field.

    Args:
        query: Search query text.
        limit: Maximum number of results to return.

    Returns:
        A list of ``(score, user_doc)`` tuples sorted by similarity (highest first).
    """
    query_embedding = await get_text_embedding(query)
    db = await get_db()
    collection = db.user_profiles

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": "user_facts_vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"global_user_id": 1, "facts": 1, "affinity": 1, "score": 1, "_id": 0}},
    ]

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)

    results = []
    for doc in docs:
        score = doc.pop("score")
        results.append((score, doc))
    return results
