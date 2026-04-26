"""Operations against the ``user_profiles`` collection.

Covers:

* Identity resolution (``resolve_global_user_id``, ``link_platform_account``)
* Profile read/create (``get_user_profile``, ``create_user_profile``)
* Affinity (``get_affinity``, ``update_affinity``)
* Relationship insight (``update_last_relationship_insight``)
* Profile memories: ``insert_profile_memories``, ``query_user_profile_memory_blocks``
* User image (three-tier): ``upsert_user_image``
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone

from kazusa_ai_chatbot.config import (
    AFFINITY_DEFAULT,
    AFFINITY_MAX,
    AFFINITY_MIN,
    CHARACTER_GLOBAL_USER_ID,
    PROFILE_MEMORY_BUDGET,
    PROFILE_MEMORY_RECENT_LIMITS,
    PROFILE_MEMORY_SEMANTIC_THRESHOLDS,
    PROFILE_MEMORY_TTL_SECONDS,
)
from kazusa_ai_chatbot.db._client import get_db, get_text_embedding
from kazusa_ai_chatbot.db.schemas import (
    ActiveCommitmentDoc,
    CharacterDiaryEntry,
    MemoryType,
    ObjectiveFactEntry,
    UserProfileMemoryDoc,
    UserProfileDoc,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC ``datetime``.

    Args:
        value: Timestamp string emitted by internal code or the LLM.

    Returns:
        A timezone-aware UTC datetime.
    """
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _expiry_from(created_at: str, memory_type: str) -> str:
    """Return the default expiry timestamp for a profile memory.

    Args:
        created_at: Memory creation time as ISO-8601.
        memory_type: One of the ``MemoryType`` string constants.

    Returns:
        ISO-8601 UTC timestamp after the configured per-type TTL.
    """
    ttl_seconds = PROFILE_MEMORY_TTL_SECONDS[memory_type]
    return (_parse_iso(created_at) + timedelta(seconds=ttl_seconds)).isoformat()


def _active_memory_filter(now_iso: str) -> dict:
    """Build the common non-deleted, non-expired profile-memory filter.

    Args:
        now_iso: Current UTC timestamp as ISO-8601.

    Returns:
        MongoDB filter fragment for live memories.
    """
    return {
        "deleted": {"$ne": True},
        "expires_at": {"$gt": now_iso},
    }


def _memory_to_diary(memory: UserProfileMemoryDoc) -> CharacterDiaryEntry:
    """Convert a diary memory document into the existing prompt-facing shape."""
    return {
        "entry": memory.get("content", ""),
        "timestamp": memory.get("created_at", ""),
        "confidence": float(memory.get("confidence", 0.8)),
        "context": memory.get("context", ""),
    }


def _memory_to_fact(memory: UserProfileMemoryDoc) -> ObjectiveFactEntry:
    """Convert an objective fact memory into the existing prompt-facing shape."""
    return {
        "fact": memory.get("content", ""),
        "category": memory.get("category", "general"),
        "timestamp": memory.get("created_at", ""),
        "source": memory.get("source", "conversation_extracted"),
        "confidence": float(memory.get("confidence", 0.85)),
    }


def _memory_to_commitment(memory: UserProfileMemoryDoc) -> ActiveCommitmentDoc:
    """Convert a commitment memory into the existing prompt-facing shape."""
    return {
        "commitment_id": memory.get("commitment_id", ""),
        "target": memory.get("target", ""),
        "action": memory.get("action", memory.get("content", "")),
        "commitment_type": memory.get("commitment_type", ""),
        "status": memory.get("status", "active"),
        "source": memory.get("source", "conversation_extracted"),
        "created_at": memory.get("created_at", ""),
        "updated_at": memory.get("updated_at", ""),
        "due_time": memory.get("due_time"),
    }


def _memory_to_milestone(memory: UserProfileMemoryDoc) -> dict:
    """Convert a milestone memory into the generated-image milestone shape."""
    return {
        "event": memory.get("content", ""),
        "timestamp": memory.get("created_at", ""),
        "category": memory.get("event_category", memory.get("category", "")),
        "fact_category": memory.get("category", ""),
        "scope": memory.get("scope", ""),
        "superseded_by": memory.get("superseded_by"),
    }


def _empty_memory_blocks() -> dict:
    """Return the canonical empty prompt-facing memory block bundle.

    Returns:
        A dict containing the same top-level keys as
        ``query_user_profile_memory_blocks`` with empty values.
    """
    return {
        "character_diary": [],
        "objective_facts": [],
        "active_commitments": [],
        "milestones": [],
        "memories": [],
    }


_RECALL_IMAGE_MAX_RECENT_WINDOW = 6
_RECALL_IMAGE_HISTORY_FALLBACK_LIMIT = 3


def _memory_to_recall_observation(memory: UserProfileMemoryDoc) -> dict | None:
    """Convert a recalled profile memory into a recall-time image observation.

    Args:
        memory: Raw profile-memory document selected for the current recall pass.

    Returns:
        ``{"timestamp": ..., "summary": ...}`` for non-empty non-commitment
        memories, otherwise ``None``.
    """
    if memory.get("memory_type") == MemoryType.COMMITMENT:
        return None

    summary = str(memory.get("content", "")).strip()
    if not summary:
        return None

    return {
        "timestamp": str(memory.get("created_at", "")),
        "summary": summary,
    }


def _build_recall_image_doc(user_profile: UserProfileDoc, memory_blocks: dict) -> dict:
    """Build a recall-time image view from merged profile memories.

    Args:
        user_profile: Lightweight profile header with the stored generated image
            artifact, if any.
        memory_blocks: Prompt-facing memory block bundle produced by the recall
            loader.

    Returns:
        An image document in the existing ``user_image`` shape whose
        ``milestones`` come from the authoritative memory collection and whose
        recent/history sections are refreshed from the recalled memory set.
    """
    existing_image = dict((user_profile or {}).get("user_image") or {})

    recall_recent = [
        observation
        for observation in (
            _memory_to_recall_observation(memory)
            for memory in (memory_blocks.get("memories") or [])
        )
        if observation is not None
    ]
    existing_recent = [
        {
            "timestamp": str(observation.get("timestamp", "")),
            "summary": str(observation.get("summary", "")).strip(),
        }
        for observation in (existing_image.get("recent_window") or [])
        if isinstance(observation, dict) and str(observation.get("summary", "")).strip()
    ]

    recent_window: list[dict[str, str]] = []
    seen_summaries: set[str] = set()
    for observation in recall_recent + existing_recent:
        summary = observation["summary"]
        if summary in seen_summaries:
            continue
        seen_summaries.add(summary)
        recent_window.append(observation)
        if len(recent_window) >= _RECALL_IMAGE_MAX_RECENT_WINDOW:
            break

    historical_summary = str(existing_image.get("historical_summary") or "").strip()
    if not historical_summary:
        fact_summaries: list[str] = []
        for fact_entry in memory_blocks.get("objective_facts") or []:
            fact_text = str(fact_entry.get("fact", "")).strip()
            if not fact_text or fact_text in fact_summaries:
                continue
            fact_summaries.append(fact_text)
            if len(fact_summaries) >= _RECALL_IMAGE_HISTORY_FALLBACK_LIMIT:
                break

        if fact_summaries:
            historical_summary = "；".join(fact_summaries)
        else:
            historical_summary = str((user_profile or {}).get("last_relationship_insight") or "").strip()

    image_doc = dict(existing_image)
    image_doc["milestones"] = memory_blocks["milestones"]
    image_doc["recent_window"] = recent_window
    image_doc["historical_summary"] = historical_summary
    return image_doc


def hydrate_user_profile_with_memory_blocks(
    user_profile: UserProfileDoc,
    memory_blocks: dict,
) -> UserProfileDoc:
    """Merge prompt-facing memory blocks into a lightweight profile header.

    Args:
        user_profile: Header document from ``user_profiles``.
        memory_blocks: Prompt-facing blocks returned by
            ``query_user_profile_memory_blocks`` or cache.

    Returns:
        A hydrated profile dict containing ``character_diary``,
        ``objective_facts``, ``active_commitments``, and a ``user_image`` whose
        recall-time sections come from the authoritative memory collection.
    """
    hydrated = dict(user_profile)
    hydrated["character_diary"] = memory_blocks["character_diary"]
    hydrated["objective_facts"] = memory_blocks["objective_facts"]
    hydrated["active_commitments"] = memory_blocks["active_commitments"]
    hydrated["user_image"] = _build_recall_image_doc(hydrated, memory_blocks)
    return hydrated


async def build_user_profile_recall_bundle(
    global_user_id: str,
    *,
    user_profile: UserProfileDoc | None = None,
    topic_embedding: list[float] | None = None,
    include_semantic: bool = False,
    budget: int = PROFILE_MEMORY_BUDGET,
) -> tuple[UserProfileDoc, dict]:
    """Build a hydrated recall bundle for a user from profile header + memories.

    Args:
        global_user_id: Owner of the profile and memory blocks.
        user_profile: Optional already-fetched lightweight profile header.
        topic_embedding: Current-topic embedding for semantic recall.
        include_semantic: Whether vector recall should augment recency.
        budget: Maximum non-commitment memory count after merge and dedup.

    Returns:
        ``(hydrated_profile, memory_blocks)`` where ``hydrated_profile`` keeps
        the prompt-facing compatibility fields while ``memory_blocks`` preserves
        the raw block structure for downstream consumers.
    """
    if not global_user_id:
        empty_blocks = _empty_memory_blocks()
        return hydrate_user_profile_with_memory_blocks({}, empty_blocks), empty_blocks

    profile_header = dict(user_profile) if user_profile is not None else await get_user_profile(global_user_id)
    memory_blocks = await query_user_profile_memory_blocks(
        global_user_id,
        topic_embedding=topic_embedding,
        include_semantic=include_semantic,
        budget=budget,
    )
    hydrated_profile = hydrate_user_profile_with_memory_blocks(profile_header, memory_blocks)
    return hydrated_profile, memory_blocks


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


async def ensure_character_identity(
    *,
    platform: str,
    platform_user_id: str,
    display_name: str,
    global_user_id: str = CHARACTER_GLOBAL_USER_ID,
) -> str:
    """Ensure the character has a first-class identity in ``user_profiles``.

    Args:
        platform: Platform where the character bot account is active.
        platform_user_id: Bot account ID on that platform.
        display_name: Current character display name.
        global_user_id: Stable internal character UUID.

    Returns:
        The stable character ``global_user_id``.
    """
    clean_global_id = global_user_id.strip()
    clean_platform = platform.strip()
    clean_platform_user_id = platform_user_id.strip()
    clean_display_name = display_name.strip()
    if not clean_global_id:
        raise ValueError("character global_user_id must not be empty")

    db = await get_db()
    now = _now_iso()

    if clean_platform and clean_platform_user_id:
        await db.user_profiles.update_many(
            {"global_user_id": {"$ne": clean_global_id}},
            {
                "$pull": {
                    "platform_accounts": {
                        "platform": clean_platform,
                        "platform_user_id": clean_platform_user_id,
                    }
                }
            },
        )

    await db.user_profiles.update_one(
        {"global_user_id": clean_global_id},
        {
            "$setOnInsert": {
                "global_user_id": clean_global_id,
                "platform_accounts": [],
                "suspected_aliases": [],
                "facts": [],
                "affinity": AFFINITY_DEFAULT,
                "last_relationship_insight": "",
            }
        },
        upsert=True,
    )

    if clean_display_name:
        account = {
            "platform": clean_platform,
            "platform_user_id": clean_platform_user_id,
            "display_name": clean_display_name,
            "linked_at": now,
        }
        match_filter = {
            "global_user_id": clean_global_id,
            "platform_accounts": {
                "$elemMatch": {
                    "platform": clean_platform,
                    "platform_user_id": clean_platform_user_id,
                }
            },
        }
        existing_account = await db.user_profiles.find_one(match_filter)
        if existing_account is not None:
            await db.user_profiles.update_one(
                match_filter,
                {
                    "$set": {
                        "platform_accounts.$.display_name": clean_display_name,
                        "platform_accounts.$.linked_at": now,
                    }
                },
            )
        else:
            await db.user_profiles.update_one(
                {"global_user_id": clean_global_id},
                {"$push": {"platform_accounts": account}},
            )

    return clean_global_id


async def backfill_character_conversation_identity(
    *,
    platform: str,
    platform_user_id: str,
    global_user_id: str = CHARACTER_GLOBAL_USER_ID,
) -> int:
    """Attach the character global ID to historical assistant messages.

    Args:
        platform: Platform where the character bot account is active.
        platform_user_id: Bot account ID on that platform.
        global_user_id: Stable internal character UUID.

    Returns:
        Number of conversation rows updated.
    """
    clean_global_id = global_user_id.strip()
    clean_platform = platform.strip()
    clean_platform_user_id = platform_user_id.strip()
    if not clean_global_id or not clean_platform or not clean_platform_user_id:
        return 0

    db = await get_db()
    result = await db.conversation_history.update_many(
        {
            "platform": clean_platform,
            "platform_user_id": clean_platform_user_id,
            "role": "assistant",
            "$or": [
                {"global_user_id": ""},
                {"global_user_id": None},
                {"global_user_id": {"$exists": False}},
            ],
        },
        {"$set": {"global_user_id": clean_global_id}},
    )
    return int(result.modified_count)


async def search_users_by_display_name(name: str, limit: int = 5) -> list[dict]:
    """Search user profiles whose platform display_name partially matches name.

    Args:
        name: Display name string to search for (case-insensitive partial match).
        limit: Maximum number of distinct users to return.

    Returns:
        List of dicts, each containing ``global_user_id``, ``display_name``,
        and ``platform`` for the first matching platform account of each user.
    """
    stripped = name.strip()
    if not stripped:
        return []
    db = await get_db()
    pattern = {"$regex": re.escape(stripped), "$options": "i"}
    cursor = db.user_profiles.find(
        {"platform_accounts": {"$elemMatch": {"display_name": pattern}}},
        {"global_user_id": 1, "platform_accounts": 1},
    ).limit(limit)
    results: list[dict] = []
    async for doc in cursor:
        for acct in doc.get("platform_accounts", []):
            if re.search(re.escape(stripped), acct.get("display_name", ""), re.IGNORECASE):
                results.append({
                    "global_user_id": doc["global_user_id"],
                    "display_name": acct["display_name"],
                    "platform": acct.get("platform", ""),
                })
                break
    return results


def _display_name_regex(value: str, operator: str) -> str:
    """Build a safe MongoDB regex for an allowed display-name predicate.

    Args:
        value: Literal display-name fragment to match.
        operator: One of equals, contains, starts_with, or ends_with.

    Returns:
        Regex string with user text escaped.
    """
    escaped = re.escape(value)
    if operator == "equals":
        return f"^{escaped}$"
    if operator == "starts_with":
        return f"^{escaped}"
    if operator == "ends_with":
        return f"{escaped}$"
    return escaped


async def list_users_by_display_name(
    value: str,
    operator: str = "contains",
    *,
    source: str = "user_profiles",
    platform: str | None = None,
    platform_channel_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List users whose display names match a constrained predicate.

    Args:
        value: Literal display-name value or fragment to match.
        operator: Match operation: equals, contains, starts_with, or ends_with.
        source: Data source to enumerate: user_profiles, conversation_participants, or both.
        platform: Optional platform filter.
        platform_channel_id: Optional conversation channel filter. Only applies to
            conversation_participants.
        limit: Maximum number of unique users to return.

    Returns:
        List of user dicts with stable identifiers and source metadata.
    """
    stripped = value.strip()
    if not stripped or limit <= 0:
        return []

    allowed_sources = {"user_profiles", "conversation_participants", "both"}
    if source not in allowed_sources:
        source = "user_profiles"

    allowed_operators = {"equals", "contains", "starts_with", "ends_with"}
    if operator not in allowed_operators:
        operator = "contains"

    db = await get_db()
    pattern = _display_name_regex(stripped, operator)
    results: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    if source in {"user_profiles", "both"}:
        elem_match: dict[str, object] = {
            "display_name": {"$regex": pattern, "$options": "i"},
        }
        if platform:
            elem_match["platform"] = platform

        cursor = db.user_profiles.find(
            {"platform_accounts": {"$elemMatch": elem_match}},
            {"_id": 0, "global_user_id": 1, "platform_accounts": 1},
        ).limit(limit)
        async for doc in cursor:
            for acct in doc.get("platform_accounts", []):
                display_name = str(acct.get("display_name", ""))
                account_platform = str(acct.get("platform", ""))
                if platform and account_platform != platform:
                    continue
                if not re.search(pattern, display_name, re.IGNORECASE):
                    continue

                key = (str(doc["global_user_id"]), display_name, account_platform)
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "global_user_id": doc["global_user_id"],
                    "display_name": display_name,
                    "platform": account_platform,
                    "platform_user_id": str(acct.get("platform_user_id", "")),
                    "source": "user_profiles",
                })
                break
            if len(results) >= limit:
                return results

    if source in {"conversation_participants", "both"} and len(results) < limit:
        match_filter: dict[str, object] = {
            "display_name": {"$regex": pattern, "$options": "i"},
            "role": "user",
        }
        if platform:
            match_filter["platform"] = platform
        if platform_channel_id:
            match_filter["platform_channel_id"] = platform_channel_id

        pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": {
                        "global_user_id": "$global_user_id",
                        "display_name": "$display_name",
                        "platform": "$platform",
                        "platform_user_id": "$platform_user_id",
                    },
                    "message_count": {"$sum": 1},
                    "last_timestamp": {"$max": "$timestamp"},
                }
            },
            {"$sort": {"message_count": -1, "last_timestamp": -1}},
            {"$limit": limit - len(results)},
        ]
        docs = await db.conversation_history.aggregate(pipeline).to_list(length=limit - len(results))
        for doc in docs:
            identity = doc["_id"]
            display_name = str(identity.get("display_name", ""))
            account_platform = str(identity.get("platform", ""))
            key = (str(identity.get("global_user_id", "")), display_name, account_platform)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "global_user_id": str(identity.get("global_user_id", "")),
                "display_name": display_name,
                "platform": account_platform,
                "platform_user_id": str(identity.get("platform_user_id", "")),
                "source": "conversation_history",
                "message_count": int(doc.get("message_count", 0)),
                "last_timestamp": str(doc.get("last_timestamp", "")),
            })

    return results


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
    """Retrieve a user profile, stripping the MongoDB internal ``_id`` field.

    Authoritative field split:
    - ``character_diary`` stores the character's subjective per-user diary notes.
    - ``objective_facts`` stores verified objective facts about the user.
    - ``facts`` is a deprecated compatibility field and must not be treated as
      the canonical source for either category in new cognition code.
    """
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return {}
    doc.pop("_id", None)
    for legacy_field in (
        "character_diary",
        "diary_updated_at",
        "objective_facts",
        "facts_updated_at",
        "active_commitments",
        "active_commitments_updated_at",
    ):
        doc.pop(legacy_field, None)
    return doc


async def create_user_profile(user_profile: UserProfileDoc) -> None:
    """Create a user profile document.

    Args:
        user_profile: Profile document to insert. Defaults for core identity,
            affinity, and memory fields are filled before persistence.

    Returns:
        None.
    """
    db = await get_db()
    user_profile.setdefault("global_user_id", str(uuid.uuid4()))
    user_profile.setdefault("platform_accounts", [])
    user_profile.setdefault("suspected_aliases", [])
    user_profile.setdefault("affinity", AFFINITY_DEFAULT)
    user_profile.setdefault("last_relationship_insight", "")
    await db.user_profiles.insert_one(user_profile)


# ── Profile memories ───────────────────────────────────────────────


async def insert_profile_memories(
    global_user_id: str,
    memories: list[UserProfileMemoryDoc],
) -> list[UserProfileMemoryDoc]:
    """Persist profile memories into ``user_profile_memories``.

    Args:
        global_user_id: Owner of the memories.
        memories: Memory docs already classified by the LLM/persistence builder.

    Returns:
        The inserted or updated memory docs with generated identifiers,
        embeddings, and expiry fields populated.
    """
    if not global_user_id or not memories:
        return []

    db = await get_db()
    now_iso = _now_iso()
    persisted: list[UserProfileMemoryDoc] = []

    for memory in memories:
        content = str(memory.get("content", "")).strip()
        memory_type = str(memory.get("memory_type", "")).strip()
        if not content or memory_type not in PROFILE_MEMORY_TTL_SECONDS:
            continue

        created_at = memory.get("created_at") or now_iso
        expires_at = memory.get("expires_at") or _expiry_from(created_at, memory_type)
        if memory_type == MemoryType.COMMITMENT:
            due_time = memory.get("due_time") or expires_at
            try:
                expires_at = _parse_iso(str(due_time)).isoformat()
            except ValueError:
                expires_at = _expiry_from(created_at, memory_type)
            memory["due_time"] = expires_at

        doc: UserProfileMemoryDoc = {
            **memory,
            "memory_id": memory.get("memory_id") or str(uuid.uuid4()),
            "global_user_id": global_user_id,
            "memory_type": memory_type,
            "content": content,
            "created_at": created_at,
            "updated_at": memory.get("updated_at") or now_iso,
            "expires_at": expires_at,
            "deleted": bool(memory.get("deleted", False)),
            "embedding": memory.get("embedding") or await get_text_embedding(content),
        }

        if memory_type == MemoryType.COMMITMENT:
            commitment_id = doc.get("commitment_id") or str(uuid.uuid4())
            doc["commitment_id"] = commitment_id
            doc["status"] = doc.get("status") or "active"
            duplicate_terms = [{"commitment_id": commitment_id}]
            if doc.get("dedup_key"):
                duplicate_terms.append({"dedup_key": doc["dedup_key"]})
            existing = await db.user_profile_memories.find_one({
                "global_user_id": global_user_id,
                "memory_type": MemoryType.COMMITMENT,
                "deleted": {"$ne": True},
                "$or": duplicate_terms,
            })
            if existing is not None:
                doc["memory_id"] = existing["memory_id"]
                doc["created_at"] = existing.get("created_at", created_at)
                await db.user_profile_memories.update_one(
                    {"memory_id": existing["memory_id"]},
                    {"$set": doc},
                )
                persisted.append(doc)
                continue

        if memory_type == MemoryType.OBJECTIVE_FACT and doc.get("dedup_key"):
            existing_fact = await db.user_profile_memories.find_one({
                "global_user_id": global_user_id,
                "memory_type": MemoryType.OBJECTIVE_FACT,
                "deleted": {"$ne": True},
                "dedup_key": doc["dedup_key"],
            })
            if existing_fact is not None:
                persisted.append(existing_fact)
                continue

        if memory_type == MemoryType.MILESTONE and doc.get("scope"):
            await db.user_profile_memories.update_many(
                {
                    "global_user_id": global_user_id,
                    "memory_type": MemoryType.MILESTONE,
                    "scope": doc["scope"],
                    "deleted": {"$ne": True},
                    "superseded_by": {"$in": [None, ""]},
                },
                {"$set": {"superseded_by": doc["memory_id"], "updated_at": now_iso}},
            )

        await db.user_profile_memories.insert_one(doc)
        persisted.append(doc)

    return persisted


async def query_profile_memories_recent(
    global_user_id: str,
    limits: dict[str, int] | None = None,
) -> list[UserProfileMemoryDoc]:
    """Load recent profile memories plus all active commitments.

    Args:
        global_user_id: Owner of the memories.
        limits: Per-type limits for diary, objective facts, and milestones.

    Returns:
        Recent live memories sorted within each type by newest first.
    """
    if not global_user_id:
        return []

    db = await get_db()
    now_iso = _now_iso()
    effective_limits = limits or PROFILE_MEMORY_RECENT_LIMITS
    results: list[UserProfileMemoryDoc] = []

    for memory_type, limit in effective_limits.items():
        cursor = db.user_profile_memories.find({
            "global_user_id": global_user_id,
            "memory_type": memory_type,
            **_active_memory_filter(now_iso),
        }).sort("created_at", -1).limit(limit)
        results.extend(await cursor.to_list(length=limit))

    commitment_cursor = db.user_profile_memories.find({
        "global_user_id": global_user_id,
        "memory_type": MemoryType.COMMITMENT,
        "status": "active",
        **_active_memory_filter(now_iso),
    }).sort("created_at", -1)
    results.extend(await commitment_cursor.to_list(length=None))
    return results


async def query_profile_memories_vector(
    global_user_id: str,
    embedding: list[float],
    thresholds: dict[str, float] | None = None,
    limit_per_type: int = 25,
) -> list[UserProfileMemoryDoc]:
    """Run vector recall over diary, objective fact, and milestone memories.

    Args:
        global_user_id: Owner of the memories.
        embedding: Query embedding for the current topic.
        thresholds: Per-type minimum vector score.
        limit_per_type: Maximum hits to request per searchable type.

    Returns:
        Live semantic hits with embeddings removed.
    """
    if not global_user_id or not embedding:
        return []

    db = await get_db()
    now_iso = _now_iso()
    effective_thresholds = thresholds or PROFILE_MEMORY_SEMANTIC_THRESHOLDS
    results: list[UserProfileMemoryDoc] = []

    for memory_type, threshold in effective_thresholds.items():
        cursor = db.user_profile_memories.aggregate([
            {
                "$vectorSearch": {
                    "index": "user_profile_memories_vector",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": limit_per_type,
                    "filter": {
                        "global_user_id": global_user_id,
                        "memory_type": memory_type,
                        "deleted": False,
                    },
                }
            },
            {"$match": {"expires_at": {"$gt": now_iso}}},
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"score": {"$gte": threshold}}},
            {"$unset": "embedding"},
        ])
        results.extend(await cursor.to_list(length=limit_per_type))
    return results


async def query_user_profile_memory_blocks(
    global_user_id: str,
    *,
    topic_embedding: list[float] | None = None,
    include_semantic: bool = False,
    budget: int = PROFILE_MEMORY_BUDGET,
) -> dict:
    """Return prompt-facing named memory blocks for a user.

    Args:
        global_user_id: Owner of the memories.
        topic_embedding: Current-topic embedding for DEEP semantic recall.
        include_semantic: Whether vector recall should augment recent memories.
        budget: Maximum non-commitment memories after merge and dedup.

    Returns:
        Dict with ``character_diary``, ``objective_facts``,
        ``active_commitments``, ``milestones``, and raw ``memories``.
    """
    recent = await query_profile_memories_recent(global_user_id)
    semantic = []
    if include_semantic and topic_embedding:
        semantic = await query_profile_memories_vector(global_user_id, topic_embedding)

    seen_ids: set[str] = set()
    merged: list[UserProfileMemoryDoc] = []
    commitments: list[UserProfileMemoryDoc] = []
    for memory in recent + semantic:
        memory_id = memory.get("memory_id", "")
        if not memory_id or memory_id in seen_ids:
            continue
        seen_ids.add(memory_id)
        if memory.get("memory_type") == MemoryType.COMMITMENT:
            commitments.append(memory)
            continue
        if len(merged) < budget:
            merged.append(memory)

    all_memories = merged + commitments
    # Milestones are a kind of fact; the write path stores each milestone
    # exactly once under MemoryType.MILESTONE and the read path surfaces it
    # in both the milestone timeline and the objective_facts block so
    # downstream prompts still see it as a fact.
    return {
        "character_diary": [
            _memory_to_diary(memory)
            for memory in all_memories
            if memory.get("memory_type") == MemoryType.DIARY_ENTRY
        ],
        "objective_facts": [
            _memory_to_fact(memory)
            for memory in all_memories
            if memory.get("memory_type") in (MemoryType.OBJECTIVE_FACT, MemoryType.MILESTONE)
        ],
        "active_commitments": [
            _memory_to_commitment(memory)
            for memory in commitments
        ],
        "milestones": [
            _memory_to_milestone(memory)
            for memory in all_memories
            if memory.get("memory_type") == MemoryType.MILESTONE
        ],
        "memories": all_memories,
    }


async def update_commitment_status(
    global_user_id: str,
    commitment_id: str,
    new_status: str,
) -> None:
    """Update one profile-memory commitment status.

    Args:
        global_user_id: Owner of the commitment.
        commitment_id: Stable commitment identifier.
        new_status: New lifecycle status.

    Returns:
        None.
    """
    if not global_user_id or not commitment_id:
        return
    db = await get_db()
    await db.user_profile_memories.update_one(
        {
            "global_user_id": global_user_id,
            "memory_type": MemoryType.COMMITMENT,
            "commitment_id": commitment_id,
        },
        {"$set": {"status": new_status, "updated_at": _now_iso()}},
    )


async def expire_overdue_profile_memories() -> int:
    """Mark active commitments past expiry as expired.

    Returns:
        Number of commitment docs updated.
    """
    db = await get_db()
    result = await db.user_profile_memories.update_many(
        {
            "memory_type": MemoryType.COMMITMENT,
            "status": "active",
            "expires_at": {"$lte": _now_iso()},
        },
        {"$set": {"status": "expired", "updated_at": _now_iso()}},
    )
    return int(result.modified_count)


# ── Prompt-shape compatibility helpers backed by user_profile_memories ──


async def get_character_diary(global_user_id: str) -> list[CharacterDiaryEntry]:
    """Retrieve prompt-facing diary entries from ``user_profile_memories``."""
    blocks = await query_user_profile_memory_blocks(global_user_id, include_semantic=False)
    return blocks["character_diary"]


async def upsert_character_diary(
    global_user_id: str,
    new_entries: list[CharacterDiaryEntry],
) -> None:
    """Insert diary entries into ``user_profile_memories``."""
    memories = [
        {
            "memory_type": MemoryType.DIARY_ENTRY,
            "content": entry.get("entry", ""),
            "created_at": entry.get("timestamp") or _now_iso(),
            "updated_at": _now_iso(),
            "confidence": entry.get("confidence", 0.8),
            "context": entry.get("context", ""),
        }
        for entry in new_entries
    ]
    await insert_profile_memories(global_user_id, memories)


# ── Objective facts (NEW — verified facts) ─────────────────────────


async def get_objective_facts(global_user_id: str) -> list[ObjectiveFactEntry]:
    """Retrieve prompt-facing objective facts from ``user_profile_memories``."""
    blocks = await query_user_profile_memory_blocks(global_user_id, include_semantic=False)
    return blocks["objective_facts"]


async def upsert_objective_facts(
    global_user_id: str,
    new_facts: list[ObjectiveFactEntry],
) -> None:
    """Insert objective facts into ``user_profile_memories``."""
    memories = [
        {
            "memory_type": MemoryType.OBJECTIVE_FACT,
            "content": fact.get("fact", ""),
            "created_at": fact.get("timestamp") or _now_iso(),
            "updated_at": _now_iso(),
            "category": fact.get("category", "general"),
            "source": fact.get("source", "conversation_extracted"),
            "confidence": fact.get("confidence", 0.85),
            "dedup_key": str(fact.get("fact", "")).strip().lower(),
        }
        for fact in new_facts
    ]
    await insert_profile_memories(global_user_id, memories)


async def upsert_active_commitments(
    global_user_id: str,
    new_commitments: list[ActiveCommitmentDoc],
) -> None:
    """Merge active commitments into ``user_profile_memories`` by dedup key.

    Args:
        global_user_id: Owner of the commitments.
        new_commitments: New or updated commitment rows to merge.

    Returns:
        None.
    """
    memories = [
        {
            "memory_type": MemoryType.COMMITMENT,
            "content": commitment.get("action", ""),
            "action": commitment.get("action", ""),
            "commitment_id": commitment.get("commitment_id", ""),
            "target": commitment.get("target", ""),
            "commitment_type": commitment.get("commitment_type", ""),
            "status": commitment.get("status", "active"),
            "source": commitment.get("source", "conversation_extracted"),
            "created_at": commitment.get("created_at") or _now_iso(),
            "updated_at": commitment.get("updated_at") or _now_iso(),
            "due_time": commitment.get("due_time"),
            "dedup_key": str(commitment.get("action", "")).strip().lower(),
        }
        for commitment in new_commitments
    ]
    await insert_profile_memories(global_user_id, memories)


async def update_active_commitment_status(
    global_user_id: str,
    commitment_id: str,
    status: str,
) -> None:
    """Update the lifecycle status of one active commitment.

    Args:
        global_user_id: Owner of the commitment.
        commitment_id: Stable commitment identifier.
        status: New lifecycle status.

    Returns:
        None.
    """
    await update_commitment_status(global_user_id, commitment_id, status)


# ── Three-tier user image ─────────────────────────────────────────


async def upsert_user_image(global_user_id: str, image_doc: dict) -> None:
    """Persist the three-tier user image document to ``user_profiles``.

    The image document shape:
    ``{milestones: [...], recent_window: [...], historical_summary: str, meta: dict}``

    Args:
        global_user_id: Owner of the image.
        image_doc: The full three-tier image document to write.
    """
    if not image_doc or not global_user_id:
        return
    db = await get_db()
    await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"user_image": image_doc}},
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
