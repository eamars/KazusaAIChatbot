"""Operations against the ``user_profiles`` collection.

Covers:

* Identity resolution (``resolve_global_user_id``, ``link_platform_account``)
* Profile read/create (``get_user_profile``, ``create_user_profile``)
* Native V2 relationship-state reads
"""

from __future__ import annotations

import logging
import re
import uuid

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.schemas import (
    UserProfileDoc,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Current UTC timestamp in the cognition contract's Z form."""

    return storage_utc_now_iso().replace("+00:00", "Z")


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
        "cognition_state": build_acquaintance_user_state(
            global_user_id=new_id,
            updated_at=_now_iso(),
        ),
    }
    await db.user_profiles.insert_one(new_profile)
    logger.info(f'Created new user profile {new_id} for {platform}/{platform_user_id}')
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
                "cognition_state": build_acquaintance_user_state(
                    global_user_id=clean_global_id,
                    updated_at=now,
                ),
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
    modified_count = int(result.modified_count)
    return modified_count


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
        return_value = []
        return return_value
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
        return_value = f"^{escaped}$"
        return return_value
    if operator == "starts_with":
        return_value = f"^{escaped}"
        return return_value
    if operator == "ends_with":
        return_value = f"{escaped}$"
        return return_value
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
        return_value = []
        return return_value

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


def _preferred_platform_account(
    platform_accounts: list[dict],
    platform: str | None,
) -> dict:
    """Choose the account used to display a user in relationship rankings.

    Args:
        platform_accounts: Linked account records from ``user_profiles``.
        platform: Optional preferred platform from the current runtime context.

    Returns:
        The preferred account dict, or an empty dict when no account exists.
    """
    if platform:
        for account in platform_accounts:
            if str(account.get("platform", "")) == platform:
                return account
    if platform_accounts:
        return platform_accounts[0]
    return_value = {}
    return return_value


async def list_users_by_affinity(
    *,
    rank_order: str = "top",
    platform: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """List profiled users ordered by relationship affinity.

    Args:
        rank_order: ``"top"`` for highest-affinity users, ``"bottom"`` for
            lowest-affinity users.
        platform: Optional preferred platform for choosing display names.
        limit: Maximum number of users to return.

    Returns:
        User dicts containing identity fields, raw affinity for internal agent
        ranking. Callers that expose results to an LLM must remove the raw
        affinity value.
    """
    if limit <= 0:
        return_value = []
        return return_value

    sort_direction = 1 if rank_order == "bottom" else -1
    db = await get_db()
    cursor = (
        db.user_profiles.find(
            {
                "global_user_id": {"$ne": CHARACTER_GLOBAL_USER_ID},
                "affinity": {"$exists": True},
                "platform_accounts.0": {"$exists": True},
            },
            {
                "_id": 0,
                "global_user_id": 1,
                "platform_accounts": 1,
                "affinity": 1,
            },
        )
        .sort("affinity", sort_direction)
        .limit(limit)
    )

    results: list[dict] = []
    async for doc in cursor:
        account = _preferred_platform_account(doc["platform_accounts"], platform)
        results.append(
            {
                "global_user_id": doc["global_user_id"],
                "display_name": str(account.get("display_name", "")),
                "platform": str(account.get("platform", "")),
                "platform_user_id": str(account.get("platform_user_id", "")),
                "affinity": doc["affinity"],
            }
        )
    return results


def _relationship_rank_score(relationship: dict) -> int:
    """Compute an internal ranking score from native V2 relationship axes."""

    return (
        int(relationship.get("positive_regard", 0))
        + int(relationship.get("trust", 0))
        + int(relationship.get("attachment", 0))
        + int(relationship.get("care", 0))
        + int(relationship.get("desired_closeness", 0))
        + int(relationship.get("perceived_closeness", 0))
        + int(relationship.get("boundary_safety", 0))
        - int(relationship.get("unresolved_injury", 0))
    )


def _relationship_semantic_band(score: int) -> tuple[str, str]:
    """Describe an internal relationship score without exposing the score."""

    if score >= 240:
        return "strong connection", "positive"
    if score >= 80:
        return "growing connection", "positive"
    if score <= -240:
        return "severely strained relationship", "negative"
    if score <= -80:
        return "strained relationship", "negative"
    return "acquaintance relationship", "neutral"


async def list_users_by_relationship(
    *,
    rank_order: str = "top",
    platform: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """List users by native V2 semantic relationship state.

    The ranking is deterministic and internal. Returned rows contain only
    qualitative relationship descriptors and stable identity metadata.
    """

    if limit <= 0:
        return []

    db = await get_db()
    cursor = db.user_profiles.find(
        {
            "global_user_id": {"$ne": CHARACTER_GLOBAL_USER_ID},
            "cognition_state.state_scope": "user",
            "cognition_state.relationship": {"$exists": True},
            "platform_accounts.0": {"$exists": True},
        },
        {
            "_id": 0,
            "global_user_id": 1,
            "platform_accounts": 1,
            "cognition_state.relationship": 1,
        },
    )
    ranked: list[tuple[int, dict]] = []
    async for document in cursor:
        relationship = document.get("cognition_state", {}).get("relationship")
        if not isinstance(relationship, dict):
            continue
        account = _preferred_platform_account(
            document.get("platform_accounts", []),
            platform,
        )
        score = _relationship_rank_score(relationship)
        label, band = _relationship_semantic_band(score)
        ranked.append((score, {
            "global_user_id": str(document.get("global_user_id", "")),
            "display_name": str(account.get("display_name", "")),
            "platform": str(account.get("platform", "")),
            "platform_user_id": str(account.get("platform_user_id", "")),
            "relationship_label": label,
            "relationship_band": band,
        }))

    ranked.sort(key=lambda row: row[0], reverse=rank_order != "bottom")
    return [row[1] for row in ranked[:limit]]


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

    User memories are no longer embedded in this document. The profile is an
    identity and relationship header; cognition-facing memory comes from
    ``user_memory_units`` through RAG.
    """
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return_value = {}
        return return_value
    doc.pop("_id", None)
    return doc


async def find_user_profile_by_identifier(
    *,
    identifier: str,
    platform: str | None = None,
) -> UserProfileDoc | None:
    """Find a user profile by global id or platform account without creating it.

    Args:
        identifier: Global user id, or platform user id when ``platform`` is
            provided or no global-id match exists.
        platform: Optional platform namespace for exact account lookup.

    Returns:
        The matching user profile without Mongo internals, or ``None``.
    """

    clean_identifier = identifier.strip()
    clean_platform = platform.strip() if platform is not None else None
    if not clean_identifier:
        return_value = None
        return return_value

    db = await get_db()
    projection = {"_id": 0}
    if clean_platform:
        document = await db.user_profiles.find_one(
            {
                "platform_accounts": {
                    "$elemMatch": {
                        "platform": clean_platform,
                        "platform_user_id": clean_identifier,
                    }
                }
            },
            projection,
        )
        if document is None:
            return_value = None
            return return_value
        return_value = dict(document)
        return return_value

    document = await db.user_profiles.find_one(
        {"global_user_id": clean_identifier},
        projection,
    )
    if document is not None:
        return_value = dict(document)
        return return_value

    document = await db.user_profiles.find_one(
        {
            "platform_accounts": {
                "$elemMatch": {"platform_user_id": clean_identifier}
            }
        },
        projection,
    )
    if document is None:
        return_value = None
        return return_value
    return_value = dict(document)
    return return_value


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
    global_user_id = user_profile["global_user_id"]
    cognition_state = user_profile.get("cognition_state")
    if cognition_state is None:
        user_profile["cognition_state"] = build_acquaintance_user_state(
            global_user_id=global_user_id,
            updated_at=_now_iso(),
        )
    else:
        user_profile["cognition_state"] = validate_cognition_state(
            cognition_state
        )
    await db.user_profiles.insert_one(user_profile)


async def get_user_cognition_state(global_user_id: str) -> dict:
    """Read and validate one user's persistent cognition state."""

    db = await get_db()
    document = await db.user_profiles.find_one(
        {"global_user_id": global_user_id},
        {"cognition_state": 1},
    )
    if document is None or document.get("cognition_state") is None:
        return build_acquaintance_user_state(
            global_user_id=global_user_id,
            updated_at=_now_iso(),
        )
    state = validate_cognition_state(document["cognition_state"])
    return state


async def replace_user_cognition_state(
    global_user_id: str,
    state: dict,
) -> None:
    """Validate and replace the cognition state embedded in one user row."""

    validated_state = validate_cognition_state(state)
    if validated_state["state_scope"] != "user":
        raise ValueError("user cognition state must be user-scoped")
    if validated_state["owner_user_id"] != global_user_id:
        raise ValueError("user cognition state owner does not match the row")
    db = await get_db()
    result = await db.user_profiles.update_one(
        {"global_user_id": global_user_id},
        {"$set": {"cognition_state": validated_state}},
        upsert=False,
    )
    if result.matched_count != 1:
        raise DatabaseOperationError(
            f"user profile {global_user_id!r} does not exist"
        )


# ── Affinity & relationship insight ────────────────────────────────


async def get_affinity(global_user_id: str) -> int:
    """Return the affinity score for a user (0–1000, default ``AFFINITY_DEFAULT``)."""
    db = await get_db()
    doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    if doc is None:
        return AFFINITY_DEFAULT
    affinity = doc["affinity"]
    return affinity


async def update_affinity(global_user_id: str, delta: int) -> int:
    """Apply a delta to the user's affinity score, clamped to ``[AFFINITY_MIN, AFFINITY_MAX]``.

    Creates the ``user_profiles`` doc if it doesn't exist yet.

    Returns:
        The new (clamped) affinity value.
    """
    try:
        current = await get_affinity(global_user_id)
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to read affinity before update: {exc}"
        ) from exc
    new_value = max(AFFINITY_MIN, min(AFFINITY_MAX, current + delta))
    try:
        db = await get_db()
        await db.user_profiles.update_one(
            {"global_user_id": global_user_id},
            {"$set": {"affinity": new_value}},
            upsert=True,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to update affinity: {exc}"
        ) from exc
    return new_value


async def update_last_relationship_insight(global_user_id: str, insight: str) -> None:
    """Update the last relationship insight for a user."""
    try:
        db = await get_db()
        await db.user_profiles.update_one(
            {"global_user_id": global_user_id},
            {"$set": {"last_relationship_insight": insight}},
            upsert=True,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to update relationship insight: {exc}"
        ) from exc
