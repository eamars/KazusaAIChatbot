"""Public database operations used by maintenance scripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.db._client import get_db, get_text_embedding
from kazusa_ai_chatbot.db.conversation import _embedding_source_text
from kazusa_ai_chatbot.db.memory import memory_embedding_source_text
from kazusa_ai_chatbot.db.user_memory_units import _semantic_text


DERIVED_EMBEDDING_FIELD = "embedding"
USER_STATE_COLLECTIONS = (
    "user_profiles",
    "user_memory_units",
    "memory",
    "conversation_episode_state",
    "scheduled_events",
    "conversation_history",
)


async def export_collection_rows(
    *,
    collection_name: str,
    filter_doc: Mapping[str, Any],
    projection: Mapping[str, Any],
    sort_doc: Mapping[str, int],
    limit: int,
) -> list[dict[str, Any]]:
    """Export rows from an arbitrary collection for operator diagnostics."""

    db = await get_db()
    cursor = db[collection_name].find(dict(filter_doc), dict(projection))
    if sort_doc:
        cursor = cursor.sort(list(sort_doc.items()))
    cursor = cursor.limit(limit)
    records = [dict(doc) for doc in await cursor.to_list(length=limit)]
    return records


async def export_memory_rows(
    *,
    query_filter: Mapping[str, Any],
    projection: Mapping[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """Export shared-memory rows in newest-first order."""

    db = await get_db()
    cursor = (
        db.memory
        .find(dict(query_filter), dict(projection))
        .sort([("updated_at", -1), ("timestamp", -1)])
        .limit(limit)
    )
    records = [dict(doc) for doc in await cursor.to_list(length=limit)]
    return records


async def find_user_profile_for_export(
    *,
    identifier: str,
    platform: str | None,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Find a user profile by global id or platform account id."""

    db = await get_db()
    if not platform:
        profile = await db.user_profiles.find_one(
            {"global_user_id": identifier},
            dict(projection),
        )
        if profile is not None:
            return_value = dict(profile)
            return return_value

    account_filter: dict[str, Any] = {"platform_user_id": identifier}
    if platform:
        account_filter["platform"] = platform
    profile = await db.user_profiles.find_one(
        {"platform_accounts": {"$elemMatch": account_filter}},
        dict(projection),
    )
    return_value = dict(profile or {})
    return return_value


async def resolve_global_user_id_for_export(
    *,
    identifier: str,
    platform: str | None,
) -> str:
    """Resolve a global user id from global id or platform account id."""

    profile = await find_user_profile_for_export(
        identifier=identifier,
        platform=platform,
        projection={"_id": 0, "global_user_id": 1},
    )
    return_value = str(profile.get("global_user_id", ""))
    return return_value


async def export_raw_user_memory_units(
    *,
    global_user_id: str,
    include_inactive: bool,
    projection: Mapping[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """Export raw user-memory-unit rows for one user."""

    db = await get_db()
    query: dict[str, Any] = {"global_user_id": global_user_id}
    if not include_inactive:
        query["status"] = "active"
    cursor = (
        db.user_memory_units
        .find(query, dict(projection))
        .sort([("last_seen_at", -1), ("updated_at", -1)])
        .limit(limit)
    )
    records = [dict(doc) for doc in await cursor.to_list(length=limit)]
    return records


async def load_character_state_snapshot_document() -> dict[str, Any]:
    """Load the singleton character-state document for local snapshotting."""

    db = await get_db()
    document = await db.character_state.find_one({"_id": "global"})
    if document is None:
        raise ValueError('character_state document "_id: global" does not exist')
    return_value = dict(document)
    return return_value


async def replace_character_state_snapshot_document(
    document: Mapping[str, Any],
) -> None:
    """Replace the singleton character-state document from a local snapshot."""

    db = await get_db()
    await db.character_state.replace_one(
        {"_id": "global"},
        dict(document),
        upsert=True,
    )


async def refresh_conversation_history_embeddings(
    *,
    batch_size: int,
) -> dict[str, int]:
    """Regenerate conversation-history embeddings for all stored rows."""

    db = await get_db()
    collection = db.conversation_history
    query: dict[str, Any] = {}
    total_count = await collection.count_documents(query)
    processed = 0
    failed = 0
    cursor = collection.find(query).batch_size(batch_size)
    async for doc in cursor:
        content = doc.get("body_text")
        if not isinstance(content, str) or not content.strip():
            content = doc.get("content", "")
        if not isinstance(content, str) or not content.strip():
            failed += 1
            continue
        embedding = await get_text_embedding(content)
        await collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"embedding": embedding}},
        )
        processed += 1

    result = {
        "total_count": total_count,
        "processed": processed,
        "failed": failed,
    }
    return result


async def count_legacy_conversation_history_rows() -> int:
    """Count conversation-history rows matching a maintenance selector."""

    db = await get_db()
    count = await db.conversation_history.count_documents(
        _legacy_conversation_query()
    )
    return count


async def list_legacy_conversation_history_rows(
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Load one deterministic batch of conversation-history rows."""

    db = await get_db()
    cursor = (
        db.conversation_history
        .find(_legacy_conversation_query())
        .sort("timestamp", 1)
        .limit(batch_size)
    )
    rows = [dict(row) for row in await cursor.to_list(length=batch_size)]
    return rows


async def update_conversation_history_row(
    *,
    row_id: Any,
    set_fields: Mapping[str, Any],
    unset_fields: Sequence[str],
) -> None:
    """Apply one maintenance update to a conversation-history row."""

    update: dict[str, Any] = {"$set": dict(set_fields)}
    if unset_fields:
        update["$unset"] = {
            field_name: ""
            for field_name in unset_fields
        }
    db = await get_db()
    await db.conversation_history.update_one({"_id": row_id}, update)


async def drop_legacy_rag_collections(
    collection_names: Sequence[str],
) -> list[str]:
    """Drop legacy RAG collections that still exist."""

    db = await get_db()
    existing = set(await db.list_collection_names())
    dropped: list[str] = []
    for collection_name in collection_names:
        if collection_name not in existing:
            continue
        await db.drop_collection(collection_name)
        dropped.append(collection_name)
    return dropped


async def load_user_state_documents(
    *,
    filters: Mapping[str, Mapping[str, Any]],
    sort_specs: Mapping[str, Sequence[tuple[str, int]]],
    collection_names: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Load all user-scoped snapshot documents grouped by collection."""

    db = await get_db()
    documents: dict[str, list[dict[str, Any]]] = {}
    for collection_name in collection_names:
        cursor = db[collection_name].find(dict(filters[collection_name]))
        cursor = cursor.sort(list(sort_specs[collection_name]))
        rows = [dict(doc) for doc in await cursor.to_list(length=None)]
        documents[collection_name] = rows
    return documents


async def load_user_state_snapshot_documents(
    *,
    global_user_id: str,
    platform_accounts: Sequence[Mapping[str, str]],
    collection_names: Sequence[str] = USER_STATE_COLLECTIONS,
) -> dict[str, list[dict[str, Any]]]:
    """Load all user-scoped snapshot documents grouped by collection."""

    filters = _user_state_collection_filters(global_user_id, platform_accounts)
    sort_specs = {
        collection_name: _user_state_sort_spec(collection_name)
        for collection_name in collection_names
    }
    documents = await load_user_state_documents(
        filters=filters,
        sort_specs=sort_specs,
        collection_names=collection_names,
    )
    return documents


async def load_user_state_alias_profile_refs(
    global_user_id: str,
) -> list[dict[str, Any]]:
    """Load profile alias backlinks that point at one user."""

    db = await get_db()
    cursor = (
        db.user_profiles
        .find(
            {
                "global_user_id": {"$ne": global_user_id},
                "suspected_aliases": global_user_id,
            },
            {
                "_id": 0,
                "global_user_id": 1,
                "suspected_aliases": 1,
            },
        )
        .sort("global_user_id", 1)
    )
    rows = [dict(doc) for doc in await cursor.to_list(length=None)]
    return rows


async def recalculate_user_state_embeddings(
    *,
    collection_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach fresh embeddings to restored snapshot rows when needed."""

    restored_rows = [dict(row) for row in rows]
    if collection_name not in {
        "conversation_history",
        "memory",
        "user_memory_units",
    }:
        return restored_rows

    for row in restored_rows:
        row.pop(DERIVED_EMBEDDING_FIELD, None)
        if collection_name == "conversation_history":
            embedding_text = _embedding_source_text(row)
        elif collection_name == "memory":
            embedding_text = memory_embedding_source_text(row)
        else:
            embedding_text = _semantic_text(row)
        row[DERIVED_EMBEDDING_FIELD] = await get_text_embedding(embedding_text)
    return restored_rows


async def replace_user_state_collection_rows(
    *,
    collection_name: str,
    filter_doc: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Replace one collection's rows inside a user-state restore."""

    db = await get_db()
    await db[collection_name].delete_many(dict(filter_doc))
    row_dicts = [dict(row) for row in rows]
    if row_dicts:
        await db[collection_name].insert_many(row_dicts)
    row_count = len(row_dicts)
    return row_count


async def replace_user_state_snapshot_collection_rows(
    *,
    global_user_id: str,
    platform_accounts: Sequence[Mapping[str, str]],
    collection_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Replace one collection's rows for a user-state snapshot restore."""

    filters = _user_state_collection_filters(global_user_id, platform_accounts)
    row_count = await replace_user_state_collection_rows(
        collection_name=collection_name,
        filter_doc=filters[collection_name],
        rows=rows,
    )
    return row_count


async def restore_user_state_alias_profile_refs(
    *,
    global_user_id: str,
    alias_refs: Sequence[Mapping[str, Any]],
) -> None:
    """Restore alias backlinks from a user-state snapshot."""

    db = await get_db()
    await db.user_profiles.update_many(
        {
            "global_user_id": {"$ne": global_user_id},
            "suspected_aliases": global_user_id,
        },
        {"$pull": {"suspected_aliases": global_user_id}},
    )
    for alias_ref in alias_refs:
        alias_global_user_id = str(alias_ref.get("global_user_id", "")).strip()
        if not alias_global_user_id:
            continue
        aliases = alias_ref.get("suspected_aliases")
        if not isinstance(aliases, list):
            aliases = []
        await db.user_profiles.update_one(
            {"global_user_id": alias_global_user_id},
            {"$set": {"suspected_aliases": aliases}},
            upsert=False,
        )


async def scan_active_user_memory_units_for_perspective(
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Load active user-memory-unit rows for perspective review."""

    db = await get_db()
    cursor = (
        db.user_memory_units
        .find({"status": "active"}, {"embedding": 0})
        .limit(limit)
    )
    rows = [dict(doc) for doc in await cursor.to_list(length=limit)]
    return rows


async def scan_user_profiles_for_perspective(
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Load user profiles that have prompt-facing relationship insight."""

    db = await get_db()
    cursor = (
        db.user_profiles
        .find({"last_relationship_insight": {"$ne": ""}}, {"_id": 0})
        .limit(limit)
    )
    rows = [dict(doc) for doc in await cursor.to_list(length=limit)]
    return rows


async def scan_persistent_memory_for_perspective(
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Load active reflection-promoted memory rows for perspective review."""

    db = await get_db()
    cursor = (
        db.memory
        .find(
            {
                "status": "active",
                "authority": "reflection_promoted",
            },
            {"embedding": 0},
        )
        .limit(limit)
    )
    rows = [dict(doc) for doc in await cursor.to_list(length=limit)]
    return rows


async def find_persistent_memory_without_embedding(
    memory_unit_id: str,
) -> dict[str, Any]:
    """Load one persistent-memory row without its vector field."""

    db = await get_db()
    existing = await db.memory.find_one(
        {"memory_unit_id": memory_unit_id},
        {"embedding": 0},
    )
    return_value = dict(existing or {})
    return return_value


def _legacy_conversation_query() -> dict[str, Any]:
    """Build the selector for rows outside the typed storage contract."""

    query: dict[str, Any] = {
        "$or": [
            {"content": {"$exists": True}},
            {"body_text": {"$exists": False}},
            {"raw_wire_text": {"$exists": False}},
            {"addressed_to_global_user_ids": {"$exists": False}},
            {"mentions": {"$exists": False}},
            {"broadcast": {"$exists": False}},
            {"attachments": {"$exists": False}},
        ]
    }
    return query


def _user_state_conversation_history_filter(
    global_user_id: str,
    platform_accounts: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Build the user-related conversation-history snapshot filter."""

    conditions: list[dict[str, Any]] = [
        {"global_user_id": global_user_id},
        {"addressed_to_global_user_ids": global_user_id},
        {"target_addressed_user_ids": global_user_id},
        {"mentions.global_user_id": global_user_id},
    ]
    for account in platform_accounts:
        platform = account["platform"]
        platform_user_id = account["platform_user_id"]
        conditions.extend([
            {
                "platform": platform,
                "platform_user_id": platform_user_id,
            },
            {
                "platform": platform,
                "reply_context.reply_to_platform_user_id": platform_user_id,
            },
            {
                "platform": platform,
                "mentions.platform_user_id": platform_user_id,
            },
        ])

    filter_doc = {"$or": conditions}
    return filter_doc


def _user_state_collection_filters(
    global_user_id: str,
    platform_accounts: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    """Build snapshot filters for authoritative user-state rows."""

    filters = {
        "user_profiles": {"global_user_id": global_user_id},
        "user_memory_units": {"global_user_id": global_user_id},
        "memory": {"source_global_user_id": global_user_id},
        "conversation_episode_state": {"global_user_id": global_user_id},
        "scheduled_events": {"source_user_id": global_user_id},
        "conversation_history": _user_state_conversation_history_filter(
            global_user_id,
            platform_accounts,
        ),
    }
    return filters


def _user_state_sort_spec(collection_name: str) -> list[tuple[str, int]]:
    """Return a stable sort order for snapshot readability."""

    sort_specs = {
        "user_profiles": [("global_user_id", 1)],
        "user_memory_units": [("unit_id", 1), ("updated_at", 1)],
        "memory": [("timestamp", 1), ("memory_name", 1)],
        "conversation_episode_state": [
            ("platform", 1),
            ("platform_channel_id", 1),
            ("updated_at", 1),
        ],
        "scheduled_events": [("execute_at", 1), ("event_id", 1)],
        "conversation_history": [("timestamp", 1), ("platform_message_id", 1)],
    }
    return_value = sort_specs[collection_name]
    return return_value
