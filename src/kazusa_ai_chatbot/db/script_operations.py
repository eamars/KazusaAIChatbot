"""Public database operations used by maintenance scripts."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.db._client import (
    build_vector_search_index_model,
    get_document_text_embedding,
    get_document_text_embeddings_batch,
    get_db,
    vector_index_definition_issues,
    vector_index_missing_filter_paths,
)
from kazusa_ai_chatbot.db.conversation import (
    CONVERSATION_VECTOR_FILTER_FIELDS,
    CONVERSATION_VECTOR_INDEX_NAME,
    _embedding_source_text,
    reset_conversation_vector_prefilter_support_cache,
)
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
VECTOR_SEARCH_INDEX_CONFIGS = {
    "conversation_history": {
        "collection": "conversation_history",
        "index_name": CONVERSATION_VECTOR_INDEX_NAME,
        "path": "embedding",
        "filter_paths": list(CONVERSATION_VECTOR_FILTER_FIELDS),
    }
}
VECTOR_INDEX_READY_STATUSES = {"READY", "STEADY", "QUERYABLE"}
TEXT_VECTOR_REEMBEDDING_COLLECTIONS = (
    "conversation_history",
    "memory",
    "user_memory_units",
)
SYNTHETIC_CONSOLIDATION_USER_ID = "self_cognition"
SYNTHETIC_CONSOLIDATION_CLEANUP_REASON = (
    "synthetic_consolidation_user_cleanup"
)
SYNTHETIC_CONSOLIDATION_USER_COUNT_KEYS = (
    "synthetic_user_profiles",
    "synthetic_scheduled_events",
    "synthetic_user_memory_units",
)


def _consolidation_target_lifecycle_filters(
    synthetic_user_id: str,
) -> dict[str, dict[str, Any]]:
    """Build exact selectors for synthetic consolidation lifecycle rows."""

    filters = {
        "synthetic_user_profiles": {
            "global_user_id": synthetic_user_id,
        },
        "user_profiles_missing_affinity": {
            "affinity": {"$exists": False},
        },
        "synthetic_scheduled_events": {
            "source_user_id": synthetic_user_id,
        },
        "synthetic_user_memory_units": {
            "global_user_id": synthetic_user_id,
        },
        "future_cognition_attempts_missing_user": {
            "action_kind": "trigger_future_cognition",
            "$or": [
                {"target_scope.scope.source_user_id": {"$exists": False}},
                {"target_scope.scope.source_user_id": ""},
                {"target_scope.scope.source_user_id": None},
            ],
        },
        "synthetic_user_profiles_with_platform_accounts": {
            "global_user_id": synthetic_user_id,
            "platform_accounts.0": {"$exists": True},
        },
    }
    return filters


def _planned_consolidation_target_lifecycle_apply(
    *,
    filters: Mapping[str, Mapping[str, Any]],
    synthetic_user_id: str,
) -> dict[str, Any]:
    """Describe approved maintenance mutations without runtime timestamps."""

    planned_apply = {
        "scheduled_events": {
            "operation": "update_many",
            "filter": dict(filters["synthetic_scheduled_events"]),
            "set_fields": {
                "status": "failed",
                "migration_reason": SYNTHETIC_CONSOLIDATION_CLEANUP_REASON,
                "migration_original_source_user_id": synthetic_user_id,
            },
            "unset_fields": ["source_user_id"],
            "runtime_fields": ["migration_applied_at"],
        },
        "user_profiles": {
            "operation": "delete_many",
            "filter": dict(filters["synthetic_user_profiles"]),
        },
        "user_memory_units": {
            "operation": "delete_many",
            "filter": dict(filters["synthetic_user_memory_units"]),
        },
    }
    return planned_apply


def _scheduled_event_synthetic_cleanup_update(
    *,
    synthetic_user_id: str,
    storage_timestamp_utc: str,
) -> dict[str, Any]:
    """Build the maintenance update that removes synthetic user ownership."""

    update_doc = {
        "$set": {
            "status": "failed",
            "migration_reason": SYNTHETIC_CONSOLIDATION_CLEANUP_REASON,
            "migration_applied_at": storage_timestamp_utc,
            "migration_original_source_user_id": synthetic_user_id,
        },
        "$unset": {"source_user_id": ""},
    }
    return update_doc


def _synthetic_consolidation_user_owned_count(
    counts: Mapping[str, int],
) -> int:
    """Count rows still owned by the forbidden synthetic user id."""

    total_count = sum(
        counts[key]
        for key in SYNTHETIC_CONSOLIDATION_USER_COUNT_KEYS
    )
    return total_count


def _vector_search_index_config(collection_name: str) -> dict[str, Any]:
    """Return the approved vector-search index config for one collection."""

    config = VECTOR_SEARCH_INDEX_CONFIGS.get(collection_name)
    if config is None:
        raise ValueError(f"unsupported vector-search collection: {collection_name}")
    return_value = dict(config)
    return return_value


async def _find_search_index(
    *,
    collection_name: str,
    index_name: str,
) -> dict[str, Any] | None:
    """Find one Atlas search-index document by name."""

    db = await get_db()
    collection = db[collection_name]
    async for index in collection.list_search_indexes():
        if index.get("name") == index_name:
            return_value = dict(index)
            return return_value
    return_value = None
    return return_value


async def inspect_vector_search_index(collection_name: str) -> dict[str, Any]:
    """Inspect whether a vector-search index needs recreation."""

    config = _vector_search_index_config(collection_name)
    index_name = str(config["index_name"])
    filter_paths = list(config["filter_paths"])
    index_document = await _find_search_index(
        collection_name=collection_name,
        index_name=index_name,
    )
    if index_document is None:
        result = {
            **config,
            "status": "missing",
            "requires_recreate": True,
            "missing_filter_paths": filter_paths,
            "definition_issues": ["missing_index"],
        }
        return result

    path = str(config["path"])
    sample_embedding = await get_document_text_embedding("test")
    definition_issues = vector_index_definition_issues(
        index_document,
        path=path,
        num_dimensions=len(sample_embedding),
        required_filter_paths=filter_paths,
    )
    missing_paths = vector_index_missing_filter_paths(
        index_document,
        filter_paths,
    )
    status = "ready"
    requires_recreate = False
    if definition_issues:
        status = "definition_mismatch"
        requires_recreate = True

    result = {
        **config,
        "status": status,
        "requires_recreate": requires_recreate,
        "missing_filter_paths": missing_paths,
        "definition_issues": definition_issues,
    }
    return result


async def _wait_for_search_index_ready(
    *,
    collection_name: str,
    index_name: str,
    timeout_seconds: int = 180,
    poll_seconds: int = 5,
) -> str:
    """Wait for Atlas to report a recreated search index as queryable."""

    elapsed_seconds = 0
    status = "UNKNOWN"
    while elapsed_seconds <= timeout_seconds:
        index_document = await _find_search_index(
            collection_name=collection_name,
            index_name=index_name,
        )
        if index_document is not None:
            raw_status = index_document.get("status")
            if isinstance(raw_status, str) and raw_status:
                status = raw_status
            if status.upper() in VECTOR_INDEX_READY_STATUSES:
                return_value = status
                return return_value

        await asyncio.sleep(poll_seconds)
        elapsed_seconds += poll_seconds

    raise TimeoutError(
        f"search index {index_name!r} on {collection_name!r} "
        f"was not ready after {timeout_seconds}s; last status={status!r}"
    )


async def apply_vector_search_index(
    collection_name: str,
    *,
    wait_ready: bool,
) -> dict[str, Any]:
    """Recreate one approved vector-search index with required filters."""

    config = _vector_search_index_config(collection_name)
    index_name = str(config["index_name"])
    path = str(config["path"])
    filter_paths = list(config["filter_paths"])
    db = await get_db()
    collection = db[collection_name]
    existing_index = await _find_search_index(
        collection_name=collection_name,
        index_name=index_name,
    )
    dropped_existing = existing_index is not None

    sample_embedding = await get_document_text_embedding("test")
    search_index_model = build_vector_search_index_model(
        index_name=index_name,
        path=path,
        num_dimensions=len(sample_embedding),
        filter_paths=filter_paths,
    )
    if dropped_existing:
        await collection.drop_search_index(index_name)

    await collection.create_search_index(search_index_model)

    ready_status = ""
    if wait_ready:
        ready_status = await _wait_for_search_index_ready(
            collection_name=collection_name,
            index_name=index_name,
        )

    reset_conversation_vector_prefilter_support_cache()
    result = {
        **config,
        "status": "applied",
        "ready_status": ready_status,
        "requires_recreate": False,
        "missing_filter_paths": [],
        "dropped_existing": dropped_existing,
    }
    return result


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


async def inspect_consolidation_target_lifecycle() -> dict[str, Any]:
    """Count rows affected by forbidden synthetic consolidation user ids.

    Returns:
        Read-only diagnostic counts and the exact filters operators should
        review before approving any cleanup action.
    """

    synthetic_user_id = SYNTHETIC_CONSOLIDATION_USER_ID
    filters = _consolidation_target_lifecycle_filters(synthetic_user_id)

    db = await get_db()
    counts = {
        "synthetic_user_profiles": await db.user_profiles.count_documents(
            filters["synthetic_user_profiles"],
        ),
        "user_profiles_missing_affinity": (
            await db.user_profiles.count_documents(
                filters["user_profiles_missing_affinity"],
            )
        ),
        "synthetic_scheduled_events": (
            await db.scheduled_events.count_documents(
                filters["synthetic_scheduled_events"],
            )
        ),
        "synthetic_user_memory_units": (
            await db.user_memory_units.count_documents(
                filters["synthetic_user_memory_units"],
            )
        ),
        "future_cognition_attempts_missing_user": (
            await db.self_cognition_action_attempts.count_documents(
                filters["future_cognition_attempts_missing_user"],
            )
        ),
        "synthetic_user_profiles_with_platform_accounts": (
            await db.user_profiles.count_documents(
                filters["synthetic_user_profiles_with_platform_accounts"],
            )
        ),
    }
    cleanup_blocked = (
        counts["synthetic_user_profiles_with_platform_accounts"] > 0
    )
    planned_apply_status = "available"
    if cleanup_blocked:
        planned_apply_status = "blocked"

    result = {
        "synthetic_user_id": synthetic_user_id,
        "mode": "dry_run",
        "counts": counts,
        "filters": filters,
        "cleanup_blocked": cleanup_blocked,
        "planned_apply_status": planned_apply_status,
        "planned_apply": _planned_consolidation_target_lifecycle_apply(
            filters=filters,
            synthetic_user_id=synthetic_user_id,
        ),
    }
    return result


async def apply_consolidation_target_lifecycle_cleanup(
    *,
    storage_timestamp_utc: str,
) -> dict[str, Any]:
    """Apply approved cleanup for forbidden synthetic consolidation user rows.

    Args:
        storage_timestamp_utc: UTC storage timestamp recorded on migrated
            scheduled-event rows.

    Returns:
        Sanitized before and after counts, write result counts, and safety
        status for the operator-approved maintenance action.
    """

    if not storage_timestamp_utc.strip():
        raise ValueError("storage_timestamp_utc is required")

    before_report = await inspect_consolidation_target_lifecycle()
    before_counts = before_report["counts"]
    filters: dict[str, dict[str, Any]] = before_report["filters"]
    synthetic_user_id = str(before_report["synthetic_user_id"])
    if before_report["cleanup_blocked"]:
        result = {
            "synthetic_user_id": synthetic_user_id,
            "mode": "apply",
            "apply_status": "blocked",
            "blocked_reason": "synthetic_profile_has_platform_accounts",
            "before_counts": before_counts,
            "after_counts": before_counts,
            "cleanup_blocked": True,
            "synthetic_user_owned_rows_after": (
                _synthetic_consolidation_user_owned_count(before_counts)
            ),
            "applied": {
                "scheduled_events_modified": 0,
                "synthetic_user_profiles_deleted": 0,
                "synthetic_user_memory_units_deleted": 0,
            },
            "filters": filters,
            "planned_apply": before_report["planned_apply"],
        }
        return result

    scheduled_event_update = _scheduled_event_synthetic_cleanup_update(
        synthetic_user_id=synthetic_user_id,
        storage_timestamp_utc=storage_timestamp_utc,
    )
    db = await get_db()
    scheduled_result = await db.scheduled_events.update_many(
        filters["synthetic_scheduled_events"],
        scheduled_event_update,
    )
    profile_result = await db.user_profiles.delete_many(
        filters["synthetic_user_profiles"],
    )
    memory_result = await db.user_memory_units.delete_many(
        filters["synthetic_user_memory_units"],
    )

    after_report = await inspect_consolidation_target_lifecycle()
    after_counts = after_report["counts"]
    result = {
        "synthetic_user_id": synthetic_user_id,
        "mode": "apply",
        "apply_status": "applied",
        "blocked_reason": "",
        "before_counts": before_counts,
        "after_counts": after_counts,
        "cleanup_blocked": after_report["cleanup_blocked"],
        "synthetic_user_owned_rows_after": (
            _synthetic_consolidation_user_owned_count(after_counts)
        ),
        "applied": {
            "scheduled_events_modified": scheduled_result.modified_count,
            "synthetic_user_profiles_deleted": profile_result.deleted_count,
            "synthetic_user_memory_units_deleted": memory_result.deleted_count,
        },
        "filters": filters,
        "scheduled_events_update": scheduled_event_update,
        "planned_apply": before_report["planned_apply"],
    }
    return result


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
        embedding = await get_document_text_embedding(content)
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


def _reembedding_source_text(collection_name: str, row: Mapping[str, Any]) -> str:
    """Build the stored-document source text for an approved collection."""

    row_copy = dict(row)
    if collection_name == "conversation_history":
        source_text = _embedding_source_text(row_copy)
    elif collection_name == "memory":
        source_text = memory_embedding_source_text(row_copy)
    elif collection_name == "user_memory_units":
        source_text = _semantic_text(row_copy)
    else:
        raise ValueError(f"unsupported re-embedding collection: {collection_name}")
    return source_text


def _skipped_reembedding_row(row: Mapping[str, Any]) -> dict[str, str]:
    """Build the operator report entry for a row with no source text."""

    row_id = str(row.get("_id", ""))
    skipped_row = {"row_id": row_id, "reason": "empty_source_text"}
    return skipped_row


async def _flush_reembedding_batch(
    *,
    collection: Any,
    rows: list[Mapping[str, Any]],
    source_texts: list[str],
    apply: bool,
) -> int:
    """Embed and update one prepared batch of non-empty source rows."""

    if not apply or not rows:
        return_value = 0
        return return_value

    embeddings = await get_document_text_embeddings_batch(list(source_texts))
    updated = 0
    for row, embedding in zip(rows, embeddings, strict=True):
        await collection.update_one(
            {"_id": row["_id"]},
            {"$set": {DERIVED_EMBEDDING_FIELD: embedding}},
        )
        updated += 1
    return updated


async def reembed_text_vector_collection(
    *,
    collection_name: str,
    batch_size: int,
    apply: bool,
) -> dict[str, Any]:
    """Dry-run or apply document-role embeddings for one approved collection."""

    if collection_name not in TEXT_VECTOR_REEMBEDDING_COLLECTIONS:
        raise ValueError(f"unsupported re-embedding collection: {collection_name}")

    db = await get_db()
    collection = db[collection_name]
    query: dict[str, Any] = {}
    total_count = await collection.count_documents(query)
    processed = 0
    skipped_rows: list[dict[str, str]] = []
    updated = 0
    cleared = 0
    pending_rows: list[Mapping[str, Any]] = []
    pending_texts: list[str] = []
    cursor = collection.find(query).batch_size(batch_size)
    async for row in cursor:
        source_text = _reembedding_source_text(collection_name, row).strip()
        if not source_text:
            skipped_rows.append(_skipped_reembedding_row(row))
            if apply:
                await collection.update_one(
                    {"_id": row["_id"]},
                    {"$unset": {DERIVED_EMBEDDING_FIELD: ""}},
                )
                cleared += 1
            continue

        processed += 1
        pending_rows.append(dict(row))
        pending_texts.append(source_text)
        if len(pending_rows) >= batch_size:
            updated += await _flush_reembedding_batch(
                collection=collection,
                rows=pending_rows,
                source_texts=pending_texts,
                apply=apply,
            )
            pending_rows = []
            pending_texts = []

    updated += await _flush_reembedding_batch(
        collection=collection,
        rows=pending_rows,
        source_texts=pending_texts,
        apply=apply,
    )

    result = {
        "collection": collection_name,
        "total_count": total_count,
        "processed": processed,
        "skipped": len(skipped_rows),
        "updated": updated,
        "cleared": cleared,
        "skipped_rows": skipped_rows,
    }
    return result


async def reembed_text_vector_embeddings(
    *,
    collection_names: Sequence[str],
    batch_size: int,
    apply: bool,
) -> dict[str, Any]:
    """Dry-run or apply document-role re-embedding for approved collections."""

    collection_results: list[dict[str, Any]] = []
    for collection_name in collection_names:
        collection_result = await reembed_text_vector_collection(
            collection_name=collection_name,
            batch_size=batch_size,
            apply=apply,
        )
        collection_results.append(collection_result)

    result = {
        "apply": apply,
        "batch_size": batch_size,
        "collections": collection_results,
        "total_count": sum(row["total_count"] for row in collection_results),
        "total_processed": sum(row["processed"] for row in collection_results),
        "total_skipped": sum(row["skipped"] for row in collection_results),
        "total_updated": sum(row["updated"] for row in collection_results),
        "total_cleared": sum(row["cleared"] for row in collection_results),
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
        row[DERIVED_EMBEDDING_FIELD] = await get_document_text_embedding(
            embedding_text
        )
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
