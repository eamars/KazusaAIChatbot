"""Public database operations used by maintenance scripts."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from bson import json_util
from bson.json_util import RELAXED_JSON_OPTIONS
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import AUDIT_LOG_TTL_DAYS, DEBUG_LOG_TTL_DAYS
from kazusa_ai_chatbot.cognition_shared.state_models import (
    validate_cognition_state,
)
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
from kazusa_ai_chatbot.logging_retention import (
    expiry_from_now,
    expiry_from_storage_iso,
)
from kazusa_ai_chatbot.db.memory import memory_embedding_source_text
from kazusa_ai_chatbot.db.user_memory_units import _semantic_text
from kazusa_ai_chatbot.time_boundary import (
    parse_storage_utc_datetime,
    storage_utc_now,
)
from kazusa_ai_chatbot.conversation_progress.policy import storage_expiry
from kazusa_ai_chatbot.llm_tracing.correlation import (
    MAX_COMPANION_CANDIDATES,
    MAX_CONVERSATION_ROWS,
    MAX_FAILURE_CAPSULES,
    MAX_PARENT_CANDIDATES,
    CorrelationSourceSurface,
    normalize_source_surface,
)


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
SEMANTIC_IDENTITY_FORBIDDEN_PATTERN = (
    r"(@mentioned-(?:user|role|entity)-\d+|#mentioned-channel-\d+|"
    r"(?<![A-Za-z0-9_])[@#]?(?:qq|discord|platform)-"
    r"(?:user|bot|role|channel|entity):[^\s]+|"
    r"\[CQ:|<@!?\d+>|<@&\d+>|<#\d+>|<a?:[A-Za-z0-9_]+:\d+>)"
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
CALENDAR_MIGRATION_PENDING_STATUS = "pending"
CALENDAR_MIGRATION_FUTURE_COGNITION_TOOL = "trigger_future_cognition"
CALENDAR_MIGRATION_SEND_MESSAGE_TOOL = "send_message"
LOGGING_RETENTION_TARGETS = (
    ("event_log_events", "occurred_at", AUDIT_LOG_TTL_DAYS),
    ("event_log_snapshots", "generated_at", AUDIT_LOG_TTL_DAYS),
    ("llm_trace_runs", "started_at", DEBUG_LOG_TTL_DAYS),
    ("llm_trace_steps", "created_at", DEBUG_LOG_TTL_DAYS),
)


async def count_dsh_plan3_drain_rows(database: Any) -> dict[str, int]:
    """Count legacy task, job, and open-interaction rows for Plan 3 drain."""

    accepted_tasks = await database["accepted_tasks"].count_documents({
        "schema_version": "accepted_task.v2",
        "task_kind": {"$in": ["task_resolution", "coding_continuation"]},
        "state": {
            "$in": [
                "enqueueing",
                "pending",
                "running",
                "result_ready",
                "failure_ready",
                "delivery_in_progress",
                "delivery_retryable",
            ]
        },
    })
    executing_jobs = await database["background_work_jobs"].count_documents({
        "schema_version": "background_work_job.v2",
        "requested_worker": "task_orchestrator",
        "worker_payload.schema_version": "task_orchestrator_worker_payload.v1",
        "status": {"$in": ["queued", "in_progress"]},
    })
    undelivered_jobs = await database["background_work_jobs"].count_documents({
        "schema_version": "background_work_job.v2",
        "requested_worker": "task_orchestrator",
        "worker_payload.schema_version": "task_orchestrator_worker_payload.v1",
        "status": {
            "$in": [
                "completed",
                "failed",
                "delivery_failed",
                "delivery_in_progress",
            ]
        },
        "delivery_state": {"$ne": "delivered"},
    })
    open_interactions = await database["dsh_interactions"].count_documents({
        "schema_version": "dsh_interaction_pending.v1",
        "$or": [
            {"status": {"$in": ["pending", "delivered", "continuation_pending"]}},
            {"grant_status": "available"},
        ],
    })
    return {
        "active_legacy_accepted_tasks": int(accepted_tasks),
        "executing_legacy_task_jobs": int(executing_jobs),
        "undelivered_legacy_task_jobs": int(undelivered_jobs),
        "open_pre_cutover_dsh_interactions": int(open_interactions),
    }


def _logging_row_expiry(
    row: Mapping[str, Any],
    *,
    timestamp_field: str,
    ttl_days: int,
) -> datetime:
    """Compute expiry for one legacy logging row."""

    raw_timestamp = row.get(timestamp_field)
    if isinstance(raw_timestamp, datetime):
        if raw_timestamp.tzinfo is None:
            source = raw_timestamp.replace(tzinfo=timezone.utc)
        else:
            source = raw_timestamp.astimezone(timezone.utc)
        expires_at = source + timedelta(days=ttl_days)
        return expires_at
    if isinstance(raw_timestamp, str) and raw_timestamp.strip():
        expires_at = expiry_from_storage_iso(raw_timestamp, ttl_days=ttl_days)
        return expires_at
    expires_at = expiry_from_now(ttl_days=ttl_days)
    return expires_at


async def apply_logging_retention(
    *,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Assign or delete legacy logging rows according to TTL policy."""

    db = await get_db()
    now = storage_utc_now()
    collections: list[dict[str, Any]] = []
    for collection_name, timestamp_field, ttl_days in LOGGING_RETENTION_TARGETS:
        collection = db[collection_name]
        missing_filter = {
            "$or": [
                {"expires_at": {"$exists": False}},
                {"expires_at": None},
            ]
        }
        total_missing = await collection.count_documents(missing_filter)
        expired_ids = []
        update_rows = []
        scanned = 0
        cursor = collection.find(missing_filter).batch_size(batch_size)
        async for row in cursor:
            scanned += 1
            expires_at = _logging_row_expiry(
                row,
                timestamp_field=timestamp_field,
                ttl_days=ttl_days,
            )
            if expires_at <= now:
                expired_ids.append(row["_id"])
            else:
                update_rows.append((row["_id"], expires_at))

        deleted = 0
        updated = 0
        if not dry_run:
            if expired_ids:
                delete_result = await collection.delete_many(
                    {"_id": {"$in": expired_ids}},
                )
                deleted = int(delete_result.deleted_count)
            for row_id, expires_at in update_rows:
                update_result = await collection.update_one(
                    {"_id": row_id},
                    {"$set": {"expires_at": expires_at}},
                )
                updated += int(update_result.modified_count)

        collections.append({
            "collection": collection_name,
            "timestamp_field": timestamp_field,
            "ttl_days": ttl_days,
            "missing_expires_at": total_missing,
            "scanned": scanned,
            "would_delete": len(expired_ids),
            "would_update": len(update_rows),
            "deleted": deleted,
            "updated": updated,
        })

    report = {
        "mode": "dry_run" if dry_run else "apply",
        "batch_size": batch_size,
        "collections": collections,
    }
    return report


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


_CORRELATION_SOURCE_FIELDS = {
    "web_control_trace_id": (
        "llm_trace_runs",
        {"trace_id": 1},
        {"trace_id": 1},
    ),
    "protected_llm_trace_id": (
        "llm_trace_runs",
        {"trace_id": 1},
        {"trace_id": 1},
    ),
    "protected_cognition_invocation_id": (
        "llm_trace_steps",
        {},
        {
            "trace_id": 1,
            "cognition_invocation_id": 1,
            "capsule.cognition_invocation_id": 1,
        },
    ),
    "protected_global_user_id": (
        "llm_trace_runs",
        {"global_user_id": 1},
        {"trace_id": 1, "global_user_id": 1},
    ),
    "protected_action_attempt_id": (
        "self_cognition_action_attempts",
        {"attempt_id": 1},
        {
            "source_llm_trace_id": 1,
            "attempt_id": 1,
            "correlation_write_status": 1,
            "correlation_conflict_source_llm_trace_id": 1,
        },
    ),
    "protected_background_work_job_id": (
        "background_work_jobs",
        {"job_id": 1},
        {
            "source_llm_trace_id": 1,
            "job_id": 1,
            "correlation_write_status": 1,
            "correlation_conflict_source_llm_trace_id": 1,
        },
    ),
    "protected_accepted_task_id": (
        "background_work_jobs",
        {"accepted_task_id": 1},
        {
            "source_llm_trace_id": 1,
            "accepted_task_id": 1,
            "correlation_write_status": 1,
            "correlation_conflict_source_llm_trace_id": 1,
        },
    ),
    "protected_calendar_schedule_id": (
        "calendar_schedules",
        {"schedule_id": 1},
        {
            "source_llm_trace_id": 1,
            "schedule_id": 1,
            "correlation_write_status": 1,
            "correlation_conflict_source_llm_trace_id": 1,
        },
    ),
    "protected_calendar_run_id": (
        "calendar_runs",
        {"run_id": 1},
        {
            "source_llm_trace_id": 1,
            "run_id": 1,
            "correlation_write_status": 1,
            "correlation_conflict_source_llm_trace_id": 1,
        },
    ),
}


def _correlation_identifier_filter(
    source_surface: CorrelationSourceSurface,
    identifier: str,
) -> dict[str, Any]:
    """Build the exact identifier filter for one protected source surface."""

    if source_surface == "protected_cognition_invocation_id":
        return {
            "$or": [
                {"cognition_invocation_id": identifier},
                {"capsule.cognition_invocation_id": identifier},
            ]
        }
    spec = _CORRELATION_SOURCE_FIELDS.get(source_surface)
    if spec is None:
        raise ValueError(f"unsupported correlation source: {source_surface}")
    filter_fields = spec[1]
    if not filter_fields:
        raise ValueError(f"source requires a dedicated filter: {source_surface}")
    return {field: identifier for field in filter_fields}


async def list_trace_correlation_candidates(
    *,
    source_surface: str,
    identifier: str,
    limit: int = MAX_PARENT_CANDIDATES,
) -> list[dict[str, Any]]:
    """Read bounded exact candidates for one typed protected source."""

    normalized_surface = normalize_source_surface(source_surface)
    if normalized_surface == "unknown":
        return []
    if limit < 1 or limit > MAX_PARENT_CANDIDATES:
        raise ValueError("parent correlation limit is outside the fixed cap")
    spec = _CORRELATION_SOURCE_FIELDS.get(normalized_surface)
    if spec is None:
        raise ValueError(f"unsupported correlation source: {normalized_surface}")
    collection_name, _filter_fields, projection = spec
    filter_doc = _correlation_identifier_filter(
        normalized_surface,
        identifier.strip(),
    )
    return await export_collection_rows(
        collection_name=collection_name,
        filter_doc=filter_doc,
        projection={"_id": 0, **projection},
        sort_doc={"trace_id": 1, "source_llm_trace_id": 1},
        limit=limit,
    )


async def list_trace_correlation_companions(
    *,
    trace_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Read bounded identifier-only companion rows for one parent trace."""

    clean_trace_id = trace_id.strip()
    if not clean_trace_id:
        return {}
    rows_by_kind: dict[str, list[dict[str, Any]]] = {}
    rows_by_kind["conversation_history"] = await export_collection_rows(
        collection_name="conversation_history",
        filter_doc={"llm_trace_id": clean_trace_id},
        projection={
            "_id": 0,
            "llm_trace_id": 1,
            "row_id": 1,
            "message_id": 1,
            "platform": 1,
            "platform_channel_id": 1,
            "platform_message_id": 1,
            "channel_type": 1,
            "role": 1,
            "timestamp": 1,
            "delivery_tracking_id": 1,
        },
        sort_doc={"timestamp": 1},
        limit=MAX_CONVERSATION_ROWS,
    )
    rows_by_kind["llm_trace_steps"] = await export_collection_rows(
        collection_name="llm_trace_steps",
        filter_doc={"trace_id": clean_trace_id},
        projection={
            "_id": 0,
            "trace_id": 1,
            "stage_name": 1,
            "sequence": 1,
            "capture_reason": 1,
            "cognition_invocation_id": 1,
            "capsule.cognition_invocation_id": 1,
            "created_at": 1,
        },
        sort_doc={"sequence": 1, "created_at": 1},
        limit=MAX_FAILURE_CAPSULES,
    )
    companion_specs = {
        "self_cognition_action_attempts": {
            "collection_name": "self_cognition_action_attempts",
            "projection": {
                "_id": 0,
                "attempt_id": 1,
                "source_llm_trace_id": 1,
                "action_kind": 1,
                "status": 1,
                "correlation_write_status": 1,
                "correlation_conflict_source_llm_trace_id": 1,
                "created_at": 1,
            },
        },
        "background_work_jobs": {
            "collection_name": "background_work_jobs",
            "projection": {
                "_id": 0,
                "job_id": 1,
                "accepted_task_id": 1,
                "source_action_attempt_id": 1,
                "source_llm_trace_id": 1,
                "status": 1,
                "correlation_write_status": 1,
                "correlation_conflict_source_llm_trace_id": 1,
                "created_at": 1,
            },
        },
        "calendar_schedules": {
            "collection_name": "calendar_schedules",
            "projection": {
                "_id": 0,
                "schedule_id": 1,
                "source_action_attempt_id": 1,
                "source_llm_trace_id": 1,
                "status": 1,
                "correlation_write_status": 1,
                "correlation_conflict_source_llm_trace_id": 1,
                "due_at": 1,
                "created_at": 1,
            },
        },
        "calendar_runs": {
            "collection_name": "calendar_runs",
            "projection": {
                "_id": 0,
                "run_id": 1,
                "source_action_attempt_id": 1,
                "source_llm_trace_id": 1,
                "source_calendar_run_id": 1,
                "status": 1,
                "correlation_write_status": 1,
                "correlation_conflict_source_llm_trace_id": 1,
                "due_at": 1,
                "created_at": 1,
            },
        },
    }
    for kind, spec in companion_specs.items():
        rows_by_kind[kind] = await export_collection_rows(
            collection_name=spec["collection_name"],
            filter_doc={"source_llm_trace_id": clean_trace_id},
            projection=spec["projection"],
            sort_doc={"created_at": 1, "due_at": 1},
            limit=MAX_COMPANION_CANDIDATES,
        )
    rows_by_kind["child_trace_runs"] = await export_collection_rows(
        collection_name="llm_trace_runs",
        filter_doc={"parent_llm_trace_id": clean_trace_id},
        projection={
            "_id": 0,
            "trace_id": 1,
            "parent_llm_trace_id": 1,
            "source_background_work_job_id": 1,
            "source_calendar_run_id": 1,
            "status": 1,
            "platform": 1,
            "started_at": 1,
            "completed_at": 1,
        },
        sort_doc={"started_at": 1},
        limit=MAX_COMPANION_CANDIDATES,
    )
    return rows_by_kind


async def load_lane_cleanup_rows(
    *,
    collection_name: str,
    filter_doc: Mapping[str, Any],
    projection: Mapping[str, Any],
    sort_doc: Sequence[tuple[str, int]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load maintenance-cleanup rows through the DB-owned boundary.

    Args:
        collection_name: Approved collection name for a lane cleanup script.
        filter_doc: MongoDB filter owned by the maintenance command.
        projection: Fields to include or exclude from the export.
        sort_doc: Stable sort fields for deterministic reports.
        limit: Optional maximum number of rows. ``None`` loads all matches.

    Returns:
        Matching documents as dictionaries.
    """

    db = await get_db()
    cursor = db[collection_name].find(dict(filter_doc), dict(projection))
    if sort_doc:
        cursor = cursor.sort(list(sort_doc))
    if limit is not None:
        cursor = cursor.limit(limit)
    rows = [dict(doc) for doc in await cursor.to_list(length=limit)]
    return rows


async def count_lane_cleanup_rows(
    *,
    collection_name: str,
    filter_doc: Mapping[str, Any],
) -> int:
    """Count maintenance-cleanup rows through the DB-owned boundary."""

    db = await get_db()
    row_count = await db[collection_name].count_documents(dict(filter_doc))
    return row_count


async def count_lane_cleanup_field_values(
    *,
    collection_name: str,
    field_name: str,
) -> dict[str, int]:
    """Count maintenance-cleanup rows grouped by one scalar string field."""

    db = await get_db()
    pipeline = [
        {"$match": {
            field_name: {
                "$exists": True,
                "$type": "string",
                "$ne": "",
            }
        }},
        {"$group": {"_id": f"${field_name}", "count": {"$sum": 1}}},
    ]
    cursor = db[collection_name].aggregate(pipeline)
    grouped_counts = {
        str(row["_id"]): int(row["count"])
        for row in await cursor.to_list(length=None)
    }
    return grouped_counts


async def find_lane_cleanup_row(
    *,
    collection_name: str,
    filter_doc: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Find one maintenance-cleanup row for drift validation."""

    db = await get_db()
    row = await db[collection_name].find_one(
        dict(filter_doc),
        dict(projection),
    )
    if row is None:
        return_value = None
        return return_value
    return_value = dict(row)
    return return_value


async def update_lane_cleanup_row(
    *,
    collection_name: str,
    filter_doc: Mapping[str, Any],
    set_fields: Mapping[str, Any],
    unset_fields: Sequence[str] = (),
    push_fields: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Apply one exact maintenance-cleanup row update.

    Args:
        collection_name: Approved collection name for a lane cleanup script.
        filter_doc: Exact row selector after drift validation.
        set_fields: Fields to set.
        unset_fields: Optional field names to unset.
        push_fields: Optional fields to append with ``$push``.

    Returns:
        Matched and modified counts.
    """

    update_doc: dict[str, Any] = {"$set": dict(set_fields)}
    if unset_fields:
        update_doc["$unset"] = {field_name: "" for field_name in unset_fields}
    if push_fields:
        update_doc["$push"] = dict(push_fields)

    db = await get_db()
    update_result = await db[collection_name].update_one(
        dict(filter_doc),
        update_doc,
    )
    result = {
        "matched_count": int(update_result.matched_count),
        "modified_count": int(update_result.modified_count),
    }
    return result


async def delete_lane_cleanup_row(
    *,
    collection_name: str,
    filter_doc: Mapping[str, Any],
) -> dict[str, int]:
    """Delete one exact maintenance-cleanup row after drift validation."""

    db = await get_db()
    delete_result = await db[collection_name].delete_one(dict(filter_doc))
    result = {"deleted_count": int(delete_result.deleted_count)}
    return result


async def export_event_log_events_for_trace_id(
    trace_id: str,
    *,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Export event-log rows linked to one LLM trace id."""

    filter_doc = {
        "$or": [
            {"correlation_id": trace_id},
            {"labels.llm_trace_id": trace_id},
            {"refs": {"$elemMatch": {
                "ref_type": "llm_trace",
                "ref_id": trace_id,
            }}},
        ]
    }
    rows = await export_collection_rows(
        collection_name="event_log_events",
        filter_doc=filter_doc,
        projection={},
        sort_doc={"occurred_at": 1},
        limit=limit,
    )
    return rows


async def list_scheduled_events_for_calendar_migration() -> list[dict[str, Any]]:
    """Load legacy scheduled-event rows for the calendar migration script."""

    db = await get_db()
    cursor = db.scheduled_events.find({})
    rows = [dict(row) for row in await cursor.to_list(length=None)]
    return rows


async def cancel_pending_send_message_for_calendar_migration(
    event_id: str,
) -> bool:
    """Cancel one pending legacy delayed-send row during calendar migration."""

    db = await get_db()
    update_result = await db.scheduled_events.update_one(
        {
            "event_id": event_id,
            "status": CALENDAR_MIGRATION_PENDING_STATUS,
            "tool": CALENDAR_MIGRATION_SEND_MESSAGE_TOOL,
        },
        {"$set": {"status": "cancelled"}},
    )
    matched = update_result.matched_count == 1
    return matched


async def mark_pending_future_cognition_migrated_for_calendar_migration(
    event_id: str,
) -> bool:
    """Mark one pending legacy future-cognition row migrated."""

    db = await get_db()
    update_result = await db.scheduled_events.update_one(
        {
            "event_id": event_id,
            "status": CALENDAR_MIGRATION_PENDING_STATUS,
            "tool": CALENDAR_MIGRATION_FUTURE_COGNITION_TOOL,
        },
        {"$set": {"status": "migrated"}},
    )
    matched = update_result.matched_count == 1
    return matched


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


async def count_legacy_conversation_history_rows(
    *,
    semantic_text_pattern: str = "",
) -> int:
    """Count conversation-history rows matching a maintenance selector.

    Args:
        semantic_text_pattern: Optional regex for dirty semantic text fields.

    Returns:
        Number of rows that need maintenance repair.
    """

    db = await get_db()
    count = await db.conversation_history.count_documents(
        _legacy_conversation_query(
            semantic_text_pattern=semantic_text_pattern,
        )
    )
    return count


async def list_legacy_conversation_history_rows(
    *,
    batch_size: int,
    semantic_text_pattern: str = "",
) -> list[dict[str, Any]]:
    """Load one deterministic batch of conversation-history rows.

    Args:
        batch_size: Maximum number of rows to load.
        semantic_text_pattern: Optional regex for dirty semantic text fields.

    Returns:
        Conversation rows that need maintenance repair.
    """

    db = await get_db()
    query = _legacy_conversation_query(
        semantic_text_pattern=semantic_text_pattern,
    )
    cursor = (
        db.conversation_history
        .find(query)
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


async def update_conversation_history_row_for_semantic_identity_repair(
    *,
    row_id: Any,
    set_fields: Mapping[str, Any],
    unset_fields: Sequence[str],
    recompute_embedding: bool,
) -> None:
    """Apply one conversation-history identity repair with optional embedding.

    Args:
        row_id: MongoDB row identifier.
        set_fields: Fields to set on the conversation row.
        unset_fields: Fields to remove from the conversation row.
        recompute_embedding: Whether to recompute the document embedding from
            the repaired semantic text.
    """

    fields = dict(set_fields)
    db = await get_db()
    if recompute_embedding:
        existing = await db.conversation_history.find_one({"_id": row_id})
        if existing is None:
            return
        repaired_row = {
            **dict(existing),
            **fields,
        }
        fields["embedding"] = await get_document_text_embedding(
            _embedding_source_text(repaired_row),
        )

    update: dict[str, Any] = {"$set": fields}
    if unset_fields:
        update["$unset"] = {
            field_name: ""
            for field_name in unset_fields
        }
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


async def list_user_cognition_states_for_relationship_maintenance_migration(
) -> list[dict[str, Any]]:
    """Load user cognition rows in the migration's canonical order."""

    db = await get_db()
    cursor = db.user_profiles.find(
        {"cognition_state.state_scope": "user"},
        {
            "_id": 0,
            "global_user_id": 1,
            "cognition_state": 1,
        },
    ).sort("global_user_id", 1)
    return [dict(row) for row in await cursor.to_list(length=None)]


def cognition_state_migration_digest(state: Mapping[str, Any]) -> str:
    """Return the canonical digest used by the maintenance migration."""

    return _extended_json_sha256(state)


async def compare_and_replace_user_cognition_state_for_migration(
    *,
    global_user_id: str,
    expected_previous_state: Mapping[str, Any],
    expected_previous_digest: str,
    replacement_state: Mapping[str, Any],
) -> bool:
    """Apply one migration row only when its reviewed base digest still matches."""

    replacement = validate_cognition_state(replacement_state)
    if (
        expected_previous_state.get("state_scope") != "user"
        or replacement["state_scope"] != "user"
        or expected_previous_state.get("owner_user_id") != global_user_id
        or replacement["owner_user_id"] != global_user_id
    ):
        raise ValueError("migration cognition state owner does not match row")
    expected_state = dict(expected_previous_state)
    actual_digest = cognition_state_migration_digest(expected_state)
    if actual_digest != expected_previous_digest:
        raise ValueError("migration expected-state digest does not match")
    db = await get_db()
    result = await db.user_profiles.update_one(
        {
            "global_user_id": global_user_id,
            "cognition_state": expected_state,
        },
        {"$set": {"cognition_state": replacement}},
        upsert=False,
    )
    return result.matched_count == 1


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


async def archive_user_memory_unit_for_semantic_identity_repair(
    *,
    unit_id: str,
    reason: str,
    storage_timestamp_utc: str,
) -> dict[str, object]:
    """Archive an active user-memory row during identity-pollution repair.

    Args:
        unit_id: Stable ``user_memory_units.unit_id`` to archive.
        reason: Maintenance reason recorded in merge history.
        storage_timestamp_utc: Storage UTC timestamp for update fields.

    Returns:
        Update counters and the merge-history row.
    """

    clean_unit_id = str(unit_id or "").strip()
    clean_reason = str(reason or "").strip()
    clean_timestamp = str(storage_timestamp_utc or "").strip()
    if not clean_unit_id:
        raise ValueError("unit_id is required")
    if not clean_reason:
        raise ValueError("reason is required")
    if not clean_timestamp:
        raise ValueError("storage_timestamp_utc is required")

    merge_history_entry = {
        "operation": "semantic_identity_repair_archive",
        "status": "archived",
        "reason": clean_reason,
        "timestamp": clean_timestamp,
    }
    db = await get_db()
    result = await db.user_memory_units.update_one(
        {
            "unit_id": clean_unit_id,
            "status": "active",
        },
        {
            "$set": {
                "status": "archived",
                "archived_at": clean_timestamp,
                "updated_at": clean_timestamp,
            },
            "$push": {"merge_history": merge_history_entry},
        },
    )
    return_value = {
        "unit_id": clean_unit_id,
        "status": "archived",
        "matched_count": result.matched_count,
        "modified_count": result.modified_count,
        "merge_history_entry": merge_history_entry,
    }
    return return_value


def _legacy_conversation_query(
    *,
    semantic_text_pattern: str = "",
) -> dict[str, Any]:
    """Build the selector for rows outside the typed storage contract.

    Args:
        semantic_text_pattern: Optional regex for dirty semantic text fields.

    Returns:
        MongoDB selector for rows that need maintenance repair.
    """

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
    if semantic_text_pattern:
        query["$or"].extend([
            {"body_text": {"$regex": semantic_text_pattern}},
            {"reply_context.reply_excerpt": {"$regex": semantic_text_pattern}},
        ])
    return query


def _semantic_identity_conversation_query() -> dict[str, Any]:
    """Build selector for conversation rows with polluted semantic fields."""

    pattern = SEMANTIC_IDENTITY_FORBIDDEN_PATTERN
    query = {
        "$or": [
            {"display_name": {"$regex": pattern}},
            {"body_text": {"$regex": pattern}},
            {"mentions.display_name": {"$regex": pattern}},
            {"reply_context.reply_to_display_name": {"$regex": pattern}},
            {"reply_context.reply_excerpt": {"$regex": pattern}},
        ]
    }
    return query


def _semantic_identity_user_memory_query() -> dict[str, Any]:
    """Build selector for active user-memory rows with polluted text."""

    pattern = SEMANTIC_IDENTITY_FORBIDDEN_PATTERN
    query = {
        "status": "active",
        "$or": [
            {"fact": {"$regex": pattern}},
            {"subjective_appraisal": {"$regex": pattern}},
            {"relationship_signal": {"$regex": pattern}},
        ],
    }
    return query


def _semantic_identity_user_profile_query() -> dict[str, Any]:
    """Build selector for profile platform accounts with polluted labels."""

    pattern = SEMANTIC_IDENTITY_FORBIDDEN_PATTERN
    query = {
        "platform_accounts.display_name": {"$regex": pattern},
    }
    return query


def _semantic_identity_shared_memory_query() -> dict[str, Any]:
    """Build selector for active shared-memory rows with polluted text."""

    pattern = SEMANTIC_IDENTITY_FORBIDDEN_PATTERN
    query = {
        "status": "active",
        "$or": [
            {"memory_name": {"$regex": pattern}},
            {"content": {"$regex": pattern}},
            {"confidence_note": {"$regex": pattern}},
        ],
    }
    return query


async def count_semantic_identity_conversation_rows() -> int:
    """Count conversation rows with polluted semantic identity fields."""

    db = await get_db()
    count = await db.conversation_history.count_documents(
        _semantic_identity_conversation_query(),
    )
    return count


async def list_semantic_identity_conversation_rows(
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Load conversation rows with polluted semantic identity fields."""

    db = await get_db()
    cursor = (
        db.conversation_history
        .find(_semantic_identity_conversation_query())
        .sort("timestamp", 1)
        .limit(batch_size)
    )
    rows = [dict(row) for row in await cursor.to_list(length=batch_size)]
    return rows


async def count_semantic_identity_user_memory_units() -> int:
    """Count active user-memory rows with polluted semantic identity fields."""

    db = await get_db()
    count = await db.user_memory_units.count_documents(
        _semantic_identity_user_memory_query(),
    )
    return count


async def count_semantic_identity_user_profile_accounts() -> int:
    """Count profiles containing polluted platform-account display labels."""

    db = await get_db()
    cursor = db.user_profiles.find(
        _semantic_identity_user_profile_query(),
        {"platform_accounts": 1},
    )
    dirty_count = 0
    pattern = SEMANTIC_IDENTITY_FORBIDDEN_PATTERN
    async for row in cursor:
        accounts = row.get("platform_accounts")
        if not isinstance(accounts, list):
            continue
        for account in accounts:
            if not isinstance(account, dict):
                continue
            display_name = account.get("display_name")
            if isinstance(display_name, str) and re.search(pattern, display_name):
                dirty_count += 1
    return dirty_count


async def list_semantic_identity_user_profile_rows(
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Load user profiles with polluted platform-account display labels."""

    db = await get_db()
    cursor = (
        db.user_profiles
        .find(_semantic_identity_user_profile_query(), {"embedding": 0})
        .sort("global_user_id", 1)
        .limit(batch_size)
    )
    rows = [dict(row) for row in await cursor.to_list(length=batch_size)]
    return rows


async def update_user_profile_platform_account_display_name(
    *,
    global_user_id: str,
    platform: str,
    platform_user_id: str,
    display_name: str,
) -> dict[str, int]:
    """Update one exact linked platform-account display label."""

    db = await get_db()
    result = await db.user_profiles.update_one(
        {
            "global_user_id": global_user_id,
            "platform_accounts": {
                "$elemMatch": {
                    "platform": platform,
                    "platform_user_id": platform_user_id,
                }
            },
        },
        {
            "$set": {
                "platform_accounts.$.display_name": display_name,
            }
        },
    )
    return_value = {
        "matched_count": int(result.matched_count),
        "modified_count": int(result.modified_count),
    }
    return return_value


async def list_semantic_identity_user_memory_units(
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Load active user-memory rows with polluted semantic identity fields."""

    db = await get_db()
    cursor = (
        db.user_memory_units
        .find(_semantic_identity_user_memory_query(), {"embedding": 0})
        .sort("updated_at", 1)
        .limit(batch_size)
    )
    rows = [dict(row) for row in await cursor.to_list(length=batch_size)]
    return rows


async def count_semantic_identity_shared_memory_units() -> int:
    """Count active shared-memory rows with polluted semantic identity fields."""

    db = await get_db()
    count = await db.memory.count_documents(
        _semantic_identity_shared_memory_query(),
    )
    return count


async def list_semantic_identity_shared_memory_units(
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Load active shared-memory rows with polluted semantic identity fields."""

    db = await get_db()
    cursor = (
        db.memory
        .find(_semantic_identity_shared_memory_query(), {"embedding": 0})
        .sort("updated_at", 1)
        .limit(batch_size)
    )
    rows = [dict(row) for row in await cursor.to_list(length=batch_size)]
    return rows


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


CONVERSATION_PROGRESS_MIGRATION_COLLECTION = 'conversation_episode_state'
CONVERSATION_PROGRESS_DRY_RUN_VERSION = (
    'conversation_progress_v2_migration_dry_run.v1'
)
CONVERSATION_PROGRESS_APPLY_VERSION = (
    'conversation_progress_v2_migration_apply.v1'
)
CONVERSATION_PROGRESS_RESTORE_VERSION = (
    'conversation_progress_v2_migration_restore.v1'
)
_CONVERSATION_PROGRESS_DRIFT_FIELDS = (
    'platform',
    'platform_channel_id',
    'global_user_id',
    'turn_count',
    'updated_at',
    'status',
)


def classify_conversation_progress_migration_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one progress row without inferring any V1 semantics."""

    schema_present = 'schema_version' in row
    schema_version = row.get('schema_version')
    scope_valid = all(
        isinstance(row.get(field_name), str)
        and bool(str(row[field_name]).strip())
        for field_name in (
            'platform',
            'global_user_id',
        )
    ) and isinstance(row.get('platform_channel_id'), str)
    turn_count = row.get('turn_count')
    turn_count_valid = (
        isinstance(turn_count, int)
        and not isinstance(turn_count, bool)
        and turn_count >= 0
    )
    timestamps_valid = all(
        _valid_migration_timestamp(row.get(field_name))
        for field_name in ('created_at', 'updated_at', 'expires_at')
    )
    status_valid = (
        isinstance(row.get('status'), str)
        and bool(str(row['status']).strip())
    )
    v2_contract_valid = (
        schema_version == 'conversation_progress.v2'
        and _valid_conversation_progress_v2_packet(row)
    )
    if v2_contract_valid:
        classification = 'already_v2'
    elif (
        schema_version in {None, 'conversation_progress.v1'}
        and
        scope_valid
        and turn_count_valid
        and timestamps_valid
        and status_valid
    ):
        classification = 'v1_eligible'
    else:
        classification = 'malformed'
    return {
        '_id': str(row.get('_id', '')),
        'classification': classification,
        'schema_present': schema_present,
        'schema_version': schema_version,
        'scope_valid': scope_valid,
        'turn_count_valid': turn_count_valid,
        'timestamps_valid': timestamps_valid,
        'status_valid': status_valid,
        'v2_contract_valid': v2_contract_valid,
        'drift': {
            field_name: row.get(field_name)
            for field_name in _CONVERSATION_PROGRESS_DRIFT_FIELDS
        },
    }


def _valid_conversation_progress_v2_packet(
    row: Mapping[str, Any],
) -> bool:
    """Apply the canonical runtime validator to a labelled V2 row."""

    from kazusa_ai_chatbot.conversation_progress import (
        validate_active_packet,
    )

    candidate = {
        field_name: value
        for field_name, value in row.items()
        if field_name != '_id'
    }
    try:
        validate_active_packet(candidate)
    except (TypeError, ValueError):
        return False
    return True


def build_conversation_progress_v2_dry_run_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    generated_at: str,
    backup_path: str,
    backup_sha256: str,
) -> dict[str, Any]:
    """Build the read-only audit report linked to one complete backup."""

    classified_rows = [
        classify_conversation_progress_migration_row(row)
        for row in rows
    ]
    counts = {
        'total': len(classified_rows),
        'v1_eligible': 0,
        'already_v2': 0,
        'malformed': 0,
    }
    for row in classified_rows:
        counts[row['classification']] += 1
    return {
        'schema_version': CONVERSATION_PROGRESS_DRY_RUN_VERSION,
        'generated_at': generated_at,
        'collection': CONVERSATION_PROGRESS_MIGRATION_COLLECTION,
        'backup_path': backup_path,
        'backup_sha256': backup_sha256,
        'backup_row_count': len(rows),
        'writes_performed': 0,
        'counts': counts,
        'rows': classified_rows,
    }


async def dry_run_conversation_progress_v2_migration(
    *,
    backup_output: Path,
    report_output: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Export every row and write a zero-write V2 migration report."""

    db = await get_db()
    rows = await db[
        CONVERSATION_PROGRESS_MIGRATION_COLLECTION
    ].find({}).to_list(length=None)
    backup_payload = {
        'schema_version': 'conversation_progress_v1_backup.v1',
        'generated_at': generated_at,
        'collection': CONVERSATION_PROGRESS_MIGRATION_COLLECTION,
        'rows': rows,
    }
    backup_text = json_util.dumps(
        backup_payload,
        json_options=RELAXED_JSON_OPTIONS,
        sort_keys=True,
        indent=2,
    )
    backup_bytes = backup_text.encode('utf-8')
    backup_sha256 = hashlib.sha256(backup_bytes).hexdigest()
    report = build_conversation_progress_v2_dry_run_report(
        rows=rows,
        generated_at=generated_at,
        backup_path=str(backup_output),
        backup_sha256=backup_sha256,
    )
    backup_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    backup_output.write_bytes(backup_bytes)
    report_output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding='utf-8',
    )
    return report


async def audit_conversation_progress_v2_rows(
    *,
    output: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Write a read-only schema/lifecycle audit without a backup artifact."""

    db = await get_db()
    rows = await db[
        CONVERSATION_PROGRESS_MIGRATION_COLLECTION
    ].find({}).to_list(length=None)
    classified = [
        classify_conversation_progress_migration_row(row)
        for row in rows
    ]
    report = {
        'schema_version': 'conversation_progress_v2_audit.v1',
        'generated_at': generated_at,
        'collection': CONVERSATION_PROGRESS_MIGRATION_COLLECTION,
        'writes_performed': 0,
        'rows': classified,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding='utf-8',
    )
    return report


async def apply_conversation_progress_v2_migration(
    *,
    dry_run_input: Path,
    backup_input: Path,
    output: Path,
    applied_at: str,
) -> dict[str, Any]:
    """Replace drift-matched V1 rows with exact closed V2 tombstones."""

    dry_run = _load_json_object(dry_run_input)
    if dry_run.get('schema_version') != CONVERSATION_PROGRESS_DRY_RUN_VERSION:
        raise ValueError('migration dry-run schema is invalid')
    backup_bytes = backup_input.read_bytes()
    backup_sha256 = hashlib.sha256(backup_bytes).hexdigest()
    if backup_sha256 != dry_run.get('backup_sha256'):
        raise ValueError('migration backup digest does not match dry-run')
    backup = json_util.loads(backup_bytes.decode('utf-8'))
    if not isinstance(backup, Mapping):
        raise ValueError('migration backup must be an object')
    backup_rows = backup.get('rows')
    if not isinstance(backup_rows, list):
        raise ValueError('migration backup rows must be a list')
    if len(backup_rows) != dry_run.get('backup_row_count'):
        raise ValueError('migration backup row count does not match dry-run')
    dry_rows = dry_run.get('rows')
    if not isinstance(dry_rows, list):
        raise ValueError('migration dry-run rows must be a list')
    dry_by_id = {
        str(row.get('_id', '')): row
        for row in dry_rows
        if isinstance(row, Mapping)
    }
    db = await get_db()
    collection = db[CONVERSATION_PROGRESS_MIGRATION_COLLECTION]
    counts = {
        'changed': 0,
        'drift_skipped': 0,
        'already_v2': 0,
        'malformed': 0,
        'blocked': 0,
        'failed': 0,
    }
    applied_rows: list[dict[str, str]] = []
    for backup_row in backup_rows:
        if not isinstance(backup_row, Mapping):
            counts['malformed'] += 1
            continue
        row_id = str(backup_row.get('_id', ''))
        dry_row = dry_by_id.get(row_id)
        if dry_row is None:
            counts['blocked'] += 1
            continue
        classification = dry_row.get('classification')
        if classification == 'already_v2':
            counts['already_v2'] += 1
            continue
        if classification != 'v1_eligible':
            counts['malformed'] += 1
            continue
        try:
            current = await collection.find_one({'_id': backup_row['_id']})
            if current is None or not _migration_row_matches_dry_run(
                current,
                dry_row,
            ):
                counts['drift_skipped'] += 1
                continue
            tombstone = build_conversation_progress_v2_tombstone(
                source_row=current,
                applied_at=applied_at,
            )
            result = await collection.replace_one(
                _migration_drift_filter(current, dry_row),
                tombstone,
                upsert=False,
            )
        except PyMongoError:
            counts['failed'] += 1
            continue
        if result.modified_count != 1:
            counts['drift_skipped'] += 1
            continue
        counts['changed'] += 1
        applied_rows.append({
            '_id': row_id,
            'tombstone_sha256': _extended_json_sha256(tombstone),
        })
    report = {
        'schema_version': CONVERSATION_PROGRESS_APPLY_VERSION,
        'applied_at': applied_at,
        'collection': CONVERSATION_PROGRESS_MIGRATION_COLLECTION,
        'dry_run_path': str(dry_run_input),
        'backup_path': str(backup_input),
        'backup_sha256': backup_sha256,
        'counts': counts,
        'applied_rows': applied_rows,
        'unrelated_collection_writes': 0,
        'conversation_history_deletes': 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding='utf-8',
    )
    return report


async def restore_conversation_progress_v1_backup(
    *,
    apply_input: Path,
    backup_input: Path,
    output: Path,
    restored_at: str,
) -> dict[str, Any]:
    """Restore only rows still equal to their exact applied tombstone."""

    apply_report = _load_json_object(apply_input)
    if apply_report.get('schema_version') != CONVERSATION_PROGRESS_APPLY_VERSION:
        raise ValueError('migration apply report schema is invalid')
    backup_bytes = backup_input.read_bytes()
    backup_sha256 = hashlib.sha256(backup_bytes).hexdigest()
    if backup_sha256 != apply_report.get('backup_sha256'):
        raise ValueError('restore backup digest does not match apply report')
    backup = json_util.loads(backup_bytes.decode('utf-8'))
    if not isinstance(backup, Mapping) or not isinstance(
        backup.get('rows'),
        list,
    ):
        raise ValueError('restore backup shape is invalid')
    backup_by_id = {
        str(row.get('_id', '')): row
        for row in backup['rows']
        if isinstance(row, Mapping)
    }
    applied_rows = apply_report.get('applied_rows')
    if not isinstance(applied_rows, list):
        raise ValueError('apply report applied_rows is invalid')
    db = await get_db()
    collection = db[CONVERSATION_PROGRESS_MIGRATION_COLLECTION]
    counts = {
        'restored': 0,
        'drift_skipped': 0,
        'active_v2_skipped': 0,
        'missing_backup': 0,
        'failed': 0,
    }
    for applied_row in applied_rows:
        if not isinstance(applied_row, Mapping):
            counts['drift_skipped'] += 1
            continue
        row_id = str(applied_row.get('_id', ''))
        backup_row = backup_by_id.get(row_id)
        if backup_row is None:
            counts['missing_backup'] += 1
            continue
        try:
            current = await collection.find_one({'_id': backup_row['_id']})
        except PyMongoError:
            counts['failed'] += 1
            continue
        if current is None:
            counts['drift_skipped'] += 1
            continue
        if (
            current.get('schema_version') == 'conversation_progress.v2'
            and current.get('status') == 'active'
            and isinstance(current.get('turn_count'), int)
            and current['turn_count'] >= 1
        ):
            counts['active_v2_skipped'] += 1
            continue
        if _extended_json_sha256(current) != applied_row.get(
            'tombstone_sha256'
        ):
            counts['drift_skipped'] += 1
            continue
        try:
            result = await collection.replace_one(
                dict(current),
                dict(backup_row),
                upsert=False,
            )
        except PyMongoError:
            counts['failed'] += 1
            continue
        if result.modified_count == 1:
            counts['restored'] += 1
        else:
            counts['drift_skipped'] += 1
    report = {
        'schema_version': CONVERSATION_PROGRESS_RESTORE_VERSION,
        'restored_at': restored_at,
        'collection': CONVERSATION_PROGRESS_MIGRATION_COLLECTION,
        'apply_path': str(apply_input),
        'backup_path': str(backup_input),
        'counts': counts,
        'unrelated_collection_writes': 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding='utf-8',
    )
    return report


def build_conversation_progress_v2_tombstone(
    *,
    source_row: Mapping[str, Any],
    applied_at: str,
) -> dict[str, Any]:
    """Build an exact closed V2 reset row with the same MongoDB identifier."""

    expires_at, purge_after = storage_expiry(applied_at)
    return {
        '_id': source_row['_id'],
        'schema_version': 'conversation_progress.v2',
        'episode_state_id': uuid4().hex,
        'platform': source_row['platform'],
        'platform_channel_id': source_row['platform_channel_id'],
        'global_user_id': source_row['global_user_id'],
        'status': 'closed',
        'continuity': 'sharp_transition',
        'turn_count': 0,
        'episode_narrative': '',
        'current_thread': '',
        'character_stance': '',
        'user_goal': '',
        'current_blocker': '',
        'emotional_trajectory': '',
        'events': [],
        'overused_moves': [],
        'recent_turn_refs': [],
        'compacted_block_refs': [],
        'created_at': applied_at,
        'updated_at': applied_at,
        'expires_at': expires_at,
        'purge_after': purge_after,
    }


def _migration_row_matches_dry_run(
    current: Mapping[str, Any],
    dry_row: Mapping[str, Any],
) -> bool:
    """Match every approved drift field plus schema presence/value."""

    drift = dry_row.get('drift')
    if not isinstance(drift, Mapping):
        return False
    if any(
        current.get(field_name) != drift.get(field_name)
        for field_name in _CONVERSATION_PROGRESS_DRIFT_FIELDS
    ):
        return False
    schema_present = dry_row.get('schema_present') is True
    if schema_present != ('schema_version' in current):
        return False
    return current.get('schema_version') == dry_row.get('schema_version')


def _valid_migration_timestamp(value: object) -> bool:
    """Return whether one migration timestamp is canonical storage UTC."""

    if not isinstance(value, str) or not value:
        return False
    try:
        parse_storage_utc_datetime(value)
    except ValueError:
        return False
    return True


def _migration_drift_filter(
    current: Mapping[str, Any],
    dry_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the atomic exact-match filter for one reviewed row."""

    filter_doc = {
        '_id': current['_id'],
        **{
            field_name: current.get(field_name)
            for field_name in _CONVERSATION_PROGRESS_DRIFT_FIELDS
        },
    }
    if dry_row.get('schema_present') is True:
        filter_doc['schema_version'] = dry_row.get('schema_version')
    else:
        filter_doc['schema_version'] = {'$exists': False}
    return filter_doc


def _extended_json_sha256(value: Mapping[str, Any]) -> str:
    """Digest one complete MongoDB document using canonical Extended JSON."""

    encoded = json_util.dumps(
        dict(value),
        json_options=RELAXED_JSON_OPTIONS,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object and reject every other root shape."""

    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return value
