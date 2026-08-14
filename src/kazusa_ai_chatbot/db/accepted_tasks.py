"""MongoDB persistence helpers for accepted delayed user tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.accepted_task.models import (
    ACCEPTED_TASK_SCHEMA_VERSION,
    ACCEPTED_TASKS_COLLECTION,
    ACTIVE_ACCEPTED_TASK_STATES,
    AcceptedTaskCreateResult,
    AcceptedTaskDoc,
    AcceptedTaskStatusCheckRequest,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError

CODING_RUN_CONTEXT_LOAD_MULTIPLIER = 4
CODING_RUN_CONTEXT_SCHEMA_VERSION = "coding_run_context.v1"
MAX_CODING_RUN_CONTEXT_TEXT_CHARS = 1200
MAX_CODING_RUN_CONTEXT_LIST_ITEMS = 8
OPEN_CODING_RUN_CONTEXT_INDEX_NAME = (
    "accepted_task_open_coding_run_context_lookup"
)
OPEN_CODING_RUN_CONTEXT_INDEX_KEYS = (
    ("source_platform", 1),
    ("source_channel_id", 1),
    ("requester_global_user_id", 1),
    ("task_kind", 1),
    ("updated_at", -1),
)
OPEN_CODING_RUN_CONTEXT_INDEX_FILTER = {
    "coding_run_context.followup_open": True,
}


async def ensure_accepted_task_indexes() -> None:
    """Create all idempotent indexes for accepted-task lifecycle rows."""

    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    try:
        await _drop_conflicting_open_coding_run_context_index(collection)
        await collection.create_index(
            "accepted_task_id",
            unique=True,
            name="accepted_task_id_unique",
        )
        await collection.create_index(
            "active_identity_key",
            unique=True,
            partialFilterExpression={"active_identity_key": {"$exists": True}},
            name="accepted_task_active_identity_unique",
        )
        await collection.create_index(
            [("state", 1), ("updated_at", 1)],
            name="accepted_task_state_updated",
        )
        await collection.create_index(
            [
                ("source_platform", 1),
                ("source_channel_id", 1),
                ("source_channel_type", 1),
                ("requester_global_user_id", 1),
                ("requester_platform_user_id", 1),
                ("state", 1),
                ("updated_at", -1),
            ],
            name="accepted_task_scope_active_lookup",
        )
        await collection.create_index(
            list(OPEN_CODING_RUN_CONTEXT_INDEX_KEYS),
            partialFilterExpression=OPEN_CODING_RUN_CONTEXT_INDEX_FILTER,
            name=OPEN_CODING_RUN_CONTEXT_INDEX_NAME,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to ensure accepted task indexes: {exc}"
        ) from exc


async def _drop_conflicting_open_coding_run_context_index(
    collection: Any,
) -> None:
    """Remove the v1 named-index definition before v2 index creation."""

    index_information = await collection.index_information()
    existing = index_information.get(OPEN_CODING_RUN_CONTEXT_INDEX_NAME)
    if not isinstance(existing, Mapping):
        return
    raw_keys = existing.get("key")
    existing_keys = tuple(
        tuple(row)
        for row in raw_keys
        if isinstance(row, (list, tuple)) and len(row) == 2
    ) if isinstance(raw_keys, list) else ()
    existing_filter = existing.get("partialFilterExpression")
    if (
        existing_keys == OPEN_CODING_RUN_CONTEXT_INDEX_KEYS
        and existing_filter == OPEN_CODING_RUN_CONTEXT_INDEX_FILTER
    ):
        return
    await collection.drop_index(OPEN_CODING_RUN_CONTEXT_INDEX_NAME)


async def insert_or_get_active_accepted_task(
    task: AcceptedTaskDoc,
    *,
    source_message_id: str,
    observed_at: str,
) -> AcceptedTaskCreateResult:
    """Insert one active task or return the exact-reference duplicate.

    An active duplicate is returned only when its stored goal continuation
    reference exactly matches the incoming reference; a mismatch fails closed
    before any provenance field changes.
    """

    _validate_v2_task_document(task)
    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    try:
        await collection.insert_one(dict(task))
    except DuplicateKeyError:
        existing = await collection.find_one(
            {
                "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
                "active_identity_key": task["active_identity_key"],
            },
            {"_id": 0},
        )
        if existing is None:
            raise DatabaseOperationError(
                "accepted task duplicate without readable active row"
            )
        incoming_ref = task.get("goal_continuation_ref")
        stored_ref = existing.get("goal_continuation_ref")
        if incoming_ref != stored_ref:
            raise DatabaseOperationError(
                "active accepted task has a different goal_continuation_ref"
            )
        active_task = await _add_related_source_message_id(
            collection,
            active_identity_key=task["active_identity_key"],
            source_message_id=source_message_id,
            observed_at=observed_at,
        )
        if active_task is None:
            raise DatabaseOperationError(
                "accepted task duplicate without readable active row"
            )
        duplicate_result: AcceptedTaskCreateResult = {
            "status": "already_active",
            "task": active_task,
        }
        return duplicate_result
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to insert accepted task: {exc}"
        ) from exc

    result: AcceptedTaskCreateResult = {
        "status": "created",
        "task": task,
    }
    return result


async def load_open_coding_run_contexts_for_scope(
    *,
    source_platform: str,
    source_channel_id: str,
    requester_global_user_id: str,
    limit: int = 3,
) -> list[dict[str, object]]:
    """Load newest unique open coding contexts for one trusted user scope.

    Args:
        source_platform: Adapter platform owning the current user turn.
        source_channel_id: Channel owning the current user turn.
        requester_global_user_id: Durable requester identity for the turn.
        limit: Maximum number of newest distinct run contexts to project.

    Returns:
        Prompt-safe contexts collapsed by coding-run reference.
    """

    if limit < 1:
        return []
    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    query = {
        "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
        "source_platform": source_platform,
        "source_channel_id": source_channel_id,
        "requester_global_user_id": requester_global_user_id,
        "task_kind": "coding_continuation",
        "coding_run_context.followup_open": True,
    }
    try:
        cursor = collection.find(
            query,
            projection={"_id": 0, "coding_run_context": 1},
        ).sort("updated_at", -1).limit(
            limit * CODING_RUN_CONTEXT_LOAD_MULTIPLIER,
        )
        rows = await cursor.to_list(
            length=limit * CODING_RUN_CONTEXT_LOAD_MULTIPLIER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to load open coding-run contexts: {exc}"
        ) from exc
    contexts: list[dict[str, object]] = []
    seen_refs: set[str] = set()
    for row in rows:
        context = _sanitize_prompt_safe_coding_context(
            row.get("coding_run_context"),
        )
        if context is None:
            continue
        context_ref = context["coding_run_ref"]
        if context_ref in seen_refs:
            continue
        seen_refs.add(context_ref)
        contexts.append(context)
        if len(contexts) >= limit:
            break
    return contexts


async def mark_accepted_task_pending(
    *,
    accepted_task_id: str,
    executor_ref: str,
    updated_at: str,
) -> AcceptedTaskDoc | None:
    """Mark an enqueueing task pending before its job becomes claimable.

    A concurrent materializer that reserved the same executor reference may
    observe the already-pending row. A different executor reference still
    fails closed.
    """

    update = {
        "$set": {
            "state": "pending",
            "executor_kind": "background_work",
            "executor_ref": executor_ref,
            "updated_at": updated_at,
        }
    }
    task = await _update_task(
        {"accepted_task_id": accepted_task_id, "state": "enqueueing"},
        update,
    )
    if task is None:
        task = await _update_task(
            {
                "accepted_task_id": accepted_task_id,
                "state": "pending",
                "executor_ref": executor_ref,
            },
            update,
        )
    return task


async def mark_accepted_task_enqueue_failed(
    *,
    accepted_task_id: str,
    failure_summary: str,
    updated_at: str,
) -> AcceptedTaskDoc | None:
    """Mark a failed enqueue and release active duplicate suppression."""

    update = {
        "$set": {
            "state": "enqueue_failed",
            "completion_status": "failed",
            "result_kind": "failed",
            "failure_summary": failure_summary,
            "updated_at": updated_at,
        },
        "$unset": {
            "active_identity_key": "",
        },
    }
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "state": {"$in": ["enqueueing", "pending"]},
        },
        update,
    )
    return task


async def recover_stale_enqueueing_tasks(
    *,
    stale_before_utc: str,
    recovered_at: str,
) -> int:
    """Release active identities for enqueueing tasks older than the cutoff."""

    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    update = {
        "$set": {
            "state": "enqueue_failed",
            "completion_status": "failed",
            "result_kind": "failed",
            "failure_summary": "Accepted task enqueue did not complete.",
            "updated_at": recovered_at,
        },
        "$unset": {
            "active_identity_key": "",
        },
    }
    try:
        update_result = await collection.update_many(
            {
                "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
                "state": "enqueueing",
                "updated_at": {"$lte": stale_before_utc},
            },
            update,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to recover stale accepted tasks: {exc}"
        ) from exc

    recovered_count = int(update_result.modified_count)
    return recovered_count


async def recover_stale_delivery_in_progress_tasks(
    *,
    stale_before_utc: str,
    recovered_at: str,
) -> int:
    """Return interrupted delivery claims to retryable state."""

    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    update = {
        "$set": {
            "state": "delivery_retryable",
            "delivery_failure_summary": (
                "Accepted task delivery did not complete."
            ),
            "updated_at": recovered_at,
        }
    }
    try:
        update_result = await collection.update_many(
            {
                "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
                "state": "delivery_in_progress",
                "updated_at": {"$lte": stale_before_utc},
            },
            update,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to recover accepted task delivery attempts: {exc}"
        ) from exc

    recovered_count = int(update_result.modified_count)
    return recovered_count


async def find_active_accepted_task_for_scope(
    request: AcceptedTaskStatusCheckRequest,
) -> AcceptedTaskDoc | None:
    """Return the newest active task matching a trusted requester scope."""

    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    query = _scope_query(request)
    if query is None:
        return_value = None
        return return_value
    query["state"] = {"$in": list(ACTIVE_ACCEPTED_TASK_STATES)}
    try:
        cursor = (
            collection.find(query, {"_id": 0})
            .sort("updated_at", -1)
            .limit(1)
        )
        rows = await cursor.to_list(length=1)
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to check accepted task status: {exc}"
        ) from exc

    if not rows:
        return_value = None
        return return_value
    return_value: AcceptedTaskDoc = dict(rows[0])
    return return_value


async def mark_accepted_task_running(
    *,
    accepted_task_id: str,
    started_at: str,
) -> AcceptedTaskDoc | None:
    """Mark a pending task running when the worker claims its job."""

    update = {
        "$set": {
            "state": "running",
            "started_at": started_at,
            "updated_at": started_at,
        }
    }
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "state": {"$in": ["pending", "running"]},
        },
        update,
    )
    return task


async def mark_tool_result_ready(
    *,
    accepted_task_id: str,
    artifact_text: str,
    result_summary: str,
    completed_at: str,
    result_kind: str,
    completion_status: str,
    remaining_needs: list[str],
    coding_run_context: dict[str, object] | None,
) -> AcceptedTaskDoc | None:
    """Record a completed artifact result for source-bound delivery."""

    update = {
        "$set": {
            "state": "result_ready",
            "completion_status": completion_status,
            "result_kind": result_kind,
            "artifact_text": artifact_text,
            "result_summary": result_summary,
            "remaining_needs": list(remaining_needs),
            "completed_at": completed_at,
            "updated_at": completed_at,
        }
    }
    if coding_run_context is not None:
        update["$set"]["coding_run_context"] = dict(coding_run_context)
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "state": {"$in": ["running", "result_ready"]},
        },
        update,
    )
    return task


async def mark_accepted_task_failure_ready(
    *,
    accepted_task_id: str,
    failure_summary: str,
    completed_at: str,
    result_kind: str,
    remaining_needs: list[str],
    coding_run_context: dict[str, object] | None,
) -> AcceptedTaskDoc | None:
    """Record a failed executor result for source-bound delivery."""

    update = {
        "$set": {
            "state": "failure_ready",
            "completion_status": "failed",
            "result_kind": result_kind,
            "failure_summary": failure_summary,
            "remaining_needs": list(remaining_needs),
            "completed_at": completed_at,
            "updated_at": completed_at,
        }
    }
    if coding_run_context is not None:
        update["$set"]["coding_run_context"] = dict(coding_run_context)
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "state": {"$in": ["running", "failure_ready"]},
        },
        update,
    )
    return task


async def mark_accepted_task_delivery_in_progress(
    *,
    accepted_task_id: str,
    delivery_tracking_id: str,
    updated_at: str,
) -> AcceptedTaskDoc | None:
    """Claim an accepted-task result for dispatcher delivery."""

    update = {
        "$set": {
            "state": "delivery_in_progress",
            "delivery_tracking_id": delivery_tracking_id,
            "updated_at": updated_at,
        }
    }
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "state": {
                "$in": [
                    "result_ready",
                    "failure_ready",
                    "delivery_retryable",
                ]
            },
        },
        update,
    )
    return task


async def mark_future_speak_accepted_task_delivered(
    *,
    accepted_task_id: str,
    delivered_at: str,
) -> AcceptedTaskDoc | None:
    """Complete a running future-speak task after scheduling succeeds."""

    update = {
        "$set": {
            "state": "delivered",
            "delivered_conversation_message_id": "",
            "delivered_at": delivered_at,
            "updated_at": delivered_at,
        },
        "$unset": {
            "active_identity_key": "",
        },
    }
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "task_kind": "future_speak",
            "state": "running",
        },
        update,
    )
    return task


async def mark_accepted_task_delivered(
    *,
    accepted_task_id: str,
    delivered_conversation_message_id: str,
    delivered_at: str,
) -> AcceptedTaskDoc | None:
    """Mark delivery success and release active duplicate suppression."""

    update = {
        "$set": {
            "state": "delivered",
            "delivered_conversation_message_id": (
                delivered_conversation_message_id
            ),
            "delivered_at": delivered_at,
            "updated_at": delivered_at,
        },
        "$unset": {
            "active_identity_key": "",
        },
    }
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "state": {"$in": ["delivery_in_progress", "delivered"]},
        },
        update,
    )
    return task


async def mark_accepted_task_delivery_failed(
    *,
    accepted_task_id: str,
    failure_summary: str,
    failed_at: str,
) -> AcceptedTaskDoc | None:
    """Record delivery failure while keeping the active task visible."""

    update = {
        "$set": {
            "state": "delivery_retryable",
            "delivery_failure_summary": failure_summary,
            "updated_at": failed_at,
        }
    }
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "state": "delivery_in_progress",
        },
        update,
    )
    return task


async def _add_related_source_message_id(
    collection: Any,
    *,
    active_identity_key: str,
    source_message_id: str,
    observed_at: str,
) -> AcceptedTaskDoc | None:
    """Attach provenance from a duplicate attempt and return the active row."""

    update: dict[str, object] = {
        "$set": {
            "updated_at": observed_at,
        }
    }
    if source_message_id:
        update["$addToSet"] = {
            "related_source_message_ids": source_message_id,
        }
    try:
        document = await collection.find_one_and_update(
            {
                "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
                "active_identity_key": active_identity_key,
            },
            update,
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to load duplicate accepted task: {exc}"
        ) from exc

    if document is None:
        return_value = None
        return return_value
    return_value: AcceptedTaskDoc = dict(document)
    return return_value


async def _update_task(
    query: dict[str, Any],
    update: dict[str, Any],
) -> AcceptedTaskDoc | None:
    """Apply one accepted-task state transition and return the row."""

    query = {"schema_version": ACCEPTED_TASK_SCHEMA_VERSION, **query}
    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    try:
        document = await collection.find_one_and_update(
            query,
            update,
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to update accepted task: {exc}"
        ) from exc

    if document is None:
        return_value = None
        return return_value
    return_value: AcceptedTaskDoc = dict(document)
    return return_value


def _scope_query(request: Mapping[str, object]) -> dict[str, object] | None:
    """Build the trusted requester/channel lookup for progress checks."""

    query: dict[str, object] = {"schema_version": ACCEPTED_TASK_SCHEMA_VERSION}
    for field_name in (
        "source_platform",
        "source_channel_id",
        "source_channel_type",
        "requester_global_user_id",
        "requester_platform_user_id",
    ):
        value = request.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return_value = None
            return return_value
        query[field_name] = value.strip()
    return query


def _validate_v2_task_document(task: AcceptedTaskDoc) -> None:
    """Reject retired task rows before a v2 lifecycle write reaches MongoDB."""

    if task.get("schema_version") != ACCEPTED_TASK_SCHEMA_VERSION:
        raise DatabaseOperationError("accepted task schema_version is invalid")


def _sanitize_prompt_safe_coding_context(
    value: object,
) -> dict[str, object] | None:
    """Validate the persisted public coding projection without coding internals."""

    if not isinstance(value, Mapping):
        return None
    expected_fields = {
        "schema_version",
        "coding_run_ref",
        "status",
        "summary",
        "limitations",
        "allowed_next_actions",
        "followup_open",
    }
    if set(value) != expected_fields:
        return None
    if value["schema_version"] != CODING_RUN_CONTEXT_SCHEMA_VERSION:
        return None
    coding_run_ref = _bounded_context_text(value["coding_run_ref"])
    status = _bounded_context_text(value["status"])
    summary = _bounded_context_text(value["summary"])
    if not coding_run_ref or not status or not summary:
        return None
    limitations = _bounded_context_text_list(value["limitations"])
    allowed_next_actions = _bounded_context_text_list(
        value["allowed_next_actions"],
    )
    if limitations is None or allowed_next_actions is None:
        return None
    followup_open = value["followup_open"]
    if not isinstance(followup_open, bool):
        return None
    context = {
        "schema_version": CODING_RUN_CONTEXT_SCHEMA_VERSION,
        "coding_run_ref": coding_run_ref,
        "status": status,
        "summary": summary,
        "limitations": limitations,
        "allowed_next_actions": allowed_next_actions,
        "followup_open": followup_open,
    }
    return context


def _bounded_context_text(value: object) -> str:
    """Return one bounded text value from an already prompt-safe projection."""

    if not isinstance(value, str):
        return ""
    return value.strip()[:MAX_CODING_RUN_CONTEXT_TEXT_CHARS]


def _bounded_context_text_list(value: object) -> list[str] | None:
    """Return bounded text list values or reject an invalid public projection."""

    if not isinstance(value, list):
        return None
    texts: list[str] = []
    for item in value[:MAX_CODING_RUN_CONTEXT_LIST_ITEMS]:
        text = _bounded_context_text(item)
        if not text:
            return None
        texts.append(text)
    return texts
