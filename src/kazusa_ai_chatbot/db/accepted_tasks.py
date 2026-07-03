"""MongoDB persistence helpers for accepted delayed user tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.accepted_task.models import (
    ACCEPTED_TASKS_COLLECTION,
    ACTIVE_ACCEPTED_TASK_STATES,
    AcceptedTaskCreateResult,
    AcceptedTaskDoc,
    AcceptedTaskStatusCheckRequest,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError


async def ensure_accepted_task_indexes() -> None:
    """Create all idempotent indexes for accepted-task lifecycle rows."""

    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    try:
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
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to ensure accepted task indexes: {exc}"
        ) from exc


async def insert_or_get_active_accepted_task(
    task: AcceptedTaskDoc,
    *,
    source_message_id: str,
    observed_at: str,
) -> AcceptedTaskCreateResult:
    """Insert one active task or return the existing duplicate."""

    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    try:
        await collection.insert_one(dict(task))
    except DuplicateKeyError:
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


async def mark_accepted_task_pending(
    *,
    accepted_task_id: str,
    executor_ref: str,
    updated_at: str,
) -> AcceptedTaskDoc | None:
    """Mark an enqueueing task pending after internal job insertion."""

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
            "result_kind": "failure",
            "failure_summary": failure_summary,
            "updated_at": updated_at,
        },
        "$unset": {
            "active_identity_key": "",
        },
    }
    task = await _update_task({"accepted_task_id": accepted_task_id}, update)
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
            "result_kind": "failure",
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


async def mark_accepted_task_result_ready(
    *,
    accepted_task_id: str,
    artifact_text: str,
    result_summary: str,
    completed_at: str,
) -> AcceptedTaskDoc | None:
    """Record a completed artifact result for source-bound delivery."""

    update = {
        "$set": {
            "state": "result_ready",
            "result_kind": "artifact",
            "artifact_text": artifact_text,
            "result_summary": result_summary,
            "completed_at": completed_at,
            "updated_at": completed_at,
        }
    }
    task = await _update_task({"accepted_task_id": accepted_task_id}, update)
    return task


async def mark_accepted_task_failure_ready(
    *,
    accepted_task_id: str,
    failure_summary: str,
    completed_at: str,
) -> AcceptedTaskDoc | None:
    """Record a failed executor result for source-bound delivery."""

    update = {
        "$set": {
            "state": "failure_ready",
            "result_kind": "failure",
            "failure_summary": failure_summary,
            "completed_at": completed_at,
            "updated_at": completed_at,
        }
    }
    task = await _update_task({"accepted_task_id": accepted_task_id}, update)
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
            "state": {"$in": ["result_ready", "failure_ready", "delivery_retryable"]},
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
    task = await _update_task({"accepted_task_id": accepted_task_id}, update)
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
    task = await _update_task({"accepted_task_id": accepted_task_id}, update)
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
            {"active_identity_key": active_identity_key},
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

    query: dict[str, object] = {}
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
