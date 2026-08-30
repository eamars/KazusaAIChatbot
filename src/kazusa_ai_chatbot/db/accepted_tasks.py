"""MongoDB persistence helpers for accepted delayed user tasks."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import uuid4

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.accepted_task.models import (
    ACCEPTED_TASK_SCHEMA_VERSION,
    ACCEPTED_TASKS_COLLECTION,
    ACTIVE_ACCEPTED_TASK_STATES,
    AcceptedTaskCreateResult,
    AcceptedTaskDoc,
    AcceptedTaskStatusCheckRequest,
    DshAcceptedTaskAffordanceV1,
    project_dsh_task_affordance,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError

DSH_FOLLOWUP_UNIQUE_INDEX_NAME = "accepted_task_open_dsh_followup_unique"
DSH_FOLLOWUP_LOOKUP_INDEX_NAME = "accepted_task_scope_dsh_followup_lookup"


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
        await collection.create_index(
            [
                ("dsh_task_session_id", 1),
            ],
            unique=True,
            partialFilterExpression={
                "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
                "task_kind": "task_resolution",
                "dsh_followup_open": True,
            },
            name=DSH_FOLLOWUP_UNIQUE_INDEX_NAME,
        )
        await collection.create_index(
            [
                ("schema_version", 1),
                ("source_platform", 1),
                ("source_channel_id", 1),
                ("source_channel_type", 1),
                ("requester_global_user_id", 1),
                ("requester_platform_user_id", 1),
                ("dsh_followup_open", 1),
                ("updated_at", -1),
            ],
            partialFilterExpression={
                "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
                "task_kind": "task_resolution",
                "dsh_followup_open": True,
            },
            name=DSH_FOLLOWUP_LOOKUP_INDEX_NAME,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to ensure accepted task indexes: {exc}"
        ) from exc


async def claim_dsh_followup(
    *,
    accepted_task_id: str,
    action_attempt_id: str,
    expected_revision: int,
    operation: str,
    instruction: str | None,
    updated_at: str | None = None,
) -> AcceptedTaskDoc:
    """Claim one open DSH follow-up with revision and attempt fencing."""

    if operation not in {"continue", "summarize", "cancel"}:
        raise ValueError("unsupported DSH follow-up operation")
    if not isinstance(action_attempt_id, str) or not action_attempt_id.strip():
        raise ValueError("action_attempt_id is required")
    if operation == "continue" and (
        not isinstance(instruction, str) or not instruction.strip()
    ):
        raise ValueError("continue follow-up instruction is required")
    if operation != "continue" and instruction is not None:
        raise ValueError("instruction is operation-specific")
    claim_query = {
        "accepted_task_id": accepted_task_id,
        "task_kind": "task_resolution",
        "dsh_followup_open": True,
        "revision": expected_revision,
    }
    candidate = await _find_task(claim_query)
    if candidate is None:
        existing = await _find_task({
            "accepted_task_id": accepted_task_id,
            "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
        })
        if (
            existing is not None
            and existing.get("dsh_followup_claim_action_attempt_id")
            == action_attempt_id
        ):
            return existing
        raise ValueError("accepted task follow-up revision or openness mismatch")
    current_generation = candidate.get("dsh_operation_generation", 0) if candidate else 0
    if not isinstance(current_generation, int):
        raise TypeError("accepted task DSH generation is invalid")
    update = {
        "$set": {
            "dsh_followup_open": False,
            "dsh_followup_claim_action_attempt_id": action_attempt_id,
            "revision": expected_revision + 1,
            "updated_at": updated_at or candidate.get("updated_at", ""),
        }
    }
    if operation == "cancel":
        update["$set"].update({
            "state": "cancelled",
            "completion_status": "failed",
            "result_kind": "failed",
            "failure_summary": "The accepted task was canceled.",
        })
    task = await _update_task(claim_query, update)
    if task is None:
        existing = await _find_task({
            "accepted_task_id": accepted_task_id,
            "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
        })
        if (
            existing is not None
            and existing.get("dsh_followup_claim_action_attempt_id")
            == action_attempt_id
        ):
            return existing
        raise ValueError("accepted task follow-up revision or openness mismatch")
    return task


async def create_followup(
    *,
    accepted_task_id: str,
    task_session_id: str,
    operation: str,
    instruction: str | None,
    action_attempt_id: str,
    operation_generation: int,
    binding: Mapping[str, object],
    expected_revision: int | None = None,
    updated_at: str | None = None,
) -> AcceptedTaskDoc:
    """Close one delivered affordance and create its next DSH task row."""

    if operation not in {"continue", "summarize", "cancel"}:
        raise ValueError("unsupported DSH follow-up operation")
    if not isinstance(task_session_id, str) or not task_session_id.strip():
        raise ValueError("task_session_id is required")
    if not isinstance(operation_generation, int) or isinstance(
        operation_generation,
        bool,
    ) or operation_generation < 1:
        raise ValueError("operation_generation is invalid")
    if not isinstance(binding, Mapping):
        raise TypeError("binding is required")
    bound_session = binding.get("task_session_id")
    if bound_session != task_session_id:
        raise ValueError("binding task session does not match follow-up")
    bound_generation = binding.get("operation_generation")
    if (
        not isinstance(bound_generation, int)
        or isinstance(bound_generation, bool)
        or operation_generation != bound_generation + 1
    ):
        raise ValueError("follow-up generation does not match binding")
    if binding.get("state") != "terminal":
        raise ValueError("follow-up requires a terminal binding")
    current = await _find_task({
        "accepted_task_id": accepted_task_id,
        "task_kind": "task_resolution",
        "dsh_task_session_id": task_session_id,
        "dsh_followup_open": True,
        "state": "delivered",
        "dsh_operation_generation": bound_generation,
        **(
            {"revision": expected_revision}
            if expected_revision is not None
            else {}
        ),
    })
    if current is None:
        replay = await _find_task({
            "task_kind": "task_resolution",
            "dsh_task_session_id": task_session_id,
            "dsh_operation_generation": operation_generation,
            "dsh_followup_claim_action_attempt_id": action_attempt_id,
        })
        if replay is not None:
            return replay
        raise ValueError("accepted task follow-up is not open")
    revision = current.get("revision", 0)
    if not isinstance(revision, int) or isinstance(revision, bool):
        raise TypeError("accepted task revision is invalid")
    claimed = await claim_dsh_followup(
        accepted_task_id=accepted_task_id,
        action_attempt_id=action_attempt_id,
        expected_revision=(
            expected_revision if expected_revision is not None else revision
        ),
        operation=operation,
        instruction=instruction,
        updated_at=updated_at,
    )
    if operation == "cancel":
        return claimed
    collection = (await get_db())[ACCEPTED_TASKS_COLLECTION]
    next_task = deepcopy(dict(claimed))
    next_task.update({
        "accepted_task_id": f"task-{uuid4().hex}",
        "active_identity_key": (
            f"{claimed.get('task_identity_key', accepted_task_id)}:dsh:{operation_generation}"
        ),
        "state": "pending",
        "completion_status": "none",
        "result_kind": "none",
        "executor_ref": "",
        "started_at": "",
        "completed_at": "",
        "delivered_at": "",
        "result_summary": "",
        "artifact_text": "",
        "remaining_needs": [],
        "failure_summary": "",
        "delivery_failure_summary": "",
        "delivery_tracking_id": "",
        "delivered_conversation_message_id": "",
        "last_progress_reported_at": "",
        "dsh_operation_generation": operation_generation,
        "dsh_followup_open": False,
        "dsh_followup_claim_action_attempt_id": action_attempt_id,
        "revision": 0,
        "updated_at": updated_at or str(claimed.get("updated_at", "")),
    })
    try:
        await collection.insert_one(next_task)
    except DuplicateKeyError:
        replay = await _find_task({
            "task_kind": "task_resolution",
            "dsh_task_session_id": task_session_id,
            "dsh_operation_generation": operation_generation,
            "dsh_followup_claim_action_attempt_id": action_attempt_id,
        })
        if replay is None:
            raise DatabaseOperationError(
                "DSH follow-up duplicate could not be loaded",
            )
        return replay
    except PyMongoError as exc:
        try:
            await _update_task(
                {
                    "accepted_task_id": accepted_task_id,
                    "task_kind": "task_resolution",
                    "dsh_task_session_id": task_session_id,
                    "dsh_followup_open": False,
                    "dsh_followup_claim_action_attempt_id": action_attempt_id,
                    "revision": revision + 1,
                },
                {
                    "$set": {"dsh_followup_open": True},
                    "$unset": {"dsh_followup_claim_action_attempt_id": ""},
                    "$inc": {"revision": 1},
                },
            )
        except (DatabaseOperationError, PyMongoError):
            pass
        raise DatabaseOperationError("failed to create DSH follow-up task") from exc
    return next_task


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
        duplicate_filter: dict[str, Any] = {
            "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
            "active_identity_key": task["active_identity_key"],
            "goal_continuation_ref": task.get("goal_continuation_ref"),
        }
        incoming_authority = task.get("scheduled_future_speech_authority")
        if incoming_authority is not None:
            duplicate_filter["scheduled_future_speech_authority"] = (
                incoming_authority
            )
        active_task = await _add_related_source_message_id(
            collection,
            source_message_id=source_message_id,
            observed_at=observed_at,
            matching_duplicate_filter=duplicate_filter,
        )
        if active_task is None:
            raise DatabaseOperationError(
                "active accepted task has a different goal_continuation_ref "
                "or scheduled authority"
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


async def load_open_dsh_task_affordances_for_scope(
    *,
    platform: str,
    source_channel_id: str,
    requester_global_user_id: str,
    source_channel_type: str | None = None,
    requester_platform_user_id: str | None = None,
    limit: int = 3,
) -> list[DshAcceptedTaskAffordanceV1]:
    """Load prompt-safe DSH task controls for one exact user scope."""

    if limit < 1:
        return []
    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    query = {
        "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
        "source_platform": platform,
        "source_channel_id": source_channel_id,
        "requester_global_user_id": requester_global_user_id,
        "task_kind": "task_resolution",
        "dsh_followup_open": True,
        "state": {
            "$in": [
                *ACTIVE_ACCEPTED_TASK_STATES,
                "delivered",
            ],
        },
    }
    if isinstance(source_channel_type, str) and source_channel_type.strip():
        query["source_channel_type"] = source_channel_type.strip()
    if (
        isinstance(requester_platform_user_id, str)
        and requester_platform_user_id.strip()
    ):
        query["requester_platform_user_id"] = requester_platform_user_id.strip()
    try:
        cursor = collection.find(query, {"_id": 0}).sort(
            "updated_at",
            -1,
        ).limit(limit)
        rows = await cursor.to_list(length=limit)
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to load open DSH task affordances: {exc}"
        ) from exc
    affordances: list[DshAcceptedTaskAffordanceV1] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            affordances.append(project_dsh_task_affordance(row, row))
        except (TypeError, ValueError):
            continue
    return affordances


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


async def find_open_dsh_followup_for_scope(
    request: AcceptedTaskStatusCheckRequest,
) -> AcceptedTaskDoc | None:
    """Return the sole delivered DSH follow-up for one trusted scope."""

    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    query = _scope_query(request)
    if query is None:
        return None
    query.update({
        "task_kind": "task_resolution",
        "dsh_followup_open": True,
        "state": "delivered",
    })
    try:
        cursor = (
            collection.find(query, {"_id": 0})
            .sort("updated_at", -1)
            .limit(1)
        )
        rows = await cursor.to_list(length=1)
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to load open DSH follow-up: {exc}"
        ) from exc
    if not rows:
        return None
    return dict(rows[0])


async def find_accepted_task_by_id(
    *,
    accepted_task_id: str,
) -> AcceptedTaskDoc | None:
    """Load one accepted-task row by its opaque durable identity."""

    if not isinstance(accepted_task_id, str) or not accepted_task_id.strip():
        raise ValueError("accepted_task_id is required")
    return await _find_task({"accepted_task_id": accepted_task_id.strip()})


async def find_dsh_followup_by_action_attempt(
    *,
    task_session_id: str,
    action_attempt_id: str,
    operation_generation: int,
) -> AcceptedTaskDoc | None:
    """Load a durable DSH follow-up replay for one action attempt."""

    if not isinstance(task_session_id, str) or not task_session_id.strip():
        raise ValueError("task_session_id is required")
    if not isinstance(action_attempt_id, str) or not action_attempt_id.strip():
        raise ValueError("action_attempt_id is required")
    if (
        not isinstance(operation_generation, int)
        or isinstance(operation_generation, bool)
        or operation_generation < 0
    ):
        raise ValueError("operation_generation is invalid")
    return await _find_task({
        "task_kind": "task_resolution",
        "dsh_task_session_id": task_session_id.strip(),
        "dsh_operation_generation": operation_generation,
        "dsh_followup_claim_action_attempt_id": action_attempt_id.strip(),
    })


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
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "state": {
                "$in": ["running", "result_ready"],
            },
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
    task = await _update_task(
        {
            "accepted_task_id": accepted_task_id,
            "state": {
                "$in": ["running", "failure_ready"],
            },
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

    query = {
        "accepted_task_id": accepted_task_id,
        "state": {"$in": ["delivery_in_progress", "delivered"]},
    }
    current = await _find_task(query)
    if current is None:
        return None
    set_fields: dict[str, object] = {
        "state": "delivered",
        "delivered_conversation_message_id": (
            delivered_conversation_message_id
        ),
        "delivered_at": delivered_at,
        "updated_at": delivered_at,
    }
    if current.get("task_kind") == "task_resolution":
        set_fields["dsh_followup_open"] = True
    update = {
        "$set": {
            **set_fields,
        },
        "$unset": {
            "active_identity_key": "",
        },
    }
    task = await _update_task(
        query,
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
    source_message_id: str,
    observed_at: str,
    matching_duplicate_filter: dict[str, Any],
) -> AcceptedTaskDoc | None:
    """Atomically attach provenance only when the duplicate row is identical.

    The comparison and the related-source mutation are one atomic update: the
    stored row is mutated only when its continuation reference and scheduled
    authority exactly match the incoming task. A mismatch returns ``None``
    without touching provenance.
    """

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
            matching_duplicate_filter,
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


async def _find_task(query: Mapping[str, object]) -> AcceptedTaskDoc | None:
    """Load one accepted-task row for a deterministic CAS preflight."""

    db = await get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    try:
        document = await collection.find_one(
            {"schema_version": ACCEPTED_TASK_SCHEMA_VERSION, **dict(query)},
            {"_id": 0},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to load accepted task: {exc}"
        ) from exc
    if document is None:
        return None
    return dict(document)


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
