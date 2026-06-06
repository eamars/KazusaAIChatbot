"""MongoDB persistence helpers for generic background-work jobs."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_JOBS_COLLECTION,
    BackgroundWorkJobDoc,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime


async def ensure_background_work_job_indexes() -> None:
    """Create all idempotent indexes for generic background-work jobs."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        await collection.create_index(
            "job_id",
            unique=True,
            name="background_work_job_id_unique",
        )
        await collection.create_index(
            "idempotency_key",
            unique=True,
            name="background_work_idempotency_unique",
        )
        await collection.create_index(
            [("status", 1), ("created_at", 1)],
            name="background_work_status_created",
        )
        await collection.create_index(
            [("lease_expires_at", 1), ("status", 1)],
            name="background_work_lease_status",
        )
        await collection.create_index(
            [("delivery_state", 1), ("updated_at", 1)],
            name="background_work_delivery_state_updated",
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to ensure background work job indexes: {exc}"
        ) from exc


async def insert_background_work_job(
    job: BackgroundWorkJobDoc,
) -> BackgroundWorkJobDoc:
    """Insert one job or return the existing idempotent row."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        await collection.insert_one(dict(job))
    except DuplicateKeyError:
        existing = await collection.find_one(
            {"idempotency_key": job["idempotency_key"]},
            {"_id": 0},
        )
        if existing is None:
            raise DatabaseOperationError(
                "background work idempotency collision without readable row"
            )
        return_value: BackgroundWorkJobDoc = dict(existing)
        return return_value
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to insert background work job: {exc}"
        ) from exc

    return job


async def claim_background_work_job(
    *,
    lease_owner: str,
    lease_seconds: int,
    now_utc: str,
    max_attempts: int,
) -> BackgroundWorkJobDoc | None:
    """Claim one ready job using a bounded lease."""

    now_dt = parse_storage_utc_datetime(now_utc)
    lease_expires_at = (now_dt + timedelta(seconds=lease_seconds)).isoformat()
    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    claim_filter = {
        "$or": [
            {"status": "queued"},
            {
                "status": "in_progress",
                "lease_expires_at": {"$lte": now_utc},
            },
        ],
        "attempt_count": {"$lt": max_attempts},
    }
    update = {
        "$set": {
            "status": "in_progress",
            "lease_owner": lease_owner,
            "lease_expires_at": lease_expires_at,
            "updated_at": now_utc,
        },
        "$inc": {"attempt_count": 1},
    }
    try:
        document = await collection.find_one_and_update(
            claim_filter,
            update,
            sort=[("created_at", 1)],
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to claim background work job: {exc}"
        ) from exc

    if document is None:
        return_value = None
        return return_value
    return_value: BackgroundWorkJobDoc = dict(document)
    return return_value


async def complete_background_work_job(
    *,
    job_id: str,
    lease_owner: str,
    router_action: str,
    worker: str,
    routed_task: str,
    router_reason: str,
    artifact_text: str,
    result_summary: str,
    worker_metadata: dict[str, object],
    completed_at: str,
) -> BackgroundWorkJobDoc | None:
    """Mark one claimed job completed with bounded worker output."""

    update = {
        "$set": {
            "status": "completed",
            "delivery_state": "ready",
            "router_action": router_action,
            "worker": worker,
            "routed_task": routed_task,
            "router_reason": router_reason,
            "artifact_text": artifact_text,
            "artifact_char_count": len(artifact_text),
            "failure_summary": "",
            "result_summary": result_summary,
            "worker_metadata": dict(worker_metadata),
            "completed_at": completed_at,
            "updated_at": completed_at,
            "lease_owner": None,
            "lease_expires_at": None,
        }
    }
    result = await _update_leased_job(
        job_id=job_id,
        lease_owner=lease_owner,
        update=update,
    )
    return result


async def fail_background_work_job(
    *,
    job_id: str,
    lease_owner: str,
    router_action: str = "",
    worker: str = "",
    routed_task: str = "",
    router_reason: str = "",
    failure_summary: str,
    result_summary: str = "",
    worker_metadata: dict[str, object] | None = None,
    failed_at: str,
) -> BackgroundWorkJobDoc | None:
    """Mark one claimed job failed and ready for failure delivery."""

    update = {
        "$set": {
            "status": "failed",
            "delivery_state": "ready",
            "router_action": router_action,
            "worker": worker,
            "routed_task": routed_task,
            "router_reason": router_reason,
            "failure_summary": failure_summary,
            "result_summary": result_summary,
            "worker_metadata": dict(worker_metadata or {}),
            "updated_at": failed_at,
            "lease_owner": None,
            "lease_expires_at": None,
        }
    }
    result = await _update_leased_job(
        job_id=job_id,
        lease_owner=lease_owner,
        update=update,
    )
    return result


async def find_deliverable_background_work_jobs(
    *,
    limit: int,
) -> list[BackgroundWorkJobDoc]:
    """Return completed or failed jobs that are ready for result delivery."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        cursor = (
            collection.find(
                {
                    "status": {"$in": ["completed", "failed", "delivery_failed"]},
                    "delivery_state": {"$in": ["ready", "failed"]},
                },
                {"_id": 0},
            )
            .sort("updated_at", 1)
            .limit(limit)
        )
        rows = await cursor.to_list(length=limit)
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to find deliverable background work jobs: {exc}"
        ) from exc

    return_value: list[BackgroundWorkJobDoc] = [dict(row) for row in rows]
    return return_value


async def mark_background_work_delivery_in_progress(
    *,
    job_id: str,
    delivery_tracking_id: str,
    started_at: str,
) -> BackgroundWorkJobDoc | None:
    """Mark one ready result as in delivery."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    claim_filter = {
        "job_id": job_id,
        "status": {"$in": ["completed", "failed", "delivery_failed"]},
        "delivery_state": {"$in": ["ready", "failed"]},
    }
    update = {
        "$set": {
            "status": "delivery_in_progress",
            "delivery_state": "in_progress",
            "delivery_tracking_id": delivery_tracking_id,
            "updated_at": started_at,
        },
        "$inc": {"delivery_attempt_count": 1},
    }
    try:
        document = await collection.find_one_and_update(
            claim_filter,
            update,
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to claim background work delivery: {exc}"
        ) from exc

    if document is None:
        return_value = None
        return return_value
    return_value: BackgroundWorkJobDoc = dict(document)
    return return_value


async def mark_background_work_delivered(
    *,
    job_id: str,
    delivered_conversation_message_id: str,
    delivered_at: str,
) -> BackgroundWorkJobDoc | None:
    """Mark one result delivered."""

    update = {
        "$set": {
            "status": "delivered",
            "delivery_state": "delivered",
            "delivered_conversation_message_id": (
                delivered_conversation_message_id
            ),
            "delivered_at": delivered_at,
            "updated_at": delivered_at,
        }
    }
    result = await _update_job(job_id=job_id, update=update)
    return result


async def mark_background_work_delivery_failed(
    *,
    job_id: str,
    failure_summary: str,
    failed_at: str,
) -> BackgroundWorkJobDoc | None:
    """Mark one result delivery failed without deleting worker output."""

    update = {
        "$set": {
            "status": "delivery_failed",
            "delivery_state": "failed",
            "failure_summary": failure_summary,
            "updated_at": failed_at,
        }
    }
    result = await _update_job(job_id=job_id, update=update)
    return result


async def _update_job(
    *,
    job_id: str,
    update: dict[str, Any],
) -> BackgroundWorkJobDoc | None:
    """Apply one state transition and return the updated document."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        document = await collection.find_one_and_update(
            {"job_id": job_id},
            update,
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to update background work job: {exc}"
        ) from exc

    if document is None:
        return_value = None
        return return_value
    return_value: BackgroundWorkJobDoc = dict(document)
    return return_value


async def _update_leased_job(
    *,
    job_id: str,
    lease_owner: str,
    update: dict[str, Any],
) -> BackgroundWorkJobDoc | None:
    """Apply one state transition only for the current worker lease."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        document = await collection.find_one_and_update(
            {
                "job_id": job_id,
                "status": "in_progress",
                "lease_owner": lease_owner,
            },
            update,
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to update background work job: {exc}"
        ) from exc

    if document is None:
        return_value = None
        return return_value
    return_value: BackgroundWorkJobDoc = dict(document)
    return return_value
