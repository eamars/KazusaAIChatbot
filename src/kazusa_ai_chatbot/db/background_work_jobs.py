"""MongoDB persistence helpers for v2 task-orchestrator background jobs."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_JOB_SCHEMA_VERSION,
    BACKGROUND_WORK_JOBS_COLLECTION,
    BackgroundWorkJobDoc,
)
from kazusa_ai_chatbot.config import BACKGROUND_WORK_DELIVERY_MAX_ATTEMPTS
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.task_resolution.contracts import (
    validate_task_resolution_checkpoint,
    validate_task_resolution_result,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime


async def ensure_background_work_job_indexes() -> None:
    """Create idempotent indexes for the v2 background-work collection."""

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
            [("schema_version", 1), ("status", 1), ("created_at", 1)],
            name="background_work_v2_status_created",
        )
        await collection.create_index(
            [("schema_version", 1), ("lease_expires_at", 1), ("status", 1)],
            name="background_work_v2_lease_status",
        )
        await collection.create_index(
            [("schema_version", 1), ("delivery_state", 1), ("updated_at", 1)],
            name="background_work_v2_delivery_state_updated",
        )
        await collection.create_index(
            [("source_llm_trace_id", 1), ("created_at", 1)],
            name="background_work_v2_source_trace_created",
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to ensure background work job indexes: {exc}"
        ) from exc


async def insert_background_work_job(
    job: BackgroundWorkJobDoc,
) -> BackgroundWorkJobDoc:
    """Insert one v2 job or return the existing idempotent v2 row.

    An existing row is returned only when its stored goal continuation
    reference exactly matches the incoming reference; a mismatch fails closed
    without reusing the stored row.
    """

    _validate_v2_job_document(job)
    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        await collection.insert_one(dict(job))
    except DuplicateKeyError:
        existing = await collection.find_one(
            {
                "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
                "idempotency_key": job["idempotency_key"],
            },
            {"_id": 0},
        )
        if existing is None:
            raise DatabaseOperationError(
                "background work idempotency collision without a v2 row"
            )
        existing_job = dict(existing)
        incoming_ref = job.get("goal_continuation_ref")
        stored_ref = existing_job.get("goal_continuation_ref")
        if incoming_ref != stored_ref:
            raise DatabaseOperationError(
                "background work idempotency row has a different "
                "goal_continuation_ref"
            )
        incoming_source = str(job.get("source_llm_trace_id") or "").strip()
        existing_source = str(
            existing_job.get("source_llm_trace_id") or ""
        ).strip()
        if (
            incoming_source
            and existing_source
            and incoming_source != existing_source
        ):
            conflict_fields = {
                "correlation_write_status": "conflict",
                "correlation_conflict_source_llm_trace_id": incoming_source,
            }
            await collection.update_one(
                {
                    "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
                    "idempotency_key": job["idempotency_key"],
                    "source_llm_trace_id": existing_source,
                },
                {"$set": conflict_fields},
            )
            existing_job.update(conflict_fields)
        return existing_job
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
    """Claim one v2 job using a deterministic bounded lease."""

    now_dt = parse_storage_utc_datetime(now_utc)
    lease_expires_at = (now_dt + timedelta(seconds=lease_seconds)).isoformat()
    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    claim_filter = {
        "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
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
        return None
    return dict(document)


async def checkpoint_background_work_job(
    *,
    job_id: str,
    lease_owner: str,
    checkpoint: dict[str, object],
    updated_at: str,
    task_resolution_result: dict[str, object] | None = None,
    release_for_resume: bool = False,
) -> BackgroundWorkJobDoc | None:
    """Persist a completed dispatch checkpoint under the current worker lease.

    Args:
        job_id: Durable job whose task checkpoint changed.
        lease_owner: Worker that still owns the job lease.
        checkpoint: Validated state after one completed specialist dispatch.
        updated_at: Durable timestamp for the checkpoint write.
        task_resolution_result: Prompt-safe result snapshot paired atomically
            with the checkpoint, when task resolution has started.
        release_for_resume: Releases the lease only when the caller has ended
            the current execution slice.

    Returns:
        The updated job when the active lease still belongs to the caller.
    """

    validated_checkpoint = validate_task_resolution_checkpoint(checkpoint)
    update_fields: dict[str, object] = {
        "worker_payload.checkpoint": dict(validated_checkpoint),
        "updated_at": updated_at,
    }
    if task_resolution_result is not None:
        validated_result = validate_task_resolution_result(
            task_resolution_result,
        )
        update_fields["task_resolution_result"] = dict(validated_result)
    if release_for_resume:
        update_fields.update({
            "status": "queued",
            "delivery_state": "queued",
            "lease_owner": None,
            "lease_expires_at": None,
        })
    update = {"$set": update_fields}
    return await _update_leased_job(
        job_id=job_id,
        lease_owner=lease_owner,
        update=update,
    )


async def complete_background_work_job(
    *,
    job_id: str,
    lease_owner: str,
    task_resolution_result: dict[str, object] | None,
    artifact_text: str,
    result_summary: str,
    completed_at: str,
    skip_result_delivery: bool = False,
) -> BackgroundWorkJobDoc | None:
    """Record one terminal worker result and release its lease.

    Deterministic future-speak jobs complete without a task-resolution result;
    task-orchestrator jobs always retain the validated result projection.
    """

    result_projection: dict[str, object] = {}
    if task_resolution_result is not None:
        result = validate_task_resolution_result(task_resolution_result)
        result_projection = dict(result)
    delivery_state = "delivered" if skip_result_delivery else "ready"
    delivered_at = completed_at if skip_result_delivery else ""
    update = {
        "$set": {
            "status": "completed",
            "delivery_state": delivery_state,
            "task_resolution_result": result_projection,
            "artifact_text": artifact_text,
            "failure_summary": "",
            "result_summary": result_summary,
            "completed_at": completed_at,
            "delivered_at": delivered_at,
            "updated_at": completed_at,
            "lease_owner": None,
            "lease_expires_at": None,
        }
    }
    return await _update_leased_job(
        job_id=job_id,
        lease_owner=lease_owner,
        update=update,
    )


async def fail_background_work_job(
    *,
    job_id: str,
    lease_owner: str,
    failure_summary: str,
    result_summary: str,
    failed_at: str,
    task_resolution_result: dict[str, object] | None = None,
    skip_result_delivery: bool = False,
) -> BackgroundWorkJobDoc | None:
    """Record one terminal operational failure without exposing internals."""

    result_projection: dict[str, object] = {}
    if task_resolution_result is not None:
        result_projection = dict(validate_task_resolution_result(
            task_resolution_result,
        ))
    delivery_state = "delivered" if skip_result_delivery else "ready"
    delivered_at = failed_at if skip_result_delivery else ""
    update = {
        "$set": {
            "status": "failed",
            "delivery_state": delivery_state,
            "task_resolution_result": result_projection,
            "failure_summary": failure_summary,
            "result_summary": result_summary,
            "updated_at": failed_at,
            "delivered_at": delivered_at,
            "lease_owner": None,
            "lease_expires_at": None,
        }
    }
    return await _update_leased_job(
        job_id=job_id,
        lease_owner=lease_owner,
        update=update,
    )


async def find_deliverable_background_work_jobs(
    *,
    limit: int,
    max_delivery_attempts: int = BACKGROUND_WORK_DELIVERY_MAX_ATTEMPTS,
) -> list[BackgroundWorkJobDoc]:
    """Return v2 terminal jobs eligible for result-ready cognition delivery."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        cursor = (
            collection.find(
                {
                    "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
                    "status": {"$in": ["completed", "failed", "delivery_failed"]},
                    "delivery_state": {"$in": ["ready", "failed"]},
                    "delivery_attempt_count": {"$lt": max_delivery_attempts},
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
    return [dict(row) for row in rows]


async def list_recent_background_work_jobs(*, limit: int) -> list[dict[str, Any]]:
    """Return bounded v2 queue rows for read-only diagnostics."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    projection = {
        "_id": 0,
        "job_id": 1,
        "accepted_task_id": 1,
        "source_action_attempt_id": 1,
        "source_llm_trace_id": 1,
        "status": 1,
        "delivery_state": 1,
        "requested_worker": 1,
        "semantic_objective": 1,
        "created_at": 1,
        "updated_at": 1,
        "completed_at": 1,
        "delivery_attempt_count": 1,
        "result_summary": 1,
        "failure_summary": 1,
        "source_platform": 1,
        "source_channel_type": 1,
        "requester_display_name": 1,
    }
    try:
        cursor = (
            collection.find(
                {"schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION},
                projection,
            )
            .sort([("updated_at", -1), ("job_id", 1)])
            .limit(limit)
        )
        rows = await cursor.to_list(length=limit)
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to list recent background work jobs: {exc}"
        ) from exc
    return [dict(row) for row in rows]


async def mark_background_work_delivery_in_progress(
    *,
    job_id: str,
    delivery_tracking_id: str,
    started_at: str,
) -> BackgroundWorkJobDoc | None:
    """Claim one terminal v2 result for source-bound cognition delivery."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    claim_filter = {
        "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
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
        return None
    return dict(document)


async def mark_background_work_delivered(
    *,
    job_id: str,
    delivered_conversation_message_id: str,
    delivered_at: str,
) -> BackgroundWorkJobDoc | None:
    """Mark one delivered v2 result after normal adapter delivery succeeds."""

    return await _update_delivery_claimed_job(
        job_id=job_id,
        update={
            "$set": {
                "status": "delivered",
                "delivery_state": "delivered",
                "delivered_conversation_message_id": (
                    delivered_conversation_message_id
                ),
                "delivered_at": delivered_at,
                "updated_at": delivered_at,
            }
        },
    )


async def mark_background_work_delivery_failed(
    *,
    job_id: str,
    failure_summary: str,
    failed_at: str,
) -> BackgroundWorkJobDoc | None:
    """Keep a v2 result retryable when result-ready cognition fails."""

    return await _update_delivery_claimed_job(
        job_id=job_id,
        update={
            "$set": {
                "status": "delivery_failed",
                "delivery_state": "failed",
                "delivery_failure_summary": failure_summary,
                "updated_at": failed_at,
            }
        },
    )


async def _update_delivery_claimed_job(
    *,
    job_id: str,
    update: dict[str, Any],
) -> BackgroundWorkJobDoc | None:
    """Apply one terminal write only to the current delivery claim."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        document = await collection.find_one_and_update(
            {
                "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
                "job_id": job_id,
                "status": "delivery_in_progress",
                "delivery_state": "in_progress",
            },
            update,
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to update background work delivery claim: {exc}"
        ) from exc
    if document is None:
        return None
    return dict(document)


async def recover_stale_background_work_delivery_in_progress(
    *,
    stale_before_utc: str,
    recovered_at: str,
) -> int:
    """Return interrupted v2 delivery claims to the retryable state."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        update_result = await collection.update_many(
            {
                "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
                "status": "delivery_in_progress",
                "delivery_state": "in_progress",
                "updated_at": {"$lte": stale_before_utc},
            },
            {
                "$set": {
                    "status": "delivery_failed",
                    "delivery_state": "failed",
                    "delivery_failure_summary": (
                        "Background work delivery did not complete."
                    ),
                    "updated_at": recovered_at,
                }
            },
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to recover background work delivery: {exc}"
        ) from exc
    return int(update_result.modified_count)


async def _update_job(
    *,
    job_id: str,
    update: dict[str, Any],
) -> BackgroundWorkJobDoc | None:
    """Apply one v2 job transition and return the normalized row."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        document = await collection.find_one_and_update(
            {
                "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
                "job_id": job_id,
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
        return None
    return dict(document)


async def _update_leased_job(
    *,
    job_id: str,
    lease_owner: str,
    update: dict[str, Any],
) -> BackgroundWorkJobDoc | None:
    """Apply one state transition only for the current v2 worker lease."""

    db = await get_db()
    collection = db[BACKGROUND_WORK_JOBS_COLLECTION]
    try:
        document = await collection.find_one_and_update(
            {
                "schema_version": BACKGROUND_WORK_JOB_SCHEMA_VERSION,
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
        return None
    return dict(document)


def _validate_v2_job_document(job: BackgroundWorkJobDoc) -> None:
    """Reject retired schema rows at the only durable v2 write boundary."""

    if job.get("schema_version") != BACKGROUND_WORK_JOB_SCHEMA_VERSION:
        raise DatabaseOperationError("background work job schema_version is invalid")
