"""Raw MongoDB owner for standalone resolution lifecycle metadata."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError

RESOLUTION_THREADS_COLLECTION = "resolution_thread_store"
RESOLUTION_THREAD_SCHEMA_VERSION = "resolution_thread_store.v2"


async def _collection() -> Any:
    db = await get_db()
    return db[RESOLUTION_THREADS_COLLECTION]


async def ensure_indexes() -> None:
    """Create idempotent standalone thread indexes outside Brain bootstrap."""

    collection = await _collection()
    try:
        await collection.create_index(
            "resolution_thread_id",
            unique=True,
            name="resolution_thread_id_unique",
        )
        await collection.create_index(
            [("state", 1), ("updated_at", 1)],
            name="resolution_thread_state_updated",
        )
        await collection.create_index(
            [("current_segment_id", 1), ("lease_epoch", -1)],
            name="resolution_thread_segment_lease",
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            "failed to ensure resolution thread indexes"
        ) from exc


async def create_thread_v2(
    *,
    resolution_thread_id: str,
    brain_conversation_ref: str,
    root_goal_ref: str,
    priority: str,
    workspace_root: str,
    workspace_fingerprint: str,
    route_digest: str,
    profile_version: str,
    standard_catalog_digest: str,
    semantic_catalog_digest: str,
    scope_fingerprint: str,
    audience_fingerprint: str,
    policy_epoch: str,
    interaction_id: str,
    segment: Mapping[str, Any],
    now: str,
) -> dict[str, Any]:
    """Insert one strict V2 thread document idempotently by thread id."""

    collection = await _collection()
    continuation = (
        datetime.fromisoformat(now.replace("Z", "+00:00"))
        + timedelta(days=1)
    ).isoformat().replace("+00:00", "Z")
    document: dict[str, Any] = {
        "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
        "resolution_thread_id": resolution_thread_id,
        "brain_conversation_ref": brain_conversation_ref,
        "root_goal_ref": root_goal_ref,
        "current_segment_id": segment["segment_id"],
        "state": "active",
        "priority": priority,
        "workspace_root": workspace_root,
        "workspace_fingerprint": workspace_fingerprint,
        "route_digest": route_digest,
        "profile_version": profile_version,
        "dsh_release": "0.1.1-rc.2",
        "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
        "standard_catalog_digest": standard_catalog_digest,
        "semantic_catalog_digest": semantic_catalog_digest,
        "policy_epoch": policy_epoch,
        "audience_fingerprint": audience_fingerprint,
        "scope_fingerprint": scope_fingerprint,
        "interaction_id": interaction_id,
        "created_at": now,
        "updated_at": now,
        "last_terminal_status": None,
        "continuation_eligible_until": continuation,
        "document_revision": 1,
        "lease_epoch": 0,
        "current_lease": None,
        "segments": [dict(segment)],
        "operations": [],
    }
    try:
        await collection.insert_one(deepcopy(document))
    except DuplicateKeyError:
        existing = await get_thread(resolution_thread_id)
        if existing is None:
            raise DatabaseOperationError(
                "duplicate resolution thread could not be loaded"
            )
        return existing
    except PyMongoError as exc:
        raise DatabaseOperationError(
            "failed to create resolution thread"
        ) from exc
    return document


async def get_thread(resolution_thread_id: str) -> dict[str, Any] | None:
    """Load one V2 thread without exposing the Mongo object id."""

    collection = await _collection()
    try:
        row = await collection.find_one(
            {
                "resolution_thread_id": resolution_thread_id,
                "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
            },
            projection={"_id": 0},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to load resolution thread") from exc
    return dict(row) if row is not None else None


async def get_operation(
    resolution_thread_id: str, operation_id: str
) -> dict[str, Any] | None:
    """Load one semantic operation projection by immutable identity."""

    collection = await _collection()
    try:
        row = await collection.find_one(
            {
                "resolution_thread_id": resolution_thread_id,
                "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
                "operations.operation_id": operation_id,
            },
            projection={"_id": 0, "operations.$": 1},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to load operation") from exc
    if row is None:
        return None
    return dict(row["operations"][0])


async def prepare_operation(
    resolution_thread_id: str,
    operation_id: str,
    payload_digest: str,
    method: str,
    segment_id: str,
    activation_id: str | None,
    lease_epoch: int | None,
) -> dict[str, Any]:
    """Reserve an immutable semantic operation or return its exact duplicate."""

    collection = await _collection()
    existing = await collection.find_one(
        {
            "resolution_thread_id": resolution_thread_id,
            "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
            "operations.operation_id": operation_id,
        },
        projection={"_id": 0, "operations.$": 1},
    )
    if existing is not None:
        operation = dict(existing["operations"][0])
        if operation["operation_payload_digest"] != payload_digest:
            raise DatabaseOperationError("OPERATION_ID_REUSE_MISMATCH")
        return operation
    operation = {
        "operation_id": operation_id,
        "operation_payload_digest": payload_digest,
        "method": method,
        "resolution_thread_id": resolution_thread_id,
        "segment_id": segment_id,
        "activation_id": activation_id,
        "lease_epoch": lease_epoch,
        "dsh_message_source_id": None,
        "disposition": "prepared",
        "last_committed_seq": None,
        "outcome_digest": None,
        "fault_code": None,
    }
    try:
        updated = await collection.find_one_and_update(
            {
                "resolution_thread_id": resolution_thread_id,
                "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
                "operations.operation_id": {"$ne": operation_id},
            },
            {
                "$push": {"operations": operation},
                "$inc": {"document_revision": 1},
            },
            return_document=ReturnDocument.AFTER,
            projection={"_id": 0},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to prepare operation") from exc
    if updated is None:
        concurrent = await prepare_operation(
            resolution_thread_id=resolution_thread_id,
            operation_id=operation_id,
            payload_digest=payload_digest,
            method=method,
            segment_id=segment_id,
            activation_id=activation_id,
            lease_epoch=lease_epoch,
        )
        return concurrent
    return operation


async def acquire_lease(
    resolution_thread_id: str,
    activation_id: str,
    owner_id: str,
    expires_at: str,
    now: str,
) -> dict[str, Any]:
    """Acquire an absent or expired lease and allocate a monotonic epoch."""

    collection = await _collection()
    try:
        document = await collection.find_one_and_update(
            {
                "resolution_thread_id": resolution_thread_id,
                "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
                "$or": [
                    {"current_lease": None},
                    {"current_lease.expires_at": {"$lte": now}},
                ],
            },
            [
                {
                    "$set": {
                        "lease_epoch": {"$add": ["$lease_epoch", 1]},
                        "document_revision": {
                            "$add": ["$document_revision", 1]
                        },
                    }
                },
                {
                    "$set": {
                        "current_lease": {
                            "activation_id": activation_id,
                            "lease_epoch": "$lease_epoch",
                            "owner_id": owner_id,
                            "expires_at": expires_at,
                        },
                        "updated_at": now,
                    }
                },
            ],
            return_document=ReturnDocument.AFTER,
            projection={"_id": 0},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to acquire resolution lease") from exc
    if document is None:
        raise DatabaseOperationError("resolution lease is already active")
    lease = dict(document["current_lease"])
    return lease


async def update_operation(
    resolution_thread_id: str,
    operation_id: str,
    *,
    disposition: str,
    dsh_message_source_id: str | None = None,
    last_committed_seq: int | None = None,
    outcome_digest: str | None = None,
    fault_code: str | None = None,
) -> dict[str, Any]:
    """CAS-update one previously admitted semantic operation."""

    collection = await _collection()
    fields = {
        "operations.$.disposition": disposition,
        "operations.$.dsh_message_source_id": dsh_message_source_id,
        "operations.$.last_committed_seq": last_committed_seq,
        "operations.$.outcome_digest": outcome_digest,
        "operations.$.fault_code": fault_code,
    }
    try:
        document = await collection.find_one_and_update(
            {
                "resolution_thread_id": resolution_thread_id,
                "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
                "operations.operation_id": operation_id,
            },
            {"$set": fields, "$inc": {"document_revision": 1}},
            return_document=ReturnDocument.AFTER,
            projection={"_id": 0},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to update operation") from exc
    if document is None:
        raise DatabaseOperationError("operation does not exist")
    return next(
        dict(item)
        for item in document["operations"]
        if item["operation_id"] == operation_id
    )


async def renew_lease(
    resolution_thread_id: str,
    activation_id: str,
    lease_epoch: int,
    expires_at: str,
) -> dict[str, Any]:
    """Renew only the exact current fenced lease."""

    collection = await _collection()
    document = await collection.find_one_and_update(
        {
            "resolution_thread_id": resolution_thread_id,
            "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
            "current_lease.activation_id": activation_id,
            "current_lease.lease_epoch": lease_epoch,
        },
        {
            "$set": {"current_lease.expires_at": expires_at},
            "$inc": {"document_revision": 1},
        },
        return_document=ReturnDocument.AFTER,
        projection={"_id": 0},
    )
    if document is None:
        raise DatabaseOperationError("STALE_ACTIVATION_OR_LEASE")
    return dict(document["current_lease"])


async def release_lease(
    resolution_thread_id: str, activation_id: str, lease_epoch: int
) -> None:
    """Release only the exact current fenced lease."""

    collection = await _collection()
    result = await collection.update_one(
        {
            "resolution_thread_id": resolution_thread_id,
            "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
            "current_lease.activation_id": activation_id,
            "current_lease.lease_epoch": lease_epoch,
        },
        {"$set": {"current_lease": None}, "$inc": {"document_revision": 1}},
    )
    if result.modified_count != 1:
        raise DatabaseOperationError("STALE_ACTIVATION_OR_LEASE")


async def validate_fence(
    resolution_thread_id: str, activation_id: str, lease_epoch: int
) -> dict[str, Any]:
    """Return the exact active lease or reject stale control."""

    collection = await _collection()
    document = await collection.find_one(
        {
            "resolution_thread_id": resolution_thread_id,
            "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
            "current_lease.activation_id": activation_id,
            "current_lease.lease_epoch": lease_epoch,
        },
        projection={"_id": 0, "current_lease": 1},
    )
    if document is None:
        raise DatabaseOperationError("STALE_ACTIVATION_OR_LEASE")
    return dict(document["current_lease"])


async def rotate_segment(
    resolution_thread_id: str, segment: Mapping[str, Any], *, reason: str
) -> dict[str, Any]:
    """Atomically append and select a compatibility-rotated segment."""

    collection = await _collection()
    current = await get_thread(resolution_thread_id)
    if current is None:
        raise DatabaseOperationError("resolution thread does not exist")
    value = dict(segment)
    value["rotation_reason"] = reason
    value["parent_segment_id"] = current["current_segment_id"]
    document = await collection.find_one_and_update(
        {
            "resolution_thread_id": resolution_thread_id,
            "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
            "current_segment_id": current["current_segment_id"],
            "document_revision": current["document_revision"],
        },
        {
            "$push": {"segments": value},
            "$set": {
                "current_segment_id": value["segment_id"],
                "current_lease": None,
            },
            "$inc": {"document_revision": 1},
        },
        return_document=ReturnDocument.AFTER,
        projection={"_id": 0},
    )
    if document is None:
        raise DatabaseOperationError("concurrent segment rotation")
    return dict(document)


async def update_segment(
    resolution_thread_id: str, segment_id: str, **changes: Any
) -> dict[str, Any]:
    """Update the bounded mutable projection of one segment."""

    allowed = {"dsh_session_id", "last_committed_seq", "state", "last_used_at"}
    if set(changes) - allowed:
        raise DatabaseOperationError("unsupported segment update")
    collection = await _collection()
    fields = {f"segments.$.{key}": value for key, value in changes.items()}
    document = await collection.find_one_and_update(
        {
            "resolution_thread_id": resolution_thread_id,
            "schema_version": RESOLUTION_THREAD_SCHEMA_VERSION,
            "segments.segment_id": segment_id,
        },
        {"$set": fields, "$inc": {"document_revision": 1}},
        return_document=ReturnDocument.AFTER,
        projection={"_id": 0},
    )
    if document is None:
        raise DatabaseOperationError("segment does not exist")
    return dict(document)


async def delete_thread(resolution_thread_id: str) -> None:
    """Delete one explicitly identified test or abandoned thread document."""

    collection = await _collection()
    try:
        await collection.delete_one({"resolution_thread_id": resolution_thread_id})
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to delete resolution thread") from exc
