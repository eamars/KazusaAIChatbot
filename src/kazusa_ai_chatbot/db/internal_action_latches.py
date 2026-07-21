"""Durable internal-thought continuation latch repository."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from uuid import uuid4

from pymongo import ReturnDocument

from kazusa_ai_chatbot.config import AUDIT_LOG_TTL_DAYS
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.schemas import (
    InternalActionLatchClaimV1,
    InternalActionLatchV1,
)
from kazusa_ai_chatbot.logging_retention import expiry_from_storage_iso
from kazusa_ai_chatbot.time_boundary import (
    parse_storage_utc_datetime,
)

INTERNAL_ACTION_LATCHES_COLLECTION = "internal_action_latches"
_MAX_ATTEMPTS = 3
_LATCH_TTL_SECONDS = 15 * 60
_CLAIM_LEASE_SECONDS = 5 * 60


def _iso_after(value: str, *, seconds: int) -> str:
    """Return a storage timestamp offset from ``value``."""

    timestamp = parse_storage_utc_datetime(value) + timedelta(seconds=seconds)
    return timestamp.isoformat()


def _purge_after(created_at: str) -> str:
    """Return the audit-retention timestamp for a latch."""

    return expiry_from_storage_iso(
        created_at,
        ttl_days=AUDIT_LOG_TTL_DAYS,
    ).isoformat()


def _semantic_latch_fields(document: dict) -> dict:
    """Select immutable fields used to verify idempotent replay."""

    return {
        field_name: document.get(field_name)
        for field_name in (
            "source_episode_id",
            "source_action_attempt_id",
            "continuation_objective",
            "evidence_refs",
            "target_scope",
            "privacy_scope",
            "continuation_depth",
        )
    }


async def ensure_internal_action_latch_indexes() -> None:
    """Create the native latch indexes."""

    db = await get_db()
    collection = db[INTERNAL_ACTION_LATCHES_COLLECTION]
    await collection.create_index(
        "idempotency_key",
        unique=True,
        name="internal_action_latch_idempotency_unique",
    )
    await collection.create_index(
        [("status", 1), ("not_before", 1), ("expires_at", 1)],
        name="internal_action_latch_due",
    )
    await collection.create_index(
        "claim_expires_at",
        name="internal_action_latch_claim_expiry",
    )
    await collection.create_index(
        "purge_after",
        expireAfterSeconds=0,
        name="internal_action_latch_purge_after_ttl",
    )


async def issue_internal_action_latch(
    *,
    source_episode_id: str,
    source_action_attempt_id: str,
    continuation_objective: str,
    evidence_refs: Sequence[dict],
    target_scope: dict,
    privacy_scope: str,
    continuation_depth: int,
    now: str,
) -> InternalActionLatchV1:
    """Insert one idempotent internal-thought continuation latch."""

    if continuation_depth > 1:
        raise ValueError("internal action latch continuation depth exceeds one")
    if continuation_depth < 0:
        raise ValueError("internal action latch continuation depth is negative")
    if not continuation_objective.strip():
        raise ValueError("internal action latch objective must be non-empty")

    idempotency_key = source_action_attempt_id
    document: InternalActionLatchV1 = {
        "schema_version": "internal_action_latch.v1",
        "latch_id": str(uuid4()),
        "idempotency_key": idempotency_key,
        "source_episode_id": source_episode_id,
        "source_action_attempt_id": source_action_attempt_id,
        "continuation_objective": continuation_objective,
        "evidence_refs": list(evidence_refs),
        "target_scope": dict(target_scope),
        "privacy_scope": privacy_scope,
        "continuation_depth": continuation_depth,
        "status": "pending",
        "not_before": now,
        "expires_at": _iso_after(now, seconds=_LATCH_TTL_SECONDS),
        "claimed_by": "",
        "claim_token": "",
        "claim_expires_at": "",
        "attempt_count": 0,
        "max_attempts": _MAX_ATTEMPTS,
        "last_error_code": "",
        "consumed_episode_id": "",
        "created_at": now,
        "updated_at": now,
        "purge_after": _purge_after(now),
    }
    db = await get_db()
    collection = db[INTERNAL_ACTION_LATCHES_COLLECTION]
    existing = await collection.find_one_and_update(
        {"idempotency_key": idempotency_key},
        {"$setOnInsert": document},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if existing is None:
        raise RuntimeError("internal action latch upsert returned no document")

    if _semantic_latch_fields(existing) != _semantic_latch_fields(document):
        raise ValueError(
            "internal action latch idempotency key conflicts with existing "
            "semantic material"
        )
    existing.pop("_id", None)
    return existing


async def claim_due_internal_action_latch(
    *,
    worker_id: str,
    now: str,
) -> InternalActionLatchClaimV1 | None:
    """Atomically claim one due or lease-expired latch."""

    db = await get_db()
    collection = db[INTERNAL_ACTION_LATCHES_COLLECTION]
    claim_token = str(uuid4())
    document = await collection.find_one_and_update(
        {
            "expires_at": {"$gt": now},
            "attempt_count": {"$lt": _MAX_ATTEMPTS},
            "$or": [
                {"status": "pending", "not_before": {"$lte": now}},
                {"status": "claimed", "claim_expires_at": {"$lte": now}},
            ],
        },
        {
            "$set": {
                "status": "claimed",
                "claimed_by": worker_id,
                "claim_token": claim_token,
                "claim_expires_at": _iso_after(
                    now,
                    seconds=_CLAIM_LEASE_SECONDS,
                ),
                "updated_at": now,
            },
            "$inc": {"attempt_count": 1},
        },
        sort=[("not_before", 1), ("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )
    if document is None:
        return None
    document.pop("_id", None)
    return {
        "latch": document,
        "claim_token": claim_token,
    }


async def _transition_claimed_latch(
    *,
    latch_id: str,
    claim_token: str,
    update: dict,
    now: str,
) -> InternalActionLatchV1:
    """Apply one token-guarded latch transition."""

    db = await get_db()
    collection = db[INTERNAL_ACTION_LATCHES_COLLECTION]
    document = await collection.find_one_and_update(
        {
            "latch_id": latch_id,
            "claim_token": claim_token,
            "status": "claimed",
        },
        {"$set": {**update, "updated_at": now}},
        return_document=ReturnDocument.AFTER,
    )
    if document is None:
        raise ValueError("internal action latch claim token is stale")
    document.pop("_id", None)
    return document


async def release_internal_action_latch(
    *,
    latch_id: str,
    claim_token: str,
    retry_at: str,
    error_code: str,
    now: str,
) -> InternalActionLatchV1:
    """Release a technical failure for retry or mark attempts exhausted."""

    db = await get_db()
    collection = db[INTERNAL_ACTION_LATCHES_COLLECTION]
    existing = await collection.find_one(
        {"latch_id": latch_id, "claim_token": claim_token, "status": "claimed"}
    )
    if existing is None:
        raise ValueError("internal action latch claim token is stale")
    next_status = "pending" if existing["attempt_count"] < _MAX_ATTEMPTS else "failed"
    return await _transition_claimed_latch(
        latch_id=latch_id,
        claim_token=claim_token,
        update={
            "status": next_status,
            "not_before": retry_at,
            "last_error_code": error_code,
            "claim_token": "",
            "claimed_by": "",
            "claim_expires_at": "",
        },
        now=now,
    )


async def consume_internal_action_latch(
    *,
    latch_id: str,
    claim_token: str,
    consumed_episode_id: str,
    now: str,
) -> InternalActionLatchV1:
    """Consume a claimed latch exactly once."""

    return await _transition_claimed_latch(
        latch_id=latch_id,
        claim_token=claim_token,
        update={
            "status": "consumed",
            "consumed_episode_id": consumed_episode_id,
            "claim_token": "",
            "claimed_by": "",
            "claim_expires_at": "",
        },
        now=now,
    )


async def fail_internal_action_latch(
    *,
    latch_id: str,
    claim_token: str,
    error_code: str,
    now: str,
) -> InternalActionLatchV1:
    """Mark a claimed latch as terminally failed."""

    return await _transition_claimed_latch(
        latch_id=latch_id,
        claim_token=claim_token,
        update={
            "status": "failed",
            "last_error_code": error_code,
            "claim_token": "",
            "claimed_by": "",
            "claim_expires_at": "",
        },
        now=now,
    )


async def expire_due_internal_action_latches(*, now: str) -> int:
    """Mark unclaimed latches past their expiry as expired."""

    db = await get_db()
    result = await db[INTERNAL_ACTION_LATCHES_COLLECTION].update_many(
        {"status": "pending", "expires_at": {"$lte": now}},
        {
            "$set": {
                "status": "expired",
                "updated_at": now,
            }
        },
    )
    return int(result.modified_count)
