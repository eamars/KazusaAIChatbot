"""Durable post-turn lifecycle and character-operational receipts."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Literal

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.config import AUDIT_LOG_TTL_DAYS
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.character import (
    _compare_and_replace_character_cognition_state_in_session,
)
from kazusa_ai_chatbot.db.schemas import (
    CharacterOperationalClaimV1,
    CharacterOperationalReceiptV1,
    PostTurnLifecycleRecordV1,
    PostTurnLifecycleRecordV2,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.logging_retention import expiry_from_storage_iso

POST_TURN_LIFECYCLE_RECORDS_COLLECTION = "post_turn_lifecycle_records"
CHARACTER_OPERATIONAL_RECEIPT_SCHEMA = "character_operational_receipt.v1"
POST_TURN_LIFECYCLE_V2_SCHEMA = "post_turn_lifecycle_record.v2"

_TERMINAL_RECEIPT_STATUSES = frozenset({
    "no_change",
    "committed",
    "failed",
    "timed_out",
})
_RECEIPT_ERROR_CODES = frozenset({
    "capacity_exceeded",
    "input_limit",
    "output_limit",
    "contract_exhausted",
    "provider_exhausted",
    "privacy_rejected",
    "deadline_exceeded",
    "state_rejected",
    "transaction_failed",
    "persistence_failed",
    "version_conflict",
    "route_invalid",
    "source_policy_rejected",
})


def _purge_after(created_at: str) -> str:
    """Compute the configured audit retention timestamp."""

    return expiry_from_storage_iso(
        created_at,
        ttl_days=AUDIT_LOG_TTL_DAYS,
    ).isoformat()


async def ensure_post_turn_lifecycle_record_indexes() -> None:
    """Create indexes for lifecycle record identity, lookup, and retention."""

    db = await get_db()
    collection = db[POST_TURN_LIFECYCLE_RECORDS_COLLECTION]
    await collection.create_index(
        "lifecycle_record_id",
        unique=True,
        name="post_turn_lifecycle_record_id_unique",
    )
    await collection.create_index(
        "source_episode_id",
        unique=True,
        name="post_turn_lifecycle_source_episode_unique",
    )
    await collection.create_index(
        [("delivery_tracking_id", 1), ("created_at", 1)],
        name="post_turn_lifecycle_delivery_created",
    )
    await collection.create_index(
        [
            ("schema_version", 1),
            ("character_operational_receipt.status", 1),
            ("character_operational_receipt.lease_expires_at", 1),
        ],
        name="post_turn_lifecycle_operational_receipt_lease",
    )
    await collection.create_index(
        "purge_after",
        expireAfterSeconds=0,
        name="post_turn_lifecycle_purge_after_ttl",
    )


def _record_without_mongo_id(record: dict) -> dict:
    """Return a comparison-safe lifecycle record."""

    normalized = dict(record)
    normalized.pop("_id", None)
    return normalized


async def upsert_post_turn_lifecycle_record(
    record: PostTurnLifecycleRecordV1,
) -> Literal["inserted", "verified"]:
    """Insert or update the one lifecycle record for an episode.

    Operational carry-over claims create the V2 audit row before response
    exposure.  The ordinary post-turn lifecycle projection later fills the
    mutable delivery/action/status fields without replacing its receipt.
    """

    if record.get("schema_version") != "post_turn_lifecycle_record.v1":
        raise ValueError("unsupported post-turn lifecycle record schema")
    if not record.get("lifecycle_record_id"):
        raise ValueError("post-turn lifecycle record id is required")

    db = await get_db()
    collection = db[POST_TURN_LIFECYCLE_RECORDS_COLLECTION]
    try:
        update_result = await collection.update_one(
            {"lifecycle_record_id": record["lifecycle_record_id"]},
            {"$setOnInsert": dict(record)},
            upsert=True,
        )
    except DuplicateKeyError:
        update_result = None
    existing = await collection.find_one(
        {"lifecycle_record_id": record["lifecycle_record_id"]}
    )
    if existing is None:
        raise RuntimeError(
            "post-turn lifecycle record upsert returned no document"
        )
    if existing.get("schema_version") == POST_TURN_LIFECYCLE_V2_SCHEMA:
        return await _update_v2_lifecycle_projection(
            collection=collection,
            existing=existing,
            record=record,
        )
    if _record_without_mongo_id(existing) != _record_without_mongo_id(record):
        raise ValueError(
            "post-turn lifecycle record conflicts with existing episode record"
        )
    if update_result is not None and update_result.upserted_id is not None:
        return "inserted"
    return "verified"


async def _update_v2_lifecycle_projection(
    *,
    collection: Any,
    existing: Mapping[str, Any],
    record: PostTurnLifecycleRecordV1,
) -> Literal["verified"]:
    """Update the mutable V2 lifecycle projection without touching receipt."""

    stable_fields = {
        "lifecycle_record_id",
        "source_episode_id",
    }
    if any(existing.get(field_name) != record.get(field_name) for field_name in stable_fields):
        raise ValueError("post-turn lifecycle record conflicts with existing episode")
    existing_tracking_id = existing.get("delivery_tracking_id")
    record_tracking_id = record.get("delivery_tracking_id")
    if (
        isinstance(existing_tracking_id, str)
        and existing_tracking_id
        and existing_tracking_id != record_tracking_id
    ):
        raise ValueError("post-turn lifecycle delivery tracking conflicts")
    document = await collection.find_one_and_update(
        {
            "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
            "lifecycle_record_id": record["lifecycle_record_id"],
        },
        {
            "$set": {
                "delivery_tracking_id": record_tracking_id,
                "action_projections": list(record["action_projections"]),
                "status": record["status"],
                "error_codes": list(record["error_codes"]),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if document is None:
        raise RuntimeError("post-turn lifecycle v2 row became unavailable")
    _receipt_from_document(document)
    return "verified"


async def claim_character_operational_receipt(
    *,
    lifecycle_record: PostTurnLifecycleRecordV2,
    sequence: int,
    base_updated_at: str,
    registered_at: str,
    lease_owner: str,
    lease_expires_at: str,
) -> CharacterOperationalClaimV1:
    """Insert or claim one operational receipt before response exposure."""

    _validate_v2_lifecycle_record(lifecycle_record)
    _validate_receipt_registration(
        sequence=sequence,
        base_updated_at=base_updated_at,
        registered_at=registered_at,
        lease_owner=lease_owner,
        lease_expires_at=lease_expires_at,
    )
    source_episode_id = lifecycle_record["source_episode_id"]
    candidate = dict(lifecycle_record)
    candidate["character_operational_receipt"] = _new_pending_receipt(
        source_episode_id=source_episode_id,
        sequence=sequence,
        base_updated_at=base_updated_at,
        registered_at=registered_at,
        lease_owner=lease_owner,
        lease_expires_at=lease_expires_at,
    )
    db = await get_db()
    collection = db[POST_TURN_LIFECYCLE_RECORDS_COLLECTION]
    try:
        document = await collection.find_one_and_update(
            {"source_episode_id": source_episode_id},
            {"$setOnInsert": candidate},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError:
        document = await collection.find_one(
            {"source_episode_id": source_episode_id},
        )
    except PyMongoError as exc:
        raise RuntimeError("character operational receipt claim failed") from exc
    if document is None:
        raise RuntimeError("character operational receipt claim returned no row")
    if document.get("schema_version") != POST_TURN_LIFECYCLE_V2_SCHEMA:
        raise ValueError("existing lifecycle row does not support receipts")
    if not _matching_lifecycle_material(document, candidate):
        raise ValueError("lifecycle record conflicts with existing episode row")
    receipt = _receipt_from_document(document)
    if (
        receipt["status"] == "pending"
        and parse_storage_utc_datetime(receipt["lease_expires_at"])
        <= parse_storage_utc_datetime(registered_at)
    ):
        timed_out = await complete_character_operational_receipt(
            source_episode_id=source_episode_id,
            lease_owner=receipt["lease_owner"],
            status="timed_out",
            completed_at=registered_at,
            error_code="deadline_exceeded",
            attempt_count=receipt["attempt_count"],
        )
        return {
            "claim_status": "terminal",
            "receipt": timed_out,
        }
    if receipt["status"] in _TERMINAL_RECEIPT_STATUSES:
        return {
            "claim_status": "terminal",
            "receipt": receipt,
        }
    if receipt["lease_owner"] != lease_owner:
        return {
            "claim_status": "in_progress",
            "receipt": receipt,
        }
    return {
        "claim_status": "claimed",
        "receipt": receipt,
    }


async def commit_character_operational_update(
    *,
    source_episode_id: str,
    lease_owner: str,
    expected_updated_at: str,
    replacement: Mapping[str, Any],
    completed_at: str,
) -> CharacterOperationalReceiptV1 | Literal["version_conflict"]:
    """Commit one state CAS and terminal receipt in the same transaction."""

    _require_text(source_episode_id, "source_episode_id")
    _require_text(lease_owner, "lease_owner")
    _require_timestamp(expected_updated_at, "expected_updated_at")
    _require_timestamp(completed_at, "completed_at")
    db = await get_db()
    session = await _start_session(db)
    collection = db[POST_TURN_LIFECYCLE_RECORDS_COLLECTION]
    try:
        async with session:
            async with session.start_transaction():
                lifecycle_document = await collection.find_one(
                    {
                        "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
                        "source_episode_id": source_episode_id,
                        "character_operational_receipt.status": "pending",
                        "character_operational_receipt.lease_owner": lease_owner,
                    },
                    session=session,
                )
                if lifecycle_document is None:
                    raise ValueError("character operational lease is stale")
                receipt = _receipt_from_document(lifecycle_document)
                if receipt["base_updated_at"] != expected_updated_at:
                    raise ValueError("character operational base version is stale")
                if (
                    parse_storage_utc_datetime(receipt["lease_expires_at"])
                    <= parse_storage_utc_datetime(completed_at)
                ):
                    document = await collection.find_one_and_update(
                        {
                            "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
                            "source_episode_id": source_episode_id,
                            "character_operational_receipt.status": "pending",
                            "character_operational_receipt.lease_owner": (
                                lease_owner
                            ),
                        },
                        {
                            "$set": {
                                "character_operational_receipt.status": (
                                    "timed_out"
                                ),
                                "character_operational_receipt.completed_at": (
                                    completed_at
                                ),
                                "character_operational_receipt.lease_expires_at": "",
                                "character_operational_receipt.error_code": (
                                    "deadline_exceeded"
                                ),
                            }
                        },
                        return_document=ReturnDocument.AFTER,
                        session=session,
                    )
                    if document is None:
                        raise RuntimeError(
                            "character operational receipt became unavailable"
                        )
                    return _receipt_from_document(document)
                committed = (
                    await _compare_and_replace_character_cognition_state_in_session(
                        db=db,
                        session=session,
                        expected_updated_at=expected_updated_at,
                        replacement=replacement,
                    )
                )
                if not committed:
                    await session.abort_transaction()
                    return "version_conflict"
                replacement_timestamp = replacement.get("updated_at")
                if not isinstance(replacement_timestamp, str):
                    raise ValueError("character replacement updated_at is invalid")
                document = await collection.find_one_and_update(
                    {
                        "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
                        "source_episode_id": source_episode_id,
                        "character_operational_receipt.status": "pending",
                        "character_operational_receipt.lease_owner": lease_owner,
                    },
                    {
                        "$set": {
                            "character_operational_receipt.status": "committed",
                            "character_operational_receipt.committed_updated_at": (
                                replacement_timestamp
                            ),
                            "character_operational_receipt.completed_at": (
                                completed_at
                            ),
                            "character_operational_receipt.lease_expires_at": "",
                            "character_operational_receipt.error_code": None,
                        }
                    },
                    return_document=ReturnDocument.AFTER,
                    session=session,
                )
                if document is None:
                    raise RuntimeError(
                        "character operational receipt became unavailable"
                    )
    except (PyMongoError, RuntimeError) as exc:
        raise RuntimeError("character operational transaction failed") from exc
    return _receipt_from_document(document)


async def complete_character_operational_receipt(
    *,
    source_episode_id: str,
    lease_owner: str,
    status: Literal["no_change", "failed", "timed_out"],
    completed_at: str,
    error_code: str | None,
    attempt_count: int,
) -> CharacterOperationalReceiptV1:
    """Terminalize a claimed receipt when no state transaction is required."""

    _require_text(source_episode_id, "source_episode_id")
    _require_text(lease_owner, "lease_owner")
    if status not in {"no_change", "failed", "timed_out"}:
        raise ValueError("receipt terminal status is invalid")
    _require_timestamp(completed_at, "completed_at")
    _validate_error_code(error_code)
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
        raise ValueError("receipt attempt_count is invalid")
    if not 0 <= attempt_count <= 3:
        raise ValueError("receipt attempt_count is out of range")
    db = await get_db()
    collection = db[POST_TURN_LIFECYCLE_RECORDS_COLLECTION]
    expired_document = await collection.find_one_and_update(
        {
            "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
            "source_episode_id": source_episode_id,
            "character_operational_receipt.status": "pending",
            "character_operational_receipt.lease_owner": lease_owner,
            "character_operational_receipt.lease_expires_at": {
                "$lte": completed_at,
            },
        },
        {
            "$set": {
                "character_operational_receipt.status": "timed_out",
                "character_operational_receipt.completed_at": completed_at,
                "character_operational_receipt.lease_expires_at": "",
                "character_operational_receipt.error_code": "deadline_exceeded",
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if expired_document is not None:
        return _receipt_from_document(expired_document)
    document = await collection.find_one_and_update(
        {
            "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
            "source_episode_id": source_episode_id,
            "character_operational_receipt.status": "pending",
            "character_operational_receipt.lease_owner": lease_owner,
            "character_operational_receipt.lease_expires_at": {
                "$gt": completed_at,
            },
        },
        {
            "$set": {
                "character_operational_receipt.status": status,
                "character_operational_receipt.completed_at": completed_at,
                "character_operational_receipt.lease_expires_at": "",
                "character_operational_receipt.attempt_count": attempt_count,
                "character_operational_receipt.error_code": error_code,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if document is not None:
        return _receipt_from_document(document)
    existing = await collection.find_one({
        "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
        "source_episode_id": source_episode_id,
    })
    if existing is None:
        raise ValueError("character operational receipt does not exist")
    receipt = _receipt_from_document(existing)
    if receipt["status"] in _TERMINAL_RECEIPT_STATUSES:
        return receipt
    raise ValueError("character operational receipt lease is stale")


async def expire_character_operational_receipts(*, now: str) -> int:
    """Expire leased receipts at startup so later work never waits forever."""

    _require_timestamp(now, "now")
    db = await get_db()
    result = await db[POST_TURN_LIFECYCLE_RECORDS_COLLECTION].update_many(
        {
            "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
            "character_operational_receipt.status": "pending",
            "character_operational_receipt.lease_expires_at": {"$lte": now},
        },
        {
            "$set": {
                "character_operational_receipt.status": "timed_out",
                "character_operational_receipt.completed_at": now,
                "character_operational_receipt.lease_expires_at": "",
                "character_operational_receipt.error_code": "deadline_exceeded",
            }
        },
    )
    return int(result.modified_count)


async def get_character_operational_receipt(
    source_episode_id: str,
) -> CharacterOperationalReceiptV1 | None:
    """Load one lifecycle-owned receipt without exposing its source payload."""

    _require_text(source_episode_id, "source_episode_id")
    db = await get_db()
    document = await db[POST_TURN_LIFECYCLE_RECORDS_COLLECTION].find_one({
        "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
        "source_episode_id": source_episode_id,
    })
    if document is None:
        return None
    return _receipt_from_document(document)


def build_character_operational_lifecycle_record(
    *,
    source_episode_id: str,
    created_at: str,
    delivery_tracking_id: str = "",
) -> PostTurnLifecycleRecordV2:
    """Build the minimal v2 lifecycle row owned by operational carry-over."""

    _require_text(source_episode_id, "source_episode_id")
    _require_timestamp(created_at, "created_at")
    if not isinstance(delivery_tracking_id, str):
        raise ValueError("delivery_tracking_id must be text")
    return {
        "schema_version": POST_TURN_LIFECYCLE_V2_SCHEMA,
        "lifecycle_record_id": f"post-turn:{source_episode_id}",
        "source_episode_id": source_episode_id,
        "delivery_tracking_id": delivery_tracking_id,
        "action_projections": [],
        "status": "skipped",
        "error_codes": [],
        "character_operational_receipt": {},
        "created_at": created_at,
        "purge_after": _purge_after(created_at),
    }


def _new_pending_receipt(
    *,
    source_episode_id: str,
    sequence: int,
    base_updated_at: str,
    registered_at: str,
    lease_owner: str,
    lease_expires_at: str,
) -> CharacterOperationalReceiptV1:
    """Build the exact receipt inserted with a v2 lifecycle row."""

    return {
        "schema_version": CHARACTER_OPERATIONAL_RECEIPT_SCHEMA,
        "source_episode_id": source_episode_id,
        "status": "pending",
        "sequence": sequence,
        "durable": True,
        "base_updated_at": base_updated_at,
        "committed_updated_at": "",
        "registered_at": registered_at,
        "completed_at": "",
        "lease_owner": lease_owner,
        "lease_expires_at": lease_expires_at,
        "attempt_count": 0,
        "error_code": None,
    }


def _validate_v2_lifecycle_record(record: Mapping[str, Any]) -> None:
    """Validate stable lifecycle material before its idempotent claim."""

    required = {
        "schema_version",
        "lifecycle_record_id",
        "source_episode_id",
        "delivery_tracking_id",
        "action_projections",
        "status",
        "error_codes",
        "character_operational_receipt",
        "created_at",
        "purge_after",
    }
    if set(record) != required:
        raise ValueError("post-turn lifecycle v2 fields are not exact")
    if record["schema_version"] != POST_TURN_LIFECYCLE_V2_SCHEMA:
        raise ValueError("unsupported post-turn lifecycle v2 schema")
    _require_text(record["lifecycle_record_id"], "lifecycle_record_id")
    _require_text(record["source_episode_id"], "source_episode_id")
    _require_timestamp(record["created_at"], "created_at")
    _require_timestamp(record["purge_after"], "purge_after")
    if not isinstance(record["delivery_tracking_id"], str):
        raise ValueError("delivery_tracking_id must be text")
    if not isinstance(record["action_projections"], list):
        raise ValueError("action_projections must be a list")
    if record["status"] not in {"skipped", "completed", "partial", "failed"}:
        raise ValueError("lifecycle status is invalid")
    if not isinstance(record["error_codes"], list):
        raise ValueError("lifecycle error_codes must be a list")
    if not isinstance(record["character_operational_receipt"], Mapping):
        raise ValueError("lifecycle receipt must be a mapping")


def _validate_receipt_registration(
    *,
    sequence: int,
    base_updated_at: str,
    registered_at: str,
    lease_owner: str,
    lease_expires_at: str,
) -> None:
    """Validate one bounded receipt claim request."""

    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ValueError("receipt sequence is invalid")
    _require_timestamp(base_updated_at, "base_updated_at")
    _require_timestamp(registered_at, "registered_at")
    _require_timestamp(lease_expires_at, "lease_expires_at")
    if parse_storage_utc_datetime(lease_expires_at) <= parse_storage_utc_datetime(
        registered_at,
    ):
        raise ValueError("receipt lease must advance past registration")
    _require_text(lease_owner, "lease_owner", maximum=200)


def _receipt_from_document(document: Mapping[str, Any]) -> CharacterOperationalReceiptV1:
    """Validate and copy one receipt embedded in a lifecycle row."""

    raw_receipt = document.get("character_operational_receipt")
    if not isinstance(raw_receipt, Mapping):
        raise ValueError("lifecycle row is missing its operational receipt")
    required = {
        "schema_version",
        "source_episode_id",
        "status",
        "sequence",
        "durable",
        "base_updated_at",
        "committed_updated_at",
        "registered_at",
        "completed_at",
        "lease_owner",
        "lease_expires_at",
        "attempt_count",
        "error_code",
    }
    if set(raw_receipt) != required:
        raise ValueError("character operational receipt fields are not exact")
    if raw_receipt["schema_version"] != CHARACTER_OPERATIONAL_RECEIPT_SCHEMA:
        raise ValueError("character operational receipt schema is invalid")
    _require_text(raw_receipt["source_episode_id"], "receipt source_episode_id")
    if raw_receipt["status"] not in {"pending", *_TERMINAL_RECEIPT_STATUSES}:
        raise ValueError("character operational receipt status is invalid")
    if (
        isinstance(raw_receipt["sequence"], bool)
        or not isinstance(raw_receipt["sequence"], int)
        or raw_receipt["sequence"] < 1
    ):
        raise ValueError("character operational receipt sequence is invalid")
    if raw_receipt["durable"] is not True:
        raise ValueError("character operational receipt must be durable")
    _require_timestamp(raw_receipt["base_updated_at"], "receipt base_updated_at")
    _require_timestamp(raw_receipt["registered_at"], "receipt registered_at")
    for field_name in (
        "committed_updated_at",
        "completed_at",
        "lease_owner",
        "lease_expires_at",
    ):
        if not isinstance(raw_receipt[field_name], str):
            raise ValueError(f"receipt {field_name} must be text")
    if raw_receipt["status"] == "pending":
        _require_text(raw_receipt["lease_owner"], "receipt lease_owner")
        _require_timestamp(raw_receipt["lease_expires_at"], "receipt lease_expires_at")
        if raw_receipt["completed_at"] or raw_receipt["committed_updated_at"]:
            raise ValueError("pending receipt cannot be completed")
    else:
        _require_timestamp(raw_receipt["completed_at"], "receipt completed_at")
        if raw_receipt["lease_expires_at"]:
            raise ValueError("terminal receipt retains a lease")
        if raw_receipt["status"] == "committed":
            _require_timestamp(
                raw_receipt["committed_updated_at"],
                "receipt committed_updated_at",
            )
        elif raw_receipt["committed_updated_at"]:
            raise ValueError("non-committed receipt has a committed version")
    if (
        isinstance(raw_receipt["attempt_count"], bool)
        or not isinstance(raw_receipt["attempt_count"], int)
        or not 0 <= raw_receipt["attempt_count"] <= 3
    ):
        raise ValueError("receipt attempt_count is invalid")
    _validate_error_code(raw_receipt["error_code"])
    return dict(raw_receipt)  # type: ignore[return-value]


def _matching_lifecycle_material(
    existing: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    """Compare immutable episode identity for an idempotent receipt claim."""

    fields = {
        "schema_version",
        "lifecycle_record_id",
        "source_episode_id",
    }
    return all(
        existing.get(field_name) == candidate.get(field_name)
        for field_name in fields
    )


async def _start_session(db: Any) -> Any:
    """Acquire a Mongo transaction session or fail closed for this commit."""

    client = getattr(db, "client", None)
    start_session = getattr(client, "start_session", None)
    if not callable(start_session):
        raise RuntimeError("MongoDB transaction support is unavailable")
    session = start_session()
    if inspect.isawaitable(session):
        session = await session
    if not hasattr(session, "start_transaction"):
        raise RuntimeError("MongoDB session cannot start transactions")
    return session


def _require_text(value: object, label: str, *, maximum: int = 1000) -> str:
    """Require bounded non-empty lifecycle text."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _require_timestamp(value: object, label: str) -> str:
    """Require a storage UTC timestamp used by lifecycle ordering."""

    text = _require_text(value, label)
    try:
        parse_storage_utc_datetime(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    return text


def _validate_error_code(value: object) -> None:
    """Require a declared public receipt failure code or no code."""

    if value is not None and value not in _RECEIPT_ERROR_CODES:
        raise ValueError("character operational receipt error code is invalid")
