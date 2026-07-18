"""Persistence for immutable post-turn lifecycle audit records."""

from __future__ import annotations

from typing import Literal

from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot.config import AUDIT_LOG_TTL_DAYS
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.schemas import PostTurnLifecycleRecordV1
from kazusa_ai_chatbot.logging_retention import expiry_from_storage_iso

POST_TURN_LIFECYCLE_RECORDS_COLLECTION = "post_turn_lifecycle_records"


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
    """Insert or verify the one lifecycle record for an episode."""

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
    if _record_without_mongo_id(existing) != _record_without_mongo_id(record):
        raise ValueError(
            "post-turn lifecycle record conflicts with existing episode record"
        )
    if update_result is not None and update_result.upserted_id is not None:
        return "inserted"
    return "verified"
