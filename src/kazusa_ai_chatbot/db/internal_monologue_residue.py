"""Persistence helpers for internal monologue residue rows."""

from __future__ import annotations

from collections.abc import Sequence

from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.schemas import (
    InternalMonologueResidueV2Doc,
    validate_internal_monologue_residue_v2_doc,
)
from kazusa_ai_chatbot.internal_monologue_residue.models import ResidueWriteResult

INTERNAL_MONOLOGUE_RESIDUE_COLLECTION = "internal_monologue_residue_state"


async def ensure_internal_monologue_residue_indexes() -> None:
    """Create collection and indexes for residue storage."""

    db = await get_db()
    existing = set(await db.list_collection_names())
    if INTERNAL_MONOLOGUE_RESIDUE_COLLECTION not in existing:
        await db.create_collection(INTERNAL_MONOLOGUE_RESIDUE_COLLECTION)

    collection = db[INTERNAL_MONOLOGUE_RESIDUE_COLLECTION]
    await collection.create_index(
        "residue_id",
        unique=True,
        name="internal_monologue_residue_id_unique",
    )
    await collection.create_index(
        [("scope_key", 1), ("created_at", -1)],
        name="internal_monologue_residue_scope_created",
    )
    await collection.create_index(
        [("character_id", 1), ("scope_kind", 1), ("created_at", -1)],
        name="internal_monologue_residue_character_scope_created",
    )
    await collection.create_index(
        "operation_id",
        unique=True,
        name="internal_monologue_residue_operation_unique",
    )
    await collection.create_index(
        "purge_at",
        expireAfterSeconds=0,
        name="internal_monologue_residue_purge_at_ttl",
    )


async def insert_internal_monologue_residue_row(
    row: InternalMonologueResidueV2Doc,
) -> ResidueWriteResult:
    """Insert one validated residue row.

    Args:
        row: Compact residue row with one semantic `residue_text` string.

    Returns:
        The idempotent disposition and logical residue id.

    Raises:
        DatabaseOperationError: If MongoDB rejects the insert.
    """

    operation_id = row.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise DatabaseOperationError(
            "internal monologue residue operation_id is required"
        )
    try:
        validate_internal_monologue_residue_v2_doc(row)
    except ValueError as exc:
        raise DatabaseOperationError(
            f"internal monologue residue row is invalid: {exc}"
        ) from exc

    try:
        collection = await _collection()
        existing = await collection.find_one(
            {"operation_id": operation_id},
            projection={"_id": 0},
        )
        if existing is not None:
            status = _compare_operation_payload(row, existing)
            return {
                "status": status,
                "residue_id": str(existing.get("residue_id") or ""),
            }
        await collection.insert_one(dict(row))
    except DuplicateKeyError:
        try:
            existing = await collection.find_one(
                {"operation_id": operation_id},
                projection={"_id": 0},
            )
        except PyMongoError as exc:
            raise DatabaseOperationError(
                f"read duplicate internal monologue residue failed: {exc}"
            ) from exc
        if existing is None:
            raise DatabaseOperationError(
                "internal monologue residue duplicate has no durable row"
            )
        status = _compare_operation_payload(row, existing)
        return {
            "status": status,
            "residue_id": str(existing.get("residue_id") or ""),
        }
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"insert internal monologue residue failed: {exc}"
        ) from exc

    residue_id = row["residue_id"]
    return {"status": "written", "residue_id": residue_id}


async def list_internal_monologue_residue_rows(
    *,
    scope_keys: Sequence[str],
    per_scope_limit: int,
) -> list[InternalMonologueResidueV2Doc]:
    """Load recent residue rows for candidate scope keys.

    Args:
        scope_keys: Candidate scope keys already selected by the runtime.
        per_scope_limit: Maximum rows to load for each scope before ranking.

    Returns:
        Rows without MongoDB `_id`, ordered newest-first within each scope.

    Raises:
        DatabaseOperationError: If MongoDB rejects the query.
    """

    rows: list[InternalMonologueResidueV2Doc] = []
    try:
        collection = await _collection()
        for scope_key in scope_keys:
            cursor = (
                collection.find(
                    {
                        "scope_key": scope_key,
                        "schema_version": "internal_monologue_residue.v2",
                    },
                    projection={"_id": 0},
                )
                .sort("created_at", -1)
                .limit(per_scope_limit)
            )
            scope_rows = await cursor.to_list(length=per_scope_limit)
            rows.extend(scope_rows)
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"load internal monologue residue failed: {exc}"
        ) from exc

    return rows


async def _collection():
    """Return the internal monologue residue collection handle."""

    db = await get_db()
    collection = db[INTERNAL_MONOLOGUE_RESIDUE_COLLECTION]
    return collection


def _compare_operation_payload(
    row: InternalMonologueResidueV2Doc,
    existing: dict[str, object],
) -> str:
    """Classify a duplicate operation without exposing document content."""

    comparable_fields = (
        "schema_version",
        "operation_id",
        "character_id",
        "scope_key",
        "scope_kind",
        "platform",
        "platform_channel_id",
        "channel_type",
        "global_user_id",
        "residue_text",
        "source_kind",
        "source_refs",
        "created_at",
        "disposition",
        "purge_at",
    )
    for field_name in comparable_fields:
        if row.get(field_name) != existing.get(field_name):
            return "conflict"
    return "duplicate_same_payload"
