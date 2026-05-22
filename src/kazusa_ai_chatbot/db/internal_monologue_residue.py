"""Persistence helpers for internal monologue residue rows."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.schemas import InternalMonologueResidueDoc

logger = logging.getLogger(__name__)

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


async def insert_internal_monologue_residue_row(
    row: InternalMonologueResidueDoc,
) -> str:
    """Insert one validated residue row.

    Args:
        row: Compact residue row with one semantic `residue_text` string.

    Returns:
        The logical residue id.

    Raises:
        DatabaseOperationError: If MongoDB rejects the insert.
    """

    try:
        collection = await _collection()
        await collection.insert_one(dict(row))
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"insert internal monologue residue failed: {exc}"
        ) from exc

    residue_id = row["residue_id"]
    return residue_id


async def list_internal_monologue_residue_rows(
    *,
    scope_keys: Sequence[str],
    per_scope_limit: int,
) -> list[InternalMonologueResidueDoc]:
    """Load recent residue rows for candidate scope keys.

    Args:
        scope_keys: Candidate scope keys already selected by the runtime.
        per_scope_limit: Maximum rows to load for each scope before ranking.

    Returns:
        Rows without MongoDB `_id`, ordered newest-first within each scope.

    Raises:
        DatabaseOperationError: If MongoDB rejects the query.
    """

    rows: list[InternalMonologueResidueDoc] = []
    try:
        collection = await _collection()
        for scope_key in scope_keys:
            cursor = (
                collection.find(
                    {"scope_key": scope_key},
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
