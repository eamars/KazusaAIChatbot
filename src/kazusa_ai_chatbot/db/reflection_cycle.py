"""MongoDB interface for production reflection-run persistence."""

from __future__ import annotations

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.schemas import CharacterReflectionRunDoc


REFLECTION_RUN_COLLECTION = "character_reflection_runs"


async def ensure_reflection_run_indexes() -> None:
    """Create the collection and indexes required by reflection workers."""

    db = await get_db()
    existing = set(await db.list_collection_names())
    if REFLECTION_RUN_COLLECTION not in existing:
        await db.create_collection(REFLECTION_RUN_COLLECTION)

    collection = db[REFLECTION_RUN_COLLECTION]
    await collection.create_index(
        [("run_id", 1)],
        unique=True,
        name="reflection_run_id_unique",
    )
    await collection.create_index(
        [("run_kind", 1), ("character_local_date", 1), ("status", 1)],
        name="reflection_kind_date_status",
    )
    await collection.create_index(
        [("scope.scope_ref", 1), ("hour_start", 1)],
        name="reflection_scope_hour",
    )
    await collection.create_index(
        [("source_reflection_run_ids", 1)],
        name="reflection_source_run_ids",
    )


async def upsert_reflection_run(document: CharacterReflectionRunDoc) -> None:
    """Upsert one reflection run document by its stable run id."""

    run_id = str(document["run_id"]).strip()
    if not run_id:
        raise ValueError("run_id is required")
    payload: CharacterReflectionRunDoc = {
        **document,
        "_id": run_id,
        "run_id": run_id,
    }
    db = await get_db()
    await db[REFLECTION_RUN_COLLECTION].replace_one(
        {"run_id": run_id},
        payload,
        upsert=True,
    )


async def find_reflection_run_by_id(
    run_id: str,
) -> CharacterReflectionRunDoc | None:
    """Return one reflection-run document by logical run id."""

    db = await get_db()
    document = await db[REFLECTION_RUN_COLLECTION].find_one({"run_id": run_id})
    if document is None:
        return_value = None
        return return_value
    return_value: CharacterReflectionRunDoc = dict(document)
    return return_value


async def list_hourly_runs_for_channel_day(
    *,
    scope_ref: str,
    character_local_date: str,
) -> list[CharacterReflectionRunDoc]:
    """Return terminal hourly reflection runs for one scope and local date."""

    db = await get_db()
    cursor = (
        db[REFLECTION_RUN_COLLECTION]
        .find({
            "run_kind": "hourly_slot",
            "scope.scope_ref": scope_ref,
            "character_local_date": character_local_date,
        })
        .sort("hour_start", 1)
    )
    documents = await cursor.to_list(length=None)
    return_value: list[CharacterReflectionRunDoc] = [
        dict(document)
        for document in documents
    ]
    return return_value


async def list_daily_channel_runs(
    *,
    character_local_date: str,
) -> list[CharacterReflectionRunDoc]:
    """Return daily channel reflection runs for one character-local date."""

    db = await get_db()
    cursor = (
        db[REFLECTION_RUN_COLLECTION]
        .find({
            "run_kind": "daily_channel",
            "character_local_date": character_local_date,
        })
        .sort("scope.scope_ref", 1)
    )
    documents = await cursor.to_list(length=None)
    return_value: list[CharacterReflectionRunDoc] = [
        dict(document)
        for document in documents
    ]
    return return_value


async def list_existing_run_ids(run_ids: list[str]) -> set[str]:
    """Return the subset of run ids that already has persisted documents."""

    if not run_ids:
        return_value: set[str] = set()
        return return_value
    db = await get_db()
    cursor = db[REFLECTION_RUN_COLLECTION].find(
        {"run_id": {"$in": run_ids}},
        {"_id": 0, "run_id": 1},
    )
    documents = await cursor.to_list(length=len(run_ids))
    existing = {
        str(document["run_id"])
        for document in documents
        if document.get("run_id")
    }
    return existing
