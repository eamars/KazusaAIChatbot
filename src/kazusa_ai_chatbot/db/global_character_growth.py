"""MongoDB interface for global character growth traits and runs."""

from __future__ import annotations

from collections.abc import Sequence

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.schemas import (
    GlobalCharacterGrowthRunDoc,
    GlobalCharacterGrowthTraitDoc,
)


GLOBAL_CHARACTER_GROWTH_TRAITS_COLLECTION = "global_character_growth_traits"
GLOBAL_CHARACTER_GROWTH_RUNS_COLLECTION = "global_character_growth_runs"


async def ensure_global_character_growth_indexes() -> None:
    """Create collections and indexes required by global growth storage."""

    db = await get_db()
    existing = set(await db.list_collection_names())
    if GLOBAL_CHARACTER_GROWTH_TRAITS_COLLECTION not in existing:
        await db.create_collection(GLOBAL_CHARACTER_GROWTH_TRAITS_COLLECTION)
    if GLOBAL_CHARACTER_GROWTH_RUNS_COLLECTION not in existing:
        await db.create_collection(GLOBAL_CHARACTER_GROWTH_RUNS_COLLECTION)

    traits = db[GLOBAL_CHARACTER_GROWTH_TRAITS_COLLECTION]
    await traits.create_index(
        [("trait_id", 1)],
        unique=True,
        name="global_growth_trait_id_unique",
    )
    await traits.create_index(
        [("status", 1), ("maturity_band", 1), ("updated_at", -1)],
        name="global_growth_trait_status_maturity",
    )
    await traits.create_index(
        [("growth_axis", 1), ("status", 1)],
        name="global_growth_trait_axis_status",
    )
    await traits.create_index(
        [("source_memory_unit_ids", 1)],
        name="global_growth_trait_source_memory",
    )

    runs = db[GLOBAL_CHARACTER_GROWTH_RUNS_COLLECTION]
    await runs.create_index(
        [("run_id", 1)],
        unique=True,
        name="global_growth_run_id_unique",
    )
    await runs.create_index(
        [("status", 1), ("updated_at", -1)],
        name="global_growth_run_status_updated",
    )
    await runs.create_index(
        [("source_memory_unit_ids", 1)],
        name="global_growth_run_source_memory",
    )
    await runs.create_index(
        [("source_reflection_run_ids", 1)],
        name="global_growth_run_source_reflection",
    )


async def list_active_growth_traits(
    *,
    limit: int = 12,
) -> list[GlobalCharacterGrowthTraitDoc]:
    """Return active trait rows for candidate generation."""

    db = await get_db()
    cursor = (
        db[GLOBAL_CHARACTER_GROWTH_TRAITS_COLLECTION]
        .find({"status": "active"})
        .sort("updated_at", -1)
        .limit(limit)
    )
    documents = await cursor.to_list(length=limit)
    return_value: list[GlobalCharacterGrowthTraitDoc] = [
        dict(document)
        for document in documents
    ]
    return return_value


async def list_prompt_visible_growth_traits(
    *,
    limit: int = 3,
) -> list[GlobalCharacterGrowthTraitDoc]:
    """Return active promoted trait rows for runtime prompt projection."""

    db = await get_db()
    cursor = (
        db[GLOBAL_CHARACTER_GROWTH_TRAITS_COLLECTION]
        .find({
            "status": "active",
            "maturity_band": "promoted",
        })
        .sort("updated_at", -1)
        .limit(limit)
    )
    documents = await cursor.to_list(length=limit)
    return_value: list[GlobalCharacterGrowthTraitDoc] = [
        dict(document)
        for document in documents
    ]
    return return_value


async def list_recent_global_character_growth_runs(
    *,
    limit: int,
) -> list[dict]:
    """Return bounded global-growth run audit rows without prompt payloads."""

    db = await get_db()
    cursor = (
        db[GLOBAL_CHARACTER_GROWTH_RUNS_COLLECTION]
        .find(
            {},
            {
                "_id": 0,
                "run_id": 1,
                "status": 1,
                "mode": 1,
                "started_at": 1,
                "updated_at": 1,
                "completed_at": 1,
                "eligible_count": 1,
                "accepted_count": 1,
                "rejected_count": 1,
                "trait_update_count": 1,
                "promoted_count": 1,
                "failure_summary": 1,
            },
        )
        .sort([("updated_at", -1), ("run_id", 1)])
        .limit(limit)
    )
    documents = await cursor.to_list(length=limit)
    return_value = [dict(document) for document in documents]
    return return_value


async def upsert_growth_trait_documents(
    trait_documents: Sequence[GlobalCharacterGrowthTraitDoc],
) -> None:
    """Upsert planned trait rows by stable trait id."""

    if not trait_documents:
        return
    db = await get_db()
    collection = db[GLOBAL_CHARACTER_GROWTH_TRAITS_COLLECTION]
    for document in trait_documents:
        trait_id = str(document["trait_id"]).strip()
        if not trait_id:
            raise ValueError("trait_id is required")
        payload: GlobalCharacterGrowthTraitDoc = {
            **document,
            "_id": trait_id,
            "trait_id": trait_id,
        }
        await collection.replace_one(
            {"trait_id": trait_id},
            payload,
            upsert=True,
        )


async def insert_growth_run_document(
    document: GlobalCharacterGrowthRunDoc,
) -> None:
    """Insert or replace one global-growth run record by run id."""

    run_id = str(document["run_id"]).strip()
    if not run_id:
        raise ValueError("run_id is required")
    payload: GlobalCharacterGrowthRunDoc = {
        **document,
        "_id": run_id,
        "run_id": run_id,
    }
    db = await get_db()
    await db[GLOBAL_CHARACTER_GROWTH_RUNS_COLLECTION].replace_one(
        {"run_id": run_id},
        payload,
        upsert=True,
    )
