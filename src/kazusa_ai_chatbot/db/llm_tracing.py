"""MongoDB helpers for protected LLM trace collections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.db._client import get_db

LLM_TRACE_RUNS_COLLECTION = "llm_trace_runs"
LLM_TRACE_STEPS_COLLECTION = "llm_trace_steps"


async def ensure_llm_trace_indexes() -> None:
    """Create LLM trace collections and indexes idempotently."""

    db = await get_db()
    existing = set(await db.list_collection_names())
    for collection_name in (
        LLM_TRACE_RUNS_COLLECTION,
        LLM_TRACE_STEPS_COLLECTION,
    ):
        if collection_name not in existing:
            await db.create_collection(collection_name)

    runs = db[LLM_TRACE_RUNS_COLLECTION]
    steps = db[LLM_TRACE_STEPS_COLLECTION]
    await runs.create_index("trace_id", unique=True, name="llm_trace_run_id_unique")
    await runs.create_index(
        [("platform", 1), ("platform_channel_id", 1), ("started_at", -1)],
        name="llm_trace_run_scope_time",
    )
    await runs.create_index(
        "expires_at",
        expireAfterSeconds=0,
        name="llm_trace_runs_expires_at_ttl",
    )
    await steps.create_index("step_id", unique=True, name="llm_trace_step_id_unique")
    await steps.create_index(
        [("trace_id", 1), ("sequence", 1)],
        name="llm_trace_step_trace_sequence",
    )
    await steps.create_index(
        [("stage_name", 1), ("created_at", -1)],
        name="llm_trace_step_stage_time",
    )
    await steps.create_index(
        "expires_at",
        expireAfterSeconds=0,
        name="llm_trace_steps_expires_at_ttl",
    )


async def upsert_trace_run(document: Mapping[str, Any]) -> str:
    """Upsert one trace-run document by trace id."""

    db = await get_db()
    trace_id = str(document["trace_id"])
    await db[LLM_TRACE_RUNS_COLLECTION].update_one(
        {"trace_id": trace_id},
        {"$setOnInsert": dict(document)},
        upsert=True,
    )
    return trace_id


async def update_trace_run(
    *,
    trace_id: str,
    update_doc: Mapping[str, Any],
) -> None:
    """Update mutable trace-run status fields."""

    db = await get_db()
    await db[LLM_TRACE_RUNS_COLLECTION].update_one(
        {"trace_id": trace_id},
        {"$set": dict(update_doc)},
        upsert=False,
    )


async def insert_trace_step(document: Mapping[str, Any]) -> str:
    """Insert one LLM trace step."""

    db = await get_db()
    await db[LLM_TRACE_STEPS_COLLECTION].insert_one(dict(document))
    step_id = str(document["step_id"])
    return step_id


async def list_llm_trace_steps_for_trace_ids(
    trace_ids: Sequence[str],
    *,
    stage_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Return selected parsed trace steps for trace-backed context.

    Args:
        trace_ids: Trace run ids to inspect.
        stage_names: Approved stage names to include.

    Returns:
        Trace-step rows with only parsed output and ordering metadata.
    """

    clean_trace_ids = _unique_strings(trace_ids)
    clean_stage_names = _unique_strings(stage_names)
    if not clean_trace_ids or not clean_stage_names:
        rows: list[dict[str, Any]] = []
        return rows

    db = await get_db()
    query = {
        "trace_id": {"$in": clean_trace_ids},
        "stage_name": {"$in": clean_stage_names},
    }
    projection = {
        "_id": 0,
        "trace_id": 1,
        "stage_name": 1,
        "sequence": 1,
        "parsed_output": 1,
        "created_at": 1,
    }
    cursor = (
        db[LLM_TRACE_STEPS_COLLECTION]
        .find(query, projection)
        .sort([("trace_id", 1), ("sequence", 1)])
    )
    rows = await cursor.to_list(length=None)
    return rows


def _unique_strings(values: Sequence[str]) -> list[str]:
    """Return stripped string values in first-seen order."""

    clean_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        clean_value = str(value).strip()
        if not clean_value or clean_value in seen_values:
            continue
        clean_values.append(clean_value)
        seen_values.add(clean_value)
    return clean_values
