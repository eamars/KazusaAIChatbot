"""MongoDB helpers for protected LLM trace collections."""

from __future__ import annotations

from collections.abc import Mapping
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
