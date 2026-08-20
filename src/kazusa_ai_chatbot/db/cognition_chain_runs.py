"""MongoDB persistence helpers for sanitized Cognition V3 chain runs."""

from __future__ import annotations

from collections.abc import Mapping

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db._client import get_db

COGNITION_CHAIN_RUNS_COLLECTION = "cognition_chain_runs"
_SCHEMA_VERSION = "cognition_chain_run.v1"
_IMMUTABLE_FIELDS = (
    "run_id",
    "llm_trace_id",
    "cognition_invocation_id",
)
_REQUIRED_FIELDS = (
    "schema_version",
    "chain_run_id",
    "engine",
    "run_id",
    "llm_trace_id",
    "cognition_invocation_id",
    "source_kind",
    "chain_model_name",
    "sidecar_model_name",
    "subconscious_enabled",
    "appraisal_group_count",
    "started_at",
    "completed_at",
    "terminal_disposition",
    "steps",
    "ledger",
    "sidecar",
    "session_events",
    "degradation_markers",
    "warning_codes",
    "expires_at",
)


def _validate_chain_run_document(document: Mapping[str, object]) -> None:
    """Validate the closed sanitized chain-run document shape."""

    if document.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported cognition_chain_run schema")
    for field_name in _REQUIRED_FIELDS:
        if field_name not in document:
            raise ValueError(
                f"cognition_chain_run is missing {field_name}"
            )
    for field_name in ("chain_run_id", "engine", *_IMMUTABLE_FIELDS):
        if not isinstance(document.get(field_name), str) or not document[
            field_name
        ].strip():
            raise ValueError(
                f"cognition_chain_run {field_name} must be non-empty"
            )
    if not isinstance(document["steps"], list):
        raise TypeError("cognition_chain_run steps must be a list")
    if not isinstance(document["ledger"], Mapping):
        raise TypeError("cognition_chain_run ledger must be a mapping")
    if not isinstance(document["sidecar"], Mapping):
        raise TypeError("cognition_chain_run sidecar must be a mapping")
    for field_name in (
        "session_events",
        "degradation_markers",
        "warning_codes",
    ):
        if not isinstance(document[field_name], list):
            raise TypeError(
                f"cognition_chain_run {field_name} must be a list"
            )


async def ensure_cognition_chain_run_indexes() -> None:
    """Create the chain-run collection and exact indexes idempotently."""

    db = await get_db()
    existing = set(await db.list_collection_names())
    if COGNITION_CHAIN_RUNS_COLLECTION not in existing:
        await db.create_collection(COGNITION_CHAIN_RUNS_COLLECTION)

    collection = db[COGNITION_CHAIN_RUNS_COLLECTION]
    await collection.create_index(
        "chain_run_id",
        unique=True,
        name="cognition_chain_run_id_unique",
    )
    await collection.create_index(
        [("run_id", 1), ("llm_trace_id", 1), ("completed_at", -1)],
        name="cognition_chain_run_correlation_completed",
    )
    await collection.create_index(
        [("cognition_invocation_id", 1), ("completed_at", -1)],
        name="cognition_chain_run_invocation_completed",
    )
    await collection.create_index(
        [("engine", 1), ("started_at", 1)],
        name="cognition_chain_run_engine_started",
    )
    await collection.create_index(
        "expires_at",
        expireAfterSeconds=0,
        name="cognition_chain_run_expires_at_ttl",
    )


async def save_cognition_chain_run(
    document: Mapping[str, object],
) -> bool:
    """Validate and idempotently upsert one chain-run document."""

    try:
        _validate_chain_run_document(document)
    except (TypeError, ValueError):
        return False

    db = await get_db()
    collection = db[COGNITION_CHAIN_RUNS_COLLECTION]
    chain_run_id = str(document["chain_run_id"])
    try:
        existing = await collection.find_one(
            {"chain_run_id": chain_run_id},
            projection={"_id": 0},
        )
        if existing is not None:
            for field_name in _IMMUTABLE_FIELDS:
                if existing.get(field_name) != document.get(field_name):
                    return False
            return True
        await collection.insert_one(dict(document))
        return True
    except PyMongoError:
        return False


async def get_cognition_chain_run(
    *,
    run_id: str,
    llm_trace_id: str,
) -> dict[str, object] | None:
    """Read the latest/terminal chain-run row for one exact correlation."""

    if not run_id.strip() or not llm_trace_id.strip():
        return None

    db = await get_db()
    collection = db[COGNITION_CHAIN_RUNS_COLLECTION]
    try:
        cursor = collection.find(
            {
                "run_id": run_id,
                "llm_trace_id": llm_trace_id,
            },
            projection={"_id": 0},
        ).sort("completed_at", -1).limit(1)
        rows = await cursor.to_list(length=1)
    except PyMongoError:
        return None
    if not rows:
        return None
    row = rows[0]
    if (
        row.get("run_id") != run_id
        or row.get("llm_trace_id") != llm_trace_id
    ):
        return None
    return dict(row)


__all__ = [
    "COGNITION_CHAIN_RUNS_COLLECTION",
    "ensure_cognition_chain_run_indexes",
    "get_cognition_chain_run",
    "save_cognition_chain_run",
]
