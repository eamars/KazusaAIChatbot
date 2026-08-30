"""Durable DSH task-session bindings with revision compare-and-set writes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from inspect import isawaitable
from typing import Any

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.task_resolution.contracts import (
    validate_dsh_resolution_ref,
    validate_dsh_task_source_scope,
    validate_dsh_task_start_spec,
    validate_task_resolution_result,
)

DSH_TASK_BINDINGS_COLLECTION = "dsh_task_bindings"
DSH_TASK_BINDING_SCHEMA_VERSION = "dsh_task_binding.v1"
DSH_TASK_BINDING_INDEXES = (
    {
        "keys": [("task_session_id", 1)],
        "unique": True,
        "name": "dsh_task_binding_session_unique",
    },
    {
        "keys": [("resolution_thread_id", 1)],
        "unique": True,
        "partialFilterExpression": {"resolution_thread_id": {"$type": "string"}},
        "name": "dsh_task_binding_thread_unique",
    },
    {
        "keys": [("current_accepted_task_id", 1)],
        "unique": True,
        "partialFilterExpression": {
            "current_accepted_task_id": {"$type": "string"},
        },
        "name": "dsh_task_binding_current_accepted_task_unique",
    },
    {
        "keys": [("current_background_work_job_id", 1)],
        "unique": True,
        "partialFilterExpression": {
            "current_background_work_job_id": {"$type": "string"},
        },
        "name": "dsh_task_binding_current_background_job_unique",
    },
    {
        "keys": [("state", 1), ("updated_at", -1)],
        "name": "dsh_task_binding_state_updated",
    },
)


async def _collection() -> Any:
    """Return the DSH task-binding Mongo collection."""

    database = get_db()
    if isawaitable(database):
        database = await database
    return database[DSH_TASK_BINDINGS_COLLECTION]


async def ensure_task_binding_indexes(collection: Any | None = None) -> None:
    """Create the exact non-TTL indexes for DSH task bindings."""

    target = collection if collection is not None else await _collection()
    try:
        for index in DSH_TASK_BINDING_INDEXES:
            options = {
                key: value
                for key, value in index.items()
                if key not in {"keys", "name"}
            }
            await target.create_index(
                index["keys"],
                name=index["name"],
                **options,
            )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            "failed to ensure DSH task-binding indexes",
        ) from exc


async def create_task_binding(
    binding: Mapping[str, object],
) -> dict[str, object]:
    """Insert one immutable task binding and return its normalized document."""

    value = _validate_binding(binding)
    collection = await _collection()
    try:
        await collection.insert_one(dict(value))
    except DuplicateKeyError:
        existing = await find_binding_by_session(
            task_session_id=str(value["task_session_id"]),
        )
        if existing is None:
            raise DatabaseOperationError(
                "duplicate DSH task binding could not be loaded",
            )
        if existing != value:
            raise ValueError("task session id was reused with a different binding")
        return existing
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to create DSH task binding") from exc
    return value


async def attach_resolution_ref(
    *,
    task_session_id: str,
    expected_revision: int,
    resolution_ref: Mapping[str, object],
    updated_at: str | None = None,
) -> dict[str, object]:
    """Attach one validated DSH reference under a revision CAS."""

    _nonnegative_int(expected_revision, "expected_revision")
    reference = validate_dsh_resolution_ref(resolution_ref)
    fields: dict[str, object] = {
        "resolution_ref": dict(reference),
        "resolution_thread_id": reference["resolution_thread_id"],
        "segment_id": reference["segment_id"],
    }
    if updated_at is not None:
        fields["updated_at"] = updated_at
    try:
        return await _cas_update(
            task_session_id=task_session_id,
            expected_revision=expected_revision,
            update={"$set": fields, "$inc": {"revision": 1}},
        )
    except ValueError:
        existing = await find_binding_by_session(
            task_session_id=task_session_id,
        )
        if (
            existing is not None
            and existing.get("resolution_ref") == dict(reference)
        ):
            return existing
        raise


async def attach_accepted_task(
    *,
    task_session_id: str,
    expected_revision: int,
    operation_generation: int,
    accepted_task_id: str,
    updated_at: str | None = None,
) -> dict[str, object]:
    """Attach one generation-owned accepted task under the binding CAS."""

    _nonnegative_int(expected_revision, "expected_revision")
    _nonnegative_int(operation_generation, "operation_generation")
    if not isinstance(accepted_task_id, str) or not accepted_task_id.strip():
        raise ValueError("accepted_task_id is required")
    fields: dict[str, object] = {
        "current_accepted_task_id": accepted_task_id.strip(),
    }
    if updated_at is not None:
        fields["updated_at"] = updated_at
    try:
        return await _cas_update(
            task_session_id=task_session_id,
            expected_revision=expected_revision,
            operation_generation=operation_generation,
            update={"$set": fields, "$inc": {"revision": 1}},
        )
    except ValueError:
        existing = await find_binding_by_session(
            task_session_id=task_session_id,
        )
        if (
            existing is not None
            and existing.get("operation_generation") == operation_generation
            and existing.get("current_accepted_task_id") == accepted_task_id
        ):
            return existing
        raise


async def attach_background_job(
    *,
    task_session_id: str,
    expected_revision: int,
    operation_generation: int,
    background_work_job_id: str,
    updated_at: str | None = None,
) -> dict[str, object]:
    """Attach one generation-owned background job under the binding CAS."""

    _nonnegative_int(expected_revision, "expected_revision")
    _nonnegative_int(operation_generation, "operation_generation")
    if not isinstance(background_work_job_id, str) or not background_work_job_id.strip():
        raise ValueError("background_work_job_id is required")
    fields: dict[str, object] = {
        "current_background_work_job_id": background_work_job_id.strip(),
    }
    if updated_at is not None:
        fields["updated_at"] = updated_at
    try:
        return await _cas_update(
            task_session_id=task_session_id,
            expected_revision=expected_revision,
            operation_generation=operation_generation,
            update={"$set": fields, "$inc": {"revision": 1}},
        )
    except ValueError:
        existing = await find_binding_by_session(
            task_session_id=task_session_id,
        )
        if (
            existing is not None
            and existing.get("operation_generation") == operation_generation
            and existing.get("current_background_work_job_id")
            == background_work_job_id
        ):
            return existing
        raise


async def transition_task_binding(
    *,
    task_session_id: str,
    expected_revision: int,
    expected_state: str,
    next_state: str,
    operation_generation: int,
    expected_operation_generation: int | None = None,
    updated_at: str | None = None,
) -> dict[str, object]:
    """Advance one binding state only when state, generation, and revision match."""

    _nonnegative_int(expected_revision, "expected_revision")
    _nonnegative_int(operation_generation, "operation_generation")
    if expected_operation_generation is not None:
        _nonnegative_int(
            expected_operation_generation,
            "expected_operation_generation",
        )
    if not isinstance(expected_state, str) or not expected_state:
        raise ValueError("expected_state is required")
    if not isinstance(next_state, str) or not next_state:
        raise ValueError("next_state is required")
    if not _allowed_transition(expected_state, next_state):
        raise ValueError("DSH task binding state transition is invalid")
    fields: dict[str, object] = {
        "state": next_state,
        "operation_generation": operation_generation,
    }
    if updated_at is not None:
        fields["updated_at"] = updated_at
    try:
        return await _cas_update(
            task_session_id=task_session_id,
            expected_revision=expected_revision,
            expected_state=expected_state,
            operation_generation=operation_generation,
            expected_operation_generation=expected_operation_generation,
            update={"$set": fields, "$inc": {"revision": 1}},
        )
    except ValueError:
        if expected_state != next_state:
            raise
        existing = await find_binding_by_session(
            task_session_id=task_session_id,
        )
        if (
            existing is not None
            and existing.get("state") == next_state
            and existing.get("operation_generation") == operation_generation
        ):
            return existing
        raise


async def reconcile_task_resolution_result(
    *,
    task_session_id: str,
    expected_revision: int,
    operation_generation: int,
    task_resolution_result: Mapping[str, object],
    updated_at: str | None = None,
) -> dict[str, object]:
    """Persist one generation-bound result exactly once under a revision CAS."""

    _nonnegative_int(expected_revision, "expected_revision")
    _nonnegative_int(operation_generation, "operation_generation")
    result = _result_mapping(task_resolution_result)
    collection = await _collection()
    query = {
        "schema_version": DSH_TASK_BINDING_SCHEMA_VERSION,
        "task_session_id": task_session_id,
        "revision": expected_revision,
        "operation_generation": operation_generation,
    }
    fields: dict[str, object] = {"latest_task_resolution_result": result}
    if updated_at is not None:
        fields["updated_at"] = updated_at
    try:
        row = await collection.find_one_and_update(
            query,
            {"$set": fields, "$inc": {"revision": 1}},
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            "failed to reconcile DSH task-resolution result",
        ) from exc
    if row is None:
        existing = await find_binding_by_session(task_session_id=task_session_id)
        if (
            existing is not None
            and existing.get("operation_generation") == operation_generation
            and existing.get("latest_task_resolution_result") == result
        ):
            return existing
        raise ValueError("DSH task-resolution result revision fence rejected")
    return dict(row)


async def find_binding_by_session(
    *,
    task_session_id: str,
) -> dict[str, object] | None:
    """Find one binding by its stable DSH session id."""

    return await _find_one({
        "schema_version": DSH_TASK_BINDING_SCHEMA_VERSION,
        "task_session_id": task_session_id,
    })


async def find_binding_by_thread(
    *,
    resolution_thread_id: str,
) -> dict[str, object] | None:
    """Find one binding by its DSH resolution thread id."""

    return await _find_one({
        "schema_version": DSH_TASK_BINDING_SCHEMA_VERSION,
        "resolution_thread_id": resolution_thread_id,
    })


async def find_binding_by_accepted_task(
    *,
    accepted_task_id: str,
) -> dict[str, object] | None:
    """Find one binding by its current accepted-task identity."""

    return await _find_one({
        "schema_version": DSH_TASK_BINDING_SCHEMA_VERSION,
        "current_accepted_task_id": accepted_task_id,
    })


async def find_binding_by_background_job(
    *,
    background_work_job_id: str,
) -> dict[str, object] | None:
    """Find one binding by its current background-job identity."""

    return await _find_one({
        "schema_version": DSH_TASK_BINDING_SCHEMA_VERSION,
        "current_background_work_job_id": background_work_job_id,
    })


async def _find_one(query: dict[str, object]) -> dict[str, object] | None:
    """Read one binding through the named repository facade."""

    collection = await _collection()
    try:
        row = await collection.find_one(query, {"_id": 0})
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to load DSH task binding") from exc
    return dict(row) if row is not None else None


async def _cas_update(
    *,
    task_session_id: str,
    expected_revision: int,
    update: dict[str, object],
    expected_state: str | None = None,
    operation_generation: int | None = None,
    expected_operation_generation: int | None = None,
) -> dict[str, object]:
    """Apply one strict revision/state/generation compare-and-set update."""

    if not isinstance(task_session_id, str) or not task_session_id:
        raise ValueError("task_session_id is required")
    _nonnegative_int(expected_revision, "expected_revision")
    if operation_generation is not None:
        _nonnegative_int(operation_generation, "operation_generation")
    if expected_operation_generation is not None:
        _nonnegative_int(
            expected_operation_generation,
            "expected_operation_generation",
        )
    query: dict[str, object] = {
        "schema_version": DSH_TASK_BINDING_SCHEMA_VERSION,
        "task_session_id": task_session_id,
        "revision": expected_revision,
    }
    if expected_state is not None:
        query["state"] = expected_state
    if expected_operation_generation is not None:
        query["operation_generation"] = expected_operation_generation
    elif operation_generation is not None:
        query["operation_generation"] = operation_generation
    collection = await _collection()
    try:
        row = await collection.find_one_and_update(
            query,
            update,
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to update DSH task binding") from exc
    if row is None:
        raise ValueError("DSH task binding revision or state fence rejected")
    return dict(row)


def _validate_binding(value: Mapping[str, object]) -> dict[str, object]:
    """Validate exact persistent binding fields before insertion."""

    expected = {
        "schema_version",
        "task_session_id",
        "semantic_objective",
        "goal_continuation_ref",
        "source_scope",
        "state",
        "start_spec",
        "resolution_thread_id",
        "segment_id",
        "resolution_ref",
        "operation_generation",
        "current_accepted_task_id",
        "current_background_work_job_id",
        "latest_task_resolution_result",
        "revision",
        "created_at",
        "updated_at",
    }
    if not isinstance(value, Mapping):
        raise TypeError("DSH task binding must be an object")
    if set(value) != expected:
        raise ValueError("DSH task binding fields are not exact")
    if value["schema_version"] != DSH_TASK_BINDING_SCHEMA_VERSION:
        raise ValueError("DSH task binding schema_version is invalid")
    if value["state"] != "queued":
        raise ValueError("DSH task binding must start in queued state")
    for field in (
        "task_session_id",
        "semantic_objective",
        "created_at",
        "updated_at",
    ):
        if not isinstance(value[field], str) or not value[field].strip():
            raise TypeError(f"DSH task binding {field} is invalid")
    source_scope = validate_dsh_task_source_scope(value["source_scope"])
    start_spec = validate_dsh_task_start_spec(value["start_spec"])
    for field in (
        "resolution_thread_id",
        "segment_id",
        "current_accepted_task_id",
        "current_background_work_job_id",
    ):
        if value[field] is not None and not isinstance(value[field], str):
            raise ValueError(f"DSH task binding {field} is invalid")
    if value["resolution_ref"] is not None:
        validate_dsh_resolution_ref(value["resolution_ref"])
    for field in ("operation_generation", "revision"):
        number = _nonnegative_int(value[field], f"DSH task binding {field}")
        if number != 0:
            raise ValueError(
                f"DSH task binding {field} must start at zero",
            )
    for field in (
        "resolution_thread_id",
        "segment_id",
        "resolution_ref",
        "current_accepted_task_id",
        "current_background_work_job_id",
        "latest_task_resolution_result",
    ):
        if value[field] is not None:
            raise ValueError(
                f"DSH task binding {field} must be empty at creation",
            )
    normalized = deepcopy(dict(value))
    normalized["source_scope"] = source_scope
    normalized["start_spec"] = start_spec
    return normalized


def _result_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize one task-resolution result before persistence."""

    return deepcopy(dict(validate_task_resolution_result(value)))


def _nonnegative_int(value: object, field: str) -> int:
    """Require a strict non-negative integer for every CAS counter."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _allowed_transition(current: str, next_state: str) -> bool:
    """Return whether one durable binding transition is in the closed graph."""

    allowed: dict[str, frozenset[str]] = {
        "queued": frozenset({"opening", "canceled", "faulted"}),
        "opening": frozenset({
            "checkpointed",
            "terminal",
            "canceled",
            "faulted",
            "consumed_inline",
        }),
        "checkpointed": frozenset({"active", "canceled", "faulted"}),
        "active": frozenset({
            "checkpointed",
            "terminal",
            "canceled",
            "faulted",
        }),
        "terminal": frozenset({"active", "canceled", "terminal"}),
        "canceled": frozenset({"canceled"}),
        "faulted": frozenset({"faulted"}),
        "consumed_inline": frozenset({"consumed_inline"}),
    }
    return next_state in allowed.get(current, frozenset())
