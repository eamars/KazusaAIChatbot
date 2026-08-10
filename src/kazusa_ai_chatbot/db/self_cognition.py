"""Persistence helpers for self-cognition ledgers."""

from __future__ import annotations

from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.schemas import (
    SelfCognitionActionAttemptDoc,
    SelfCognitionGroupReviewWindowDoc,
)

SELF_COGNITION_ACTION_ATTEMPTS_COLLECTION = "self_cognition_action_attempts"
SELF_COGNITION_GROUP_REVIEW_WINDOWS_COLLECTION = (
    "self_cognition_group_review_windows"
)

_GROUP_REVIEW_FAILED_STATUSES = {
    "target_binding_failed",
    "review_failed",
}
_GROUP_REVIEW_SKIPPED_STATUSES = {
    "coalesced_skipped",
    "stale_skipped",
}
MAX_GROUP_REVIEW_WINDOW_LIMIT = 100


async def upsert_self_cognition_action_attempt(
    attempt: SelfCognitionActionAttemptDoc,
) -> None:
    """Persist the latest state for one self-cognition action identity.

    Args:
        attempt: Action-attempt row keyed by ``idempotency_key``.

    Raises:
        DatabaseOperationError: When MongoDB rejects the write.
    """

    db = await get_db()
    try:
        collection = db.self_cognition_action_attempts
        key_filter = {"idempotency_key": attempt["idempotency_key"]}
        incoming_source = str(attempt.get("source_llm_trace_id") or "").strip()
        mutable_attempt = dict(attempt)
        mutable_attempt.pop("source_llm_trace_id", None)
        update: dict[str, dict[str, object]] = {"$set": mutable_attempt}
        if incoming_source:
            update["$setOnInsert"] = {
                "source_llm_trace_id": incoming_source,
            }
        await collection.update_one(key_filter, update, upsert=True)
        if incoming_source:
            existing = await collection.find_one(
                key_filter,
                {"_id": 0, "source_llm_trace_id": 1},
            )
            existing_source = (
                str(existing.get("source_llm_trace_id") or "").strip()
                if isinstance(existing, dict)
                else ""
            )
            if existing_source and existing_source != incoming_source:
                await collection.update_one(
                    {
                        **key_filter,
                        "source_llm_trace_id": existing_source,
                    },
                    {
                        "$set": {
                            "correlation_write_status": "conflict",
                            "correlation_conflict_source_llm_trace_id": (
                                incoming_source
                            ),
                        },
                    },
                )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to upsert self-cognition action attempt: {exc}"
        ) from exc


async def reserve_self_cognition_action_attempt(
    attempt: SelfCognitionActionAttemptDoc,
) -> bool:
    """Atomically reserve one source-window action identity before dialog.

    Args:
        attempt: Candidate action-attempt row keyed by its idempotency key.

    Returns:
        True when this call inserted the reservation; False when an existing
        unique identity already owns the source window.

    Raises:
        DatabaseOperationError: When MongoDB rejects the reservation.
    """

    db = await get_db()
    try:
        result = await db.self_cognition_action_attempts.update_one(
            {"idempotency_key": attempt["idempotency_key"]},
            {"$setOnInsert": dict(attempt)},
            upsert=True,
        )
    except DuplicateKeyError:
        return False
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to reserve self-cognition action attempt: {exc}"
        ) from exc
    return bool(result.upserted_id)


async def list_self_cognition_action_attempts(
    *,
    limit: int = 1000,
) -> list[SelfCognitionActionAttemptDoc]:
    """Return recent self-cognition attempts for duplicate suppression.

    Args:
        limit: Maximum number of recent attempt rows to return.

    Returns:
        Recent attempt documents sorted from newest to oldest.
    """

    db = await get_db()
    cursor = (
        db.self_cognition_action_attempts.find({}, {"_id": 0})
        .sort("recorded_at", -1)
        .limit(limit)
    )
    attempts = await cursor.to_list(length=limit)
    return attempts


async def find_self_cognition_group_review_window(
    source_id: str,
) -> SelfCognitionGroupReviewWindowDoc | None:
    """Return one terminal group-review ledger row by source window id.

    Args:
        source_id: Durable group activity-window source identity.

    Returns:
        The stored terminal ledger row without Mongo internals, or ``None``.

    Raises:
        DatabaseOperationError: When MongoDB rejects the read.
    """

    db = await get_db()
    collection = getattr(db, SELF_COGNITION_GROUP_REVIEW_WINDOWS_COLLECTION)
    try:
        document = await collection.find_one(
            {"source_id": source_id},
            {"_id": 0},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to find self-cognition group review window: {exc}"
        ) from exc

    if document is None:
        return_value = None
        return return_value
    return_value: SelfCognitionGroupReviewWindowDoc = dict(document)
    return return_value


async def list_group_review_windows(
    *,
    platform: str,
    platform_channel_id: str,
    limit: int,
) -> list[SelfCognitionGroupReviewWindowDoc]:
    """Return bounded terminal review state for one exact group scope.

    Args:
        platform: Exact source platform.
        platform_channel_id: Exact group-channel identifier.
        limit: Maximum number of terminal review rows to return.

    Returns:
        Review rows ordered from newest to oldest without MongoDB internals.

    Raises:
        DatabaseOperationError: When MongoDB rejects the read.
    """

    effective_limit = max(1, min(limit, MAX_GROUP_REVIEW_WINDOW_LIMIT))
    db = await get_db()
    collection = getattr(db, SELF_COGNITION_GROUP_REVIEW_WINDOWS_COLLECTION)
    try:
        cursor = (
            collection.find(
                {
                    "platform": platform,
                    "platform_channel_id": platform_channel_id,
                    "channel_type": "group",
                },
                {"_id": 0},
            )
            .sort([
                ("reviewed_at", -1),
                ("source_id", 1),
            ])
            .limit(effective_limit)
        )
        documents = await cursor.to_list(length=effective_limit)
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to list self-cognition group review windows: {exc}"
        ) from exc

    windows: list[SelfCognitionGroupReviewWindowDoc] = [
        dict(document)
        for document in documents
    ]
    return windows


async def upsert_self_cognition_group_review_window(
    window: SelfCognitionGroupReviewWindowDoc,
) -> SelfCognitionGroupReviewWindowDoc:
    """Insert one terminal group-review ledger row if it is not recorded.

    Args:
        window: Terminal reviewed-window ledger row keyed by ``source_id``.

    Returns:
        The existing terminal row for this source, or the newly inserted row.

    Raises:
        ValueError: If the row is not a valid terminal ledger status.
        DatabaseOperationError: When MongoDB rejects the read or insert.
    """

    _validate_group_review_window(window)

    db = await get_db()
    collection = getattr(db, SELF_COGNITION_GROUP_REVIEW_WINDOWS_COLLECTION)
    try:
        existing = await collection.find_one(
            {"source_id": window["source_id"]},
            {"_id": 0},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to find self-cognition group review window: {exc}"
        ) from exc

    if existing is not None:
        return_value: SelfCognitionGroupReviewWindowDoc = dict(existing)
        return return_value

    try:
        await collection.insert_one(dict(window))
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to insert self-cognition group review window: {exc}"
        ) from exc

    return window


def _validate_group_review_window(
    window: SelfCognitionGroupReviewWindowDoc,
) -> None:
    """Validate terminal status invariants for reviewed group windows."""

    required_text_fields = (
        "source_id",
        "scope_ref",
        "platform",
        "platform_channel_id",
        "window_start",
        "window_end",
        "reviewed_at",
    )
    for field_name in required_text_fields:
        if not window.get(field_name):
            raise ValueError(f"{field_name} is required")

    if window.get("channel_type") != "group":
        raise ValueError("channel_type must be group")

    status = window.get("status")
    if status == "reviewed":
        _validate_reviewed_group_review_window(window)
        return
    if status in _GROUP_REVIEW_FAILED_STATUSES:
        _validate_failed_group_review_window(window)
        return
    if status in _GROUP_REVIEW_SKIPPED_STATUSES:
        _validate_skipped_group_review_window(window)
        return

    raise ValueError(f"unknown group review window status: {status!r}")


def _validate_reviewed_group_review_window(
    window: SelfCognitionGroupReviewWindowDoc,
) -> None:
    """Validate a successfully reviewed window ledger row."""

    if not window.get("case_id"):
        raise ValueError("case_id is required for reviewed rows")
    if window.get("skip_reason") is not None:
        raise ValueError("skip_reason must be None for reviewed rows")


def _validate_failed_group_review_window(
    window: SelfCognitionGroupReviewWindowDoc,
) -> None:
    """Validate a failed terminal window ledger row."""

    if not window.get("case_id"):
        raise ValueError("case_id is required for failed rows")
    if not window.get("skip_reason"):
        raise ValueError("skip_reason is required for failed rows")


def _validate_skipped_group_review_window(
    window: SelfCognitionGroupReviewWindowDoc,
) -> None:
    """Validate a skipped terminal window ledger row."""

    if not window.get("skip_reason"):
        raise ValueError("skip_reason is required for skipped rows")
    blocked_fields = ("case_id", "selected_route", "dispatch_status")
    for field_name in blocked_fields:
        if window.get(field_name) is not None:
            raise ValueError(f"{field_name} must be None for skipped rows")
