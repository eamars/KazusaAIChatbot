"""Persistence helpers for self-cognition ledgers."""

from __future__ import annotations

from pymongo.errors import PyMongoError

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
        await db.self_cognition_action_attempts.replace_one(
            {"idempotency_key": attempt["idempotency_key"]},
            attempt,
            upsert=True,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to upsert self-cognition action attempt: {exc}"
        ) from exc


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
