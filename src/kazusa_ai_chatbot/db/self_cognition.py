"""Persistence helpers for self-cognition action attempts."""

from __future__ import annotations

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.schemas import SelfCognitionActionAttemptDoc

SELF_COGNITION_ACTION_ATTEMPTS_COLLECTION = "self_cognition_action_attempts"


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
