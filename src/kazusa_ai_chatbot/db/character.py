"""Operational character state and latest-identity composition."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot.character_identity_growth.models import (
    TOP_LEVEL_IDENTITY_KEYS,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.character_identity_growth import (
    get_current_identity,
)
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.time_boundary import (
    parse_storage_utc_datetime,
    storage_utc_now_iso,
)


RUNTIME_CHARACTER_STATE_FIELDS = (
    "cognition_state",
    "updated_at",
)
_OPERATIONAL_DOCUMENT_KEYS = frozenset({
    "_id",
    *RUNTIME_CHARACTER_STATE_FIELDS,
})


class LegacyCharacterStateError(DatabaseOperationError):
    """Raised when semantic profile data remains in ``character_state``."""


def split_character_profile_runtime_state(
    profile: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split a composed graph profile into identity and runtime state."""

    static_profile = {
        key: deepcopy(profile[key])
        for key in TOP_LEVEL_IDENTITY_KEYS
        if key in profile
    }
    runtime_state = {
        key: deepcopy(profile[key])
        for key in RUNTIME_CHARACTER_STATE_FIELDS
        if key in profile
    }
    return static_profile, runtime_state


def compose_character_profile(
    static_profile: Mapping[str, object],
    runtime_state: Mapping[str, object],
    global_user_id: str,
) -> dict[str, object]:
    """Compose latest semantic identity with operational graph state."""

    return {
        **deepcopy(dict(static_profile)),
        **deepcopy(dict(runtime_state)),
        "global_user_id": global_user_id,
    }


async def ensure_operational_character_state() -> str:
    """Insert or verify the cognition-state-only singleton."""

    db = await get_db()
    existing = await db.character_state.find_one({"_id": "global"})
    if existing is None:
        updated_at = storage_utc_now_iso().replace("+00:00", "Z")
        document = {
            "_id": "global",
            "cognition_state": build_character_production_state(
                updated_at=updated_at,
            ),
            "updated_at": updated_at,
        }
        try:
            await db.character_state.insert_one(deepcopy(document))
        except DuplicateKeyError as exc:
            existing = await db.character_state.find_one({"_id": "global"})
            if existing is None:
                raise DatabaseOperationError(
                    "operational character-state insert raced without "
                    "a readable singleton"
                ) from exc
        else:
            return "inserted"

    _validate_operational_character_state_document(existing)
    return "verified"


async def get_character_profile(
    *,
    character_id: str = CHARACTER_GLOBAL_USER_ID,
) -> dict[str, object]:
    """Compose the latest identity revision with operational state."""

    revision = await get_current_identity(character_id=character_id)
    runtime_state = await get_character_runtime_state()
    return compose_character_profile(
        revision["effective_identity"],
        runtime_state,
        character_id,
    )


async def get_character_runtime_state() -> dict[str, object]:
    """Retrieve and validate the operational singleton state."""

    db = await get_db()
    document = await db.character_state.find_one({"_id": "global"})
    if document is None:
        return {}
    validated = _validate_operational_character_state_document(document)
    validated.pop("_id")
    return validated


async def get_character_state() -> dict[str, object]:
    """Return the operational singleton without semantic identity."""

    return await get_character_runtime_state()


async def get_character_cognition_state() -> dict[str, object]:
    """Read and validate the singleton character cognition state."""

    runtime_state = await get_character_runtime_state()
    cognition_state = runtime_state.get("cognition_state")
    if cognition_state is None:
        raise DatabaseOperationError(
            "global character state document is missing cognition_state"
        )
    return validate_cognition_state(cognition_state)


async def replace_character_cognition_state(state: dict) -> None:
    """Validate and replace the singleton character cognition state."""

    validated_state = validate_cognition_state(state)
    if validated_state["state_scope"] != "character":
        raise ValueError("character cognition state must be character-scoped")
    db = await get_db()
    result = await db.character_state.update_one(
        {"_id": "global"},
        {
            "$set": {
                "cognition_state": validated_state,
                "updated_at": storage_utc_now_iso().replace("+00:00", "Z"),
            }
        },
        upsert=False,
    )
    if result.matched_count != 1:
        raise DatabaseOperationError(
            "global character state document does not exist"
        )


def _validate_operational_character_state_document(
    raw_document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact operational singleton shape."""

    actual_keys = frozenset(raw_document)
    unknown_keys = sorted(actual_keys.difference(_OPERATIONAL_DOCUMENT_KEYS))
    if unknown_keys:
        raise LegacyCharacterStateError(
            "character_state contains semantic or legacy fields "
            f"{unknown_keys}; start from a clean target database"
        )
    missing_keys = sorted(_OPERATIONAL_DOCUMENT_KEYS.difference(actual_keys))
    if missing_keys:
        raise DatabaseOperationError(
            f"character_state is missing operational fields {missing_keys}"
        )
    if raw_document["_id"] != "global":
        raise DatabaseOperationError(
            "character_state singleton must use _id='global'"
        )
    cognition_state = validate_cognition_state(
        raw_document["cognition_state"]
    )
    updated_at = _validate_updated_at(raw_document["updated_at"])
    return {
        "_id": "global",
        "cognition_state": cognition_state,
        "updated_at": updated_at,
    }


def _validate_updated_at(value: object) -> str:
    """Require one timezone-aware ISO runtime timestamp."""

    if not isinstance(value, str) or not value.strip():
        raise DatabaseOperationError(
            "character_state updated_at must be nonempty text"
        )
    text = value.strip()
    try:
        parse_storage_utc_datetime(text)
    except ValueError as exc:
        raise DatabaseOperationError(
            "character_state updated_at must be a storage UTC datetime"
        ) from exc
    return text


__all__ = [
    "LegacyCharacterStateError",
    "RUNTIME_CHARACTER_STATE_FIELDS",
    "compose_character_profile",
    "ensure_operational_character_state",
    "get_character_cognition_state",
    "get_character_profile",
    "get_character_runtime_state",
    "get_character_state",
    "replace_character_cognition_state",
    "split_character_profile_runtime_state",
]
