"""Operations against the singleton ``character_state`` document."""

from __future__ import annotations

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.schemas import CharacterProfileDoc
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso


RUNTIME_CHARACTER_STATE_FIELDS = (
    "self_image",
    "cognition_state",
    "updated_at",
)


def split_character_profile_runtime_state(profile: dict) -> tuple[dict, dict]:
    """Separate static profile fields from mutable runtime character state.

    Args:
        profile: Full singleton ``character_state`` document with ``_id``
            already stripped or still present.

    Returns:
        A ``(static_profile, runtime_state)`` pair. Runtime fields contain only
        keys present in ``profile``; static fields exclude runtime keys and
        MongoDB's internal ``_id``.
    """
    runtime_field_names = set(RUNTIME_CHARACTER_STATE_FIELDS)
    static_profile = {
        key: value
        for key, value in profile.items()
        if (
            key not in runtime_field_names
            and key != "_id"
        )
    }
    runtime_state = {
        key: value
        for key, value in profile.items()
        if key in runtime_field_names
    }
    return_value = (static_profile, runtime_state)
    return return_value


def compose_character_profile(
    static_profile: dict,
    runtime_state: dict,
    global_user_id: str,
) -> dict:
    """Build the graph-facing character profile for one turn.

    Args:
        static_profile: Process-local immutable profile fields.
        runtime_state: Current mutable runtime state fields.
        global_user_id: Internal character identity for the active adapter
            account.

    Returns:
        Character profile payload consumed by the persona graph.
    """
    character_profile = {
        **static_profile,
        **runtime_state,
        "global_user_id": global_user_id,
    }
    return character_profile


async def get_character_profile() -> CharacterProfileDoc | dict:
    """Retrieve the full ``_id: "global"`` document (profile + runtime state).

    Returns:
        The document with ``_id`` stripped, or an empty dict if missing.
    """
    db = await get_db()
    doc = await db.character_state.find_one({"_id": "global"})
    if doc is None:
        return_value = {}
        return return_value
    doc.pop("_id", None)
    return doc


async def get_character_runtime_state() -> dict:
    """Retrieve only runtime fields from the singleton character document.

    Returns:
        Runtime state fields with ``_id`` stripped, or an empty dict if the
        singleton document is missing.
    """
    db = await get_db()
    projection = {field_name: 1 for field_name in RUNTIME_CHARACTER_STATE_FIELDS}
    doc = await db.character_state.find_one({"_id": "global"}, projection)
    if doc is None:
        return_value = {}
        return return_value
    doc.pop("_id", None)
    return doc


async def save_character_profile(profile: dict) -> None:
    """Persist personality profile fields to the global document.

    Each key in ``profile`` is written as a top-level field on
    ``_id: "global"``. Runtime state fields are untouched.
    """
    db = await get_db()
    await db.character_state.update_one(
        {"_id": "global"},
        {"$set": profile},
        upsert=True,
    )


async def get_character_state() -> CharacterProfileDoc | dict:
    """Alias for :func:`get_character_profile` — returns the same document."""
    return_value = await get_character_profile()
    return return_value


async def get_character_cognition_state() -> dict:
    """Read and validate the singleton character cognition state."""

    db = await get_db()
    document = await db.character_state.find_one(
        {"_id": "global"},
        {"cognition_state": 1},
    )
    if document is None or document.get("cognition_state") is None:
        return build_character_production_state(
            updated_at=storage_utc_now_iso(),
        )
    return validate_cognition_state(document["cognition_state"])


async def replace_character_cognition_state(state: dict) -> None:
    """Validate and replace the singleton character cognition state."""

    validated_state = validate_cognition_state(state)
    if validated_state["state_scope"] != "character":
        raise ValueError("character cognition state must be character-scoped")
    db = await get_db()
    result = await db.character_state.update_one(
        {"_id": "global"},
        {"$set": {"cognition_state": validated_state}},
        upsert=False,
    )
    if result.matched_count != 1:
        raise DatabaseOperationError(
            "global character state document does not exist"
        )


async def upsert_character_self_image(image_doc: dict) -> None:
    """Persist the three-tier character self-image document to ``character_state``.

    Args:
        image_doc: The full three-tier image document to write.
    """
    if not image_doc:
        return
    try:
        db = await get_db()
        await db.character_state.update_one(
            {"_id": "global"},
            {"$set": {"self_image": image_doc}},
            upsert=True,
        )
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to upsert character self image: {exc}"
        ) from exc
