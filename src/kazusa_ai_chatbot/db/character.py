"""Operations against the singleton ``character_state`` document."""

from __future__ import annotations

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.schemas import CharacterProfileDoc


async def get_character_profile() -> CharacterProfileDoc | dict:
    """Retrieve the full ``_id: "global"`` document (profile + runtime state).

    Returns:
        The document with ``_id`` stripped, or an empty dict if missing.
    """
    db = await get_db()
    doc = await db.character_state.find_one({"_id": "global"})
    if doc is None:
        return {}
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
    return await get_character_profile()


async def upsert_character_state(
    mood: str,
    global_vibe: str,
    reflection_summary: str,
    timestamp: str,
) -> None:
    """Update the runtime character state.

    Empty-string arguments are treated as "leave unchanged" — the existing
    value for that field is preserved. ``timestamp`` always overwrites
    ``updated_at``.
    """
    db = await get_db()
    existing = await get_character_state()

    if mood == "":
        mood = existing.get("mood", "")
    if global_vibe == "":
        global_vibe = existing.get("global_vibe", "")
    if reflection_summary == "":
        reflection_summary = existing.get("reflection_summary", "")

    await db.character_state.update_one(
        {"_id": "global"},
        {
            "$set": {
                "mood": mood,
                "global_vibe": global_vibe,
                "reflection_summary": reflection_summary,
                "updated_at": timestamp,
            }
        },
        upsert=True,
    )
