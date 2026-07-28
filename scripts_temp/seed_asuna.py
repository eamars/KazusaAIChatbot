"""Seed asuna_core_v2 with the Asuna V2 profile as revision 0.

Uses CHARACTER_GLOBAL_USER_ID so the seeded revision is discoverable
by the running service.
"""
import asyncio
import os
from pathlib import Path

os.environ["MONGODB_DB_NAME"] = "asuna_core_v2"

from kazusa_ai_chatbot.character_profile import load_character_profile_seed
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db._client import get_db, close_db
from kazusa_ai_chatbot.db.character_identity_growth import (
    GROWTH_COLLECTION_NAMES,
    ensure_character_identity_growth_indexes,
    ensure_seed_identity,
    get_current_identity,
)
from kazusa_ai_chatbot.db.character import ensure_operational_character_state

_PROFILE_PATH = Path(__file__).resolve().parents[1] / "personalities" / "asuna.json"


async def seed():
    db = await get_db()
    character_id = CHARACTER_GLOBAL_USER_ID
    print(f"  character_id: {character_id}")
    print(f"  profile: {_PROFILE_PATH}")

    owned = list(GROWTH_COLLECTION_NAMES) + ["character_state"]
    for name in owned:
        await db.drop_collection(name)
        print(f"  dropped {name}")

    await ensure_character_identity_growth_indexes()
    print("  indexes created")

    await ensure_operational_character_state()
    print("  operational state ensured")

    seed_data = load_character_profile_seed(_PROFILE_PATH)
    rev = await ensure_seed_identity(
        character_id=character_id,
        seed=seed_data,
    )
    eid = rev["effective_identity"]
    print(f"  seeded revision 0: name={eid['name']}")

    verify = await get_current_identity(character_id=character_id)
    assert verify["effective_identity"]["name"] == eid["name"]
    print(f"  verified: get_current_identity returns {verify['effective_identity']['name']}")

    names = await db.list_collection_names()
    print(f"  total collections: {len(names)}")

    await close_db()


asyncio.run(seed())
