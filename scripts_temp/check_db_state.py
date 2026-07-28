"""Check current character_state and identity revisions in DB."""
import asyncio
import json
import os

os.environ["MONGODB_DB_NAME"] = "asuna_core_v2"

from kazusa_ai_chatbot.db._client import get_db, close_db


async def check():
    db = await get_db()

    revisions = await db["character_identity_revisions"].find({}).to_list(100)
    print(f"=== identity revisions: {len(revisions)} ===")
    for rev in revisions:
        rev.pop("_id", None)
        eid = rev.get("effective_identity", {})
        print(f"  rev={rev.get('revision_number')} "
              f"character_id={rev.get('character_id')} "
              f"name={eid.get('name')}")

    state = await db["character_state"].find_one({})
    if state:
        state.pop("_id", None)
        print("=== character_state ===")
        print(json.dumps(state, indent=2, ensure_ascii=False, default=str))
    else:
        print("character_state: EMPTY")

    rev = await db["character_identity_revisions"].find_one({})
    if rev:
        rev.pop("_id", None)
        print("\n=== character_identity_revisions (first doc) ===")
        print(json.dumps(rev, indent=2, ensure_ascii=False, default=str))
    else:
        print("character_identity_revisions: EMPTY")

    await close_db()


asyncio.run(check())
