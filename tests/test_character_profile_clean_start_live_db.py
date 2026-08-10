"""Live Asuna-database proof for the manual identity-seed boundary."""

from __future__ import annotations

from copy import deepcopy
import os

import pytest


_AUTHORIZED_DATABASE_NAME = "asuna_core_v2"

pytestmark = [pytest.mark.asyncio, pytest.mark.live_db]


def _require_authorized_database() -> None:
    """Require the explicit read-only Asuna live-DB guard."""

    if os.environ.get("IDENTITY_GROWTH_DATABASE_GUARD") != "1":
        pytest.skip("manual-seed proof requires the identity-growth guard")
    if (
        os.environ.get("IDENTITY_GROWTH_TEST_DATABASE")
        != _AUTHORIZED_DATABASE_NAME
    ):
        pytest.skip("manual-seed proof is restricted to asuna_core_v2")
    if os.environ.get("MONGODB_DB_NAME") != _AUTHORIZED_DATABASE_NAME:
        pytest.skip("manual-seed proof is restricted to asuna_core_v2")


async def test_live_startup_requires_seed_without_mutating_clean_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing identity crashes before startup creates operational state."""

    _require_authorized_database()

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.db import close_db
    from kazusa_ai_chatbot.db._client import get_db

    database = await get_db()
    revision_count_before = (
        await database.character_identity_revisions.count_documents({})
    )
    state_before = await database.character_state.find_one(
        {},
        {"_id": 0},
    )
    missing_character_id = (
        f"{service.CHARACTER_GLOBAL_USER_ID}:missing-seed-proof"
    )
    monkeypatch.setattr(
        service,
        "CHARACTER_GLOBAL_USER_ID",
        missing_character_id,
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="No character identity revision",
        ):
            await service._load_startup_character_profile()

        revision_count_after = (
            await database.character_identity_revisions.count_documents({})
        )
        state_after = await database.character_state.find_one(
            {},
            {"_id": 0},
        )
        assert revision_count_after == revision_count_before
        assert state_after == state_before
    finally:
        await close_db()


async def test_live_startup_loads_the_existing_manual_seed() -> None:
    """The authorized manually seeded ledger supplies startup identity."""

    _require_authorized_database()

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.db import close_db

    try:
        identity, runtime_state = (
            await service._load_startup_character_profile()
        )
        identity_snapshot = deepcopy(identity)

        assert identity_snapshot["name"]
        assert identity_snapshot["self_image"]["self_concept"]
        assert runtime_state["cognition_state"]["state_scope"] == "character"
    finally:
        await close_db()
