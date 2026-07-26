"""Live disposable-database proof for packaged character startup."""

from __future__ import annotations

import os

import pytest
from pymongo import MongoClient


_STAGE3_DATABASE_NAME = "_test_kazusa_core_v2"

pytestmark = [pytest.mark.asyncio, pytest.mark.live_db]


async def test_brain_clean_start_and_restart_preserve_packaged_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean database should start twice without external profile config."""

    if os.environ.get("MONGODB_DB_NAME") != _STAGE3_DATABASE_NAME:
        pytest.skip("clean-start proof requires the Stage 3 database")
    if os.environ.get("KAZUSA_TEST_DB_GUARD") != "1":
        pytest.skip("clean-start proof requires KAZUSA_TEST_DB_GUARD=1")
    if os.environ.get("STAGE3_DATABASE_GUARD") != "1":
        pytest.skip("clean-start proof requires STAGE3_DATABASE_GUARD=1")
    if os.environ.get("CHARACTER_PROFILE_PATH"):
        raise AssertionError("clean-start proof forbids profile path config")

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.character_profile import (
        load_packaged_character_profile_seed,
    )
    from kazusa_ai_chatbot.db import (
        close_db,
        get_character_cognition_state,
        get_character_profile,
    )

    for setting_name in (
        "BACKGROUND_WORK_WORKER_ENABLED",
        "CALENDAR_SCHEDULER_ENABLED",
        "REFLECTION_CYCLE_ENABLED",
        "SELF_COGNITION_ENABLED",
    ):
        monkeypatch.setattr(service, setting_name, False)

    client = MongoClient(
        os.environ["MONGODB_URI"],
        serverSelectionTimeoutMS=5_000,
    )
    client.admin.command("ping")
    client.drop_database(_STAGE3_DATABASE_NAME)
    packaged_seed = load_packaged_character_profile_seed()

    try:
        async with service.lifespan(service.app):
            first_profile = await get_character_profile()
            first_state = await get_character_cognition_state()
            health_response = await service.health()
            for field_name, expected_value in packaged_seed.items():
                assert first_profile[field_name] == expected_value
            assert first_state["state_scope"] == "character"
            assert first_state["schema_version"] == "cognition_state.v2"
            assert health_response.status == "ok"
            assert health_response.db is True

        async with service.lifespan(service.app):
            restart_profile = await get_character_profile()
            restart_state = await get_character_cognition_state()
            assert restart_profile == first_profile
            assert restart_state == first_state
    finally:
        await close_db()
        client.drop_database(_STAGE3_DATABASE_NAME)
        client.close()
