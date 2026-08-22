"""Deterministic CAS tests for user cognition persistence."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.db import users

_TIMESTAMP = "2026-08-18T00:00:00Z"


class _UserProfiles:
    """Small collection fake that performs exact-document CAS matching."""

    def __init__(self, state: dict[str, object]) -> None:
        self.state = deepcopy(state)
        self.last_filter: dict[str, object] | None = None

    async def update_one(
        self,
        selector: dict[str, object],
        update: dict[str, object],
        *,
        upsert: bool,
    ) -> SimpleNamespace:
        del upsert
        self.last_filter = selector
        if selector.get("cognition_state") != self.state:
            return SimpleNamespace(matched_count=0)
        self.state = deepcopy(update["$set"]["cognition_state"])
        return SimpleNamespace(matched_count=1)


class _Database:
    """Database fake exposing the user profile collection."""

    def __init__(self, state: dict[str, object]) -> None:
        self.user_profiles = _UserProfiles(state)


@pytest.mark.asyncio
async def test_compare_and_replace_user_cognition_state_rejects_stale_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a base state that no longer matches the stored document."""

    current = build_acquaintance_user_state(
        global_user_id="cas-stale-user",
        updated_at=_TIMESTAMP,
    )
    database = _Database(current)

    async def get_database() -> _Database:
        """Return the fake database through the async DB boundary."""

        return database

    monkeypatch.setattr(users, "get_db", get_database)
    stale = deepcopy(current)
    stale["relationship"]["trust"] = 4
    replacement = deepcopy(current)
    replacement["updated_at"] = "2026-08-18T00:01:00Z"

    committed = await users.compare_and_replace_user_cognition_state(
        "cas-stale-user",
        stale,
        replacement,
    )

    assert committed is False
    assert database.user_profiles.state == current


@pytest.mark.asyncio
async def test_compare_and_replace_user_cognition_state_rejects_same_timestamp_stale_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare complete state content even when timestamps collide."""

    current = build_acquaintance_user_state(
        global_user_id="cas-same-timestamp-user",
        updated_at=_TIMESTAMP,
    )
    database = _Database(current)

    async def get_database() -> _Database:
        """Return the fake database through the async DB boundary."""

        return database

    monkeypatch.setattr(users, "get_db", get_database)
    stale = deepcopy(current)
    stale["relationship"]["attachment"] = 9
    replacement = deepcopy(current)
    replacement["relationship"]["trust"] = 8

    committed = await users.compare_and_replace_user_cognition_state(
        "cas-same-timestamp-user",
        stale,
        replacement,
    )

    assert committed is False
    assert database.user_profiles.last_filter is not None
    assert database.user_profiles.last_filter["cognition_state"] == stale
