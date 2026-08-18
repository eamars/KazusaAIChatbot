"""Deterministic database-owned migration helper tests."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.db import script_operations
from scripts.migrate_cognition_relationship_maintenance import (
    build_relationship_maintenance_state,
)

_TIMESTAMP = "2026-08-18T00:00:00Z"


class _Collection:
    """Collection fake capturing the helper's exact CAS selector."""

    def __init__(self, current: dict[str, object]) -> None:
        self.current = deepcopy(current)
        self.selector: dict[str, object] | None = None

    async def update_one(
        self,
        selector: dict[str, object],
        update: dict[str, object],
        *,
        upsert: bool,
    ) -> SimpleNamespace:
        del upsert
        self.selector = selector
        if selector["cognition_state"] != self.current["cognition_state"]:
            return SimpleNamespace(matched_count=0)
        self.current["cognition_state"] = update["$set"]["cognition_state"]
        return SimpleNamespace(matched_count=1)


class _Database:
    """Database fake for the migration helper."""

    def __init__(self, current: dict[str, object]) -> None:
        self.user_profiles = _Collection(current)


@pytest.mark.asyncio
async def test_relationship_maintenance_migration_helpers_use_expected_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the reviewed old-state digest and complete-state selector."""

    current_state = build_acquaintance_user_state(
        global_user_id="migration-helper-user",
        updated_at=_TIMESTAMP,
    )
    old_state = deepcopy(current_state)
    del old_state["relationship"]["relationship_maintenance"]
    replacement = build_relationship_maintenance_state(old_state)
    database = _Database({
        "global_user_id": "migration-helper-user",
        "cognition_state": old_state,
    })

    async def get_database() -> _Database:
        """Return the fake database through the async DB boundary."""

        return database

    monkeypatch.setattr(script_operations, "get_db", get_database)

    committed = await script_operations.compare_and_replace_user_cognition_state_for_migration(
        global_user_id="migration-helper-user",
        expected_previous_state=old_state,
        expected_previous_digest=script_operations.cognition_state_migration_digest(
            old_state
        ),
        replacement_state=replacement,
    )

    assert committed is True
    assert database.user_profiles.selector == {
        "global_user_id": "migration-helper-user",
        "cognition_state": old_state,
    }


@pytest.mark.asyncio
async def test_relationship_maintenance_migration_helpers_report_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a guarded miss when the stored row has drifted."""

    current_state = build_acquaintance_user_state(
        global_user_id="migration-drift-user",
        updated_at=_TIMESTAMP,
    )
    old_state = deepcopy(current_state)
    del old_state["relationship"]["relationship_maintenance"]
    replacement = build_relationship_maintenance_state(old_state)
    drifted = deepcopy(old_state)
    drifted["relationship"]["trust"] = 5
    database = _Database({
        "global_user_id": "migration-drift-user",
        "cognition_state": drifted,
    })

    async def get_database() -> _Database:
        """Return the fake database through the async DB boundary."""

        return database

    monkeypatch.setattr(script_operations, "get_db", get_database)

    committed = await script_operations.compare_and_replace_user_cognition_state_for_migration(
        global_user_id="migration-drift-user",
        expected_previous_state=old_state,
        expected_previous_digest=script_operations.cognition_state_migration_digest(
            old_state
        ),
        replacement_state=replacement,
    )

    assert committed is False
