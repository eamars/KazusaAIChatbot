"""Deterministic integration-loop coverage for idempotent maintenance."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import facade_helpers as facade
from kazusa_ai_chatbot.cognition_shared.output_projection import (
    build_state_update,
    project_affect,
    project_relationship,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    apply_relationship_maintenance,
    apply_state_update,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.db import users


class _UserProfiles:
    """In-memory profile collection with exact-document CAS behavior."""

    def __init__(self) -> None:
        self.documents: dict[str, dict[str, object]] = {}

    async def insert_one(self, document: dict[str, object]) -> None:
        """Persist one complete profile document for the loop test."""

        user_id = document["global_user_id"]
        self.documents[user_id] = deepcopy(document)

    async def find_one(
        self,
        selector: dict[str, object],
        projection: dict[str, int] | None = None,
    ) -> dict[str, object] | None:
        """Read one profile using the same selector shape as the DB owner."""

        user_id = selector["global_user_id"]
        document = self.documents.get(user_id)
        if document is None:
            return None
        if projection == {"cognition_state": 1}:
            return {
                "cognition_state": deepcopy(document["cognition_state"]),
            }
        return deepcopy(document)

    async def update_one(
        self,
        selector: dict[str, object],
        update: dict[str, object],
        *,
        upsert: bool,
    ) -> SimpleNamespace:
        """Apply the complete-state selector and return a match count."""

        del upsert
        user_id = selector["global_user_id"]
        document = self.documents.get(user_id)
        if (
            document is None
            or document["cognition_state"] != selector["cognition_state"]
        ):
            return SimpleNamespace(matched_count=0)
        document["cognition_state"] = deepcopy(
            update["$set"]["cognition_state"]
        )
        return SimpleNamespace(matched_count=1)


class _Database:
    """Expose the profile collection at the project DB boundary."""

    def __init__(self) -> None:
        self.user_profiles = _UserProfiles()


def test_relationship_maintenance_loop_replay_is_idempotent() -> None:
    """A replayed episode cannot create a second familiarity increment."""

    state = build_acquaintance_user_state(
        global_user_id="maintenance-loop-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    first = apply_relationship_maintenance(
        state,
        source_episode_id="loop-episode",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    replay = apply_relationship_maintenance(
        first,
        source_episode_id="loop-episode",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )

    assert replay == first


@pytest.mark.asyncio
async def test_relationship_maintenance_round_trips_create_consume_update_persist_reread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run maintenance through the facade, envelope, CAS, and fresh read."""

    database = _Database()
    monkeypatch.setattr(users, "get_db", lambda: _database(database))
    created = build_acquaintance_user_state(
        global_user_id="maintenance-round-trip-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    created["relationship"].update({
        "attachment": 60,
        "trust": 60,
        "desired_closeness": 80,
        "perceived_closeness": 0,
        "salience": 39,
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": "episode:round-trip-episode",
            "occurred_at": "2026-08-18T00:00:00Z",
            "semantic_summary": "The episode carries relationship evidence.",
        }],
    })
    await users.create_user_profile({
        "global_user_id": "maintenance-round-trip-user",
        "cognition_state": created,
    })
    previous = await users.get_user_cognition_state(
        "maintenance-round-trip-user"
    )
    receipts = [{
        "target_path": "relationship.trust",
        "relationship_axis": "trust",
        "requested_delta": 2,
        "applied_delta": 2,
        "previous_value": 60,
        "next_value": 62,
        "evidence_refs": [],
        "duplicate_disposition": "unique",
    }]
    maintained = facade._apply_final_relationship_maintenance(
        previous,
        episode={
            "episode_id": "round-trip-episode",
            "created_at": "2026-08-18T00:00:00Z",
        },
        elapsed_seconds=0,
        accepted_relationship_deltas=receipts,
        direct_facts=[],
    )
    derived = apply_state_update(
        maintained,
        elapsed_seconds=0,
        updated_at="2026-08-18T00:00:00Z",
    )
    final = create_deterministic_goals(
        derived,
        updated_at="2026-08-18T00:00:00Z",
        reconcile_salience_gated_goals=True,
    )
    final = validate_cognition_state(final)
    state_update = build_state_update(previous, final)
    committed = await users.compare_and_replace_user_cognition_state(
        "maintenance-round-trip-user",
        state_update["expected_previous_state"],
        state_update["replacement_state"],
    )
    reread = await users.get_user_cognition_state(
        "maintenance-round-trip-user"
    )
    relationship_projection = project_relationship(reread["relationship"])
    affect_projection = project_affect(
        reread["affect_activations"],
        reread,
    )

    assert committed is True
    assert state_update["expected_previous_state"] == previous
    assert reread["relationship"]["familiarity"] == 12
    assert reread["relationship"]["salience"] == 41
    assert reread["relationship"]["relationship_maintenance"][
        "last_source_id"
    ] == "episode:round-trip-episode"
    assert any(
        goal["goal_kind"] == "relationship_connection"
        for goal in reread["goals"]
    )
    assert any(
        activation["emotion_id"] == "love_attachment"
        for activation in reread["affect_activations"]
    )
    assert any(
        row["emotion"] == "love_attachment"
        for row in affect_projection
    )
    assert "relationship_maintenance" not in relationship_projection


async def _database(database: _Database) -> _Database:
    """Return the fake DB through the async project boundary."""

    return database
