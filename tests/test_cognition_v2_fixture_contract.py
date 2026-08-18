"""Fixture contract checks for the maintenance metadata transition."""

from __future__ import annotations

import json
from pathlib import Path

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    validate_cognition_state,
)


def test_fixture_state_preserves_relationship_axes_with_maintenance_metadata() -> None:
    """A canonical fixture keeps relationship axes and adds bookkeeping."""

    state = build_acquaintance_user_state(
        global_user_id="fixture-contract-user",
        updated_at="2026-08-18T00:00:00Z",
    )

    validated = validate_cognition_state(state)

    assert validated["relationship"]["trust"] == 0
    assert validated["relationship"]["relationship_maintenance"][
        "processed_source_ids"
    ] == []


def test_mongo_seed_user_states_include_relationship_maintenance() -> None:
    """Every strict user-state seed carries the new metadata object."""

    fixture_path = Path(__file__).resolve().parent / "fixtures" / (
        "cognition_core_v2_mongo_seed.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    user_documents = [
        row["document"]
        for row in fixture["seed_documents"]
        if row.get("collection") == "user_profiles"
    ]

    assert user_documents
    for document in user_documents:
        maintenance = document["cognition_state"]["relationship"][
            "relationship_maintenance"
        ]
        assert maintenance["schema_version"] == (
            "relationship_maintenance.v1"
        )
