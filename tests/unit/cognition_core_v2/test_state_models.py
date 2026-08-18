"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/state_models.py."""

from __future__ import annotations

from importlib import import_module

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    build_acquaintance_user_state,
    validate_cognition_state,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.state_models"
EXPECTED_SYMBOLS = ["RelationshipMaintenanceV1", "validate_cognition_state"]
_TIMESTAMP = "2026-08-18T00:00:00Z"


def test_state_models_exposes_owned_contract() -> None:
    """Keep the module's named owner contract discoverable."""

    module = import_module(MODULE_PATH)
    missing_symbols = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if not hasattr(module, symbol)
    ]

    assert not missing_symbols, (
        f"{MODULE_PATH} is missing owner symbols: {missing_symbols}"
    )


def test_relationship_maintenance_metadata_is_required_and_validated() -> None:
    """Require the reducer-owned relationship maintenance contract."""

    state = build_acquaintance_user_state(
        global_user_id="state-model-user",
        updated_at=_TIMESTAMP,
    )
    del state["relationship"]["relationship_maintenance"]

    with pytest.raises(CognitionStateError, match="relationship fields"):
        validate_cognition_state(state)

    state["relationship"]["relationship_maintenance"] = {
        "schema_version": "relationship_maintenance.v1",
        "last_interaction_date_utc": "2026-08-18",
        "last_bonus_date_utc": None,
        "last_source_id": "episode:state-model-episode",
        "processed_source_ids": ["episode:state-model-episode"],
    }

    validated = validate_cognition_state(state)

    assert validated["relationship"]["relationship_maintenance"][
        "schema_version"
    ] == "relationship_maintenance.v1"


def test_relationship_maintenance_rejects_unbounded_source_ledger() -> None:
    """Reject a source ledger that could grow without a deterministic cap."""

    state = build_acquaintance_user_state(
        global_user_id="state-model-overflow-user",
        updated_at=_TIMESTAMP,
    )
    state["relationship"]["relationship_maintenance"] = {
        "schema_version": "relationship_maintenance.v1",
        "last_interaction_date_utc": "2026-08-18",
        "last_bonus_date_utc": None,
        "last_source_id": "episode:state-model-overflow-episode",
        "processed_source_ids": [
            f"episode:overflow-{index}" for index in range(257)
        ],
    }

    with pytest.raises(CognitionStateError, match="source ledger"):
        validate_cognition_state(state)
