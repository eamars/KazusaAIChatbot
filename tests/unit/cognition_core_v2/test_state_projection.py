"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py."""

from __future__ import annotations

from importlib import import_module

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_relationship_context,
)


MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.state_projection"
EXPECTED_SYMBOLS = ["project_state_for_prompt"]


def test_state_projection_exposes_owned_contract() -> None:
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


def test_relationship_maintenance_metadata_is_not_projected() -> None:
    """Keep maintenance bookkeeping outside the model-facing projection."""

    state = build_acquaintance_user_state(
        global_user_id="projection-maintenance-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    relationship = state["relationship"]
    relationship["relationship_maintenance"] = {
        "schema_version": "relationship_maintenance.v1",
        "last_interaction_date_utc": "2026-08-18",
        "last_bonus_date_utc": None,
        "last_source_id": "episode:projection",
        "processed_source_ids": ["episode:projection"],
    }

    projected = project_relationship_context(relationship)

    assert "relationship_maintenance" not in projected
    assert "relationship_maintenance" not in projected["axes"]
