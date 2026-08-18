"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/output_projection.py."""

from __future__ import annotations

from importlib import import_module

from kazusa_ai_chatbot.cognition_core_v2.output_projection import (
    build_state_update,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.output_projection"
EXPECTED_SYMBOLS = ["build_state_update", "project_affect"]
_TIMESTAMP = "2026-08-18T00:00:00Z"


def test_output_projection_exposes_owned_contract() -> None:
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


def test_state_update_carries_expected_previous_state() -> None:
    """Carry the complete CAS base state beside the replacement state."""

    previous = build_acquaintance_user_state(
        global_user_id="output-projection-user",
        updated_at=_TIMESTAMP,
    )
    replacement = build_acquaintance_user_state(
        global_user_id="output-projection-user",
        updated_at="2026-08-18T00:01:00Z",
    )

    update = build_state_update(previous, replacement)

    assert update["expected_previous_state"] == previous
