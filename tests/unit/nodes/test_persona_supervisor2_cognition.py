"""Deterministic ownership test for src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py."""

from __future__ import annotations

from importlib import import_module


MODULE_PATH = "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition"
EXPECTED_SYMBOLS = ["build_cognition_input_from_global_state"]


def test_persona_supervisor2_cognition_exposes_owned_contract() -> None:
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
