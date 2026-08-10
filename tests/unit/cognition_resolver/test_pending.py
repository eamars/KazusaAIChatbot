"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_resolver/pending.py."""

from __future__ import annotations

from importlib import import_module


MODULE_PATH = "kazusa_ai_chatbot.cognition_resolver.pending"
EXPECTED_SYMBOLS = ["build_pending_resume_record"]


def test_pending_exposes_owned_contract() -> None:
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
