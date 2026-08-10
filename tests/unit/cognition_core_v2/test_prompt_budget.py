"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/prompt_budget.py."""

from __future__ import annotations

from importlib import import_module


MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.prompt_budget"
EXPECTED_SYMBOLS = ["fit_evidence_texts_to_budget"]


def test_prompt_budget_exposes_owned_contract() -> None:
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
