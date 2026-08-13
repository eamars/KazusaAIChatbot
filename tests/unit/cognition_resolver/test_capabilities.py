"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_resolver/capabilities.py."""

from __future__ import annotations

from importlib import import_module

from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    project_resolver_observation_for_cognition,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_resolver.capabilities"
EXPECTED_SYMBOLS = ["project_resolver_observation_for_cognition"]


def test_capabilities_exposes_owned_contract() -> None:
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


def test_resolver_observation_evidence_has_typed_authority() -> None:
    """Resolver context is explicitly non-current-event evidence."""

    evidence, direct_facts = project_resolver_observation_for_cognition(
        {
            'observation_id': 'resolver-observation-1',
            'semantic_summary': 'bounded resolver context',
            'capability': 'lookup',
        },
        occurred_at='2026-07-30T00:00:00Z',
    )

    assert evidence['authority'] == 'contextual_fact_only'
    assert direct_facts == []
