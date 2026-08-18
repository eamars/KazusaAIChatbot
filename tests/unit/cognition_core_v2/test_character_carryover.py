"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/character_carryover.py."""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module

from kazusa_ai_chatbot.cognition_core_v2 import character_carryover
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)


MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.character_carryover"
EXPECTED_SYMBOLS = ["run_character_carryover_cognition"]


def test_character_carryover_exposes_owned_contract() -> None:
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


def test_character_carryover_consumes_canonical_delta_receipts(monkeypatch) -> None:
    """Consume the reducer envelope's updated state without re-reducing deltas."""

    base_state = build_character_production_state(
        updated_at="2026-08-18T00:00:00Z",
    )
    reduced_state = deepcopy(base_state)
    reduced_state["drives"]["connection"]["pressure"] = 21

    monkeypatch.setattr(
        character_carryover,
        "_build_native_appraisal",
        lambda appraisal, *, evidence: (
            appraisal,
            {"self": {"kind": "meaning", "entity_id": "meaning:character"}},
        ),
    )
    monkeypatch.setattr(
        character_carryover,
        "apply_semantic_appraisals",
        lambda *args, **kwargs: {
            "updated_state": reduced_state,
            "accepted_delta_receipts": [{"target_path": "drives.connection.pressure"}],
            "rejected_delta_receipts": [],
        },
    )
    monkeypatch.setattr(
        character_carryover,
        "apply_state_update",
        lambda state, **kwargs: state,
    )

    result = character_carryover._reduce_apply_decision(
        base_state=base_state,
        effective_at="2026-08-18T00:00:00Z",
        decision_payload={"semantic_appraisal": {}},
        evidence=[],
    )

    assert result is not None
    assert result["replacement_state"]["drives"]["connection"][
        "pressure"
    ] == 21
