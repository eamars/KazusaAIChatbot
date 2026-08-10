"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py."""

from __future__ import annotations

from importlib import import_module

import pytest

from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    validate_selection_goal_draft,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    NO_ROLE,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.goal_cognition"
EXPECTED_SYMBOLS = ["run_goal_cognition"]


def _input_operation() -> dict[str, object]:
    """Build the required-selection operation before character judgment."""

    return {
        "operation": "the character chooses a reward",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": NO_ROLE,
        "embedded_target_role": NO_ROLE,
    }


def _selection_draft(
    selected_operation: dict[str, object],
) -> dict[str, object]:
    """Build one complete selection draft for the validator."""

    return {
        "selection": "choose a reward",
        "selected_response_operation": selected_operation,
        "reason": "the current episode requires a concrete choice",
        "private_monologue": "choose from the current request",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the selected reward is stated"],
        "confidence": "high",
    }


def test_goal_cognition_exposes_owned_contract() -> None:
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


def test_required_selection_emits_selected_response_operation() -> None:
    """Selection validation returns the operation chosen by cognition."""

    selected_operation = {
        **_input_operation(),
        "operation": "the user gives the selected reward to the character",
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }
    validated = validate_selection_goal_draft(
        _selection_draft(selected_operation),
        evidence_handles={"e1"},
        role_handles=set(),
        required_evidence_handles={"e1"},
        required_operations=[{
            "evidence_handle": "e1",
            "response_operation": _input_operation(),
        }],
        maximum_evidence_handles=4,
    )

    assert validated["selected_response_operation"] == selected_operation


def test_required_selection_rejects_fixed_role_conflict() -> None:
    """Selection validation rejects reversal of a known input endpoint."""

    selected_operation = {
        **_input_operation(),
        "operation": "the character gives the selected reward to the user",
        "embedded_actor_role": CURRENT_CHARACTER_ROLE,
        "embedded_target_role": CURRENT_USER_ROLE,
    }

    with pytest.raises(ValueError, match="known input role"):
        validate_selection_goal_draft(
            _selection_draft(selected_operation),
            evidence_handles={"e1"},
            role_handles=set(),
            required_evidence_handles={"e1"},
            required_operations=[{
                "evidence_handle": "e1",
                "response_operation": {
                    **_input_operation(),
                    "embedded_actor_role": CURRENT_USER_ROLE,
                },
            }],
            maximum_evidence_handles=4,
        )
