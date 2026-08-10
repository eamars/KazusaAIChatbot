"""Direct ownership tests for selected response-operation validation."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    CognitiveEpisodeValidationError,
    NO_ROLE,
    validate_selected_response_operation,
)


def _input_operation() -> dict[str, object]:
    """Build the episode operation with ungrounded action endpoints."""

    return {
        "operation": "the character chooses a reward",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": NO_ROLE,
        "embedded_target_role": NO_ROLE,
    }


def _selected_operation() -> dict[str, object]:
    """Build the post-selection operation with resolved action endpoints."""

    return {
        "operation": "the user gives the selected reward to the character",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }


def test_selected_response_operation_preserves_fixed_roles() -> None:
    """Selection may resolve only endpoints left ungrounded by the episode."""

    validated = validate_selected_response_operation(
        _selected_operation(),
        _input_operation(),
    )

    assert validated["embedded_actor_role"] == CURRENT_USER_ROLE
    assert validated["embedded_target_role"] == CURRENT_CHARACTER_ROLE
    assert validated["response_owner_role"] == CURRENT_CHARACTER_ROLE
    assert validated["selection_owner_role"] == CURRENT_CHARACTER_ROLE


def test_selected_response_operation_rejects_conflicting_roles() -> None:
    """A selected endpoint cannot reverse a known input endpoint."""

    input_operation = _input_operation()
    input_operation["embedded_actor_role"] = CURRENT_CHARACTER_ROLE

    with pytest.raises(
        CognitiveEpisodeValidationError,
        match="known input role",
    ):
        validate_selected_response_operation(
            _selected_operation(),
            input_operation,
        )
