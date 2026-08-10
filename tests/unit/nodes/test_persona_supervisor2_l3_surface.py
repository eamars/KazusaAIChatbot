"""Direct ownership tests for the L3 surface handoff."""

from __future__ import annotations

from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    NO_ROLE,
)
from tests.unit.cognition_core_v2.surface_fixtures import (
    build_relational_decision,
    build_surface_state,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


def test_l3_surface_preserves_relational_willingness_v2() -> None:
    """L3 carries the exact selected stance into the surface input."""

    decision = build_relational_decision(stance="conditional_accept")
    payload = l3_surface.build_text_surface_input_from_global_state(
        build_surface_state(decision),
        interaction_style_context="brief and natural",
    )

    assert payload["relational_willingness"] == decision
    assert "relationship_willingness" not in payload


def test_l3_surface_preserves_selected_response_operation() -> None:
    """L3 carries the post-selection role carrier beside the intention."""

    input_operation = {
        "operation": "the character chooses a reward",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": NO_ROLE,
        "embedded_target_role": NO_ROLE,
    }
    selected_operation = {
        **input_operation,
        "operation": "the user gives the selected reward to the character",
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }
    state = build_surface_state(build_relational_decision())
    state["cognitive_episode"] = canonical_episode(
        episode_id="selected-operation-surface-episode",
        content="current reward request",
        metadata={"response_operation": input_operation},
    )
    output = state["cognition_core_output"]
    output["intention"]["selected_response_operation"] = selected_operation
    output["admitted_bid"]["selected_response_operation"] = selected_operation

    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["selected_response_operation"] == selected_operation
    assert payload["intention"]["selected_response_operation"] == (
        selected_operation
    )


def test_persona_supervisor2_l3_surface_exposes_owned_contract() -> None:
    """Keep the L3 surface builder attached to this source owner."""

    assert callable(l3_surface.build_text_surface_input_from_global_state)
