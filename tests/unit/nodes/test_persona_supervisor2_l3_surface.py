"""Direct ownership tests for the L3 surface handoff."""

from __future__ import annotations

from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from tests.unit.nodes.surface_fixtures import (
    build_relational_decision,
    build_surface_state,
)


def test_l3_surface_preserves_relational_willingness() -> None:
    """L3 carries the exact selected stance into the surface input."""

    decision = build_relational_decision(stance="conditional_accept")
    payload = l3_surface.build_text_surface_input_from_global_state(
        build_surface_state(decision),
        interaction_style_context="brief and natural",
    )

    assert payload["relational_willingness"] == decision
    assert "relationship_willingness" not in payload


def test_l3_surface_preserves_selected_response_operation() -> None:
    """L3 carries the canonical response goal beside the semantic goal."""

    state = build_surface_state(build_relational_decision())
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )

    assert payload["response_plan"]["response_goal"] == (
        "acknowledge the grounded episode"
    )


def test_persona_supervisor2_l3_surface_exposes_owned_contract() -> None:
    """Keep the L3 surface builder attached to this source owner."""

    assert callable(l3_surface.build_text_surface_input_from_global_state)
