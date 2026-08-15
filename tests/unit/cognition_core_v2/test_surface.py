"""Direct ownership tests for typed surface propagation."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v2.surface import (
    build_degraded_text_surface,
    run_text_surface_planning,
    _project_surface_payload,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from tests.unit.cognition_core_v2.surface_fixtures import (
    build_relational_decision,
    build_surface_state,
)


def test_surface_output_preserves_relational_willingness_v2() -> None:
    """Degraded surface output preserves the exact selected stance."""

    decision = build_relational_decision(stance="accept")
    payload = l3_surface.build_text_surface_input_from_global_state(
        build_surface_state(decision),
        interaction_style_context="brief and natural",
    )

    output = build_degraded_text_surface(payload)

    assert output["relational_willingness"] == decision


def test_surface_score_fallback_preserves_canonical_intent_and_limits() -> None:
    """The baseline fallback keeps authoritative intent and runtime limits."""

    payload = build_surface_state(
        build_relational_decision(stance="conditional_accept"),
    )
    surface_input = l3_surface.build_text_surface_input_from_global_state(
        payload,
        interaction_style_context="brief and natural",
    )
    surface_input["runtime_capability_limits"] = [
        "the requested capability is unavailable",
    ]

    output = build_degraded_text_surface(surface_input)

    assert output["selected_surface_intent"] == (
        surface_input["intention"]["intention"]
    )
    assert output["runtime_capability_limits"] == (
        surface_input["runtime_capability_limits"]
    )
    assert output["permitted_action_results"] == (
        surface_input["permitted_action_results"]
    )


def test_surface_prompt_omits_selected_response_operation() -> None:
    """The role carrier stays outside the writable surface prompt projection."""

    decision = build_relational_decision(stance="conditional_accept")
    payload = l3_surface.build_text_surface_input_from_global_state(
        build_surface_state(decision),
        interaction_style_context="brief and natural",
    )
    selected_operation = {
        "operation": "the user gives the selected reward to the character",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }
    payload["selected_response_operation"] = dict(selected_operation)
    payload["intention"]["selected_response_operation"] = dict(
        selected_operation
    )

    projected = _project_surface_payload(payload)

    assert "selected_response_operation" not in projected
    assert "selected_response_operation" not in projected["intention"]


def test_surface_exposes_owned_contract() -> None:
    """Keep the degraded surface entrypoint attached to this owner."""

    assert callable(build_degraded_text_surface)
    assert callable(run_text_surface_planning)
