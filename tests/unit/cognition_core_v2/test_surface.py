"""Direct ownership tests for typed surface propagation."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v2.surface import (
    build_degraded_text_surface,
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


def test_surface_exposes_owned_contract() -> None:
    """Keep the degraded surface entrypoint attached to this owner."""

    assert callable(build_degraded_text_surface)
