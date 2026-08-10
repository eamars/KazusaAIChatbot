"""Cross-boundary relational stance propagation tests."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v2.surface import (
    build_degraded_text_surface,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from tests.unit.cognition_core_v2.surface_fixtures import (
    build_relational_decision,
    build_surface_state,
)


def test_relational_stance_preserves_polarity_through_surface_and_dialog() -> None:
    """Surface preparation preserves the exact stance for dialog ownership."""

    decision = build_relational_decision(stance="accept")
    surface_input = l3_surface.build_text_surface_input_from_global_state(
        build_surface_state(decision),
        interaction_style_context="brief and natural",
    )
    surface_output = build_degraded_text_surface(surface_input)

    assert surface_input["relational_willingness"] == decision
    assert surface_output["relational_willingness"] == decision
