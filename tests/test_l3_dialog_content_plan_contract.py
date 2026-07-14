"""Focused tests for the canonical V2 L3-to-dialog handoff."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    validate_text_surface_output,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as surface_module


def _cognition_output() -> dict[str, Any]:
    """Build the semantic output shape required by surface projection."""

    return {
        "intention": {"route": "speech", "intention": "acknowledge", "target_roles": [], "reason": "grounded"},
        "supporting_bids": [],
        "affect_projection": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "warm",
            "intensity": "restrained",
            "directness": "balanced",
        },
    }


def _state() -> dict[str, Any]:
    """Build a surface-planner state without legacy dialog directives."""

    return {
        "cognition_core_output": _cognition_output(),
        "cognitive_episode": {
            "semantic_scene": "conversation",
        },
        "user_input": "hello",
        "action_results": [],
    }


def test_surface_input_uses_native_v2_contract() -> None:
    """Surface input contains semantic projections rather than directive bags."""

    payload = surface_module.build_text_surface_input_from_global_state(_state())

    assert payload["schema_version"] == "text_surface_input.v2"
    assert payload["intention"]["route"] == "speech"
    assert "action_directives" not in payload


def test_surface_output_validation_requires_exact_v2_fields() -> None:
    """The dialog boundary validates the exact TextSurfaceOutputV2 shape."""

    output = {
        "schema_version": "text_surface_output.v2",
        "content_plan": "Say hello.",
        "visible_boundaries": [],
        "addressee_plan": ["current user"],
        "style_guidance": "brief",
        "pacing_guidance": "one sentence",
        "selected_surface_intent": "acknowledge",
    }

    assert validate_text_surface_output(output)["content_plan"] == "Say hello."


@pytest.mark.asyncio
async def test_surface_handler_returns_native_output(monkeypatch) -> None:
    """L3 returns the V2 surface directly for dialog consumption."""

    expected = {
        "schema_version": "text_surface_output.v2",
        "content_plan": "Say hello.",
        "visible_boundaries": [],
        "addressee_plan": ["current user"],
        "style_guidance": "brief",
        "pacing_guidance": "one sentence",
        "selected_surface_intent": "acknowledge",
    }

    async def _fake_planner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return expected

    monkeypatch.setattr(
        surface_module,
        "run_text_surface_planning",
        _fake_planner,
    )

    result = await surface_module.call_l3_text_surface_handler(_state())

    assert result == {"text_surface_output_v2": expected}
    assert "action_directives" not in result
