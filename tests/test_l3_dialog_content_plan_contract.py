"""Focused tests for the canonical V2 L3-to-dialog handoff."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    validate_text_surface_output,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as surface_module
from tests.cognition_core_v2_test_helpers import (
    canonical_cognition_output,
    canonical_episode,
)


def _cognition_output() -> dict[str, Any]:
    """Build the semantic output shape required by surface projection."""

    return canonical_cognition_output()


def _state() -> dict[str, Any]:
    """Build a surface-planner state without legacy dialog directives."""

    return {
        "cognition_core_output": _cognition_output(),
        "cognitive_episode": canonical_episode(
            episode_id="l3-dialog-content-plan",
            content="conversation",
        ),
        "user_input": "hello",
        "action_results": [],
        "character_profile": _character_profile(),
    }


def _character_profile() -> dict[str, Any]:
    """Build the required wording-only character voice source."""

    return {
        "name": "Kazusa",
        "personality_brief": {
            "logic": "analytical",
            "tempo": "moderate",
            "defense": "reserved",
            "quirks": "occasional hesitation",
            "taboos": "stay in character",
        },
        "linguistic_texture_profile": {
            "hesitation_density": 0.4,
            "fragmentation": 0.4,
            "emotional_leakage": 0.4,
            "rhythmic_bounce": 0.4,
            "direct_assertion": 0.4,
            "softener_density": 0.4,
            "counter_questioning": 0.4,
            "formalism_avoidance": 0.4,
            "abstraction_reframing": 0.4,
            "self_deprecation": 0.4,
        },
    }


def test_surface_input_uses_native_v2_contract() -> None:
    """Surface input contains semantic projections rather than directive bags."""

    payload = surface_module.build_text_surface_input_from_global_state(
        _state(),
        interaction_style_context="brief and natural",
    )

    assert payload["schema_version"] == "text_surface_input.v2"
    assert payload["intention"]["route"] == "speech"
    assert "action_directives" not in payload


def test_surface_output_validation_requires_exact_v2_fields() -> None:
    """The dialog boundary validates the exact TextSurfaceOutputV2 shape."""

    output = {
        "schema_version": "text_surface_output.v2",
        "content_plan": "Say hello.",
        "content_requirements": ["Address the current user."],
        "visible_boundaries": [],
        "addressee_plan": ["current user"],
        "style_guidance": "brief",
        "selected_surface_intent": "acknowledge",
        "permitted_action_results": [],
    }

    assert validate_text_surface_output(output)["content_plan"] == "Say hello."


@pytest.mark.asyncio
async def test_surface_handler_returns_native_output(monkeypatch) -> None:
    """L3 returns the V2 surface directly for dialog consumption."""

    expected = {
        "schema_version": "text_surface_output.v2",
        "content_plan": "Say hello.",
        "content_requirements": ["Address the current user."],
        "visible_boundaries": [],
        "addressee_plan": ["current user"],
        "style_guidance": "brief",
        "selected_surface_intent": "acknowledge",
        "permitted_action_results": [],
    }
    expected_visual = {
        "schema_version": "visual_surface_output.v2",
        "visual_directives": "private image composition",
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

    async def _fake_visual_planner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return expected_visual

    monkeypatch.setattr(
        surface_module,
        "run_visual_surface_planning",
        _fake_visual_planner,
    )

    async def _style_context(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "user_style": {
                "speech_guidelines": [],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [],
                "confidence": "",
            },
            "application_order": ["user_style"],
        }

    monkeypatch.setattr(
        surface_module,
        "build_interaction_style_context",
        _style_context,
    )

    result = await surface_module.call_l3_text_surface_handler(_state())

    assert result == {
        "text_surface_output_v2": expected,
        "visual_surface_output_v2": expected_visual,
    }
    assert "action_directives" not in result
