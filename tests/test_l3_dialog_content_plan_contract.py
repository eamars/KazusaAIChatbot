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
    canonical_service_character_profile,
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
        "interaction_style_context": {
            "schema_version": "interaction_style_turn_snapshot.v1",
            "surface": {
                "user": {
                    "overlay": {
                        "speech_guidelines": [],
                        "social_guidelines": [],
                        "pacing_guidelines": [],
                        "engagement_guidelines": [],
                    },
                },
            },
            "application_order": ["user"],
        },
    }


def _character_profile() -> dict[str, Any]:
    """Build the required wording-only character voice source."""

    profile = canonical_service_character_profile(marker="l3-dialog")
    profile.update({
        "name": "Kazusa",
        "personality_brief": {
            "mbti": "ISTP",
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
    })
    return profile


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
        "addressee_plan": [{
            "handle": "current_user",
            "display_name": "current user",
            "semantic_role": "direct_recipient",
            "wording_policy": "second_person_allowed",
        }],
        "delivery_profile": {
            "lexical_register": "plain",
            "sentence_shape": "brief",
            "rhythm": "steady",
            "hesitation": "minimal",
            "punctuation": "restrained",
        },
        "selected_surface_intent": "acknowledge",
        "permitted_action_results": [],
        "relational_willingness": dict(
            _cognition_output()["relational_willingness"]
        ),
    }

    assert validate_text_surface_output(output)["content_plan"] == "Say hello."


@pytest.mark.asyncio
async def test_surface_handler_returns_native_output(monkeypatch) -> None:
    """L3 retains its canonical input with both terminal surface outputs."""

    expected = {
        "schema_version": "text_surface_output.v2",
        "content_plan": "Say hello.",
        "content_requirements": ["Address the current user."],
        "visible_boundaries": [],
        "addressee_plan": [{
            "handle": "current_user",
            "display_name": "current user",
            "semantic_role": "direct_recipient",
            "wording_policy": "second_person_allowed",
        }],
        "delivery_profile": {
            "lexical_register": "plain",
            "sentence_shape": "brief",
            "rhythm": "steady",
            "hesitation": "minimal",
            "punctuation": "restrained",
        },
        "selected_surface_intent": "acknowledge",
        "permitted_action_results": [],
        "relational_willingness": dict(
            _cognition_output()["relational_willingness"]
        ),
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

    result = await surface_module.call_l3_text_surface_handler(_state())

    assert result["text_surface_output_v2"] == expected
    assert result["visual_surface_output_v2"] == expected_visual
    assert result["text_surface_input_v2"]["schema_version"] == (
        "text_surface_input.v2"
    )
    assert "action_directives" not in result
