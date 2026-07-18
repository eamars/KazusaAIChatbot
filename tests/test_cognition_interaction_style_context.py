"""V2 interaction-style surface ownership tests."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    STYLE_SYSTEM_PROMPT,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as surface_module
from tests.cognition_core_v2_test_helpers import (
    canonical_cognition_output,
    canonical_episode,
)


def _overlay(*, speech: list[str] | None = None, engagement: list[str] | None = None) -> dict[str, Any]:
    """Build one sanitized runtime style overlay."""

    return {
        "speech_guidelines": list(speech or []),
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": list(engagement or []),
        "confidence": "medium" if speech or engagement else "",
    }


def _state(*, channel_type: str = "private") -> dict[str, Any]:
    """Build a committed cognition state at the V2 L3 boundary."""

    episode = canonical_episode(
        episode_id=f"interaction-style-{channel_type}",
        content="current conversation",
    )
    episode["target_scope"]["channel_type"] = channel_type
    return {
        "global_user_id": "internal-user-id",
        "channel_type": channel_type,
        "platform": "debug",
        "platform_channel_id": "private-channel-id",
        "cognitive_episode": episode,
        "cognition_core_output": canonical_cognition_output(),
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


def test_style_context_is_owned_by_the_surface_stage() -> None:
    """Visible style stays downstream of cognition route selection."""

    style_prompt = STYLE_SYSTEM_PROMPT.casefold()

    assert "style guidance" in style_prompt
    assert "rather than final dialog" in style_prompt
    assert "cognition state" not in style_prompt


@pytest.mark.asyncio
async def test_private_style_load_uses_user_scope_without_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private L3 loads only the current participant's sanitized style."""

    captured: dict[str, str] = {}

    async def _load(**kwargs: str) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "user_style": _overlay(speech=["Use compact warmth."]),
            "application_order": ["user_style"],
        }

    monkeypatch.setattr(
        surface_module,
        "build_interaction_style_context",
        _load,
    )

    rendered = await surface_module._load_interaction_style_context(_state())

    assert captured == {
        "global_user_id": "internal-user-id",
        "channel_type": "private",
        "platform": "debug",
        "platform_channel_id": "private-channel-id",
    }
    assert rendered == "Current participant style speech: Use compact warmth."
    assert "group" not in rendered.casefold()


def test_group_style_projection_is_ordered_bounded_and_allowlisted() -> None:
    """User guidance precedes group guidance without storage metadata leaks."""

    context = {
        "user_style": _overlay(speech=["Use compact warmth."]),
        "group_channel_style": _overlay(
            engagement=["Join loose topics only when there is a grounded reason."]
        ),
        "application_order": ["user_style", "group_channel_style"],
        "style_image_id": "secret-style-image-id",
        "revision": 98,
        "source_reflection_run_ids": ["secret-run-id"],
    }

    rendered = surface_module._render_interaction_style_context(context)

    assert rendered.index("Current participant") < rendered.index("Current group")
    assert len(rendered) <= 500
    assert "secret" not in rendered
    assert "98" not in rendered


@pytest.mark.asyncio
async def test_surface_handler_passes_loaded_style_to_v2_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real connector places learned guidance in TextSurfaceInputV2."""

    async def _load(**kwargs: str) -> dict[str, Any]:
        del kwargs
        return {
            "user_style": _overlay(speech=["Prefer short direct sentences."]),
            "application_order": ["user_style"],
        }

    captured: dict[str, Any] = {}

    async def _plan(payload: dict[str, Any], services: object) -> dict[str, Any]:
        del services
        captured.update(payload)
        return {
            "schema_version": "text_surface_output.v2",
            "content_plan": "Acknowledge the exchange.",
            "content_requirements": ["Address the current participant."],
            "visible_boundaries": [],
            "addressee_plan": ["current participant"],
            "style_guidance": "brief",
            "selected_surface_intent": "acknowledge the current participant",
            "permitted_action_results": [],
        }

    monkeypatch.setattr(
        surface_module,
        "build_interaction_style_context",
        _load,
    )
    monkeypatch.setattr(surface_module, "run_text_surface_planning", _plan)

    await surface_module.call_l3_text_surface_handler(_state())

    assert captured["interaction_style_context"] == (
        "Current participant style speech: Prefer short direct sentences."
    )
    assert "internal-user-id" not in captured["interaction_style_context"]
    voice = captured["character_voice_context"]
    assert "fragmentation=" not in voice
    assert "hesitation_density=" not in voice
    assert "0.4" not in voice
    assert len(voice) > 300


def test_empty_style_context_has_explicit_semantic_fallback() -> None:
    """An empty learned overlay still satisfies the exact text contract."""

    rendered = surface_module._render_interaction_style_context({
        "user_style": _overlay(),
        "application_order": ["user_style"],
    })

    assert rendered == "No learned interaction style guidance is available."
