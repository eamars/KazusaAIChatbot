"""V2 interaction-style surface ownership tests."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_shared.surface_stages import (
    CONTENT_PLAN_SYSTEM_PROMPT,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as surface_module
from tests.cognition_test_helpers import (
    canonical_cognition_output,
    canonical_episode,
    canonical_service_character_profile,
)


def _overlay(
    *,
    speech: list[str] | None = None,
    engagement: list[str] | None = None,
) -> dict[str, Any]:
    """Build one sanitized runtime style overlay."""

    return {
        "speech_guidelines": list(speech or []),
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": list(engagement or []),
        "confidence": "medium" if speech or engagement else "",
    }


def _snapshot(
    *,
    user: dict[str, Any] | None = None,
    group: dict[str, Any] | None = None,
    **metadata: object,
) -> dict[str, Any]:
    """Build the service-owned prompt-safe turn snapshot consumed by L3."""

    surface = {
        "user": {
            "overlay": user or _overlay(),
        },
    }
    application_order = ["user"]
    if group is not None:
        surface["group_channel"] = {"overlay": group}
        application_order.append("group_channel")
    return {
        "schema_version": "interaction_style_turn_snapshot.v1",
        "surface": surface,
        "application_order": application_order,
        **metadata,
    }


def _state(*, channel_type: str = "private") -> dict[str, Any]:
    """Build a committed cognition state at the V2 L3 boundary."""

    episode = canonical_episode(
        episode_id=f"interaction-style-{channel_type}",
        content="current conversation",
    )
    episode["target_scope"]["channel_type"] = channel_type
    episode["origin_metadata"]["debug_modes"][
        "no_visual_directives"
    ] = True
    return {
        "global_user_id": "internal-user-id",
        "channel_type": channel_type,
        "platform": "debug",
        "platform_channel_id": "private-channel-id",
        "cognitive_episode": episode,
        "cognition_core_output": canonical_cognition_output(),
        "action_results": [],
        "character_profile": _character_profile(),
        "interaction_style_context": _snapshot(),
    }


def _character_profile() -> dict[str, Any]:
    """Build the required wording-only character voice source."""

    profile = canonical_service_character_profile(marker="style-context")
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


def test_interaction_style_is_owned_by_unified_content_planning() -> None:
    """Learned expression guidance stays downstream of cognition."""

    content_prompt = CONTENT_PLAN_SYSTEM_PROMPT.casefold()

    assert "interaction style" in content_prompt
    assert "delivery_profile" in content_prompt
    assert "最终对话由 dialog 渲染器生成" in content_prompt
    assert "cognition state" not in content_prompt


@pytest.mark.asyncio
async def test_private_style_load_uses_preloaded_user_snapshot() -> None:
    """Private L3 consumes only the service-owned sanitized turn snapshot."""

    state = _state()
    state["interaction_style_context"] = _snapshot(
        user=_overlay(speech=["Use compact warmth."]),
    )

    rendered = await surface_module._load_interaction_style_context(state)

    assert rendered == "当前用户风格 语言: Use compact warmth."
    assert "group" not in rendered.casefold()


def test_group_style_projection_is_ordered_bounded_and_allowlisted() -> None:
    """User guidance precedes group guidance without storage metadata leaks."""

    context = _snapshot(
        user=_overlay(speech=["Use compact warmth."]),
        group=_overlay(
            engagement=["Join loose topics only when there is a grounded reason."]
        ),
        style_image_id="secret-style-image-id",
        revision=98,
        source_reflection_run_ids=["secret-run-id"],
    )

    rendered = surface_module._render_interaction_style_context(context)

    assert rendered.index("当前用户风格") < rendered.index("当前群聊风格")
    assert len(rendered) <= 500
    assert "secret" not in rendered
    assert "98" not in rendered


def test_chinese_style_projection_uses_chinese_role_labels() -> None:
    """Chinese guidance keeps the model-facing style vocabulary Chinese."""

    context = _snapshot(
        user=_overlay(speech=["使用简洁、温和的句子。"]),
        group=_overlay(
            engagement=["只在有依据时加入群聊话题。"]
        ),
    )

    rendered = surface_module._render_interaction_style_context(context)

    assert "当前用户风格 语言" in rendered
    assert "当前群聊风格 互动" in rendered
    assert "当前用户风格" in rendered
    assert "当前群聊风格" in rendered


@pytest.mark.asyncio
async def test_surface_handler_passes_loaded_style_to_v2_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real connector places learned guidance in TextSurfaceInputV2."""

    captured: dict[str, Any] = {}

    async def _plan(payload: dict[str, Any], services: object) -> dict[str, Any]:
        del services
        captured.update(payload)
        result = {
            "schema_version": "text_surface_output.v2",
            "content_plan": "Acknowledge the exchange.",
            "content_requirements": ["Address the current participant."],
            "visible_boundaries": [],
            "addressee_plan": [{
                "handle": "current_user",
                "display_name": "current participant",
                "semantic_role": "direct_recipient",
                "wording_policy": "second_person_allowed",
            }],
            "delivery_profile": {
                "lexical_register": "plain",
                "sentence_shape": "brief",
                "rhythm": "steady",
                "hesitation": "light",
                "punctuation": "restrained",
            },
            "selected_surface_intent": "acknowledge the current participant",
            "permitted_action_results": [],
        }
        if "relational_willingness" in captured:
            result["relational_willingness"] = dict(
                captured["relational_willingness"]
            )
        return result

    monkeypatch.setattr(surface_module, "run_text_surface_planning", _plan)

    state = _state()
    state["interaction_style_context"] = _snapshot(
        user=_overlay(speech=["Prefer short direct sentences."]),
    )
    await surface_module.call_l3_text_surface_handler(state)

    assert captured["interaction_style_context"] == (
        "当前用户风格 语言: Prefer short direct sentences."
    )
    assert "internal-user-id" not in captured["interaction_style_context"]
    expression = captured["character_expression_context"]
    assert expression["tempo"] == "moderate"
    texture = expression["linguistic_texture"]
    assert "fragmentation=" not in texture
    assert "hesitation_density=" not in texture
    assert "0.4" not in texture
    assert len(texture) > 300
    assert len(captured["visual_character_context"]) > 50
    assert "visual-style-context" in captured["visual_character_context"]


def test_empty_style_context_has_explicit_semantic_fallback() -> None:
    """An empty learned overlay still satisfies the exact text contract."""

    rendered = surface_module._render_interaction_style_context(_snapshot())

    assert rendered == "没有可用的已学习互动风格指引。"
