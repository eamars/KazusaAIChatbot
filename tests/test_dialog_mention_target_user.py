"""Tests for dialog-authored inline mention tags."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.llm_interface.contracts import BackendDescriptor, LLMResponse
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module


class _CapturingLLM:
    """Capture dialog-generator messages and return a fixed payload."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = []

    async def ainvoke(self, messages, *, config=None):
        self.messages = messages
        backend = BackendDescriptor(
            route_name="test",
            backend_kind="test",
            model_family="test",
            model="test",
            normalized_base_url="test",
            thinking_strategy="unsupported",
            confidence="high",
            generation=1,
        )
        response = LLMResponse(
            content=json.dumps(self.payload),
            backend=backend,
            raw_response=None,
            usage={},
        )
        return response


def _character_profile() -> dict:
    """Return the minimal character profile needed by dialog prompt rendering."""

    profile = {
        "name": "Kazusa",
        "personality_brief": {
            "mbti": "INTJ",
            "logic": "precise",
            "tempo": "measured",
            "defense": "guarded",
            "quirks": "dry",
            "taboos": "physical action narration",
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.4,
            "hesitation_density": 0.2,
            "counter_questioning": 0.2,
            "softener_density": 0.3,
            "formalism_avoidance": 0.6,
            "abstraction_reframing": 0.4,
            "direct_assertion": 0.6,
            "emotional_leakage": 0.3,
            "rhythmic_bounce": 0.2,
            "self_deprecation": 0.1,
        },
    }
    return profile


def _dialog_state() -> dict:
    """Build a reusable dialog-generator state fixture.

    Returns:
        Dialog-agent state with deterministic ASCII content.
    """

    state = {
        "internal_monologue": "answer directly",
        "action_directives": {
            "linguistic_directives": {
                "rhetorical_strategy": "direct",
                "linguistic_style": "brief",
                "accepted_user_preferences": [],
                "content_plan": {
                    "semantic_content": "answer",
                    "rendering": "short",
                },
                "forbidden_phrases": [],
            },
            "contextual_directives": {
                "social_distance": "friendly",
                "emotional_intensity": "low",
                "vibe_check": "calm",
                "relational_dynamic": "cooperative",
            },
        },
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "platform-user-1",
        "platform_bot_id": "bot-1",
        "global_user_id": "global-user-1",
        "user_name": "User",
        "user_profile": {"affinity": 700},
        "character_profile": _character_profile(),
        "debug_modes": {},
        "should_respond": True,
        "dialog_usage_mode": "live_visible_reply",
        "messages": [],
    }
    return state


@pytest.mark.asyncio
async def test_dialog_generator_preserves_inline_tag_without_delivery_context(
    monkeypatch,
) -> None:
    """Dialog generator should preserve authored tags without delivery data."""

    fake_llm = _CapturingLLM({
        "final_dialog": ["@User answer"],
    })
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_module.dialog_generator(_dialog_state())

    assert result == {"final_dialog": ["@User answer"]}

    human_payload = json.loads(fake_llm.messages[1].content)
    assert "delivery_context" not in human_payload
    assert "channel_type" not in human_payload
    assert "use_reply_feature" not in human_payload
    assert "single_target_user" not in human_payload
    assert "platform_user_id" not in human_payload
    assert "global_user_id" not in human_payload
    assert "scope_users" not in human_payload


@pytest.mark.asyncio
async def test_dialog_generator_does_not_require_mention_flag(
    monkeypatch,
) -> None:
    """Dialog generator output shape should only require final_dialog."""

    fake_llm = _CapturingLLM({
        "final_dialog": ["answer"],
    })
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_module.dialog_generator(_dialog_state())

    assert result == {"final_dialog": ["answer"]}


@pytest.mark.asyncio
async def test_dialog_agent_returns_no_mention_flag(
    monkeypatch,
) -> None:
    """Dialog agent should not expose retired delivery trigger fields."""

    fake_generator = _CapturingLLM({
        "final_dialog": ["@User answer"],
    })
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_generator)

    result = await dialog_module.dialog_agent(_dialog_state())

    assert result["final_dialog"] == ["@User answer"]
    retired_field = "mention" + "_target_user"
    assert retired_field not in result
