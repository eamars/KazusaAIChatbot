from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module


class _FakeStyleLlm:
    """Capture style-agent payload and return a valid style result."""

    def __init__(self) -> None:
        """Create the fake style LLM."""

        self.payload: dict | None = None

    async def ainvoke(self, messages: list) -> SimpleNamespace:
        """Capture the human message payload."""

        self.payload = json.loads(messages[1].content)
        content = json.dumps(
            {
                "rhetorical_strategy": "保持轻快的确认。",
                "linguistic_style": "短句，轻微调侃。",
                "forbidden_phrases": [],
            },
            ensure_ascii=False,
        )
        return_value = SimpleNamespace(content=content)
        return return_value


def _character_profile() -> dict:
    """Build the minimal character profile used by L3 style tests."""

    return_value = {
        "name": "Test Character",
        "mood": "neutral",
        "global_vibe": "calm",
        "personality_brief": {
            "logic": "direct",
            "tempo": "quick",
            "defense": "teasing",
            "quirks": "none",
            "taboos": [],
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.2,
            "hesitation_density": 0.2,
            "counter_questioning": 0.2,
            "softener_density": 0.2,
            "formalism_avoidance": 0.8,
            "abstraction_reframing": 0.2,
            "direct_assertion": 0.7,
            "emotional_leakage": 0.4,
            "rhythmic_bounce": 0.6,
            "self_deprecation": 0.1,
        },
    }
    return return_value


def _style_state(*, channel_type: str = "private") -> dict:
    """Build the minimal state consumed by ``call_style_agent``."""

    return_value = {
        "character_profile": _character_profile(),
        "user_profile": {"last_relationship_insight": "neutral"},
        "internal_monologue": "The request is harmless.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "chat_history_recent": [],
        "channel_type": channel_type,
        "interaction_style_context": {
            "user_style": {
                "speech_guidelines": ["Use compact warmth."],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [],
                "confidence": "medium",
            },
            "application_order": ["user_style"],
        },
    }
    return return_value


def _global_state(*, channel_type: str) -> dict:
    """Build the minimal global persona state for cognition subgraph tests."""

    return_value = {
        "character_profile": _character_profile(),
        "timestamp": "2026-05-06T00:00:00+00:00",
        "time_context": {},
        "user_input": "hello",
        "prompt_message_context": {},
        "platform": "qq",
        "platform_channel_id": f"{channel_type}-channel",
        "channel_type": channel_type,
        "platform_message_id": "message-1",
        "platform_user_id": "platform-user-1",
        "global_user_id": "global-user-1",
        "user_name": "User",
        "user_profile": {
            "affinity": 500,
            "last_relationship_insight": "neutral",
        },
        "platform_bot_id": "bot-1",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "promoted_reflection_context": {},
        "referents": [],
        "debug_modes": {},
        "decontexualized_input": "hello",
        "rag_result": {
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [],
                    "milestones": [],
                    "active_commitments": [],
                }
            }
        },
    }
    return return_value


@pytest.mark.asyncio
async def test_interaction_style_context_loader_falls_back_without_group_for_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader failure returns private-safe empty context without group style."""

    monkeypatch.setattr(
        l3_module,
        "build_interaction_style_context",
        AsyncMock(side_effect=RuntimeError("db unavailable")),
    )

    result = await l3_module.call_interaction_style_context_loader(
        {
            "global_user_id": "user-1",
            "channel_type": "private",
            "platform": "qq",
            "platform_channel_id": "private-1",
        }
    )

    context = result["interaction_style_context"]
    assert context["application_order"] == ["user_style"]
    assert "group_channel_style" not in context


@pytest.mark.asyncio
async def test_style_agent_receives_private_interaction_style_without_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Style agent prompt payload receives sanitized private style context."""

    fake_llm = _FakeStyleLlm()
    monkeypatch.setattr(l3_module, "_style_agent_llm", fake_llm)

    result = await l3_module.call_style_agent(_style_state())

    assert result["linguistic_style"] == "短句，轻微调侃。"
    assert fake_llm.payload["interaction_style_context"] == {
        "user_style": {
            "speech_guidelines": ["Use compact warmth."],
            "social_guidelines": [],
            "pacing_guidelines": [],
            "engagement_guidelines": [],
            "confidence": "medium",
        },
        "application_order": ["user_style"],
    }


@pytest.mark.asyncio
async def test_cognition_subgraph_plumbs_channel_scope_into_l3_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Platform and channel fields reach ``CognitionState`` for L3 loading."""

    captured_states: list[dict] = []

    async def fake_l1(_state: dict) -> dict:
        return_value = {
            "interaction_subtext": "neutral",
            "emotional_appraisal": "calm",
        }
        return return_value

    async def fake_l2a(_state: dict) -> dict:
        return_value = {
            "internal_monologue": "safe",
            "character_intent": "PROVIDE",
        }
        return return_value

    async def fake_l2b(_state: dict) -> dict:
        return_value = {"boundary_core_assessment": {}}
        return return_value

    async def fake_l2c(_state: dict) -> dict:
        return_value = {
            "logical_stance": "CONFIRM",
            "judgment_note": "ok",
        }
        return return_value

    async def fake_loader(state: dict) -> dict:
        captured_states.append(dict(state))
        return_value = {
            "interaction_style_context": {
                "user_style": {
                    "speech_guidelines": [],
                    "social_guidelines": [],
                    "pacing_guidelines": [],
                    "engagement_guidelines": [],
                    "confidence": "",
                },
                "application_order": ["user_style"],
            }
        }
        return return_value

    async def fake_contextual(_state: dict) -> dict:
        return_value = {
            "social_distance": "neutral",
            "emotional_intensity": "calm",
            "vibe_check": "daily",
            "relational_dynamic": "stable",
            "expression_willingness": "open",
        }
        return return_value

    async def fake_style(_state: dict) -> dict:
        return_value = {
            "rhetorical_strategy": "plain",
            "linguistic_style": "plain",
            "forbidden_phrases": [],
        }
        return return_value

    async def fake_content(_state: dict) -> dict:
        return_value = {"content_anchors": []}
        return return_value

    async def fake_preference(_state: dict) -> dict:
        return_value = {"accepted_user_preferences": []}
        return return_value

    async def fake_visual(_state: dict) -> dict:
        return_value = {
            "facial_expression": [],
            "body_language": [],
            "gaze_direction": [],
            "visual_vibe": [],
        }
        return return_value

    async def fake_collector(_state: dict) -> dict:
        return_value = {"action_directives": {"ok": True}}
        return return_value

    monkeypatch.setattr(cognition_module, "call_cognition_subconscious", fake_l1)
    monkeypatch.setattr(cognition_module, "call_cognition_consciousness", fake_l2a)
    monkeypatch.setattr(cognition_module, "call_boundary_core_agent", fake_l2b)
    monkeypatch.setattr(cognition_module, "call_judgment_core_agent", fake_l2c)
    monkeypatch.setattr(
        cognition_module,
        "call_interaction_style_context_loader",
        fake_loader,
    )
    monkeypatch.setattr(cognition_module, "call_contextual_agent", fake_contextual)
    monkeypatch.setattr(cognition_module, "call_style_agent", fake_style)
    monkeypatch.setattr(cognition_module, "call_content_anchor_agent", fake_content)
    monkeypatch.setattr(cognition_module, "call_preference_adapter", fake_preference)
    monkeypatch.setattr(cognition_module, "call_visual_agent", fake_visual)
    monkeypatch.setattr(cognition_module, "call_collector", fake_collector)

    for channel_type in ("private", "group"):
        await cognition_module.call_cognition_subgraph(
            _global_state(channel_type=channel_type)
        )

    assert [state["channel_type"] for state in captured_states] == [
        "private",
        "group",
    ]
    assert [state["platform"] for state in captured_states] == ["qq", "qq"]
    assert [state["platform_channel_id"] for state in captured_states] == [
        "private-channel",
        "group-channel",
    ]
