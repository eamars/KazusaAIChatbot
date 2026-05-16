from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    build_text_chat_cognitive_episode,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as surface_module


class _FakeStyleLlm:
    """Capture style-agent payload and return a valid style result."""

    def __init__(self) -> None:
        """Create the fake style LLM."""

        self.payload: dict | None = None

    async def ainvoke(self, messages: list) -> SimpleNamespace:
        """Capture the human message payload.

        Args:
            messages: Prompt messages supplied by the caller.

        Returns:
            Fake response namespace with JSON content.
        """

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


class _FakeContentAnchorLlm:
    """Capture content-anchor payload and return a valid anchor result."""

    def __init__(self) -> None:
        """Create the fake content-anchor LLM."""

        self.payload: dict | None = None
        self.system_prompt = ""

    async def ainvoke(self, messages: list) -> SimpleNamespace:
        """Capture prompt messages and return a valid content-anchor result.

        Args:
            messages: Prompt messages supplied by the caller.

        Returns:
            Fake response namespace with JSON content.
        """

        self.system_prompt = messages[0].content
        self.payload = json.loads(messages[1].content)
        content = json.dumps(
            {
                "content_anchors": [
                    "[DECISION] 接住当前轻松分享",
                    "[SOCIAL] 轻轻追问图片背景",
                    "[SCOPE] 简短覆盖立场和追问",
                ],
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


def _cognitive_episode(*, channel_type: str = "private") -> CognitiveEpisode:
    """Build a valid text-chat episode for cognition state fixtures.

    Args:
        channel_type: Channel type represented by the fixture.

    Returns:
        Valid Stage 02 text-chat cognitive episode.
    """
    return build_text_chat_cognitive_episode(
        episode_id=f"interaction-style-{channel_type}-episode",
        percept_id=f"interaction-style-{channel_type}-percept",
        timestamp="2026-05-06T00:00:00+00:00",
        time_context={
            "current_local_datetime": "2026-05-09 19:30",
            "current_local_weekday": "Saturday",
        },
        user_input="hello",
        platform="qq",
        platform_channel_id=f"{channel_type}-channel",
        channel_type=channel_type,
        platform_message_id="message-1",
        platform_user_id="platform-user-1",
        global_user_id="global-user-1",
        user_name="User",
        active_turn_platform_message_ids=["message-1"],
        active_turn_conversation_row_ids=[],
        debug_modes={},
    )


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
        "cognitive_episode": _cognitive_episode(channel_type=channel_type),
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


def _content_anchor_state(*, channel_type: str = "group") -> dict:
    """Build the minimal state consumed by ``call_content_anchor_agent``."""

    return_value = {
        "character_profile": {"name": "Test Character"},
        "decontexualized_input": "用户分享了一张旧照片，像是在等回应。",
        "referents": [],
        "rag_result": {
            "answer": "",
            "user_image": {},
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {"unknown_slots": [], "loop_count": 0},
        },
        "internal_monologue": "This is a harmless share worth a light follow-up.",
        "logical_stance": "CONFIRM",
        "character_intent": "BANTER",
        "conversation_progress": None,
        "channel_type": channel_type,
        "cognitive_episode": _cognitive_episode(channel_type=channel_type),
        "interaction_style_context": {
            "user_style": {
                "speech_guidelines": [],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [
                    "主动承接用户分享意图，通过追问参与。"
                ],
                "confidence": "medium",
            },
            "group_channel_style": {
                "speech_guidelines": [],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [
                    "根据频道主题判断是否参与松散话题。"
                ],
                "confidence": "medium",
            },
            "application_order": ["user_style", "group_channel_style"],
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
        "cognitive_episode": _cognitive_episode(channel_type=channel_type),
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
async def test_content_anchor_agent_receives_interaction_style_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Content-anchor prompt payload should receive sanitized style context."""

    fake_llm = _FakeContentAnchorLlm()
    monkeypatch.setattr(l3_module, "_content_anchor_agent_llm", fake_llm)

    result = await l3_module.call_content_anchor_agent(_content_anchor_state())

    assert result["content_anchors"][0].startswith("[DECISION]")
    assert fake_llm.payload["interaction_style_context"] == (
        _content_anchor_state()["interaction_style_context"]
    )
    assert "只能作为 `[SOCIAL]`、`[PROGRESSION]`" in fake_llm.system_prompt


@pytest.mark.asyncio
async def test_cognition_subgraph_plumbs_channel_scope_into_l3_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Platform and channel fields reach ``CognitionState`` for L3 loading."""

    captured_states: list[dict] = []
    captured_content_states: list[dict] = []

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

    async def fake_style(_state: dict) -> dict:
        return_value = {
            "rhetorical_strategy": "plain",
            "linguistic_style": "plain",
            "forbidden_phrases": [],
        }
        return return_value

    async def fake_content(state: dict) -> dict:
        captured_content_states.append(dict(state))
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

    monkeypatch.setattr(
        surface_module,
        "call_interaction_style_context_loader",
        fake_loader,
    )
    monkeypatch.setattr(surface_module, "call_style_agent", fake_style)
    monkeypatch.setattr(surface_module, "call_content_anchor_agent", fake_content)
    monkeypatch.setattr(surface_module, "call_preference_adapter", fake_preference)
    monkeypatch.setattr(surface_module, "call_visual_agent", fake_visual)
    monkeypatch.setattr(surface_module, "call_surface_directive_collector", fake_collector)

    for channel_type in ("private", "group"):
        state = _global_state(channel_type=channel_type)
        state.update(
            {
                "emotional_appraisal": "calm",
                "interaction_subtext": "neutral",
                "internal_monologue": "safe",
                "character_intent": "PROVIDE",
                "logical_stance": "CONFIRM",
                "judgment_note": "ok",
                "social_distance": "neutral",
                "emotional_intensity": "calm",
                "vibe_check": "daily",
                "relational_dynamic": "stable",
            }
        )
        await surface_module.call_l3_text_surface_handler(state)

    assert [state["channel_type"] for state in captured_states] == [
        "private",
        "group",
    ]
    assert [state["platform"] for state in captured_states] == ["qq", "qq"]
    assert [state["platform_channel_id"] for state in captured_states] == [
        "private-channel",
        "group-channel",
    ]
    assert [
        state["interaction_style_context"]["application_order"]
        for state in captured_content_states
    ] == [
        ["user_style"],
        ["user_style"],
    ]

