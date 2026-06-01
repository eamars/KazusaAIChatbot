"""Replay-style cognition plumbing tests for promoted global growth."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as l2_module
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import empty_user_memory_context
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


@pytest.mark.asyncio
async def test_l2_receives_promoted_global_growth_for_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promoted global growth should reach L2 through promoted reflection context."""

    response = _Response(json.dumps({
        "internal_monologue": "Use the promoted global growth as background only.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
    }))
    conscious_llm = AsyncMock()
    conscious_llm.ainvoke = AsyncMock(return_value=response)
    monkeypatch.setattr(l2_module, "_conscious_llm", conscious_llm)

    await l2_module.call_cognition_consciousness(_state(
        promoted_reflection_context={
            "promoted_global_growth": [{
                "growth_axis": "clarity",
                "guidance": "保持关心可见，但不要催促同意。",
                "maturity": "promoted",
                "updated_at": "2026-05-05T10:00:00+00:00",
            }],
            "retrieval_notes": [
                "Only active promoted global character-growth traits are included.",
            ],
        },
    ))

    rendered_messages = conscious_llm.ainvoke.await_args.args[0]
    prompt_payload = json.loads(rendered_messages[1].content)
    assert prompt_payload["promoted_reflection_context"]["promoted_global_growth"] == [{
        "growth_axis": "clarity",
        "guidance": "保持关心可见，但不要催促同意。",
        "maturity": "promoted",
        "updated_at": "2026-05-05T10:00:00+00:00",
    }]
    assert "shadow_projection" not in rendered_messages[0].content
    assert "strength" not in rendered_messages[1].content


@pytest.mark.asyncio
async def test_l2_keeps_global_growth_absent_when_not_projected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unpromoted traits are absent because context projection omits them."""

    response = _Response(json.dumps({
        "internal_monologue": "No promoted growth supplied.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
    }))
    conscious_llm = AsyncMock()
    conscious_llm.ainvoke = AsyncMock(return_value=response)
    monkeypatch.setattr(l2_module, "_conscious_llm", conscious_llm)

    await l2_module.call_cognition_consciousness(_state(promoted_reflection_context={}))

    rendered_messages = conscious_llm.ainvoke.await_args.args[0]
    prompt_payload = json.loads(rendered_messages[1].content)
    assert prompt_payload["promoted_reflection_context"] == {}


def test_l2_prompt_mentions_promoted_global_growth_as_general_context() -> None:
    """The L2 system prompt should distinguish growth from user facts."""

    prompt = l2_module._COGNITION_CONSCIOUSNESS_PROMPT

    assert "promoted_global_growth" in prompt
    assert "全局人格成长" in prompt
    assert "不得把它当成当前用户事实" in prompt
    assert "不得覆盖" in prompt


class _Response:
    """Minimal async LLM response fixture."""

    def __init__(self, content: str) -> None:
        self.content = content


def _state(*, promoted_reflection_context: dict) -> dict:
    """Build a minimal L2 cognition state fixture."""

    storage_timestamp_utc = "2026-05-05T10:00:00+00:00"
    turn_clock = build_turn_clock_from_storage_utc(storage_timestamp_utc)
    episode = build_text_chat_cognitive_episode(
        episode_id="global-growth-replay",
        percept_id="global-growth-replay-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="Can you answer directly?",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="private",
        platform_message_id="msg-1",
        platform_user_id="platform-user",
        global_user_id="global-user",
        user_name="User",
        active_turn_platform_message_ids=["msg-1"],
        active_turn_conversation_row_ids=[],
        debug_modes={},
        target_addressed_user_ids=["bot"],
        target_broadcast=False,
    )
    user_memory_context = empty_user_memory_context()
    return {
        "user_profile": {
            "affinity": 500,
            "last_relationship_insight": "steady baseline",
        },
        "rag_result": {
            "answer": "",
            "memory_evidence": [],
            "world_evidence": [],
            "user_image": {
                "user_memory_context": user_memory_context,
            },
        },
        "cognitive_episode": episode,
        "local_time_context": turn_clock["local_time_context"],
        "character_profile": {
            "name": "Character",
            "personality_brief": {"mbti": "INTJ"},
            "mood": "calm",
            "global_vibe": "steady",
        },
        "decontexualized_input": "Can you answer directly?",
        "conversation_progress": {"status": "active"},
        "indirect_speech_context": "",
        "emotional_appraisal": "steady",
        "interaction_subtext": "routine",
        "promoted_reflection_context": promoted_reflection_context,
    }
