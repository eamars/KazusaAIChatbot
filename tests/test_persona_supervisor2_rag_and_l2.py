from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as cognition_l2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag as rag_module


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


class _CapturingAsyncLLM:
    def __init__(self, response_payload: dict):
        self.messages = None
        self._response_payload = response_payload

    async def ainvoke(self, messages):
        self.messages = messages
        return _DummyResponse(json.dumps(self._response_payload, ensure_ascii=False))


@pytest.mark.asyncio
async def test_call_cognition_consciousness_uses_character_diary_not_legacy_facts(monkeypatch):
    llm = _CapturingAsyncLLM(
        {
            "internal_monologue": "记住了。",
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
        }
    )
    monkeypatch.setattr(cognition_l2_module, "_conscious_llm", llm)

    state = {
        "character_profile": {
            "name": "Kazusa",
            "personality_brief": {"mbti": "INTJ"},
            "mood": "calm",
            "global_vibe": "quiet",
            "reflection_summary": "她还记得上次的气氛。",
        },
        "user_profile": {
            "affinity": 650,
            "character_diary": [
                {"entry": "这是主观日记一。"},
                {"entry": "这是主观日记二。"},
            ],
            "facts": ["这是旧 facts，不应再被读取。"],
            "active_commitments": [],
            "last_relationship_insight": "关系正在慢慢变近。",
        },
        "research_facts": {},
        "decontexualized_input": "你还记得我吗？",
        "indirect_speech_context": "",
        "emotional_appraisal": "有点在意。",
        "interaction_subtext": "对方在确认关系连续性。",
    }

    await cognition_l2_module.call_cognition_consciousness(state)

    human_payload = json.loads(llm.messages[1].content)
    assert human_payload["diary_entry"] == ["这是主观日记一。", "这是主观日记二。"]
    assert "这是旧 facts，不应再被读取。" not in human_payload["diary_entry"]


def test_result_confidence_honors_explicit_empty_flag():
    assert rag_module._result_confidence("This branch returned real-looking text.", is_empty_result=True) == 0.0
    assert rag_module._result_confidence("This branch returned real-looking text.", is_empty_result=False) > 0.0


@pytest.mark.asyncio
async def test_call_web_search_agent_preserves_explicit_empty_flag(monkeypatch):
    async def _fake_web_search_agent(*, task: str, context: dict, expected_response: str):
        return {
            "response": "This text should still be treated as empty by explicit flag.",
            "is_empty_result": True,
        }

    monkeypatch.setattr(rag_module, "web_search_agent", _fake_web_search_agent)

    result = await rag_module.call_web_search_agent(
        {
            "external_rag_task": "查天气",
            "external_rag_context": {},
            "external_rag_expected_response": "简短回答",
        }
    )

    assert result["external_rag_results"] == ["This text should still be treated as empty by explicit flag."]
    assert result["external_rag_is_empty_result"] is True


@pytest.mark.asyncio
async def test_call_memory_retriever_agent_preserves_explicit_empty_flag(monkeypatch):
    async def _fake_memory_retriever_agent(*, task: str, context: dict, expected_response: str):
        return {
            "response": "This text should still be treated as empty by explicit flag.",
            "is_empty_result": True,
        }

    monkeypatch.setattr(rag_module, "memory_retriever_agent", _fake_memory_retriever_agent)

    result = await rag_module.call_memory_retriever_agent_input_context_rag(
        {
            "input_context_context": {},
            "user_name": "TestUser",
            "global_user_id": "uuid-1",
            "platform": "discord",
            "platform_channel_id": "chan-1",
            "input_context_to_timestamp": "2026-04-24T00:00:00+00:00",
            "platform_bot_id": "bot-1",
            "input_context_task": "查之前聊过什么",
            "input_context_expected_response": "简短回答",
        }
    )

    assert result["input_context_results"] == ["This text should still be treated as empty by explicit flag."]
    assert result["input_context_is_empty_result"] is True
