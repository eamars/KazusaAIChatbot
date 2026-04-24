from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as cognition_l2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag as rag_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_executors as rag_executors_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_resolution as rag_resolution_module


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

    monkeypatch.setattr(rag_executors_module, "web_search_agent", _fake_web_search_agent)

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

    monkeypatch.setattr(rag_executors_module, "memory_retriever_agent", _fake_memory_retriever_agent)

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


# ── Phase 7 — Third-Party Recall unit tests ──────────────────────


def _make_rag_state(**overrides) -> dict:
    base = {
        "timestamp": "2026-04-24T12:00:00+00:00",
        "platform": "discord",
        "platform_channel_id": "chan-1",
        "platform_message_id": "msg-1",
        "decontexualized_input": "好笛有最近聊了什么？",
        "channel_topic": "闲聊",
        "input_context_to_timestamp": "2026-04-24T12:00:00+00:00",
        "chat_history_recent": [],
        "user_name": "TestUser",
        "global_user_id": "uuid-self",
        "platform_bot_id": "bot-1",
        "character_profile": {"name": "Kazusa"},
        "user_profile": {"affinity": 650},
        "input_embedding": [0.0] * 10,
        "depth": "DEEP",
        "depth_confidence": 0.9,
        "cache_hit": False,
        "trigger_dispatchers": [],
        "rag_metadata": {},
        "continuation_context": {},
        "retrieval_plan": {},
        "resolved_entities": [],
        "retrieval_ledger": {},
        "channel_recent_entity_results": "",
        "third_party_profile_results": "",
        "entity_resolution_notes": "",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_continuation_resolver_complete_input(monkeypatch):
    llm = _CapturingAsyncLLM({
        "needs_context_resolution": False,
        "resolved_task": "好笛有最近聊了什么？",
        "known_slots": {},
        "missing_slots": [],
        "confidence": 0.95,
        "evidence": [],
    })
    monkeypatch.setattr(rag_resolution_module, "_continuation_resolver_llm", llm)
    state = _make_rag_state()
    result = await rag_resolution_module.continuation_resolver(state)
    assert result["continuation_context"]["needs_context_resolution"] is False
    assert result["continuation_context"]["confidence"] == 0.95


@pytest.mark.asyncio
async def test_continuation_resolver_continuation_input(monkeypatch):
    llm = _CapturingAsyncLLM({
        "needs_context_resolution": True,
        "resolved_task": "好笛有最近的量化讨论结果是什么？",
        "known_slots": {"subject": "好笛有"},
        "missing_slots": [],
        "confidence": 0.8,
        "evidence": ["recent_turn"],
    })
    monkeypatch.setattr(rag_resolution_module, "_continuation_resolver_llm", llm)
    state = _make_rag_state(decontexualized_input="然后呢？")
    result = await rag_resolution_module.continuation_resolver(state)
    assert result["continuation_context"]["needs_context_resolution"] is True
    assert "量化" in result["continuation_context"]["resolved_task"]


@pytest.mark.asyncio
async def test_rag_planner_emits_channel_recent_entity(monkeypatch):
    llm = _CapturingAsyncLLM({
        "retrieval_mode": "CHANNEL_RECENT_ENTITY",
        "active_sources": ["CHANNEL_RECENT_ENTITY"],
        "task": "查找好笛有的近期对话",
        "entities": [{"surface_form": "好笛有", "entity_type": "person", "resolution_confidence": 0.9}],
        "subject": {"kind": "third_party_user", "primary_entity": "好笛有"},
        "time_scope": {"kind": "recent", "lookback_hours": 72},
        "search_scope": {"same_channel": True, "cross_channel": False, "current_user_only": False},
        "external_task_hint": "",
        "expected_response": "好笛有的近期对话摘要",
    })
    monkeypatch.setattr(rag_resolution_module, "_rag_planner_llm", llm)
    state = _make_rag_state(continuation_context={
        "needs_context_resolution": False,
        "resolved_task": "好笛有最近聊了什么？",
        "known_slots": {},
        "missing_slots": [],
        "confidence": 0.95,
        "evidence": [],
    })
    result = await rag_resolution_module.rag_planner(state)
    plan = result["retrieval_plan"]
    assert plan["retrieval_mode"] == "CHANNEL_RECENT_ENTITY"
    assert "CHANNEL_RECENT_ENTITY" in plan["active_sources"]
    assert len(plan["entities"]) == 1
    assert plan["entities"][0]["surface_form"] == "好笛有"


@pytest.mark.asyncio
async def test_entity_grounder_resolves_exact_display_name():
    state = _make_rag_state(
        retrieval_plan={
            "entities": [{"surface_form": "好笛有", "entity_type": "person"}],
        },
        chat_history_recent=[
            {"display_name": "好笛有", "global_user_id": "uuid-haodieyou", "content": "test"},
            {"display_name": "TestUser", "global_user_id": "uuid-self", "content": "reply"},
        ],
    )
    result = await rag_resolution_module.entity_grounder(state)
    assert len(result["resolved_entities"]) == 1
    entity = result["resolved_entities"][0]
    assert entity["resolved_global_user_id"] == "uuid-haodieyou"
    assert entity["resolution_method"] == "exact_display_name"
    assert entity["resolution_confidence"] == 0.95
    assert "resolved" in result["entity_resolution_notes"]


@pytest.mark.asyncio
async def test_entity_grounder_partial_match():
    state = _make_rag_state(
        retrieval_plan={
            "entities": [{"surface_form": "好笛", "entity_type": "person"}],
        },
        chat_history_recent=[
            {"display_name": "好笛有", "global_user_id": "uuid-haodieyou", "content": "test"},
        ],
    )
    result = await rag_resolution_module.entity_grounder(state)
    assert len(result["resolved_entities"]) == 1
    entity = result["resolved_entities"][0]
    assert entity["resolved_global_user_id"] == "uuid-haodieyou"
    assert entity["resolution_method"] == "partial_display_name"


@pytest.mark.asyncio
async def test_entity_grounder_no_entities_returns_empty():
    state = _make_rag_state(retrieval_plan={"entities": []})
    result = await rag_resolution_module.entity_grounder(state)
    assert result["resolved_entities"] == []
    assert result["entity_resolution_notes"] == ""


@pytest.mark.asyncio
async def test_channel_recent_entity_rag_skips_when_not_active():
    state = _make_rag_state(
        retrieval_plan={"active_sources": []},
        resolved_entities=[{"surface_form": "好笛有", "resolved_global_user_id": "uuid-1"}],
    )
    result = await rag_executors_module.channel_recent_entity_rag(state)
    assert result["channel_recent_entity_results"] == ""


@pytest.mark.asyncio
async def test_channel_recent_entity_rag_fetches_by_id(monkeypatch):
    async def _fake_get_history(**kwargs):
        return [
            {"display_name": "好笛有", "content": "这是好笛有的消息", "timestamp": "2026-04-24T10:00:00", "platform_message_id": "m1"},
        ]

    monkeypatch.setattr(rag_executors_module, "get_conversation_history", _fake_get_history)

    state = _make_rag_state(
        retrieval_plan={
            "active_sources": ["CHANNEL_RECENT_ENTITY"],
            "time_scope": {"kind": "recent", "lookback_hours": 72},
        },
        resolved_entities=[{
            "surface_form": "好笛有",
            "entity_type": "person",
            "resolved_global_user_id": "uuid-haodieyou",
            "resolution_confidence": 0.95,
            "resolution_method": "exact_display_name",
        }],
    )
    result = await rag_executors_module.channel_recent_entity_rag(state)
    assert "好笛有" in result["channel_recent_entity_results"]
    assert "这是好笛有的消息" in result["channel_recent_entity_results"]


@pytest.mark.asyncio
async def test_third_party_profile_rag_skips_when_not_active():
    state = _make_rag_state(
        retrieval_plan={"active_sources": []},
        resolved_entities=[{"surface_form": "X", "resolved_global_user_id": "uuid-x"}],
    )
    result = await rag_executors_module.third_party_profile_rag(state)
    assert result["third_party_profile_results"] == ""


@pytest.mark.asyncio
async def test_third_party_profile_rag_skips_current_user():
    state = _make_rag_state(
        retrieval_plan={"active_sources": ["THIRD_PARTY_PROFILE"]},
        resolved_entities=[{
            "surface_form": "TestUser",
            "entity_type": "person",
            "resolved_global_user_id": "uuid-self",
            "resolution_confidence": 0.95,
            "resolution_method": "exact_display_name",
        }],
    )
    result = await rag_executors_module.third_party_profile_rag(state)
    assert result["third_party_profile_results"] == ""


@pytest.mark.asyncio
async def test_third_party_profile_rag_fetches_profile(monkeypatch):
    async def _fake_get_user_profile(global_user_id):
        return {
            "user_image": {
                "milestones": [{"event": "Met Kazusa", "category": "social"}],
                "historical_summary": "一个量化爱好者",
                "recent_observations": ["最近在讨论模型"],
            },
            "character_diary": [
                {"entry": "好笛有让千纱教他编程。"},
            ],
        }

    monkeypatch.setattr(rag_executors_module, "get_user_profile", _fake_get_user_profile)

    state = _make_rag_state(
        retrieval_plan={"active_sources": ["THIRD_PARTY_PROFILE"]},
        resolved_entities=[{
            "surface_form": "好笛有",
            "entity_type": "person",
            "resolved_global_user_id": "uuid-haodieyou",
            "resolution_confidence": 0.95,
            "resolution_method": "exact_display_name",
        }],
    )
    result = await rag_executors_module.third_party_profile_rag(state)
    assert "好笛有" in result["third_party_profile_results"]
    assert "量化爱好者" in result["third_party_profile_results"]
    assert "编程" in result["third_party_profile_results"]


def test_tier_gate_should_run_tier2():
    state = _make_rag_state(retrieval_plan={"active_sources": ["CHANNEL_RECENT_ENTITY"]})
    assert rag_executors_module._should_run_tier2(state) == "run"
    state = _make_rag_state(retrieval_plan={"active_sources": ["THIRD_PARTY_PROFILE"]})
    assert rag_executors_module._should_run_tier2(state) == "run"
    state = _make_rag_state(retrieval_plan={"active_sources": ["EXTERNAL_KNOWLEDGE"]})
    assert rag_executors_module._should_run_tier2(state) == "skip"
    state = _make_rag_state(retrieval_plan={"active_sources": []})
    assert rag_executors_module._should_run_tier2(state) == "skip"


def test_tier_gate_should_run_tier3():
    state = _make_rag_state(retrieval_plan={"active_sources": ["EXTERNAL_KNOWLEDGE"]})
    assert rag_executors_module._should_run_tier3(state) == "run"
    state = _make_rag_state(retrieval_plan={"active_sources": ["CHANNEL_RECENT_ENTITY"]})
    assert rag_executors_module._should_run_tier3(state) == "skip"
