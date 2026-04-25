from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as cognition_l2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag as rag_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_executors as rag_executors_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_resolution as rag_resolution_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor as rag_supervisor_module


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
    assert "reflection_summary" not in human_payload


def test_build_character_profile_results_uses_strict_allowlist():
    profile_results = rag_module._build_character_profile_results(
        {
            "name": "杏山千纱",
            "description": "公开描述",
            "gender": "女",
            "age": 15,
            "birthday": "8月5日",
            "backstory": "公开背景",
            "boundary_profile": {"self_integrity": 0.6},
            "linguistic_texture_profile": {"fragmentation": 0.3},
        }
    )

    assert "### 角色公开资料" in profile_results
    assert "- 姓名: 杏山千纱" in profile_results
    assert "- 年龄: 15" in profile_results
    assert "boundary_profile" not in profile_results
    assert "linguistic_texture_profile" not in profile_results


def test_merge_objective_facts_appends_character_profile_results():
    merged = rag_module._merge_objective_facts(
        "用户住在奥克兰",
        "### 角色公开资料\n- 年龄: 15",
    )

    assert "用户住在奥克兰" in merged
    assert "### 角色公开资料" in merged
    assert "- 年龄: 15" in merged


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


@pytest.mark.asyncio
async def test_call_memory_retriever_agent_keeps_dispatcher_subject_scope(monkeypatch):
    captured: dict = {}

    async def _fake_memory_retriever_agent(*, task: str, context: dict, expected_response: str):
        captured["task"] = task
        captured["context"] = dict(context)
        captured["expected_response"] = expected_response
        return {
            "response": "ok",
            "is_empty_result": False,
        }

    monkeypatch.setattr(rag_executors_module, "memory_retriever_agent", _fake_memory_retriever_agent)

    await rag_module.call_memory_retriever_agent_input_context_rag(
        {
            "input_context_context": {
                "target_user_name": "ThirdParty",
                "target_global_user_id": "uuid-third-party",
                "entities": ["ThirdParty"],
            },
            "user_name": "CurrentUser",
            "global_user_id": "uuid-current-user",
            "platform": "discord",
            "platform_channel_id": "chan-1",
            "input_context_to_timestamp": "2026-04-24T00:00:00+00:00",
            "platform_bot_id": "bot-1",
            "input_context_task": "查第三方最近聊过什么",
            "input_context_expected_response": "简短回答",
        }
    )

    assert captured["context"]["target_user_name"] == "ThirdParty"
    assert captured["context"]["target_global_user_id"] == "uuid-third-party"
    assert captured["context"]["target_platform"] == "discord"
    assert captured["context"]["target_platform_channel_id"] == "chan-1"
    assert captured["context"]["target_to_timestamp"] == "2026-04-24T00:00:00+00:00"


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
        "reply_context": {},
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
async def test_continuation_resolver_includes_reply_context_for_reply_only_confirmation(monkeypatch):
    llm = _CapturingAsyncLLM({
        "needs_context_resolution": True,
        "resolved_task": "用户确认自己是在要求千纱说明白对自己的具体评价。",
        "known_slots": {"target": "current_user", "task": "specific_self_evaluation"},
        "missing_slots": [],
        "confidence": 0.92,
        "evidence": ["reply_context", "recent_turn"],
    })
    monkeypatch.setattr(rag_resolution_module, "_continuation_resolver_llm", llm)

    state = _make_rag_state(
        decontexualized_input="是的",
        chat_history_recent=[
            {"role": "assistant", "content": "你是想让我怎么定义你呀？是想要一个具体的评价，还是仅仅在随口试探……唔。"},
            {"role": "user", "content": "要千纱的具体评价", "display_name": "TestUser"},
            {"role": "assistant", "content": "评价这种事……你是说，要我说明白对你的看法吗？唔……突然问这些，感觉胸口闷闷的。"},
        ],
        reply_context={
            "reply_to_current_bot": True,
            "reply_to_display_name": "Kazusa",
            "reply_excerpt": "评价这种事……你是说，要我说明白对你的看法吗？唔……突然问这些，感觉胸口闷闷的。",
        },
    )
    result = await rag_resolution_module.continuation_resolver(state)

    payload = json.loads(llm.messages[1].content)
    assert payload["reply_context"]["reply_to_current_bot"] is True
    assert "说明白对你的看法" in payload["reply_context"]["reply_excerpt"]
    assert result["continuation_context"]["needs_context_resolution"] is True
    assert "具体评价" in result["continuation_context"]["resolved_task"]


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
async def test_rag_planner_receives_character_name_in_system_prompt(monkeypatch):
    llm = _CapturingAsyncLLM({
        "retrieval_mode": "NONE",
        "active_sources": [],
        "task": "回答角色自己的公开资料",
        "entities": [],
        "subject": {"kind": "character_self", "primary_entity": "Kazusa"},
        "time_scope": {"kind": "none", "lookback_hours": 72},
        "search_scope": {"same_channel": True, "cross_channel": False, "current_user_only": False},
        "external_task_hint": "",
        "expected_response": "角色公开资料",
    })
    monkeypatch.setattr(rag_resolution_module, "_rag_planner_llm", llm)

    state = _make_rag_state(
        character_profile={"name": "Kazusa"},
        decontexualized_input="你几岁了？",
        continuation_context={
            "needs_context_resolution": False,
            "resolved_task": "你几岁了？",
            "known_slots": {},
            "missing_slots": [],
            "confidence": 1.0,
            "evidence": [],
        },
    )

    result = await rag_resolution_module.rag_planner(state)

    system_prompt = llm.messages[0].content
    payload = json.loads(llm.messages[1].content)
    assert "当前固定扮演的角色名为：Kazusa" in system_prompt
    assert "character_name" not in payload
    assert result["retrieval_plan"]["subject"]["kind"] == "character_self"


@pytest.mark.asyncio
async def test_call_cognition_consciousness_passes_objective_facts(monkeypatch):
    llm = _CapturingAsyncLLM(
        {
            "internal_monologue": "这些属于我可以公开说的资料。",
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
            "reflection_summary": "一切正常。",
        },
        "user_profile": {
            "affinity": 650,
            "character_diary": [],
            "active_commitments": [],
            "last_relationship_insight": "关系稳定。",
        },
        "research_facts": {
            "objective_facts": "### 角色公开资料\n- 姓名: Kazusa\n- 年龄: 15\n- 生日: 8月5日"
        },
        "decontexualized_input": "你几岁了，生日是什么时候？",
        "indirect_speech_context": "",
        "emotional_appraisal": "平静。",
        "interaction_subtext": "对方在询问公开资料。",
        "internal_monologue": "这些属于我可以公开说的资料。",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
    }

    await cognition_l2_module.call_cognition_consciousness(state)

    payload = json.loads(llm.messages[1].content)
    assert "角色公开资料" in payload["research_facts"]["objective_facts"]
    assert "年龄: 15" in payload["research_facts"]["objective_facts"]


def test_evidence_presentation_level_is_branch_based_only():
    assert rag_executors_module._select_evidence_presentation_level("input_context") == "raw_evidence"
    assert rag_executors_module._select_evidence_presentation_level("external") == "source_brief"
    assert rag_executors_module._select_evidence_presentation_level("third_party_profile") == "structured_facts"


def test_merge_expected_response_contract_preserves_existing_contract():
    merged = rag_executors_module._merge_expected_response_contract(
        branch="input_context",
        expected_response="返回最近相关记录",
    )

    assert "返回最近相关记录" in merged
    assert "附加约束" in merged
    assert "不要替角色回答" in merged


@pytest.mark.asyncio
async def test_input_context_dispatcher_uses_resolved_task(monkeypatch):
    llm = _CapturingAsyncLLM({
        "next_action": "memory_retriever_agent",
        "reasoning": "需要内部记忆",
        "task": "检索这条链接之前的讨论",
        "context": {"target_user_input": "用户在确认之前的链接"},
        "expected_response": "简短回答",
    })
    monkeypatch.setattr(rag_executors_module, "_input_context_rag_dispatcher_llm", llm)

    state = _make_rag_state(
        decontexualized_input="这个 https://example.com/page",
        continuation_context={
            "needs_context_resolution": True,
            "resolved_task": "用户在确认是否记得这条链接及其之前讨论内容：https://example.com/page",
            "known_slots": {},
            "missing_slots": [],
            "confidence": 0.9,
            "evidence": ["recent_turn"],
        },
    )
    await rag_executors_module.input_context_rag_dispatcher(state)

    payload = json.loads(llm.messages[1].content)
    assert payload["user_input"] == state["continuation_context"]["resolved_task"]
    assert payload["raw_user_input"] == "这个 https://example.com/page"

    result = await rag_executors_module.input_context_rag_dispatcher(state)
    assert "简短回答" in result["input_context_expected_response"]
    assert "附加约束" in result["input_context_expected_response"]
    assert "不要替角色回答" in result["input_context_expected_response"]


@pytest.mark.asyncio
async def test_external_dispatcher_uses_resolved_task(monkeypatch):
    llm = _CapturingAsyncLLM({
        "next_action": "web_search_agent",
        "reasoning": "需要外部知识",
        "task": "查询该链接对应页面的公开资料",
        "context": {"target_user_input": "链接页面"},
        "expected_response": "简短回答",
    })
    monkeypatch.setattr(rag_executors_module, "_external_rag_dispatcher_llm", llm)

    state = _make_rag_state(
        decontexualized_input="这个 https://example.com/page",
        continuation_context={
            "needs_context_resolution": True,
            "resolved_task": "用户在确认是否记得这条链接及其之前讨论内容：https://example.com/page",
            "known_slots": {},
            "missing_slots": [],
            "confidence": 0.9,
            "evidence": ["recent_turn"],
        },
    )
    await rag_executors_module.external_rag_dispatcher(state)

    payload = json.loads(llm.messages[1].content)
    assert payload["user_input"] == state["continuation_context"]["resolved_task"]
    assert payload["raw_user_input"] == "这个 https://example.com/page"

    result = await rag_executors_module.external_rag_dispatcher(state)
    assert "简短回答" in result["external_rag_expected_response"]
    assert "附加约束" in result["external_rag_expected_response"]
    assert "不要替角色回答" in result["external_rag_expected_response"]


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
    payload = json.loads(result["channel_recent_entity_results"])
    assert payload[0]["entity"] == "好笛有"
    assert payload[0]["messages"][0]["content"] == "这是好笛有的消息"


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
    retriever = AsyncMock(return_value=({
        "user_image": {
            "milestones": [{"event": "Met Kazusa", "category": "social"}],
            "recent_window": [{"summary": "最近在讨论模型"}],
            "historical_summary": "一个量化爱好者",
        },
        "character_diary": [{"entry": "好笛有让千纱教他编程。"}],
        "last_relationship_insight": "擅长提技术问题的人",
    }, {
        "milestones": [{"event": "Met Kazusa", "category": "social"}],
        "character_diary": [{"entry": "好笛有让千纱教他编程。"}],
        "objective_facts": [],
        "active_commitments": [],
        "memories": [],
    }))
    monkeypatch.setattr(
        rag_executors_module,
        "user_image_retriever_agent",
        retriever,
    )
    monkeypatch.setattr(
        rag_executors_module,
        "_third_party_profile_finalizer_llm",
        _CapturingAsyncLLM({
            "response": "好笛有是一个量化爱好者，擅长提技术问题，最近还会来找千纱讨论模型和编程。",
            "is_empty_result": False,
            "reason": "资料足够概括稳定画像。",
        }),
    )

    state = _make_rag_state(
        input_embedding=[0.2, 0.3],
        depth="DEEP",
        retrieval_plan={
            "active_sources": ["THIRD_PARTY_PROFILE"],
            "task": "检索关于好笛有的持久画像/印象信息",
            "expected_response": "中文80-120字总结该人物的稳定画像，突出2-3个特征；若证据有限，只写已知印象，不补全。",
        },
        resolved_entities=[{
            "surface_form": "好笛有",
            "entity_type": "person",
            "resolved_global_user_id": "uuid-haodieyou",
            "resolution_confidence": 0.95,
            "resolution_method": "exact_display_name",
        }],
    )
    result = await rag_executors_module.third_party_profile_rag(state)
    retriever.assert_awaited_once_with(
        "uuid-haodieyou",
        input_embedding=[0.2, 0.3],
        depth="DEEP",
    )
    assert result["third_party_profile_results"] == (
        "好笛有是一个量化爱好者，擅长提技术问题，最近还会来找千纱讨论模型和编程。"
    )


def test_metadata_confidence_bundle_counts_third_party_profile_results():
    confidence_scores, response_confidence = rag_module._metadata_confidence_bundle(
        input_context_results="",
        input_context_is_empty_result=False,
        external_rag_results="",
        external_rag_is_empty_result=False,
        channel_recent_entity_results="",
        third_party_profile_results='[{"entity":"小钳子","historical_summary":"量化爱好者"}]',
    )

    assert confidence_scores["third_party_profile"] > 0.0
    assert response_confidence == confidence_scores["third_party_profile"]


def test_tier_gate_should_run_tier2():
    state = _make_rag_state(retrieval_plan={"active_sources": ["CHANNEL_RECENT_ENTITY"]})
    assert rag_executors_module._should_run_tier2(state) == "run"
    state = _make_rag_state(retrieval_plan={"active_sources": ["THIRD_PARTY_PROFILE"]})
    assert rag_executors_module._should_run_tier2(state) == "run"
    state = _make_rag_state(retrieval_plan={"active_sources": ["EXTERNAL_KNOWLEDGE"]})
    assert rag_executors_module._should_run_tier2(state) == "skip"
    state = _make_rag_state(retrieval_plan={"active_sources": []})
    assert rag_executors_module._should_run_tier2(state) == "skip"


def test_tier_gate_should_run_input_context():
    state = _make_rag_state(retrieval_plan={"retrieval_mode": "CASCADED", "active_sources": ["EXTERNAL_KNOWLEDGE"]})
    assert rag_executors_module._should_run_input_context(state) == "run"
    state = _make_rag_state(retrieval_plan={"retrieval_mode": "EXTERNAL_KNOWLEDGE", "active_sources": ["EXTERNAL_KNOWLEDGE"]})
    assert rag_executors_module._should_run_input_context(state) == "skip"
    state = _make_rag_state(retrieval_plan={"retrieval_mode": "THIRD_PARTY_PROFILE", "active_sources": ["THIRD_PARTY_PROFILE"]})
    assert rag_executors_module._should_run_input_context(state) == "skip"


def test_tier_gate_should_run_tier3():
    state = _make_rag_state(retrieval_plan={"active_sources": ["EXTERNAL_KNOWLEDGE"]})
    assert rag_executors_module._should_run_tier3(state) == "run"
    state = _make_rag_state(retrieval_plan={"active_sources": ["CHANNEL_RECENT_ENTITY"]})
    assert rag_executors_module._should_run_tier3(state) == "skip"


@pytest.mark.asyncio
async def test_evaluator_repair_requires_newly_revealed_entity(monkeypatch):
    llm = _CapturingAsyncLLM({
        "verdict": "needs_repair",
        "reasoning": "原实体缺失",
        "coverage_score": 0.5,
        "missing_entities": ["好笛有"],
        "repair_sources": ["CHANNEL_RECENT_ENTITY"],
    })
    monkeypatch.setattr(rag_supervisor_module, "_evaluator_llm", llm)

    state = _make_rag_state(
        continuation_context={"resolved_task": "好笛有最近聊了什么？"},
        retrieval_plan={
            "retrieval_mode": "CASCADED",
            "active_sources": ["CHANNEL_RECENT_ENTITY"],
            "entities": [{"surface_form": "好笛有", "entity_type": "person"}],
        },
        resolved_entities=[
            {
                "surface_form": "好笛有",
                "entity_type": "person",
                "resolved_global_user_id": "",
                "resolution_confidence": 0.0,
                "resolution_method": "unresolved",
            }
        ],
    )

    result = await rag_supervisor_module.rag_supervisor_evaluator(state)
    assert result["needs_repair"] is False
    assert result["repair_entities"] == []


@pytest.mark.asyncio
async def test_deep_retrieval_graph_runs_external_without_tier2(monkeypatch):
    call_order: list[str] = []

    async def _fake_input_context_dispatcher(state):
        call_order.append("input_context")
        return {
            "input_context_next_action": "end",
            "input_context_task": "",
            "input_context_context": {},
            "input_context_expected_response": "",
        }

    async def _fake_external_dispatcher(state):
        call_order.append("external_dispatcher")
        return {
            "external_rag_next_action": "web_search_agent",
            "external_rag_task": "查外部资料",
            "external_rag_context": {},
            "external_rag_expected_response": "简短回答",
        }

    async def _fake_web_search_agent(state):
        call_order.append("web_search")
        return {
            "external_rag_results": ["外部结果"],
            "external_rag_is_empty_result": False,
        }

    monkeypatch.setattr(rag_module, "input_context_rag_dispatcher", _fake_input_context_dispatcher)
    monkeypatch.setattr(rag_module, "external_rag_dispatcher", _fake_external_dispatcher)
    monkeypatch.setattr(rag_module, "call_web_search_agent", _fake_web_search_agent)

    graph = rag_module._build_retrieval_graph("DEEP", 80)
    result = await graph.ainvoke(
        _make_rag_state(
            retrieval_plan={
                "retrieval_mode": "EXTERNAL_KNOWLEDGE",
                "active_sources": ["EXTERNAL_KNOWLEDGE"],
                "entities": [],
            }
        )
    )

    assert call_order == ["external_dispatcher", "web_search"]
    assert result["external_rag_results"] == ["外部结果"]
