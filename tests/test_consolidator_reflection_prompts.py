"""Tests for consolidator reflection prompt contracts."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_reflection as reflection_module


class _DummyResponse:
    """Minimal async LLM response wrapper."""

    def __init__(self, content: str) -> None:
        self.content = content


class _CapturingAsyncLLM:
    """Capture messages passed to an async LLM call."""

    def __init__(self, response_payload: dict) -> None:
        self.messages = []
        self._response_payload = response_payload

    async def ainvoke(self, messages):
        self.messages = messages
        response = _DummyResponse(
            json.dumps(self._response_payload, ensure_ascii=False),
        )
        return response


def _state() -> dict:
    return {
        "character_profile": {
            "name": "杏山千纱",
            "personality_brief": {"mbti": "ISTJ"},
        },
        "user_profile": {"affinity": 500},
        "user_name": "测试用户甲",
        "decontexualized_input": "不是在拉开距离，只是顺手整理线材。",
        "internal_monologue": "一瞬间有些局促，但这只是普通任务说明。",
        "emotional_appraisal": "轻微局促。",
        "interaction_subtext": "事务协作。",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "final_dialog": ["那就按你说的来吧。"],
        "action_directives": {
            "linguistic_directives": {
                "content_anchors": ["[FACTUAL] 按普通事务回应。"],
            },
        },
    }


@pytest.mark.asyncio
async def test_global_state_updater_receives_grounding_fields(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({
        "mood": "Neutral",
        "global_vibe": "Softened",
        "reflection_summary": "只是普通整理，没有留下强烈情绪。",
    })
    monkeypatch.setattr(reflection_module, "_global_state_updater_llm", llm)

    await reflection_module.global_state_updater(_state())

    system_prompt = llm.messages[0].content
    payload = json.loads(llm.messages[1].content)
    assert "不会收到完整角色资料" in system_prompt
    assert "final_dialog" in system_prompt
    assert "弱证据" in system_prompt
    assert "强负面状态准入" in system_prompt
    assert payload["logical_stance"] == "CONFIRM"
    assert payload["decontexualized_input"] == "不是在拉开距离，只是顺手整理线材。"


@pytest.mark.asyncio
async def test_relationship_recorder_receives_reassurance_context(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({
        "skip": True,
        "subjective_appraisals": ["should be dropped when skip is true"],
        "affinity_delta": 0,
        "last_relationship_insight": "should not persist",
    })
    monkeypatch.setattr(reflection_module, "_relationship_recorder_llm", llm)

    result = await reflection_module.relationship_recorder(_state())

    system_prompt = llm.messages[0].content
    payload = json.loads(llm.messages[1].content)
    assert "输入纠偏规则" in system_prompt
    assert "不会收到完整角色资料" in system_prompt
    assert "证据分层规则" in system_prompt
    assert "普通任务默认跳过" in system_prompt
    assert payload["decontexualized_input"] == "不是在拉开距离，只是顺手整理线材。"
    assert payload["final_dialog"] == ["那就按你说的来吧。"]
    assert payload["content_anchors"] == ["[FACTUAL] 按普通事务回应。"]
    assert result["affinity_delta"] == 0
    assert result["subjective_appraisals"] == []
    assert result["last_relationship_insight"] == ""
