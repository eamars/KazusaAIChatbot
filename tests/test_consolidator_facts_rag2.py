"""Tests for fact harvester prompts against the RAG2 payload shape."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_facts as facts_module


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
        return _DummyResponse(json.dumps(self._response_payload, ensure_ascii=False))


def _state() -> dict:
    return {
        "character_profile": {"name": "杏山千纱"},
        "user_name": "提拉米苏",
        "timestamp": "2026-04-27T00:00:00+12:00",
        "decontexualized_input": "记住我喜欢红茶。",
        "rag_result": {
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [
                        {
                            "fact": "提拉米苏喜欢绿茶",
                            "subjective_appraisal": "这是已有饮品偏好。",
                            "relationship_signal": "饮品相关回应可参考该事实。",
                        }
                    ],
                    "milestones": [],
                    "active_commitments": [],
                }
            },
            "character_image": {"self_image": {"historical_summary": "谨慎"}},
            "memory_evidence": [{"summary": "旧记忆", "content": "提拉米苏喜欢绿茶"}],
            "conversation_evidence": ["刚才聊到饮料"],
            "external_evidence": [],
            "third_party_profiles": [],
            "supervisor_trace": {"loop_count": 1, "unknown_slots": [], "dispatched": []},
        },
        "existing_dedup_keys": {"drink_preference_green_tea"},
        "action_directives": {"linguistic_directives": {"content_anchors": ["[DECISION] 接受记住"]}},
        "final_dialog": ["嗯，我会记住你喜欢红茶。"],
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "fact_harvester_feedback_message": [],
    }


@pytest.mark.asyncio
async def test_facts_harvester_receives_rag2_payload_and_dedup_keys(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({"new_facts": [], "future_promises": []})
    monkeypatch.setattr(facts_module, "_facts_harvester_llm", llm)

    await facts_module.facts_harvester(_state())

    system_prompt = llm.messages[0].content
    payload = json.loads(llm.messages[1].content)
    assert "research_facts" not in system_prompt
    assert "RAG 元信息" not in system_prompt
    assert "义务主体是不是" in system_prompt
    assert "角色建议用户怎么做，不等于角色承诺自己会做" in system_prompt
    assert "建议/方案不是承诺" in system_prompt
    assert "主语替换自检" in system_prompt
    assert payload["rag_result"]["memory_evidence"][0]["summary"] == "旧记忆"
    assert payload["supervisor_trace"]["loop_count"] == 1
    assert payload["existing_dedup_keys"] == ["drink_preference_green_tea"]


@pytest.mark.asyncio
async def test_fact_harvester_evaluator_reads_rag2_field_names(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({
        "should_stop": True,
        "feedback": "通过审计，无需修改",
        "contradiction_flags": [],
    })
    monkeypatch.setattr(facts_module, "_fact_harvester_evaluator_llm", llm)

    state = {
        **_state(),
        "new_facts": [],
        "future_promises": [],
        "fact_harvester_retry": 0,
        "metadata": {},
    }
    await facts_module.fact_harvester_evaluator(state)

    system_prompt = llm.messages[0].content
    payload = json.loads(llm.messages[1].content)
    assert "rag_result.user_image" in system_prompt
    assert "rag_result.memory_evidence" in system_prompt
    assert "research_facts" not in system_prompt
    memory_context = payload["rag_result"]["user_image"]["user_memory_context"]
    assert memory_context["objective_facts"][0]["fact"] == "提拉米苏喜欢绿茶"
