"""Tests for fact harvester prompts against the RAG2 payload shape."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_facts as facts_module
from kazusa_ai_chatbot.time_context import build_character_time_context


_COMPACT_CANDIDATE_FIELDS = {
    "unit_id",
    "unit_type",
    "fact",
    "dedup_key",
    "updated_at",
}
_RAW_CANDIDATE_MARKERS = (
    "RAW_CONTENT_MARKER",
    "RAW_APPRAISAL_MARKER",
    "RAW_SIGNAL_MARKER",
    "RAW_SOURCE_MARKER",
    "RAW_EVIDENCE_MARKER",
    "RAW_MERGE_MARKER",
    "RAW_ARBITRARY_MARKER",
)


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
        "time_context": build_character_time_context("2026-04-27T00:00:00+12:00"),
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
            "recall_evidence": [
                {
                    "selected_summary": "当前进度是已经约好九点半出发。",
                    "primary_source": "conversation_progress",
                    "supporting_sources": [],
                }
            ],
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


def _oversized_memory_unit_candidates() -> list[dict]:
    candidates = []
    for index in range(20):
        candidates.append({
            "unit_id": f"unit-{index}",
            "unit_type": "objective_fact",
            "fact": f"candidate fact {index} " + ("long fact " * 80),
            "dedup_key": f"candidate_{index}",
            "updated_at": "2026-05-13T07:00:00+00:00",
            "content": f"RAW_CONTENT_MARKER_{index}" + (" raw" * 100),
            "subjective_appraisal": f"RAW_APPRAISAL_MARKER_{index}",
            "relationship_signal": f"RAW_SIGNAL_MARKER_{index}",
            "source_refs": [{"source": f"RAW_SOURCE_MARKER_{index}"}],
            "evidence_refs": [{"source": f"RAW_EVIDENCE_MARKER_{index}"}],
            "merge_history": [{"reason": f"RAW_MERGE_MARKER_{index}"}],
            "arbitrary_metadata": f"RAW_ARBITRARY_MARKER_{index}",
            "embedding": [0.1, 0.2, 0.3],
        })

    return_value = candidates
    return return_value


def _state_with_oversized_candidates() -> dict:
    state = _state()
    rag_result = dict(state["rag_result"])
    rag_result["user_memory_unit_candidates"] = _oversized_memory_unit_candidates()
    state["rag_result"] = rag_result
    return_value = state
    return return_value


def _assert_compact_memory_unit_candidates(payload: dict) -> None:
    candidates = payload["rag_result"]["user_memory_unit_candidates"]
    assert len(candidates) == 12

    for index, candidate in enumerate(candidates):
        assert set(candidate) == _COMPACT_CANDIDATE_FIELDS
        assert candidate["unit_id"] == f"unit-{index}"
        assert candidate["unit_type"] == "objective_fact"
        assert candidate["dedup_key"] == f"candidate_{index}"
        assert candidate["updated_at"]
        assert len(candidate["fact"]) <= 240

    rendered_candidates = json.dumps(candidates, ensure_ascii=False)
    for marker in _RAW_CANDIDATE_MARKERS:
        assert marker not in rendered_candidates


@pytest.mark.asyncio
async def test_facts_harvester_receives_rag2_payload_and_dedup_keys(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({"new_facts": [], "future_promises": []})
    monkeypatch.setattr(facts_module, "_facts_harvester_llm", llm)

    await facts_module.facts_harvester(_state())

    system_prompt = llm.messages[0].content
    payload = json.loads(llm.messages[1].content)
    assert "research_facts" not in system_prompt
    assert "RAG" not in system_prompt
    assert "RAG 元信息" not in system_prompt
    assert "`rag_result.memory_evidence`、`conversation_evidence`、`external_evidence`" in system_prompt
    assert "`rag_result.recall_evidence`" in system_prompt
    assert "conversation_progress" in system_prompt
    assert "只能作为回合操作证据" in system_prompt
    assert "现实义务主体是不是" in system_prompt
    assert "来源权威性" in system_prompt
    assert "生成回复自污染禁止" in system_prompt
    assert "若候选事实主语是" in system_prompt
    assert "第一人称回答只属于本轮台词" in system_prompt
    assert "只是建议用户怎么做" in system_prompt
    assert "不输出该候选 promise" in system_prompt
    assert "`action` 只写承诺本体" in system_prompt
    assert "`due_time`" in system_prompt
    assert payload["rag_result"]["memory_evidence"][0]["summary"] == "旧记忆"
    assert payload["rag_result"]["recall_evidence"][0]["primary_source"] == "conversation_progress"
    assert payload["supervisor_trace"]["loop_count"] == 1
    assert payload["existing_dedup_keys"] == ["drink_preference_green_tea"]


@pytest.mark.asyncio
async def test_facts_harvester_compacts_raw_memory_unit_candidates(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({"new_facts": [], "future_promises": []})
    monkeypatch.setattr(facts_module, "_facts_harvester_llm", llm)

    await facts_module.facts_harvester(_state_with_oversized_candidates())

    payload = json.loads(llm.messages[1].content)
    _assert_compact_memory_unit_candidates(payload)


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
    assert "rag_result.recall_evidence" in system_prompt
    assert "progress-only recall" in system_prompt
    assert "RAG" not in system_prompt
    assert "`supervisor_trace` 只能作为检索充分性参考" in system_prompt
    assert "来源权威性审计" in system_prompt
    assert "第一人称回答" in system_prompt
    assert "四步链" in system_prompt
    assert "建议用户怎么做" in system_prompt
    assert "时间性事实审计" in system_prompt
    assert "future_promises 红线" in system_prompt
    assert "future_promises.action" in system_prompt
    assert "due_time" in system_prompt
    assert "research_facts" not in system_prompt
    memory_context = payload["rag_result"]["user_image"]["user_memory_context"]
    assert memory_context["objective_facts"][0]["fact"] == "提拉米苏喜欢绿茶"


@pytest.mark.asyncio
async def test_fact_harvester_evaluator_compacts_raw_memory_unit_candidates(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({
        "should_stop": True,
        "feedback": "通过审计，无需修改",
        "contradiction_flags": [],
    })
    monkeypatch.setattr(facts_module, "_fact_harvester_evaluator_llm", llm)

    state = {
        **_state_with_oversized_candidates(),
        "new_facts": [],
        "future_promises": [],
        "fact_harvester_retry": 0,
        "metadata": {},
    }
    await facts_module.fact_harvester_evaluator(state)

    payload = json.loads(llm.messages[1].content)
    _assert_compact_memory_unit_candidates(payload)
