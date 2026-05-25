"""Tests for fact harvester prompts against the RAG2 payload shape."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.consolidation import facts as facts_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from kazusa_ai_chatbot.utils import DEFAULT_LLM_MAX_COMPLETION_TOKENS


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


def test_fact_harvester_llms_use_shared_completion_token_budget() -> None:
    """Fact extraction calls should not inherit backend default output caps."""

    assert facts_module._facts_harvester_llm.max_tokens == (
        DEFAULT_LLM_MAX_COMPLETION_TOKENS
    )
    assert facts_module._facts_harvester_llm._default_params[
        "max_completion_tokens"
    ] == DEFAULT_LLM_MAX_COMPLETION_TOKENS
    assert facts_module._fact_harvester_evaluator_llm.max_tokens == (
        DEFAULT_LLM_MAX_COMPLETION_TOKENS
    )
    assert facts_module._fact_harvester_evaluator_llm._default_params[
        "max_completion_tokens"
    ] == DEFAULT_LLM_MAX_COMPLETION_TOKENS


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
    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    return {
        "character_profile": {"name": "杏山千纱"},
        "user_name": "提拉米苏",
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "consolidation_origin": {
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "channel_type": "private",
            "platform_message_id": "message-1",
            "active_turn_platform_message_ids": ["message-1"],
            "active_turn_conversation_row_ids": ["row-1"],
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "global-user-1",
            "current_display_name": "提拉米苏",
        },
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
async def test_facts_harvester_receives_prompt_safe_action_trace(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({"new_facts": [], "future_promises": []})
    monkeypatch.setattr(facts_module, "_facts_harvester_llm", llm)
    state = _state()
    state["episode_trace_projection"] = {
        "schema_version": "episode_trace_projection.v1",
        "episode_id": "episode-1",
        "trigger_source": "self_cognition",
        "action_results": [
            {
                "schema_version": "consolidation_action_projection.v1",
                "action_kind": "apply_memory_lifecycle_update",
                "status": "executed",
                "visibility": "private",
                "semantic_decision": "角色决定放弃这个过期承诺。",
                "result_summary": (
                    "apply_memory_lifecycle_update executed: cancelled"
                ),
                "evidence_refs": [],
            }
        ],
        "surface_outputs": [],
    }

    await facts_module.facts_harvester(state)

    system_prompt = llm.messages[0].content
    payload = json.loads(llm.messages[1].content)
    action_trace = payload["episode_trace_projection"]
    rendered = json.dumps(action_trace, ensure_ascii=False)
    assert "episode_trace_projection" in system_prompt
    assert action_trace["action_results"][0]["action_kind"] == (
        "apply_memory_lifecycle_update"
    )
    assert "handler_id" not in rendered
    assert "raw_params" not in rendered
    assert "user_memory_units" not in rendered


@pytest.mark.asyncio
async def test_facts_harvester_accepts_no_surface_action(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({"new_facts": [], "future_promises": []})
    monkeypatch.setattr(facts_module, "_facts_harvester_llm", llm)
    state = _state()
    state["action_directives"] = {}

    await facts_module.facts_harvester(state)

    payload = json.loads(llm.messages[1].content)
    assert payload["content_anchors"] == []


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
async def test_fact_harvester_evaluator_accepts_no_surface_action(monkeypatch) -> None:
    llm = _CapturingAsyncLLM({
        "should_stop": True,
        "feedback": "通过审计，无需修改",
        "contradiction_flags": [],
    })
    monkeypatch.setattr(facts_module, "_fact_harvester_evaluator_llm", llm)
    state = {
        **_state(),
        "action_directives": {},
        "new_facts": [],
        "future_promises": [],
        "fact_harvester_retry": 0,
        "metadata": {},
    }

    await facts_module.fact_harvester_evaluator(state)

    payload = json.loads(llm.messages[1].content)
    assert payload["content_anchors"] == []


@pytest.mark.asyncio
async def test_fact_harvester_evaluator_receives_prompt_safe_action_trace(
    monkeypatch,
) -> None:
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
        "episode_trace_projection": {
            "schema_version": "episode_trace_projection.v1",
            "episode_id": "episode-1",
            "trigger_source": "self_cognition",
            "action_results": [
                {
                    "schema_version": "consolidation_action_projection.v1",
                    "action_kind": "apply_memory_lifecycle_update",
                    "status": "executed",
                    "visibility": "private",
                    "semantic_decision": "角色决定放弃这个过期承诺。",
                    "result_summary": (
                        "apply_memory_lifecycle_update executed: cancelled"
                    ),
                    "evidence_refs": [],
                }
            ],
            "surface_outputs": [],
        },
    }

    await facts_module.fact_harvester_evaluator(state)

    system_prompt = llm.messages[0].content
    payload = json.loads(llm.messages[1].content)
    rendered = json.dumps(payload["episode_trace_projection"], ensure_ascii=False)
    assert "episode_trace_projection" in system_prompt
    assert payload["episode_trace_projection"]["action_results"][0][
        "action_kind"
    ] == "apply_memory_lifecycle_update"
    assert "handler_id" not in rendered
    assert "raw_params" not in rendered


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
