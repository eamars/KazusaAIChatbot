from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_evaluator as module


async def _noop_record_llm_stage_event(**_: object) -> None:
    """Replace event logging in tests that should not write telemetry."""


class _FakeFinalizerLLM:
    def __init__(self, content: str = "final answer") -> None:
        self.payload: dict[str, object] | None = None
        self.calls = 0
        self.content = content

    async def ainvoke(self, messages: list[object]) -> object:
        self.calls += 1
        human_message = messages[-1]
        self.payload = json.loads(human_message.content)

        class _Response:
            content = ""

        response = _Response()
        response.content = self.content
        return response


@pytest.mark.asyncio
async def test_rag_finalizer_passes_time_context_to_llm(monkeypatch) -> None:
    """Finalizer should see local time so relative dates remain grounded."""

    llm = _FakeFinalizerLLM()
    monkeypatch.setattr(module, "_finalizer_llm", llm)
    monkeypatch.setattr(
        module.event_logging,
        "record_llm_stage_event",
        _noop_record_llm_stage_event,
    )

    result = await module.rag_finalizer(
        {
            "original_query": "did we discuss this yesterday",
            "known_facts": [],
            "context": {
                "time_context": {
                    "current_local_datetime": "2026-05-12 08:40",
                    "current_local_weekday": "Tuesday",
                }
            },
        }
    )

    assert result == {"final_answer": "final answer"}
    assert llm.payload is not None
    assert llm.payload["time_context"] == {
        "current_local_datetime": "2026-05-12 08:40",
        "current_local_weekday": "Tuesday",
    }


def test_unresolved_summary_distinguishes_observation_candidates() -> None:
    """Unresolved facts with candidate rows should not say retrieval was empty."""

    summary = module._unresolved_summary(
        "Conversation-evidence: retrieve GPU discussion",
        {
            "observation_candidates": [
                {
                    "content": "Tester at 2026-05-11 20:50: GPU topic",
                    "source": "conversation:msg-1",
                }
            ],
            "missing_context": ["conversation_evidence"],
        },
    )

    assert '检索到了候选证据' in summary
    assert '没有足够确认当前槽位' in summary
    assert '检索没有返回相关的确认结果' not in summary


def test_unresolved_summary_describes_incompatible_route() -> None:
    """Incompatible routes should not be summarized as missing records."""

    summary = module._unresolved_summary(
        "Conversation-evidence: retrieve active agreement",
        {"missing_context": ["incompatible_intent:Recall"]},
    )

    assert '检索来源不匹配' in summary
    assert 'Recall' in summary
    assert '检索没有返回相关的确认结果' not in summary


def test_unresolved_summary_describes_missing_context() -> None:
    """Missing inputs should be described as missing context."""

    summary = module._unresolved_summary(
        "Conversation-evidence: retrieve speaker-specific messages",
        {"missing_context": ["person_ref"]},
    )

    assert '检索缺少必要上下文' in summary
    assert 'person_ref' in summary
    assert '检索没有返回相关的确认结果' not in summary


@pytest.mark.asyncio
async def test_rag_evaluator_promotes_direct_observation_candidate(
    monkeypatch,
) -> None:
    """Existing continuation review can promote a directly answering candidate."""

    async def _fake_assess_continuation(**_: object) -> dict[str, object]:
        decision = {
            "should_continue": False,
            "refined_query": "",
            "reason": '候选证据直接回答当前槽位。',
            "promote_candidate": True,
            "promoted_candidate_indexes": [0],
            "promotion_summary": (
                '提拉米苏在耗材厂看到 jayo、alizee、esun 和拓竹。'
            ),
            "promotion_uncertainty": '无',
        }
        return decision

    monkeypatch.setattr(module, "_assess_continuation", _fake_assess_continuation)

    result = await module.rag_evaluator(
        {
            "original_query": '耗材厂那段提拉米苏看到了哪些牌子？',
            "current_slot": (
                "Conversation-evidence: retrieve messages mentioning "
                "'提拉米苏' and '耗材厂'"
            ),
            "unknown_slots": [
                (
                    "Conversation-evidence: retrieve messages mentioning "
                    "'提拉米苏' and '耗材厂'"
                )
            ],
            "known_facts": [],
            "loop_count": 1,
            "last_agent_result": {
                "agent": "conversation_evidence_agent",
                "resolved": False,
                "attempts": 1,
                "result": {
                    "selected_summary": "",
                    "missing_context": ["conversation_evidence"],
                    "observation_candidates": [
                        {
                            "content": (
                                '提拉米苏: 去耗材厂看了一圈，啥标啥包装都有。'
                                '什么jayo alizee esun 甚至还有拓竹……'
                            ),
                            "source": "conversation:platform_message_id:319042975",
                        }
                    ],
                    "projection_payload": {
                        "rows": [
                            {
                                "summary": (
                                    '提拉米苏: 去耗材厂看了一圈，啥标啥包装都有。'
                                    '什么jayo alizee esun 甚至还有拓竹……'
                                ),
                                "display_name": '提拉米苏',
                            }
                        ]
                    },
                },
            },
        }
    )

    fact = result["known_facts"][0]
    assert fact["resolved"] is True
    assert fact["summary"] == '提拉米苏在耗材厂看到 jayo、alizee、esun 和拓竹。'
    assert fact["raw_result"]["selected_summary"] == fact["summary"]
    assert fact["raw_result"]["promotion_source"] == "candidate_evidence"
    assert fact["continuation"]["promote_candidate"] is True
    assert result["unknown_slots"] == []


def test_unresolved_finalizer_keeps_nearby_candidates_concise() -> None:
    """All-unresolved final output should not dump every failed slot."""

    answer = module._unresolved_finalizer_answer(
        [
            {
                "slot": "Recall: retrieve active agreement",
                "agent": "recall_agent",
                "resolved": False,
                "summary": "large internal summary",
                "raw_result": {
                    "missing_context": ["recall_evidence"],
                    "candidates": [
                        {
                            "source": "user_memory_units",
                            "claim": "A nearby but unconfirmed commitment.",
                        },
                        {
                            "source": "user_memory_units",
                            "claim": "Another nearby but unconfirmed commitment.",
                        },
                    ],
                },
                "attempts": 1,
            },
            {
                "slot": "Conversation-evidence: retrieve exact message",
                "agent": "conversation_evidence_agent",
                "resolved": False,
                "summary": "another large internal summary",
                "raw_result": {
                    "missing_context": ["conversation_evidence"],
                },
                "attempts": 1,
            },
        ]
    )

    assert '本次 RAG 没有找到已确认事实' in answer
    assert '已检查来源' in answer
    assert '附近但未确认的候选' in answer
    assert "A nearby but unconfirmed commitment" in answer
    assert "未解决槽位" not in answer
    assert len(answer) < 500


@pytest.mark.asyncio
async def test_rag_finalizer_all_unresolved_uses_deterministic_summary(
    monkeypatch,
) -> None:
    """All-unresolved RAG output should not ask the LLM to infer absence."""

    llm = _FakeFinalizerLLM()
    monkeypatch.setattr(module, "_finalizer_llm", llm)
    monkeypatch.setattr(
        module.event_logging,
        "record_llm_stage_event",
        _noop_record_llm_stage_event,
    )

    result = await module.rag_finalizer(
        {
            "original_query": "did the character ask for a definition",
            "known_facts": [
                {
                    "slot": "Conversation-evidence: retrieve definition request",
                    "agent": "conversation_evidence_agent",
                    "resolved": False,
                    "summary": 'Source mismatch; slot not confirmed.',
                    "raw_result": {
                        "missing_context": ["incompatible_intent:Recall"],
                    },
                    "attempts": 1,
                }
            ],
            "context": {},
        }
    )

    assert llm.calls == 0
    assert '没有找到已确认事实' in result["final_answer"]
    assert '来源不匹配' in result["final_answer"]


@pytest.mark.asyncio
async def test_rag_finalizer_fact_without_resolved_uses_llm(
    monkeypatch,
) -> None:
    """Only explicit unresolved facts should use the deterministic branch."""

    llm = _FakeFinalizerLLM()
    monkeypatch.setattr(module, "_finalizer_llm", llm)
    monkeypatch.setattr(
        module.event_logging,
        "record_llm_stage_event",
        _noop_record_llm_stage_event,
    )

    result = await module.rag_finalizer(
        {
            "original_query": "legacy fact shape",
            "known_facts": [
                {
                    "slot": "Legacy slot",
                    "agent": "legacy_agent",
                    "summary": "Legacy summary",
                    "raw_result": {"summary": "Legacy summary"},
                    "attempts": 1,
                }
            ],
            "context": {},
        }
    )

    assert llm.calls == 1
    assert result["final_answer"] == "final answer"


@pytest.mark.asyncio
async def test_rag_finalizer_compacts_resolved_recall_raw_result(
    monkeypatch,
) -> None:
    """Resolved Recall raw internals should not enter the finalizer prompt."""

    llm = _FakeFinalizerLLM()
    monkeypatch.setattr(module, "_finalizer_llm", llm)
    monkeypatch.setattr(
        module.event_logging,
        "record_llm_stage_event",
        _noop_record_llm_stage_event,
    )

    await module.rag_finalizer(
        {
            "original_query": "is the noodle promise still active",
            "known_facts": [
                {
                    "slot": "Recall: retrieve active agreement",
                    "agent": "recall_agent",
                    "resolved": True,
                    "summary": "The user promised noodles as compensation.",
                    "raw_result": {
                        "selected_summary": (
                            "The user promised noodles as compensation."
                        ),
                        "recall_type": "durable_commitment",
                        "primary_source": "user_memory_units",
                        "freshness_basis": (
                            "Active-episode state was unavailable."
                        ),
                        "candidates": [
                            {
                                "source": "user_memory_units",
                                "claim": "The user promised noodles.",
                                "evidence_time": (
                                    "2026-05-22T10:16:47.993342+00:00"
                                ),
                            }
                        ],
                    },
                    "attempts": 1,
                }
            ],
            "context": {},
        }
    )

    assert llm.payload is not None
    finalizer_facts = llm.payload["known_facts"]
    recall_raw = finalizer_facts[0]["raw_result"]
    assert recall_raw == {
        "selected_summary": "The user promised noodles as compensation.",
        "freshness_basis": "Active-episode state was unavailable.",
    }
    payload_text = json.dumps(llm.payload, ensure_ascii=False)
    assert "user_memory_units" not in payload_text
    assert "2026-05-22T10:16:47.993342+00:00" not in payload_text


@pytest.mark.asyncio
async def test_rag_finalizer_sanitizes_prompt_unsafe_output(
    monkeypatch,
) -> None:
    """Finalizer output should be public-facing even when the LLM leaks terms."""

    llm = _FakeFinalizerLLM(
        content=(
            "recall_agent says user_memory_units confirmed durable_commitment "
            "at 2026-05-22T10:16:47.993342+00:00."
        )
    )
    monkeypatch.setattr(module, "_finalizer_llm", llm)
    monkeypatch.setattr(
        module.event_logging,
        "record_llm_stage_event",
        _noop_record_llm_stage_event,
    )

    result = await module.rag_finalizer(
        {
            "original_query": "is the noodle promise still active",
            "known_facts": [
                {
                    "slot": "Recall: retrieve active agreement",
                    "agent": "recall_agent",
                    "resolved": True,
                    "summary": "The user promised noodles.",
                    "raw_result": {
                        "selected_summary": "The user promised noodles.",
                        "recall_type": "durable_commitment",
                    },
                    "attempts": 1,
                }
            ],
            "context": {},
        }
    )

    answer = result["final_answer"]
    assert "2026-05-22T10:16:47.993342+00:00" not in answer
    assert "user_memory_units" not in answer
    assert "recall_agent" not in answer
    assert "durable_commitment" not in answer
    assert "2026-05-22 22:16" in answer


@pytest.mark.asyncio
async def test_rag_finalizer_hides_unresolved_raw_candidates(
    monkeypatch,
) -> None:
    """Mixed finalizer input must not expose unresolved candidates as facts."""

    llm = _FakeFinalizerLLM()
    monkeypatch.setattr(module, "_finalizer_llm", llm)
    monkeypatch.setattr(
        module.event_logging,
        "record_llm_stage_event",
        _noop_record_llm_stage_event,
    )

    await module.rag_finalizer(
        {
            "original_query": "what commitment is active",
            "known_facts": [
                {
                    "slot": "Conversation-evidence: retrieve meal plan",
                    "agent": "conversation_evidence_agent",
                    "resolved": True,
                    "summary": "The user made dinner.",
                    "raw_result": {
                        "capability": "conversation_evidence",
                        "selected_summary": "The user made dinner.",
                        "evidence": ["The user made dinner."],
                        "missing_context": [],
                    },
                    "attempts": 1,
                },
                {
                    "slot": "Recall: retrieve active agreement",
                    "agent": "recall_agent",
                    "resolved": False,
                    "summary": "Recall did not confirm the active agreement.",
                    "raw_result": {
                        "missing_context": ["recall_evidence"],
                        "candidates": [
                            {
                                "claim": (
                                    "This unconfirmed candidate must not enter "
                                    "the finalizer prompt."
                                )
                            }
                        ],
                    },
                    "attempts": 1,
                },
            ],
            "context": {},
        }
    )

    assert llm.payload is not None
    finalizer_facts = llm.payload["known_facts"]
    unresolved_fact = finalizer_facts[1]
    assert unresolved_fact["raw_result"] == {
        "missing_context": ["recall_evidence"],
        "selected_summary": "",
    }
    payload_text = json.dumps(llm.payload, ensure_ascii=False)
    assert "This unconfirmed candidate must not enter" not in payload_text
