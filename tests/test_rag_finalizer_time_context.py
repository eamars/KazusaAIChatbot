from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_evaluator as module


async def _noop_record_llm_stage_event(**_: object) -> None:
    """Replace event logging in tests that should not write telemetry."""


class _FakeFinalizerLLM:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None
        self.calls = 0

    async def ainvoke(self, messages: list[object]) -> object:
        self.calls += 1
        human_message = messages[-1]
        self.payload = json.loads(human_message.content)

        class _Response:
            content = "final answer"

        return _Response()


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

    assert "检索到候选结果" in summary
    assert "未确认足以解决槽位" in summary
    assert "检索未返回相关结果" not in summary


def test_unresolved_summary_describes_incompatible_route() -> None:
    """Incompatible routes should not be summarized as missing records."""

    summary = module._unresolved_summary(
        "Conversation-evidence: retrieve active agreement",
        {"missing_context": ["incompatible_intent:Recall"]},
    )

    assert '检索来源不匹配' in summary
    assert 'Recall' in summary
    assert '检索未返回相关结果' not in summary


def test_unresolved_summary_describes_missing_context() -> None:
    """Missing inputs should be described as missing context."""

    summary = module._unresolved_summary(
        "Conversation-evidence: retrieve speaker-specific messages",
        {"missing_context": ["person_ref"]},
    )

    assert '缺少必要上下文' in summary
    assert 'person_ref' in summary
    assert '检索未返回相关结果' not in summary


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
                    "summary": '检索来源不匹配，未能确认槽位。',
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
    assert '没有得到已确认事实' in result["final_answer"]
    assert '检索来源不匹配' in result["final_answer"]


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
