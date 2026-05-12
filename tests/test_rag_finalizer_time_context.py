from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_evaluator as module


class _FakeFinalizerLLM:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    async def ainvoke(self, messages: list[object]) -> object:
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
