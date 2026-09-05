"""RAG-side integration tests for past-dialog cognition residual."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import pytest
from pymongo.errors import PyMongoError



def _rag_result_with_conversation_ref(*, row_id: str = "row-1") -> dict[str, Any]:
    return {
        "answer": "A prior Kazusa dialog was retrieved.",
        "conversation_evidence": [{
            "summary": "Kazusa said the idea needed time.",
        }],
        "supervisor_trace": {
            "dispatched": [{
                "agent": "conversation_evidence_agent",
                "source_refs": [{
                    "conversation_row_id": row_id,
                    "platform_message_id": "platform-only-is-not-enough",
                }],
            }],
        },
    }


@pytest.mark.asyncio
async def test_rag_source_refs_resolve_rows_by_row_id_without_public_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    captured_row_ids: list[str] = []
    captured_trace_ids: list[str] = []

    async def list_rows(
        row_ids: Sequence[str],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        captured_row_ids.extend(row_ids)
        assert limit == 3
        return [{
            "_id": "mongo-row-1",
            "conversation_row_id": "row-1",
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "role": "assistant",
            "platform_message_id": "platform-1",
            "global_user_id": "character-1",
            "display_name": "Kazusa",
            "body_text": "The idea needed time.",
            "llm_trace_id": "trace-1",
            "timestamp": "2026-06-01T00:00:00Z",
        }]

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        captured_trace_ids.extend(trace_ids)
        return [{
            "trace_id": "trace-1",
            "stage_name": "l2a_conscious_framing",
            "sequence": 1,
            "parsed_output": {
                "internal_monologue": "private residual from the old turn",
            },
            "created_at": "2026-06-01T00:00:01Z",
        }]

    monkeypatch.setattr(runtime, "list_conversation_rows_by_row_ids", list_rows)
    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )
    rag_result = _rag_result_with_conversation_ref()
    public_before = deepcopy(rag_result)

    result = await runtime.build_past_dialog_cognition_context_from_rag_result(
        rag_result,
        character_global_user_id="character-1",
    )

    assert captured_row_ids == ["row-1"]
    assert captured_trace_ids == ["trace-1"]
    assert result["selected_count"] == 1
    assert "private residual from the old turn" in (
        result["past_dialog_cognition_context"]
    )
    assert rag_result == public_before
    assert "private residual from the old turn" not in str(rag_result)


@pytest.mark.asyncio
async def test_rag_source_refs_without_row_ids_do_not_use_platform_message_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    async def list_rows(
        row_ids: Sequence[str],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        raise AssertionError(f"unexpected row lookup: {list(row_ids)}")

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        raise AssertionError(f"unexpected trace lookup: {list(trace_ids)}")

    monkeypatch.setattr(runtime, "list_conversation_rows_by_row_ids", list_rows)
    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )
    rag_result = _rag_result_with_conversation_ref(row_id="")

    result = await runtime.build_past_dialog_cognition_context_from_rag_result(
        rag_result,
        character_global_user_id="character-1",
    )

    assert result["past_dialog_cognition_context"] == ""
    assert result["selected_count"] == 0


@pytest.mark.asyncio
async def test_rag_row_database_error_is_forgotten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    async def list_rows(
        row_ids: Sequence[str],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        del row_ids, limit
        raise PyMongoError("conversation store unavailable")

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        raise AssertionError(f"unexpected trace lookup: {list(trace_ids)}")

    monkeypatch.setattr(runtime, "list_conversation_rows_by_row_ids", list_rows)
    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    result = await runtime.build_past_dialog_cognition_context_from_rag_result(
        _rag_result_with_conversation_ref(),
        character_global_user_id="character-1",
    )

    assert result["past_dialog_cognition_context"] == ""
    assert result["status"] == "row_lookup_failed"
    assert result["selected_count"] == 0


