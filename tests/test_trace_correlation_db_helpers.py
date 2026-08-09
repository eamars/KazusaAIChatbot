from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.db import script_operations


@pytest.mark.asyncio
async def test_trace_candidate_reader_uses_typed_exact_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = AsyncMock(return_value=[])
    monkeypatch.setattr(script_operations, "export_collection_rows", reader)

    await script_operations.list_trace_correlation_candidates(
        source_surface="protected_action_attempt_id",
        identifier="attempt-1",
    )

    reader.assert_awaited_once_with(
        collection_name="self_cognition_action_attempts",
        filter_doc={"attempt_id": "attempt-1"},
        projection={
            "_id": 0,
            "source_llm_trace_id": 1,
            "attempt_id": 1,
            "correlation_write_status": 1,
            "correlation_conflict_source_llm_trace_id": 1,
        },
        sort_doc={"trace_id": 1, "source_llm_trace_id": 1},
        limit=2,
    )


@pytest.mark.asyncio
async def test_trace_companion_reader_uses_bounded_identifier_joins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = AsyncMock(return_value=[])
    monkeypatch.setattr(script_operations, "export_collection_rows", reader)

    companions = await script_operations.list_trace_correlation_companions(
        trace_id="llmtrace-parent-1",
    )

    assert set(companions) == {
        "conversation_history",
        "llm_trace_steps",
        "self_cognition_action_attempts",
        "background_work_jobs",
        "calendar_schedules",
        "calendar_runs",
        "child_trace_runs",
    }
    assert reader.await_count == 7
    calls_by_collection = {
        call.kwargs["collection_name"]: call.kwargs
        for call in reader.await_args_list
    }
    for kwargs in calls_by_collection.values():
        assert kwargs["filter_doc"] in (
            {"llm_trace_id": "llmtrace-parent-1"},
            {"trace_id": "llmtrace-parent-1"},
            {"source_llm_trace_id": "llmtrace-parent-1"},
            {"parent_llm_trace_id": "llmtrace-parent-1"},
        )
        assert kwargs["limit"] in (32, 64)
        projection_text = str(kwargs["projection"])
        assert "body_text" not in projection_text
        assert "prompt" not in projection_text
        assert "response" not in projection_text
        assert "embedding" not in projection_text
