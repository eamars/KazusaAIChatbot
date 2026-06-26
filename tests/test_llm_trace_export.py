from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from scripts import export_llm_trace


@pytest.mark.asyncio
async def test_build_trace_export_uses_trace_id(monkeypatch):
    run = {"trace_id": "trace-1", "status": "succeeded"}
    step = {"trace_id": "trace-1", "stage_name": "dialog_generator"}
    event = {"correlation_id": "trace-1", "event_family": "llm_stage"}
    conversation = {"llm_trace_id": "trace-1", "body_text": "hello"}

    monkeypatch.setattr(
        export_llm_trace.script_operations,
        "export_collection_rows",
        AsyncMock(side_effect=[[run], [step], [event], [conversation]]),
    )

    document = await export_llm_trace.build_trace_export(trace_id="trace-1")

    assert document["query"]["trace_id"] == "trace-1"
    assert document["llm_trace_runs"] == [run]
    assert document["llm_trace_steps"] == [step]
    assert document["event_log_events"] == [event]
    assert document["conversation_history"] == [conversation]


@pytest.mark.asyncio
async def test_resolve_trace_id_from_dialog_text(monkeypatch):
    monkeypatch.setattr(
        export_llm_trace.script_operations,
        "export_collection_rows",
        AsyncMock(return_value=[{"llm_trace_id": "trace-from-dialog"}]),
    )

    trace_id = await export_llm_trace.resolve_trace_id(
        trace_id="",
        dialog_text="14:30了",
        delivery_tracking_id="",
        platform_message_id="",
    )

    assert trace_id == "trace-from-dialog"
