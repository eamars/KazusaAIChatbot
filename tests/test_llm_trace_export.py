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


@pytest.mark.asyncio
async def test_build_trace_export_groups_failure_capsules(monkeypatch):
    ordinary_step = {
        "trace_id": "trace-1",
        "stage_name": "cognition_goal",
        "sequence": 1,
    }
    capsule = {
        "schema_version": "cognition_failure_capsule.v1",
        "trace_id": "trace-1",
        "cognition_invocation_id": "invocation-1",
        "entrypoint": "run_cognition",
        "input_payload": {"episode": {"content": "exact input"}},
        "attempts": [{"attempt_index": 1, "raw_response_text": "raw"}],
        "failure_events": [],
        "outcome": "terminal_failure",
        "exception": {"type": "RuntimeError", "message": "failed"},
    }
    capsule_step = {
        "trace_id": "trace-1",
        "stage_name": "cognition_failure_capsule",
        "capture_reason": "cognition_failure_capsule",
        "cognition_invocation_id": "invocation-1",
        "capsule": capsule,
    }
    monkeypatch.setattr(
        export_llm_trace.script_operations,
        "export_collection_rows",
        AsyncMock(
            side_effect=[
                [],
                [ordinary_step, capsule_step],
                [],
            ]
        ),
    )
    monkeypatch.setattr(
        export_llm_trace.script_operations,
        "export_event_log_events_for_trace_id",
        AsyncMock(return_value=[]),
    )

    document = await export_llm_trace.build_trace_export(trace_id="trace-1")

    assert document["llm_trace_steps"] == [ordinary_step, capsule_step]
    assert document["cognition_failure_capsules"] == [capsule]


@pytest.mark.asyncio
async def test_build_trace_export_selects_one_cognition_invocation(monkeypatch):
    first_capsule = {
        "schema_version": "cognition_failure_capsule.v1",
        "trace_id": "trace-1",
        "cognition_invocation_id": "invocation-1",
        "input_payload": {"value": "first"},
        "attempts": [{"attempt_index": 1}],
    }
    second_capsule = {
        "schema_version": "cognition_failure_capsule.v1",
        "trace_id": "trace-1",
        "cognition_invocation_id": "invocation-2",
        "input_payload": {"value": "second"},
        "attempts": [{"attempt_index": 1}],
    }
    steps = [
        {
            "trace_id": "trace-1",
            "capture_reason": "cognition_failure_capsule",
            "cognition_invocation_id": "invocation-1",
            "capsule": first_capsule,
        },
        {
            "trace_id": "trace-1",
            "capture_reason": "cognition_failure_capsule",
            "cognition_invocation_id": "invocation-2",
            "capsule": second_capsule,
        },
    ]
    monkeypatch.setattr(
        export_llm_trace.script_operations,
        "export_collection_rows",
        AsyncMock(side_effect=[[], steps, []]),
    )
    monkeypatch.setattr(
        export_llm_trace.script_operations,
        "export_event_log_events_for_trace_id",
        AsyncMock(return_value=[]),
    )

    document = await export_llm_trace.build_trace_export(
        trace_id="trace-1",
        cognition_invocation_id="invocation-2",
    )

    assert document["query"] == {
        "trace_id": "trace-1",
        "cognition_invocation_id": "invocation-2",
    }
    assert document["llm_trace_steps"] == steps
    assert document["cognition_failure_capsules"] == [second_capsule]
