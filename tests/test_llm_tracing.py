from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import kazusa_ai_chatbot.llm_tracing as tracing


@pytest.mark.asyncio
async def test_record_trace_step_metadata_mode_omits_raw_payload(monkeypatch):
    written: list[dict] = []

    async def insert_step(document: dict) -> str:
        written.append(document)
        return document["step_id"]

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    result = await tracing.record_llm_trace_step(
        trace_id="trace-1",
        stage_name="dialog_generator",
        route_name="DIALOG_GENERATOR_LLM",
        model_name="model-a",
        messages=[
            SystemMessage(content="system secret"),
            HumanMessage(content="hello"),
        ],
        response_text='{"final_dialog":["hi"]}',
        parsed_output={"final_dialog": ["hi"]},
        parse_status="succeeded",
        status="succeeded",
        duration_ms=10,
        output_state_fields=["final_dialog"],
    )

    assert result["status"] == "recorded"
    assert len(written) == 1
    doc = written[0]
    assert doc["prompt_chars"] == len("system secret") + len("hello")
    assert doc["output_chars"] == len('{"final_dialog":["hi"]}')
    assert doc["prompt_sha256"]
    assert doc["output_sha256"]
    assert doc["raw_messages"] == []
    assert doc["raw_response_text"] == ""
    assert doc["parsed_output"] == {}
    assert isinstance(doc["expires_at"], datetime)


@pytest.mark.asyncio
async def test_record_trace_step_off_mode_skips_db_write(monkeypatch):
    insert_step = AsyncMock()

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "off")
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    result = await tracing.record_llm_trace_step(
        trace_id="trace-1",
        stage_name="stage",
        route_name="route",
        model_name="model",
        messages=[HumanMessage(content="hello")],
        response_text="{}",
        parsed_output={},
        parse_status="succeeded",
        status="succeeded",
        duration_ms=1,
        output_state_fields=[],
    )

    assert result["status"] == "skipped"
    insert_step.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_trace_run_updates_status(monkeypatch):
    update_run = AsyncMock()

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(tracing.db_llm_tracing, "update_trace_run", update_run)

    await tracing.finalize_llm_trace_run(
        trace_id="trace-1",
        status="succeeded",
        final_dialog_count=1,
        delivery_tracking_id="delivery-1",
    )

    update_doc = update_run.await_args.kwargs["update_doc"]
    assert update_doc["status"] == "succeeded"
    assert update_doc["final_dialog_count"] == 1
    assert update_doc["delivery_tracking_id"] == "delivery-1"


def test_build_trace_id_is_prefixed():
    trace_id = tracing.build_trace_id()

    assert trace_id.startswith("llmtrace_")
