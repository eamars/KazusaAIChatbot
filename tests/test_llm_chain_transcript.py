"""Deterministic tests for protected Cognition V3 chain-transcript capture."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot import llm_tracing as tracing
from kazusa_ai_chatbot.llm_tracing import chain_transcript


@pytest.mark.asyncio
async def test_chain_transcript_capture_obeys_off_metadata_full_modes(
    monkeypatch,
):
    """Off skips, metadata hashes, and full stores exact messages."""

    written: list[dict] = []

    async def insert_step(document: dict) -> str:
        written.append(document)
        return document["step_id"]

    messages = [
        SystemMessage(content="system manual"),
        HumanMessage(content="first question"),
        AIMessage(content="accepted answer"),
    ]
    steps = [
        {
            "step_id": "A1",
            "status": "accepted",
            "attempt_count": 1,
        }
    ]

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "off")
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    off_result = await tracing.record_cognition_chain_transcript(
        trace_id="trace-1",
        run_id="run-1",
        messages=messages,
        steps=steps,
        terminal_disposition="complete",
    )
    assert off_result["status"] == "skipped"
    assert written == []

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    metadata_result = await tracing.record_cognition_chain_transcript(
        trace_id="trace-1",
        run_id="run-1",
        messages=messages,
        steps=steps,
        terminal_disposition="complete",
    )
    assert metadata_result["status"] == "recorded"
    metadata_doc = written[-1]
    assert metadata_doc["raw_messages"] == []
    assert metadata_doc["steps"] == []
    assert metadata_doc["message_count"] == 3
    assert len(metadata_doc["message_hashes"]) == 3
    assert metadata_doc["message_lengths"] == [
        len("system manual"),
        len("first question"),
        len("accepted answer"),
    ]
    assert metadata_doc["terminal_disposition"] == "complete"

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "full")
    full_result = await tracing.record_cognition_chain_transcript(
        trace_id="trace-1",
        run_id="run-1",
        messages=messages,
        steps=steps,
        terminal_disposition="complete",
        chain_model_name="chain-model",
        sidecar_model_name="sidecar-model",
    )
    assert full_result["status"] == "recorded"
    full_doc = written[-1]
    assert full_doc["raw_messages"] == [
        {"role": "system", "content": "system manual"},
        {"role": "human", "content": "first question"},
        {"role": "ai", "content": "accepted answer"},
    ]
    assert full_doc["steps"] == steps
    assert full_doc["chain_model_name"] == "chain-model"
    assert full_doc["sidecar_model_name"] == "sidecar-model"


def test_trace_facade_exposes_only_scoped_chain_capture() -> None:
    """The chain-transcript facade exposes the exact scoped function."""

    assert chain_transcript.record_cognition_chain_transcript is (
        tracing.record_cognition_chain_transcript
    )
    assert chain_transcript.__all__ == [
        "record_cognition_chain_transcript"
    ]
