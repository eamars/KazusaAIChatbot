"""Deterministic contracts for Cognition V3 chain-run event logging."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.event_logging import recording
from kazusa_ai_chatbot.event_logging.sanitization import (
    sanitize_cognition_chain_event_fields,
)


def _fields() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "cognition_invocation_id": "invocation-1",
        "terminal_disposition": "complete",
        "chain_model_name": "chain-model",
        "sidecar_model_name": "sidecar-model",
        "step_count": 8,
        "repair_count": 0,
        "cold_start_count": 1,
        "prompt_chars_total": 100,
        "new_suffix_chars_total": 30,
        "prefix_share_ratio": 0.7,
        "max_estimated_prompt_tokens": 1000,
        "max_reserved_completion_tokens": 4096,
        "max_estimated_total_context_tokens": 5096,
        "active_total_ceiling_tokens": 50000,
        "extension_available": False,
        "extension_used": False,
        "reanchor_used": False,
        "session_disposition": "cold",
        "duration_ms": 120,
        "deadline_ms": 240000,
        "deadline_consumption_ratio": 0.5,
        "l1_stream_count": 1,
        "json_repair_call_count": 0,
        "action_auth_attempt_count": 0,
        "resolver_auth_attempt_count": 0,
        "sidecar_queue_wait_ms_total": 0,
        "sidecar_max_in_flight": 1,
        "l1_preempted_by_repair": False,
        "sidecar_cancellation_count": 0,
        "warning_codes": ["bounded_warning"],
    }


def test_cognition_chain_event_model_rejects_unknown_and_unbounded_fields():
    sanitized = sanitize_cognition_chain_event_fields(_fields())
    assert sanitized["run_id"] == "run-1"
    assert sanitized["warning_codes"] == ["bounded_warning"]

    with pytest.raises(ValueError, match="exact"):
        sanitize_cognition_chain_event_fields(
            {**_fields(), "raw_prompt": "forbidden"}
        )

    with pytest.raises(ValueError, match="non-negative integer"):
        sanitize_cognition_chain_event_fields(
            {**_fields(), "step_count": -1}
        )


@pytest.mark.asyncio
async def test_cognition_chain_recorder_is_keyword_only_bounded_and_best_effort(
    monkeypatch,
):
    written = AsyncMock()
    monkeypatch.setattr(recording.repository, "write_event", written)

    result = await recording.record_cognition_chain_event(
        run_id="run-1",
        cognition_invocation_id="invocation-1",
        terminal_disposition="complete",
        step_count=8,
        duration_ms=120,
    )

    assert result["status"] == "recorded"
    written.assert_awaited_once()
    event_doc = written.await_args.args[0]
    assert event_doc["event_family"] == "cognition_chain"
    assert event_doc["cognition_chain"]["step_count"] == 8
    assert event_doc["cognition_chain"]["warning_codes"] == []


def test_cognition_chain_event_sanitizer_removes_secret_and_raw_content():
    forbidden = _fields()
    forbidden["raw_output"] = "secret model text"
    with pytest.raises(ValueError):
        sanitize_cognition_chain_event_fields(forbidden)

    nested = {**_fields(), "payload": {"evidence_text": "secret"}}
    with pytest.raises(ValueError):
        sanitize_cognition_chain_event_fields(nested)


def test_cognition_chain_event_facade_exports_exact_recorder():
    assert event_logging.record_cognition_chain_event is (
        recording.record_cognition_chain_event
    )
