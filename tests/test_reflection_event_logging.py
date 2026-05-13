"""Tests for reflection worker event logging."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.reflection_cycle import worker as worker_module
from kazusa_ai_chatbot.reflection_cycle.models import (
    PromptBuildResult,
    ReflectionLLMResult,
    ReflectionScopeInput,
)


def _scope() -> ReflectionScopeInput:
    """Build a message-bearing reflection scope."""

    scope = ReflectionScopeInput(
        scope_ref="scope_channel",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        assistant_message_count=1,
        user_message_count=1,
        total_message_count=2,
        first_timestamp="2026-05-04T10:01:00+00:00",
        last_timestamp="2026-05-04T10:02:00+00:00",
        messages=[
            {
                "role": "user",
                "body_text": "hello",
                "timestamp": "2026-05-04T10:01:00+00:00",
            },
            {
                "role": "assistant",
                "body_text": "reply",
                "timestamp": "2026-05-04T10:02:00+00:00",
            },
        ],
    )
    return scope


def _prompt() -> PromptBuildResult:
    """Build a prompt metadata fixture."""

    prompt = PromptBuildResult(
        system_prompt="system text",
        human_payload={},
        human_prompt="human text",
        prompt_chars=20,
        prompt_preview="system text human text",
        validation_warnings=[],
    )
    return prompt


@pytest.mark.asyncio
async def test_worker_tick_records_deferred_event(monkeypatch) -> None:
    """Busy reflection ticks should record a deferred worker event."""

    record_worker_event = AsyncMock()
    monkeypatch.setattr(
        worker_module.event_logging,
        "record_worker_event",
        record_worker_event,
    )

    results = await worker_module._run_worker_tick(
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: True,
    )

    assert results[0].deferred is True
    record_worker_event.assert_awaited_once()
    kwargs = record_worker_event.await_args.kwargs
    assert kwargs["component"] == "reflection_cycle.worker"
    assert kwargs["run_kind"] == "reflection_tick"
    assert kwargs["status"] == "deferred"


@pytest.mark.asyncio
async def test_hourly_slot_records_reflection_run_upsert(monkeypatch) -> None:
    """Persisted reflection docs should record approved DB operation metadata."""

    record_database_operation_event = AsyncMock()
    monkeypatch.setattr(
        worker_module.event_logging,
        "record_database_operation_event",
        record_database_operation_event,
    )
    monkeypatch.setattr(
        worker_module.repository,
        "upsert_run",
        AsyncMock(),
    )

    run_doc = await worker_module._run_one_hourly_slot(
        hourly_scope=_scope(),
        dry_run=True,
    )

    assert run_doc["run_kind"] == "hourly_slot"
    record_database_operation_event.assert_awaited_once()
    kwargs = record_database_operation_event.await_args.kwargs
    assert kwargs["component"] == "reflection_cycle.worker"
    assert kwargs["collection"] == "character_reflection_runs"
    assert kwargs["document_ref"] == run_doc["run_id"]
    assert "body_text" not in str(kwargs)


@pytest.mark.asyncio
async def test_hourly_llm_stage_records_metadata_and_contract_warning(
    monkeypatch,
) -> None:
    """Reflection LLM telemetry should store sizes and warning codes only."""

    prompt = _prompt()
    llm_result = ReflectionLLMResult(
        scope_ref="scope_channel",
        prompt=prompt,
        raw_output='{"topic_summary": "ok"}',
        parsed_output={"topic_summary": "ok"},
        validation_warnings=["missing_privacy_notes"],
        llm_skipped=False,
    )
    record_llm_stage_event = AsyncMock()
    record_model_contract_event = AsyncMock()
    monkeypatch.setattr(
        worker_module,
        "build_hourly_reflection_prompt",
        lambda scope: prompt,
    )
    monkeypatch.setattr(
        worker_module,
        "run_hourly_reflection_llm",
        AsyncMock(return_value=llm_result),
    )
    monkeypatch.setattr(
        worker_module.event_logging,
        "record_llm_stage_event",
        record_llm_stage_event,
    )
    monkeypatch.setattr(
        worker_module.event_logging,
        "record_model_contract_event",
        record_model_contract_event,
    )

    result, attempt_count = await worker_module._run_hourly_with_retry(_scope())

    assert result is llm_result
    assert attempt_count == 1
    record_llm_stage_event.assert_awaited_once()
    stage_kwargs = record_llm_stage_event.await_args.kwargs
    assert stage_kwargs["stage_name"] == "hourly_reflection"
    assert stage_kwargs["prompt_chars"] == 20
    assert stage_kwargs["output_chars"] == len(llm_result.raw_output)
    assert stage_kwargs["parse_status"] == "warning"
    record_model_contract_event.assert_awaited_once()
    contract_kwargs = record_model_contract_event.await_args.kwargs
    assert contract_kwargs["invalid_fields"] == ["missing_privacy_notes"]
    assert llm_result.raw_output not in str(stage_kwargs)
    assert llm_result.raw_output not in str(contract_kwargs)
