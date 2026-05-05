"""Deterministic tests for production reflection worker behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.reflection_cycle import worker as worker_module
from kazusa_ai_chatbot.reflection_cycle.models import (
    PromptBuildResult,
    REFLECTION_RUN_KIND_HOURLY,
    REFLECTION_STATUS_DRY_RUN,
    ReflectionInputSet,
    ReflectionLLMResult,
    ReflectionScopeInput,
)


@pytest.mark.asyncio
async def test_hourly_worker_disables_evaluation_fallback(monkeypatch) -> None:
    """Production hourly collection should idle instead of using fallback."""

    captured_kwargs = {}

    async def _collect_reflection_inputs(**kwargs):
        captured_kwargs.update(kwargs)
        input_set = _input_set([])
        return input_set

    monkeypatch.setattr(
        worker_module,
        "collect_reflection_inputs",
        _collect_reflection_inputs,
    )

    result = await worker_module.run_hourly_reflection_cycle(
        now=datetime(2026, 5, 5, tzinfo=timezone.utc),
        dry_run=True,
    )

    assert captured_kwargs["allow_fallback"] is False
    assert result.run_kind == REFLECTION_RUN_KIND_HOURLY
    assert result.skipped_count == 1
    assert result.defer_reason == "no monitored message-bearing hourly slots"


@pytest.mark.asyncio
async def test_hourly_worker_dry_run_persists_hourly_doc(monkeypatch) -> None:
    """Hourly dry-run should persist inspectable run docs without LLM calls."""

    persisted = []
    monkeypatch.setattr(
        worker_module,
        "collect_reflection_inputs",
        AsyncMock(return_value=_input_set([_channel_scope()])),
    )
    monkeypatch.setattr(
        worker_module.repository,
        "existing_run_ids",
        AsyncMock(return_value=set()),
    )

    async def _upsert(document):
        persisted.append(document)

    monkeypatch.setattr(worker_module.repository, "upsert_run", _upsert)
    run_hourly_llm = AsyncMock()
    monkeypatch.setattr(worker_module, "run_hourly_reflection_llm", run_hourly_llm)

    result = await worker_module.run_hourly_reflection_cycle(
        now=datetime(2026, 5, 5, tzinfo=timezone.utc),
        dry_run=True,
    )

    assert result.processed_count == 2
    assert result.skipped_count == 2
    assert [document["status"] for document in persisted] == [
        REFLECTION_STATUS_DRY_RUN,
        REFLECTION_STATUS_DRY_RUN,
    ]
    assert run_hourly_llm.await_count == 0


@pytest.mark.asyncio
async def test_worker_tick_defers_when_primary_interaction_is_busy() -> None:
    """A busy service probe should prevent reflection work from starting."""

    result = await worker_module._run_worker_tick(
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: True,
    )

    assert result[0].deferred is True
    assert result[0].defer_reason == "primary interaction busy"


@pytest.mark.asyncio
async def test_worker_tick_passes_busy_probe_to_promotion(monkeypatch) -> None:
    """Promotion scheduling should keep the service busy probe available."""

    captured = {}
    hourly_result = worker_module.ReflectionWorkerResult(
        run_kind="hourly_slot",
        dry_run=False,
    )
    daily_result = worker_module.ReflectionWorkerResult(
        run_kind="daily_channel",
        dry_run=False,
    )
    promotion_result = worker_module.ReflectionPromotionResult(
        run_kind="daily_global_promotion",
        dry_run=False,
    )

    async def _run_global_reflection_promotion(**kwargs):
        captured.update(kwargs)
        return promotion_result

    monkeypatch.setattr(
        worker_module,
        "_run_hourly_reflection_cycle",
        AsyncMock(return_value=hourly_result),
    )
    monkeypatch.setattr(
        worker_module,
        "_run_daily_channel_reflection_cycle",
        AsyncMock(return_value=daily_result),
    )
    monkeypatch.setattr(
        worker_module,
        "_run_global_reflection_promotion",
        _run_global_reflection_promotion,
    )
    monkeypatch.setattr(worker_module, "_local_time_is_after", lambda *_: True)

    def _busy_probe() -> bool:
        return False

    results = await worker_module._run_worker_tick(
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=_busy_probe,
    )

    assert results == [hourly_result, daily_result, promotion_result]
    assert captured["is_primary_interaction_busy"] is _busy_probe


@pytest.mark.asyncio
async def test_hourly_retry_builds_prompt_directly(monkeypatch) -> None:
    """Hourly retry should build the prompt without creating a skipped result."""

    prompt = _prompt_result()
    llm_result = ReflectionLLMResult(
        scope_ref="scope_channel",
        prompt=prompt,
        raw_output="{}",
        parsed_output={"topic_summary": "ok"},
        validation_warnings=[],
        llm_skipped=False,
    )
    build_prompt = MagicMock(return_value=prompt)
    build_skipped = MagicMock(
        side_effect=AssertionError("skipped result should not build retry prompt"),
    )
    monkeypatch.setattr(
        worker_module,
        "build_hourly_reflection_prompt",
        build_prompt,
    )
    monkeypatch.setattr(
        worker_module,
        "build_skipped_hourly_result",
        build_skipped,
    )
    run_hourly_llm = AsyncMock(return_value=llm_result)
    monkeypatch.setattr(
        worker_module,
        "run_hourly_reflection_llm",
        run_hourly_llm,
    )

    result, attempt_count = await worker_module._run_hourly_with_retry(
        _channel_scope(),
    )

    assert result is llm_result
    assert attempt_count == 1
    build_prompt.assert_called_once()
    build_skipped.assert_not_called()
    run_hourly_llm.assert_awaited_once_with(
        scope_ref="scope_channel",
        prompt=prompt,
    )


@pytest.mark.asyncio
async def test_daily_worker_counts_failed_hourly_as_terminal(monkeypatch) -> None:
    """Failed hourly docs should not permanently block daily synthesis."""

    persisted = []
    channel_scope = _channel_scope()
    monkeypatch.setattr(
        worker_module,
        "collect_reflection_inputs",
        AsyncMock(return_value=_input_set([channel_scope])),
    )
    monkeypatch.setattr(
        worker_module.repository,
        "existing_run_ids",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        worker_module.repository,
        "hourly_runs_for_channel_day",
        AsyncMock(return_value=[_hourly_doc(status="failed")]),
    )

    async def _upsert(document):
        persisted.append(document)

    monkeypatch.setattr(worker_module.repository, "upsert_run", _upsert)
    run_daily_llm = AsyncMock()
    monkeypatch.setattr(worker_module, "run_daily_synthesis_llm", run_daily_llm)

    result = await worker_module.run_daily_channel_reflection_cycle(
        character_local_date="2026-05-04",
        dry_run=True,
    )

    assert result.processed_count == 1
    assert persisted[0]["run_kind"] == "daily_channel"
    assert persisted[0]["source_reflection_run_ids"] == ["hourly-run-1"]
    assert run_daily_llm.await_count == 0


@pytest.mark.asyncio
async def test_daily_worker_records_partial_hourly_input_warning(
    monkeypatch,
) -> None:
    """Daily synthesis should reveal when non-terminal hourly docs were skipped."""

    persisted = []
    channel_scope = _channel_scope()
    monkeypatch.setattr(
        worker_module,
        "collect_reflection_inputs",
        AsyncMock(return_value=_input_set([channel_scope])),
    )
    monkeypatch.setattr(
        worker_module.repository,
        "existing_run_ids",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        worker_module.repository,
        "hourly_runs_for_channel_day",
        AsyncMock(
            return_value=[
                _hourly_doc(status="succeeded"),
                _hourly_doc(status="running", run_id="hourly-run-2"),
            ],
        ),
    )

    async def _upsert(document):
        persisted.append(document)

    monkeypatch.setattr(worker_module.repository, "upsert_run", _upsert)

    result = await worker_module.run_daily_channel_reflection_cycle(
        character_local_date="2026-05-04",
        dry_run=True,
    )

    assert result.processed_count == 1
    assert persisted[0]["source_reflection_run_ids"] == ["hourly-run-1"]
    assert (
        "partial_hourly_input terminal_count=1 total_count=2"
        in persisted[0]["validation_warnings"]
    )


def _input_set(scopes: list[ReflectionScopeInput]) -> ReflectionInputSet:
    """Build a worker input set fixture."""

    input_set = ReflectionInputSet(
        lookback_hours=24,
        requested_start="2026-05-04T00:00:00+00:00",
        requested_end="2026-05-05T00:00:00+00:00",
        effective_start="2026-05-04T00:00:00+00:00",
        effective_end="2026-05-05T00:00:00+00:00",
        fallback_used=False,
        fallback_reason="",
        selected_scopes=scopes,
        query_diagnostics={},
    )
    return input_set


def _channel_scope() -> ReflectionScopeInput:
    """Build a channel with two message-bearing hours."""

    messages = [
        {
            "role": "user",
            "body_text": "first hour",
            "timestamp": "2026-05-04T10:01:00+00:00",
        },
        {
            "role": "assistant",
            "body_text": "first reply",
            "timestamp": "2026-05-04T10:02:00+00:00",
        },
        {
            "role": "assistant",
            "body_text": "second reply",
            "timestamp": "2026-05-04T11:02:00+00:00",
        },
    ]
    scope = ReflectionScopeInput(
        scope_ref="scope_channel",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        assistant_message_count=2,
        user_message_count=1,
        total_message_count=3,
        first_timestamp="2026-05-04T10:01:00+00:00",
        last_timestamp="2026-05-04T11:02:00+00:00",
        messages=messages,
    )
    return scope


def _prompt_result() -> PromptBuildResult:
    """Build a prompt result fixture for hourly retry tests."""

    prompt = PromptBuildResult(
        system_prompt="system",
        human_payload={},
        human_prompt="human",
        prompt_chars=11,
        prompt_preview="system human",
        validation_warnings=[],
    )
    return prompt


def _hourly_doc(*, status: str, run_id: str = "hourly-run-1") -> dict:
    """Build an hourly run doc for daily synthesis tests."""

    doc = {
        "run_id": run_id,
        "run_kind": "hourly_slot",
        "status": status,
        "prompt_version": "readonly_reflection_v1",
        "attempt_count": 1,
        "scope": {
            "scope_ref": "scope_channel",
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
        },
        "hour_start": "2026-05-04T10:00:00+00:00",
        "hour_end": "2026-05-04T11:00:00+00:00",
        "character_local_date": "2026-05-04",
        "source_message_refs": [],
        "source_reflection_run_ids": [],
        "output": {
            "hourly_scope_ref": "scope_channel_20260504T10Z",
            "topic_summary": "A reflection summary.",
            "conversation_quality_feedback": ["Stay concrete."],
            "privacy_notes": ["No private details."],
            "confidence": "medium",
        },
        "promotion_decisions": [],
        "validation_warnings": [],
        "error": "",
    }
    return doc
