"""Deterministic tests for production reflection worker behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.reflection_cycle import worker as worker_module
from kazusa_ai_chatbot.reflection_cycle.phase_scheduler import (
    REFLECTION_PHASE_GROUPS_PER_SLOT,
)
from kazusa_ai_chatbot.reflection_cycle.models import (
    PromptBuildResult,
    REFLECTION_RUN_KIND_HOURLY,
    REFLECTION_STATUS_DRY_RUN,
    ReflectionInputSet,
    ReflectionLLMResult,
    ReflectionScopeInput,
)


@pytest.fixture(autouse=True)
def _disable_event_log_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deterministic reflection worker tests off MongoDB."""

    monkeypatch.setattr(
        worker_module.event_logging,
        "record_worker_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        worker_module.event_logging,
        "record_database_operation_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        worker_module.event_logging,
        "record_llm_stage_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        worker_module.event_logging,
        "record_model_contract_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        worker_module.event_logging,
        "record_runtime_error_event",
        AsyncMock(),
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
async def test_group_review_defers_for_same_scope_foreground(
    monkeypatch,
) -> None:
    """Group review should not collect cases while its channel is foreground."""

    from kazusa_ai_chatbot.runtime_coordination import (
        PipelineCoordinator,
        PipelineScope,
    )

    coordinator = PipelineCoordinator()
    channel_scope = _channel_scope(
        scope_ref="scope_group",
        platform_channel_id="chan-1",
        channel_type="group",
    )
    scope = PipelineScope(
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
    )
    foreground = await coordinator.start_run(
        scope=scope,
        owner="service",
        precedence="foreground",
        run_kind="chat",
    )
    monkeypatch.setattr(
        worker_module,
        "get_character_profile",
        AsyncMock(side_effect=AssertionError("deferred review should not fetch")),
    )

    assert foreground.handle is not None
    async with foreground.handle:
        result = await worker_module._run_group_self_cognition_review_for_scope(
            now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
            channel_scope=channel_scope,
            is_primary_interaction_busy=lambda: False,
            pipeline_coordinator=coordinator,
        )

    assert result.deferred is True
    assert result.defer_reason == "same_scope_foreground_active"
    assert result.processed_count == 0


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
    style_result = worker_module.ReflectionWorkerResult(
        run_kind="daily_interaction_style_update",
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
        "_run_daily_interaction_style_update",
        AsyncMock(return_value=style_result),
    )
    monkeypatch.setattr(
        worker_module,
        "_run_global_reflection_promotion",
        _run_global_reflection_promotion,
    )
    monkeypatch.setattr(worker_module, "GLOBAL_CHARACTER_GROWTH_PASS_ENABLED", False)
    monkeypatch.setattr(worker_module, "_local_time_is_after", lambda *_: True)

    def _busy_probe() -> bool:
        return False

    results = await worker_module._run_worker_tick(
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=_busy_probe,
    )

    assert results == [hourly_result, daily_result, style_result, promotion_result]
    assert captured["is_primary_interaction_busy"] is _busy_probe


@pytest.mark.asyncio
async def test_worker_tick_executes_due_phase_intents_from_provider(
    monkeypatch,
) -> None:
    """Reflection ticks should consume calendar-shaped phase run intents."""

    now = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    provider = _FakePhaseRunProvider([
        _phase_intent(_channel_scope(), due_at=now),
    ])
    phase_result = worker_module.ReflectionWorkerResult(
        run_kind="reflection_phase_slot",
        dry_run=False,
        processed_count=1,
    )
    captured_intents = []

    async def _run_reflection_phase_intent(**kwargs):
        captured_intents.append(kwargs["intent"])
        return [phase_result]

    monkeypatch.setattr(
        worker_module,
        "_run_reflection_phase_intent",
        _run_reflection_phase_intent,
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "collect_reflection_inputs",
        AsyncMock(side_effect=AssertionError("tick must use phase provider")),
    )
    monkeypatch.setattr(worker_module, "_local_time_is_after", lambda *_: False)

    def _busy_probe() -> bool:
        return False

    results = await worker_module._run_worker_tick(
        now=now,
        is_primary_interaction_busy=_busy_probe,
        phase_run_provider=provider,
    )

    assert results == [phase_result]
    assert captured_intents == provider.intents
    assert provider.period_requests == [
        datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
    ]


@pytest.mark.asyncio
async def test_phase_intent_runs_hourly_and_group_for_selected_scope(
    monkeypatch,
) -> None:
    """One phase intent should execute handlers for its selected scope only."""

    now = datetime(2026, 5, 5, 18, 5, tzinfo=timezone.utc)
    selected_scope = _channel_scope(
        scope_ref="scope_selected",
        platform_channel_id="group-selected",
    )
    intent = _phase_intent(selected_scope, due_at=now)
    hourly_scopes = []
    group_scopes = []

    async def _collect_phase_scope_input(**kwargs):
        assert kwargs["intent"] is intent
        return selected_scope

    async def _run_hourly_reflection_for_scope(**kwargs):
        hourly_scopes.append(kwargs["channel_scope"])
        return worker_module.ReflectionWorkerResult(
            run_kind=REFLECTION_RUN_KIND_HOURLY,
            dry_run=False,
            processed_count=1,
            succeeded_count=1,
            run_ids=["hourly-selected"],
        )

    async def _run_group_self_cognition_review_for_scope(**kwargs):
        group_scopes.append(kwargs["channel_scope"])
        return worker_module.ReflectionWorkerResult(
            run_kind="group_self_cognition_review",
            dry_run=False,
            processed_count=1,
            succeeded_count=1,
        )

    monkeypatch.setattr(
        worker_module,
        "_collect_phase_scope_input",
        _collect_phase_scope_input,
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "_run_hourly_reflection_for_scope",
        _run_hourly_reflection_for_scope,
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "_run_group_self_cognition_review_for_scope",
        _run_group_self_cognition_review_for_scope,
        raising=False,
    )

    results = await worker_module._run_reflection_phase_intent(
        intent=intent,
        now=now,
        dry_run=False,
        is_primary_interaction_busy=lambda: False,
        adapter_registry_provider=None,
    )

    assert [scope.scope_ref for scope in hourly_scopes] == ["scope_selected"]
    assert [scope.scope_ref for scope in group_scopes] == ["scope_selected"]
    assert [result.run_kind for result in results] == [
        REFLECTION_RUN_KIND_HOURLY,
        "group_self_cognition_review",
    ]


@pytest.mark.asyncio
async def test_phase_intent_skips_group_review_for_private_scope(
    monkeypatch,
) -> None:
    """Private phase slots should not run group self-cognition."""

    now = datetime(2026, 5, 5, 18, 5, tzinfo=timezone.utc)
    selected_scope = _private_channel_scope()
    intent = _phase_intent(selected_scope, due_at=now)
    group_review = AsyncMock(side_effect=AssertionError("private group review"))

    monkeypatch.setattr(
        worker_module,
        "_collect_phase_scope_input",
        AsyncMock(return_value=selected_scope),
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "_run_hourly_reflection_for_scope",
        AsyncMock(return_value=worker_module.ReflectionWorkerResult(
            run_kind=REFLECTION_RUN_KIND_HOURLY,
            dry_run=False,
        )),
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "_run_group_self_cognition_review_for_scope",
        group_review,
        raising=False,
    )

    results = await worker_module._run_reflection_phase_intent(
        intent=intent,
        now=now,
        dry_run=False,
        is_primary_interaction_busy=lambda: False,
        adapter_registry_provider=None,
    )

    assert [result.run_kind for result in results] == [REFLECTION_RUN_KIND_HOURLY]
    group_review.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_review_passes_adapter_registry_provider_to_self_cognition(
    monkeypatch,
) -> None:
    """Selected group review should pass one case to the normal worker."""

    captured: dict[str, object] = {}
    ledger_rows = []
    now = datetime(2026, 5, 5, 18, 15, tzinfo=timezone.utc)
    adapter_provider = lambda: object()
    character_profile = {"name": "Character", "platform_bot_id": "bot-1"}
    channel_scope = _group_scope_with_window_minutes(
        [1],
        base_date="2026-05-05",
        hour=18,
    )

    async def _collect_group_review_cases(**kwargs):
        assert kwargs["now"] == now
        assert kwargs["character_profile"] == character_profile
        assert kwargs["max_cases"] == REFLECTION_PHASE_GROUPS_PER_SLOT
        captured["windows"] = kwargs["windows"]
        return [_group_review_case_from_window(kwargs["windows"][0])]

    async def _run_self_cognition_tick(**kwargs):
        captured.update(kwargs)
        selected_cases = await kwargs["collect_cases_func"](
            now=now,
            max_cases=kwargs["max_cases"],
        )
        captured["selected_cases"] = selected_cases
        return worker_module.self_cognition_worker.SelfCognitionWorkerResult(
            processed_count=1,
        )

    async def _find_group_review_window(source_id: str):
        del source_id
        return None

    async def _upsert_group_review_window(row: dict):
        ledger_rows.append(row)
        return row

    monkeypatch.setattr(
        worker_module,
        "get_character_profile",
        AsyncMock(return_value=character_profile),
    )
    monkeypatch.setattr(
        worker_module.self_cognition_sources,
        "collect_group_review_cases",
        _collect_group_review_cases,
    )
    monkeypatch.setattr(
        worker_module.self_cognition_worker,
        "run_self_cognition_worker_tick",
        _run_self_cognition_tick,
    )
    monkeypatch.setattr(
        worker_module,
        "is_self_cognition_sleep_period",
        lambda now: False,
    )
    monkeypatch.setattr(
        worker_module,
        "find_self_cognition_group_review_window",
        _find_group_review_window,
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "upsert_self_cognition_group_review_window",
        _upsert_group_review_window,
        raising=False,
    )

    result = await worker_module._run_group_self_cognition_review_for_scope(
        now=now,
        channel_scope=channel_scope,
        is_primary_interaction_busy=lambda: False,
        adapter_registry_provider=adapter_provider,
    )

    assert result.processed_count == 1
    assert captured["adapter_registry_provider"] is adapter_provider
    assert captured["collect_cases_func"] is not None
    assert captured["max_cases"] == REFLECTION_PHASE_GROUPS_PER_SLOT
    assert [
        window.source_id
        for window in captured["windows"]
    ] == [
        _source_id(
            "scope_group",
            "2026-05-05T18:00:00+00:00",
            "2026-05-05T18:15:00+00:00",
        )
    ]
    assert captured["selected_cases"][0]["case_id"].startswith(
        "group_activity_window:"
    )
    assert [row["status"] for row in ledger_rows] == ["reviewed"]
    assert ledger_rows[0]["skip_reason"] is None


@pytest.mark.asyncio
async def test_group_self_cognition_review_skips_cases_during_sleep(
    monkeypatch,
) -> None:
    """The reflection sidecar should not collect group cases while asleep."""

    monkeypatch.setattr(
        worker_module,
        "is_self_cognition_sleep_period",
        lambda now: True,
    )
    monkeypatch.setattr(
        worker_module,
        "get_character_profile",
        AsyncMock(side_effect=AssertionError("profile fetch should sleep")),
    )
    monkeypatch.setattr(
        worker_module.self_cognition_sources,
        "collect_group_chat_review_cases",
        AsyncMock(side_effect=AssertionError("group review should sleep")),
    )
    monkeypatch.setattr(
        worker_module.self_cognition_worker,
        "run_self_cognition_worker_tick",
        AsyncMock(side_effect=AssertionError("worker tick should sleep")),
    )

    result = await worker_module._run_group_self_cognition_review_for_scope(
        now=datetime(2026, 5, 12, 14, 30, tzinfo=timezone.utc),
        channel_scope=_channel_scope(),
        is_primary_interaction_busy=lambda: False,
    )

    assert result.skipped_count == 1
    assert result.defer_reason == "self-cognition sleep period"


@pytest.mark.asyncio
async def test_group_review_suppresses_reviewed_and_coalesces_older_windows(
    monkeypatch,
) -> None:
    """A group phase should review newest unreviewed and skip older backlog."""

    now = datetime(2026, 5, 12, 4, 45, tzinfo=timezone.utc)
    channel_scope = _group_scope_with_window_minutes([1, 18, 31])
    newest_source_id = _source_id(
        "scope_group",
        "2026-05-12T04:30:00+00:00",
        "2026-05-12T04:45:00+00:00",
    )
    selected_source_id = _source_id(
        "scope_group",
        "2026-05-12T04:15:00+00:00",
        "2026-05-12T04:30:00+00:00",
    )
    older_source_id = _source_id(
        "scope_group",
        "2026-05-12T04:00:00+00:00",
        "2026-05-12T04:15:00+00:00",
    )
    ledger_rows = []
    captured_windows = []

    async def _find_group_review_window(source_id: str):
        if source_id == newest_source_id:
            return {
                "source_id": source_id,
                "status": "reviewed",
                "case_id": f"group_activity_window:{source_id}",
            }
        return None

    async def _upsert_group_review_window(row: dict):
        ledger_rows.append(row)
        return row

    async def _collect_group_review_cases(**kwargs):
        captured_windows.extend(kwargs["windows"])
        return [_group_review_case_from_window(kwargs["windows"][0])]

    async def _run_self_cognition_tick(**kwargs):
        await kwargs["collect_cases_func"](
            now=now,
            max_cases=kwargs["max_cases"],
        )
        return worker_module.self_cognition_worker.SelfCognitionWorkerResult(
            processed_count=1,
        )

    monkeypatch.setattr(
        worker_module,
        "is_self_cognition_sleep_period",
        lambda checked_now: False,
    )
    monkeypatch.setattr(
        worker_module,
        "get_character_profile",
        AsyncMock(return_value={"name": "Character"}),
    )
    monkeypatch.setattr(
        worker_module.self_cognition_sources,
        "collect_group_review_cases",
        _collect_group_review_cases,
    )
    monkeypatch.setattr(
        worker_module.self_cognition_worker,
        "run_self_cognition_worker_tick",
        _run_self_cognition_tick,
    )
    monkeypatch.setattr(
        worker_module,
        "find_self_cognition_group_review_window",
        _find_group_review_window,
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "upsert_self_cognition_group_review_window",
        _upsert_group_review_window,
        raising=False,
    )

    result = await worker_module._run_group_self_cognition_review_for_scope(
        now=now,
        channel_scope=channel_scope,
        is_primary_interaction_busy=lambda: False,
    )

    assert result.processed_count == 1
    assert [window.source_id for window in captured_windows] == [selected_source_id]
    row_by_source = {row["source_id"]: row for row in ledger_rows}
    assert row_by_source[selected_source_id]["status"] == "reviewed"
    assert row_by_source[selected_source_id]["skip_reason"] is None
    assert row_by_source[older_source_id]["status"] == "coalesced_skipped"
    assert row_by_source[older_source_id]["skip_reason"] == (
        "coalesced_by_newer_group_phase_window"
    )
    assert newest_source_id not in row_by_source


@pytest.mark.asyncio
async def test_group_review_records_stale_skip_when_no_case_is_built(
    monkeypatch,
) -> None:
    """Selected windows that produce no case should be suppressed once."""

    now = datetime(2026, 5, 12, 18, 30, tzinfo=timezone.utc)
    channel_scope = _group_scope_with_window_minutes(
        [1],
        base_date="2026-05-12",
        hour=18,
    )
    ledger_rows = []

    async def _find_group_review_window(source_id: str):
        del source_id
        return None

    async def _upsert_group_review_window(row: dict):
        ledger_rows.append(row)
        return row

    monkeypatch.setattr(
        worker_module,
        "is_self_cognition_sleep_period",
        lambda checked_now: False,
    )
    monkeypatch.setattr(
        worker_module,
        "get_character_profile",
        AsyncMock(return_value={"name": "Character"}),
    )
    monkeypatch.setattr(
        worker_module.self_cognition_sources,
        "collect_group_review_cases",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        worker_module.self_cognition_worker,
        "run_self_cognition_worker_tick",
        AsyncMock(side_effect=AssertionError("no case should not run worker")),
    )
    monkeypatch.setattr(
        worker_module,
        "find_self_cognition_group_review_window",
        _find_group_review_window,
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "upsert_self_cognition_group_review_window",
        _upsert_group_review_window,
        raising=False,
    )

    result = await worker_module._run_group_self_cognition_review_for_scope(
        now=now,
        channel_scope=channel_scope,
        is_primary_interaction_busy=lambda: False,
    )

    assert result.skipped_count == 1
    assert result.defer_reason == "no group review cases"
    assert [row["status"] for row in ledger_rows] == ["stale_skipped"]
    assert ledger_rows[0]["skip_reason"] == "no_group_review_case_built"


@pytest.mark.asyncio
async def test_group_review_records_target_binding_failed_terminal_row(
    monkeypatch,
) -> None:
    """Target-binding failures should be terminal without pretending review ran."""

    now = datetime(2026, 5, 12, 18, 30, tzinfo=timezone.utc)
    channel_scope = _group_scope_with_window_minutes(
        [1],
        base_date="2026-05-12",
        hour=18,
    )
    ledger_rows = []

    async def _find_group_review_window(source_id: str):
        del source_id
        return None

    async def _upsert_group_review_window(row: dict):
        ledger_rows.append(row)
        return row

    async def _collect_group_review_cases(**kwargs):
        case = _group_review_case_from_window(kwargs["windows"][0])
        case.pop("delivery_target", None)
        case["target_binding_status"] = "failed"
        case["target_binding_failure"] = {
            "status": "target_binding_failed",
            "reason": "missing_delivery_target",
        }
        return [case]

    async def _run_self_cognition_tick(**kwargs):
        await kwargs["collect_cases_func"](
            now=now,
            max_cases=kwargs["max_cases"],
        )
        return worker_module.self_cognition_worker.SelfCognitionWorkerResult(
            skipped_count=1,
        )

    monkeypatch.setattr(
        worker_module,
        "is_self_cognition_sleep_period",
        lambda checked_now: False,
    )
    monkeypatch.setattr(
        worker_module,
        "get_character_profile",
        AsyncMock(return_value={"name": "Character"}),
    )
    monkeypatch.setattr(
        worker_module.self_cognition_sources,
        "collect_group_review_cases",
        _collect_group_review_cases,
    )
    monkeypatch.setattr(
        worker_module.self_cognition_worker,
        "run_self_cognition_worker_tick",
        _run_self_cognition_tick,
    )
    monkeypatch.setattr(
        worker_module,
        "find_self_cognition_group_review_window",
        _find_group_review_window,
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "upsert_self_cognition_group_review_window",
        _upsert_group_review_window,
        raising=False,
    )

    result = await worker_module._run_group_self_cognition_review_for_scope(
        now=now,
        channel_scope=channel_scope,
        is_primary_interaction_busy=lambda: False,
    )

    assert result.skipped_count == 1
    assert [row["status"] for row in ledger_rows] == ["target_binding_failed"]
    assert ledger_rows[0]["skip_reason"] == "missing_delivery_target"


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
async def test_worker_tick_runs_period_maintenance_once_for_many_phase_intents(
    monkeypatch,
) -> None:
    """Daily and promotion maintenance should not run once per phase slot."""

    now = datetime(2026, 5, 5, 18, 10, tzinfo=timezone.utc)
    provider = _FakePhaseRunProvider([
        _phase_intent(_channel_scope(scope_ref="scope_a"), due_at=now),
        _phase_intent(_channel_scope(scope_ref="scope_b"), due_at=now, slot_index=1),
    ])
    daily_result = worker_module.ReflectionWorkerResult(
        run_kind="daily_channel",
        dry_run=False,
    )
    style_result = worker_module.ReflectionWorkerResult(
        run_kind="daily_interaction_style_update",
        dry_run=False,
    )
    promotion_result = worker_module.ReflectionPromotionResult(
        run_kind="daily_global_promotion",
        dry_run=False,
    )
    phase_call_count = 0

    async def _run_reflection_phase_intent(**kwargs):
        nonlocal phase_call_count
        phase_call_count += 1
        return [
            worker_module.ReflectionWorkerResult(
                run_kind="reflection_phase_slot",
                dry_run=False,
                processed_count=1,
            )
        ]

    monkeypatch.setattr(
        worker_module,
        "_run_reflection_phase_intent",
        _run_reflection_phase_intent,
        raising=False,
    )
    daily = AsyncMock(return_value=daily_result)
    style = AsyncMock(return_value=style_result)
    promotion = AsyncMock(return_value=promotion_result)
    monkeypatch.setattr(worker_module, "_run_daily_channel_reflection_cycle", daily)
    monkeypatch.setattr(worker_module, "_run_daily_interaction_style_update", style)
    monkeypatch.setattr(worker_module, "_run_global_reflection_promotion", promotion)
    monkeypatch.setattr(worker_module, "GLOBAL_CHARACTER_GROWTH_PASS_ENABLED", False)
    monkeypatch.setattr(worker_module, "_local_time_is_after", lambda *_: True)

    results = await worker_module._run_worker_tick(
        now=now,
        is_primary_interaction_busy=lambda: False,
        phase_run_provider=provider,
    )

    assert phase_call_count == 2
    assert results.count(daily_result) == 1
    assert results.count(style_result) == 1
    assert results.count(promotion_result) == 1
    daily.assert_awaited_once()
    style.assert_awaited_once()
    promotion.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_tick_runs_due_daily_affect_settling_once(
    monkeypatch,
) -> None:
    """The wake-window affect pass should run once per worker period."""

    now = datetime(2026, 5, 5, 23, 40, tzinfo=timezone.utc)
    provider = _FakePhaseRunProvider([
        _phase_intent(_channel_scope(scope_ref="scope_a"), due_at=now),
        _phase_intent(_channel_scope(scope_ref="scope_b"), due_at=now),
    ])
    daily_result = worker_module.ReflectionWorkerResult(
        run_kind="daily_channel",
        dry_run=False,
    )
    style_result = worker_module.ReflectionWorkerResult(
        run_kind="daily_interaction_style_update",
        dry_run=False,
    )
    promotion_result = worker_module.ReflectionPromotionResult(
        run_kind="daily_global_promotion",
        dry_run=False,
    )
    affect_result = worker_module.ReflectionWorkerResult(
        run_kind="daily_affect_settling",
        dry_run=False,
        succeeded_count=1,
    )
    refresh = AsyncMock()

    async def _run_reflection_phase_intent(**kwargs):
        del kwargs
        return [
            worker_module.ReflectionWorkerResult(
                run_kind="reflection_phase_slot",
                dry_run=False,
                processed_count=1,
            )
        ]

    monkeypatch.setattr(
        worker_module,
        "_run_reflection_phase_intent",
        _run_reflection_phase_intent,
        raising=False,
    )
    monkeypatch.setattr(
        worker_module,
        "_run_daily_channel_reflection_cycle",
        AsyncMock(return_value=daily_result),
    )
    monkeypatch.setattr(
        worker_module,
        "_run_daily_interaction_style_update",
        AsyncMock(return_value=style_result),
    )
    monkeypatch.setattr(
        worker_module,
        "_run_global_reflection_promotion",
        AsyncMock(return_value=promotion_result),
    )
    affect = AsyncMock(return_value=affect_result)
    monkeypatch.setattr(
        worker_module,
        "_run_daily_affect_settling",
        affect,
        raising=False,
    )
    monkeypatch.setattr(worker_module, "GLOBAL_CHARACTER_GROWTH_PASS_ENABLED", False)
    monkeypatch.setattr(worker_module, "_local_time_is_after", lambda *_: True)
    monkeypatch.setattr(
        worker_module,
        "settling_local_date_for_due_affect_settling",
        lambda *_: "2026-05-06",
    )

    executed_maintenance: set[tuple[datetime, str, str]] = set()
    results = await worker_module._run_worker_tick(
        now=now,
        is_primary_interaction_busy=lambda: False,
        phase_run_provider=provider,
        executed_period_maintenance_keys=executed_maintenance,
        character_state_refresh_callback=refresh,
    )

    assert results.count(affect_result) == 1
    affect.assert_awaited_once_with(
        settling_local_date="2026-05-06",
        dry_run=False,
        enable_character_state_write=True,
        character_state_refresh_callback=refresh,
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


@pytest.mark.asyncio
async def test_daily_worker_defers_when_expected_hourly_docs_are_missing(
    monkeypatch,
) -> None:
    """Daily synthesis must not silently summarize partial phase output."""

    persisted = []
    channel_scope = _channel_scope()
    expected = worker_module.ExpectedDailyChannelHourlyRuns(
        channel_scope=channel_scope,
        expected_run_ids=["expected-hour-1", "expected-hour-2"],
    )
    provider = _FakePhaseRunProvider([], expected_daily=[expected])

    monkeypatch.setattr(
        worker_module,
        "collect_reflection_inputs",
        AsyncMock(side_effect=AssertionError("daily must use phase readiness")),
    )
    monkeypatch.setattr(
        worker_module.repository,
        "existing_run_ids",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        worker_module.repository,
        "hourly_runs_for_channel_day",
        AsyncMock(return_value=[_hourly_doc(
            status="succeeded",
            run_id="expected-hour-1",
        )]),
    )

    async def _upsert(document):
        persisted.append(document)

    monkeypatch.setattr(worker_module.repository, "upsert_run", _upsert)

    result = await worker_module._run_daily_channel_reflection_cycle(
        character_local_date="2026-05-04",
        dry_run=True,
        is_primary_interaction_busy=lambda: False,
        phase_run_provider=provider,
    )

    assert result.processed_count == 0
    assert result.skipped_count == 1
    assert result.defer_reason == "missing expected hourly reflection runs"
    assert result.validation_warnings == [
        "missing_expected_hourly_runs scope_ref=scope_channel missing_count=1"
    ]
    assert persisted == []


@pytest.mark.asyncio
async def test_daily_readiness_comes_from_phase_provider_not_runtime_snapshot(
    monkeypatch,
) -> None:
    """Daily expected IDs should derive from prior phase materialization."""

    persisted = []
    previous_day_scope = _channel_scope(
        scope_ref="scope_previous_day",
        platform_channel_id="group-previous-day",
    )
    runtime_scope = _channel_scope(
        scope_ref="scope_runtime",
        platform_channel_id="group-runtime",
    )
    expected = worker_module.ExpectedDailyChannelHourlyRuns(
        channel_scope=previous_day_scope,
        expected_run_ids=["hourly-run-1"],
    )
    provider = _FakePhaseRunProvider([], expected_daily=[expected])

    monkeypatch.setattr(
        worker_module,
        "collect_reflection_inputs",
        AsyncMock(return_value=_input_set([runtime_scope])),
    )
    monkeypatch.setattr(
        worker_module.repository,
        "existing_run_ids",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        worker_module.repository,
        "hourly_runs_for_channel_day",
        AsyncMock(return_value=[_hourly_doc(status="succeeded")]),
    )

    async def _upsert(document):
        persisted.append(document)

    monkeypatch.setattr(worker_module.repository, "upsert_run", _upsert)

    result = await worker_module._run_daily_channel_reflection_cycle(
        character_local_date="2026-05-04",
        dry_run=True,
        is_primary_interaction_busy=lambda: False,
        phase_run_provider=provider,
    )

    assert result.processed_count == 1
    assert persisted[0]["scope"]["scope_ref"] == "scope_previous_day"


@pytest.mark.asyncio
async def test_daily_readiness_uses_only_closed_hours_for_phase_due_at(
    monkeypatch,
) -> None:
    """Expected daily hourly rows should not include open phase-bucket hours."""

    provider = worker_module.LocalReflectionPhaseRunProvider()
    due_at = datetime(2026, 5, 4, 11, 15, tzinfo=timezone.utc)
    channel_scope = _channel_scope()
    intent = _phase_intent(channel_scope, due_at=due_at)
    period_requests = []

    async def _period_run_intents(*, period_start_utc):
        period_requests.append(period_start_utc)
        if period_start_utc == due_at:
            return [intent]
        return []

    monkeypatch.setattr(provider, "period_run_intents", _period_run_intents)
    monkeypatch.setattr(
        worker_module,
        "_collect_phase_scope_input",
        AsyncMock(return_value=channel_scope),
    )

    expected_rows = await provider.expected_hourly_runs_for_character_local_date(
        character_local_date="2026-05-04",
    )

    expected_run_id = worker_module.repository.hourly_run_id(
        scope_ref=worker_module.repository.channel_scope_ref_for_hourly(
            channel_scope.scope_ref,
        ),
        hour_start=datetime(
            2026,
            5,
            4,
            10,
            tzinfo=timezone.utc,
        ).isoformat(),
    )
    assert len(expected_rows) == 1
    assert expected_rows[0].expected_run_ids == [expected_run_id]
    assert due_at in period_requests


def test_next_phase_wait_targets_next_offset_not_full_period() -> None:
    """Worker wait math should wake for the next due offset in the period."""

    period_start = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    first_intent = _phase_intent(
        _channel_scope(scope_ref="scope_a"),
        due_at=period_start,
    )
    second_intent = _phase_intent(
        _channel_scope(scope_ref="scope_b"),
        due_at=period_start + timedelta(minutes=5),
        slot_index=1,
    )

    timeout = worker_module._next_reflection_phase_wait_seconds(
        now=period_start,
        period_start_utc=period_start,
        intents=[first_intent, second_intent],
        executed_run_ids={first_intent["run_id"]},
    )

    assert timeout == 300


@pytest.mark.asyncio
async def test_worker_loop_waits_on_stop_event_until_next_phase_due(
    monkeypatch,
) -> None:
    """The loop should be stoppable while waiting for the next offset."""

    period_start = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    first_intent = _phase_intent(
        _channel_scope(scope_ref="scope_a"),
        due_at=period_start,
    )
    second_intent = _phase_intent(
        _channel_scope(scope_ref="scope_b"),
        due_at=period_start + timedelta(minutes=5),
        slot_index=1,
    )
    provider = _FakePhaseRunProvider([first_intent, second_intent])
    stop_event = worker_module.asyncio.Event()
    wait_timeouts = []

    async def _run_worker_tick(**kwargs):
        kwargs["executed_phase_run_ids"].add(first_intent["run_id"])
        return []

    async def _wait_for(awaitable, *, timeout):
        wait_timeouts.append(timeout)
        awaitable.close()
        stop_event.set()
        return None

    monkeypatch.setattr(worker_module, "storage_utc_now", lambda: period_start)
    monkeypatch.setattr(worker_module, "_run_worker_tick", _run_worker_tick)
    monkeypatch.setattr(worker_module.asyncio, "wait_for", _wait_for)

    await worker_module._reflection_worker_loop(
        stop_event=stop_event,
        is_primary_interaction_busy=lambda: False,
        adapter_registry_provider=None,
        phase_run_provider=provider,
    )

    assert wait_timeouts == [300]


@pytest.mark.asyncio
async def test_worker_loop_uses_post_tick_time_for_next_phase_wait(
    monkeypatch,
) -> None:
    """Long phase work should not make the next due offset wait again."""

    period_start = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    first_intent = _phase_intent(
        _channel_scope(scope_ref="scope_a"),
        due_at=period_start,
    )
    second_intent = _phase_intent(
        _channel_scope(scope_ref="scope_b"),
        due_at=period_start + timedelta(minutes=5),
        slot_index=1,
    )
    provider = _FakePhaseRunProvider([first_intent, second_intent])
    stop_event = worker_module.asyncio.Event()
    wait_timeouts = []
    now_values = [
        period_start,
        period_start + timedelta(minutes=6),
    ]

    async def _run_worker_tick(**kwargs):
        kwargs["executed_phase_run_ids"].add(first_intent["run_id"])
        return []

    def _storage_utc_now():
        if now_values:
            return now_values.pop(0)
        return period_start + timedelta(minutes=6)

    async def _wait_for(awaitable, *, timeout):
        wait_timeouts.append(timeout)
        awaitable.close()
        stop_event.set()
        return None

    monkeypatch.setattr(worker_module, "storage_utc_now", _storage_utc_now)
    monkeypatch.setattr(worker_module, "_run_worker_tick", _run_worker_tick)
    monkeypatch.setattr(worker_module.asyncio, "wait_for", _wait_for)

    await worker_module._reflection_worker_loop(
        stop_event=stop_event,
        is_primary_interaction_busy=lambda: False,
        adapter_registry_provider=None,
        phase_run_provider=provider,
    )

    assert wait_timeouts == [0]


@pytest.mark.asyncio
async def test_worker_loop_runs_period_maintenance_once_across_phase_wakes(
    monkeypatch,
) -> None:
    """Maintenance gates should not rerun for each intra-period phase wake."""

    period_start = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    first_intent = _phase_intent(
        _channel_scope(scope_ref="scope_a"),
        due_at=period_start,
    )
    second_intent = _phase_intent(
        _channel_scope(scope_ref="scope_b"),
        due_at=period_start + timedelta(minutes=5),
        slot_index=1,
    )
    provider = _FakePhaseRunProvider([first_intent, second_intent])
    stop_event = worker_module.asyncio.Event()
    wait_timeouts = []
    now_values = [
        period_start,
        period_start,
        period_start + timedelta(minutes=5),
        period_start + timedelta(minutes=5),
    ]

    async def _run_reflection_phase_intent(**kwargs):
        del kwargs
        return [
            worker_module.ReflectionWorkerResult(
                run_kind="reflection_phase_slot",
                dry_run=False,
                processed_count=1,
            )
        ]

    def _storage_utc_now():
        if now_values:
            return now_values.pop(0)
        return period_start + timedelta(minutes=5)

    async def _wait_for(awaitable, *, timeout):
        wait_timeouts.append(timeout)
        awaitable.close()
        if len(wait_timeouts) == 1:
            raise TimeoutError
        stop_event.set()
        return None

    daily = AsyncMock(return_value=worker_module.ReflectionWorkerResult(
        run_kind="daily_channel",
        dry_run=False,
    ))
    style = AsyncMock(return_value=worker_module.ReflectionWorkerResult(
        run_kind="daily_interaction_style_update",
        dry_run=False,
    ))
    promotion = AsyncMock(return_value=worker_module.ReflectionPromotionResult(
        run_kind="daily_global_promotion",
        dry_run=False,
    ))
    monkeypatch.setattr(worker_module, "storage_utc_now", _storage_utc_now)
    monkeypatch.setattr(
        worker_module,
        "_run_reflection_phase_intent",
        _run_reflection_phase_intent,
        raising=False,
    )
    monkeypatch.setattr(worker_module, "_run_daily_channel_reflection_cycle", daily)
    monkeypatch.setattr(worker_module, "_run_daily_interaction_style_update", style)
    monkeypatch.setattr(worker_module, "_run_global_reflection_promotion", promotion)
    monkeypatch.setattr(worker_module, "GLOBAL_CHARACTER_GROWTH_PASS_ENABLED", False)
    monkeypatch.setattr(worker_module, "_local_time_is_after", lambda *_: True)
    monkeypatch.setattr(worker_module.asyncio, "wait_for", _wait_for)

    await worker_module._reflection_worker_loop(
        stop_event=stop_event,
        is_primary_interaction_busy=lambda: False,
        adapter_registry_provider=None,
        phase_run_provider=provider,
    )

    assert wait_timeouts == [300, 600]
    daily.assert_awaited_once()
    style.assert_awaited_once()
    promotion.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_loop_recovers_when_wait_planning_provider_fails(
    monkeypatch,
) -> None:
    """Provider failures during wait planning should not terminate the loop."""

    period_start = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    provider = _FailingAfterTickPhaseProvider()
    stop_event = worker_module.asyncio.Event()
    wait_timeouts = []

    async def _run_worker_tick(**kwargs):
        del kwargs
        return []

    async def _wait_for(awaitable, *, timeout):
        wait_timeouts.append(timeout)
        awaitable.close()
        stop_event.set()
        return None

    monkeypatch.setattr(worker_module, "storage_utc_now", lambda: period_start)
    monkeypatch.setattr(worker_module, "_run_worker_tick", _run_worker_tick)
    monkeypatch.setattr(worker_module.asyncio, "wait_for", _wait_for)

    await worker_module._reflection_worker_loop(
        stop_event=stop_event,
        is_primary_interaction_busy=lambda: False,
        adapter_registry_provider=None,
        phase_run_provider=provider,
    )

    assert wait_timeouts == [900]
    assert (
        worker_module.event_logging.record_runtime_error_event.await_args.kwargs[
            "stack_fingerprint"
        ]
        == "reflection_phase_wait_planning"
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


class _FakePhaseRunProvider:
    """Small fake for the reflection phase provider contract."""

    def __init__(
        self,
        intents: list[dict],
        *,
        expected_daily: list[object] | None = None,
    ) -> None:
        self.intents = intents
        self.expected_daily = expected_daily or []
        self.period_requests: list[datetime] = []
        self.daily_requests: list[str] = []

    async def period_run_intents(
        self,
        *,
        period_start_utc: datetime,
    ) -> list[dict]:
        """Return configured run intents and capture the requested period."""

        self.period_requests.append(period_start_utc)
        return list(self.intents)

    async def expected_hourly_runs_for_character_local_date(
        self,
        *,
        character_local_date: str,
    ) -> list[object]:
        """Return configured daily-readiness rows."""

        self.daily_requests.append(character_local_date)
        return list(self.expected_daily)


class _FailingAfterTickPhaseProvider:
    """Fake provider that fails only when the loop plans its next wait."""

    async def period_run_intents(
        self,
        *,
        period_start_utc: datetime,
    ) -> list[dict]:
        """Raise to prove loop wait planning is recovered."""

        del period_start_utc
        raise RuntimeError("phase provider unavailable")


def _phase_intent(
    scope: ReflectionScopeInput,
    *,
    due_at: datetime,
    slot_index: int = 0,
) -> dict:
    """Build a calendar-shaped phase intent for worker tests."""

    period_start = due_at.replace(minute=(due_at.minute // 15) * 15)
    period_start = period_start.replace(second=0, microsecond=0)
    run_id = f"phase-run-{scope.scope_ref}-{slot_index}"
    intent = {
        "run_id": run_id,
        "trigger_kind": "reflection_phase_slot",
        "due_at": due_at.isoformat(),
        "period_start_utc": period_start.isoformat(),
        "slot_index": slot_index,
        "offset_seconds": int((due_at - period_start).total_seconds()),
        "source_scope": {
            "scope_ref": scope.scope_ref,
            "platform": scope.platform,
            "platform_channel_id": scope.platform_channel_id,
            "channel_type": scope.channel_type,
        },
        "payload": {
            "phase_period_seconds": 900,
            "max_slots_per_period": 3,
            "prompt_version": "readonly_reflection_v1",
            "allowed_actions": [
                "reflection_hourly_slot",
                "group_self_cognition_review",
            ],
        },
        "idempotency_key": run_id,
    }
    return intent


def _channel_scope(
    *,
    scope_ref: str = "scope_channel",
    platform_channel_id: str = "chan-1",
    channel_type: str = "group",
) -> ReflectionScopeInput:
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
        scope_ref=scope_ref,
        platform="qq",
        platform_channel_id=platform_channel_id,
        channel_type=channel_type,
        assistant_message_count=2,
        user_message_count=1,
        total_message_count=3,
        first_timestamp="2026-05-04T10:01:00+00:00",
        last_timestamp="2026-05-04T11:02:00+00:00",
        messages=messages,
    )
    return scope


def _private_channel_scope() -> ReflectionScopeInput:
    """Build a private channel scope for phase-handler tests."""

    scope = _channel_scope(
        scope_ref="scope_private",
        platform_channel_id="dm-1",
        channel_type="private",
    )
    return scope


def _group_scope_with_window_minutes(
    minutes: list[int],
    *,
    base_date: str = "2026-05-12",
    hour: int = 4,
) -> ReflectionScopeInput:
    """Build a group scope with one message in each requested minute."""

    messages = [
        {
            "role": "user",
            "body_text": f"group window {minute}",
            "timestamp": f"{base_date}T{hour:02d}:{minute:02d}:00+00:00",
            "display_name": "user",
            "platform_message_id": f"msg-{minute}",
            "global_user_id": f"user-{minute}",
            "platform_user_id": f"qq-user-{minute}",
            "addressed_to_global_user_ids": [],
            "mentions": [],
        }
        for minute in minutes
    ]
    scope = ReflectionScopeInput(
        scope_ref="scope_group",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        assistant_message_count=0,
        user_message_count=len(messages),
        total_message_count=len(messages),
        first_timestamp=str(messages[0]["timestamp"]),
        last_timestamp=str(messages[-1]["timestamp"]),
        messages=messages,
    )
    return scope


def _group_review_case_from_window(window: object) -> dict[str, object]:
    """Build a self-cognition case from a selected activity window fixture."""

    source_id = str(window.source_id)
    case = {
        "case_name": "group_chat_review",
        "case_id": f"group_activity_window:{source_id}",
        "source_refs": [
            {
                "source_kind": "reflection_activity_window",
                "source_id": source_id,
                "due_at": None,
                "summary": "group window",
            }
        ],
        "target_scope": {
            "platform": window.platform,
            "platform_channel_id": window.platform_channel_id,
            "channel_type": "group",
            "user_id": None,
        },
        "delivery_target": {
            "platform": window.platform,
            "platform_channel_id": window.platform_channel_id,
            "channel_type": "group",
        },
    }
    return case


def _source_id(scope_ref: str, window_start: str, window_end: str) -> str:
    """Build the activity-window source id used by reviewed-window ledger."""

    source_id = f"{scope_ref}:{window_start}:{window_end}"
    return source_id


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
