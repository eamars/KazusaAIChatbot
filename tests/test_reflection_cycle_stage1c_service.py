"""FastAPI lifespan scheduling tests for the reflection worker."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.reflection_cycle import worker as worker_module


@pytest.fixture(autouse=True)
def _stub_service_event_logging(monkeypatch) -> None:
    """Keep lifespan scheduling tests off the event-log database."""

    recorder_names = (
        "record_process_event",
        "record_resource_health_event",
        "record_runtime_error_event",
    )
    for recorder_name in recorder_names:
        monkeypatch.setattr(
            service_module.event_logging,
            recorder_name,
            AsyncMock(),
        )


@pytest.mark.asyncio
async def test_lifespan_starts_reflection_worker_by_default(monkeypatch) -> None:
    """FastAPI lifespan should start reflection when the worker flag is enabled."""

    calls = await _run_lifespan(monkeypatch, enabled=True)

    assert calls["started"] == 1
    assert calls["stopped"] == 1
    assert callable(calls["busy_probe"])
    assert calls["busy_probe"]() is False
    assert callable(calls["reflection_adapter_registry_provider"])
    assert calls["reflection_adapter_registry_provider"]() is (
        service_module._adapter_registry
    )
    assert calls["reflection_phase_provider"] is calls["calendar_phase_provider"]
    assert callable(calls["reflection_state_refresh_callback"])


@pytest.mark.asyncio
async def test_lifespan_does_not_start_reflection_worker_when_explicitly_disabled(
    monkeypatch,
) -> None:
    """The positive worker flag should be the startup skip path."""

    calls = await _run_lifespan(monkeypatch, enabled=False)

    assert calls["started"] == 0
    assert calls["stopped"] == 0


@pytest.mark.asyncio
async def test_lifespan_stops_reflection_worker_on_shutdown(monkeypatch) -> None:
    """Shutdown should stop the owned reflection worker handle before exit."""

    calls = await _run_lifespan(monkeypatch, enabled=True)

    assert calls["started"] == 1
    assert calls["stopped"] == 1


@pytest.mark.asyncio
async def test_lifespan_starts_self_cognition_worker_only_when_enabled(
    monkeypatch,
) -> None:
    """Self-cognition worker startup should be explicitly gated by config."""

    disabled_calls = await _run_lifespan(
        monkeypatch,
        enabled=False,
        self_cognition_enabled=False,
    )
    enabled_calls = await _run_lifespan(
        monkeypatch,
        enabled=False,
        self_cognition_enabled=True,
    )

    assert disabled_calls["self_cognition_started"] == 0
    assert disabled_calls["self_cognition_stopped"] == 0
    assert enabled_calls["self_cognition_started"] == 1
    assert enabled_calls["self_cognition_stopped"] == 1
    assert callable(enabled_calls["self_cognition_busy_probe"])
    assert enabled_calls["self_cognition_busy_probe"]() is False
    assert callable(enabled_calls["self_cognition_affect_pause_probe"])


@pytest.mark.asyncio
async def test_lifespan_starts_calendar_scheduler_worker_by_default(
    monkeypatch,
) -> None:
    """Service startup should own the durable calendar worker lifecycle."""

    calls = await _run_lifespan(
        monkeypatch,
        enabled=False,
        calendar_enabled=True,
    )

    assert calls["calendar_started"] == 1
    assert calls["calendar_stopped"] == 1
    calendar_kwargs = calls["calendar_worker_kwargs"]
    assert isinstance(calendar_kwargs, dict)
    assert calendar_kwargs["poll_interval_seconds"] == 30
    assert calendar_kwargs["lease_duration_seconds"] == 300
    assert calendar_kwargs["claim_limit"] == 10
    assert calendar_kwargs["max_attempts"] == 3
    handler_registry = calendar_kwargs["handler_registry"]
    assert handler_registry.get("reflection_phase_slot") is not None


@pytest.mark.asyncio
async def test_lifespan_does_not_start_calendar_scheduler_when_disabled(
    monkeypatch,
) -> None:
    """The calendar worker should be gated by its own positive flag."""

    calls = await _run_lifespan(
        monkeypatch,
        enabled=False,
        calendar_enabled=False,
    )

    assert calls["calendar_started"] == 0
    assert calls["calendar_stopped"] == 0


@pytest.mark.asyncio
async def test_reflection_worker_defers_while_primary_interaction_is_busy() -> None:
    """Worker tick should defer when the service busy probe is true."""

    results = await worker_module._run_worker_tick(
        now=datetime(
            2026,
            5,
            4,
            18,
            0,
            tzinfo=timezone.utc,
        ),
        is_primary_interaction_busy=lambda: True,
    )

    assert results[0].deferred is True
    assert results[0].defer_reason == "primary interaction busy"


@pytest.mark.asyncio
async def test_reflection_probe_ignores_chat_queue_state(monkeypatch) -> None:
    """Reflection should not serialize itself behind active chat work."""

    class _Queue:
        def __init__(self, count: int) -> None:
            self.count = count

        def pending_count(self) -> int:
            return self.count

    monkeypatch.setattr(service_module, "_chat_input_queue", _Queue(1))
    monkeypatch.setattr(service_module, "_primary_interaction_active_count", 1)

    calls = await _run_lifespan(monkeypatch, enabled=True)
    busy_probe = calls["busy_probe"]

    assert callable(busy_probe)
    assert busy_probe() is False


@pytest.mark.asyncio
async def test_calendar_reflection_handler_passes_runtime_coordinator(
    monkeypatch,
) -> None:
    """Calendar reflection should use the generic runtime coordinator."""

    captured_kwargs: dict[str, object] = {}
    coordinator = object()

    async def _handle_calendar_run(run: dict[str, object], **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "status": "completed",
            "run_id": run["run_id"],
        }

    monkeypatch.setattr(
        service_module,
        "_pipeline_coordinator",
        coordinator,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "handle_reflection_phase_calendar_run",
        _handle_calendar_run,
    )

    result = await service_module._handle_calendar_reflection_phase_run(
        {"run_id": "phase-1"},
    )

    assert result == {
        "status": "completed",
        "run_id": "phase-1",
    }
    assert captured_kwargs["pipeline_coordinator"] is coordinator
    assert (
        captured_kwargs["is_primary_interaction_busy"]
        is service_module._reflection_cycle_primary_interaction_busy
    )


async def _run_lifespan(
    monkeypatch,
    *,
    enabled: bool,
    self_cognition_enabled: bool = False,
    calendar_enabled: bool = True,
    background_work_enabled: bool = False,
) -> dict[str, object]:
    """Run service lifespan with external dependencies patched."""

    calls: dict[str, object] = {
        "started": 0,
        "stopped": 0,
        "busy_probe": None,
        "self_cognition_started": 0,
        "self_cognition_stopped": 0,
        "self_cognition_busy_probe": None,
        "self_cognition_affect_pause_probe": None,
        "reflection_adapter_registry_provider": None,
        "reflection_phase_provider": None,
        "reflection_state_refresh_callback": None,
        "calendar_phase_provider": object(),
        "calendar_started": 0,
        "calendar_stopped": 0,
        "calendar_worker_kwargs": None,
        "background_work_started": 0,
        "background_work_stopped": 0,
    }
    handle = SimpleNamespace(
        task=asyncio.create_task(_sleep_forever()),
        stop_event=None,
    )
    self_cognition_handle: SimpleNamespace | None = None

    def _start_reflection_cycle_worker(
        *,
        is_primary_interaction_busy,
        adapter_registry_provider=None,
        phase_run_provider=None,
        character_state_refresh_callback=None,
    ):
        calls["started"] = int(calls["started"]) + 1
        calls["busy_probe"] = is_primary_interaction_busy
        calls["reflection_adapter_registry_provider"] = adapter_registry_provider
        calls["reflection_phase_provider"] = phase_run_provider
        calls["reflection_state_refresh_callback"] = (
            character_state_refresh_callback
        )
        return handle

    async def _stop_reflection_cycle_worker(_handle):
        calls["stopped"] = int(calls["stopped"]) + 1
        handle.task.cancel()
        try:
            await handle.task
        except asyncio.CancelledError:
            pass

    def _start_self_cognition_worker(**kwargs):
        nonlocal self_cognition_handle
        calls["self_cognition_started"] = (
            int(calls["self_cognition_started"]) + 1
        )
        calls["self_cognition_busy_probe"] = kwargs["is_primary_interaction_busy"]
        calls["self_cognition_affect_pause_probe"] = (
            kwargs["should_pause_for_affect_settling"]
        )
        self_cognition_handle = SimpleNamespace(
            task=asyncio.create_task(_sleep_forever()),
            stop_event=None,
        )
        return self_cognition_handle

    async def _stop_self_cognition_worker(_handle):
        calls["self_cognition_stopped"] = (
            int(calls["self_cognition_stopped"]) + 1
        )
        assert self_cognition_handle is not None
        self_cognition_handle.task.cancel()
        try:
            await self_cognition_handle.task
        except asyncio.CancelledError:
            pass

    calendar_handle = SimpleNamespace(
        task=asyncio.create_task(_sleep_forever()),
        stop_event=None,
    )

    def _start_calendar_scheduler_worker(**kwargs):
        calls["calendar_started"] = int(calls["calendar_started"]) + 1
        calls["calendar_worker_kwargs"] = kwargs
        return calendar_handle

    async def _stop_calendar_scheduler_worker(_handle):
        calls["calendar_stopped"] = int(calls["calendar_stopped"]) + 1
        calendar_handle.task.cancel()
        try:
            await calendar_handle.task
        except asyncio.CancelledError:
            pass

    def _calendar_phase_provider_factory():
        return calls["calendar_phase_provider"]

    background_work_handle = SimpleNamespace(
        task=asyncio.create_task(_sleep_forever()),
        stop_event=None,
    )

    def _start_background_work_runtime(**kwargs):
        calls["background_work_started"] = (
            int(calls["background_work_started"]) + 1
        )
        return background_work_handle

    async def _stop_background_work_runtime(_handle):
        calls["background_work_stopped"] = (
            int(calls["background_work_stopped"]) + 1
        )
        background_work_handle.task.cancel()
        try:
            await background_work_handle.task
        except asyncio.CancelledError:
            pass

    monkeypatch.setattr(
        service_module,
        "BACKGROUND_WORK_WORKER_ENABLED",
        background_work_enabled,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "start_background_work_runtime",
        _start_background_work_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "stop_background_work_runtime",
        _stop_background_work_runtime,
        raising=False,
    )
    monkeypatch.setattr(service_module, "REFLECTION_CYCLE_ENABLED", enabled)
    monkeypatch.setattr(
        service_module,
        "SELF_COGNITION_ENABLED",
        self_cognition_enabled,
    )
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_ENABLED",
        calendar_enabled,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_POLL_INTERVAL_SECONDS",
        30,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_CLAIM_LIMIT",
        10,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_LEASE_SECONDS",
        300,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "CALENDAR_SCHEDULER_MAX_ATTEMPTS",
        3,
        raising=False,
    )
    monkeypatch.setattr(service_module, "db_bootstrap", AsyncMock())
    monkeypatch.setattr(
        service_module,
        "_hydrate_media_descriptor_cache",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        service_module,
        "get_character_profile",
        AsyncMock(return_value={"name": "Character"}),
    )
    monkeypatch.setattr(
        service_module,
        "_build_graph",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(service_module.mcp_manager, "start", AsyncMock())
    monkeypatch.setattr(service_module.mcp_manager, "stop", AsyncMock())
    monkeypatch.setattr(
        service_module,
        "Pending" + "TaskIndex",
        _ForbiddenLegacyRuntime,
        raising=False,
    )
    legacy_scheduler = getattr(service_module, "scheduler", None)
    if legacy_scheduler is not None:
        monkeypatch.setattr(
            legacy_scheduler,
            "load" + "_pending_events",
            _forbidden_legacy_async,
            raising=False,
        )
        monkeypatch.setattr(
            legacy_scheduler,
            "shutdown",
            _forbidden_legacy_async,
            raising=False,
        )
        monkeypatch.setattr(
            legacy_scheduler,
            "configure_runtime",
            _forbidden_legacy_sync,
            raising=False,
        )
    monkeypatch.setattr(
        service_module,
        "render_llm_route_table",
        MagicMock(return_value="routes"),
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_chat_input_worker_started",
        MagicMock(),
    )
    monkeypatch.setattr(service_module, "_stop_chat_input_worker", AsyncMock())
    monkeypatch.setattr(service_module, "close_db", AsyncMock())
    monkeypatch.setattr(
        service_module,
        "start_reflection_cycle_worker",
        _start_reflection_cycle_worker,
    )
    monkeypatch.setattr(
        service_module,
        "stop_reflection_cycle_worker",
        _stop_reflection_cycle_worker,
    )
    monkeypatch.setattr(
        service_module,
        "start_self_cognition_worker",
        _start_self_cognition_worker,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "stop_self_cognition_worker",
        _stop_self_cognition_worker,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "start_calendar_scheduler_worker",
        _start_calendar_scheduler_worker,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "stop_calendar_scheduler_worker",
        _stop_calendar_scheduler_worker,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "CalendarReflectionPhaseRunProvider",
        _calendar_phase_provider_factory,
        raising=False,
    )

    async with service_module.lifespan(FastAPI()):
        pass

    return calls


async def _sleep_forever() -> None:
    """Provide a cancellable fake worker task."""

    await asyncio.Event().wait()


class _ForbiddenLegacyRuntime:
    """Fail if the retired delayed-task runtime is still constructed."""

    def __init__(self) -> None:
        raise AssertionError("retired scheduler runtime should not start")


async def _forbidden_legacy_async(*args, **kwargs) -> None:
    """Fail if an async legacy runtime hook is still called."""

    raise AssertionError("retired scheduler runtime should not be called")


def _forbidden_legacy_sync(*args, **kwargs) -> None:
    """Fail if a sync legacy runtime hook is still called."""

    raise AssertionError("retired scheduler runtime should not be called")
