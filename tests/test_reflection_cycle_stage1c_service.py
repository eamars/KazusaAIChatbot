"""FastAPI lifespan scheduling tests for the reflection worker."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.reflection_cycle import worker as worker_module


@pytest.mark.asyncio
async def test_lifespan_starts_reflection_worker_by_default(monkeypatch) -> None:
    """FastAPI lifespan should start reflection unless explicitly disabled."""

    calls = await _run_lifespan(monkeypatch, disabled=False)

    assert calls["started"] == 1
    assert calls["stopped"] == 1
    assert callable(calls["busy_probe"])


@pytest.mark.asyncio
async def test_lifespan_does_not_start_reflection_worker_when_explicitly_disabled(
    monkeypatch,
) -> None:
    """The explicit disable flag should be the only startup skip path."""

    calls = await _run_lifespan(monkeypatch, disabled=True)

    assert calls["started"] == 0
    assert calls["stopped"] == 0


@pytest.mark.asyncio
async def test_lifespan_stops_reflection_worker_on_shutdown(monkeypatch) -> None:
    """Shutdown should stop the owned reflection worker handle before exit."""

    calls = await _run_lifespan(monkeypatch, disabled=False)

    assert calls["started"] == 1
    assert calls["stopped"] == 1


@pytest.mark.asyncio
async def test_reflection_worker_defers_while_primary_interaction_is_busy() -> None:
    """Worker tick should defer when the service busy probe is true."""

    results = await worker_module._run_worker_tick(
        now=worker_module.datetime(2026, 5, 4, 18, 0, tzinfo=worker_module.timezone.utc),
        is_primary_interaction_busy=lambda: True,
    )

    assert results[0].deferred is True
    assert results[0].defer_reason == "primary interaction busy"


def test_primary_interaction_busy_uses_queue_and_active_count(monkeypatch) -> None:
    """Service busy probe should cover queued and active primary work."""

    class _Queue:
        def __init__(self, count: int) -> None:
            self.count = count

        def pending_count(self) -> int:
            return self.count

    monkeypatch.setattr(service_module, "_chat_input_queue", _Queue(1))
    monkeypatch.setattr(service_module, "_primary_interaction_active_count", 0)
    assert service_module._primary_interaction_busy() is True

    monkeypatch.setattr(service_module, "_chat_input_queue", _Queue(0))
    monkeypatch.setattr(service_module, "_primary_interaction_active_count", 1)
    assert service_module._primary_interaction_busy() is True

    monkeypatch.setattr(service_module, "_primary_interaction_active_count", 0)
    assert service_module._primary_interaction_busy() is False


async def _run_lifespan(monkeypatch, *, disabled: bool) -> dict[str, object]:
    """Run service lifespan with external dependencies patched."""

    calls: dict[str, object] = {
        "started": 0,
        "stopped": 0,
        "busy_probe": None,
    }
    handle = SimpleNamespace(task=asyncio.create_task(_sleep_forever()), stop_event=None)

    def _start_reflection_cycle_worker(*, is_primary_interaction_busy):
        calls["started"] = int(calls["started"]) + 1
        calls["busy_probe"] = is_primary_interaction_busy
        return handle

    async def _stop_reflection_cycle_worker(_handle):
        calls["stopped"] = int(calls["stopped"]) + 1
        handle.task.cancel()
        try:
            await handle.task
        except asyncio.CancelledError:
            pass

    monkeypatch.setattr(service_module, "REFLECTION_CYCLE_DISABLED", disabled)
    monkeypatch.setattr(service_module, "db_bootstrap", AsyncMock())
    monkeypatch.setattr(
        service_module,
        "_hydrate_rag_initializer_cache",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        service_module,
        "get_character_profile",
        AsyncMock(return_value={"name": "Character"}),
    )
    monkeypatch.setattr(service_module, "_build_graph", MagicMock(return_value=object()))
    monkeypatch.setattr(service_module.mcp_manager, "start", AsyncMock())
    monkeypatch.setattr(service_module.mcp_manager, "stop", AsyncMock())
    monkeypatch.setattr(service_module, "PendingTaskIndex", _FakePendingTaskIndex)
    monkeypatch.setattr(service_module.scheduler, "load_pending_events", AsyncMock())
    monkeypatch.setattr(service_module.scheduler, "shutdown", AsyncMock())
    monkeypatch.setattr(service_module.scheduler, "configure_runtime", MagicMock())
    monkeypatch.setattr(service_module, "configure_task_dispatcher", MagicMock())
    monkeypatch.setattr(service_module, "render_llm_route_table", MagicMock(return_value="routes"))
    monkeypatch.setattr(service_module, "_ensure_chat_input_worker_started", MagicMock())
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

    async with service_module.lifespan(FastAPI()):
        pass

    return calls


async def _sleep_forever() -> None:
    """Provide a cancellable fake worker task."""

    await asyncio.Event().wait()


class _FakePendingTaskIndex:
    """Fake scheduler pending index with no database reads."""

    async def rebuild_from_db(self) -> None:
        """No-op rebuild for service lifespan tests."""

        return None
