"""Worker integration tests for global character growth."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.reflection_cycle import worker as worker_module


@pytest.mark.asyncio
async def test_worker_runs_growth_after_global_reflection_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default-on worker should run growth after daily global promotion."""

    growth_pass = AsyncMock(return_value={"run_kind": "global_character_growth"})
    _patch_due_reflection_tick(monkeypatch)
    monkeypatch.setattr(worker_module, "GLOBAL_CHARACTER_GROWTH_PASS_ENABLED", True)
    monkeypatch.setattr(
        worker_module,
        "run_global_character_growth_pass",
        growth_pass,
    )

    results = await worker_module._run_worker_tick(
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
    )

    assert results[-1] == {"run_kind": "global_character_growth"}
    growth_pass.assert_awaited_once_with(
        character_local_date="2026-05-05",
        dry_run=False,
        enable_trait_writes=True,
    )


@pytest.mark.asyncio
async def test_worker_skips_growth_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback flag should stop the growth pass without stopping reflection."""

    growth_pass = AsyncMock()
    _patch_due_reflection_tick(monkeypatch)
    monkeypatch.setattr(worker_module, "GLOBAL_CHARACTER_GROWTH_PASS_ENABLED", False)
    monkeypatch.setattr(
        worker_module,
        "run_global_character_growth_pass",
        growth_pass,
    )

    results = await worker_module._run_worker_tick(
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
    )

    assert [result.run_kind for result in results] == [
        "hourly_slot",
        "daily_channel",
        "daily_interaction_style_update",
        "daily_global_promotion",
    ]
    growth_pass.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_skips_growth_if_busy_after_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new chat turn after promotion should defer the growth pass."""

    growth_pass = AsyncMock()
    _patch_due_reflection_tick(monkeypatch)
    monkeypatch.setattr(worker_module, "GLOBAL_CHARACTER_GROWTH_PASS_ENABLED", True)
    monkeypatch.setattr(
        worker_module,
        "run_global_character_growth_pass",
        growth_pass,
    )
    calls = {"count": 0}

    def _busy_after_promotion() -> bool:
        calls["count"] += 1
        return calls["count"] >= 5

    await worker_module._run_worker_tick(
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=_busy_after_promotion,
    )

    growth_pass.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_skips_growth_when_promotion_writes_no_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Growth should not spend an LLM call when promotion had no new memory."""

    growth_pass = AsyncMock()
    _patch_due_reflection_tick(monkeypatch)
    monkeypatch.setattr(worker_module, "GLOBAL_CHARACTER_GROWTH_PASS_ENABLED", True)
    monkeypatch.setattr(
        worker_module,
        "run_global_character_growth_pass",
        growth_pass,
    )
    monkeypatch.setattr(
        worker_module,
        "_run_global_reflection_promotion",
        AsyncMock(return_value=worker_module.ReflectionPromotionResult(
            run_kind="daily_global_promotion",
            dry_run=False,
            skipped_count=1,
            defer_reason="no promoted memory mutations",
        )),
    )

    results = await worker_module._run_worker_tick(
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
    )

    assert results[-1].run_kind == "daily_global_promotion"
    growth_pass.assert_not_awaited()


def _patch_due_reflection_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the reflection stages so a tick reaches global promotion."""

    monkeypatch.setattr(worker_module, "_local_time_is_after", lambda *_: True)
    monkeypatch.setattr(
        worker_module,
        "_run_hourly_reflection_cycle",
        AsyncMock(return_value=worker_module.ReflectionWorkerResult(
            run_kind="hourly_slot",
            dry_run=False,
        )),
    )
    monkeypatch.setattr(
        worker_module,
        "_run_daily_channel_reflection_cycle",
        AsyncMock(return_value=worker_module.ReflectionWorkerResult(
            run_kind="daily_channel",
            dry_run=False,
        )),
    )
    monkeypatch.setattr(
        worker_module,
        "_run_daily_interaction_style_update",
        AsyncMock(return_value=worker_module.ReflectionWorkerResult(
            run_kind="daily_interaction_style_update",
            dry_run=False,
        )),
    )
    monkeypatch.setattr(
        worker_module,
        "_run_global_reflection_promotion",
        AsyncMock(return_value=worker_module.ReflectionPromotionResult(
            run_kind="daily_global_promotion",
            dry_run=False,
            succeeded_count=1,
        )),
    )
