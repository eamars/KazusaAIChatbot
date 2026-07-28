"""Worker integration tests for daily character identity growth."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.reflection_cycle import worker as worker_module


@pytest.mark.asyncio
async def test_worker_runs_identity_after_global_reflection_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default-on worker runs identity after daily global promotion."""

    growth_pass = AsyncMock(return_value={"status": "no_change"})
    _patch_due_reflection_tick(monkeypatch)
    monkeypatch.setattr(
        worker_module,
        "CHARACTER_IDENTITY_GROWTH_ENABLED",
        True,
    )
    monkeypatch.setattr(
        worker_module,
        "run_reflection_identity_growth_pass",
        growth_pass,
    )

    results = await worker_module._run_worker_tick(
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
    )

    assert results[-1] == {"status": "no_change"}
    growth_pass.assert_awaited_once_with(
        character_local_date="2026-05-05",
        source_reflection_run_ids=["daily-run-1"],
        dry_run=False,
        enable_revision_writes=True,
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_worker_skips_identity_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The restart-applied flag stops identity while retaining reflection."""

    growth_pass = AsyncMock(return_value={"status": "no_change"})
    _patch_due_reflection_tick(monkeypatch)
    monkeypatch.setattr(
        worker_module,
        "CHARACTER_IDENTITY_GROWTH_ENABLED",
        False,
    )
    monkeypatch.setattr(
        worker_module,
        "run_reflection_identity_growth_pass",
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
async def test_worker_skips_identity_if_busy_after_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new chat turn after promotion should defer the identity pass."""

    growth_pass = AsyncMock()
    _patch_due_reflection_tick(monkeypatch)
    monkeypatch.setattr(
        worker_module,
        "CHARACTER_IDENTITY_GROWTH_ENABLED",
        True,
    )
    monkeypatch.setattr(
        worker_module,
        "run_reflection_identity_growth_pass",
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
async def test_worker_runs_identity_when_promotion_writes_no_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity evaluation is independent of the memory mutation outcome."""

    growth_pass = AsyncMock(return_value={"status": "no_change"})
    _patch_due_reflection_tick(monkeypatch)
    monkeypatch.setattr(
        worker_module,
        "CHARACTER_IDENTITY_GROWTH_ENABLED",
        True,
    )
    monkeypatch.setattr(
        worker_module,
        "run_reflection_identity_growth_pass",
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

    assert results[-1] == {"status": "no_change"}
    growth_pass.assert_awaited_once_with(
        character_local_date="2026-05-05",
        source_reflection_run_ids=["daily-run-1"],
        dry_run=False,
        enable_revision_writes=True,
        now=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
    )


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
    monkeypatch.setattr(
        worker_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[{"run_id": "daily-run-1"}]),
    )
