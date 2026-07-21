"""Deterministic tests for daily sleep recovery."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.reflection_cycle import affect_settling
from kazusa_ai_chatbot.reflection_cycle.models import (
    REFLECTION_STATUS_SUCCEEDED,
)


def test_affect_settling_due_time_uses_later_of_promotion_and_wake_prep():
    due_time = affect_settling.compute_affect_settling_due_local_time(
        sleep_local_period="02:00-12:00",
        promotion_run_after_local_time="05:00",
        after_promotion_grace_minutes=15,
        wake_prep_minutes=30,
    )

    assert due_time == "11:30"


def test_affect_settling_window_closes_after_wake_defer_grace():
    assert affect_settling.local_datetime_is_in_affect_settling_window(
        "2026-05-05T11:40:00+12:00",
        sleep_local_period="02:00-12:00",
        promotion_run_after_local_time="05:00",
        after_promotion_grace_minutes=15,
        wake_prep_minutes=30,
        wake_defer_grace_minutes=15,
    )
    assert not affect_settling.local_datetime_is_in_affect_settling_window(
        "2026-05-05T12:20:00+12:00",
        sleep_local_period="02:00-12:00",
        promotion_run_after_local_time="05:00",
        after_promotion_grace_minutes=15,
        wake_prep_minutes=30,
        wake_defer_grace_minutes=15,
    )


def test_affect_settling_worker_due_allows_after_grace_catch_up():
    settling_date = affect_settling.settling_local_date_for_due_affect_settling(
        "2026-05-05T13:00:00+12:00",
        sleep_local_period="02:00-12:00",
        promotion_run_after_local_time="05:00",
        after_promotion_grace_minutes=15,
        wake_prep_minutes=30,
    )

    assert settling_date == "2026-05-05"


def test_sleep_recovery_changes_transient_state_only():
    state = build_character_production_state(
        updated_at="2026-07-14T00:00:00Z",
    )
    state["drives"]["connection"]["pressure"] = 60
    state["meaning_state"]["salience"] = 60
    before = state.copy()

    recovered, artifact = affect_settling.sleep_recovery(
        state,
        local_date_key="2026-07-14",
        elapsed_sleep_seconds=7200,
        started_at="2026-07-14T08:00:00Z",
        completed_at="2026-07-14T08:00:01Z",
    )

    assert recovered["drives"]["connection"]["pressure"] == 32
    assert recovered["drives"]["connection"]["importance"] == 70
    assert recovered["meaning_state"]["salience"] == 60
    assert state == before
    assert artifact["status"] == "completed"
    assert artifact["local_date_key"] == "2026-07-14"
    assert artifact["state_scope"] == "character"
    assert artifact["elapsed_sleep_seconds"] == 7200
    assert isinstance(artifact["semantic_recovery_summary"], str)
    assert recovered["updated_at"] == "2026-07-14T08:00:01Z"


@pytest.mark.asyncio
async def test_daily_sleep_recovery_persists_once_without_llm(monkeypatch):
    state = build_character_production_state(
        updated_at="2026-07-14T00:00:00Z",
    )
    persisted_docs: list[dict] = []
    replace_state = AsyncMock()
    monkeypatch.setattr(
        affect_settling.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        affect_settling.db,
        "get_character_cognition_state",
        AsyncMock(return_value=state),
    )
    monkeypatch.setattr(
        affect_settling.db,
        "replace_character_cognition_state",
        replace_state,
    )
    monkeypatch.setattr(
        affect_settling.db,
        "get_character_runtime_state",
        AsyncMock(side_effect=AssertionError("legacy runtime state read")),
    )

    async def _persist(document: dict) -> None:
        persisted_docs.append(document)

    monkeypatch.setattr(affect_settling.repository, "upsert_run", _persist)

    result = await affect_settling.run_daily_affect_settling(
        settling_local_date="2026-07-14",
        dry_run=False,
        enable_character_state_write=True,
    )

    assert result.succeeded_count == 1
    replace_state.assert_awaited_once()
    assert persisted_docs[-1]["status"] == REFLECTION_STATUS_SUCCEEDED
    output = persisted_docs[-1]["output"]["sleep_recovery"]
    assert output["local_date_key"] == "2026-07-14"
    assert output["elapsed_sleep_seconds"] == 10 * 60 * 60
    assert not hasattr(affect_settling, "run_affect_settling_proposal_llm")
    assert not hasattr(affect_settling, "run_affect_settling_review_llm")


@pytest.mark.asyncio
async def test_completed_local_date_reentry_does_not_mutate_state(monkeypatch):
    replace_state = AsyncMock()
    monkeypatch.setattr(
        affect_settling.repository,
        "reflection_run_by_id",
        AsyncMock(return_value={
            "status": REFLECTION_STATUS_SUCCEEDED,
            "output": {"sleep_recovery": {"status": "completed"}},
        }),
    )
    monkeypatch.setattr(
        affect_settling.db,
        "replace_character_cognition_state",
        replace_state,
    )

    result = await affect_settling.run_daily_affect_settling(
        settling_local_date="2026-07-14",
        dry_run=False,
        enable_character_state_write=True,
    )

    assert result.skipped_count == 1
    replace_state.assert_not_awaited()
