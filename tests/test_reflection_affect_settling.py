"""Deterministic tests for daily affect settling."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.reflection_cycle import affect_settling
from kazusa_ai_chatbot.reflection_cycle.models import (
    REFLECTION_STATUS_SKIPPED,
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


def test_affect_settling_prompt_excludes_operational_metadata():
    payload = affect_settling.build_affect_settling_payload(
        settling_local_date="2026-05-05",
        character_state={
            "mood": "刺々しいけど、眠気で角が少し丸い",
            "global_vibe": "still annoyed, quieter around the edges",
            "reflection_summary": "She remembers the argument, not just anger.",
            "updated_at": "state-token-1",
        },
        daily_docs=[
            {
                "run_id": "daily-run-1",
                "source_run_ids": ["hourly-run-hidden"],
                "output": {
                    "day_summary": "The day ended tense.",
                    "conversation_quality_patterns": ["pressure lingered"],
                    "synthesis_limitations": [],
                },
            }
        ],
        sleep_window_docs=[],
    )

    prompt = affect_settling.build_affect_settling_prompt(payload)
    prompt_text = f"{prompt.system_prompt}\n{prompt.human_prompt}"

    assert "刺々しいけど、眠気で角が少し丸い" in prompt_text
    assert "still annoyed, quieter around the edges" in prompt_text
    assert "state-token-1" not in prompt_text
    assert "updated_at" not in prompt_text
    assert "daily-run-1" not in prompt_text
    assert "hourly-run-hidden" not in prompt_text


@pytest.mark.asyncio
async def test_run_daily_affect_settling_persists_stale_state_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted_docs: list[dict] = []
    refresh = AsyncMock()
    monkeypatch.setattr(
        affect_settling.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        affect_settling.db,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "mood": "angry",
            "global_vibe": "hostile",
            "reflection_summary": "A fight still feels fresh.",
            "updated_at": "state-token-1",
        }),
    )
    monkeypatch.setattr(
        affect_settling,
        "load_affect_settling_source_documents",
        AsyncMock(return_value=([], [])),
    )
    monkeypatch.setattr(
        affect_settling,
        "run_affect_settling_proposal_llm",
        AsyncMock(return_value={
            "mood": "irritated but less explosive after sleep",
            "global_vibe": "guarded, not actively hostile",
            "reflection_summary": "Sleep softened the immediate heat.",
        }),
    )
    monkeypatch.setattr(
        affect_settling,
        "run_affect_settling_review_llm",
        AsyncMock(return_value={
            "write_decision": "accept",
            "review_reason": "The change is gradual and grounded.",
        }),
    )
    monkeypatch.setattr(
        affect_settling.db,
        "compare_and_upsert_character_state",
        AsyncMock(return_value=False),
    )

    async def _persist(document: dict) -> None:
        persisted_docs.append(document)

    monkeypatch.setattr(affect_settling.repository, "upsert_run", _persist)

    result = await affect_settling.run_daily_affect_settling(
        settling_local_date="2026-05-05",
        dry_run=False,
        enable_character_state_write=True,
        character_state_refresh_callback=refresh,
    )

    assert result.skipped_count == 1
    assert result.failed_count == 0
    assert persisted_docs[-1]["status"] == REFLECTION_STATUS_SKIPPED
    assert persisted_docs[-1]["output"]["skip_reason"] == "stale_character_state"
    assert persisted_docs[-1]["output"]["retryable"] is False
    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_daily_affect_settling_writes_free_form_llm_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted_docs: list[dict] = []
    refresh = AsyncMock()
    expected_mood = "less sharp; still proud enough to stay distant"
    expected_global_vibe = "quietly annoyed, no longer looking for a fight"
    expected_summary = "The anger survives, but sleep made it less immediate."
    compare_write = AsyncMock(return_value=True)
    monkeypatch.setattr(
        affect_settling.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        affect_settling.db,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "mood": "furious",
            "global_vibe": "sharp and rejecting",
            "reflection_summary": "She is still hurt.",
            "updated_at": "state-token-1",
        }),
    )
    monkeypatch.setattr(
        affect_settling,
        "load_affect_settling_source_documents",
        AsyncMock(return_value=([], [])),
    )
    monkeypatch.setattr(
        affect_settling,
        "run_affect_settling_proposal_llm",
        AsyncMock(return_value={
            "mood": expected_mood,
            "global_vibe": expected_global_vibe,
            "reflection_summary": expected_summary,
        }),
    )
    monkeypatch.setattr(
        affect_settling,
        "run_affect_settling_review_llm",
        AsyncMock(return_value={
            "write_decision": "accept",
            "review_reason": "No deterministic vocabulary rewrite was needed.",
        }),
    )
    monkeypatch.setattr(
        affect_settling.db,
        "compare_and_upsert_character_state",
        compare_write,
    )

    async def _persist(document: dict) -> None:
        persisted_docs.append(document)

    monkeypatch.setattr(affect_settling.repository, "upsert_run", _persist)

    result = await affect_settling.run_daily_affect_settling(
        settling_local_date="2026-05-05",
        dry_run=False,
        enable_character_state_write=True,
        character_state_refresh_callback=refresh,
    )

    assert result.succeeded_count == 1
    assert persisted_docs[-1]["status"] == REFLECTION_STATUS_SUCCEEDED
    compare_write.assert_awaited_once()
    compare_kwargs = compare_write.await_args.kwargs
    assert compare_kwargs["expected_updated_at"] == "state-token-1"
    assert compare_kwargs["mood"] == expected_mood
    assert compare_kwargs["global_vibe"] == expected_global_vibe
    assert compare_kwargs["reflection_summary"] == expected_summary
    refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_daily_affect_settling_keeps_success_when_refresh_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted_docs: list[dict] = []
    expected_mood = "less sharp after rest"
    expected_global_vibe = "guarded, but no longer actively hostile"
    expected_summary = "Sleep softened the immediate anger."
    compare_write = AsyncMock(return_value=True)
    runtime_error_event = AsyncMock()
    monkeypatch.setattr(
        affect_settling.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        affect_settling.db,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "mood": "furious",
            "global_vibe": "sharp and rejecting",
            "reflection_summary": "She is still hurt.",
            "updated_at": "state-token-1",
        }),
    )
    monkeypatch.setattr(
        affect_settling,
        "load_affect_settling_source_documents",
        AsyncMock(return_value=([], [])),
    )
    monkeypatch.setattr(
        affect_settling,
        "run_affect_settling_proposal_llm",
        AsyncMock(return_value={
            "mood": expected_mood,
            "global_vibe": expected_global_vibe,
            "reflection_summary": expected_summary,
        }),
    )
    monkeypatch.setattr(
        affect_settling,
        "run_affect_settling_review_llm",
        AsyncMock(return_value={
            "write_decision": "accept",
            "review_reason": "The change remains gradual.",
        }),
    )
    monkeypatch.setattr(
        affect_settling.db,
        "compare_and_upsert_character_state",
        compare_write,
    )
    monkeypatch.setattr(
        affect_settling.event_logging,
        "record_runtime_error_event",
        runtime_error_event,
    )

    async def _persist(document: dict) -> None:
        persisted_docs.append(document)

    async def _refresh() -> None:
        raise RuntimeError("refresh unavailable")

    monkeypatch.setattr(affect_settling.repository, "upsert_run", _persist)

    result = await affect_settling.run_daily_affect_settling(
        settling_local_date="2026-05-05",
        dry_run=False,
        enable_character_state_write=True,
        character_state_refresh_callback=_refresh,
    )

    assert result.succeeded_count == 1
    assert result.failed_count == 0
    assert persisted_docs[-1]["status"] == REFLECTION_STATUS_SUCCEEDED
    compare_write.assert_awaited_once()
    runtime_error_event.assert_awaited_once()
    error_kwargs = runtime_error_event.await_args.kwargs
    assert error_kwargs["component"] == "reflection_cycle.affect_settling"
    assert error_kwargs["recovered"] is True
