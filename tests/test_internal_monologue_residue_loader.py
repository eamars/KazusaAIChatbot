"""Deterministic loader tests for internal monologue residue."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.internal_monologue_residue import loader
from kazusa_ai_chatbot.internal_monologue_residue.loader import (
    build_scope_key,
    load_residue_context,
    select_residue_window,
)
from kazusa_ai_chatbot.internal_monologue_residue.models import (
    ResidueTriggerScope,
)


def _row(
    *,
    residue_id: str,
    scope_kind: str,
    platform_channel_id: str,
    global_user_id: str,
    created_at: str,
    disposition: str = "append",
    residue_text: str | None = None,
) -> dict:
    """Build a candidate row fixture.

    Args:
        residue_id: Stable row identifier.
        scope_kind: Candidate scope kind.
        platform_channel_id: Channel associated with the candidate.
        global_user_id: User associated with the candidate.
        created_at: UTC timestamp used for recency sorting.

    Returns:
        Candidate residue row.
    """

    return_value = {
        "residue_id": residue_id,
        "character_id": "character-1",
        "scope_key": f"{scope_kind}:qq:{platform_channel_id}:{global_user_id}",
        "scope_kind": scope_kind,
        "platform": "qq",
        "platform_channel_id": platform_channel_id,
        "channel_type": "group",
        "global_user_id": global_user_id,
        "residue_text": (
            residue_text
            if residue_text is not None
            else f"residue {residue_id}"
        ),
        "source_kind": "chat",
        "source_refs": [],
        "created_at": created_at,
        "schema_version": "internal_monologue_residue.v2",
        "operation_id": f"operation-{residue_id}",
        "disposition": disposition,
        "purge_at": datetime(2026, 5, 22, tzinfo=timezone.utc),
    }
    return return_value


def test_select_residue_window_filters_scope_and_respects_priority() -> None:
    """Exact user residue outranks group and global residue."""

    trigger_scope = ResidueTriggerScope(
        character_id="character-1",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        global_user_id="user-1",
    )
    selected = select_residue_window(
        trigger_scope=trigger_scope,
        rows=[
            _row(
                residue_id="other-channel",
                scope_kind="group_scene",
                platform_channel_id="group-2",
                global_user_id="user-1",
                created_at="2026-05-20T00:12:00+00:00",
            ),
            _row(
                residue_id="group-new",
                scope_kind="group_scene",
                platform_channel_id="group-1",
                global_user_id="",
                created_at="2026-05-20T00:11:00+00:00",
            ),
            _row(
                residue_id="user-old",
                scope_kind="user_thread",
                platform_channel_id="group-1",
                global_user_id="user-1",
                created_at="2026-05-20T00:01:00+00:00",
            ),
            _row(
                residue_id="global-new",
                scope_kind="user_thread",
                platform_channel_id="",
                global_user_id="",
                created_at="2026-05-20T00:10:00+00:00",
            ),
        ],
        window_size=2,
    )

    selected_ids = [row["residue_id"] for row in selected]
    assert selected_ids == ["user-old", "group-new"]


def test_select_residue_window_keeps_newest_rows_within_same_priority() -> None:
    """Within one scope priority, newest rows are selected first."""

    trigger_scope = ResidueTriggerScope(
        character_id="character-1",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        global_user_id="user-1",
    )
    selected = select_residue_window(
        trigger_scope=trigger_scope,
        rows=[
            _row(
                residue_id="user-old",
                scope_kind="user_thread",
                platform_channel_id="group-1",
                global_user_id="user-1",
                created_at="2026-05-20T00:01:00+00:00",
            ),
            _row(
                residue_id="user-new",
                scope_kind="user_thread",
                platform_channel_id="group-1",
                global_user_id="user-1",
                created_at="2026-05-20T00:09:00+00:00",
            ),
        ],
        window_size=1,
    )

    assert [row["residue_id"] for row in selected] == ["user-new"]


@pytest.mark.asyncio
async def test_load_residue_context_can_disable_read_telemetry(monkeypatch) -> None:
    """Read-only inspectors can load residue without writing an event row."""

    from kazusa_ai_chatbot.internal_monologue_residue import loader

    list_rows = AsyncMock(return_value=[])
    record_event = AsyncMock()
    monkeypatch.setattr(
        loader.db,
        "list_internal_monologue_residue_rows",
        list_rows,
    )
    monkeypatch.setattr(
        loader.event_logging,
        "record_continuity_boundary_event",
        record_event,
    )
    trigger_scope = ResidueTriggerScope(
        character_id="character-1",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        global_user_id="user-1",
    )

    result = await loader.load_residue_context(
        trigger_scope=trigger_scope,
        current_timestamp_utc="2026-07-27T00:00:00+00:00",
        record_telemetry=False,
    )

    assert result["status"] == "empty"
    record_event.assert_not_awaited()

    await loader.load_residue_context(
        trigger_scope=trigger_scope,
        current_timestamp_utc="2026-07-27T00:00:00+00:00",
    )
    record_event.assert_awaited_once()


def test_select_residue_window_stops_at_scoped_empty_clear_barrier() -> None:
    """A clear marker hides older rows while later appends remain visible."""

    trigger_scope = ResidueTriggerScope(
        character_id="character-1",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        global_user_id="user-1",
    )
    selected = select_residue_window(
        trigger_scope=trigger_scope,
        rows=[
            _row(
                residue_id="old",
                scope_kind="user_thread",
                platform_channel_id="group-1",
                global_user_id="user-1",
                created_at="2026-05-20T00:01:00+00:00",
                disposition="append",
            ),
            _row(
                residue_id="clear",
                scope_kind="user_thread",
                platform_channel_id="group-1",
                global_user_id="user-1",
                created_at="2026-05-20T00:02:00+00:00",
                disposition="clear_scope",
                residue_text="",
            ),
            _row(
                residue_id="after",
                scope_kind="user_thread",
                platform_channel_id="group-1",
                global_user_id="user-1",
                created_at="2026-05-20T00:03:00+00:00",
                disposition="append",
            ),
        ],
        window_size=8,
    )

    selected_ids = {row['residue_id'] for row in selected}
    assert selected_ids == {'clear', 'after'}


def test_clear_barrier_keeps_other_user_residue_isolated() -> None:
    """Clearing one user thread cannot hide or expose another user's row."""

    trigger_scope = ResidueTriggerScope(
        character_id="character-1",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        global_user_id="user-1",
    )
    selected = select_residue_window(
        trigger_scope=trigger_scope,
        rows=[
            _row(
                residue_id="user-one-clear",
                scope_kind="user_thread",
                platform_channel_id="group-1",
                global_user_id="user-1",
                created_at="2026-05-20T00:03:00+00:00",
                disposition="clear_scope",
                residue_text="",
            ),
            _row(
                residue_id="user-two-row",
                scope_kind="user_thread",
                platform_channel_id="group-1",
                global_user_id="user-2",
                created_at="2026-05-20T00:04:00+00:00",
            ),
        ],
        window_size=8,
    )

    assert [row['residue_id'] for row in selected] == ['user-one-clear']


def test_noncanonical_rows_are_excluded_from_the_residue_window() -> None:
    """Rows without the canonical v2 contract never reach prompt context."""

    trigger_scope = ResidueTriggerScope(
        character_id="character-1",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        global_user_id="user-1",
    )
    noncanonical = _row(
        residue_id="noncanonical",
        scope_kind="user_thread",
        platform_channel_id="group-1",
        global_user_id="user-1",
        created_at="2026-05-20T00:01:00+00:00",
    )
    noncanonical.pop("schema_version")
    noncanonical.pop("operation_id")
    noncanonical.pop("disposition")
    noncanonical.pop("purge_at")

    selected = select_residue_window(
        trigger_scope=trigger_scope,
        rows=[noncanonical],
        window_size=8,
    )

    assert selected == []


@pytest.mark.asyncio
async def test_clear_barrier_load_reports_cleared_status(monkeypatch) -> None:
    """The load contract distinguishes a clear barrier from an empty scope."""

    scope_key = build_scope_key(
        character_id='character-1',
        scope_kind='user_thread',
        platform='qq',
        platform_channel_id='group-1',
        global_user_id='user-1',
    )
    clear_row = _row(
        residue_id='clear-marker',
        scope_kind='user_thread',
        platform_channel_id='group-1',
        global_user_id='user-1',
        created_at='2026-05-20T00:03:00+00:00',
        disposition='clear_scope',
        residue_text='',
    )
    clear_row['scope_key'] = scope_key
    monkeypatch.setattr(
        loader.db,
        'list_internal_monologue_residue_rows',
        AsyncMock(return_value=[clear_row]),
    )
    event_recorder = AsyncMock()
    monkeypatch.setattr(
        loader.event_logging,
        'record_continuity_boundary_event',
        event_recorder,
    )

    result = await load_residue_context(
        trigger_scope=ResidueTriggerScope(
            character_id='character-1',
            platform='qq',
            platform_channel_id='group-1',
            channel_type='group',
            global_user_id='user-1',
        ),
        current_timestamp_utc='2026-05-20T00:10:00+00:00',
    )

    assert result['status'] == 'cleared'
    assert result['barrier_disposition'] == 'clear_scope'
    assert result['internal_monologue_residue_context'] == ''
    event_recorder.assert_awaited_once()
    assert event_recorder.await_args.kwargs['barrier_disposition'] == (
        'clear_scope'
    )
