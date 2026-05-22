"""Deterministic loader tests for internal monologue residue."""

from __future__ import annotations

from kazusa_ai_chatbot.internal_monologue_residue.loader import (
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
        "residue_text": f"residue {residue_id}",
        "source_kind": "chat",
        "source_refs": [],
        "created_at": created_at,
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
                scope_kind="character_global",
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
