"""Deterministic projection tests for internal monologue residue."""

from __future__ import annotations

from kazusa_ai_chatbot.internal_monologue_residue.projection import (
    project_residue_window,
)


def _row(
    *,
    residue_id: str,
    residue_text: str,
    created_at: str,
    scope_kind: str = "group_scene",
) -> dict:
    """Build a residue row fixture.

    Args:
        residue_id: Stable row identifier.
        residue_text: Compact first-person carry-over text.
        created_at: UTC timestamp for age-label projection.
        scope_kind: Deterministic scope kind for the row.

    Returns:
        Residue row shaped like the DB facade returns it.
    """

    return_value = {
        "_id": "mongo-internal-id",
        "residue_id": residue_id,
        "character_id": "character-1",
        "scope_key": f"{scope_kind}:qq:group-1:user-1",
        "scope_kind": scope_kind,
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "global_user_id": "user-1",
        "residue_text": residue_text,
        "source_kind": "chat",
        "source_refs": [{"platform_message_id": "raw-message-id"}],
        "adapter_message_id": "adapter-message-id",
        "created_at": created_at,
    }
    return return_value


def test_project_residue_window_returns_empty_string_for_no_rows() -> None:
    """No selected residue produces no prompt-facing text."""

    projected = project_residue_window(
        rows=[],
        current_timestamp_utc="2026-05-20T00:10:00+00:00",
        context_char_limit=3000,
    )

    assert projected == ""


def test_project_residue_window_includes_age_and_excludes_raw_metadata() -> None:
    """Projection renders semantic residue, not storage or adapter metadata."""

    projected = project_residue_window(
        rows=[
            _row(
                residue_id="residue-1",
                residue_text='我还记得 Tobacco 用提拉米苏和赌约逗我。',
                created_at="2026-05-20T00:02:00+00:00",
            ),
        ],
        current_timestamp_utc="2026-05-20T00:10:00+00:00",
        context_char_limit=3000,
    )

    assert '8分钟前' in projected
    assert '我还记得 Tobacco 用提拉米苏和赌约逗我。' in projected
    assert 'residue-1' not in projected
    assert 'mongo-internal-id' not in projected
    assert 'raw-message-id' not in projected
    assert 'adapter-message-id' not in projected
    assert 'source_refs' not in projected


def test_project_residue_window_prefers_newer_rows_when_budget_is_tight() -> None:
    """When projection budget is tight, newer residue wins."""

    projected = project_residue_window(
        rows=[
            _row(
                residue_id="old",
                residue_text='我记得旧的低落。',
                created_at="2026-05-20T00:00:00+00:00",
            ),
            _row(
                residue_id="new",
                residue_text='我记得新的期待。',
                created_at="2026-05-20T00:09:00+00:00",
            ),
        ],
        current_timestamp_utc="2026-05-20T00:10:00+00:00",
        context_char_limit=48,
    )

    assert '我记得新的期待。' in projected
    assert '我记得旧的低落。' not in projected
