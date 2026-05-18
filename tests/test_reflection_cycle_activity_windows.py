"""Deterministic tests for reflection-attached group activity windows."""

from __future__ import annotations

from datetime import datetime, timezone

from kazusa_ai_chatbot.reflection_cycle import activity_windows
from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput


def test_build_group_activity_windows_projects_non_empty_15_minute_windows(
) -> None:
    """Group activity windows should be semantic, bounded, and non-empty."""

    scope = _group_scope([
        _message(
            role="user",
            timestamp="2026-05-18T04:01:00+00:00",
            body_text="Please look at this group issue.",
            global_user_id="user-1",
            platform_user_id="qq-user-1",
            addressed=True,
        ),
        _message(
            role="assistant",
            timestamp="2026-05-18T04:05:00+00:00",
            body_text="I can help with the outline.",
            global_user_id="character-global",
            platform_user_id="bot-1",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:18:00+00:00",
            body_text="Another participant adds context.",
            global_user_id="user-2",
            platform_user_id="qq-user-2",
        ),
    ])

    windows = activity_windows.build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 45, tzinfo=timezone.utc),
        now=datetime(2026, 5, 18, 4, 45, tzinfo=timezone.utc),
        character_global_user_id="character-global",
        platform_bot_id="bot-1",
    )

    assert [window.window_index for window in windows] == [1, 2]
    assert [
        window.window_start.isoformat()
        for window in windows
    ] == [
        "2026-05-18T04:00:00+00:00",
        "2026-05-18T04:15:00+00:00",
    ]
    assert windows[0].labels == {
        "activity_level": "quiet",
        "speaker_diversity": "one_speaker",
        "assistant_presence": "present",
        "bot_addressing": "directly_addressed",
        "message_recency": "recent",
        "noise_level": "low",
        "response_risk": "low",
        "window_summary": (
            "quiet group activity, one_speaker speakers, "
            "directly_addressed, present, risk low"
        ),
    }
    assert windows[0].visible_context == [
        {
            "timestamp": "2026-05-18T04:01:00+00:00",
            "role": "user",
            "display_name": "user",
            "body_text": "Please look at this group issue.",
            "platform_message_id": "msg-20260518T0401000000",
        },
        {
            "timestamp": "2026-05-18T04:05:00+00:00",
            "role": "assistant",
            "display_name": "assistant",
            "body_text": "I can help with the outline.",
            "platform_message_id": "msg-20260518T0405000000",
        },
    ]
    assert windows[0].source_refs == [
        {
            "source_kind": "reflection_activity_window",
            "source_id": (
                "scope_group:"
                "2026-05-18T04:00:00+00:00:"
                "2026-05-18T04:15:00+00:00"
            ),
            "due_at": None,
            "summary": windows[0].labels["window_summary"],
        }
    ]
    assert windows[1].labels["assistant_presence"] == "not_in_window"
    assert windows[1].labels["response_risk"] == "unclear"


def test_group_activity_window_visible_context_is_bounded() -> None:
    """Only recent bounded rows should enter self-cognition context."""

    messages = [
        _message(
            role="user",
            timestamp=f"2026-05-18T04:{minute:02d}:00+00:00",
            body_text=(
                "long body "
                + str(minute)
                + " "
                + ("x" * 400)
            ),
            global_user_id=f"user-{minute}",
            platform_user_id=f"qq-user-{minute}",
        )
        for minute in range(12)
    ]
    scope = _group_scope(messages)

    windows = activity_windows.build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
        now=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
    )

    assert len(windows) == 1
    assert len(windows[0].visible_context) == 6
    assert windows[0].visible_context[0]["timestamp"] == (
        "2026-05-18T04:06:00+00:00"
    )
    assert windows[0].visible_context[-1]["timestamp"] == (
        "2026-05-18T04:11:00+00:00"
    )
    assert len(windows[0].visible_context[-1]["body_text"]) <= 280
    assert windows[0].labels["activity_level"] == "bursty"
    assert windows[0].labels["speaker_diversity"] == "many_speakers"
    assert windows[0].labels["noise_level"] == "high"


def test_group_activity_window_direct_address_requires_character_identity(
) -> None:
    """Mentions of other users must not be labeled as bot addressing."""

    other_user_message = _message(
        role="user",
        timestamp="2026-05-18T04:01:00+00:00",
        body_text="Can someone else check this?",
        global_user_id="user-1",
        platform_user_id="qq-user-1",
    )
    other_user_message["addressed_to_global_user_ids"] = ["other-user"]
    other_user_message["mentions"] = [
        {
            "entity_kind": "user",
            "global_user_id": "other-user",
            "platform_user_id": "qq-other",
        }
    ]
    character_message = _message(
        role="user",
        timestamp="2026-05-18T04:16:00+00:00",
        body_text="Character, can you check this?",
        global_user_id="user-2",
        platform_user_id="qq-user-2",
    )
    character_message["addressed_to_global_user_ids"] = ["character-global"]
    scope = _group_scope([other_user_message, character_message])

    windows = activity_windows.build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 30, tzinfo=timezone.utc),
        now=datetime(2026, 5, 18, 4, 30, tzinfo=timezone.utc),
        character_global_user_id="character-global",
        platform_bot_id="bot-1",
    )

    assert windows[0].labels["bot_addressing"] == "ambient_group_context"
    assert windows[0].labels["response_risk"] == "unclear"
    assert windows[1].labels["bot_addressing"] == "directly_addressed"
    assert windows[1].labels["response_risk"] == "low"


def test_build_hourly_aggregation_preview_groups_four_windows() -> None:
    """Four 15-minute windows should remain compatible with hourly reflection."""

    scope = _group_scope([
        _message("user", "2026-05-18T04:01:00+00:00", "first"),
        _message("user", "2026-05-18T04:16:00+00:00", "second"),
        _message("assistant", "2026-05-18T04:31:00+00:00", "third"),
        _message("user", "2026-05-18T04:46:00+00:00", "fourth"),
    ])
    windows = activity_windows.build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 5, 0, tzinfo=timezone.utc),
        now=datetime(2026, 5, 18, 5, 0, tzinfo=timezone.utc),
    )

    preview = activity_windows.build_hourly_aggregation_preview(windows)

    assert preview == [
        {
            "hour_start": "2026-05-18T04:00:00+00:00",
            "window_count": 4,
            "message_count": 4,
            "summary_labels": [
                "quiet group activity, one_speaker speakers, "
                "ambient_group_context, not_in_window, risk unclear",
                "quiet group activity, one_speaker speakers, "
                "ambient_group_context, not_in_window, risk unclear",
                "quiet group activity, no_speakers speakers, "
                "ambient_group_context, present, risk low",
                "quiet group activity, one_speaker speakers, "
                "ambient_group_context, not_in_window, risk unclear",
            ],
        }
    ]


def test_build_group_activity_windows_ignores_private_scopes() -> None:
    """Group review must not turn private channels into group windows."""

    scope = ReflectionScopeInput(
        scope_ref="scope_private",
        platform="qq",
        platform_channel_id="dm-1",
        channel_type="private",
        assistant_message_count=1,
        user_message_count=1,
        total_message_count=2,
        first_timestamp="2026-05-18T04:01:00+00:00",
        last_timestamp="2026-05-18T04:02:00+00:00",
        messages=[
            _message("user", "2026-05-18T04:01:00+00:00", "private"),
            _message("assistant", "2026-05-18T04:02:00+00:00", "reply"),
        ],
    )

    windows = activity_windows.build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
        now=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
    )

    assert windows == []


def _group_scope(messages: list[dict[str, object]]) -> ReflectionScopeInput:
    """Build one monitored group scope for activity-window tests."""

    scope = ReflectionScopeInput(
        scope_ref="scope_group",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        assistant_message_count=sum(
            1
            for message in messages
            if message["role"] == "assistant"
        ),
        user_message_count=sum(
            1
            for message in messages
            if message["role"] == "user"
        ),
        total_message_count=len(messages),
        first_timestamp=str(messages[0]["timestamp"]),
        last_timestamp=str(messages[-1]["timestamp"]),
        messages=messages,
    )
    return scope


def _message(
    role: str,
    timestamp: str,
    body_text: str,
    global_user_id: str = "",
    platform_user_id: str = "",
    addressed: bool = False,
) -> dict[str, object]:
    """Build one conversation row for group-window tests."""

    message = {
        "role": role,
        "timestamp": timestamp,
        "body_text": body_text,
        "display_name": role,
        "global_user_id": global_user_id,
        "platform_user_id": platform_user_id,
        "platform_message_id": (
            "msg-" + timestamp.replace("-", "").replace(":", "")
        ),
        "addressed_to_global_user_ids": (
            ["character-global"] if addressed else []
        ),
        "mentions": [],
    }
    return message
