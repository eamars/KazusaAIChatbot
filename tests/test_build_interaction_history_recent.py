"""Tests for typed current-user interaction-history slicing."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.utils import build_interaction_history_recent


def _row(
    role: str,
    platform_user_id: str,
    global_user_id: str,
    content: str,
    *,
    addressed_to_global_user_ids: list[str] | None = None,
    broadcast: bool = False,
    platform_message_id: str = "",
) -> dict:
    """Build one trimmed history row for interaction-slice tests."""

    if addressed_to_global_user_ids is None and role == "user":
        addressed_to_global_user_ids = [CHARACTER_GLOBAL_USER_ID]
    return {
        "role": role,
        "platform_user_id": platform_user_id,
        "global_user_id": global_user_id,
        "body_text": content,
        "addressed_to_global_user_ids": addressed_to_global_user_ids or [],
        "broadcast": broadcast,
        "platform_message_id": platform_message_id,
        "platform_channel_id": "group-1",
        "timestamp": platform_message_id,
    }


def test_typed_history_filters_assistant_rows_by_addressee() -> None:
    """Bot replies to another user should not enter the current user's slice."""

    history = [
        _row("user", "user-a-platform", "user-a", "A asks", platform_message_id="a1"),
        _row(
            "assistant",
            "bot-platform",
            "character-global",
            "bot to A",
            addressed_to_global_user_ids=["user-a"],
            platform_message_id="bot-a1",
        ),
        _row("user", "user-b-platform", "user-b", "B asks", platform_message_id="b1"),
        _row(
            "user",
            "user-a-platform",
            "user-a",
            "A side comment",
            addressed_to_global_user_ids=[],
            platform_message_id="a-side",
        ),
        _row(
            "assistant",
            "bot-platform",
            "character-global",
            "bot to B",
            addressed_to_global_user_ids=["user-b"],
            platform_message_id="bot-b1",
        ),
        _row("user", "user-a-platform", "user-a", "A returns", platform_message_id="a2"),
        _row(
            "assistant",
            "bot-platform",
            "character-global",
            "bot to A again",
            addressed_to_global_user_ids=["user-a"],
            platform_message_id="bot-a2",
        ),
    ]

    result = build_interaction_history_recent(
        history,
        current_platform_user_id="user-a-platform",
        platform_bot_id="bot-platform",
        current_global_user_id="user-a",
    )

    assert [row["body_text"] for row in result] == [
        "A asks",
        "bot to A",
        "A side comment",
        "A returns",
        "bot to A again",
    ]


def test_typed_history_includes_broadcast_assistant_rows() -> None:
    """Broadcast assistant rows should be visible to every user slice."""

    history = [
        _row("user", "user-a-platform", "user-a", "A asks", platform_message_id="a1"),
        _row(
            "assistant",
            "bot-platform",
            "character-global",
            "channel notice",
            broadcast=True,
            platform_message_id="bot-broadcast",
        ),
    ]

    result = build_interaction_history_recent(
        history,
        current_platform_user_id="user-a-platform",
        platform_bot_id="bot-platform",
        current_global_user_id="user-a",
    )

    assert [row["body_text"] for row in result] == ["A asks", "channel notice"]


def test_untyped_history_raises_contract_error() -> None:
    """Rows without typed addressee fields violate the history contract."""

    history = [
        {
            "role": "user",
            "platform_user_id": "user-a-platform",
            "body_text": "untyped A",
            "platform_message_id": "untyped-a-e1",
            "platform_channel_id": "group-1",
            "timestamp": "untyped-a-e1",
        },
        {
            "role": "assistant",
            "platform_user_id": "bot-platform",
            "body_text": "untyped bot",
            "platform_message_id": "untyped-bot-e1",
            "platform_channel_id": "group-1",
            "timestamp": "untyped-bot-e1",
        },
    ]

    with pytest.raises(KeyError):
        build_interaction_history_recent(
            history,
            current_platform_user_id="user-a-platform",
            platform_bot_id="bot-platform",
            current_global_user_id="user-a",
        )


def test_missing_current_global_user_id_returns_empty() -> None:
    """Typed filtering requires the current user's global id."""

    history = [
        _row("user", "user-a-platform", "user-a", "A asks", platform_message_id="a1"),
        _row(
            "assistant",
            "bot-platform",
            "character-global",
            "bot to A",
            addressed_to_global_user_ids=["user-a"],
            platform_message_id="bot-a1",
        ),
    ]

    result = build_interaction_history_recent(
        history,
        current_platform_user_id="user-a-platform",
        platform_bot_id="bot-platform",
        current_global_user_id="",
    )

    assert result == []


def test_current_user_slice_hides_third_party_exchange_needed_for_group_referents() -> None:
    """Current-user slicing can hide a recent exchange referenced by group banter."""

    dangal_global_user_id = "745d7818-a9d3-4889-b7f3-8555078a2061"
    target_global_user_id = "256e8a10-c406-47e9-ac8f-efd270d18160"
    bot_platform_id = "3768713357"
    history = [
        _row(
            "user",
            "67889018",
            dangal_global_user_id,
            '反正现在有AI',
            platform_message_id="2026-05-08T01:48:45.000000+00:00",
        ),
        _row(
            "user",
            "67889018",
            dangal_global_user_id,
            '你应付下就好了',
            platform_message_id="2026-05-08T01:48:52.000000+00:00",
        ),
        _row(
            "user",
            "673225019",
            target_global_user_id,
            '把对方解决掉也是解决问题的方式之一哦',
            platform_message_id="2026-05-08T01:48:58.000000+00:00",
        ),
        _row(
            "assistant",
            bot_platform_id,
            CHARACTER_GLOBAL_USER_ID,
            '不不不，这个一点都不好笑。\n'
            '你说这种话就像被泼了冷水一样，我超不舒服的。\n'
            '真的不喜欢，别再提这种话了。',
            addressed_to_global_user_ids=[target_global_user_id],
            platform_message_id="2026-05-08T01:49:02.000000+00:00",
        ),
    ]

    result = build_interaction_history_recent(
        history,
        current_platform_user_id="67889018",
        platform_bot_id=bot_platform_id,
        current_global_user_id=dangal_global_user_id,
    )

    assert [row["body_text"] for row in result] == [
        '反正现在有AI',
        '你应付下就好了',
    ]
    assert '把对方解决掉也是解决问题的方式之一哦' in [
        row["body_text"] for row in history
    ]
    assert any('真的不喜欢' in row["body_text"] for row in history)
    assert all('真的不喜欢' not in row["body_text"] for row in result)


def test_full_history_exposes_unrelated_addressed_subthread_removed_by_slice() -> None:
    """Full group history includes other addressed subthreads that the slice removes."""

    history = [
        _row(
            "user",
            "platform-current",
            "global-current",
            "current user asks about their config",
            platform_message_id="current-1",
        ),
        _row(
            "assistant",
            "platform-bot",
            "character-global",
            "answer to current user",
            addressed_to_global_user_ids=["global-current"],
            platform_message_id="bot-current-1",
        ),
        _row(
            "user",
            "platform-other",
            "global-other",
            "other user private setup detail",
            platform_message_id="other-1",
        ),
        _row(
            "assistant",
            "platform-bot",
            "character-global",
            "answer to other user",
            addressed_to_global_user_ids=["global-other"],
            platform_message_id="bot-other-1",
        ),
        _row(
            "user",
            "platform-current",
            "global-current",
            "current user followup",
            platform_message_id="current-2",
        ),
    ]

    result = build_interaction_history_recent(
        history,
        current_platform_user_id="platform-current",
        platform_bot_id="platform-bot",
        current_global_user_id="global-current",
    )

    assert [row["body_text"] for row in history] == [
        "current user asks about their config",
        "answer to current user",
        "other user private setup detail",
        "answer to other user",
        "current user followup",
    ]
    assert [row["body_text"] for row in result] == [
        "current user asks about their config",
        "answer to current user",
        "current user followup",
    ]
