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
