"""Tests for reflection-owned group scene digest source hydration."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.reflection_cycle import group_scene_digest
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    build_group_activity_windows,
)
from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput


def test_group_scene_digest_payload_preserves_names_and_orders_rows() -> None:
    """Digest input should preserve visible names without exposing raw ids."""

    window = _duplicate_answer_window()

    payload = group_scene_digest.build_group_scene_digest_prompt_payload(
        window,
    )

    assert payload["window"] == {
        "window_start": "2026-05-18T04:00:00+00:00",
        "window_end": "2026-05-18T04:15:00+00:00",
        "message_count": 3,
    }
    assert payload["activity_labels"]["assistant_presence"] == "present"
    assert [
        row["display_name"]
        for row in payload["message_rows"]
    ] == [
        "user",
        "assistant",
        "user",
    ]
    assert [
        row["content_activity"]
        for row in payload["message_rows"]
    ] == [
        "text",
        "text",
        "empty_or_media_only",
    ]

    payload_text = str(payload)
    assert "user-1" not in payload_text
    assert "qq-user-1" not in payload_text
    assert "bot-1" not in payload_text
    assert "group-1" not in payload_text
    assert "participant_1" not in payload_text
    assert "https://example.invalid/image.png" not in payload_text


def test_group_scene_digest_payload_preserves_display_names_for_cognition() -> None:
    """Digest input should preserve visible names instead of aliasing speakers."""

    window = _named_duplicate_answer_window()

    payload = group_scene_digest.build_group_scene_digest_prompt_payload(
        window,
    )

    message_rows = payload["message_rows"]
    rendered_payload = str(payload)

    assert [
        row["display_name"]
        for row in message_rows
    ] == [
        "蚝爹油",
        "杏山千纱",
        "蚝爹油",
    ]
    assert "participant_1" not in rendered_payload
    assert "global_user_id" not in rendered_payload
    assert "platform_user_id" not in rendered_payload


def test_group_scene_digest_payload_truncates_oldest_rows_not_latest() -> None:
    """Busy windows should preserve newest rows when prompt rows are capped."""

    window = _busy_truncation_window()

    payload = group_scene_digest.build_group_scene_digest_prompt_payload(
        window,
    )

    visible_text = "\n".join(
        row["text"]
        for row in payload["message_rows"]
    )

    assert "old noisy row 01" not in visible_text
    assert "old noisy row 02" not in visible_text
    assert "old noisy row 03" in visible_text
    assert "蚝爹油：谢谢千纱" in visible_text
    assert "能帮到你就好呀" in visible_text
    assert "张伟：后面我只是补一句主板供电" in visible_text


def test_group_scene_digest_prompt_contract_is_first_person_only() -> None:
    """Prompt rendering should keep one-string first-person summary rules."""

    messages = group_scene_digest.build_group_scene_digest_messages(
        _duplicate_answer_window(),
    )
    rendered = "\n".join(str(message.content) for message in messages)

    assert "第一人称" in rendered
    assert "digest" in rendered
    assert "display_name" in rendered
    assert '# 生成步骤' in rendered
    assert 'assistant 行的 text 当作原文引用' in rendered
    assert '引用里的 `你`、`我`、称呼和语气词保持原样' in rendered
    assert '先看 activity_labels.assistant_presence' in rendered
    assert 'assistant_presence="present"' in rendered
    assert '最后补一句当前角色发言后的状态' in rendered
    assert "participant_1" not in rendered
    assert "Kazusa" not in rendered
    assert "杏山千纱" not in rendered
    assert "development plan" not in rendered
    assert "bugfix plan" not in rendered
    assert "implementation plan" not in rendered
    assert "delivery_target" not in rendered
    assert "global_user_id" not in rendered
    assert "platform_user_id" not in rendered


def test_group_scene_digest_normalizes_one_string_output() -> None:
    """Only the one-string JSON contract should be accepted and bounded."""

    raw_digest = " 我看到这段群聊里有人问词义，我已经解释过；" + ("x" * 900)

    normalized = group_scene_digest.normalize_group_scene_digest_output({
        "digest": raw_digest,
    })

    assert normalized is not None
    assert set(normalized) == {"digest"}
    assert normalized["digest"].startswith("我看到这段群聊里")
    assert (
        len(normalized["digest"])
        <= group_scene_digest.GROUP_SCENE_DIGEST_MAX_CHARS
    )


@pytest.mark.parametrize(
    "parsed_output",
    [
        {},
        {"digest": ""},
        {"digest": 7},
        {"digest": "我看到这段群聊有人问问题。", "confidence": "high"},
        {"digest": "你刚刚已经回答过了，不要再说。"},
        {"should_speak": False, "digest": "我看到这段群聊有人问问题。"},
        {"action_recommendation": "should_stay_silent"},
        {"digest": "resolved issue; suppress another answer."},
    ],
)
def test_group_scene_digest_rejects_invalid_or_guidance_output(
    parsed_output: dict[str, Any],
) -> None:
    """Invalid shapes and action guidance must omit the digest."""

    normalized = group_scene_digest.normalize_group_scene_digest_output(
        parsed_output,
    )

    assert normalized is None


@pytest.mark.asyncio
async def test_build_group_scene_digest_invokes_llm_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder should return parsed one-string JSON from one LLM call."""

    captured_messages: list[Any] = []

    class FakeDigestLLM:
        async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
            captured_messages.extend(messages)
            response = SimpleNamespace(
                content=(
                    '{"digest": "这段群聊里，user 问了词义，'
                    '我（assistant）随后已经解释过；后面只有媒体/空内容。"}'
                ),
            )
            return response

    monkeypatch.setattr(
        group_scene_digest,
        "_group_scene_digest_llm",
        FakeDigestLLM(),
    )

    result = await group_scene_digest.build_group_scene_digest(
        _duplicate_answer_window(),
    )

    assert result == {
        "digest": (
            "这段群聊里，user 问了词义，"
            "我（assistant）随后已经解释过；后面只有媒体/空内容。"
        )
    }
    assert len(captured_messages) == 2


def _duplicate_answer_window():
    """Build one duplicate-answer-shaped group activity window."""

    messages = [
        _message(
            role="user",
            timestamp="2026-05-18T04:01:00+00:00",
            body_text="耄耋是什么意思？",
            global_user_id="user-1",
            platform_user_id="qq-user-1",
            addressed=True,
        ),
        _message(
            role="assistant",
            timestamp="2026-05-18T04:07:00+00:00",
            body_text="耄耋一般指八九十岁的年纪。",
            global_user_id="character-global",
            platform_user_id="bot-1",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:08:00+00:00",
            body_text="",
            global_user_id="user-1",
            platform_user_id="qq-user-1",
            addressed=False,
        ),
    ]
    messages[-1]["attachments"] = [
        {"url": "https://example.invalid/image.png"},
    ]
    scope = ReflectionScopeInput(
        scope_ref="scope_group",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        assistant_message_count=1,
        user_message_count=2,
        total_message_count=3,
        first_timestamp="2026-05-18T04:01:00+00:00",
        last_timestamp="2026-05-18T04:08:00+00:00",
        messages=messages,
    )
    windows = build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
        now=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
        character_global_user_id="character-global",
        platform_bot_id="bot-1",
    )
    assert len(windows) == 1
    return windows[0]


def _named_duplicate_answer_window():
    """Build a duplicate-answer-shaped window with visible speaker names."""

    messages = [
        _message(
            role="user",
            timestamp="2026-05-18T04:01:00+00:00",
            body_text="@杏山千纱 谢谢千纱",
            global_user_id="user-1",
            platform_user_id="qq-user-1",
            display_name="蚝爹油",
            addressed=True,
        ),
        _message(
            role="assistant",
            timestamp="2026-05-18T04:02:00+00:00",
            body_text="能帮到你就好呀",
            global_user_id="character-global",
            platform_user_id="bot-1",
            display_name="杏山千纱",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:03:00+00:00",
            body_text="",
            global_user_id="user-1",
            platform_user_id="qq-user-1",
            display_name="蚝爹油",
            addressed=False,
        ),
    ]
    scope = ReflectionScopeInput(
        scope_ref="scope_group_named",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        assistant_message_count=1,
        user_message_count=2,
        total_message_count=3,
        first_timestamp="2026-05-18T04:01:00+00:00",
        last_timestamp="2026-05-18T04:03:00+00:00",
        messages=messages,
    )
    windows = build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
        now=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
        character_global_user_id="character-global",
        platform_bot_id="bot-1",
    )
    assert len(windows) == 1
    return windows[0]


def _busy_truncation_window():
    """Build a busy window where latest rows must survive row limiting."""

    messages = [
        _message(
            role="user",
            timestamp="2026-05-18T04:00:10+00:00",
            body_text="old noisy row 01",
            global_user_id="old-1",
            platform_user_id="qq-old-1",
            display_name="旧话题一",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:00:20+00:00",
            body_text="old noisy row 02",
            global_user_id="old-2",
            platform_user_id="qq-old-2",
            display_name="旧话题二",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:00:30+00:00",
            body_text="old noisy row 03",
            global_user_id="old-3",
            platform_user_id="qq-old-3",
            display_name="旧话题三",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:00:40+00:00",
            body_text="old noisy row 04",
            global_user_id="old-4",
            platform_user_id="qq-old-4",
            display_name="旧话题四",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:01:00+00:00",
            body_text="总是跌倒的企鹅：用串口 Linux 启动",
            global_user_id="user-penguin",
            platform_user_id="qq-penguin",
            display_name="总是跌倒的企鹅",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:02:00+00:00",
            body_text="白狐：工作站 POST 阶段不结束",
            global_user_id="user-baihu",
            platform_user_id="qq-baihu",
            display_name="白狐",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:03:00+00:00",
            body_text="白狐：USB 都不带有电的",
            global_user_id="user-baihu",
            platform_user_id="qq-baihu",
            display_name="白狐",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:04:00+00:00",
            body_text="蚝爹油：谢谢千纱",
            global_user_id="user-oyster",
            platform_user_id="qq-oyster",
            display_name="蚝爹油",
            addressed=True,
        ),
        _message(
            role="assistant",
            timestamp="2026-05-18T04:05:00+00:00",
            body_text="能帮到你就好呀",
            global_user_id="character-global",
            platform_user_id="bot-1",
            display_name="杏山千纱",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:06:00+00:00",
            body_text="张伟：后面我只是补一句主板供电",
            global_user_id="user-zhang",
            platform_user_id="qq-zhang",
            display_name="张伟",
        ),
    ]
    scope = ReflectionScopeInput(
        scope_ref="scope_group_busy",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        assistant_message_count=1,
        user_message_count=9,
        total_message_count=len(messages),
        first_timestamp="2026-05-18T04:00:10+00:00",
        last_timestamp="2026-05-18T04:06:00+00:00",
        messages=messages,
    )
    windows = build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
        now=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
        character_global_user_id="character-global",
        platform_bot_id="bot-1",
    )
    assert len(windows) == 1
    return windows[0]


def _message(
    *,
    role: str,
    timestamp: str,
    body_text: str,
    global_user_id: str,
    platform_user_id: str,
    display_name: str | None = None,
    addressed: bool = False,
) -> dict[str, Any]:
    """Build one activity-window source row."""

    visible_name = display_name or role
    message = {
        "role": role,
        "timestamp": timestamp,
        "body_text": body_text,
        "display_name": visible_name,
        "global_user_id": global_user_id,
        "platform_user_id": platform_user_id,
        "platform_message_id": "msg-" + timestamp.replace("-", "").replace(
            ":",
            "",
        ),
        "addressed_to_global_user_ids": (
            ["character-global"] if addressed else []
        ),
        "mentions": [],
    }
    return message
