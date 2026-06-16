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
    window.digest_participant_rows[1].pop("display_name", None)

    payload = group_scene_digest.build_group_scene_digest_prompt_payload(
        window,
    )

    assert payload["window"] == {
        "window_start": "2026-05-18T04:00:00+00:00",
        "window_end": "2026-05-18T04:15:00+00:00",
        "message_count": 3,
    }
    assert payload["activity_labels"]["assistant_presence"] == "present"
    lines = payload["message_lines"]
    assert len(lines) == 3
    assert "user:" in lines[0]
    assert "当前角色:" in lines[1]
    assert "user:" in lines[2]

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

    message_lines = payload["message_lines"]
    rendered_payload = str(payload)

    assert [
        line.split("] ")[1].split(":")[0] if "] " in line else line.split(":")[0]
        for line in message_lines
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

    visible_text = "\n".join(payload["message_lines"])

    assert "old noisy row 01" not in visible_text
    assert "old noisy row 02" not in visible_text
    assert "old noisy row 03" in visible_text
    assert "蚝爹油：谢谢千纱" in visible_text
    assert "能帮到你就好呀" in visible_text
    assert "张伟：后面我只是补一句主板供电" in visible_text


def test_group_scene_digest_payload_uses_wide_digest_rows() -> None:
    """Digest input should include rolling same-scope rows before the window."""

    scope = ReflectionScopeInput(
        scope_ref="scope_group_laxi",
        platform="qq",
        platform_channel_id="group-laxi",
        channel_type="group",
        assistant_message_count=0,
        user_message_count=5,
        total_message_count=5,
        first_timestamp="2026-06-12T00:03:00+00:00",
        last_timestamp="2026-06-12T00:26:00+00:00",
        messages=[
            _message(
                role="user",
                timestamp="2026-06-12T00:03:00+00:00",
                body_text="山风太冷，温格受凉了。",
                global_user_id="user-1816",
                platform_user_id="qq-1816",
                display_name="1816",
            ),
            _message(
                role="user",
                timestamp="2026-06-12T00:10:00+00:00",
                body_text="温格高艾菲波加查说刚才爬山没力气。",
                global_user_id="user-wenge",
                platform_user_id="qq-wenge",
                display_name="温格高艾菲波加查",
            ),
            _message(
                role="user",
                timestamp="2026-06-12T00:17:00+00:00",
                body_text="正常拉稀怎么会还有力气骑。",
                global_user_id="user-w",
                platform_user_id="qq-w",
                display_name="W",
            ),
            _message(
                role="user",
                timestamp="2026-06-12T00:18:00+00:00",
                body_text="你这不像没力气。",
                global_user_id="user-w",
                platform_user_id="qq-w",
                display_name="W",
            ),
            _message(
                role="user",
                timestamp="2026-06-12T00:26:00+00:00",
                body_text="future row must not leak into the selected window digest.",
                global_user_id="user-future",
                platform_user_id="qq-future",
                display_name="未来发言者",
            ),
        ],
    )

    windows = build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 6, 12, 0, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 6, 12, 0, 25, tzinfo=timezone.utc),
        now=datetime(2026, 6, 12, 0, 25, tzinfo=timezone.utc),
        character_global_user_id="character-global",
        platform_bot_id="bot-1",
    )
    selected_window = next(
        window for window in windows
        if window.window_start == datetime(2026, 6, 12, 0, 15, tzinfo=timezone.utc)
    )

    payload = group_scene_digest.build_group_scene_digest_prompt_payload(
        selected_window,
    )
    visible_window_text = "\n".join(
        row["body_text"]
        for row in selected_window.visible_context
    )
    digest_text = "\n".join(payload["message_lines"])

    assert "温格高艾菲波加查说刚才爬山没力气" not in visible_window_text
    assert "温格高艾菲波加查说刚才爬山没力气" in digest_text
    assert "正常拉稀怎么会还有力气骑" in digest_text
    assert "future row must not leak" not in digest_text


def test_group_scene_digest_prompt_contract_is_first_person_only() -> None:
    """Prompt rendering should keep one-string first-person summary rules."""

    messages = group_scene_digest.build_group_scene_digest_messages(
        _duplicate_answer_window(),
    )
    rendered = "\n".join(str(message.content) for message in messages)

    assert "第一人称" in rendered
    assert "digest" in rendered
    assert "summary" in rendered
    assert "说话人" in rendered
    assert '# 生成步骤' in rendered
    assert '当前角色自己的行当作原文引用' in rendered
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


def test_group_scene_digest_normalizes_optional_short_summary() -> None:
    """A short string topic summary may accompany the full digest."""

    normalized = group_scene_digest.normalize_group_scene_digest_output({
        "digest": "我看到 W 的调侃承接的是温格前面说身体不舒服。",
        "summary": "W 和温格围绕受凉、拉稀和没力气开玩笑。",
    })

    assert normalized == {
        "digest": "我看到 W 的调侃承接的是温格前面说身体不舒服。",
        "summary": "W 和温格围绕受凉、拉稀和没力气开玩笑。",
    }


@pytest.mark.parametrize(
    "parsed_output",
    [
        {"digest": "我看到这段群聊有人问问题。", "summary": 7},
        {"digest": "我看到这段群聊有人问问题。", "summary": ""},
        {
            "digest": "我看到这段群聊有人问问题。",
            "summary": "这段应该保持沉默。",
        },
    ],
)
def test_group_scene_digest_omits_invalid_optional_summary(
    parsed_output: dict[str, Any],
) -> None:
    """An invalid optional summary should not suppress a valid digest."""

    normalized = group_scene_digest.normalize_group_scene_digest_output(
        parsed_output,
    )

    assert normalized == {"digest": "我看到这段群聊有人问问题。"}


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
        async def ainvoke(self, messages: list[Any], *, config=None) -> SimpleNamespace:
            captured_messages.extend(messages)
            response = SimpleNamespace(
                content=(
                    '{"digest": "这段群聊里，user 问了词义，'
                    '我（assistant）随后已经解释过；后面只有媒体/空内容。",'
                    '"summary": "user 问词义，我已经解释过。"}'
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
        ),
        "summary": "user 问词义，我已经解释过。",
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
            timestamp="2026-05-18T04:00:50+00:00",
            body_text="old noisy row 05",
            global_user_id="old-5",
            platform_user_id="qq-old-5",
            display_name="旧话题五",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:00:55+00:00",
            body_text="old noisy row 06",
            global_user_id="old-6",
            platform_user_id="qq-old-6",
            display_name="旧话题六",
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
        user_message_count=11,
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
