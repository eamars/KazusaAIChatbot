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


def test_group_scene_digest_payload_deidentifies_and_orders_rows() -> None:
    """Digest input should preserve flow without exposing source identities."""

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
        row["speaker_ref"]
        for row in payload["message_rows"]
    ] == [
        "participant_1",
        "active_character",
        "participant_1",
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
    assert "https://example.invalid/image.png" not in payload_text


def test_group_scene_digest_prompt_contract_is_first_person_only() -> None:
    """Prompt rendering should keep one-string first-person summary rules."""

    messages = group_scene_digest.build_group_scene_digest_messages(
        _duplicate_answer_window(),
    )
    rendered = "\n".join(str(message.content) for message in messages)

    assert "第一人称" in rendered
    assert "digest" in rendered
    assert "active_character" in rendered
    assert "participant_1" in rendered
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
                    '{"digest": "这段群聊里，participant_1 问了词义，'
                    '我随后已经解释过；后面只有媒体/空内容。"}'
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
            "这段群聊里，participant_1 问了词义，"
            "我随后已经解释过；后面只有媒体/空内容。"
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


def _message(
    *,
    role: str,
    timestamp: str,
    body_text: str,
    global_user_id: str,
    platform_user_id: str,
    addressed: bool = False,
) -> dict[str, Any]:
    """Build one activity-window source row."""

    message = {
        "role": role,
        "timestamp": timestamp,
        "body_text": body_text,
        "display_name": role,
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
