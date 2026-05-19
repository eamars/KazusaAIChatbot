"""Tests for reflection-attached group self-cognition source collection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from kazusa_ai_chatbot.reflection_cycle.models import (
    ReflectionInputSet,
    ReflectionScopeInput,
)
from kazusa_ai_chatbot.self_cognition import models, projection, runner, sources


@pytest.mark.asyncio
async def test_collect_group_chat_review_cases_builds_same_group_cases(
) -> None:
    """Non-empty group windows should become bounded self-cognition cases."""

    now = datetime(2026, 5, 18, 4, 45, tzinfo=timezone.utc)

    async def collect_inputs(**kwargs: Any) -> ReflectionInputSet:
        assert kwargs == {
            "lookback_hours": 3,
            "now": now,
            "allow_fallback": False,
        }
        return _input_set([
            _group_scope(),
            _private_scope(),
        ])

    cases = await sources.collect_group_chat_review_cases(
        now=now,
        character_profile={
            "name": "Character",
            "global_user_id": "character-global",
            "platform_bot_id": "bot-1",
            "mood": "focused",
            "global_vibe": "steady",
        },
        max_cases=3,
        collect_reflection_inputs_func=collect_inputs,
    )

    assert len(cases) == 2
    newest_case = cases[0]
    assert newest_case["case_name"] == models.CASE_GROUP_CHAT_REVIEW
    assert newest_case["trigger_kind"] == models.TRIGGER_GROUP_CHAT_REVIEW
    assert newest_case["actionability"] == (
        "active_group_review_same_channel_no_fallback"
    )
    assert newest_case["target_scope"] == {
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "user_id": None,
    }
    assert newest_case["delivery_target"]["platform_channel_id"] == "group-1"
    assert newest_case["delivery_target"]["channel_type"] == "group"
    assert newest_case["delivery_target"]["source_kind"] == (
        "self_cognition_source_channel"
    )
    assert newest_case["delivery_target"]["fallback_reason"] == ""
    assert newest_case["current_mood"] == "focused"
    assert newest_case["global_vibe"] == "steady"
    addressed_case = next(
        case
        for case in cases
        if case["conversation_progress"]["window_start"]
        == "2026-05-18T04:00:00+00:00"
    )
    assert addressed_case["source_refs"][0]["source_kind"] == (
        "reflection_activity_window"
    )
    assert addressed_case["source_refs"][0]["source_id"] == (
        "scope_group:"
        "2026-05-18T04:00:00+00:00:"
        "2026-05-18T04:15:00+00:00"
    )
    assert addressed_case["conversation_progress"]["source"] == (
        "reflection_activity_window"
    )
    assert addressed_case["group_activity_window"]["source"] == (
        "reflection_activity_window"
    )
    assert addressed_case["group_activity_window"]["semantic_labels"][
        "bot_addressing"
    ] == "directly_addressed"
    assert addressed_case["conversation_progress"]["activity_labels"][
        "bot_addressing"
    ] == "directly_addressed"
    assert len(addressed_case["visible_context"]) == 2
    assert all(
        case["target_scope"]["channel_type"] == "group"
        for case in cases
    )


@pytest.mark.asyncio
async def test_collect_group_chat_review_cases_skips_empty_windows() -> None:
    """Empty 15-minute periods should not call self-cognition."""

    async def collect_inputs(**kwargs: Any) -> ReflectionInputSet:
        del kwargs
        return _input_set([
            ReflectionScopeInput(
                scope_ref="scope_group",
                platform="qq",
                platform_channel_id="group-1",
                channel_type="group",
                assistant_message_count=1,
                user_message_count=0,
                total_message_count=1,
                first_timestamp="2026-05-18T04:31:00+00:00",
                last_timestamp="2026-05-18T04:31:00+00:00",
                messages=[
                    _message(
                        role="assistant",
                        timestamp="2026-05-18T04:31:00+00:00",
                        body_text="Only one active window.",
                        global_user_id="character-global",
                        platform_user_id="bot-1",
                    )
                ],
            )
        ])

    cases = await sources.collect_group_chat_review_cases(
        now=datetime(2026, 5, 18, 4, 45, tzinfo=timezone.utc),
        character_profile={"name": "Character"},
        max_cases=12,
        collect_reflection_inputs_func=collect_inputs,
    )

    assert len(cases) == 1
    assert cases[0]["conversation_progress"]["window_start"] == (
        "2026-05-18T04:30:00+00:00"
    )


@pytest.mark.asyncio
async def test_collect_group_chat_review_cases_prefers_newest_windows(
) -> None:
    """Bounded group review should process current windows before stale ones."""

    messages = [
        _message(
            "user",
            f"2026-05-18T04:{minute:02d}:00+00:00",
            f"window {minute}",
        )
        for minute in (1, 16, 31, 46)
    ]
    scope = ReflectionScopeInput(
        scope_ref="scope_group",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        assistant_message_count=0,
        user_message_count=len(messages),
        total_message_count=len(messages),
        first_timestamp="2026-05-18T04:01:00+00:00",
        last_timestamp="2026-05-18T04:46:00+00:00",
        messages=messages,
    )

    async def collect_inputs(**kwargs: Any) -> ReflectionInputSet:
        del kwargs
        return ReflectionInputSet(
            lookback_hours=3,
            requested_start="2026-05-18T02:00:00+00:00",
            requested_end="2026-05-18T05:00:00+00:00",
            effective_start="2026-05-18T02:00:00+00:00",
            effective_end="2026-05-18T05:00:00+00:00",
            fallback_used=False,
            fallback_reason="",
            selected_scopes=[scope],
            query_diagnostics={},
        )

    cases = await sources.collect_group_chat_review_cases(
        now=datetime(2026, 5, 18, 5, 0, tzinfo=timezone.utc),
        character_profile={"name": "Character"},
        max_cases=3,
        collect_reflection_inputs_func=collect_inputs,
    )

    window_starts = [
        case["conversation_progress"]["window_start"]
        for case in cases
    ]
    assert window_starts == [
        "2026-05-18T04:45:00+00:00",
        "2026-05-18T04:30:00+00:00",
        "2026-05-18T04:15:00+00:00",
    ]


@pytest.mark.asyncio
async def test_collect_self_cognition_cases_does_not_schedule_group_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standalone self-cognition worker must not own group cadence."""

    async def no_scheduled(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []

    async def no_commitments(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []

    async def group_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        raise AssertionError("group review belongs to reflection cadence")

    monkeypatch.setattr(
        sources,
        "collect_scheduled_future_cognition_cases",
        no_scheduled,
    )
    monkeypatch.setattr(sources, "collect_active_commitment_cases", no_commitments)
    monkeypatch.setattr(sources, "collect_group_chat_review_cases", group_cases)

    cases = await sources.collect_self_cognition_cases(
        now=datetime(2026, 5, 18, 4, 45, tzinfo=timezone.utc),
        character_profile={"name": "Character"},
        max_cases=3,
    )

    assert cases == []


def test_group_review_source_packet_uses_active_group_review_contract() -> None:
    """Group review should use first-person chat-window data framing."""

    case = _group_review_case()

    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    serialized_packet = json.dumps(source_packet, ensure_ascii=False)

    assert source_packet["instruction"] != models.SELF_COGNITION_INPUT_TEXT
    assert source_packet["instruction"] == (
        "我刚看到群里刚刚发生的一段现场。里面有人把话题指向我。"
    )
    assert rendered_packet.startswith(
        "我刚看到群里刚刚发生的一段现场。里面有人把话题指向我。"
        "\n\n# 当前聊天窗口"
    )
    _assert_no_group_source_trigger_labels(rendered_packet)
    assert (
        "我刚看到群里刚刚发生的一段现场。我之前没有插话，"
        "这段里也没有人把话题交给我。"
    ) not in rendered_packet
    assert "group_chat_trigger_review" in serialized_packet
    assert "active_group_review_same_channel_no_fallback" in serialized_packet
    assert "group_chat_trigger_review" not in rendered_packet
    assert "active_group_review_same_channel_no_fallback" not in rendered_packet
    assert "group_activity_window" in serialized_packet
    assert "semantic_labels" in serialized_packet
    assert "# 群聊窗口信息" in rendered_packet
    assert "当前自检" not in rendered_packet
    assert "自然路线" not in rendered_packet
    assert "delivery_target" not in serialized_packet
    assert "dm-" not in serialized_packet
    assert "self_cognition_delivery_target" not in serialized_packet


def test_group_review_source_packet_uses_ambient_sentence_when_not_addressed(
) -> None:
    """Ambient group review should not imply anyone addressed the character."""

    case = _group_review_case()
    case["conversation_progress"]["activity_labels"]["bot_addressing"] = (
        "ambient_group_context"
    )
    case["group_activity_window"]["semantic_labels"]["bot_addressing"] = (
        "ambient_group_context"
    )

    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)

    assert source_packet["instruction"] == (
        "我刚看到群里刚刚发生的一段现场。我之前没有插话，"
        "这段里也没有人把话题交给我。"
    )
    assert rendered_packet.startswith(
        "我刚看到群里刚刚发生的一段现场。我之前没有插话，"
        "这段里也没有人把话题交给我。"
        "\n\n# 当前聊天窗口"
    )
    _assert_no_group_source_trigger_labels(rendered_packet)
    assert (
        "我刚看到群里刚刚发生的一段现场。里面有人把话题指向我。"
        not in rendered_packet
    )


def test_group_review_cognition_state_does_not_invent_target_user() -> None:
    """Channel-scoped group review should not fabricate a semantic target."""

    case = _group_review_case()

    cognition_state = runner._build_cognition_state(
        case,
        "rendered group source packet",
    )
    episode_target_scope = cognition_state["cognitive_episode"]["target_scope"]

    assert cognition_state["global_user_id"] == ""
    assert cognition_state["platform_user_id"] == ""
    assert cognition_state["user_name"] == "group audience"
    assert episode_target_scope["current_global_user_id"] == ""
    assert episode_target_scope["current_platform_user_id"] == ""
    assert episode_target_scope["current_display_name"] == "group audience"
    assert episode_target_scope["target_addressed_user_ids"] == []


def test_non_group_source_packet_omits_group_window_section() -> None:
    """Existing self-cognition sources should not receive empty group framing."""

    case = {
        "case_name": models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "commitment:unit-1",
        "idle_timestamp_utc": "2026-05-18T04:45:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-18T04:05:00+00:00",
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": models.DUE_STATE_PAST_DUE,
        "actionability": "contact_is_socially_available",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "dm-1",
            "channel_type": "private",
            "user_id": "user-1",
        },
        "source_refs": [
            {
                "source_kind": "future_promise",
                "source_id": "unit-1",
                "due_at": "2026-05-18T04:00:00+00:00",
                "summary": "A commitment became due.",
            }
        ],
        "visible_context": [],
    }

    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    serialized_packet = json.dumps(source_packet, ensure_ascii=False)

    assert "group_activity_window" not in serialized_packet
    assert "# 群聊窗口信息" not in rendered_packet


def _input_set(scopes: list[ReflectionScopeInput]) -> ReflectionInputSet:
    """Build a reflection input set for group-review source tests."""

    input_set = ReflectionInputSet(
        lookback_hours=3,
        requested_start="2026-05-18T03:45:00+00:00",
        requested_end="2026-05-18T04:45:00+00:00",
        effective_start="2026-05-18T03:45:00+00:00",
        effective_end="2026-05-18T04:45:00+00:00",
        fallback_used=False,
        fallback_reason="",
        selected_scopes=scopes,
        query_diagnostics={},
    )
    return input_set


def _assert_no_group_source_trigger_labels(rendered_packet: str) -> None:
    """Assert group-review source text has no label or action-pressure cue."""

    for forbidden in (
        "来源位置：",
        "出现原因：",
        "数据身份：",
        "进入注意的原因：",
        "阅读方式",
        "自检",
        "需要接上",
    ):
        assert forbidden not in rendered_packet


def _group_scope() -> ReflectionScopeInput:
    """Build one monitored group with two non-empty windows."""

    messages = [
        _message(
            role="user",
            timestamp="2026-05-18T04:01:00+00:00",
            body_text="Can you look at this in the group?",
            global_user_id="user-1",
            platform_user_id="qq-user-1",
            addressed=True,
        ),
        _message(
            role="assistant",
            timestamp="2026-05-18T04:05:00+00:00",
            body_text="I can help.",
            global_user_id="character-global",
            platform_user_id="bot-1",
            platform_message_id="assistant-msg-1",
        ),
        _message(
            role="user",
            timestamp="2026-05-18T04:18:00+00:00",
            body_text="Second window.",
            global_user_id="user-2",
            platform_user_id="qq-user-2",
        ),
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
        last_timestamp="2026-05-18T04:18:00+00:00",
        messages=messages,
    )
    return scope


def _private_scope() -> ReflectionScopeInput:
    """Build a private scope that group review must ignore."""

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
    return scope


def _message(
    role: str,
    timestamp: str,
    body_text: str,
    global_user_id: str = "",
    platform_user_id: str = "",
    platform_message_id: str = "",
    addressed: bool = False,
) -> dict[str, Any]:
    """Build one conversation message row."""

    message_id = platform_message_id
    if not message_id:
        message_id = "msg-" + timestamp.replace("-", "").replace(":", "")
    message = {
        "role": role,
        "timestamp": timestamp,
        "body_text": body_text,
        "display_name": role,
        "global_user_id": global_user_id,
        "platform_user_id": platform_user_id,
        "platform_message_id": message_id,
        "addressed_to_global_user_ids": (
            ["character-global"] if addressed else []
        ),
        "mentions": [],
    }
    return message


def _group_review_case() -> dict[str, Any]:
    """Build one group review case for projection contract tests."""

    case = {
        "case_name": models.CASE_GROUP_CHAT_REVIEW,
        "case_id": (
            "group_activity_window:"
            "scope_group:"
            "2026-05-18T04:00:00+00:00:"
            "2026-05-18T04:15:00+00:00"
        ),
        "idle_timestamp_utc": "2026-05-18T04:45:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-18T04:05:00+00:00",
        "trigger_kind": models.TRIGGER_GROUP_CHAT_REVIEW,
        "semantic_due_state": None,
        "actionability": "active_group_review_same_channel_no_fallback",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_type": "group",
            "user_id": None,
        },
        "source_refs": [
            {
                "source_kind": "reflection_activity_window",
                "source_id": (
                    "scope_group:"
                    "2026-05-18T04:00:00+00:00:"
                    "2026-05-18T04:15:00+00:00"
                ),
                "due_at": None,
                "summary": (
                    "quiet group activity, one_speaker speakers, "
                    "directly_addressed, present, risk low"
                ),
            }
        ],
        "visible_context": [],
        "conversation_progress": {
            "source": "reflection_activity_window",
            "window_start": "2026-05-18T04:00:00+00:00",
            "window_end": "2026-05-18T04:15:00+00:00",
            "activity_labels": {
                "activity_level": "quiet",
                "speaker_diversity": "one_speaker",
                "assistant_presence": "present",
                "bot_addressing": "directly_addressed",
                "message_recency": "recent",
                "noise_level": "low",
                "response_risk": "low",
            },
        },
        "group_activity_window": {
            "source": "reflection_activity_window",
            "window_start": "2026-05-18T04:00:00+00:00",
            "window_end": "2026-05-18T04:15:00+00:00",
            "semantic_labels": {
                "activity_level": "quiet",
                "bot_addressing": "directly_addressed",
            },
            "delivery_target": "dm-1",
        },
        "delivery_target": {
            "schema_version": "self_cognition_delivery_target.v1",
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_type": "group",
            "target_global_user_id": None,
            "target_platform_user_id": None,
            "source_kind": "self_cognition_source_channel",
            "source_ref": (
                "scope_group:"
                "2026-05-18T04:00:00+00:00:"
                "2026-05-18T04:15:00+00:00"
            ),
            "source_platform_channel_id": "group-1",
            "source_channel_type": "group",
            "source_message_id": "assistant-msg-1",
            "source_global_user_id": None,
            "source_platform_bot_id": "bot-1",
            "source_character_name": "Character",
            "guild_id": None,
            "bot_permission_role": "user",
            "fallback_reason": "",
        },
    }
    return case
