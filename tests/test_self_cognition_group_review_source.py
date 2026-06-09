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
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    build_group_activity_windows,
)
from kazusa_ai_chatbot.self_cognition import models, projection, runner, sources


@pytest.fixture(autouse=True)
def _skip_default_scene_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep source-collector tests off the real digest LLM by default."""

    async def no_scene_digest(**kwargs: Any) -> None:
        del kwargs
        return None

    monkeypatch.setattr(
        sources,
        "build_group_scene_digest",
        no_scene_digest,
        raising=False,
    )


@pytest.mark.asyncio
async def test_collect_group_chat_review_cases_builds_same_group_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-empty group windows should become bounded self-cognition cases."""

    now = datetime(2026, 5, 18, 4, 45, tzinfo=timezone.utc)

    async def build_participant_context(**kwargs: Any) -> None:
        del kwargs
        return None

    monkeypatch.setattr(
        sources,
        "build_group_review_participant_context",
        build_participant_context,
    )

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
async def test_group_review_case_profile_uses_configured_character_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Projected group-review cases should carry configured character identity."""

    now = datetime(2026, 5, 18, 4, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(
        sources,
        "CHARACTER_GLOBAL_USER_ID",
        "configured-character",
    )

    async def build_participant_context(**kwargs: Any) -> None:
        del kwargs
        return None

    monkeypatch.setattr(
        sources,
        "build_group_review_participant_context",
        build_participant_context,
    )

    async def collect_inputs(**kwargs: Any) -> ReflectionInputSet:
        assert kwargs == {
            "lookback_hours": 3,
            "now": now,
            "allow_fallback": False,
        }
        return _input_set([_group_scope()])

    cases = await sources.collect_group_chat_review_cases(
        now=now,
        character_profile={
            "name": "Character",
            "mood": "focused",
            "platform_bot_id": "bot-1",
        },
        max_cases=1,
        collect_reflection_inputs_func=collect_inputs,
    )

    assert cases
    for case in cases:
        assert case["character_profile"]["global_user_id"] == (
            "configured-character"
        )
    assert cases[0]["character_profile"]["global_user_id"] == (
        "configured-character"
    )


@pytest.mark.asyncio
async def test_group_review_case_profile_preserves_profile_character_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Projected group-review cases should preserve explicit profile identity."""

    now = datetime(2026, 5, 18, 4, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(
        sources,
        "CHARACTER_GLOBAL_USER_ID",
        "configured-character",
    )

    async def build_participant_context(**kwargs: Any) -> None:
        del kwargs
        return None

    monkeypatch.setattr(
        sources,
        "build_group_review_participant_context",
        build_participant_context,
    )

    async def collect_inputs(**kwargs: Any) -> ReflectionInputSet:
        assert kwargs == {
            "lookback_hours": 3,
            "now": now,
            "allow_fallback": False,
        }
        return _input_set([_group_scope()])

    cases = await sources.collect_group_chat_review_cases(
        now=now,
        character_profile={
            "name": "Character",
            "global_user_id": "profile-character",
            "mood": "focused",
            "platform_bot_id": "bot-1",
        },
        max_cases=1,
        collect_reflection_inputs_func=collect_inputs,
    )

    assert cases
    assert cases[0]["character_profile"]["global_user_id"] == (
        "profile-character"
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


def test_group_activity_window_projects_internal_participant_rows() -> None:
    """Participant rows should carry ids without changing visible context."""

    now = datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc)
    scope = ReflectionScopeInput(
        scope_ref="scope_group",
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
        assistant_message_count=0,
        user_message_count=2,
        total_message_count=2,
        first_timestamp="2026-05-18T04:01:00+00:00",
        last_timestamp="2026-05-18T04:02:00+00:00",
        messages=[
            _message(
                role="user",
                timestamp="2026-05-18T04:01:00+00:00",
                body_text="Can you answer this?",
                global_user_id="user-1",
                platform_user_id="qq-user-1",
                addressed=True,
            ),
            _message(
                role="user",
                timestamp="2026-05-18T04:02:00+00:00",
                body_text="Replying to the bot.",
                global_user_id="user-2",
                platform_user_id="qq-user-2",
                reply_context={
                    "reply_to_platform_user_id": "bot-1",
                    "reply_to_message_id": "assistant-msg-1",
                },
            ),
        ],
    )

    windows = build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc),
        now=now,
        character_global_user_id="character-global",
        platform_bot_id="bot-1",
    )

    assert len(windows) == 1
    window = windows[0]
    assert [row["global_user_id"] for row in window.participant_rows] == [
        "user-1",
        "user-2",
    ]
    assert window.participant_rows[0]["is_directed_at_character"] is False
    assert window.participant_rows[0]["addressed_to_global_user_ids"] == [
        "character-global",
    ]
    assert window.participant_rows[1]["reply_context"] == {
        "reply_to_platform_user_id": "bot-1",
        "reply_to_message_id": "assistant-msg-1",
    }
    assert "global_user_id" not in window.visible_context[0]
    assert "platform_user_id" not in window.visible_context[0]
    assert window.visible_context[0]["body_text"] == "Can you answer this?"


@pytest.mark.asyncio
async def test_collect_group_review_cases_attaches_participant_context(
) -> None:
    """Source collection should attach participant context before cognition."""

    now = datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc)
    windows = build_group_activity_windows(
        scope=_group_scope(),
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 30, tzinfo=timezone.utc),
        now=now,
        character_global_user_id="character-global",
        platform_bot_id="bot-1",
    )
    captured_builder_payload: dict[str, Any] = {}

    async def participant_context_builder(**kwargs: Any) -> dict[str, Any]:
        captured_builder_payload.update(kwargs)
        context = {
            "source": "group_review_participant_context",
            "context_shape": "single_flow_focus",
            "focus_mode": "direct_reply",
            "guidance": "reply to the primary target only",
            "primary_reply_target": {
                "display_name": "user",
                "reply_target_fit": "high",
                "role_in_window": ["direct_cue"],
                "relationship_label": "Neutral",
                "relationship_band": "neutral",
                "last_relationship_insight": "",
                "engagement_guidelines": [],
                "nearby_conversation_evidence": [],
                "visible_samples": ["Can you look at this in the group?"],
            },
            "background_flow": {
                "mode": "ambient_group",
                "summary": "one other row is visible",
                "participant_count_label": "few",
            },
        }
        return context

    cases = await sources.collect_group_review_cases(
        now=now,
        character_profile={
            "name": "Character",
            "global_user_id": "character-global",
            "platform_bot_id": "bot-1",
        },
        windows=windows,
        max_cases=1,
        participant_context_builder=participant_context_builder,
    )

    assert len(cases) == 1
    case = cases[0]
    assert captured_builder_payload["target_scope"]["user_id"] is None
    assert captured_builder_payload["window_start_utc"] == (
        "2026-05-18T04:00:00+00:00"
    )
    assert captured_builder_payload["participant_rows"][0][
        "global_user_id"
    ] == "user-1"
    assert case["conversation_progress"]["participant_context"]["source"] == (
        "group_review_participant_context"
    )
    assert case["target_scope"]["user_id"] is None
    assert case["delivery_target"]["platform_channel_id"] == "group-1"

    source_packet = projection.build_source_packet(case)
    serialized_packet = json.dumps(source_packet, ensure_ascii=False)
    assert "participant_context" in serialized_packet
    assert "delivery_target" not in serialized_packet
    assert "user-1" not in serialized_packet


@pytest.mark.asyncio
async def test_collect_group_review_cases_attaches_scene_digest(
) -> None:
    """Valid scene digest should reach the rendered source packet."""

    now = datetime(2026, 5, 18, 4, 15, tzinfo=timezone.utc)
    windows = build_group_activity_windows(
        scope=_group_scope(),
        window_start=datetime(2026, 5, 18, 4, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 18, 4, 30, tzinfo=timezone.utc),
        now=now,
        character_global_user_id="character-global",
        platform_bot_id="bot-1",
    )
    captured_window = None

    async def participant_context_builder(**kwargs: Any) -> None:
        del kwargs
        return None

    async def scene_digest_builder(**kwargs: Any) -> dict[str, str]:
        nonlocal captured_window
        captured_window = kwargs["window"]
        digest = {
            "digest": (
                "这段群聊里，participant_1 问了我一个问题，"
                "我已经在窗口里接过一次。"
            ),
        }
        return digest

    cases = await sources.collect_group_review_cases(
        now=now,
        character_profile={
            "name": "Character",
            "global_user_id": "character-global",
            "platform_bot_id": "bot-1",
        },
        windows=windows,
        max_cases=1,
        participant_context_builder=participant_context_builder,
        scene_digest_builder=scene_digest_builder,
    )

    assert len(cases) == 1
    assert captured_window == windows[0]
    case = cases[0]
    assert case["conversation_progress"]["group_scene_digest"] == {
        "digest": (
            "这段群聊里，participant_1 问了我一个问题，"
            "我已经在窗口里接过一次。"
        ),
    }

    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    serialized_packet = json.dumps(source_packet, ensure_ascii=False)

    assert "group_scene_digest" in rendered_packet
    assert "participant_1 问了我一个问题" in rendered_packet
    assert "delivery_target" not in serialized_packet
    assert "user-1" not in serialized_packet
    assert "qq-user-1" not in serialized_packet
    assert "bot-1" not in serialized_packet


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
    reply_context: dict[str, Any] | None = None,
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
        "is_directed_at_character": False,
        "reply_context": dict(reply_context or {}),
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
