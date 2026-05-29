"""Focused tests for group-review participant context source hydration."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.self_cognition import (
    group_review_participant_context as context_module,
)


class RecordingConversationAgent:
    """Record bounded conversation-evidence calls from the context builder."""

    def __init__(self, evidence: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.evidence = list(evidence or [])

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Return a deterministic conversation-evidence helper result."""

        self.calls.append(
            {
                "task": task,
                "context": context,
                "max_attempts": max_attempts,
            },
        )
        result = {
            "resolved": bool(self.evidence),
            "result": {
                "evidence": list(self.evidence),
                "selected_summary": "\n".join(self.evidence),
            },
        }
        return result


@pytest.mark.asyncio
async def test_direct_cue_selects_one_primary_social_beat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct cue should hydrate one primary target, not a roster."""

    profile_calls: list[str] = []
    engagement_calls: list[str] = []

    async def get_user_profile(global_user_id: str) -> dict[str, Any]:
        profile_calls.append(global_user_id)
        profile = {
            "global_user_id": global_user_id,
            "display_name": "Profile Bob",
            "affinity": 700,
            "last_relationship_insight": "Bob tests boundaries playfully.",
        }
        return profile

    async def build_user_engagement_relevance_context(
        global_user_id: str,
    ) -> dict[str, Any]:
        engagement_calls.append(global_user_id)
        context = {
            "engagement_guidelines": ["Keep it brief and do not fan out."],
            "confidence": "high",
        }
        return context

    monkeypatch.setattr(
        context_module,
        "get_user_profile",
        get_user_profile,
    )
    monkeypatch.setattr(
        context_module,
        "build_user_engagement_relevance_context",
        build_user_engagement_relevance_context,
    )
    conversation_agent = RecordingConversationAgent(
        evidence=[
            "Bob: earlier boundary tease",
            "Character: short rebuttal",
            "Bob: follow-up joke",
            "Bob: extra row beyond cap",
        ],
    )

    result = await context_module.build_group_review_participant_context(
        participant_rows=[
            _participant_row(
                display_name="Alice",
                global_user_id="user-a",
                platform_user_id="qq-a",
                body_text="Side thread detail.",
                timestamp="2026-05-18T04:01:00+00:00",
            ),
            _participant_row(
                display_name="Bob",
                global_user_id="user-b",
                platform_user_id="qq-b",
                body_text="You crossed the line here.",
                timestamp="2026-05-18T04:02:00+00:00",
                addressed_to_global_user_ids=["character-global"],
            ),
            _participant_row(
                display_name="Alice",
                global_user_id="user-a",
                platform_user_id="qq-a",
                body_text="Another side comment.",
                timestamp="2026-05-18T04:03:00+00:00",
            ),
        ],
        target_scope=_target_scope(),
        character_profile={
            "global_user_id": "character-global",
            "platform_bot_id": "bot-1",
        },
        window_start_utc="2026-05-18T04:00:00+00:00",
        current_timestamp_utc="2026-05-18T04:15:00+00:00",
        conversation_agent=conversation_agent,
    )

    assert result is not None
    assert result["source"] == "group_review_participant_context"
    assert result["context_shape"] == "single_flow_focus"
    assert result["focus_mode"] == "direct_reply"
    assert "participants" not in result
    assert "selected_reply_target" not in result

    primary = result["primary_reply_target"]
    assert primary["display_name"] == "Bob"
    assert primary["reply_target_fit"] == "high"
    assert primary["relationship_label"] == "Warm"
    assert primary["relationship_band"] == "positive"
    assert primary["last_relationship_insight"] == (
        "Bob tests boundaries playfully."
    )
    assert primary["engagement_guidelines"] == [
        "Keep it brief and do not fan out.",
    ]
    assert primary["nearby_conversation_evidence"] == [
        "Bob: earlier boundary tease",
        "Character: short rebuttal",
        "Bob: follow-up joke",
    ]
    assert primary["visible_samples"] == ["You crossed the line here."]
    assert "direct_cue" in primary["role_in_window"]
    assert result["background_flow"]["participant_count_label"] == "few"

    assert profile_calls == ["user-b"]
    assert engagement_calls == ["user-b"]
    assert len(conversation_agent.calls) == 1
    assert conversation_agent.calls[0]["max_attempts"] == 1
    assert conversation_agent.calls[0]["context"]["platform_channel_id"] == (
        "group-1"
    )


@pytest.mark.asyncio
async def test_missing_global_user_id_degrades_to_visible_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing global user id should not trigger profile or conversation lookup."""

    async def fail_get_user_profile(global_user_id: str) -> dict[str, Any]:
        raise AssertionError(f"profile lookup should not run: {global_user_id}")

    async def fail_build_user_engagement_relevance_context(
        global_user_id: str,
    ) -> dict[str, Any]:
        raise AssertionError(
            f"engagement lookup should not run: {global_user_id}",
        )

    monkeypatch.setattr(
        context_module,
        "get_user_profile",
        fail_get_user_profile,
    )
    monkeypatch.setattr(
        context_module,
        "build_user_engagement_relevance_context",
        fail_build_user_engagement_relevance_context,
    )
    conversation_agent = RecordingConversationAgent(evidence=["unused"])

    result = await context_module.build_group_review_participant_context(
        participant_rows=[
            _participant_row(
                display_name="Display Only",
                global_user_id="",
                platform_user_id="qq-display",
                body_text="A visible-only comment.",
                timestamp="2026-05-18T04:01:00+00:00",
            ),
        ],
        target_scope=_target_scope(),
        character_profile={"global_user_id": "character-global"},
        window_start_utc="2026-05-18T04:00:00+00:00",
        current_timestamp_utc="2026-05-18T04:15:00+00:00",
        conversation_agent=conversation_agent,
    )

    assert result is not None
    primary = result["primary_reply_target"]
    assert primary["display_name"] == "Display Only"
    assert primary["relationship_label"] == "unknown"
    assert primary["relationship_band"] == "unknown"
    assert primary["last_relationship_insight"] == ""
    assert primary["engagement_guidelines"] == []
    assert primary["nearby_conversation_evidence"] == []
    assert primary["visible_samples"] == ["A visible-only comment."]
    assert conversation_agent.calls == []


@pytest.mark.asyncio
async def test_group_pile_on_collapses_high_fit_participants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple direct cues should become one shared-flow focus."""

    async def get_user_profile(global_user_id: str) -> dict[str, Any]:
        profile = {
            "global_user_id": global_user_id,
            "display_name": global_user_id,
            "affinity": 500,
            "last_relationship_insight": "",
        }
        return profile

    async def build_user_engagement_relevance_context(
        global_user_id: str,
    ) -> dict[str, Any]:
        del global_user_id
        context = {"engagement_guidelines": [], "confidence": ""}
        return context

    monkeypatch.setattr(
        context_module,
        "get_user_profile",
        get_user_profile,
    )
    monkeypatch.setattr(
        context_module,
        "build_user_engagement_relevance_context",
        build_user_engagement_relevance_context,
    )

    result = await context_module.build_group_review_participant_context(
        participant_rows=[
            _participant_row(
                display_name="Alice",
                global_user_id="user-a",
                platform_user_id="qq-a",
                body_text="Character, explain this.",
                timestamp="2026-05-18T04:01:00+00:00",
                addressed_to_global_user_ids=["character-global"],
            ),
            _participant_row(
                display_name="Bob",
                global_user_id="user-b",
                platform_user_id="qq-b",
                body_text="Yes, why did you say that?",
                timestamp="2026-05-18T04:02:00+00:00",
                addressed_to_global_user_ids=["character-global"],
            ),
        ],
        target_scope=_target_scope(),
        character_profile={"global_user_id": "character-global"},
        window_start_utc="2026-05-18T04:00:00+00:00",
        current_timestamp_utc="2026-05-18T04:15:00+00:00",
        conversation_agent=RecordingConversationAgent(),
    )

    assert result is not None
    assert result["focus_mode"] == "group_pile_on"
    assert result["primary_reply_target"]["display_name"] == "Bob"
    assert result["background_flow"]["mode"] == "multi_person_pile_on"
    assert "participants" not in result
    assert "selected_reply_target" not in result
    assert "one by one" in result["guidance"]


@pytest.mark.asyncio
async def test_caps_visible_samples_and_conversation_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt-facing participant context should cap samples and evidence."""

    async def get_user_profile(global_user_id: str) -> dict[str, Any]:
        profile = {
            "global_user_id": global_user_id,
            "display_name": "Topic Author",
            "affinity": 500,
            "last_relationship_insight": "stable",
        }
        return profile

    async def build_user_engagement_relevance_context(
        global_user_id: str,
    ) -> dict[str, Any]:
        del global_user_id
        context = {"engagement_guidelines": [], "confidence": ""}
        return context

    monkeypatch.setattr(
        context_module,
        "get_user_profile",
        get_user_profile,
    )
    monkeypatch.setattr(
        context_module,
        "build_user_engagement_relevance_context",
        build_user_engagement_relevance_context,
    )
    long_text = "x" * 240
    rows = [
        _participant_row(
            display_name="Topic Author",
            global_user_id="user-topic",
            platform_user_id="qq-topic",
            body_text=f"{long_text}-{index}",
            timestamp=f"2026-05-18T04:0{index}:00+00:00",
        )
        for index in range(1, 5)
    ]
    conversation_agent = RecordingConversationAgent(
        evidence=[
            "evidence 1",
            "evidence 2",
            "evidence 3",
            "evidence 4",
        ],
    )

    result = await context_module.build_group_review_participant_context(
        participant_rows=rows,
        target_scope=_target_scope(),
        character_profile={"global_user_id": "character-global"},
        window_start_utc="2026-05-18T04:00:00+00:00",
        current_timestamp_utc="2026-05-18T04:15:00+00:00",
        conversation_agent=conversation_agent,
    )

    assert result is not None
    primary = result["primary_reply_target"]
    assert primary["nearby_conversation_evidence"] == [
        "evidence 1",
        "evidence 2",
        "evidence 3",
    ]
    assert len(primary["visible_samples"]) == 3
    assert all(len(sample) <= 160 for sample in primary["visible_samples"])
    assert len(conversation_agent.calls) == 1


@pytest.mark.asyncio
async def test_returns_none_when_no_user_rows() -> None:
    """Assistant-only windows should not attach participant context."""

    result = await context_module.build_group_review_participant_context(
        participant_rows=[
            _participant_row(
                role="assistant",
                display_name="Character",
                global_user_id="character-global",
                platform_user_id="bot-1",
                body_text="Assistant row.",
                timestamp="2026-05-18T04:01:00+00:00",
            ),
        ],
        target_scope=_target_scope(),
        character_profile={"global_user_id": "character-global"},
        window_start_utc="2026-05-18T04:00:00+00:00",
        current_timestamp_utc="2026-05-18T04:15:00+00:00",
        conversation_agent=RecordingConversationAgent(),
    )

    assert result is None


def _participant_row(
    *,
    display_name: str,
    global_user_id: str,
    platform_user_id: str,
    body_text: str,
    timestamp: str,
    role: str = "user",
    platform_message_id: str = "",
    addressed_to_global_user_ids: list[str] | None = None,
    mentions: list[dict[str, str]] | None = None,
    is_directed_at_character: bool = False,
    reply_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one participant-row input for context-builder tests."""

    message_id = platform_message_id
    if not message_id:
        message_id = f"msg-{timestamp}"
    row = {
        "timestamp": timestamp,
        "role": role,
        "display_name": display_name,
        "body_text": body_text,
        "platform_message_id": message_id,
        "global_user_id": global_user_id,
        "platform_user_id": platform_user_id,
        "addressed_to_global_user_ids": list(
            addressed_to_global_user_ids or [],
        ),
        "mentions": list(mentions or []),
        "is_directed_at_character": is_directed_at_character,
        "reply_context": dict(reply_context or {}),
    }
    return row


def _target_scope() -> dict[str, str | None]:
    """Build the group target scope used by focused participant tests."""

    target_scope = {
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "user_id": None,
    }
    return target_scope
