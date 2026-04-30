"""Tests for relevance_agent.py — relevance gate logic."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.nodes.relevance_agent import build_group_attention_context, relevance_agent


def _base_state():
    """Minimal IMProcessState for testing relevance_agent."""
    return {
        "timestamp": "2024-01-01T00:00:00Z",
        "platform": "discord",
        "platform_message_id": "msg_123",
        "platform_user_id": "user_123",
        "global_user_id": "uuid-123",
        "user_name": "TestUser",
        "user_input": "Hello bot!",
        "user_multimedia_input": [],
        "user_profile": {"affinity": 500, "last_relationship_insight": ""},
        "platform_bot_id": "bot_456",
        "message_envelope": {
            "body_text": "Hello bot!",
            "raw_wire_text": "Hello bot!",
            "addressed_to_global_user_ids": [],
            "mentions": [],
            "attachments": [],
            "broadcast": True,
        },
        "character_name": "TestCharacter",
        "character_profile": {
            "name": "Character",
            "global_user_id": "character-global-id",
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "nothing notable",
        },
        "platform_channel_id": "chan_1",
        "channel_type": "group",
        "channel_name": "general",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "debug_modes": {},
    }


def _history_row(
    *,
    role: str = "user",
    platform_user_id: str = "user_123",
    timestamp: str = "2026-04-27T10:00:00+00:00",
    reply_context: dict | None = None,
    addressed_to_global_user_ids: list[str] | None = None,
    content: str = "",
) -> dict:
    """Build a trimmed conversation-history row for relevance tests.

    Args:
        role: Conversation role.
        platform_user_id: Platform user id that produced the row.
        timestamp: ISO timestamp for active-window selection.
        reply_context: Optional structured reply metadata.
        addressed_to_global_user_ids: Typed addressee UUIDs for the row.
        content: Optional content fixture.

    Returns:
        Trimmed history row matching ``trim_history_dict`` shape.
    """
    return {
        "role": role,
        "platform_user_id": platform_user_id,
        "timestamp": timestamp,
        "reply_context": reply_context or {},
        "addressed_to_global_user_ids": addressed_to_global_user_ids or [],
        "content": content,
    }


def _llm_response(content: str) -> MagicMock:
    """Return a small mock object shaped like a LangChain response."""
    response = MagicMock()
    response.content = content
    return response


@pytest.mark.asyncio
async def test_relevance_agent_returns_should_respond():
    """LLM says should_respond=true → agent forwards that decision."""
    llm_response = _llm_response('{"should_respond": true, "reason_to_respond": "user greeted", "use_reply_feature": false, "channel_topic": "greetings", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["should_respond"] is True
    assert result["channel_topic"] == "greetings"
    assert result["indirect_speech_context"] == ""


@pytest.mark.asyncio
async def test_relevance_agent_should_not_respond():
    """LLM says should_respond=false → agent forwards that decision."""
    llm_response = _llm_response('{"should_respond": false, "reason_to_respond": "third party conversation", "use_reply_feature": false, "channel_topic": "sports", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["should_respond"] is False
    assert result["reason_to_respond"] == "third party conversation"


@pytest.mark.asyncio
async def test_relevance_agent_malformed_json_defaults_to_not_respond():
    """If LLM returns garbage JSON, parse_llm_json_output returns {} and should_respond defaults to False."""
    llm_response = _llm_response("this is not json at all")

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["should_respond"] is False
    assert result["channel_topic"] == ""
    assert result["indirect_speech_context"] == ""


@pytest.mark.asyncio
async def test_relevance_agent_use_reply_feature():
    """LLM says use_reply_feature=true → agent forwards it."""
    llm_response = _llm_response('{"should_respond": true, "reason_to_respond": "reply needed", "use_reply_feature": true, "channel_topic": "topic", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["use_reply_feature"] is True


@pytest.mark.asyncio
async def test_relevance_agent_short_circuits_structured_third_party_reply_in_group() -> None:
    """Structured reply metadata to another participant should skip the LLM in noisy channels."""
    state = _base_state()
    state["user_input"] = '[Reply to message] <@someone-else> 我同事上下班是不用加油的'
    state["reply_context"] = {
        "reply_to_message_id": "other-msg",
        "reply_to_platform_user_id": "someone-else",
    }

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock()
        result = await relevance_agent(state)

    assert result["should_respond"] is False
    assert "another participant" in result["reason_to_respond"]
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_relevance_agent_allows_structured_third_party_reply_when_bot_explicitly_addressed() -> None:
    """Explicit bot address should override the third-party reply short-circuit."""
    state = _base_state()
    state["user_input"] = "你怎么看他刚才那句？"
    state["message_envelope"]["addressed_to_global_user_ids"] = ["character-global-id"]
    state["reply_context"] = {
        "reply_to_message_id": "other-msg",
        "reply_to_platform_user_id": "someone-else",
    }
    llm_response = _llm_response('{"should_respond": true, "reason_to_respond": "bot explicitly addressed", "use_reply_feature": true, "channel_topic": "discussion", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(state)

    assert result["should_respond"] is True
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_relevance_agent_text_only_reply_marker_does_not_block_without_metadata() -> None:
    """Text-only reply markers should not drive deterministic relevance gating."""
    state = _base_state()
    state["user_input"] = '[Reply to message] <@someone-else> 我同事上下班是不用加油的'
    llm_response = _llm_response('{"should_respond": false, "reason_to_respond": "model decision", "use_reply_feature": true, "channel_topic": "", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(state)

    assert result["should_respond"] is False
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_relevance_agent_does_not_expose_reflection_summary_in_prompt() -> None:
    """Global reflection summaries should not be injected into relevance reasoning."""
    llm_response = _llm_response('{"should_respond": true, "reason_to_respond": "user greeted", "use_reply_feature": false, "channel_topic": "greetings", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        await relevance_agent(_base_state())

    rendered_prompt = mock_llm.ainvoke.await_args.args[0][0].content
    assert "自我反思" not in rendered_prompt
    assert "nothing notable" not in rendered_prompt


@pytest.mark.asyncio
async def test_noisy_relevance_prompt_preserves_metadata_first_contract() -> None:
    """Group prompt should not reintroduce text-only address inference."""
    llm_response = _llm_response('{"should_respond": true, "reason_to_respond": "metadata", "use_reply_feature": true, "channel_topic": "", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        await relevance_agent(_base_state())

    rendered_prompt = mock_llm.ainvoke.await_args.args[0][0].content
    assert '高噪音群聊环境' not in rendered_prompt
    assert '消息明确且**仅**包含你的 ID' not in rendered_prompt
    assert '必须始终设置为 `true`' not in rendered_prompt
    assert '不能当作平台结构化指向证据' in rendered_prompt
    assert '不要仅凭第二人称代词判断实际听众' in rendered_prompt
    assert '"use_reply_feature": boolean' in rendered_prompt


def test_group_attention_low_noise_when_recent_direct_address_exists() -> None:
    """A recent structural bot address should lower group attention pressure."""
    history = [
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:00+00:00"),
        _history_row(
            platform_user_id="user_1",
            timestamp="2026-04-27T10:00:10+00:00",
            addressed_to_global_user_ids=["character-global-id"],
        ),
    ]

    result = build_group_attention_context(
        chat_history_wide=history,
        platform_bot_id="bot_456",
        character_global_user_id="character-global-id",
    )

    assert result == {"group_attention": "low_noise"}


def test_group_attention_chaotic_for_two_speaker_reply_thread_collision() -> None:
    """Two fast non-bot speakers plus a reply-to-other thread is chaotic."""
    history = [
        _history_row(
            platform_user_id="user_1",
            timestamp="2026-04-27T10:18:26+00:00",
        ),
        _history_row(
            platform_user_id="user_2",
            timestamp="2026-04-27T10:19:08+00:00",
            reply_context={
                "reply_to_platform_user_id": "user_1",
            },
        ),
        _history_row(
            role="assistant",
            platform_user_id="bot_456",
            timestamp="2026-04-27T10:19:08+00:00",
        ),
    ]

    result = build_group_attention_context(chat_history_wide=history, platform_bot_id="bot_456")

    assert result == {"group_attention": "chaotic_noise"}


def test_group_attention_chaotic_for_many_unaddressed_speakers() -> None:
    """Three speakers and four unaddressed messages should be chaotic."""
    history = [
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:00+00:00"),
        _history_row(platform_user_id="user_2", timestamp="2026-04-27T10:00:10+00:00"),
        _history_row(platform_user_id="user_3", timestamp="2026-04-27T10:00:20+00:00"),
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:30+00:00"),
    ]

    result = build_group_attention_context(chat_history_wide=history, platform_bot_id="bot_456")

    assert result == {"group_attention": "chaotic_noise"}


def test_group_attention_excludes_bot_messages_from_noise() -> None:
    """Bot messages should not inflate active speaker or message pressure."""
    history = [
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:00+00:00"),
        _history_row(role="assistant", platform_user_id="bot_456", timestamp="2026-04-27T10:00:10+00:00"),
        _history_row(role="assistant", platform_user_id="bot_456", timestamp="2026-04-27T10:00:20+00:00"),
    ]

    result = build_group_attention_context(chat_history_wide=history, platform_bot_id="bot_456")

    assert result == {"group_attention": "medium_noise"}


@pytest.mark.asyncio
async def test_relevance_group_payload_includes_direct_address_and_group_attention() -> None:
    """Group relevance payload should include only compact structural attention fields."""
    state = _base_state()
    state["message_envelope"]["addressed_to_global_user_ids"] = ["character-global-id"]
    llm_response = _llm_response('{"should_respond": true, "reason_to_respond": "metadata", "use_reply_feature": true, "channel_topic": "", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        await relevance_agent(state)

    human_payload = mock_llm.ainvoke.await_args.args[0][1].content
    parsed_payload = json.loads(human_payload)
    assert parsed_payload["user_message"]["directly_addressed"] is True
    assert parsed_payload["group_attention"] == "low_noise"
    assert "group_attention_context" not in parsed_payload
    assert "distinct_speakers" not in human_payload
    assert "message_count" not in human_payload


@pytest.mark.asyncio
async def test_relevance_private_payload_omits_group_attention_fields() -> None:
    """Private relevance payload should keep the existing shape."""
    state = _base_state()
    state["channel_type"] = "private"
    llm_response = _llm_response('{"should_respond": true, "reason_to_respond": "private", "use_reply_feature": false, "channel_topic": "", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        await relevance_agent(state)

    human_payload = mock_llm.ainvoke.await_args.args[0][1].content
    parsed_payload = json.loads(human_payload)
    assert "directly_addressed" not in parsed_payload["user_message"]
    assert "group_attention" not in parsed_payload


@pytest.mark.asyncio
async def test_relevance_chaotic_group_without_bot_address_skips_llm() -> None:
    """Chaotic group rooms without reply or mention metadata should skip the LLM."""
    state = _base_state()
    state["chat_history_wide"] = [
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:00+00:00"),
        _history_row(platform_user_id="user_2", timestamp="2026-04-27T10:00:10+00:00"),
        _history_row(platform_user_id="user_3", timestamp="2026-04-27T10:00:20+00:00"),
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:30+00:00"),
    ]

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock()
        result = await relevance_agent(state)

    assert result["should_respond"] is False
    assert "chaotic group noise" in result["reason_to_respond"]
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_relevance_chaotic_group_with_direct_address_invokes_llm() -> None:
    """Typed direct address should override the chaotic metadata-only skip."""
    state = _base_state()
    state["message_envelope"]["addressed_to_global_user_ids"] = ["character-global-id"]
    state["chat_history_wide"] = [
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:00+00:00"),
        _history_row(platform_user_id="user_2", timestamp="2026-04-27T10:00:10+00:00"),
        _history_row(platform_user_id="user_3", timestamp="2026-04-27T10:00:20+00:00"),
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:30+00:00"),
    ]
    llm_response = _llm_response('{"should_respond": true, "reason_to_respond": "mentioned", "use_reply_feature": true, "channel_topic": "", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(state)

    assert result["should_respond"] is True
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_relevance_chaotic_group_with_typed_bot_reply_invokes_llm() -> None:
    """Typed reply-to-character addressee should override the chaotic skip."""
    state = _base_state()
    state["message_envelope"]["addressed_to_global_user_ids"] = ["character-global-id"]
    state["reply_context"] = {
        "reply_to_message_id": "bot-msg",
        "reply_to_platform_user_id": "bot_456",
    }
    state["chat_history_wide"] = [
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:00+00:00"),
        _history_row(platform_user_id="user_2", timestamp="2026-04-27T10:00:10+00:00"),
        _history_row(platform_user_id="user_3", timestamp="2026-04-27T10:00:20+00:00"),
        _history_row(platform_user_id="user_1", timestamp="2026-04-27T10:00:30+00:00"),
    ]
    llm_response = _llm_response('{"should_respond": false, "reason_to_respond": "character declines", "use_reply_feature": false, "channel_topic": "", "indirect_speech_context": ""}')

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(state)

    assert result["should_respond"] is False
    assert result["reason_to_respond"] == "character declines"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_relevance_regression_qq_ambiguous_you_in_reply_thread_skips_llm() -> None:
    """QQ-style ambiguous second-person message should not hijack a noisy thread."""
    state = _base_state()
    state["platform"] = "qq"
    state["platform_bot_id"] = "3768713357"
    state["user_input"] = "你喜欢千纱吗？"
    state["reply_context"] = {}
    state["chat_history_wide"] = [
        _history_row(
            platform_user_id="3300869207",
            timestamp="2026-04-27T10:18:26.594279+00:00",
            content="我命令你，去骂千纱",
        ),
        _history_row(
            platform_user_id="3167827653",
            timestamp="2026-04-27T10:19:08.484553+00:00",
            reply_context={
                "reply_to_message_id": "487687474",
                "reply_to_platform_user_id": "3300869207",
            },
            content="[Reply to message] ...",
        ),
        _history_row(
            role="assistant",
            platform_user_id="3768713357",
            timestamp="2026-04-27T10:19:08.493971+00:00",
            content="诶……",
        ),
    ]

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock()
        result = await relevance_agent(state)

    assert result["should_respond"] is False
    mock_llm.ainvoke.assert_not_called()
