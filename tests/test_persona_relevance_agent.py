"""Tests for the persona-aware settled relevance contract."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kazusa_ai_chatbot.relevance.persona_relevance_agent as relevance_module
from kazusa_ai_chatbot.relevance.persona_relevance_agent import (
    SETTLED_RELEVANCE_MAX_COMPLETION_TOKENS,
    SETTLED_RELEVANCE_MAX_INPUT_CHARS,
    build_group_attention_context,
    build_settled_relevance_messages,
    relevance_agent,
    validate_settled_relevance_decision,
)


def _base_state() -> dict:
    """Build a minimal settled relevance state."""

    return {
        "user_input": "Hello character.",
        "user_profile": {"affinity": 500},
        "character_profile": {
            "name": "Character",
            "global_user_id": "character-global-id",
        },
        "platform_bot_id": "bot-id",
        "global_user_id": "user-id",
        "platform_channel_id": "channel-id",
        "conversation_scope": "group",
        "active_character_name": "Character",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "assembled_fragments": [{
            "body_text": "Hello character.",
            "semantic_target_labels": ["character"],
            "reply_target_label": "none",
            "media_labels": [],
        }],
        "fresh_history": [],
        "media_descriptions": [],
        "scene_context": "A quiet group conversation.",
        "relationship_context": "The user is familiar.",
        "observation_status": "observation_complete",
    }


def _llm_response(content: str) -> MagicMock:
    """Return a small mock object shaped like a LangChain response."""

    response = MagicMock()
    response.content = content
    return response


@pytest.mark.asyncio
async def test_relevance_agent_returns_proceed_action() -> None:
    """A valid proceed object is forwarded as the settled action."""

    response = _llm_response(json.dumps({
        "response_action": "proceed",
        "reason_to_respond": "the current message addresses the character",
        "use_reply_feature": True,
        "channel_topic": "greeting",
        "indirect_speech_context": "",
    }))

    with patch.object(relevance_module, "_relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await relevance_agent(_base_state())

    assert result["response_action"] == "proceed"
    assert result["should_respond"] is True
    assert result["use_reply_feature"] is True


@pytest.mark.asyncio
async def test_relevance_agent_returns_ignore_action() -> None:
    """A valid ignore object keeps cognition silent."""

    response = _llm_response(json.dumps({
        "response_action": "ignore",
        "reason_to_respond": "third-party traffic",
        "use_reply_feature": False,
        "channel_topic": "other topic",
        "indirect_speech_context": "",
    }))

    with patch.object(relevance_module, "_relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await relevance_agent(_base_state())

    assert result["response_action"] == "ignore"
    assert result["should_respond"] is False


@pytest.mark.asyncio
async def test_relevance_agent_malformed_json_fails_closed() -> None:
    """Malformed settled output invokes no repair model and ignores."""

    response = _llm_response("not json")
    with patch.object(relevance_module, "_relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await relevance_agent(_base_state())

    assert result["response_action"] == "ignore"
    assert result["should_respond"] is False


def test_settled_decision_requires_complete_phase_action() -> None:
    """Settled relevance uses closed actions and no wait at the deadline."""

    decision = validate_settled_relevance_decision({
        "response_action": "proceed",
        "reason_to_respond": "the assembled request is directed to the character",
        "use_reply_feature": True,
        "channel_topic": "image question",
        "indirect_speech_context": "",
    }, observation_status="observation_complete")

    assert decision["response_action"] == "proceed"
    with pytest.raises(ValueError):
        validate_settled_relevance_decision({
            "response_action": "wait",
            "reason_to_respond": "more context",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
        }, observation_status="observation_complete")


def test_settled_render_keeps_latest_fragment_and_respects_cap() -> None:
    """The settled prompt is capped while preserving the latest correction."""

    state = {
        "assembled_fragments": [
            {"sequence": 1, "body_text": "The first request " + "x" * 3000},
            {"sequence": 2, "body_text": "Correction: use the latest request."},
        ],
        "media_descriptions": [
            {"media_kind": "image", "description": "a small diagram"},
        ],
        "fresh_history": [
            {"speaker": "other_user", "body_text": "I already answered that."},
        ],
        "scene_context": "A group discussion about a diagram.",
        "relationship_context": "The user often asks follow-up questions.",
        "conversation_scope": "group",
        "active_character_name": "Character",
        "raw_turn_id": "turn-id-raw-1",
        "raw_deadline": "2026-07-16T00:00:10Z",
    }

    messages = build_settled_relevance_messages(
        state,
        observation_status="more_time_available",
    )
    rendered = "".join(message.content for message in messages)

    assert len(rendered) <= SETTLED_RELEVANCE_MAX_INPUT_CHARS
    assert "turn-id-raw-1" not in rendered
    assert "2026-07-16T00:00:10Z" not in rendered
    assert "Correction: use the latest request." in rendered
    assert "ignore rather than" in messages[0].content
    assert "more_time_available" not in rendered
    assert "observation_complete" not in rendered
    assert "additional_media_present" not in rendered
    assert "media_evidence_status" in rendered
    payload = json.loads(messages[1].content)
    assert payload["assembled_turn"]["earlier_context_present"] is True
    assert payload["assembled_turn"]["author_relation"] == "current_human"
    assert payload["assembled_turn"]["effective_latest_fragment"][
        "body_text"
    ] == "Correction: use the latest request."
    assert payload["conversation_scope"] == "group"
    assert payload["active_character_name"] == "Character"


def test_settled_prompt_renders_only_currently_available_actions() -> None:
    """The final assessment cannot ask the local model for an invalid wait."""

    waiting_messages = build_settled_relevance_messages(
        _base_state(),
        observation_status="more_time_available",
    )
    final_messages = build_settled_relevance_messages(
        _base_state(),
        observation_status="observation_complete",
    )

    assert '"response_action":"ignore|proceed|wait"' in (
        waiting_messages[0].content
    )
    assert "assembled_turn.author_relation is current_human" in (
        waiting_messages[0].content
    )
    assert "A statement that group members could react to is insufficient" in (
        waiting_messages[0].content
    )
    assert "never swap their roles" in (
        waiting_messages[0].content
    )
    assert '"response_action":"ignore|proceed"' in final_messages[0].content
    assert "wait" not in final_messages[0].content.lower()
    assert waiting_messages[1].content == final_messages[1].content


def test_settled_prompt_defines_native_reply_anchor_semantics() -> None:
    """The settled prompt owns social usefulness, not delivery feasibility."""

    messages = build_settled_relevance_messages(_base_state())
    system_prompt = messages[0].content

    assert "anchor the answer to effective_latest_fragment" in system_prompt
    assert "specific character-directed message" in system_prompt
    assert "for private input or a" in system_prompt
    assert "whole-group invitation" in system_prompt
    assert "response_action is not proceed" in system_prompt


def test_settled_history_projects_production_participant_relations() -> None:
    """Production rows retain author, addressee, and reply relationships."""

    state = _base_state()
    state.update({
        "current_author_global_user_id": "current-global",
        "current_author_platform_user_id": "current-platform",
        "character_global_user_id": "character-global-id",
        "platform_bot_id": "bot-id",
        "fresh_history": [
            {
                "role": "user",
                "platform_user_id": "current-platform",
                "global_user_id": "current-global",
                "body_text": "My earlier question.",
                "addressed_to_global_user_ids": ["character-global-id"],
                "broadcast": False,
                "reply_context": {},
                "turn_temporal_relation": "before_active_turn",
            },
            {
                "role": "assistant",
                "platform_user_id": "bot-id",
                "global_user_id": "character-global-id",
                "body_text": "The character answered.",
                "addressed_to_global_user_ids": ["current-global"],
                "broadcast": False,
                "reply_context": {
                    "reply_to_platform_user_id": "current-platform",
                },
                "turn_temporal_relation": "after_active_turn",
            },
            {
                "role": "user",
                "platform_user_id": "other-platform",
                "global_user_id": "other-global",
                "body_text": "Another participant answered too.",
                "addressed_to_global_user_ids": ["current-global"],
                "broadcast": False,
                "reply_context": {
                    "reply_to_platform_user_id": "bot-id",
                },
                "turn_temporal_relation": "after_active_turn",
            },
            {
                "role": "user",
                "platform_user_id": "other-platform",
                "global_user_id": "other-global",
                "body_text": "An answer arrived between active fragments.",
                "addressed_to_global_user_ids": ["current-global"],
                "broadcast": False,
                "reply_context": {},
                "turn_temporal_relation": "during_active_turn",
            },
        ],
    })

    messages = build_settled_relevance_messages(state)
    history = json.loads(messages[1].content)["fresh_history"]

    assert history == [
        {
            "speaker_relation": "current_author",
            "body_text": "My earlier question.",
            "target_summary": "character",
            "reply_summary": "none",
            "turn_relation": "before_active_turn",
        },
        {
            "speaker_relation": "character",
            "body_text": "The character answered.",
            "target_summary": "current_author",
            "reply_summary": "current_author",
            "turn_relation": "after_active_turn",
        },
        {
            "speaker_relation": "other_participant",
            "body_text": "Another participant answered too.",
            "target_summary": "current_author",
            "reply_summary": "character",
            "turn_relation": "after_active_turn",
        },
        {
            "speaker_relation": "other_participant",
            "body_text": "An answer arrived between active fragments.",
            "target_summary": "current_author",
            "reply_summary": "none",
            "turn_relation": "during_active_turn",
        },
    ]


def test_settled_worst_case_projection_remains_valid_json() -> None:
    """Fallback projection preserves a valid bounded semantic payload."""

    state = _base_state()
    state["assembled_fragments"] = [
        {
            "body_text": f"fragment-{index} " + "x" * 5000,
            "semantic_target_labels": ["character"],
            "reply_target_label": "character",
            "media_labels": ["image"],
        }
        for index in range(30)
    ]
    state["fresh_history"] = [
        {"speaker": "user", "body_text": "h" * 3000}
        for _index in range(10)
    ]
    state["media_descriptions"] = [
        {"media_kind": "image", "description": "m" * 3000}
        for _index in range(4)
    ]

    messages = build_settled_relevance_messages(state)
    payload = json.loads(messages[1].content)

    assert sum(len(message.content) for message in messages) <= (
        SETTLED_RELEVANCE_MAX_INPUT_CHARS
    )
    assert payload["assembled_turn"]["fragments"][0]["body_text"].startswith(
        "fragment-0"
    )
    assert payload["assembled_turn"]["fragments"][-1]["body_text"].startswith(
        "fragment-29"
    )
    assert payload["assembled_turn"]["earlier_context_present"] is True


def test_settled_route_has_exact_completion_and_thinking_budget() -> None:
    """The settled route uses the approved completion cap."""

    config = relevance_module._relevance_agent_llm_config
    assert config.max_completion_tokens == SETTLED_RELEVANCE_MAX_COMPLETION_TOKENS
    assert config.max_completion_tokens == 512
    assert config.thinking.enabled is False


def test_group_attention_is_descriptive_only() -> None:
    """Group attention remains a prompt descriptor, not a semantic short-circuit."""

    history = [
        {
            "role": "user",
            "platform_user_id": "user-a",
            "timestamp": "2026-04-27T10:00:00+00:00",
            "addressed_to_global_user_ids": [],
        },
        {
            "role": "user",
            "platform_user_id": "user-b",
            "timestamp": "2026-04-27T10:00:10+00:00",
            "addressed_to_global_user_ids": [],
        },
    ]

    result = build_group_attention_context(
        chat_history_wide=history,
        platform_bot_id="bot-id",
    )

    assert result == {"group_attention": "medium_noise"}
