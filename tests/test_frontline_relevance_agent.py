"""Contract tests for the compact frontline relevance stage."""

from __future__ import annotations

import json
from importlib import import_module
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.relevance.frontline_relevance_agent import (
    FRONTLINE_RELEVANCE_MAX_COMPLETION_TOKENS,
    FRONTLINE_RELEVANCE_MAX_INPUT_CHARS,
    build_frontline_messages,
    frontline_relevance_agent,
    validate_frontline_decision,
)


frontline_module = import_module(
    "kazusa_ai_chatbot.relevance.frontline_relevance_agent"
)


def _frontline_state() -> dict:
    """Build a semantic-only frontline projection with identity sentinels."""

    return {
        "current_message": {
            "body_text": "Could you check this image?",
            "semantic_target_labels": ["character"],
            "reply_target_label": "character",
            "media_labels": ["image"],
        },
        "open_turns": [
            {
                "slot": "open_1",
                "author_relation": "same_author",
                "latest_intent": "question about the image",
                "target_summary": "character",
            },
            {
                "slot": "open_2",
                "author_relation": "same_author",
                "latest_intent": "unrelated topic",
                "target_summary": "other_user",
            },
        ],
        "recent_preludes": [
            {
                "slot": "prelude_1",
                "summary": "The user introduced a photo.",
            },
        ],
        "latest_bot_continuity": "The character answered the same image topic.",
        "identity_sentinel": "platform-user-raw-123",
        "timestamp_sentinel": "2026-07-16T00:00:00Z",
    }


def test_frontline_decision_has_closed_enums_and_bounded_cards() -> None:
    """Frontline output accepts only the compact slot vocabulary."""

    decision = validate_frontline_decision({
        "intake_action": "append",
        "append_target": "open_2",
        "prelude_targets": [],
        "reason": "same author and same image topic",
    })

    assert decision == {
        "intake_action": "append",
        "append_target": "open_2",
        "prelude_targets": [],
        "reason": "same author and same image topic",
    }

    with pytest.raises(ValueError):
        validate_frontline_decision({
            "intake_action": "append",
            "append_target": "turn-id-from-model",
            "prelude_targets": [],
            "reason": "invalid slot",
        })

    with pytest.raises(ValueError):
        validate_frontline_decision({
            "intake_action": "append",
            "append_target": "open_1",
            "prelude_targets": ["prelude_1"],
            "reason": "preludes belong to a new promoted turn",
        })


def test_frontline_decision_truncates_reason_and_limits_preludes() -> None:
    """Structural validation enforces the model-facing output budget."""

    decision = validate_frontline_decision({
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": ["prelude_1", "prelude_2", "prelude_3"],
        "reason": "x" * 200,
    })

    assert decision["append_target"] == "none"
    assert decision["prelude_targets"] == ["prelude_1", "prelude_2"]
    assert len(decision["reason"]) <= 80


def test_frontline_render_is_bounded_and_omits_raw_identity_and_time() -> None:
    """The fast route receives semantic labels rather than transport metadata."""

    messages = build_frontline_messages(_frontline_state())
    rendered = "".join(message.content for message in messages)

    assert len(rendered) <= FRONTLINE_RELEVANCE_MAX_INPUT_CHARS
    assert "platform-user-raw-123" not in rendered
    assert "2026-07-16T00:00:00Z" not in rendered
    assert '"open_1"' in rendered
    assert '"prelude_1"' in rendered
    json.loads(messages[1].content)


def test_frontline_worst_case_projection_remains_valid_json() -> None:
    """Hard-cap fitting never sends a sliced JSON document to the model."""

    state = _frontline_state()
    state["current_message"]["body_text"] = "x" * 20000
    state["open_turns"] = [
        {
            "author_relation": "same_author",
            "latest_intent": "y" * 2000,
            "opening_excerpt": "z" * 2000,
            "target_summary": "character",
            "reply_summary": "character",
            "media_summary": "image" * 100,
        }
        for _index in range(3)
    ]

    messages = build_frontline_messages(state)

    assert sum(len(message.content) for message in messages) <= (
        FRONTLINE_RELEVANCE_MAX_INPUT_CHARS
    )
    json.loads(messages[1].content)


def test_frontline_route_has_exact_completion_and_thinking_budget() -> None:
    """The configured fast route must stay within the approved call envelope."""

    config = frontline_module._frontline_relevance_agent_llm_config

    assert config.max_completion_tokens == FRONTLINE_RELEVANCE_MAX_COMPLETION_TOKENS
    assert config.max_completion_tokens == 256
    assert config.thinking.enabled is False


@pytest.mark.asyncio
async def test_frontline_agent_uses_structural_parser_and_returns_decision(
    monkeypatch,
) -> None:
    """A valid model object becomes the closed frontline decision contract."""

    response = MagicMock()
    response.content = json.dumps({
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "direct request",
    })
    llm = frontline_module._frontline_relevance_agent_llm
    invoke = AsyncMock(return_value=response)
    monkeypatch.setattr(llm, "ainvoke", invoke)

    result = await frontline_relevance_agent(_frontline_state())

    assert result["intake_action"] == "start"
    assert result["append_target"] == "none"
    invoke.assert_awaited_once()
    assert invoke.await_args.kwargs["config"] is (
        frontline_module._frontline_relevance_agent_llm_config
    )
