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
        "conversation_scope": "group",
        "active_character_name": "Kazusa",
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
    payload = json.loads(messages[1].content)
    assert payload["conversation_scope"] == "group"
    assert payload["active_character_name"] == "Kazusa"


def test_frontline_authoritative_prompt_limits_work_to_semantic_linkage() -> None:
    """Typed participation removes the contradictory discard workload."""

    messages = build_frontline_messages(_frontline_state())
    system_prompt = messages[0].content

    assert "already established participation" in system_prompt
    assert "clearly continues exactly one" in system_prompt
    assert "slot number, list order" in system_prompt.lower()
    assert "never treat this payload\nas private input" in system_prompt
    assert "Recipient\n   withdrawal" in system_prompt
    assert '"intake_action":"start|append"' in system_prompt
    assert '"intake_action":"discard|start|append"' not in system_prompt
    assert "Otherwise discard" not in system_prompt


def test_frontline_ordinary_group_retains_participation_judgment() -> None:
    """Untargeted group traffic keeps semantic discard and participation rules."""

    state = _frontline_state()
    state["current_message"]["semantic_target_labels"] = []
    state["current_message"]["reply_target_label"] = "none"

    system_prompt = build_frontline_messages(state)[0].content

    assert "explicit whole-group invitation" in system_prompt
    assert "answerability" in system_prompt
    assert "latest_bot_continuity is context, never an open slot" in (
        system_prompt
    )
    assert 'elliptical reference such as "that one"' in system_prompt
    assert "only a direct character summon or explicitly" in system_prompt
    assert "append is mandatory and start is invalid" in system_prompt
    assert "target none and reply none, start is valid only" in system_prompt
    assert '"intake_action":"discard|start|append"' in system_prompt


def test_frontline_private_prompt_has_no_group_suppression_workload() -> None:
    """Private intake uses its smaller scope-specific routing contract."""

    state = _frontline_state()
    state["conversation_scope"] = "private"
    messages = build_frontline_messages(state)
    system_prompt = messages[0].content

    assert "conversation_scope is private" in system_prompt
    assert "always has a character participation basis" in system_prompt
    assert "Group Participation" not in system_prompt


def test_frontline_prompt_hides_actions_for_absent_candidate_slots() -> None:
    """The local model sees only actions supported by supplied candidates."""

    state = _frontline_state()
    state["current_message"]["semantic_target_labels"] = []
    state["current_message"]["reply_target_label"] = "none"
    state["open_turns"] = []
    state["recent_preludes"] = []
    messages = build_frontline_messages(state)
    system_prompt = messages[0].content

    assert '"intake_action":"discard|start"' in system_prompt
    assert '"intake_action":"discard|start|append"' not in system_prompt
    assert "The append action is unavailable" in system_prompt
    assert "Return prelude_targets as [] exactly" in system_prompt


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


@pytest.mark.asyncio
async def test_frontline_direct_without_candidates_starts_without_model_call(
    monkeypatch,
) -> None:
    """Typed participation with no linkage candidates is admitted directly."""

    invoke = AsyncMock()
    monkeypatch.setattr(
        frontline_module._frontline_relevance_agent_llm,
        "ainvoke",
        invoke,
    )
    state = _frontline_state()
    state["open_turns"] = []
    state["recent_preludes"] = []

    result = await frontline_relevance_agent(state)

    assert result == {
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "authoritative character participation",
    }
    invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_frontline_direct_open_turn_rejects_discard_without_retry(
    monkeypatch,
) -> None:
    """Unavailable discard cannot override typed participation or add a call."""

    discarded = MagicMock()
    discarded.content = json.dumps({
        "intake_action": "discard",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "invalid unavailable action",
    })
    invoke = AsyncMock(return_value=discarded)
    monkeypatch.setattr(
        frontline_module._frontline_relevance_agent_llm,
        "ainvoke",
        invoke,
    )
    state = _frontline_state()

    result = await frontline_relevance_agent(state)

    assert result == {
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "invalid authoritative frontline output",
    }
    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_frontline_broadcast_without_candidates_starts_without_call(
    monkeypatch,
) -> None:
    """Typed whole-group participation is admitted like typed direct input."""

    invoke = AsyncMock()
    monkeypatch.setattr(
        frontline_module._frontline_relevance_agent_llm,
        "ainvoke",
        invoke,
    )
    state = _frontline_state()
    state["current_message"]["semantic_target_labels"] = ["broadcast"]
    state["current_message"]["reply_target_label"] = "none"
    state["open_turns"] = []
    state["recent_preludes"] = []

    result = await frontline_relevance_agent(state)

    assert result["intake_action"] == "start"
    invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_frontline_does_not_recheck_untargeted_discard(
    monkeypatch,
) -> None:
    """An ordinary group discard keeps the single-call path."""

    response = MagicMock()
    response.content = json.dumps({
        "intake_action": "discard",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "not addressed to the character",
    })
    invoke = AsyncMock(return_value=response)
    monkeypatch.setattr(
        frontline_module._frontline_relevance_agent_llm,
        "ainvoke",
        invoke,
    )
    state = _frontline_state()
    state["current_message"]["semantic_target_labels"] = []
    state["current_message"]["reply_target_label"] = "none"
    state["open_turns"] = []

    result = await frontline_relevance_agent(state)

    assert result["intake_action"] == "discard"
    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_frontline_agent_fails_closed_on_unsupplied_model_slot(
    monkeypatch,
) -> None:
    """A vocabulary-valid but absent slot cannot pass model validation."""

    response = MagicMock()
    response.content = json.dumps({
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": ["prelude_1"],
        "reason": "invented slot",
    })
    llm = frontline_module._frontline_relevance_agent_llm
    monkeypatch.setattr(llm, "ainvoke", AsyncMock(return_value=response))
    state = _frontline_state()
    state["current_message"]["semantic_target_labels"] = []
    state["current_message"]["reply_target_label"] = "none"
    state["recent_preludes"] = []

    result = await frontline_relevance_agent(state)

    assert result == {
        "intake_action": "discard",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "invalid frontline output",
    }


@pytest.mark.asyncio
async def test_frontline_agent_fails_closed_on_unsupplied_append_slot(
    monkeypatch,
) -> None:
    """An absent open slot cannot pass as a vocabulary-valid append."""

    response = MagicMock()
    response.content = json.dumps({
        "intake_action": "append",
        "append_target": "open_1",
        "prelude_targets": [],
        "reason": "invented slot",
    })
    llm = frontline_module._frontline_relevance_agent_llm
    monkeypatch.setattr(llm, "ainvoke", AsyncMock(return_value=response))
    state = _frontline_state()
    state["current_message"]["semantic_target_labels"] = []
    state["current_message"]["reply_target_label"] = "none"
    state["open_turns"] = []

    result = await frontline_relevance_agent(state)

    assert result == {
        "intake_action": "discard",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "invalid frontline output",
    }
