"""Contract tests for the compact frontline relevance stage."""

from __future__ import annotations

import json
from importlib import import_module
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.relevance.frontline_relevance_agent import (
    FRONTLINE_RELEVANCE_MAX_COMPLETION_TOKENS,
    FRONTLINE_RELEVANCE_MAX_INPUT_CHARS,
    build_frontline_messages,
    frontline_relevance_agent,
    validate_frontline_decision,
)
from kazusa_ai_chatbot.relevance.participation_evidence import (
    validate_participation_assessment,
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
        "character_cognition_state": build_character_production_state(
            updated_at="2026-07-16T00:00:00Z",
        ),
        "identity_sentinel": "platform-user-raw-123",
        "timestamp_sentinel": "2026-07-16T00:00:00Z",
    }


def _active_goal_state(description: str) -> dict:
    """Build one active native character goal for relevance tests."""

    state = build_character_production_state(
        updated_at="2026-07-16T00:00:00Z",
    )
    state["goals"] = [{
        "description": description,
        "status": "pursuing",
        "salience": 80,
    }]
    return state


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

    assert "已经确认当前角色参与" in system_prompt
    assert "当前含义明确延续恰好一个" in system_prompt
    assert "slot 编号、列表顺序" in system_prompt
    assert "conversation_scope 为 group" in system_prompt
    assert "接收者撤回" in system_prompt
    assert '"intake_action":"start|append"' in system_prompt
    assert '"intake_action":"discard|start|append"' not in system_prompt
    assert "群聊参与依据" not in system_prompt


def test_frontline_ordinary_group_retains_participation_judgment() -> None:
    """Untargeted group traffic keeps semantic discard and participation rules."""

    state = _frontline_state()
    state["current_message"]["semantic_target_labels"] = []
    state["current_message"]["reply_target_label"] = "none"

    system_prompt = build_frontline_messages(state)[0].content

    assert "明确邀请全群" in system_prompt
    assert "可回答性" in system_prompt
    assert "latest_bot_continuity 只提供语境，不是 open slot" in (
        system_prompt
    )
    assert "类似“那个”的省略指代" in system_prompt
    assert "只是直接召唤角色" in system_prompt
    assert "除非当前消息转移或撤回" in system_prompt
    assert "target 与 reply 都是 none 时" in system_prompt
    assert '"intake_action":"discard|start|append"' in system_prompt


def test_frontline_private_prompt_has_no_group_suppression_workload() -> None:
    """Private intake uses its smaller scope-specific routing contract."""

    state = _frontline_state()
    state["conversation_scope"] = "private"
    messages = build_frontline_messages(state)
    system_prompt = messages[0].content

    assert "conversation_scope 为 private" in system_prompt
    assert "始终具有角色参与依据" in system_prompt
    assert "群聊参与依据" not in system_prompt


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
    assert "本次调用不可选择 append" in system_prompt
    assert "prelude_targets 必须恰好为 []" in system_prompt


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


def test_frontline_cap_drops_weakest_state_and_invalidates_its_ref(
    monkeypatch,
) -> None:
    """Cap fitting retains stronger state and rejects a removed state ref."""

    state = _frontline_state()
    state["character_cognition_state"]["goals"] = [
        {
            "description": f"active goal {index} " + "x" * 120,
            "status": "pursuing",
            "salience": 90 - index,
        }
        for index in range(6)
    ]
    baseline_messages = build_frontline_messages(state)
    monkeypatch.setattr(
        frontline_module,
        "FRONTLINE_RELEVANCE_MAX_INPUT_CHARS",
        sum(len(message.content) for message in baseline_messages) - 1,
    )

    messages = build_frontline_messages(state)
    payload = json.loads(messages[1].content)

    assert [
        item["ref"] for item in payload["character_state_evidence"]
    ] == ["state_1", "state_2", "state_3", "state_4", "state_5"]
    with pytest.raises(ValueError, match="requires state evidence"):
        validate_participation_assessment(
            {
                "recipient_relation": "character",
                "admission_basis": "character_state_salience",
                "interaction_evidence_refs": ["target_character"],
                "character_state_refs": ["state_6"],
            },
            interaction_evidence=payload["interaction_evidence"],
            character_state_evidence=payload["character_state_evidence"],
            stage="frontline",
            action="start",
            append_target="none",
            use_reply_feature=False,
        )


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
        "recipient_relation": "character",
        "admission_basis": "interaction_relevance",
        "interaction_evidence_refs": ["target_character"],
        "character_state_refs": [],
        "reason": "direct request",
    })
    llm = frontline_module._frontline_relevance_agent_llm
    invoke = AsyncMock(return_value=response)
    monkeypatch.setattr(llm, "ainvoke", invoke)

    result = await frontline_relevance_agent(_frontline_state())

    assert result["decision"]["intake_action"] == "start"
    assert result["decision"]["append_target"] == "none"
    assert set(result) == {
        "decision",
        "attempt_diagnostics",
    }
    assert set(result["decision"]) == {
        "intake_action",
        "append_target",
        "prelude_targets",
        "reason",
    }
    assert result["attempt_diagnostics"] == []
    invoke.assert_awaited_once()
    assert invoke.await_args.kwargs["config"] is (
        frontline_module._frontline_relevance_agent_llm_config
    )


@pytest.mark.asyncio
async def test_frontline_provider_exhaustion_starts_authoritative_turn(
    monkeypatch,
) -> None:
    """Two provider failures use the typed authoritative start decision."""

    invoke = AsyncMock(
        side_effect=[RuntimeError("provider unavailable")] * 2,
    )
    record_trace = AsyncMock()
    monkeypatch.setattr(
        frontline_module._frontline_relevance_agent_llm,
        "ainvoke",
        invoke,
    )
    monkeypatch.setattr(
        frontline_module.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    state = _frontline_state()
    state["llm_trace_id"] = "frontline-provider-trace"

    result = await frontline_relevance_agent(state)

    assert result["decision"] == {
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "authoritative character participation",
    }
    assert result["attempt_diagnostics"] == [{
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "frontline_relevance",
        "error_code": "frontline_relevance_deterministic_degraded",
        "attempt_count": 2,
        "safe_checkpoint": "pre_state_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }]
    assert invoke.await_count == 2
    assert invoke.await_args_list[0].args[0] == (
        invoke.await_args_list[1].args[0]
    )
    assert [call.kwargs["attempt_index"] for call in record_trace.await_args_list] == [
        1,
        2,
        0,
    ]
    assert [call.kwargs["parse_status"] for call in record_trace.await_args_list] == [
        "provider_error",
        "provider_error",
        "deterministic",
    ]
    assert [call.kwargs["status"] for call in record_trace.await_args_list] == [
        "failed",
        "failed",
        "accepted_degraded",
    ]
    assert record_trace.await_args_list[-1].kwargs["stage_name"] == (
        "frontline_relevance_agent.deterministic"
    )
    assert record_trace.await_args_list[-1].kwargs["response_text"] == ""


@pytest.mark.asyncio
async def test_frontline_provider_exhaustion_discards_non_authoritative_turn(
    monkeypatch,
) -> None:
    """Two provider failures use the typed ordinary discard decision."""

    invoke = AsyncMock(
        side_effect=[RuntimeError("provider unavailable")] * 2,
    )
    record_trace = AsyncMock()
    monkeypatch.setattr(
        frontline_module._frontline_relevance_agent_llm,
        "ainvoke",
        invoke,
    )
    monkeypatch.setattr(
        frontline_module.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    state = _frontline_state()
    state["llm_trace_id"] = "frontline-provider-trace"
    state["current_message"]["semantic_target_labels"] = []
    state["current_message"]["reply_target_label"] = "none"
    state["open_turns"] = []
    state["recent_preludes"] = []

    result = await frontline_relevance_agent(state)

    assert result["decision"]["intake_action"] == "discard"
    assert result["decision"]["append_target"] == "none"
    assert result["decision"]["prelude_targets"] == []
    assert result["decision"]["reason"] == "frontline provider exhausted"
    assert result["attempt_diagnostics"][0]["error_code"] == (
        "frontline_relevance_deterministic_degraded"
    )
    assert invoke.await_count == 2
    assert [call.kwargs["parse_status"] for call in record_trace.await_args_list] == [
        "provider_error",
        "provider_error",
        "deterministic",
    ]
    assert record_trace.await_args_list[-1].kwargs["status"] == (
        "accepted_degraded"
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

    assert result["decision"] == {
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "authoritative character participation",
    }
    assert result["attempt_diagnostics"] == []
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
    record_trace = AsyncMock()
    monkeypatch.setattr(
        frontline_module.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    state = _frontline_state()
    state["llm_trace_id"] = "frontline-normalized-trace"

    result = await frontline_relevance_agent(state)

    assert result["decision"] == {
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "invalid authoritative frontline output",
    }
    assert result["attempt_diagnostics"] == []
    invoke.assert_awaited_once()
    assert record_trace.await_count == 1
    assert record_trace.await_args.kwargs["parse_status"] == "normalized"
    assert record_trace.await_args.kwargs["status"] == "succeeded"
    assert record_trace.await_args.kwargs["attempt_index"] == 1


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

    assert result["decision"]["intake_action"] == "start"
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
        "recipient_relation": "unknown",
        "admission_basis": "none",
        "interaction_evidence_refs": [],
        "character_state_refs": [],
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

    assert result["decision"]["intake_action"] == "discard"
    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_frontline_incident_rejects_message_only_character_claim(
    monkeypatch,
) -> None:
    """The captured pronoun and incidental glyph cannot ground admission."""

    response = MagicMock()
    response.content = json.dumps({
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "recipient_relation": "character",
        "admission_basis": "interaction_relevance",
        "interaction_evidence_refs": ["message_1"],
        "character_state_refs": [],
        "reason": "the pronoun addresses the character",
    })
    monkeypatch.setattr(
        frontline_module._frontline_relevance_agent_llm,
        "ainvoke",
        AsyncMock(return_value=response),
    )
    state = _frontline_state()
    state["active_character_name"] = "一之濑明日奈"
    state["current_message"] = {
        "body_text": "直接找你一换一是吧",
        "semantic_target_labels": [],
        "reply_target_label": "none",
        "media_labels": [],
    }
    state["open_turns"] = []
    state["recent_preludes"] = []
    state["latest_bot_continuity"] = ""

    result = await frontline_relevance_agent(state)

    assert result["decision"]["intake_action"] == "discard"
    assert result["decision"]["reason"] == "invalid frontline output"


@pytest.mark.asyncio
async def test_frontline_quoted_canonical_name_remains_model_judged(
    monkeypatch,
) -> None:
    """A full-name candidate does not deterministically force admission."""

    response = MagicMock()
    response.content = json.dumps({
        "intake_action": "discard",
        "append_target": "none",
        "prelude_targets": [],
        "recipient_relation": "unknown",
        "admission_basis": "none",
        "interaction_evidence_refs": ["name_1", "message_1"],
        "character_state_refs": [],
        "reason": "the name is quoted rather than used as an address",
    })
    invoke = AsyncMock(return_value=response)
    monkeypatch.setattr(
        frontline_module._frontline_relevance_agent_llm,
        "ainvoke",
        invoke,
    )
    state = _frontline_state()
    state["active_character_name"] = "一之濑明日奈"
    state["current_message"] = {
        "body_text": '小林说“一之濑明日奈”只是书里的名字',
        "semantic_target_labels": [],
        "reply_target_label": "none",
        "media_labels": [],
    }
    state["open_turns"] = []
    state["recent_preludes"] = []
    state["latest_bot_continuity"] = ""

    result = await frontline_relevance_agent(state)

    assert result["decision"]["intake_action"] == "discard"
    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_frontline_state_salience_preserves_other_recipient(
    monkeypatch,
) -> None:
    """Active-state relevance admits speech without changing its recipient."""

    response = MagicMock()
    response.content = json.dumps({
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "recipient_relation": "other_participant",
        "admission_basis": "character_state_salience",
        "interaction_evidence_refs": ["target_other", "message_1"],
        "character_state_refs": ["state_1"],
        "reason": "the warning intersects an active safety goal",
    })
    monkeypatch.setattr(
        frontline_module._frontline_relevance_agent_llm,
        "ainvoke",
        AsyncMock(return_value=response),
    )
    state = _frontline_state()
    state["current_message"] = {
        "body_text": "Alex, the current challenge will put you in danger.",
        "semantic_target_labels": ["other_participant"],
        "reply_target_label": "other_participant",
        "media_labels": [],
    }
    state["character_cognition_state"] = _active_goal_state(
        "Prevent the current challenge from harming a participant.",
    )
    state["open_turns"] = []
    state["recent_preludes"] = []
    state["latest_bot_continuity"] = ""

    result = await frontline_relevance_agent(state)

    assert result["decision"]["intake_action"] == "start"
    assert "recipient_relation" not in result["decision"]


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
    invoke = AsyncMock(return_value=response)
    record_trace = AsyncMock()
    monkeypatch.setattr(llm, "ainvoke", invoke)
    monkeypatch.setattr(
        frontline_module.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    state = _frontline_state()
    state["llm_trace_id"] = "frontline-normalized-trace"
    state["current_message"]["semantic_target_labels"] = []
    state["current_message"]["reply_target_label"] = "none"
    state["recent_preludes"] = []

    result = await frontline_relevance_agent(state)

    assert result["decision"] == {
        "intake_action": "discard",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "invalid frontline output",
    }
    assert result["attempt_diagnostics"] == []
    invoke.assert_awaited_once()
    assert record_trace.await_count == 1
    assert record_trace.await_args.kwargs["parse_status"] == "normalized"
    assert record_trace.await_args.kwargs["status"] == "succeeded"
    assert record_trace.await_args.kwargs["attempt_index"] == 1


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

    assert result["decision"] == {
        "intake_action": "discard",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "invalid frontline output",
    }
    assert result["attempt_diagnostics"] == []
