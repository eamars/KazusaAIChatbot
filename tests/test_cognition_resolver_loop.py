"""Tests for cognition resolver capability execution."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_resolver import capabilities as capabilities_module
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_OBSERVATION_VERSION,
    ResolverValidationError,
)
from kazusa_ai_chatbot.cognition_resolver.loop import call_cognition_resolver_loop
from kazusa_ai_chatbot.time_boundary import build_turn_clock


def _resolver_request(
    *,
    capability_kind: str = "rag_evidence",
    objective: str = "检索当前用户与这个问题有关的关系和记忆证据。",
) -> dict:
    return {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": capability_kind,
        "objective": objective,
        "reason": "当前认知循环缺少足够证据。",
        "priority": "now",
    }


def _resolver_state() -> dict:
    turn_clock = build_turn_clock("2026-05-30 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="resolver-capability-episode",
        percept_id="resolver-capability-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="Need an evidence-backed answer.",
        platform="debug",
        platform_channel_id="channel-123",
        channel_type="private",
        platform_message_id="message-123",
        platform_user_id="platform-user-123",
        global_user_id="global-user-123",
        user_name="Test User",
        active_turn_platform_message_ids=["message-123"],
        active_turn_conversation_row_ids=["row-123"],
        debug_modes={},
        target_addressed_user_ids=["character-123"],
        target_broadcast=False,
    )
    return {
        "decontexualized_input": "Original user request about trust.",
        "referents": [],
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-123",
        },
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "channel_type": "private",
        "platform_message_id": "message-123",
        "platform_bot_id": "bot-123",
        "global_user_id": "global-user-123",
        "user_name": "Test User",
        "user_profile": {"affinity": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "prompt_message_context": {
            "body_text": "Need an evidence-backed answer.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-123"],
            "broadcast": False,
        },
        "channel_topic": "debug",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": {
            "current_thread": "trust question",
        },
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
        "active_turn_platform_message_ids": ["message-123"],
        "active_turn_conversation_row_ids": ["row-123"],
        "cognitive_episode": episode,
    }


def _cognition_result(
    *,
    internal_monologue: str,
    action_specs: list[dict] | None = None,
    resolver_requests: list[dict] | None = None,
) -> dict:
    return {
        "internal_monologue": internal_monologue,
        "interaction_subtext": f"{internal_monologue} 的互动潜台词",
        "emotional_appraisal": f"{internal_monologue} 的情绪判断",
        "character_intent": f"{internal_monologue} 的角色意图",
        "logical_stance": f"{internal_monologue} 的逻辑立场",
        "judgment_note": f"{internal_monologue} 的判断备注",
        "social_distance": "close",
        "emotional_intensity": "low",
        "vibe_check": "steady",
        "relational_dynamic": "trusted",
        "action_specs": action_specs or [],
        "resolver_capability_requests": resolver_requests or [],
    }


def _speak_action_spec(reason: str = "已经有足够证据，可以进入可见回复。") -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "speak",
        "cognition_mode": "deliberative",
        "source_refs": [],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_user",
            "target_id": None,
            "owner": "l2d",
            "scope": {},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {},
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": reason,
    }


@pytest.mark.asyncio
async def test_loop_runs_cognition_capability_then_cognition_again() -> None:
    """The resolver must recur through cognition after a capability observation."""

    request = _resolver_request(
        objective="检索信任判断需要的关系证据。",
    )
    final_action = _speak_action_spec()
    cognition_inputs: list[dict] = []
    capability_inputs: list[tuple[dict, dict]] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            return _cognition_result(
                internal_monologue="第一轮：证据不足",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="第二轮：证据足够",
            action_specs=[final_action],
        )

    async def execute_capability(
        capability_request: dict,
        state: dict,
    ) -> dict:
        capability_inputs.append((capability_request, dict(state)))
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_trust_memory",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "succeeded",
            "prompt_safe_summary": "找到一条信任相关记忆。",
            "rag_result": {
                "answer": "用户曾经稳定支持过她的判断。",
                "memory_evidence": [
                    {
                        "summary": "用户在一次困难讨论里支持过她。",
                    },
                ],
            },
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 2
    assert len(capability_inputs) == 1
    assert capability_inputs[0][0] == request
    assert cognition_inputs[0]["rag_result"]["answer"] == ""
    assert cognition_inputs[0]["resolver_context"].startswith("resolver_state:")
    assert "找到一条信任相关记忆" in cognition_inputs[1]["resolver_context"]
    assert "用户曾经稳定支持过她的判断" in cognition_inputs[1]["resolver_context"]
    assert result["action_specs"] == [final_action]
    assert result["rag_result"]["answer"] == "用户曾经稳定支持过她的判断。"

    resolver_state = result["resolver_state"]
    assert resolver_state["status"] == "terminal"
    assert resolver_state["held_action_specs"] == [final_action]
    assert len(resolver_state["observations"]) == 1
    assert len(resolver_state["cycle_traces"]) == 2
    assert resolver_state["cycle_traces"][0]["selected_capability_kind"] == (
        "rag_evidence"
    )
    assert resolver_state["cycle_traces"][0]["observation_ids"] == [
        "resolver_obs_trust_memory"
    ]
    assert resolver_state["cycle_traces"][1]["selected_capability_kind"] == ""
    assert resolver_state["cycle_traces"][1]["final_surface_decision"].startswith(
        "action_specs="
    )
    assert resolver_state["cycle_traces"][1]["terminal_reason"] == (
        "no resolver capability request"
    )


@pytest.mark.asyncio
async def test_loop_records_timeout_observation_then_returns_to_cognition() -> None:
    """Capability timeouts should become observations, not Python decisions."""

    request = _resolver_request(objective="检索一个会超时的证据目标。")
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            return _cognition_result(
                internal_monologue="第一轮：需要证据",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="第二轮：看到超时阻塞",
            action_specs=[_speak_action_spec("证据工具超时，所以说明限制。")],
        )

    async def execute_capability(_request: dict, _state: dict) -> dict:
        await AsyncMock()()
        await asyncio.sleep(1.0)
        raise AssertionError("wait_for should timeout before this returns")

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=0.01,
    )

    assert len(cognition_inputs) == 2
    assert "timed out" in cognition_inputs[1]["resolver_context"]
    observation = result["resolver_state"]["observations"][0]
    assert observation["status"] == "failed"
    assert observation["capability_kind"] == "rag_evidence"
    assert observation["request_objective"] == request["objective"]
    assert "timed out" in observation["prompt_safe_summary"]


@pytest.mark.asyncio
async def test_loop_runs_final_cognition_with_max_cycle_blocker() -> None:
    """When capped, the blocker still returns through cognition."""

    request = _resolver_request(objective="持续检索仍然不足的证据。")
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            return _cognition_result(
                internal_monologue="第一轮：还想继续查",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="封顶轮：必须收束",
            action_specs=[_speak_action_spec("循环封顶后收束。")],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_partial",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "证据仍不足。",
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=1,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 2
    assert "max_cycles" in cognition_inputs[1]["resolver_context"]
    assert "maximum resolver cycles" in cognition_inputs[1]["resolver_context"]
    resolver_state = result["resolver_state"]
    assert resolver_state["status"] == "max_cycles"
    assert resolver_state["terminal_reason"] == "maximum resolver cycles reached"
    assert len(resolver_state["observations"]) == 2
    assert len(resolver_state["cycle_traces"]) == 2


@pytest.mark.asyncio
async def test_rag_capability_uses_objective_and_preserves_original_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG resolver execution should use objective while preserving context."""

    captured: dict = {}

    async def call_rag_supervisor(
        *,
        fresh_query: str,
        reply_context: dict,
        character_name: str,
        context: dict,
    ) -> dict[str, object]:
        captured["fresh_query"] = fresh_query
        captured["reply_context"] = reply_context
        captured["character_name"] = character_name
        captured["context"] = context
        result = {
            "answer": "找到一条关系记忆。",
            "known_facts": [
                {
                    "slot": "memory",
                    "agent": "memory_evidence_agent",
                    "resolved": True,
                    "summary": "存在一条信任相关记忆。",
                    "raw_result": {
                        "projection_payload": {"memory_rows": []},
                    },
                }
            ],
            "unknown_slots": [],
            "loop_count": 1,
        }
        return result

    monkeypatch.setattr(
        capabilities_module,
        "call_quote_aware_rag_supervisor",
        call_rag_supervisor,
    )
    monkeypatch.setattr(
        capabilities_module.event_logging,
        "record_rag_stage_event",
        AsyncMock(),
    )
    request = _resolver_request()

    observation = await capabilities_module.execute_resolver_capability_request(
        request,
        _resolver_state(),
    )

    assert captured["fresh_query"] == request["objective"]
    assert captured["context"]["original_user_request"] == (
        "Original user request about trust."
    )
    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "rag_evidence"
    assert observation["request_objective"] == request["objective"]
    assert observation["request_reason"] == request["reason"]
    assert observation["rag_result"]["answer"] == "找到一条关系记忆。"
    assert "memory_evidence" in observation["rag_result"]
    assert "user_image" in observation["rag_result"]


@pytest.mark.asyncio
async def test_empty_resolver_objective_fails_before_rag_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed resolver requests must fail before invoking RAG."""

    call_rag_supervisor = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "call_quote_aware_rag_supervisor",
        call_rag_supervisor,
    )
    request = _resolver_request(objective=" ")

    with pytest.raises(ResolverValidationError, match="objective"):
        await capabilities_module.execute_resolver_capability_request(
            request,
            _resolver_state(),
        )

    call_rag_supervisor.assert_not_awaited()


@pytest.mark.asyncio
async def test_blocked_capabilities_return_prompt_safe_observations() -> None:
    """Clarification and approval capabilities should block without side effects."""

    state = _resolver_state()

    clarification = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="human_clarification",
            objective="请只问用户所在城市。",
        ),
        state,
    )
    approval = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="approval_preparation",
            objective="说明准备创建提醒，但等待用户确认。",
        ),
        state,
    )

    assert clarification["status"] == "blocked"
    assert clarification["capability_kind"] == "human_clarification"
    assert "请只问用户所在城市" in clarification["prompt_safe_summary"]
    assert approval["status"] == "blocked"
    assert approval["capability_kind"] == "approval_preparation"
    assert "等待用户确认" in approval["prompt_safe_summary"]


@pytest.mark.asyncio
async def test_self_goal_resolution_blocks_user_message_source() -> None:
    """User-message turns must not spawn private self-goal execution."""

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="self_goal_resolution",
            objective="整理一个内部目标。",
        ),
        _resolver_state(),
    )

    assert observation["status"] == "blocked"
    assert observation["capability_kind"] == "self_goal_resolution"
