"""Tests for RAG and dialog event logging boundaries."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.cognition_resolver import capabilities as capabilities_module
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2 as supervisor_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_dispatch as dispatch_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as rag2_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock


class _StaticLLM:
    """Return one configured model-compatible response."""

    def __init__(self, content: str) -> None:
        """Store the raw response content for the fake model."""

        self.content = content
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any], *, config=None) -> AIMessage:
        """Return the configured content and keep the prompt for inspection."""

        self.messages = list(messages)
        response = AIMessage(content=self.content)
        return response


def _serialized(value: object) -> str:
    """Serialize event call kwargs for privacy assertions."""

    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return rendered


def _minimal_episode() -> dict[str, object]:
    """Build a text-chat episode accepted by RAG request construction."""

    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-event-log",
        percept_id="percept-event-log",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="private stage input",
        platform="qq",
        platform_channel_id="private-channel",
        channel_type="group",
        platform_message_id="msg-event-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=["msg-event-1"],
        active_turn_conversation_row_ids=["row-event-1"],
        debug_modes={},
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    return episode


def _stage_1_state(*, referents: list[dict[str, object]]) -> dict[str, object]:
    """Build a persona state slice for direct stage-one RAG tests."""

    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    state = {
        "decontexualized_input": "private query text",
        "referents": referents,
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-1",
        },
        "platform": "qq",
        "platform_channel_id": "private-channel",
        "channel_type": "group",
        "platform_message_id": "msg-event-1",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "prompt_message_context": {
            "body_text": "private prompt context",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "private topic",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": {
            "status": "active",
            "continuity": "same_episode",
            "current_thread": "private thread",
        },
        "conversation_episode_state": {
            "updated_at": "2026-04-26T23:00:00+00:00",
            "turn_count": 1,
        },
        "cognitive_episode": _minimal_episode(),
    }
    return state


def _progressive_state() -> dict[str, object]:
    """Build the progressive RAG state needed by wrapper node tests."""

    state = {
        "original_query": "private original query",
        "character_name": "Kazusa",
        "context": {
            "platform": "qq",
            "platform_message_id": "msg-event-1",
            "platform_channel_id": "private-channel",
        },
        "unknown_slots": ["private slot text"],
        "current_slot": "private slot text",
        "known_facts": [],
        "messages": [],
        "initializer_cache": {},
        "current_dispatch": {
            "agent_name": "memory_evidence_agent",
            "task": "private task text",
            "context": {},
            "max_attempts": 1,
        },
        "last_agent_result": {},
        "loop_count": 1,
        "final_answer": "",
    }
    return state


def _dialog_global_state() -> dict[str, object]:
    """Build a dialog state with all required character voice fields."""

    state = {
        "internal_monologue": "private internal thought",
        "text_surface_output_v2": {
            "schema_version": "text_surface_output.v2",
            "content_plan": "Acknowledge the request.",
            "visible_boundaries": [],
            "addressee_plan": ["current user"],
            "style_guidance": "concise",
            "pacing_guidance": "one message",
            "selected_surface_intent": "acknowledge",
        },
        "chat_history_wide": [],
        "chat_history_recent": [],
        "debug_modes": {},
        "should_respond": True,
        "dialog_usage_mode": "live_visible_reply",
        "channel_type": "group",
        "use_reply_feature": False,
        "platform_user_id": "platform-user-1",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {
            "global_user_id": "user-1",
            "cognition_state": {"owner_user_id": "user-1"},
        },
        "character_profile": {
            "name": "Kazusa",
            "personality_brief": {
                "logic": "analytical",
                "tempo": "steady",
                "defense": "reserved",
                "quirks": "dry humor",
                "taboos": "break character",
                "mbti": "INTJ",
            },
            "linguistic_texture_profile": {
                "hesitation_density": 0.4,
                "fragmentation": 0.4,
                "emotional_leakage": 0.4,
                "rhythmic_bounce": 0.4,
                "direct_assertion": 0.4,
                "softener_density": 0.4,
                "counter_questioning": 0.4,
                "formalism_avoidance": 0.4,
                "abstraction_reframing": 0.4,
                "self_deprecation": 0.4,
            },
        },
    }
    return state


def _dialog_node_state() -> dict[str, object]:
    """Build a dialog-node state with graph-managed fields present."""

    state = _dialog_global_state()
    state.update(
        {
            "final_dialog": ["private generated dialog"],
            "target_addressed_user_ids": [],
            "target_broadcast": False,
        }
    )
    return state


def _local_context_packet(rag_result: dict[str, object]) -> dict[str, object]:
    """Build the minimal packet shape consumed by the capability wrapper."""

    return {
        "rag_result": rag_result,
        "graph": {
            "nodes": {
                "root": {},
                "task_1": {},
            },
        },
        "trace_summary": {
            "cache_hits": 0,
        },
    }


def _rag3_result(
    *,
    answer: str = "",
    memory_evidence: list[dict[str, object]] | None = None,
    supervisor_trace: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a retained RAG3 result payload for event tests."""

    return {
        "answer": answer,
        "user_image": {
            "user_memory_context": {
                "stable_patterns": [],
                "recent_shifts": [],
                "objective_facts": [],
                "milestones": [],
                "active_commitments": [],
            },
        },
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": list(memory_evidence or []),
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": supervisor_trace or {
            "resolver": "local_context_resolver",
            "node_count": 2,
        },
    }


@pytest.mark.asyncio
async def test_rag_evidence_skip_records_event_without_query_text(monkeypatch) -> None:
    """Unresolved referent skips should record only sanitized RAG metadata."""

    record_rag_stage_event = AsyncMock()
    resolve_local_context = AsyncMock()
    monkeypatch.setattr(
        capabilities_module.event_logging,
        "record_rag_stage_event",
        record_rag_stage_event,
    )
    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        resolve_local_context,
    )
    monkeypatch.setattr(
        capabilities_module,
        "project_local_context_packet",
        lambda packet: packet["rag_result"],
    )

    state = _stage_1_state(
        referents=[
            {
                "phrase": "that",
                "referent_role": "object",
                "status": "unresolved",
            },
        ],
    )

    rag_result = await supervisor_module.run_rag_evidence_for_persona_state(
        state,
        agent_name="resolver_rag_evidence",
    )

    assert rag_result["answer"] == ""
    resolve_local_context.assert_not_awaited()
    record_rag_stage_event.assert_awaited_once()
    kwargs = record_rag_stage_event.await_args.kwargs
    assert kwargs["status"] == "skipped"
    assert kwargs["agent_name"] == "resolver_rag_evidence"
    assert kwargs["correlation_id"] == "rag:qq:msg-event-1"
    assert "private query text" not in _serialized(kwargs)


@pytest.mark.asyncio
async def test_rag_evidence_success_records_counts_without_evidence(monkeypatch) -> None:
    """Successful RAG projection should count evidence without logging it."""

    record_rag_stage_event = AsyncMock()
    monkeypatch.setattr(
        capabilities_module.event_logging,
        "record_rag_stage_event",
        record_rag_stage_event,
    )

    async def resolve_local_context(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        """Return one projected memory evidence row."""

        rag_result = _rag3_result(
            answer="private synthesized answer",
            memory_evidence=[{
                "content": "secret retrieved evidence",
                "source_system": "memory",
            }],
        )
        return _local_context_packet(rag_result)

    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        resolve_local_context,
    )
    monkeypatch.setattr(
        capabilities_module,
        "project_local_context_packet",
        lambda packet: packet["rag_result"],
    )

    state = _stage_1_state(referents=[])

    rag_result = await supervisor_module.run_rag_evidence_for_persona_state(
        state,
        agent_name="resolver_rag_evidence",
    )

    assert rag_result["memory_evidence"]
    record_rag_stage_event.assert_awaited_once()
    kwargs = record_rag_stage_event.await_args.kwargs
    assert kwargs["status"] == "succeeded"
    assert kwargs["retrieval_count"] == 1
    event_text = _serialized(kwargs)
    assert "secret retrieved evidence" not in event_text
    assert "private synthesized answer" not in event_text


@pytest.mark.asyncio
async def test_rag_evidence_success_records_safety_recovery_count(monkeypatch) -> None:
    """RAG telemetry should expose recovery counts without unsafe content."""

    record_rag_stage_event = AsyncMock()
    monkeypatch.setattr(
        capabilities_module.event_logging,
        "record_rag_stage_event",
        record_rag_stage_event,
    )

    async def resolve_local_context(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        """Return a projected result with one safety recovery incident."""

        rag_result = _rag3_result(
            supervisor_trace={
                "resolver": "local_context_resolver",
                "node_count": 2,
                "safety_recovery": [
                    "dropped_text_line:rag_result.answer",
                ],
            },
        )
        return _local_context_packet(rag_result)

    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        resolve_local_context,
    )
    monkeypatch.setattr(
        capabilities_module,
        "project_local_context_packet",
        lambda packet: packet["rag_result"],
    )

    state = _stage_1_state(referents=[])

    rag_result = await supervisor_module.run_rag_evidence_for_persona_state(
        state,
        agent_name="resolver_rag_evidence",
    )

    assert rag_result["answer"] == ""
    record_rag_stage_event.assert_awaited_once()
    kwargs = record_rag_stage_event.await_args.kwargs
    assert kwargs["safety_recovery_count"] > 0
    assert kwargs["safety_recovery_first"] == "dropped_text_line:rag_result.answer"
    assert "[CQ:" not in _serialized(kwargs)


@pytest.mark.asyncio
async def test_rag_dispatcher_records_llm_metadata_without_slot_text(monkeypatch) -> None:
    """RAG dispatcher telemetry should store counts and status, not slot text."""

    record_llm_stage_event = AsyncMock()
    record_model_contract_event = AsyncMock()
    dispatch = {
        "agent_name": "user_lookup_agent",
        "task": "private dispatched task",
        "context": {},
        "max_attempts": 1,
    }
    monkeypatch.setattr(dispatch_module, "_dispatcher_llm", _StaticLLM(json.dumps(dispatch)))
    monkeypatch.setattr(
        dispatch_module.event_logging,
        "record_llm_stage_event",
        record_llm_stage_event,
    )
    monkeypatch.setattr(
        dispatch_module.event_logging,
        "record_model_contract_event",
        record_model_contract_event,
    )

    state = _progressive_state()
    state["unknown_slots"] = ["unprefixed private dispatch slot"]
    result = await dispatch_module.rag_dispatcher(state)

    assert result["current_dispatch"]["agent_name"] == "user_lookup_agent"
    record_llm_stage_event.assert_awaited_once()
    record_model_contract_event.assert_not_awaited()
    kwargs = record_llm_stage_event.await_args.kwargs
    assert kwargs["stage_name"] == "rag_dispatcher"
    assert kwargs["route_name"] == "dispatcher_llm"
    event_text = _serialized(kwargs)
    assert "unprefixed private dispatch slot" not in event_text
    assert "private dispatched task" not in event_text
    assert "private-channel" not in event_text


@pytest.mark.asyncio
async def test_rag_executor_wrapper_records_metadata_without_raw_result(
    monkeypatch,
) -> None:
    """RAG executor wrapper should not copy helper evidence into telemetry."""

    record_rag_stage_event = AsyncMock()
    monkeypatch.setattr(
        rag2_module.event_logging,
        "record_rag_stage_event",
        record_rag_stage_event,
    )

    async def rag_executor(state: dict[str, object]) -> dict[str, object]:
        """Return one raw helper result containing private evidence."""

        del state
        result = {
            "last_agent_result": {
                "agent": "memory_evidence_agent",
                "resolved": True,
                "result": {"evidence": ["secret helper evidence"]},
                "attempts": 1,
            },
            "messages": [],
        }
        return result

    monkeypatch.setattr(rag2_module._dispatch_domain, "rag_executor", rag_executor)

    state = _progressive_state()
    result = await rag2_module.rag_executor(state)

    assert result["last_agent_result"]["resolved"] is True
    record_rag_stage_event.assert_awaited_once()
    kwargs = record_rag_stage_event.await_args.kwargs
    assert kwargs["agent_name"] == "memory_evidence_agent"
    assert kwargs["retrieval_count"] == 1
    event_text = _serialized(kwargs)
    assert "secret helper evidence" not in event_text
    assert "private task text" not in event_text


@pytest.mark.asyncio
async def test_dialog_generator_records_llm_metadata_without_generated_text(
    monkeypatch,
) -> None:
    """Dialog generator telemetry should omit generated dialog fragments."""

    record_llm_stage_event = AsyncMock()
    record_model_contract_event = AsyncMock()
    llm = _StaticLLM(
        '{"final_dialog": ["secret generated dialog"]}'
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", llm)
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_llm_stage_event",
        record_llm_stage_event,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_model_contract_event",
        record_model_contract_event,
    )

    state = _dialog_node_state()
    result = await dialog_module.dialog_generator(state)

    assert result["final_dialog"] == ["secret generated dialog"]
    record_llm_stage_event.assert_awaited_once()
    record_model_contract_event.assert_not_awaited()
    kwargs = record_llm_stage_event.await_args.kwargs
    assert kwargs["stage_name"] == "dialog_generator"
    assert "secret generated dialog" not in _serialized(kwargs)


@pytest.mark.asyncio
async def test_dialog_agent_records_quality_without_dialog_text(monkeypatch) -> None:
    """Full dialog graph should record quality metadata after normal output."""

    record_dialog_quality_event = AsyncMock()
    record_llm_stage_event = AsyncMock()
    record_model_contract_event = AsyncMock()
    monkeypatch.setattr(
        dialog_module,
        "_dialog_generator_llm",
        _StaticLLM(
            '{"final_dialog": ["secret full graph reply"]}'
        ),
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_dialog_quality_event",
        record_dialog_quality_event,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_llm_stage_event",
        record_llm_stage_event,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_model_contract_event",
        record_model_contract_event,
    )

    result = await dialog_module.dialog_agent(_dialog_global_state())

    assert result["final_dialog"] == ["secret full graph reply"]
    record_dialog_quality_event.assert_awaited_once()
    kwargs = record_dialog_quality_event.await_args.kwargs
    assert kwargs["usage_mode"] == "live_visible_reply"
    assert kwargs["quality_status"] == "passed"
    assert kwargs["retry_count"] == 0
    assert kwargs["content_plan_entry_count"] == 1
    assert "secret full graph reply" not in _serialized(kwargs)
