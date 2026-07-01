"""Tests for L2d resolver capability selection contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_PENDING_RESOLUTION_VERSION,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l1 as l1_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2 as l2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2c2 as l2c2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2d as l2d_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from llm_test_helpers import bind_test_llm


class _FakeLLM:
    """Capture the L2d prompt call and return one configured JSON payload."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any], *, config=None) -> SimpleNamespace:
        self.messages = messages
        response = SimpleNamespace(content=self.content)
        return response


def _resolver_request() -> dict:
    return {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": "local_context_recall",
        "objective": "检索当前用户与这个问题有关的关系和记忆证据。",
        "reason": "没有记忆证据时无法可靠判断。",
        "priority": "now",
    }


def _pending_resolution() -> dict:
    return {
        "schema_version": RESOLVER_PENDING_RESOLUTION_VERSION,
        "resume_id": "resolver-pending-001",
        "decision": "answered",
        "reason": "用户已经补充了缺失地点。",
    }


def _episode() -> dict:
    turn_clock = build_turn_clock("2026-05-30 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-123",
        percept_id="percept-123",
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
        target_addressed_user_ids=["character-123"],
        target_broadcast=False,
    )
    return episode


def _l2d_state() -> dict:
    turn_clock = build_turn_clock("2026-05-30 09:00:00")
    return {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "character_profile": {
            "name": "Test Character",
            "global_user_id": "character-123",
        },
        "user_input": "Need an evidence-backed answer.",
        "prompt_message_context": {
            "body_text": "Need an evidence-backed answer.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-123"],
            "broadcast": False,
        },
        "cognitive_episode": _episode(),
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "channel_type": "private",
        "global_user_id": "global-user-123",
        "user_name": "Test User",
        "user_profile": {"last_relationship_insight": "stable"},
        "platform_bot_id": "bot-123",
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "debug",
        "decontexualized_input": "Need an evidence-backed answer.",
        "referents": [],
        "rag_result": {
            "answer": "",
            "user_image": {"user_memory_context": {"active_commitments": []}},
            "memory_evidence": [],
        },
        "emotional_appraisal": "calm",
        "interaction_subtext": "direct request",
        "internal_monologue": "Evidence is missing.",
        "character_intent": "CLARIFY",
        "logical_stance": "TENTATIVE",
        "judgment_note": "Need retrieval before final answer.",
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "acceptance": "allow",
            "stance_bias": "tentative",
        },
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "calm",
        "relational_dynamic": "stable",
        "resolver_context": (
            "resolver_state: status=running; cycle_index=0; max_cycles=3"
        ),
    }


def _persona_state() -> dict:
    state = _l2d_state()
    state.update({
        "platform_message_id": "message-123",
        "platform_user_id": "platform-user-123",
        "user_multimedia_input": [],
        "chat_history_wide": [],
        "debug_modes": {},
        "should_respond": True,
    })
    return state


@pytest.mark.asyncio
async def test_action_selection_returns_resolver_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2d should return resolver requests before terminal action specs."""

    fake_llm = _FakeLLM(json.dumps({
        "resolver_capability_requests": [_resolver_request()],
        "semantic_action_requests": [
            {
                "capability": "speak",
                "decision": "visible_reply",
                "detail": "Answer now.",
                "reason": "This should be ignored while resolving.",
            }
        ],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_selection_llm", bind_test_llm(fake_llm, "action_selection_llm"))

    result = await l2d_module.select_semantic_actions(_l2d_state())

    assert result["semantic_action_requests"] == []
    assert result["resolver_capability_requests"] == [_resolver_request()]
    prompt_payload = fake_llm.messages[1].content
    assert "resolver_state: status=running" in prompt_payload


@pytest.mark.asyncio
async def test_cognition_subgraph_propagates_resolver_requests() -> None:
    """Resolver requests from L2d must return to the persona graph."""

    async def l1_agent(_state: dict) -> dict:
        return {
            "emotional_appraisal": "calm",
            "interaction_subtext": "direct request",
        }

    async def l2a_agent(state: dict) -> dict:
        assert state["resolver_context"].startswith("resolver_state:")
        return {
            "internal_monologue": "Evidence is missing.",
            "logical_stance": "TENTATIVE",
            "character_intent": "CLARIFY",
        }

    async def l2b_agent(_state: dict) -> dict:
        return {
            "boundary_core_assessment": {
                "boundary_issue": "none",
                "acceptance": "allow",
                "stance_bias": "tentative",
            },
        }

    async def l2c1_agent(_state: dict) -> dict:
        return {
            "logical_stance": "TENTATIVE",
            "character_intent": "CLARIFY",
            "judgment_note": "Need retrieval before final answer.",
        }

    async def l2c2_agent(_state: dict) -> dict:
        return {
            "social_distance": "friendly",
            "emotional_intensity": "low",
            "vibe_check": "calm",
            "relational_dynamic": "stable",
        }

    async def l2d_agent(state: dict) -> dict:
        assert state["resolver_context"].startswith("resolver_state:")
        return {
            "semantic_action_requests": [],
            "resolver_capability_requests": [_resolver_request()],
            "resolver_pending_resolution": _pending_resolution(),
        }

    with (
        patch.object(l1_module, "call_cognition_subconscious", l1_agent),
        patch.object(l2_module, "call_cognition_consciousness", l2a_agent),
        patch.object(l2_module, "call_boundary_core_agent", l2b_agent),
        patch.object(l2_module, "call_judgment_core_agent", l2c1_agent),
        patch.object(
            l2c2_module,
            "call_social_context_appraisal",
            l2c2_agent,
        ),
        patch.object(l2d_module, "select_semantic_actions", l2d_agent),
    ):
        result = await cognition_module.call_cognition_subgraph(_persona_state())

    assert result["resolver_capability_requests"] == [_resolver_request()]
    assert result["resolver_pending_resolution"] == _pending_resolution()
    assert result["action_specs"] == []
