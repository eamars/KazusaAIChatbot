from __future__ import annotations

import pytest
pytest.skip("Stage 1 assertions replaced by the V2 contract suite", allow_module_level=True)

import json

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    build_text_chat_cognitive_episode,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from llm_test_helpers import bind_test_llm


_TURN_CLOCK = build_turn_clock("2026-05-09 19:30:00")


class _DummyResponse:
    """Small LangChain-like response wrapper for cognition tests."""

    def __init__(self, content: str) -> None:
        """Create a dummy response.

        Args:
            content: LLM response content.
        """
        self.content = content


class _CapturingAsyncLLM:
    """Async LLM fake that records messages and returns a JSON payload."""

    def __init__(self, payload: dict) -> None:
        """Create a capturing fake LLM.

        Args:
            payload: JSON-serializable response payload.
        """
        self.payload = payload
        self.messages: list = []

    async def ainvoke(self, messages: list, *, config=None) -> _DummyResponse:
        """Record prompt messages and return the configured payload.

        Args:
            messages: Prompt messages supplied by the caller.

        Returns:
            Dummy response with JSON content.
        """
        self.messages = messages
        return _DummyResponse(json.dumps(self.payload))


def _cognitive_episode() -> CognitiveEpisode:
    """Build a valid text-chat episode for direct handler tests.

    Returns:
        Valid Stage 02 text-chat cognitive episode.
    """
    return build_text_chat_cognitive_episode(
        episode_id="clarification-consumer-episode",
        percept_id="clarification-consumer-percept",
        storage_timestamp_utc=_TURN_CLOCK["storage_timestamp_utc"],
        local_time_context=_TURN_CLOCK["local_time_context"],
        user_input="hello",
        platform="qq",
        platform_channel_id="private-channel",
        channel_type="private",
        platform_message_id="message-1",
        platform_user_id="platform-user-1",
        global_user_id="global-user-1",
        user_name="User",
        active_turn_platform_message_ids=["message-1"],
        active_turn_conversation_row_ids=[],
        debug_modes={},
    )


def _judgment_state() -> dict:
    """Build the minimal state required by Judgment Core.

    Returns:
        Cognition-state subset for the L2c test.
    """
    return {
        "character_profile": {"name": "Kazusa"},
        "user_profile": {"relationship_state": 500},
        "internal_monologue": "I might answer if I knew the object.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "cognitive_episode": _cognitive_episode(),
        "user_input": "hello",
        "prompt_message_context": {},
        "reply_context": {},
        "user_name": "User",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
        ],
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "boundary_summary": "none",
            "behavior_primary": "comply",
            "behavior_secondary": "none",
            "acceptance": "allow",
            "stance_bias": "confirm",
            "identity_policy": "accept",
            "pressure_policy": "absorb",
            "trajectory": "stable",
        },
    }


def _content_plan_state() -> dict:
    """Build the minimal state required by the content-plan agent.

    Returns:
        Cognition-state subset for the L3b test.
    """
    return {
        "character_profile": {"name": "Kazusa"},
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
        ],
        "rag_result": {
            "answer": "",
            "user_image": {},
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {"unknown_slots": [], "loop_count": 0},
        },
        "internal_monologue": "I need to ask what these refers to.",
        "logical_stance": "TENTATIVE",
        "character_intent": "CLARIFY",
        "cognitive_episode": _cognitive_episode(),
        "user_input": "hello",
        "prompt_message_context": {},
        "reply_context": {},
        "user_name": "User",
        "conversation_progress": None,
    }


@pytest.mark.asyncio
async def test_judgment_core_consumes_unresolved_referents(monkeypatch) -> None:
    """Judgment Core should force clarification from structured referents."""
    fake_llm = _CapturingAsyncLLM({
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "would otherwise answer",
    })
    monkeypatch.setattr(l2_module, "_judgement_core_llm", bind_test_llm(fake_llm, "judgement_core_llm"))

    result = await l2_module.call_judgment_core_agent(_judgment_state())

    payload = json.loads(fake_llm.messages[1].content)
    assert payload["referents"] == [
        {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
    ]
    assert result["logical_stance"] == "TENTATIVE"
    assert result["character_intent"] == "CLARIFY"
    assert "不要用宽泛旧上下文" in result["judgment_note"]


@pytest.mark.asyncio
async def test_judgment_core_requires_referents(monkeypatch) -> None:
    """Judgment Core should enforce the structured referents contract."""

    fake_llm = _CapturingAsyncLLM({
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "would otherwise answer",
    })
    monkeypatch.setattr(l2_module, "_judgement_core_llm", bind_test_llm(fake_llm, "judgement_core_llm"))

    state = _judgment_state()
    state.pop("referents")

    with pytest.raises(KeyError):
        await l2_module.call_judgment_core_agent(state)


@pytest.mark.asyncio
async def test_content_plan_agent_receives_referent_signal(monkeypatch) -> None:
    """Content-plan agent should receive referent clarification fields."""
    fake_llm = _CapturingAsyncLLM({
        "content_plan": {
            "visible_goal": "先追问缺失对象。",
            "semantic_content": "你说的这些具体是指什么？",
            "rendering": "简短追问即可。",
        }
    })
    monkeypatch.setattr(l3_module, "_content_plan_agent_llm", bind_test_llm(fake_llm, "content_plan_agent_llm"))

    result = await l3_module.call_content_plan_agent(_content_plan_state())

    payload = json.loads(fake_llm.messages[1].content)
    assert payload["referents"] == [
        {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
    ]
    assert result["content_plan"]["semantic_content"] == "你说的这些具体是指什么？"


@pytest.mark.asyncio
async def test_content_plan_agent_requires_referents(monkeypatch) -> None:
    """Content-plan agent should enforce the structured referents contract."""
    fake_llm = _CapturingAsyncLLM({
        "content_plan": {
            "visible_goal": "先追问缺失对象。",
            "semantic_content": "你说的这些具体是指什么？",
            "rendering": "简短追问即可。",
        }
    })
    monkeypatch.setattr(l3_module, "_content_plan_agent_llm", bind_test_llm(fake_llm, "content_plan_agent_llm"))

    state = _content_plan_state()
    state.pop("referents")

    with pytest.raises(KeyError):
        await l3_module.call_content_plan_agent(state)
