from __future__ import annotations

import pytest
pytest.skip("Stage 1 assertions replaced by the V2 contract suite", allow_module_level=True)

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.nodes import persona_supervisor2 as supervisor_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from llm_test_helpers import bind_test_llm


class _DummyResponse:
    """Small LangChain-like response wrapper for cognition shape tests."""

    def __init__(self, content: str) -> None:
        """Create a dummy response.

        Args:
            content: LLM response content.
        """
        self.content = content


class _StaticAsyncLLM:
    """Async LLM fake returning a fixed content-plan payload."""

    async def ainvoke(self, _messages: list, *, config=None) -> _DummyResponse:
        """Return a minimal content-plan response.

        Args:
            _messages: Prompt messages supplied by the caller.

        Returns:
            Dummy response with JSON content.
        """
        return_value = _DummyResponse(
            '{"content_plan": {"visible_goal": "先澄清对象", '
            '"semantic_content": "你说的这些具体是指什么？", '
            '"rendering": "简短追问"}}'
        )
        return return_value


def _clarification_state() -> dict:
    """Build a persona state whose unresolved reference skips RAG.

    Returns:
        Global persona-state subset for the RAG routing node.
    """
    return {
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"},
        ],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"relationship_state": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
    }


def _minimal_text_chat_episode() -> dict:
    """Build a valid text-chat cognitive episode for cognition node fixtures.

    Returns:
        Valid user-message cognitive episode.
    """
    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-rag-skip",
        percept_id="percept-rag-skip",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="clean body",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        platform_message_id="msg-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=[],
        active_turn_conversation_row_ids=[],
        debug_modes={},
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    return episode


@pytest.mark.asyncio
async def test_rag_evidence_skip_preserves_projected_shape(monkeypatch) -> None:
    """Skipped RAG should still return the full cognition-consumed payload."""

    async def _resolve_local_context(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError(
            "local-context resolver should not run when clarification is needed"
        )

    monkeypatch.setattr(
        capabilities,
        "resolve_local_context",
        _resolve_local_context,
    )

    rag_result = await supervisor_module.run_rag_evidence_for_persona_state(
        _clarification_state(),
        agent_name="resolver_rag_evidence",
    )

    assert rag_result["user_image"]["user_memory_context"]["stable_patterns"] == []
    assert rag_result["character_image"] == {}
    assert rag_result["memory_evidence"] == []
    assert rag_result["conversation_evidence"] == []
    assert rag_result["external_evidence"] == []
    assert rag_result["supervisor_trace"]["unknown_slots"] == []
    assert rag_result["supervisor_trace"]["loop_count"] == 0


@pytest.mark.asyncio
async def test_skip_branch_does_not_call_adapter(monkeypatch) -> None:
    """Skipped RAG should not invoke the local-context resolver."""

    async def _resolve_local_context(*_args: object, **_kwargs: object) -> dict:
        """Fail if the skip branch reaches resolver execution."""
        raise AssertionError(
            "local-context resolver should not run when clarification is needed"
        )

    monkeypatch.setattr(
        capabilities,
        "resolve_local_context",
        _resolve_local_context,
    )

    rag_result = await supervisor_module.run_rag_evidence_for_persona_state(
        _clarification_state(),
        agent_name="resolver_rag_evidence",
    )

    assert rag_result["answer"] == ""
    assert rag_result["supervisor_trace"]["loop_count"] == 0


@pytest.mark.asyncio
async def test_content_plan_accepts_skipped_rag_result_shape(monkeypatch) -> None:
    """Content-plan agent should not raise on skipped-RAG projection shape."""
    monkeypatch.setattr(
        l3_module,
        "_content_plan_agent_llm",
        bind_test_llm(_StaticAsyncLLM(), "content_plan_agent"),
    )
    rag_result = await supervisor_module.run_rag_evidence_for_persona_state(
        _clarification_state(),
        agent_name="resolver_rag_evidence",
    )

    result = await l3_module.call_content_plan_agent({
        "character_profile": {"name": "Kazusa"},
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"},
        ],
        "rag_result": rag_result,
        "internal_monologue": "I need to ask what these refers to.",
        "logical_stance": "TENTATIVE",
        "character_intent": "CLARIFY",
        "conversation_progress": None,
        "cognitive_episode": _minimal_text_chat_episode(),
        "user_input": "clean body",
        "prompt_message_context": {},
        "reply_context": {},
        "user_name": "User",
    })

    assert result["content_plan"]["semantic_content"] == "你说的这些具体是指什么？"
