"""Tests for memory_retriever_agent.py — memory search tools and agent orchestration."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall

from kazusa_ai_chatbot.agents.memory_retriever_agent import (
    search_user_facts,
    search_conversation,
    search_conversation_keyword,
    get_conversation,
    search_persistent_memory,
    search_persistent_memory_keyword,
    memory_search_tool_call_generator,
    MemoryRetrieverState,
    _ALL_TOOLS,
    _TOOLS_BY_NAME,
)

logger = logging.getLogger(__name__)


class TestToolRegistration:
    def test_all_tools_list_has_6_tools(self):
        assert len(_ALL_TOOLS) == 6

    def test_tools_by_name_matches(self):
        assert set(_TOOLS_BY_NAME.keys()) == {t.name for t in _ALL_TOOLS}

    def test_expected_tool_names(self):
        expected = {
            "search_user_facts",
            "search_conversation",
            "search_conversation_keyword",
            "get_conversation",
            "search_persistent_memory",
            "search_persistent_memory_keyword",
        }
        assert set(_TOOLS_BY_NAME.keys()) == expected




@pytest.mark.asyncio
async def test_search_conversation_tool():
    """search_conversation tool should delegate to db.search_conversation_history."""
    mock_results = [
        (0.9, {"content": "hello", "timestamp": "t1", "channel_id": "c1", "user_id": "u1"}),
    ]
    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent.search_conversation_history", new_callable=AsyncMock, return_value=mock_results):
        result = await search_conversation.ainvoke({"search_query": "hello"})

    assert isinstance(result, list)
    assert len(result) == 1
    # Returns tuples of (score, message_dict)
    score, doc = result[0]
    assert score == 0.9
    assert doc["content"] == "hello"


@pytest.mark.asyncio
async def test_get_conversation_tool():
    """get_conversation tool should delegate to db.get_conversation_history."""
    mock_msgs = [
        {"content": "hi", "timestamp": "t1", "channel_id": "c1", "user_id": "u1", "embedding": [0.1]},
    ]
    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent.get_conversation_history", new_callable=AsyncMock, return_value=mock_msgs):
        result = await get_conversation.ainvoke({})

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["content"] == "hi"


@pytest.mark.asyncio
async def test_search_persistent_memory_tool():
    """search_persistent_memory tool should delegate to db.search_memory."""
    mock_results = [
        (0.85, {"memory_name": "test_mem", "content": "some data", "timestamp": "t1"}),
    ]
    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent.search_memory_db", new_callable=AsyncMock, return_value=mock_results):
        result = await search_persistent_memory.ainvoke({"search_query": "test"})

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["cosine_similarity"] == 0.85
    assert result[0]["memory_name"] == "test_mem"
    assert result[0]["content"] == "some data"


# ── Live LLM integration tests ─────────────────────────────────────
# Run with: pytest -m live_llm
#
# These tests call memory_search_tool_call_generator with a real LLM and assert
# on which tool was chosen and how the query/keyword was constructed.
# No real DB calls are made — tool execution is not triggered here.


def _make_state(task: str, context: dict | None = None) -> MemoryRetrieverState:
    """Minimal MemoryRetrieverState for testing the generator only."""
    return MemoryRetrieverState(
        task=task,
        context=context or {},
        next_tool="",
        expected_response="",
        messages=[],
        should_stop=False,
        retry=0,
        knowledge_metadata={},
        final_response="",
        final_status="",
        final_reason="",
    )


def _first_tool_call(result: MemoryRetrieverState) -> ToolCall:
    """Extract the first tool call from a generator result."""
    messages = result.get("messages", [])
    assert messages, "Generator produced no messages"
    ai_msg = messages[-1]
    assert isinstance(ai_msg, AIMessage), f"Expected AIMessage, got {type(ai_msg)}"
    assert ai_msg.tool_calls, "Generator produced no tool calls"
    return ai_msg.tool_calls[0]


def _log_live_routing(task: str, tool_call: ToolCall) -> None:
    logger.info("memory_retriever task=%r tool_call=%r", task, tool_call)


@pytest.mark.live_llm
class TestGeneratorToolRoutingLive:
    """Validate that the real LLM routes to keyword vs. vector tools correctly."""

    async def test_technical_term_routes_to_keyword_search(self):
        """'指令跟随逻辑' is a specific technical phrase — must use keyword search."""
        state = _make_state(
            task="检索用户关于'指令跟随逻辑'的对话记录",
            context={"target_global_user_id": "76a37e60-982e-45cb-af28-6d8c6b533297"},
        )
        result = await memory_search_tool_call_generator(state)
        tc = _first_tool_call(result)
        _log_live_routing(state["task"], tc)

        assert tc["name"] in ("search_conversation_keyword", "search_persistent_memory_keyword"), (
            f"Expected keyword tool, got '{tc['name']}' with args {tc['args']}"
        )
        keyword = tc["args"].get("keyword", "")
        assert len(keyword) <= 15, (
            f"Keyword should be a short term, not a full sentence; got: {keyword!r}"
        )
        assert "指令跟随" in keyword or "指令" in keyword, (
            f"Keyword should contain the target term; got: {keyword!r}"
        )

    async def test_product_name_routes_to_keyword_search(self):
        """'DDR5' is a specific product name — must use keyword search with a short term."""
        state = _make_state(
            task="检索用户关于DDR5内存的讨论记录",
            context={"target_global_user_id": "76a37e60-982e-45cb-af28-6d8c6b533297"},
        )
        result = await memory_search_tool_call_generator(state)
        tc = _first_tool_call(result)
        _log_live_routing(state["task"], tc)

        assert tc["name"] in ("search_conversation_keyword", "search_persistent_memory_keyword"), (
            f"Expected keyword tool for product name, got '{tc['name']}'"
        )
        keyword = tc["args"].get("keyword", "")
        assert "DDR5" in keyword.upper(), (
            f"Keyword should contain 'DDR5'; got: {keyword!r}"
        )

    async def test_semantic_impression_routes_to_vector_search(self):
        """Opinion/feeling queries have no exact wording — must use vector search."""
        state = _make_state(
            task="检索用户对杏山千纱的整体印象和情感态度",
            context={"target_global_user_id": "76a37e60-982e-45cb-af28-6d8c6b533297"},
        )
        result = await memory_search_tool_call_generator(state)
        tc = _first_tool_call(result)
        _log_live_routing(state["task"], tc)

        assert tc["name"] in ("search_conversation", "search_persistent_memory"), (
            f"Expected vector tool for semantic query, got '{tc['name']}'"
        )

    async def test_third_party_entity_does_not_call_search_user_facts(self):
        """search_user_facts requires a UUID — must never be called for a name-based entity."""
        state = _make_state(
            task="检索关于'Glitch'这个人与杏山千纱互动的记录",
            context={
                "target_global_user_id": "76a37e60-982e-45cb-af28-6d8c6b533297",
                "entities": ["Glitch"],
            },
        )
        result = await memory_search_tool_call_generator(state)
        messages = result.get("messages", [])
        ai_msg = messages[-1]
        called = [tc["name"] for tc in (ai_msg.tool_calls or [])]
        logger.info("memory_retriever task=%r tool_calls=%r", state["task"], ai_msg.tool_calls)

        assert "search_user_facts" not in called, (
            f"search_user_facts must not be called for third-party name entities; calls: {called}"
        )

    async def test_bug_report_case_uses_short_keyword(self):
        """Bug-report scenario: retrieve what the user actually said about 指令跟随逻辑.

        This is a lookup for the user's own statements on a specific technical term,
        so the generator must use keyword search with a short term — not a long
        semantic query that was causing the original vector search to fail.
        """
        state = _make_state(
            task="检索用户蚝爹油关于'指令跟随逻辑'的历史发言，以便了解他的具体观点内容",
            context={"target_global_user_id": "76a37e60-982e-45cb-af28-6d8c6b533297"},
        )
        result = await memory_search_tool_call_generator(state)
        tc = _first_tool_call(result)
        _log_live_routing(state["task"], tc)

        assert tc["name"] in ("search_conversation_keyword", "search_persistent_memory_keyword"), (
            f"Expected keyword tool for specific-term user lookup, got '{tc['name']}' args={tc['args']}"
        )
        keyword = tc["args"].get("keyword", "")
        assert len(keyword) <= 15, (
            f"Keyword must be short (≤15 chars), not a prose sentence; got: {keyword!r}"
        )
