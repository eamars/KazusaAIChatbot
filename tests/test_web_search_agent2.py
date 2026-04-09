from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from kazusa_ai_chatbot.agents.web_search_agent2 import WebSearchAgent2
from kazusa_ai_chatbot.state import BotState


@pytest.mark.asyncio
async def test_web_search_agent2_success_flow():
    agent = WebSearchAgent2()
    bot_state = BotState()
    bot_state["timestamp"] = "2026-04-09T12:00:00+00:00"

    # Mock the LLM
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    mock_llm_with_tools = AsyncMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    # Generator response: tool call to web_search
    generator_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "web_search",
            "args": {"query": "test query"},
            "id": "tool_1"
        }]
    )
    mock_llm_with_tools.ainvoke.return_value = generator_response

    # Evaluator response: should_stop = True
    evaluator_response = AIMessage(
        content=json.dumps({
            "feedback": "Search complete",
            "should_stop": True
        })
    )
    # Finalizer response: final answer
    finalizer_response = AIMessage(
        content=json.dumps({
            "response": "Here is the test result.",
            "score": 90,
            "reason": "Accurate information found."
        })
    )
    mock_llm.ainvoke.side_effect = [evaluator_response, finalizer_response]

    # Mock the tools
    mock_web_search = MagicMock()
    mock_web_search.name = "web_search"
    mock_web_search.ainvoke = AsyncMock(return_value="Test search results snippet.")
    mock_web_url_read = MagicMock()
    mock_web_url_read.name = "web_url_read"
    mock_web_url_read.ainvoke = AsyncMock()

    with patch("kazusa_ai_chatbot.agents.web_search_agent2._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.web_search_agent2._TOOLS_BY_NAME", {"web_search": mock_web_search, "web_url_read": mock_web_url_read}), \
         patch("kazusa_ai_chatbot.agents.web_search_agent2._ALL_TOOLS", [mock_web_search, mock_web_url_read]):
        
        result = await agent.run(
            state=bot_state,
            task="Find test query",
            expected_response="A short summary"
        )

    assert result["agent"] == "web_search_agent2"
    assert result["status"] == "success"
    assert result["summary"] == "Here is the test result."


@pytest.mark.asyncio
async def test_web_search_agent2_partial_flow():
    agent = WebSearchAgent2()
    bot_state = BotState()
    bot_state["timestamp"] = "2026-04-09T12:00:00+00:00"

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    mock_llm_with_tools = AsyncMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    # Generator response: tool call to web_search
    generator_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "web_search",
            "args": {"query": "partial query"},
            "id": "tool_1"
        }]
    )
    mock_llm_with_tools.ainvoke.return_value = generator_response

    # Evaluator response: should_stop = True
    evaluator_response = AIMessage(
        content=json.dumps({
            "feedback": "Partially found",
            "should_stop": True
        })
    )
    # Finalizer response: partial score
    finalizer_response = AIMessage(
        content=json.dumps({
            "response": "Only found some details.",
            "score": 60,
            "reason": "Missing some parts."
        })
    )
    mock_llm.ainvoke.side_effect = [evaluator_response, finalizer_response]

    mock_web_search = MagicMock()
    mock_web_search.name = "web_search"
    mock_web_search.ainvoke = AsyncMock(return_value="Partial snippet.")

    with patch("kazusa_ai_chatbot.agents.web_search_agent2._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.web_search_agent2._TOOLS_BY_NAME", {"web_search": mock_web_search}), \
         patch("kazusa_ai_chatbot.agents.web_search_agent2._ALL_TOOLS", [mock_web_search]):
        
        result = await agent.run(
            state=bot_state,
            task="Find partial query",
            expected_response="Details"
        )

    assert result["status"] == "partial"
    assert result["summary"] == "Only found some details."


@pytest.mark.asyncio
async def test_web_search_agent2_max_iterations_reached():
    agent = WebSearchAgent2()
    bot_state = BotState()
    bot_state["timestamp"] = "2026-04-09T12:00:00+00:00"

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    mock_llm_with_tools = AsyncMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    # Generator will always return a new tool call
    def generator_side_effect(*args, **kwargs):
        import uuid
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "web_search",
                "args": {"query": "looping query"},
                "id": str(uuid.uuid4())
            }]
        )
    mock_llm_with_tools.ainvoke.side_effect = generator_side_effect

    # Evaluator always says should_stop = False
    evaluator_response = AIMessage(
        content=json.dumps({
            "feedback": "Keep searching",
            "should_stop": False
        })
    )
    
    # After MAX_TOOL_ITERATIONS (e.g. 5), the graph forces it to stop and calls finalizer
    finalizer_response = AIMessage(
        content=json.dumps({
            "response": "Stopped after max iterations.",
            "score": 10,
            "reason": "Could not find final answer."
        })
    )
    
    # Evaluator is called MAX_TOOL_ITERATIONS times, then Finalizer is called once
    from kazusa_ai_chatbot.config import MAX_TOOL_ITERATIONS
    mock_llm.ainvoke.side_effect = [evaluator_response] * MAX_TOOL_ITERATIONS + [finalizer_response]

    mock_web_search = MagicMock()
    mock_web_search.name = "web_search"
    mock_web_search.ainvoke = AsyncMock(return_value="Still looking...")

    with patch("kazusa_ai_chatbot.agents.web_search_agent2._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.web_search_agent2._TOOLS_BY_NAME", {"web_search": mock_web_search}), \
         patch("kazusa_ai_chatbot.agents.web_search_agent2._ALL_TOOLS", [mock_web_search]):
        
        result = await agent.run(
            state=bot_state,
            task="Find looping query",
            expected_response="Eventually stops"
        )

    assert result["status"] == "not_found"
    assert result["summary"] == "Stopped after max iterations."
    # The tool should be called exactly MAX_TOOL_ITERATIONS times
    assert mock_web_search.ainvoke.call_count == MAX_TOOL_ITERATIONS
