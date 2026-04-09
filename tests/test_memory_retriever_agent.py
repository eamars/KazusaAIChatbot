from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from kazusa_ai_chatbot.agents.memory_retriever_agent import MemoryRetrieverAgent
from kazusa_ai_chatbot.state import BotState


@pytest.mark.asyncio
async def test_memory_retriever_agent_success_flow():
    agent = MemoryRetrieverAgent()
    bot_state = BotState()

    # Mock the LLM
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    mock_llm_with_tools = AsyncMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    # Generator response: tool call to search_user_facts
    generator_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_user_facts",
            "args": {"user_id": "test_user"},
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
            "response": "User fact is ABC.",
            "score": 90,
            "reason": "Accurate information found."
        })
    )
    mock_llm.ainvoke.side_effect = [evaluator_response, finalizer_response]

    # Mock the tools
    mock_search_user_facts = MagicMock()
    mock_search_user_facts.name = "search_user_facts"
    mock_search_user_facts.ainvoke = AsyncMock(return_value=["Fact 1", "Fact 2"])

    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.memory_retriever_agent._llm_with_tools", mock_llm_with_tools), \
         patch("kazusa_ai_chatbot.agents.memory_retriever_agent._TOOLS_BY_NAME", {"search_user_facts": mock_search_user_facts}), \
         patch("kazusa_ai_chatbot.agents.memory_retriever_agent._ALL_TOOLS", [mock_search_user_facts]):
        
        result = await agent.run(
            state=bot_state,
            task="Find user fact",
            expected_response="A short summary"
        )

    assert result["agent"] == "memory_retriever_agent"
    assert result["status"] == "complete"
    assert result["summary"] == "User fact is ABC."


@pytest.mark.asyncio
async def test_memory_retriever_agent_partial_flow():
    agent = MemoryRetrieverAgent()
    bot_state = BotState()

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    mock_llm_with_tools = AsyncMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    # Generator response: tool call
    generator_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_memory",
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

    mock_search_memory = MagicMock()
    mock_search_memory.name = "search_memory"
    mock_search_memory.ainvoke = AsyncMock(return_value="Partial snippet.")

    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.memory_retriever_agent._llm_with_tools", mock_llm_with_tools), \
         patch("kazusa_ai_chatbot.agents.memory_retriever_agent._TOOLS_BY_NAME", {"search_memory": mock_search_memory}), \
         patch("kazusa_ai_chatbot.agents.memory_retriever_agent._ALL_TOOLS", [mock_search_memory]):
        
        result = await agent.run(
            state=bot_state,
            task="Find partial query",
            expected_response="Details"
        )

    assert result["status"] == "partial"
    assert result["summary"] == "Only found some details."


@pytest.mark.asyncio
async def test_memory_retriever_agent_max_iterations_reached():
    agent = MemoryRetrieverAgent()
    bot_state = BotState()

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
                "name": "search_conversation",
                "args": {"search_query": "looping query"},
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

    mock_search_conversation = MagicMock()
    mock_search_conversation.name = "search_conversation"
    mock_search_conversation.ainvoke = AsyncMock(return_value="Still looking...")

    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent._get_llm", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.memory_retriever_agent._llm_with_tools", mock_llm_with_tools), \
         patch("kazusa_ai_chatbot.agents.memory_retriever_agent._TOOLS_BY_NAME", {"search_conversation": mock_search_conversation}), \
         patch("kazusa_ai_chatbot.agents.memory_retriever_agent._ALL_TOOLS", [mock_search_conversation]):
        
        result = await agent.run(
            state=bot_state,
            task="Find looping query",
            expected_response="Eventually stops"
        )

    assert result["status"] == "incomplete"
    assert result["summary"] == "Stopped after max iterations."
    # The tool should be called exactly MAX_TOOL_ITERATIONS times
    assert mock_search_conversation.ainvoke.call_count == MAX_TOOL_ITERATIONS
