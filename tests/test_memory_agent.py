from __future__ import annotations

from unittest.mock import ANY, AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.agents.memory_agent import MemoryAgent


@pytest.mark.asyncio
async def test_memory_agent_recalls_stored_memory():
    agent = MemoryAgent()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(
            content='I will search for relevant memory about the embedding guide.',
            tool_calls=[{
                "name": "recall_memory",
                "args": {"query": "embedding guide", "method": "vector", "limit": 3},
                "id": "tool_call_1",
                "type": "tool_call"
            }]
        ),
        AIMessage(content='''{"status": "success", "summary": "Stored memory says the embedding guide explained that vector search should be used for semantic recall."}'''),
    ])

    with patch("kazusa_ai_chatbot.agents.memory_agent._get_llm_with_tools", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.memory_agent.search_memory", new_callable=AsyncMock, return_value=[
             (0.92, {
                 "memory_name": "Embedding guide",
                 "content": "Use vector search for semantic recall and keyword search as a fallback.",
             })
         ]):
        result = await agent.run(
            {"user_id": "user_123", "channel_id": "chan_456", "message_text": "Do you remember the embedding guide?"},
            "Do you remember the embedding guide?",
            "Recall any stored memory about the embedding guide.",
        )

    assert result["status"] == "success"
    assert "embedding guide" in result["summary"].lower()
    assert result["tool_history"][0]["tool"] == "recall_memory"
    assert result["tool_history"][0]["args"]["query"] == "embedding guide"


@pytest.mark.asyncio
async def test_memory_agent_checks_existing_memory_before_saving_and_overwrites():
    agent = MemoryAgent()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(
            content='I will first check if memory exists about the langgraph supervisor article.',
            tool_calls=[{
                "name": "recall_memory",
                "args": {"query": "langgraph supervisor article", "method": "vector", "limit": 3},
                "id": "tool_call_1",
                "type": "tool_call"
            }]
        ),
        AIMessage(
            content='I will now store the updated memory.',
            tool_calls=[{
                "name": "store_memory",
                "args": {"memory_name": "LangGraph supervisor article", "content": "Source: https://example.com/langgraph. The article explains the supervisor plus sub-agent pattern, emphasizes isolated agent contexts, and recommends explicit supervisor-authored instructions."},
                "id": "tool_call_2",
                "type": "tool_call"
            }]
        ),
        AIMessage(content='''{"status": "success", "summary": "Updated stored memory for LangGraph supervisor article with a richer normalized summary."}'''),
    ])

    with patch("kazusa_ai_chatbot.agents.memory_agent._get_llm_with_tools", return_value=mock_llm), \
         patch("kazusa_ai_chatbot.agents.memory_agent.search_memory", new_callable=AsyncMock, return_value=[
             (0.88, {
                 "memory_name": "LangGraph supervisor article",
                 "content": "Older shorter summary.",
             })
         ]) as mock_search, \
         patch("kazusa_ai_chatbot.agents.memory_agent.save_memory", new_callable=AsyncMock) as mock_save:
        result = await agent.run(
            {"user_id": "user_123", "channel_id": "chan_456", "message_text": "Please remember this LangGraph article for later."},
            "Please remember this LangGraph article for later.",
            "Check whether knowledge about the shared LangGraph supervisor article already exists, and if the new information is better then overwrite it.",
        )

    assert result["status"] == "success"
    assert "updated stored memory" in result["summary"].lower()
    assert [call["tool"] for call in result["tool_history"]] == [
        "recall_memory",
        "store_memory",
    ]
    mock_search.assert_awaited_once()
    mock_save.assert_awaited_once_with(
        memory_name="LangGraph supervisor article",
        content="Source: https://example.com/langgraph. The article explains the supervisor plus sub-agent pattern, emphasizes isolated agent contexts, and recommends explicit supervisor-authored instructions.",
        timestamp=ANY,
    )


@pytest.mark.asyncio
async def test_memory_agent_failure_returns_error_result():
    agent = MemoryAgent()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    with patch("kazusa_ai_chatbot.agents.memory_agent._get_llm_with_tools", return_value=mock_llm):
        result = await agent.run(
            {"user_id": "user_123", "channel_id": "chan_456"},
            "Remember this article.",
            "Store the article if it is new knowledge.",
        )

    assert result["status"] == "error"
    assert "LLM unavailable" in result["summary"]
    assert result["tool_history"] == []


@pytest.mark.asyncio
async def test_memory_agent_returns_needs_clarification_for_implicit_request_summary():
    agent = MemoryAgent()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="""{"status": "needs_clarification", "summary": "您好！我注意到您提到要让我记一个清单，但消息中似乎没有包含具体的内容呢。请问这个清单具体是什么？您可以把需要记录的内容发给我吗？"}"""))

    with patch("kazusa_ai_chatbot.agents.memory_agent._get_llm_with_tools", return_value=mock_llm):
        result = await agent.run(
            {"user_id": "user_123", "channel_id": "chan_456", "message_text": "千纱能把这个清单记一下么"},
            "千纱能把这个清单记一下么",
            "Remember the referenced checklist for later.",
        )

    assert result["status"] == "needs_clarification"
    assert "清单" in result["summary"]
    assert result["tool_history"] == []


# ── Live LLM tests ────────────────────────────────────────────────────
# Requires a running LM Studio instance with a chat model loaded.
# Run with:  pytest -m live_llm -v

live_llm = pytest.mark.live_llm


@live_llm
@pytest.mark.asyncio
async def test_live_memory_agent_recalls_stored_memory():
    """Real LLM should recall stored memory when asked about previously saved content."""
    import kazusa_ai_chatbot.agents.memory_agent as ma
    from kazusa_ai_chatbot.db import save_memory, enable_memory_vector_index

    # Reset cached LLM so a real one is created
    ma._llm = None

    # Ensure memory vector index is enabled
    await enable_memory_vector_index()

    # First save a test memory entry
    await save_memory(
        memory_name="Test programming guide",
        content="Python is a versatile programming language often used for web development, data science, and automation.",
        timestamp="2026-04-05T00:00:00+00:00",
    )

    # Now test recall
    result = await ma.MemoryAgent().run(
        {"user_id": "user_123", "channel_id": "chan_456", "message_text": "What do you remember about Python programming?"},
        "What do you remember about Python programming?",
        "Recall any stored memory about Python programming.",
    )

    assert result["agent"] == "memory_agent"
    assert result["status"] in ["success", "needs_context"]
    if result["status"] == "success":
        assert "python" in result["summary"].lower()
        assert len(result["tool_history"]) > 0
        assert result["tool_history"][0]["tool"] == "recall_memory"


@live_llm
@pytest.mark.asyncio
async def test_live_memory_agent_stores_new_memory():
    """Real LLM should store new memory when given explicit content to remember."""
    import kazusa_ai_chatbot.agents.memory_agent as ma
    from kazusa_ai_chatbot.db import search_memory, enable_memory_vector_index

    # Reset cached LLM so a real one is created
    ma._llm = None

    # Ensure memory vector index is enabled
    await enable_memory_vector_index()

    # Test storing new memory
    result = await ma.MemoryAgent().run(
        {"user_id": "user_123", "channel_id": "chan_456", "message_text": "Please remember this API endpoint: https://api.example.com/v1/users"},
        "Please remember this API endpoint: https://api.example.com/v1/users",
        "Store the API endpoint information for future reference.",
    )

    assert result["agent"] == "memory_agent"
    assert result["status"] in ["success", "needs_context"]
    if result["status"] == "success":
        assert len(result["tool_history"]) > 0
        assert any(call["tool"] == "store_memory" for call in result["tool_history"])

    # Verify the memory was stored by searching for it
    search_results = await search_memory(query="API endpoint", limit=5, method="vector")
    assert len(search_results) > 0
    memory_names = [doc.get("memory_name", "") for score, doc in search_results]
    assert any("api" in name.lower() or "endpoint" in name.lower() for name in memory_names)


@live_llm
@pytest.mark.asyncio
async def test_live_memory_agent_stores_and_recalls_matcha_chiffon_cake_recipe():
    """Real LLM should store and recall matcha chiffon cake recipe from Xiachufang."""
    import kazusa_ai_chatbot.agents.memory_agent as ma
    from kazusa_ai_chatbot.db import search_memory, enable_memory_vector_index

    # Reset cached LLM so a real one is created
    ma._llm = None

    # Ensure memory vector index is enabled
    await enable_memory_vector_index()

    # Test storing the matcha chiffon cake recipe
    result = await ma.MemoryAgent().run(
        {"user_id": "user_123", "channel_id": "chan_456", "message_text": "请记住这个抹茶戚风蛋糕的食谱：https://m.xiachufang.com/recipe/105888249/"},
        "请记住这个抹茶戚风蛋糕的食谱：https://m.xiachufang.com/recipe/105888249/",
        "Store the matcha chiffon cake recipe from Xiachufang for future reference.",
    )

    assert result["agent"] == "memory_agent"
    assert result["status"] in ["success", "needs_context"]
    if result["status"] == "success":
        assert len(result["tool_history"]) > 0
        assert any(call["tool"] == "store_memory" for call in result["tool_history"])

    # Now test recalling the stored recipe
    recall_result = await ma.MemoryAgent().run(
        {"user_id": "user_123", "channel_id": "chan_456", "message_text": "你还记得那个抹茶戚风蛋糕的食谱吗？"},
        "你还记得那个抹茶戚风蛋糕的食谱吗？",
        "Recall the matcha chiffon cake recipe that was previously stored.",
    )

    assert recall_result["agent"] == "memory_agent"
    assert recall_result["status"] in ["success", "needs_context"]
    if recall_result["status"] == "success":
        assert "抹茶" in recall_result["summary"] or "matcha" in recall_result["summary"].lower()
        assert len(recall_result["tool_history"]) > 0
        assert recall_result["tool_history"][0]["tool"] == "recall_memory"

    # Verify the memory was stored by searching for it
    search_results = await search_memory(query="抹茶戚风蛋糕", limit=5, method="vector")
    assert len(search_results) > 0
    memory_names = [doc.get("memory_name", "") for score, doc in search_results]
    assert any("抹茶" in name or "matcha" in name.lower() or "蛋糕" in name for name in memory_names)
