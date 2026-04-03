"""Tests for Stage 3 — RAG Retriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.db import _cosine_similarity, get_text_embedding
from kazusa_ai_chatbot.nodes.rag import rag_retriever

# Mark for tests that require a running LM Studio instance.
# Run with:  pytest -m live_llm
live_llm = pytest.mark.live_llm


@pytest.mark.asyncio
async def test_rag_skipped_when_flag_false(routed_state):
    routed_state["retrieve_rag"] = False
    result = await rag_retriever(routed_state)
    assert result["rag_results"] == []


@pytest.mark.asyncio
async def test_rag_skipped_when_query_empty(routed_state):
    routed_state["rag_query"] = ""
    result = await rag_retriever(routed_state)
    assert result["rag_results"] == []


@pytest.mark.asyncio
async def test_rag_returns_results(routed_state):
    mock_embedding = [0.1] * 128
    mock_docs = [
        {"text": "The gate was breached.", "source": "lore/events", "score": 0.9},
        {"text": "Voss reported casualties.", "source": "lore/npcs", "score": 0.8},
    ]

    with (
        patch("kazusa_ai_chatbot.db.get_text_embedding", new_callable=AsyncMock, return_value=mock_embedding),
        patch("kazusa_ai_chatbot.nodes.rag.vector_search", new_callable=AsyncMock, return_value=mock_docs),
    ):
        result = await rag_retriever(routed_state)

    assert len(result["rag_results"]) == 2
    assert result["rag_results"][0]["text"] == "The gate was breached."
    assert result["rag_results"][0]["score"] == 0.9


@pytest.mark.asyncio
async def test_rag_handles_embed_failure(routed_state):
    with patch("kazusa_ai_chatbot.db.get_text_embedding", new_callable=AsyncMock, side_effect=Exception("embed error")):
        result = await rag_retriever(routed_state)

    assert result["rag_results"] == []


@pytest.mark.asyncio
async def test_rag_handles_search_failure(routed_state):
    with (
        patch("kazusa_ai_chatbot.db.get_text_embedding", new_callable=AsyncMock, return_value=[0.1] * 128),
        patch("kazusa_ai_chatbot.nodes.rag.vector_search", new_callable=AsyncMock, side_effect=Exception("db error")),
    ):
        result = await rag_retriever(routed_state)

    assert result["rag_results"] == []


@pytest.mark.asyncio
async def test_rag_returns_only_owned_fields(routed_state):
    """Parallel fan-out safety: RAG should NOT return the full state."""
    routed_state["retrieve_rag"] = False
    result = await rag_retriever(routed_state)
    assert "user_id" not in result
    assert "rag_results" in result


# ── Live LM Studio embedding tests ──────────────────────────────────
# Require a running LM Studio instance with an embedding model loaded.
# Run with:  pytest -m live_llm -v


@live_llm
@pytest.mark.asyncio
async def test_live_embed_returns_valid_vector():
    """Call the real LM Studio embedding endpoint and validate the vector shape."""
    embedding = await get_text_embedding("The northern gate was attacked by shadow wolves.")

    # Should be a non-empty list of floats
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)

    # Typical embedding dimensions: 384, 768, 1024, 1536, 4096, etc.
    assert len(embedding) >= 128, f"Unexpectedly small embedding dimension: {len(embedding)}"

    # Vector should not be all zeros
    assert any(x != 0.0 for x in embedding)


@live_llm
@pytest.mark.asyncio
async def test_live_embed_semantic_similarity():
    """Verify that semantically similar texts are closer than dissimilar ones."""
    emb_gate_attack = await get_text_embedding("The northern gate was attacked by shadow wolves last night.")
    emb_gate_breach = await get_text_embedding("Shadow wolves breached the north gate during the night.")
    emb_dessert = await get_text_embedding("I would like a chocolate cake with strawberries please.")

    sim_related = _cosine_similarity(emb_gate_attack, emb_gate_breach)
    sim_unrelated = _cosine_similarity(emb_gate_attack, emb_dessert)

    # Two sentences about the same event should be more similar than
    # a sentence about an unrelated topic
    assert sim_related > sim_unrelated, (
        f"Expected related similarity ({sim_related:.4f}) > "
        f"unrelated similarity ({sim_unrelated:.4f})"
    )


@live_llm
@pytest.mark.asyncio
async def test_live_embed_sematic_similarity_chinese():
    """Verify that semantically similar texts in Chinese are closer than dissimilar ones."""
    emb_question = await get_text_embedding("今天天气好么？")
    emb_warnings = await get_text_embedding("Kazusa要一起去甜点店么？")
    emb_command = await get_text_embedding("北门受到了攻击！")

    sim_related = _cosine_similarity(emb_question, emb_warnings)
    sim_unrelated = _cosine_similarity(emb_question, emb_command)

    # Two sentences about the same event should be more similar than
    # a sentence about an unrelated topic
    assert sim_related > sim_unrelated, (
        f"Expected related similarity ({sim_related:.4f}) > "
        f"unrelated similarity ({sim_unrelated:.4f})"
    )