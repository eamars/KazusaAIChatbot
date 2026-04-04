"""Tests for embedding utilities used by semantic search helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


import pytest

from kazusa_ai_chatbot.db import _cosine_similarity, get_text_embedding

# Mark for tests that require a running LM Studio instance.
# Run with:  pytest -m live_llm
live_llm = pytest.mark.live_llm


@pytest.mark.asyncio
async def test_get_text_embedding_calls_embedding_client():
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("kazusa_ai_chatbot.db._get_embed_client", return_value=mock_client):
        embedding = await get_text_embedding("hello")

    assert embedding == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_awaited_once_with(input=["hello"], model="text-embedding-model")


def test_cosine_similarity_identical_vectors():
    assert _cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


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