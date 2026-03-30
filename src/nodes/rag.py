"""Stage 3 — RAG Retriever (no generative LLM call).

Embeds the query via LM Studio's embedding endpoint, then runs a
MongoDB Atlas $vectorSearch to fetch the top-K relevant chunks.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from config import EMBEDDING_BASE_URL, EMBEDDING_MODEL, LLM_API_KEY, RAG_TOP_K
from db import vector_search
from state import BotState, RagResult

logger = logging.getLogger(__name__)

# Lazily initialised embedding client
_embed_client: AsyncOpenAI | None = None


def _get_embed_client() -> AsyncOpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = AsyncOpenAI(
            base_url=EMBEDDING_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _embed_client


async def _embed(text: str) -> list[float]:
    """Get embedding vector for a single text string."""
    client = _get_embed_client()
    resp = await client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding


async def rag_retriever(state: BotState) -> BotState:
    """Retrieve relevant lore / knowledge chunks from MongoDB."""
    if not state.get("retrieve_rag"):
        return {"rag_results": []}

    query = state.get("rag_query", "")
    if not query:
        return {"rag_results": []}

    try:
        embedding = await _embed(query)
        raw_results = await vector_search(
            collection_name="lore",
            query_embedding=embedding,
            top_k=RAG_TOP_K,
        )
        results: list[RagResult] = [
            RagResult(
                text=doc.get("text", ""),
                source=doc.get("source", "unknown"),
                score=doc.get("score", 0.0),
            )
            for doc in raw_results
        ]
    except Exception:
        logger.exception("RAG retrieval failed — continuing without context")
        results = []

    return {"rag_results": results}
