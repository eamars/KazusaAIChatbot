"""Stage 3 — RAG Retriever (no generative LLM call).

Embeds the query via LM Studio's embedding endpoint, then runs a
MongoDB Atlas $vectorSearch to fetch the top-K relevant chunks.
"""

from __future__ import annotations

import logging

from kazusa_ai_chatbot.db import vector_search, get_text_embedding
from kazusa_ai_chatbot.state import BotState, RagResult
from kazusa_ai_chatbot.config import RAG_TOP_K

logger = logging.getLogger(__name__)




async def rag_retriever(state: BotState) -> BotState:
    """Retrieve relevant lore / knowledge chunks from MongoDB."""
    if not state.get("retrieve_rag"):
        return {"rag_results": []}

    query = state.get("rag_query", "")
    if not query:
        return {"rag_results": []}

    try:
        embedding = await get_text_embedding(query)
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
