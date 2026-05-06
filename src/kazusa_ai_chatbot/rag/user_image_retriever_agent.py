from __future__ import annotations

from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import build_user_memory_context_bundle


async def user_image_retriever_agent(
    global_user_id: str,
    *,
    user_profile: dict | None = None,
    input_embedding: list[float],
    include_semantic: bool,
    time_context: dict | None = None,
) -> tuple[dict, dict]:
    """Hydrate one user profile with the unified memory-unit context.

    Args:
        global_user_id: Internal UUID for the user being read.
        user_profile: Optional base profile document already loaded by caller.
        input_embedding: Current-topic embedding for semantic memory recall.
        include_semantic: Whether to include vector memory recall.
        time_context: Current runtime time context for due-date status labels.

    Returns:
        Tuple of hydrated profile and raw RAG memory context.
    """
    user_memory_context, source_units = await build_user_memory_context_bundle(
        global_user_id,
        query_embedding=input_embedding,
        include_semantic=include_semantic,
        time_context=time_context,
    )
    hydrated = dict(user_profile or {})
    hydrated["user_memory_context"] = user_memory_context
    hydrated["_user_memory_units"] = source_units
    return_value = hydrated, {
        "user_memory_context": user_memory_context,
        "user_memory_units": source_units,
    }
    return return_value
