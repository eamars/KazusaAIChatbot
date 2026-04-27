from __future__ import annotations

from kazusa_ai_chatbot.config import PROFILE_MEMORY_BUDGET
from kazusa_ai_chatbot.db import build_user_profile_recall_bundle, hydrate_user_profile_with_memory_blocks


async def user_image_retriever_agent(
    global_user_id: str,
    *,
    user_profile: dict | None = None,
    input_embedding: list[float],
    include_semantic: bool,
    budget: int = PROFILE_MEMORY_BUDGET,
) -> tuple[dict, dict]:
    """Hydrate one user profile with authoritative memory blocks.

    Args:
        global_user_id: Internal UUID for the user being read.
        user_profile: Optional base profile document already loaded by caller.
        input_embedding: Current-topic embedding for semantic memory recall.
        include_semantic: Whether to include vector memory recall.
        budget: Per-type memory recall budget.

    Returns:
        Tuple of hydrated profile and raw memory blocks used for hydration.
    """
    _, memory_blocks = await build_user_profile_recall_bundle(
        global_user_id,
        user_profile=user_profile,
        topic_embedding=input_embedding,
        include_semantic=include_semantic,
        budget=budget,
    )

    hydrated = hydrate_user_profile_with_memory_blocks(user_profile or {}, memory_blocks)
    return hydrated, memory_blocks
