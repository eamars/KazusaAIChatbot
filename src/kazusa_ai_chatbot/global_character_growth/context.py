"""Runtime context facade for global character growth."""

from __future__ import annotations

from kazusa_ai_chatbot.db import global_character_growth as growth_store
from kazusa_ai_chatbot.global_character_growth.models import (
    RUNTIME_CONTEXT_LIMIT,
    GlobalCharacterGrowthContext,
)
from kazusa_ai_chatbot.global_character_growth.projection import project_runtime_context


async def build_global_character_growth_context(
    *,
    limit: int = RUNTIME_CONTEXT_LIMIT,
) -> GlobalCharacterGrowthContext:
    """Return prompt-visible global growth context when available."""

    trait_rows = await growth_store.list_prompt_visible_growth_traits(limit=limit)
    return_value = project_runtime_context(trait_rows, limit=limit)
    return return_value
