"""Prompt-facing promoted reflection context."""

from __future__ import annotations

from typing import TypedDict

from kazusa_ai_chatbot.config import REFLECTION_CONTEXT_ENABLED
from kazusa_ai_chatbot.memory_evolution import (
    MemorySourceKind,
    find_active_memory_units,
)
from kazusa_ai_chatbot.reflection_cycle.promotion import PROMOTION_LANE_MEMORY_TYPE


class PromotedReflectionContext(TypedDict, total=False):
    """Bounded reflection-derived context allowed into normal chat prompts."""

    promoted_lore: list[dict]
    promoted_self_guidance: list[dict]
    source_dates: list[str]
    retrieval_notes: list[str]


async def build_promoted_reflection_context(
    *,
    enabled: bool = REFLECTION_CONTEXT_ENABLED,
    limit_per_lane: int = 3,
) -> PromotedReflectionContext:
    """Return compact promoted lore and self-guidance context when enabled."""

    if not enabled:
        return_value: PromotedReflectionContext = {}
        return return_value

    lore = await _project_lane(
        memory_type=PROMOTION_LANE_MEMORY_TYPE["lore"],
        limit=limit_per_lane,
    )
    self_guidance = await _project_lane(
        memory_type=PROMOTION_LANE_MEMORY_TYPE["self_guidance"],
        limit=limit_per_lane,
    )
    if not lore and not self_guidance:
        return_value = {}
        return return_value

    source_dates = _source_dates(lore + self_guidance)
    context: PromotedReflectionContext = {
        "promoted_lore": lore,
        "promoted_self_guidance": self_guidance,
        "source_dates": source_dates,
        "retrieval_notes": [
            "Only active reflection-promoted memory rows are included.",
        ],
    }
    return context


async def _project_lane(*, memory_type: str, limit: int) -> list[dict]:
    """Project one reflection-promoted memory lane for prompt use."""

    rows = await find_active_memory_units(
        query={
            "source_kind": MemorySourceKind.REFLECTION_INFERRED,
            "source_global_user_id": "",
            "memory_type": memory_type,
        },
        limit=limit,
    )
    projected: list[dict] = []
    for _, document in rows:
        projected.append({
            "memory_name": str(document["memory_name"]),
            "content": str(document["content"]),
            "memory_type": str(document["memory_type"]),
            "updated_at": str(document.get("updated_at", "")),
            "confidence_note": str(document.get("confidence_note", "")),
        })
    return projected


def _source_dates(rows: list[dict]) -> list[str]:
    """Return unique source dates derived from projected updated timestamps."""

    dates: list[str] = []
    for row in rows:
        updated_at = str(row.get("updated_at", ""))
        if len(updated_at) < 10:
            continue
        source_date = updated_at[:10]
        if source_date not in dates:
            dates.append(source_date)
    return dates
