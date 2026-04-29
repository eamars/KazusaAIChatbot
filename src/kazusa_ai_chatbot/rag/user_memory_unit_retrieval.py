"""RAG-owned retrieval and projection for user memory units."""

from __future__ import annotations

from kazusa_ai_chatbot.db import query_user_memory_units, search_user_memory_units_by_vector
from kazusa_ai_chatbot.db.schemas import (
    UserMemoryContextDoc,
    UserMemoryContextEntry,
    UserMemoryUnitStatus,
    UserMemoryUnitType,
)
from kazusa_ai_chatbot.utils import text_or_empty


MEMORY_CONTEXT_CATEGORY_ORDER = (
    "stable_patterns",
    "recent_shifts",
    "objective_facts",
    "milestones",
    "active_commitments",
)

UNIT_TYPE_TO_CATEGORY = {
    UserMemoryUnitType.STABLE_PATTERN: "stable_patterns",
    UserMemoryUnitType.RECENT_SHIFT: "recent_shifts",
    UserMemoryUnitType.OBJECTIVE_FACT: "objective_facts",
    UserMemoryUnitType.MILESTONE: "milestones",
    UserMemoryUnitType.ACTIVE_COMMITMENT: "active_commitments",
}

CATEGORY_TO_UNIT_TYPE = {
    category: unit_type
    for unit_type, category in UNIT_TYPE_TO_CATEGORY.items()
}

DEFAULT_USER_MEMORY_CONTEXT_BUDGET = {
    "stable_patterns": {"max_items": 3, "max_chars": 900},
    "recent_shifts": {"max_items": 4, "max_chars": 1000},
    "objective_facts": {"max_items": 4, "max_chars": 900},
    "milestones": {"max_items": 3, "max_chars": 700},
    "active_commitments": {"max_items": 4, "max_chars": 900},
}


def _entry_char_count(entry: dict) -> int:
    char_count = sum(
        len(text_or_empty(entry.get(field)))
        for field in ("fact", "subjective_appraisal", "relationship_signal", "updated_at")
    )
    return char_count


def _project_unit(unit: dict) -> UserMemoryContextEntry:
    entry: UserMemoryContextEntry = {
        "fact": text_or_empty(unit.get("fact")),
        "subjective_appraisal": text_or_empty(unit.get("subjective_appraisal")),
        "relationship_signal": text_or_empty(unit.get("relationship_signal")),
    }
    updated_at = text_or_empty(unit.get("updated_at"))
    if updated_at:
        entry["updated_at"] = updated_at
    return entry


def _category_budget(category: str, budget: dict[str, dict[str, int]] | None) -> dict[str, int]:
    if budget and category in budget:
        return {
            "max_items": int(budget[category].get("max_items", DEFAULT_USER_MEMORY_CONTEXT_BUDGET[category]["max_items"])),
            "max_chars": int(budget[category].get("max_chars", DEFAULT_USER_MEMORY_CONTEXT_BUDGET[category]["max_chars"])),
        }
    return DEFAULT_USER_MEMORY_CONTEXT_BUDGET[category]


def empty_user_memory_context() -> UserMemoryContextDoc:
    """Build an empty prompt-facing user memory context.

    Returns:
        A context dict with all expected categories present.
    """

    return {
        "stable_patterns": [],
        "recent_shifts": [],
        "objective_facts": [],
        "milestones": [],
        "active_commitments": [],
    }


def project_user_memory_units(
    units: list[dict],
    *,
    budget: dict[str, dict[str, int]] | None = None,
) -> UserMemoryContextDoc:
    """Project stored memory units into the cognition-facing category payload.

    Args:
        units: Stored memory-unit documents.
        budget: Optional per-category item and character budget override.

    Returns:
        Category-balanced prompt-facing memory context.
    """

    context = empty_user_memory_context()
    spent_chars = {category: 0 for category in MEMORY_CONTEXT_CATEGORY_ORDER}
    spent_items = {category: 0 for category in MEMORY_CONTEXT_CATEGORY_ORDER}

    for unit in units:
        category = UNIT_TYPE_TO_CATEGORY.get(text_or_empty(unit.get("unit_type")))
        if not category:
            continue
        category_budget = _category_budget(category, budget)
        if spent_items[category] >= category_budget["max_items"]:
            continue

        entry = _project_unit(unit)
        if not entry["fact"] or not entry["subjective_appraisal"] or not entry["relationship_signal"]:
            continue

        entry_chars = _entry_char_count(entry)
        if spent_chars[category] + entry_chars > category_budget["max_chars"]:
            continue

        context[category].append(entry)
        spent_chars[category] += entry_chars
        spent_items[category] += 1

    return context


def _merge_units(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Merge two memory-unit result lists by stable unit id.

    Args:
        primary: Higher-priority units, usually semantic hits.
        secondary: Lower-priority units, usually recency hits.

    Returns:
        Deduplicated units preserving first occurrence order.
    """

    merged: list[dict] = []
    seen: set[str] = set()
    for unit in primary + secondary:
        unit_id = text_or_empty(unit.get("unit_id"))
        if not unit_id or unit_id in seen:
            continue
        seen.add(unit_id)
        merged.append(unit)
    return merged


async def retrieve_user_memory_units_for_context(
    global_user_id: str,
    *,
    query_embedding: list[float] | None,
    include_semantic: bool,
    limit: int = 100,
) -> list[dict]:
    """Retrieve RAG-owned memory units for projection and consolidation reuse.

    Args:
        global_user_id: Internal UUID for the memory owner.
        query_embedding: Current-turn embedding generated by the RAG profile
            agent. Required for semantic recall.
        include_semantic: Whether semantic recall should be attempted.
        limit: Maximum merged units to return.

    Returns:
        Deduplicated active memory units, with semantic hits first and recency
        hits filling the remaining budget.
    """

    semantic_units: list[dict] = []
    if include_semantic and query_embedding:
        semantic_units = await search_user_memory_units_by_vector(
            global_user_id,
            query_embedding,
            statuses=[UserMemoryUnitStatus.ACTIVE],
            limit=limit,
        )

    recent_units = await query_user_memory_units(
        global_user_id,
        statuses=[UserMemoryUnitStatus.ACTIVE],
        limit=limit,
    )
    merged_units = _merge_units(semantic_units, recent_units)[:limit]
    return merged_units


async def build_user_memory_context_bundle(
    global_user_id: str,
    *,
    query_embedding: list[float] | None,
    include_semantic: bool,
    budget: dict[str, dict[str, int]] | None = None,
) -> tuple[UserMemoryContextDoc, list[dict]]:
    """Build both the cognition projection and RAG-surfaced source units.

    Args:
        global_user_id: Internal UUID for the memory owner.
        query_embedding: Current-turn embedding generated by RAG.
        include_semantic: Whether semantic recall should be attempted.
        budget: Optional per-category projection budget.

    Returns:
        Tuple of prompt-facing context and the bounded source-unit list that
        consolidation can reuse for merge/evolve/create decisions.
    """

    units = await retrieve_user_memory_units_for_context(
        global_user_id,
        query_embedding=query_embedding,
        include_semantic=include_semantic,
    )
    user_memory_context = project_user_memory_units(units, budget=budget)
    return user_memory_context, units


async def build_user_memory_context(
    global_user_id: str,
    *,
    query_text: str,
    include_semantic: bool,
    budget: dict[str, dict[str, int]] | None = None,
) -> UserMemoryContextDoc:
    """Read and project user memory units for cognition.

    Args:
        global_user_id: Internal UUID for the memory owner.
        query_text: Current query text. Reserved for future semantic retrieval.
        include_semantic: Kept for parity with existing RAG profile APIs.
        budget: Optional per-category projection budget.

    Returns:
        Prompt-facing user memory context.
    """

    del query_text, include_semantic
    units = await query_user_memory_units(
        global_user_id,
        statuses=[UserMemoryUnitStatus.ACTIVE],
        limit=100,
    )
    user_memory_context = project_user_memory_units(units, budget=budget)
    return user_memory_context


async def retrieve_memory_unit_merge_candidates(
    global_user_id: str,
    *,
    candidate_unit: dict,
    surfaced_units: list[dict],
    limit: int = 6,
) -> list[dict]:
    """Retrieve candidate units for a consolidator LLM merge judgment.

    Args:
        global_user_id: Internal UUID for the memory owner.
        candidate_unit: Newly extracted candidate unit.
        surfaced_units: Units already surfaced by response-time RAG.
        limit: Maximum candidates to return.

    Returns:
        Projected existing units for the merge judge. The caller's LLM decides
        create, merge, or evolve; this function only retrieves candidates.
    """

    unit_type = text_or_empty(candidate_unit.get("unit_type"))
    surfaced_by_id = {
        text_or_empty(unit.get("unit_id")): unit
        for unit in surfaced_units
        if text_or_empty(unit.get("unit_id"))
    }
    candidates = list(surfaced_by_id.values())
    if len(candidates) < limit:
        fetched = await query_user_memory_units(
            global_user_id,
            unit_types=[unit_type] if unit_type else None,
            statuses=[UserMemoryUnitStatus.ACTIVE],
            limit=limit,
        )
        for unit in fetched:
            unit_id = text_or_empty(unit.get("unit_id"))
            if unit_id and unit_id not in surfaced_by_id:
                candidates.append(unit)
                surfaced_by_id[unit_id] = unit
            if len(candidates) >= limit:
                break

    return candidates[:limit]
