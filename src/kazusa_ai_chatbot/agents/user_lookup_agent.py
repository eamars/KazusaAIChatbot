"""Inner-loop agent for resolving a display name to global_user_id."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.db import search_conversation_history, search_users_by_display_name
from kazusa_ai_chatbot.rag.cache2_policy import (
    USER_LOOKUP_CACHE_NAME,
    build_user_lookup_cache_key,
    build_user_lookup_dependencies,
)
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)


_EXTRACTOR_PROMPT = """\
Extract the exact display name to look up from the slot description below.

# Return Format: 
Valid JSON without makrdown wrap. Only include the following keys
{
    "display_name": "the name string"
}
"""

_PICKER_PROMPT = """\
You are matching a target display name against a list of candidates.
Return the global_user_id of the best match, or null if none is close enough.

# Return Format: 
Valid JSON without makrdown wrap. Only include the following keys
{
    "global_user_id": "uuid" or null
}
"""

_extractor_llm = get_llm(temperature=0.0, top_p=1.0)
_picker_llm = get_llm(temperature=0.0, top_p=1.0)


async def _extract_display_name(task: str, context: dict) -> str:
    """Extract the literal display name from the slot description.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Runtime hints (passed for context only).

    Returns:
        The display name string to search for, or empty string on failure.
    """
    system_prompt = SystemMessage(content=_EXTRACTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps({"task": task, "context": context}, ensure_ascii=False, default=str)
    )
    response = await _extractor_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(str(response.content))
    if not isinstance(result, dict):
        return ""
    return str(result.get("display_name", "")).strip()


async def _pick_best(target_name: str, candidates: list[dict]) -> dict | None:
    """Ask the LLM to pick the closest match from a list of candidates.

    Args:
        target_name: The display name we are trying to resolve.
        candidates: List of dicts, each with global_user_id, display_name, platform.

    Returns:
        The best-matching candidate dict, or None if no close match found.
    """
    if len(candidates) == 1:
        return candidates[0]

    system_prompt = SystemMessage(content=_PICKER_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(
            {"target": target_name, "candidates": candidates},
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _picker_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(str(response.content))
    if not isinstance(result, dict):
        return None

    chosen_id = result.get("global_user_id")
    if not chosen_id:
        return None

    for candidate in candidates:
        if candidate.get("global_user_id") == chosen_id:
            return candidate
    return None


async def _vector_search_candidates(display_name: str, context: dict) -> list[dict]:
    """Fall back to vector search over conversation history to find candidate users.

    Args:
        display_name: The name to use as the semantic query.
        context: Runtime hints supplying platform and channel filters.

    Returns:
        Deduplicated list of dicts with global_user_id, display_name, platform.
    """
    channel_id = str(
        context.get("platform_channel_id")
        or context.get("target_platform_channel_id")
        or ""
    ).strip() or None
    platform = str(context.get("platform") or "").strip() or None

    try:
        results = await search_conversation_history(
            query=display_name,
            platform=platform,
            platform_channel_id=channel_id,
            limit=10,
            method="vector",
        )
    except Exception as exc:
        logger.exception("user_lookup_agent vector search failed for %r", display_name)
        logger.debug("vector search error detail: %s", exc)
        return []

    seen: set[str] = set()
    candidates: list[dict] = []
    for _, msg in results:
        uid = str(msg.get("global_user_id", "")).strip()
        if uid and uid not in seen:
            seen.add(uid)
            candidates.append({
                "global_user_id": uid,
                "display_name": str(msg.get("display_name", "")),
                "platform": str(msg.get("platform", "")),
            })
    return candidates


async def user_lookup_agent(
    task: str,
    context: dict,
    max_attempts: int = 3,
) -> dict:
    """Resolve a display name to global_user_id.

    Step 1: exact substring match in user_profiles.
    Step 2: vector search over conversation history to build candidates.
    Step 3: LLM picks the best candidate from either step.

    Args:
        task: Slot description containing the display name to resolve.
        context: Runtime hints supplying platform and channel filters.
        max_attempts: Unused; kept for interface compatibility.

    Returns:
        Dict with resolved (bool), result (JSON list with best match), and attempts (int).
    """
    del max_attempts

    display_name = await _extract_display_name(task, context)
    if not display_name:
        return {"resolved": False, "result": None, "attempts": 1}

    cache = get_rag_cache2_runtime()
    cache_key = build_user_lookup_cache_key(display_name, context)
    cached_result = await cache.get(cache_key)
    if cached_result is not None:
        return {
            "resolved": True,
            "result": cached_result,
            "attempts": 0,
            "cache_hit": True,
        }

    # Step 1: exact lookup in user_profiles
    try:
        exact_results = await search_users_by_display_name(display_name)
    except Exception as exc:
        logger.exception("user_lookup_agent exact search failed for %r", display_name)
        exact_results = []
        logger.debug("exact search error detail: %s", exc)

    if exact_results:
        best = await _pick_best(display_name, exact_results)
        if best:
            await cache.store(
                cache_key=cache_key,
                cache_name=USER_LOOKUP_CACHE_NAME,
                result=best,
                dependencies=build_user_lookup_dependencies(display_name, context),
                metadata={"lookup_mode": "profile"},
            )
            return {
                "resolved": True,
                "result": best,
                "attempts": 1,
                "cache_hit": False,
            }

    # Step 2: vector search fallback over conversation history
    candidates = await _vector_search_candidates(display_name, context)
    if candidates:
        best = await _pick_best(display_name, candidates)
        if best:
            return {
                "resolved": True,
                "result": best,
                "attempts": 2,
            }

    return {"resolved": False, "result": None, "attempts": 2}


async def test_main() -> None:
    """Run a manual smoke check for the user lookup agent."""
    result = await user_lookup_agent(
        task="蚝爹油",
        context={
            "platform": "qq",
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # result = await user_lookup_agent(
    #     task="蚝爹油",
    #     context={
    #         "platform": "qq",
    #     },
    # )
    # print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_main())
