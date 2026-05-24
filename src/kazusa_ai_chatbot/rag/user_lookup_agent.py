"""RAG helper agent: resolve a display name to global_user_id."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import RAG_SUBAGENT_LLM_API_KEY, RAG_SUBAGENT_LLM_BASE_URL, RAG_SUBAGENT_LLM_MODEL
from kazusa_ai_chatbot.db import search_conversation_history, search_users_by_display_name
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
from kazusa_ai_chatbot.rag.cache2_policy import (
    USER_LOOKUP_CACHE_NAME,
    build_user_lookup_cache_key,
    build_user_lookup_dependencies,
    normalize_cache_text,
)
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.prompt_projection import project_runtime_context_for_llm
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)

_USER_PROFILE_CACHE_SOURCE = "user_profile"

_EXTRACTOR_PROMPT = '''\
从下面的槽位描述中抽取要查询的精确 display name。

# 生成步骤
1. 读取 `task`，找出正在解析的字面显示名。
2. `context` 只能用于消歧，不能用来编造名字。
3. 只有确实没有显示名时，才输出空字符串。

# 输入格式
{
    "task": "外层 RAG supervisor 给出的槽位描述",
    "context": "已知事实和运行时提示"
}

# 输出格式
只返回有效 JSON，不要 markdown 包裹。只包含以下键：
{
    "display_name": "名字字符串"
}
'''

_extractor_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def _extract_display_name_with_llm(
    task: str, context: dict[str, Any]
) -> str:
    """Extract the literal display name from the slot description.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Runtime hints (passed for context only).

    Returns:
        The display name string to search for, or empty string on failure.
    """
    system_prompt = SystemMessage(content=_EXTRACTOR_PROMPT)
    llm_context = project_runtime_context_for_llm(context)
    human_message = HumanMessage(
        content=json.dumps(
            {"task": task, "context": llm_context},
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _extractor_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)
    if not isinstance(result, dict):
        return ""
    return_value = str(result.get("display_name", "")).strip()
    return return_value


_PICKER_PROMPT = '''\
你要把目标 display name 与候选列表进行匹配。
返回最佳匹配的 global_user_id；如果没有足够接近的候选，返回 null。

# 生成步骤
1. 将 `target` 视为正在解析的名字。
2. 只在给定的 `candidates` 内比较。
3. 根据 display name 和 platform 上下文选择最接近的候选。
4. 如果没有足够接近的候选，返回 null。

# 输入格式
{
    "target": "需要解析的 display name",
    "candidates": [{"global_user_id": "uuid", "display_name": "候选名字", "platform": "可选平台"}]
}

# 输出格式
只返回有效 JSON，不要 markdown 包裹。只包含以下键：
{
    "global_user_id": "uuid 或 null"
}
'''

_picker_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def _pick_best_candidate_with_llm(
    target_name: str, candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
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
    result = parse_llm_json_output(response.content)
    if not isinstance(result, dict):
        return None

    chosen_id = result.get("global_user_id")
    if not chosen_id:
        return None

    for candidate in candidates:
        if candidate.get("global_user_id") == chosen_id:
            return candidate
    return None


class UserLookupAgent(BaseRAGHelperAgent):
    """RAG helper agent that resolves a display name to global_user_id.

    Uses a two-step strategy: exact user-profile lookup first, then a vector
    search over conversation history as a fallback. Results from the profile
    step are cached via the inherited Cache 2 interface.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime: RAGCache2Runtime | None = None) -> None:
        super().__init__(
            name="user_lookup_agent",
            cache_name=USER_LOOKUP_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Resolve a display name to global_user_id.

        Step 1: exact substring match in user_profiles (result is cached).
        Step 2: vector search over conversation history as a fallback.
        Step 3: LLM picks the best candidate from either step.

        Args:
            task: Slot description containing the display name to resolve.
            context: Runtime hints supplying platform and channel filters.
            max_attempts: Unused; kept for interface compatibility.

        Returns:
            Dict with resolved (bool), result (best-match dict or None),
            attempts (int), and standardized cache metadata.
        """
        del max_attempts

        display_name = await _extract_display_name_with_llm(task, context)
        if not display_name:
            return_value = self.with_cache_status(
                {"resolved": False, "result": None, "attempts": 1},
                hit=False,
                reason="skipped_missing_display_name",
            )
            return return_value

        cache_key = build_user_lookup_cache_key(display_name, context)
        cached = await self.read_cache(cache_key)
        if cached is not None:
            return_value = self.with_cache_status(
                {"resolved": True, "result": cached, "attempts": 0},
                hit=True,
                reason="hit",
                cache_key=cache_key,
            )
            return return_value

        # Step 1: exact lookup in user_profiles
        best: dict[str, Any] | None = None
        attempts = 1
        try:
            exact_results = await search_users_by_display_name(display_name)
        except Exception as exc:
            logger.exception(
                f"user_lookup_agent exact search failed for {display_name!r}: {exc}"
            )
            exact_results = []

        if exact_results:
            best = await _pick_best_candidate_with_llm(display_name, exact_results)

        # Step 2: vector search fallback over conversation history
        if best is None:
            attempts = 2
            candidates = await self._vector_search_candidates(display_name, context)
            if candidates:
                best = await _pick_best_candidate_with_llm(display_name, candidates)

        if best is None:
            return_value = self.with_cache_status(
                {"resolved": False, "result": None, "attempts": attempts},
                hit=False,
                reason="miss_unresolved",
                cache_key=cache_key,
            )
            return return_value

        global_user_id = str(best.get("global_user_id", "")).strip()
        await self.write_cache(
            cache_key=cache_key,
            result=best,
            dependencies=build_user_lookup_dependencies(global_user_id, context),
            metadata={"lookup_mode": "profile" if attempts == 1 else "vector"},
        )
        return_value = self.with_cache_status(
            {"resolved": True, "result": best, "attempts": attempts},
            hit=False,
            reason="miss_stored",
            cache_key=cache_key,
        )
        return return_value

    async def invalidate_for_profile(
        self,
        *,
        platform: str = "",
        platform_channel_id: str = "",
        display_name: str = "",
        reason: str = "user lookup cache invalidation",
    ) -> int:
        """Invalidate cached entries for a profile/display-name scope.

        Args:
            platform: Optional platform scope. Empty matches any platform.
            platform_channel_id: Optional channel scope. Empty matches any channel.
            display_name: Optional display-name scope. Empty invalidates all
                user-lookup entries that depend on user profiles.
            reason: Human-readable reason for logs and cache metrics.

        Returns:
            Number of Cache 2 entries invalidated.
        """
        return_value = await self.invalidate_cache(
            CacheInvalidationEvent(
                source=_USER_PROFILE_CACHE_SOURCE,
                platform=normalize_cache_text(platform),
                platform_channel_id=normalize_cache_text(platform_channel_id),
                display_name=normalize_cache_text(display_name),
                reason=reason,
            )
        )
        return return_value

    async def _vector_search_candidates(
        self, display_name: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Fall back to vector search over conversation history for candidate users.

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
            logger.exception(
                f"user_lookup_agent vector search failed for {display_name!r}: {exc}"
            )
            return_value = []
            return return_value

        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
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


async def _test_main() -> None:
    """Run a manual smoke check for UserLookupAgent."""
    agent = UserLookupAgent()
    result = await agent.run(
        task='<named user>',
        context={"platform": "qq"},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    result = await agent.run(
        task='<named user>',
        context={"platform": "qq"},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    result = await agent.run(
        task='<partial user alias>',
        context={"platform": "qq"},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Test performance of caching
    result = await agent.run(
        task='<partial user alias>',
        context={"platform": "qq"},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
