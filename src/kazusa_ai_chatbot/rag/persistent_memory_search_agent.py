"""RAG helper agent: semantic persistent-memory search."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.config import RAG_SUBAGENT_LLM_API_KEY, RAG_SUBAGENT_LLM_BASE_URL, RAG_SUBAGENT_LLM_MODEL
from kazusa_ai_chatbot.rag.memory_retrieval_tools import search_persistent_memory
from kazusa_ai_chatbot.rag.cache2_policy import (
    PERSISTENT_MEMORY_SEARCH_CACHE_NAME,
    build_persistent_memory_search_cache_key,
    build_persistent_memory_search_dependencies,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = """\
你是一个只负责 `search_persistent_memory` 的检索参数生成器。

# 你的唯一职责
- 只为 `search_persistent_memory` 生成参数。
- `search_query` 必须写成自然语言的记忆查询，不要退化成关键词列表。
- 这类查询要偏向"印象 / 事实 / 看法 / 关系线索"的 framing，例如"千纱对 X 的看法""关于 X 的已知承诺"。
- 如果 `context` 或 `known_facts` 明确给出 `memory_type`、`status` 或来源用户，请使用这些过滤条件。
- 如果 `feedback` 指出记忆类型不对、查询太抽象或没有相关记忆，下一轮必须改写。

# 字段约束（严格遵守）
- `source_global_user_id`：只有在 context 或 known_facts 中存在**明确的 UUID 格式**用户 ID 时才填写；平台频道 ID、用户名、昵称等均不是 UUID，一律省略。
- `memory_type`：只能取以下枚举值之一，否则省略：fact | promise | impression | narrative | defense_rule
- `status`：只能取 active | fulfilled | expired | superseded，否则省略。
- `source_kind`：只能取 conversation_extracted | relationship_inferred | reflection_inferred | seeded_manual | external_imported，否则省略。

# 输出格式
请只返回合法 JSON：
{
  "search_query": "string",
  "top_k": 5,
  "source_global_user_id": "UUID string or omitted",
  "memory_type": "fact|promise|impression|narrative|defense_rule or omitted",
  "source_kind": "string or omitted",
  "status": "string or omitted",
  "expiry_before": "ISO-8601 or omitted",
  "expiry_after": "ISO-8601 or omitted"
}
"""
_generator_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

_JUDGE_PROMPT = """\
你是 `search_persistent_memory` 的结果评估器。

# 任务
- 判断当前结果是否已经足以解决槽位。
- 如果未解决，反馈必须具体可执行。

# 常见反馈方向
- 查询太抽象，需要改成某人的看法/承诺/事实
- `memory_type` 错了，需要换 impression / fact / promise 等
- 返回记忆不相关，需要换主题角度
- 没有相关记忆，需要放宽过滤或改写查询

# 输出格式
请只返回合法 JSON：
{
  "resolved": true or false,
  "feedback": "string"
}
"""
_judge_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


def _normalize_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Keep only valid `search_persistent_memory` arguments.

    Args:
        raw_args: Raw dict from the generator LLM.

    Returns:
        Validated argument dict for ``search_persistent_memory``.
    """
    args: dict[str, Any] = {}

    search_query = str(raw_args.get("search_query", "")).strip()
    if search_query:
        args["search_query"] = search_query

    top_k = raw_args.get("top_k", 5)
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
        args["top_k"] = top_k
    else:
        args["top_k"] = 5

    for key in (
        "source_global_user_id",
        "memory_type",
        "source_kind",
        "status",
        "expiry_before",
        "expiry_after",
    ):
        raw_val = raw_args.get(key)
        if raw_val is None:
            continue
        value = str(raw_val).strip()
        if value:
            args[key] = value

    return args


def _slot_number(task: str) -> int | None:
    """Extract a referenced slot number from a task description.

    Args:
        task: Slot description that may contain "slot N".

    Returns:
        Integer slot number, or None when no reference is present.
    """
    match = re.search(r"slot\s+(\d+)", task, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _resolved_user_from_known_facts(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Find the user resolved by the slot referenced in the memory-search task.

    Args:
        task: Slot description produced by the outer-loop supervisor.
        context: Agent context containing known_facts.

    Returns:
        Dict with known resolved-user fields, or an empty dict when unavailable.
    """
    slot_number = _slot_number(task)
    known_facts = context.get("known_facts", [])
    if slot_number is None or not isinstance(known_facts, list):
        return {}
    if slot_number < 1 or slot_number > len(known_facts):
        return {}

    fact = known_facts[slot_number - 1]
    if not isinstance(fact, dict):
        return {}

    raw_result = fact.get("raw_result")
    if isinstance(raw_result, dict):
        return {
            "global_user_id": str(raw_result.get("global_user_id", "")).strip(),
            "display_name": str(raw_result.get("display_name", "")).strip(),
        }
    return {}


def _is_about_resolved_user_search(task: str) -> bool:
    """Return whether the memory task asks about a resolved user as subject.

    Args:
        task: Slot description produced by the outer-loop supervisor.

    Returns:
        True when the slot is about impressions/opinions/facts of a resolved user,
        not memories sourced by that user.
    """
    task_lower = task.lower()
    references_resolved_user = "user resolved in slot" in task_lower
    asks_about_subject = any(
        marker in task_lower
        for marker in ("about", "impression", "opinion", "看法", "印象", "评价")
    )
    return references_resolved_user and asks_about_subject


def _apply_resolved_subject_user(
    args: dict[str, Any], task: str, context: dict[str, Any]
) -> dict[str, Any]:
    """Adjust memory-search args for third-party subject-user queries.

    ``source_global_user_id`` means the user who sourced/triggered a memory, not
    necessarily the user the memory is about. For "about the user resolved in
    slot N" tasks, keep the resolved user's display name in the semantic query
    and remove the source filter so memories learned from other users remain
    searchable.

    Args:
        args: Tool arguments generated by the LLM.
        task: Slot description produced by the outer-loop supervisor.
        context: Agent context containing known_facts.

    Returns:
        Adjusted tool arguments.
    """
    if not _is_about_resolved_user_search(task):
        return args

    adjusted_args = dict(args)
    adjusted_args.pop("source_global_user_id", None)

    resolved_user = _resolved_user_from_known_facts(task, context)
    display_name = resolved_user.get("display_name", "")
    search_query = str(adjusted_args.get("search_query", "")).strip()
    if display_name and display_name not in search_query:
        adjusted_args["search_query"] = f"{search_query} {display_name}".strip()

    if "memory_type" not in adjusted_args and any(
        marker in task.lower()
        for marker in ("impression", "opinion", "看法", "印象", "评价")
    ):
        adjusted_args["memory_type"] = "impression"

    return adjusted_args


async def _generator(task: str, context: dict[str, Any], feedback: str) -> dict[str, Any]:
    """Generate one `search_persistent_memory` argument dict.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Known facts and runtime hints.
        feedback: Judge feedback from the previous attempt, or empty string.

    Returns:
        Normalized arguments for ``search_persistent_memory``.
    """
    system_prompt = SystemMessage(content=_GENERATOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(
            {"task": task, "context": context, "feedback": feedback},
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _generator_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(str(response.content))
    if not isinstance(result, dict):
        return {}
    return _normalize_args(result)


async def _tool(args: dict[str, Any]) -> object:
    """Execute `search_persistent_memory` exactly once.

    Args:
        args: Normalized arguments for the tool.

    Returns:
        Tool result or an error dict on invalid arguments.
    """
    try:
        return await search_persistent_memory.ainvoke(args)
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info("persistent_memory_search_agent invalid args: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest semantic memory-search result resolves the slot.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        result: Tool result from the current attempt.

    Returns:
        Tuple of (resolved, feedback).
    """
    system_prompt = SystemMessage(content=_JUDGE_PROMPT)
    human_message = HumanMessage(
        content=json.dumps({"task": task, "result": result}, ensure_ascii=False)
    )
    response = await _judge_llm.ainvoke([system_prompt, human_message])
    verdict = parse_llm_json_output(str(response.content))
    if not isinstance(verdict, dict):
        return False, "评估输出无效，请把查询改成更具体的记忆描述。"

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return resolved, feedback


class PersistentMemorySearchAgent(BaseRAGHelperAgent):
    """RAG helper agent that resolves a slot through iterative semantic search over persistent memories.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="persistent_memory_search_agent",
            cache_name=PERSISTENT_MEMORY_SEARCH_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Resolve one slot through iterative semantic search over persistent memories.

        Args:
            task: Slot description produced by the outer-loop supervisor.
            context: Known facts and runtime hints that may constrain the search.
            max_attempts: Maximum generator-tool-judge iterations to attempt.

        Returns:
            Dict with resolved (bool), result (last tool result), and attempts count.
        """
        cache_key = build_persistent_memory_search_cache_key(task, context)
        cached = await self.read_cache(cache_key)
        if cached is not None:
            return self.with_cache_status(
                {"resolved": True, "result": cached, "attempts": 0},
                hit=True,
                reason="hit",
                cache_key=cache_key,
            )

        feedback = ""
        result = None
        resolved = False
        attempt = 0
        args: dict[str, Any] = {}
        cache_stored = False

        for attempt in range(max_attempts):
            args = await _generator(task, context, feedback)
            args = _apply_resolved_subject_user(args, task, context)
            result = await _tool(args)
            resolved, feedback = await _judge(task, result)
            if resolved:
                break

        if resolved:
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_persistent_memory_search_dependencies(args),
                metadata={},
            )
            cache_stored = True

        if cache_stored:
            cache_reason = "miss_stored"
        elif resolved:
            cache_reason = "miss_not_cacheable"
        else:
            cache_reason = "miss_unresolved"

        return self.with_cache_status(
            {
                "resolved": resolved,
                "result": result,
                "attempts": attempt + 1,
            },
            hit=False,
            reason=cache_reason,
            cache_key=cache_key,
        )


async def _test_main() -> None:
    """Run a manual smoke check for PersistentMemorySearchAgent."""
    agent = PersistentMemorySearchAgent()
    result = await agent.run(
        task="找出和'洗车'有关的持久记忆条目",
        context={"known_facts": []},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    result = await agent.run(
        task="找出和'洗车'有关的持久记忆条目",
        context={"known_facts": []},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
