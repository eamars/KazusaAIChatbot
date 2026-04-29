"""RAG helper agent: keyword-based persistent-memory search."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.memory_retrieval_tools import search_persistent_memory_keyword
from kazusa_ai_chatbot.rag.cache2_policy import (
    PERSISTENT_MEMORY_KEYWORD_CACHE_NAME,
    build_persistent_memory_keyword_cache_key,
    build_persistent_memory_keyword_dependencies,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = """\
你是一个只负责 `search_persistent_memory_keyword` 的检索参数生成器。

# 你的唯一职责
- 只为 `search_persistent_memory_keyword` 生成参数。
- `keyword` 必须是最短且不歧义的核心词或短语，不能是完整句子。
- 如果目标是专有名词、昵称、事件名、文件名或短标签，就优先保留字面锚点。
- 如果 `feedback` 指出词太具体、无结果或需要更泛化的表达，下一轮必须显著调整。
- `source_global_user_id` 是隐私边界，不是相关性提示。默认省略；只有任务明确要求查找"由某个用户触发/提供/承诺"的记忆时才填写。

# 输出格式
请只返回合法 JSON：
{
  "keyword": "string",
  "top_k": 5,
  "source_global_user_id": "string or omitted"
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
你是 `search_persistent_memory_keyword` 的结果评估器。

# 任务
- 判断当前结果是否已经足以解决槽位。
- 如果未解决，反馈必须明确告诉下一轮怎么改关键词。

# 常见反馈方向
- 关键词太长，请收缩到核心词
- 关键词太细，请换成更常见或更泛一点的叫法
- 没有匹配，请换同义词或去掉多余修饰
- 需要补充/移除来源用户过滤

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
    """Keep only valid `search_persistent_memory_keyword` arguments.

    Args:
        raw_args: Raw dict from the generator LLM.

    Returns:
        Validated argument dict for ``search_persistent_memory_keyword``.
    """
    args: dict[str, Any] = {}

    keyword = text_or_empty(raw_args.get("keyword"))
    if keyword:
        args["keyword"] = keyword

    top_k = raw_args.get("top_k", 5)
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
        args["top_k"] = top_k
    else:
        args["top_k"] = 5

    for key in ("source_global_user_id",):
        raw_val = raw_args.get(key)
        if raw_val is None:
            continue
        value = text_or_empty(raw_val)
        if value:
            args[key] = value

    _erase_character_source_global_user_id(args)
    return args


def _erase_character_source_global_user_id(args: dict[str, Any]) -> None:
    """Erase the character ID when it is used as a memory source filter.

    Args:
        args: Mutable normalized subagent arguments.
    """
    source_global_user_id = args.get("source_global_user_id")
    if source_global_user_id == CHARACTER_GLOBAL_USER_ID:
        args.pop("source_global_user_id")


async def _generator(task: str, context: dict[str, Any], feedback: str) -> dict[str, Any]:
    """Generate one `search_persistent_memory_keyword` argument dict.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Known facts and runtime hints.
        feedback: Judge feedback from the previous attempt, or empty string.

    Returns:
        Normalized arguments for ``search_persistent_memory_keyword``.
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
    result = parse_llm_json_output(response.content)
    if not isinstance(result, dict):
        return_value = {}
        return return_value
    return_value = _normalize_args(result)
    return return_value


async def _tool(args: dict[str, Any]) -> object:
    """Execute `search_persistent_memory_keyword` exactly once.

    Args:
        args: Normalized arguments for the tool.

    Returns:
        Tool result or an error dict on invalid arguments.
    """
    try:
        return_value = await search_persistent_memory_keyword.ainvoke(args)
        return return_value
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info(f'persistent_memory_keyword_agent invalid args: {exc}')
        return_value = {"error": f"{type(exc).__name__}: {exc}"}
        return return_value


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest keyword memory-search result resolves the slot.

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
    verdict = parse_llm_json_output(response.content)
    if not isinstance(verdict, dict):
        return_value = False, "评估输出无效，请把关键词改成更短或更常见的叫法。"
        return return_value

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return_value = resolved, feedback
    return return_value


class PersistentMemoryKeywordAgent(BaseRAGHelperAgent):
    """RAG helper agent that resolves a slot through iterative keyword search over persistent memories.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="persistent_memory_keyword_agent",
            cache_name=PERSISTENT_MEMORY_KEYWORD_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Resolve one slot through iterative keyword search over persistent memories.

        Args:
            task: Slot description produced by the outer-loop supervisor.
            context: Known facts and runtime hints that may constrain the search.
            max_attempts: Maximum generator-tool-judge iterations to attempt.

        Returns:
            Dict with resolved (bool), result (last tool result), and attempts count.
        """
        cache_key = build_persistent_memory_keyword_cache_key(task, context)
        cached = await self.read_cache(cache_key)
        if cached is not None:
            return_value = self.with_cache_status(
                {"resolved": True, "result": cached, "attempts": 0},
                hit=True,
                reason="hit",
                cache_key=cache_key,
            )
            return return_value

        feedback = ""
        result = None
        resolved = False
        attempt = 0
        args: dict[str, Any] = {}
        cache_stored = False

        for attempt in range(max_attempts):
            args = await _generator(task, context, feedback)
            result = await _tool(args)
            resolved, feedback = await _judge(task, result)
            if resolved:
                break

        if resolved:
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_persistent_memory_keyword_dependencies(args),
                metadata={},
            )
            cache_stored = True

        if cache_stored:
            cache_reason = "miss_stored"
        elif resolved:
            cache_reason = "miss_not_cacheable"
        else:
            cache_reason = "miss_unresolved"

        return_value = self.with_cache_status(
            {
                "resolved": resolved,
                "result": result,
                "attempts": attempt + 1,
            },
            hit=False,
            reason=cache_reason,
            cache_key=cache_key,
        )
        return return_value


async def _test_main() -> None:
    """Run a manual smoke check for PersistentMemoryKeywordAgent."""
    agent = PersistentMemoryKeywordAgent()
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
