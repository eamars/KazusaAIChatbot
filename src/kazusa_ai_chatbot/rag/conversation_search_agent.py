"""RAG helper agent: semantic conversation search."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.config import RAG_SUBAGENT_LLM_API_KEY, RAG_SUBAGENT_LLM_BASE_URL, RAG_SUBAGENT_LLM_MODEL
from kazusa_ai_chatbot.rag.memory_retrieval_tools import search_conversation
from kazusa_ai_chatbot.rag.cache2_policy import (
    CONVERSATION_SEARCH_CACHE_NAME,
    build_conversation_search_cache_key,
    build_conversation_search_dependencies,
    is_closed_historical_range,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = """\
你是一个只负责 `search_conversation` 的检索参数生成器。

# 你的唯一职责
- 目标是为当前槽位生成一次高质量的语义检索参数。
- 你只能为 `search_conversation` 生成参数，禁止改调其它工具。
- `search_query` 必须是自然语言语义查询，不要退化成关键词列表。
- 如果 `context` 里已经给出 platform / platform_channel_id / global_user_id / 时间边界，就优先利用这些明确线索。
- `feedback` 来自上一轮评估，必须优先吸收；如果反馈说"太泛""角度不对""结果不相关"，就重写查询意图而不是重复上一轮。

# 输出格式
请只返回合法 JSON：
{
  "search_query": "string",
  "global_user_id": "string or omitted",
  "top_k": 5,
  "platform": "string or omitted",
  "platform_channel_id": "string or omitted",
  "from_timestamp": "ISO-8601 or omitted",
  "to_timestamp": "ISO-8601 or omitted"
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
你是 `search_conversation` 的结果评估器。

# 任务
- 判断当前结果是否已经足以解决槽位。
- 如果未解决，`feedback` 必须给出下一轮可执行的修正建议。

# 常见反馈方向
- 查询太泛，需要换成更具体的语义表述
- 查询角度错了，需要聚焦人物/链接/事件
- 缺少过滤条件，需要利用已知用户或时间范围
- 返回消息不相关，需要改写成"对什么的看法/提到什么内容"

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
    """Keep only valid `search_conversation` arguments with safe scalar coercion.

    Args:
        raw_args: Raw dict from the generator LLM.

    Returns:
        Validated argument dict for ``search_conversation``.
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
        "global_user_id",
        "platform",
        "platform_channel_id",
        "from_timestamp",
        "to_timestamp",
    ):
        raw_val = raw_args.get(key)
        if raw_val is None:
            continue
        value = str(raw_val).strip()
        if value:
            args[key] = value

    return args


async def _generator(task: str, context: dict[str, Any], feedback: str) -> dict[str, Any]:
    """Generate one `search_conversation` argument dict for the current attempt.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Known facts and runtime hints.
        feedback: Judge feedback from the previous attempt, or empty string.

    Returns:
        Normalized arguments for ``search_conversation``.
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
    """Execute `search_conversation` exactly once and return the result.

    Args:
        args: Normalized arguments for the tool.

    Returns:
        Tool result or an error dict on invalid arguments.
    """
    try:
        return await search_conversation.ainvoke(args)
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info("conversation_search_agent invalid args: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest semantic search result resolves the slot.

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
        return False, "评估输出无效，请把语义查询改得更具体。"

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return resolved, feedback


class ConversationSearchAgent(BaseRAGHelperAgent):
    """RAG helper agent that resolves a slot through iterative semantic conversation search.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="conversation_search_agent",
            cache_name=CONVERSATION_SEARCH_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Resolve one slot through iterative semantic search over conversation history.

        Args:
            task: Slot description produced by the outer-loop supervisor.
            context: Known facts and runtime hints that may constrain the search.
            max_attempts: Maximum generator-tool-judge iterations to attempt.

        Returns:
            Dict with resolved (bool), result (last tool result), and attempts count.
        """
        cache_key = build_conversation_search_cache_key(task, context)
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
            result = await _tool(args)
            resolved, feedback = await _judge(task, result)
            if resolved:
                break

        if resolved and is_closed_historical_range(args):
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_conversation_search_dependencies(args, context),
                metadata={},
            )
            cache_stored = True

        if cache_stored:
            cache_reason = "miss_stored"
        elif resolved:
            cache_reason = "miss_open_conversation_range"
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
    """Run a manual smoke check for ConversationSearchAgent."""
    agent = ConversationSearchAgent()
    result = await agent.run(
        task="最近提到的小红书链接",
        context={
            "platform": "qq",
            "known_facts": [],
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    result = await agent.run(
        task="关于vibe coding的话题",
        context={
            "platform": "qq",
            "known_facts": [],
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
