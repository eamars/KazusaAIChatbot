"""RAG helper agent: structured conversation filtering."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.config import RAG_SUBAGENT_LLM_API_KEY, RAG_SUBAGENT_LLM_BASE_URL, RAG_SUBAGENT_LLM_MODEL
from kazusa_ai_chatbot.rag.memory_retrieval_tools import get_conversation
from kazusa_ai_chatbot.rag.cache2_policy import (
    CONVERSATION_FILTER_CACHE_NAME,
    build_conversation_filter_cache_key,
    build_conversation_filter_dependencies,
    is_closed_historical_range,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.prompt_projection import (
    project_runtime_context_for_llm,
    project_tool_result_for_llm,
)
from kazusa_ai_chatbot.time_context import (
    structured_llm_time_to_utc_iso,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = """\
你是一个只负责 `get_conversation` 的结构化筛选参数生成器。

# 你的唯一职责
- 只为 `get_conversation` 生成参数。
- 优先从 `context` 与 `known_facts` 里提取明确的 platform / channel / user / display_name / time range。
- 如果 task 或 context.original_query 明确要求数量（例如“最近3条 / last 3 messages”），必须把这个数量写入 `limit`。
- 如果上一轮 `feedback` 说结果太少，就优先扩大时间范围或提高 `limit`。
- 如果上一轮 `feedback` 说用户错了，就改 user filter；如果说时间错了，就改时间范围。
- 不要凭空猜不存在的 UUID；没有就留空。

# 生成步骤
1. 先读取 `task`，确认需要拉取哪段结构化对话记录。
2. 从 `context` 和 `known_facts` 提取明确的 platform、channel、user、display_name、time range 和 limit。
3. 若 `feedback` 指出结果太少、用户错或时间错，调整对应过滤条件。
4. 不要猜 UUID；没有明确 UUID 时使用 display_name 或省略用户过滤。

# 输入格式
{
  "task": "外层 RAG supervisor 生成的槽位描述",
  "context": "已知事实与运行时提示",
  "feedback": "上一轮评估反馈，或空字符串"
}

# 输出格式
请只返回合法 JSON：
{
  "platform": "string or omitted",
  "platform_channel_id": "string or omitted",
  "limit": 5,
  "global_user_id": "string or omitted",
  "display_name": "string or omitted",
  "from_timestamp": "local YYYY-MM-DD HH:MM or omitted",
  "to_timestamp": "local YYYY-MM-DD HH:MM or omitted"
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
你是 `get_conversation` 的结果评估器。

# 任务
- 判断当前结果是否已经足以解决槽位。
- 如果未解决，反馈必须具体到下一轮该怎么调 filter。

# 审计步骤
1. 先读取 `task`，确认需要的对话记录范围。
2. 检查 `result` 是否包含足够数量、正确用户和正确时间范围的记录。
3. 结果足以解决槽位时返回 `resolved: true`。
4. 未解决时，指出下一轮应放宽时间、提高 limit、换用户过滤或调整方向。

# 输入格式
{
  "task": "外层 RAG supervisor 生成的槽位描述",
  "result": "get_conversation 的工具结果"
}

# 常见反馈方向
- 结果太少，请放宽时间范围或提高 limit
- 过滤错了，请更换 global_user_id / display_name
- 时间范围太窄或方向反了
- 已经拿到相关记录，可以停止

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
    """Keep only valid `get_conversation` arguments.

    Args:
        raw_args: Raw dict from the generator LLM.

    Returns:
        Validated argument dict for ``get_conversation``.
    """
    args: dict[str, Any] = {}

    limit = raw_args.get("limit", 5)
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        args["limit"] = limit
    else:
        args["limit"] = 5

    for key in (
        "platform",
        "platform_channel_id",
        "global_user_id",
        "display_name",
    ):
        raw_val = raw_args.get(key)
        if raw_val is None:
            continue
        value = text_or_empty(raw_val)
        if value:
            args[key] = value

    for key in ("from_timestamp", "to_timestamp"):
        value = text_or_empty(raw_args.get(key))
        if not value:
            continue
        try:
            args[key] = structured_llm_time_to_utc_iso(value)
        except ValueError as exc:
            logger.debug(f"Dropping invalid {key} from LLM output: {exc}")

    return args


async def _generator(task: str, context: dict[str, Any], feedback: str) -> dict[str, Any]:
    """Generate one `get_conversation` argument dict.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Known facts and runtime hints.
        feedback: Judge feedback from the previous attempt, or empty string.

    Returns:
        Normalized arguments for ``get_conversation``.
    """
    system_prompt = SystemMessage(content=_GENERATOR_PROMPT)
    llm_context = project_runtime_context_for_llm(context)
    human_message = HumanMessage(
        content=json.dumps(
            {"task": task, "context": llm_context, "feedback": feedback},
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
    """Execute `get_conversation` exactly once.

    Args:
        args: Normalized arguments for the tool.

    Returns:
        Tool result or an error dict on invalid arguments.
    """
    try:
        return_value = await get_conversation.ainvoke(args)
        return return_value
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info(f'conversation_filter_agent invalid args: {exc}')
        return_value = {"error": f"{type(exc).__name__}: {exc}"}
        return return_value


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest structured conversation result resolves the slot.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        result: Tool result from the current attempt.

    Returns:
        Tuple of (resolved, feedback).
    """
    system_prompt = SystemMessage(content=_JUDGE_PROMPT)
    llm_result = project_tool_result_for_llm(result)
    human_message = HumanMessage(
        content=json.dumps({"task": task, "result": llm_result}, ensure_ascii=False)
    )
    response = await _judge_llm.ainvoke([system_prompt, human_message])
    verdict = parse_llm_json_output(response.content)
    if not isinstance(verdict, dict):
        return_value = False, "评估输出无效，请调整时间范围或提高 limit。"
        return return_value

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return_value = resolved, feedback
    return return_value


class ConversationFilterAgent(BaseRAGHelperAgent):
    """RAG helper agent that resolves a slot through iterative structured conversation filtering.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="conversation_filter_agent",
            cache_name=CONVERSATION_FILTER_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Resolve one slot through iterative structured conversation filtering.

        Args:
            task: Slot description produced by the outer-loop supervisor.
            context: Known facts and runtime hints that may constrain the filters.
            max_attempts: Maximum generator-tool-judge iterations to attempt.

        Returns:
            Dict with resolved (bool), result (last tool result), and attempts count.
        """
        cache_key = build_conversation_filter_cache_key(task, context)
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

        if resolved and is_closed_historical_range(args):
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_conversation_filter_dependencies(args, context),
                metadata={},
            )
            cache_stored = True

        if cache_stored:
            cache_reason = "miss_stored"
        elif resolved:
            cache_reason = "miss_open_conversation_range"
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
    """Run a manual smoke check for ConversationFilterAgent."""
    agent = ConversationFilterAgent()
    result = await agent.run(
        task="<active character> 的最近3条发言",
        context={
            "platform": "qq",
            "platform_channel_id": "54369546",
            "current_timestamp": "2026-04-25T00:00:00+00:00",
            "known_facts": [],
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
