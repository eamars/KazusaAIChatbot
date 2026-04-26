"""Inner-loop agent for structured conversation filtering."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.agents.memory_retriever_agent import get_conversation
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)


_GENERATOR_PROMPT = """\
你是一个只负责 `get_conversation` 的结构化筛选参数生成器。

# 你的唯一职责
- 只为 `get_conversation` 生成参数。
- 优先从 `context` 与 `known_facts` 里提取明确的 platform / channel / user / display_name / time range。
- 如果上一轮 `feedback` 说结果太少，就优先扩大时间范围或提高 `limit`。
- 如果上一轮 `feedback` 说用户错了，就改 user filter；如果说时间错了，就改时间范围。
- 不要凭空猜不存在的 UUID；没有就留空。

# 输出格式
请只返回合法 JSON：
{
  "platform": "string or omitted",
  "platform_channel_id": "string or omitted",
  "limit": 5,
  "global_user_id": "string or omitted",
  "display_name": "string or omitted",
  "from_timestamp": "ISO-8601 or omitted",
  "to_timestamp": "ISO-8601 or omitted"
}
"""
_generator_llm = get_llm(temperature=0.0, top_p=1.0)

_JUDGE_PROMPT = """\
你是 `get_conversation` 的结果评估器。

# 任务
- 判断当前结果是否已经足以解决槽位。
- 如果未解决，反馈必须具体到下一轮该怎么调 filter。

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
_judge_llm = get_llm(temperature=0.0, top_p=1.0)


def _normalize_args(raw_args: dict) -> dict:
    """Keep only valid `get_conversation` arguments."""
    args: dict[str, object] = {}

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


async def _generator(task: str, context: dict, feedback: str) -> dict:
    """Generate one `get_conversation` argument dict."""
    system_prompt = SystemMessage(content=_GENERATOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(
            {
                "task": task,
                "context": context,
                "feedback": feedback,
            },
            ensure_ascii=False,
            default=str,
        )
    )

    response = await _generator_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(str(response.content))
    if not isinstance(result, dict):
        return {}
    return _normalize_args(result)


async def _tool(args: dict) -> object:
    """Execute `get_conversation` exactly once."""
    try:
        result = await get_conversation.ainvoke(args)
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info("conversation_filter_agent invalid args: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
    return result


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest structured conversation result resolves the slot."""
    system_prompt = SystemMessage(content=_JUDGE_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(
            {
                "task": task,
                "result": result,
            },
            ensure_ascii=False,
        )
    )

    response = await _judge_llm.ainvoke([system_prompt, human_message])
    verdict = parse_llm_json_output(str(response.content))
    if not isinstance(verdict, dict):
        return False, "评估输出无效，请调整时间范围或提高 limit。"

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return resolved, feedback


async def conversation_filter_agent(
    task: str,
    context: dict,
    max_attempts: int = 3,
) -> dict:
    """Resolve one slot through iterative structured conversation filtering.

    Args:
        task: Slot description produced by the outer-loop supervisor.
        context: Known facts and runtime hints that may constrain the filters.
        max_attempts: Maximum generator-tool-judge iterations to attempt.

    Returns:
        A dict containing whether the slot was resolved, the last serialized
        tool result, and the number of attempts consumed.
    """
    feedback = ""
    result = None
    resolved = False
    attempt = 0

    for attempt in range(max_attempts):
        args = await _generator(task, context, feedback)
        result = await _tool(args)
        resolved, feedback = await _judge(task, result)
        if resolved:
            break

    return {
        "resolved": resolved,
        "result": result,
        "attempts": attempt + 1,
    }


async def test_main() -> None:
    """Run a manual smoke check for the structured conversation-filter agent."""
    result = await conversation_filter_agent(
        task="在最近一周里筛出他提到该链接的几条原始消息",
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

    asyncio.run(test_main())
