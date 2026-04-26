"""Inner-loop agent for keyword-based conversation search."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.agents.memory_retriever_agent import search_conversation_keyword
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)


_GENERATOR_PROMPT = """\
You are a parameter generator for `search_conversation_keyword`.
The tool accepts ONE keyword string and runs a case-insensitive regex match against message content.

## Keyword selection rules

1. Pick the single MOST SPECIFIC term — the one least likely to appear in unrelated messages.
   - GOOD: "qwen27b"  (unique technical name)
   - BAD:  "5090 qwen27b"  (space between terms will NOT match "5090跑qwen27b" — different separator)

2. When the task mentions two or more terms, do NOT join them.
   Choose whichever one term is rarest/most distinctive.
   Example: task says "find messages with '5090' and 'qwen27b'" → use keyword="qwen27b", not "5090 qwen27b".

3. For URLs, filenames, or exact phrases — keep the shortest unambiguous anchor
   (e.g. "xhslink.com", "cookie管理器", "play的一环").

4. Never pass a full sentence as the keyword.

5. If feedback says "no results" or "too many results", change the keyword meaningfully:
   - No results → try a shorter term or a synonym
   - Too many results → add filters (global_user_id, platform_channel_id, time range) rather than lengthening the keyword

## Context filters
If context contains platform / platform_channel_id / global_user_id / time bounds, include them.

## Output format
Return valid JSON only:
{
  "keyword": "string",
  "global_user_id": "string or omitted",
  "top_k": 5,
  "platform": "string or omitted",
  "platform_channel_id": "string or omitted",
  "from_timestamp": "ISO-8601 or omitted",
  "to_timestamp": "ISO-8601 or omitted"
}
"""
_generator_llm = get_llm(temperature=0.0, top_p=1.0)

_JUDGE_PROMPT = """\
你是 `search_conversation_keyword` 的结果评估器。

# 任务
- 判断结果是否已经解决槽位。
- 如果未解决，必须输出可以立刻执行的反馈。

# 常见反馈方向
- 关键词太长，请只保留核心 noun / phrase
- 没有匹配，请尝试更短或更常见的同义词
- 需要加入或移除用户/时间过滤
- 命中内容偏题，需要换成真正的锚点词

# 输出格式
请只返回合法 JSON：
{
  "resolved": true or false,
  "feedback": "string"
}
"""
_judge_llm = get_llm(temperature=0.0, top_p=1.0)


def _normalize_args(raw_args: dict) -> dict:
    """Keep only valid `search_conversation_keyword` arguments."""
    args: dict[str, object] = {}

    keyword = str(raw_args.get("keyword", "")).strip()
    if keyword:
        args["keyword"] = keyword

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


async def _generator(task: str, context: dict, feedback: str) -> dict:
    """Generate one `search_conversation_keyword` argument dict."""
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
    """Execute `search_conversation_keyword` exactly once."""
    try:
        result = await search_conversation_keyword.ainvoke(args)
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info("conversation_keyword_agent invalid args: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
    return result


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest keyword search result resolves the slot."""
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
        return False, "评估输出无效，请把关键词缩短成最核心的词。"

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return resolved, feedback


async def conversation_keyword_agent(
    task: str,
    context: dict,
    max_attempts: int = 3,
) -> dict:
    """Resolve one slot through iterative keyword search over conversation history.

    Args:
        task: Slot description produced by the outer-loop supervisor.
        context: Known facts and runtime hints that may constrain the search.
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
    """Run a manual smoke check for the conversation keyword agent."""
    result = await conversation_keyword_agent(
        task="姐姐",
        context={
            "platform": "qq",
            "platform_channel_id": "902317662",
            "known_facts": [],
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_main())
