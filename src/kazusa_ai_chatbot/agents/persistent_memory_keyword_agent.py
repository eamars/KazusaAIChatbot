"""Inner-loop agent for keyword-based persistent-memory search."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.agents.memory_retriever_agent import search_persistent_memory_keyword
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)


_GENERATOR_PROMPT = """\
你是一个只负责 `search_persistent_memory_keyword` 的检索参数生成器。

# 你的唯一职责
- 只为 `search_persistent_memory_keyword` 生成参数。
- `keyword` 必须是最短且不歧义的核心词或短语，不能是完整句子。
- 如果目标是专有名词、昵称、事件名、文件名或短标签，就优先保留字面锚点。
- 如果 `feedback` 指出词太具体、无结果或需要更泛化的表达，下一轮必须显著调整。
- 如果 `context` 明确给出来源用户或记忆类型，可以加上过滤。

# 输出格式
请只返回合法 JSON：
{
  "keyword": "string",
  "top_k": 5,
  "source_global_user_id": "string or omitted",
  "memory_type": "string or omitted"
}
"""
_generator_llm = get_llm(temperature=0.0, top_p=1.0)

_JUDGE_PROMPT = """\
你是 `search_persistent_memory_keyword` 的结果评估器。

# 任务
- 判断当前结果是否已经足以解决槽位。
- 如果未解决，反馈必须明确告诉下一轮怎么改关键词。

# 常见反馈方向
- 关键词太长，请收缩到核心词
- 关键词太细，请换成更常见或更泛一点的叫法
- 没有匹配，请换同义词或去掉多余修饰
- 需要补充/移除 memory_type 过滤

# 输出格式
请只返回合法 JSON：
{
  "resolved": true or false,
  "feedback": "string"
}
"""
_judge_llm = get_llm(temperature=0.0, top_p=1.0)


def _normalize_args(raw_args: dict) -> dict:
    """Keep only valid `search_persistent_memory_keyword` arguments."""
    args: dict[str, object] = {}

    keyword = str(raw_args.get("keyword", "")).strip()
    if keyword:
        args["keyword"] = keyword

    top_k = raw_args.get("top_k", 5)
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
        args["top_k"] = top_k
    else:
        args["top_k"] = 5

    for key in ("source_global_user_id", "memory_type"):
        raw_val = raw_args.get(key)
        if raw_val is None:
            continue
        value = str(raw_val).strip()
        if value:
            args[key] = value

    return args


async def _generator(task: str, context: dict, feedback: str) -> dict:
    """Generate one `search_persistent_memory_keyword` argument dict."""
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
    """Execute `search_persistent_memory_keyword` exactly once."""
    try:
        result = await search_persistent_memory_keyword.ainvoke(args)
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info("persistent_memory_keyword_agent invalid args: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
    return result


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest keyword memory-search result resolves the slot."""
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
        return False, "评估输出无效，请把关键词改成更短或更常见的叫法。"

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return resolved, feedback


async def persistent_memory_keyword_agent(
    task: str,
    context: dict,
    max_attempts: int = 3,
) -> dict:
    """Resolve one slot through iterative keyword search over persistent memories.

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
    """Run a manual smoke check for the persistent-memory keyword agent."""
    result = await persistent_memory_keyword_agent(
        task="找出和‘小钳子’有关的持久记忆条目",
        context={"known_facts": []},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_main())
