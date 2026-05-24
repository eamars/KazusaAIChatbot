"""RAG helper agent: keyword-based conversation search."""

from __future__ import annotations

import json
import logging
from string import Template
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.config import (
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_SEARCH_MAX_TOP_K,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.memory_retrieval_tools import search_conversation_keyword
from kazusa_ai_chatbot.rag.cache2_policy import (
    CONVERSATION_KEYWORD_CACHE_NAME,
    build_conversation_keyword_cache_key,
    build_conversation_keyword_dependencies,
    is_closed_historical_range,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.prompt_projection import (
    project_runtime_context_for_llm,
    project_tool_result_for_llm,
)
from kazusa_ai_chatbot.rag.search_runtime import (
    apply_conversation_runtime_constraints,
)
from kazusa_ai_chatbot.time_boundary import local_llm_datetime_to_storage_utc_iso
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = Template('''\
你是 `search_conversation_keyword` 的参数生成器。
该工具只接受一个 keyword 字符串，并对消息内容执行大小写不敏感的 regex 匹配。

## Keyword 选择规则

1. 选择唯一且最具体的词，也就是最不容易出现在无关消息里的词。
   - GOOD: "qwen27b"  (unique technical name)
   - BAD:  "5090 qwen27b"  (词间空格不会匹配 "5090跑qwen27b"，分隔符不同)

2. 当任务提到两个或更多词时，不要把它们拼接。
   选择其中最稀有、最有辨识度的一个。
   示例：task 说 "find messages with '5090' and 'qwen27b'"，
   应使用 keyword="qwen27b"，不要用 "5090 qwen27b"。

3. URLs、filenames 或 exact phrases 使用最短且不歧义的 anchor
   (e.g. "xhslink.com", "cookie管理器", "play的一环")。

4. 绝不要把完整句子作为 keyword。

5. 如果 feedback 说 "no results" 或 "too many results"，要有意义地改变 keyword：
   - No results → 尝试更短的词或同义词
   - Too many results → 添加过滤器 (global_user_id, platform_channel_id, time range)，不要拉长 keyword

## Context filters
如果 context 包含 platform / platform_channel_id / global_user_id / time bounds，就带上它们。

# 生成步骤
1. 读取 `task`，识别最有辨识度的字面 anchor。
2. `context` 只用于 platform、channel、user 或 time bounds 等过滤器。
3. 如果 `feedback` 指出 no results 或 too many results，有意义地修改 keyword 或过滤器。
4. 输出一个 keyword 字符串；不要把多个词拼成一句话。

# 输入格式
{
  "task": "外层 RAG supervisor 给出的槽位描述",
  "context": "已知事实和运行时提示",
  "feedback": "上一轮 judge feedback，或空字符串"
}

## 输出格式
只返回有效 JSON：
{
  "keyword": "string",
  "global_user_id": "string or omitted",
  "top_k": $default_top_k,
  "platform": "string or omitted",
  "platform_channel_id": "string or omitted",
  "from_timestamp": "local YYYY-MM-DD HH:MM or omitted",
  "to_timestamp": "local YYYY-MM-DD HH:MM or omitted"
}
''').substitute(default_top_k=RAG_SEARCH_DEFAULT_TOP_K)
_generator_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

_JUDGE_PROMPT = '''\
你判断 `search_conversation_keyword` 的结果是否解决当前槽位。

# 任务
- 判断结果是否解决槽位。
- 如果未解决，输出下一次 keyword/filter 尝试可以直接使用的 feedback。

# 生成步骤
1. 读取 `task`，识别必须匹配的字面 anchor。
2. 检查 `result` 中的相关消息、错误或空输出。
3. 只有结果足够解决槽位时，才返回 `resolved: true`。
4. 如果未解决，为下一次 keyword 或 filters 给出具体修正。
5. 生成的控制/状态说明使用中文。保留 source message text、display names、
   literal anchors、URLs 和 filenames 的原始语言。

# 输入格式
{
  "task": "外层 RAG supervisor 给出的槽位描述",
  "result": "search_conversation_keyword 的工具结果"
}

# 常见反馈方向
- Keyword 太长；只保留核心名词或短语。
- 没有匹配；尝试更短的词或常见同义词。
- 添加或移除用户/时间过滤器。
- 命中偏题；改用真正的 anchor term。

# 输出格式
只返回有效 JSON：
{
  "resolved": true or false,
  "feedback": "string"
}
'''
_judge_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


def _normalize_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Keep only valid `search_conversation_keyword` arguments.

    Args:
        raw_args: Raw dict from the generator LLM.

    Returns:
        Validated argument dict for ``search_conversation_keyword``.
    """
    args: dict[str, Any] = {}

    keyword = text_or_empty(raw_args.get("keyword"))
    if keyword:
        args["keyword"] = keyword

    top_k = raw_args.get("top_k", RAG_SEARCH_DEFAULT_TOP_K)
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
        args["top_k"] = min(top_k, RAG_SEARCH_MAX_TOP_K)
    else:
        args["top_k"] = RAG_SEARCH_DEFAULT_TOP_K

    for key in (
        "global_user_id",
        "platform",
        "platform_channel_id",
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
            args[key] = local_llm_datetime_to_storage_utc_iso(value)
        except ValueError as exc:
            logger.debug(f"Dropping invalid {key} from LLM output: {exc}")

    return args


async def _generator(task: str, context: dict[str, Any], feedback: str) -> dict[str, Any]:
    """Generate one `search_conversation_keyword` argument dict.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Known facts and runtime hints.
        feedback: Judge feedback from the previous attempt, or empty string.

    Returns:
        Normalized arguments for ``search_conversation_keyword``.
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
    """Execute `search_conversation_keyword` exactly once.

    Args:
        args: Normalized arguments for the tool.

    Returns:
        Tool result or an error dict on invalid arguments.
    """
    try:
        return_value = await search_conversation_keyword.ainvoke(args)
        return return_value
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info(f'conversation_keyword_agent invalid args: {exc}')
        return_value = {"error": f"{type(exc).__name__}: {exc}"}
        return return_value


def _apply_runtime_constraints(
    args: dict[str, Any],
    task: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Apply runtime-owned filters after keyword argument generation."""

    constrained = apply_conversation_runtime_constraints(
        args,
        context=context,
        task=task,
        literal_anchor_limit=1,
    )
    constrained.pop("literal_anchors", None)
    return constrained


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest keyword search result resolves the slot.

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
        return_value = False, "judge 输出无效；把 keyword 缩短为核心词。"
        return return_value

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return_value = resolved, feedback
    return return_value


class ConversationKeywordAgent(BaseRAGHelperAgent):
    """RAG helper agent that resolves a slot through iterative keyword conversation search.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="conversation_keyword_agent",
            cache_name=CONVERSATION_KEYWORD_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Resolve one slot through iterative keyword search over conversation history.

        Args:
            task: Slot description produced by the outer-loop supervisor.
            context: Known facts and runtime hints that may constrain the search.
            max_attempts: Maximum generator-tool-judge iterations to attempt.

        Returns:
            Dict with resolved (bool), result (last tool result), and attempts count.
        """
        cache_key = build_conversation_keyword_cache_key(task, context)
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
            args = _apply_runtime_constraints(args, task, context)
            result = await _tool(args)
            resolved, feedback = await _judge(task, result)
            if resolved:
                break

        if resolved and is_closed_historical_range(args):
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_conversation_keyword_dependencies(args, context),
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
    """Run a manual smoke check for ConversationKeywordAgent."""
    agent = ConversationKeywordAgent()
    result = await agent.run(
        task="包含了姐姐关键词",
        context={
            "platform": "qq",
            "platform_channel_id": "902317662",
            "known_facts": [],
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
