"""RAG helper agent: keyword-based persistent-memory search."""

from __future__ import annotations

import json
import logging
from string import Template
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.config import (

    CHARACTER_GLOBAL_USER_ID,
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_SEARCH_MAX_TOP_K,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
    RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    RAG_SUBAGENT_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.rag.memory_retrieval_tools import search_persistent_memory_keyword
from kazusa_ai_chatbot.rag.cache2_policy import (
    PERSISTENT_MEMORY_KEYWORD_CACHE_NAME,
    build_persistent_memory_keyword_cache_key,
    build_persistent_memory_keyword_dependencies,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.prompt_projection import (
    project_runtime_context_for_llm,
    project_tool_result_for_llm,
)
from kazusa_ai_chatbot.rag.search_runtime import (
    apply_source_memory_runtime_constraints,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = Template('''\
你只为 `search_persistent_memory_keyword` 生成检索参数。

# 范围
- 只生成 `search_persistent_memory_keyword` 的参数。
- `keyword` 必须是最短且不歧义的核心词或短语，不是完整句子。
- proper nouns、nicknames、event names、filenames 或 short tags 优先使用字面 anchor。
- 如果 `feedback` 指出词太具体、没有结果或需要更宽表达，下一次 keyword 必须有意义地改变。
- `source_global_user_id` 是隐私边界，不是相关性提示。默认省略；只有任务明确
  要求某个来源用户触发、提供或承诺的记忆时才填写。
- 生成的控制/状态说明使用中文。保留 literal anchors、names、quotes、URLs、
  filenames 和 source text 的原始语言。

# 生成步骤
1. 读取 `task`，找出最短且不歧义的字面 anchor。
2. 读取 `context`；只有任务明确要求 memory source-user filter 时才使用
   `source_global_user_id`。
3. 如果 feedback 说无结果、太具体或需要变宽，显著改变 keyword。
4. 输出一个 keyword 加必要过滤字段。

# 输入格式
{
  "task": "外层 RAG supervisor 给出的槽位描述",
  "context": "已知事实和运行时提示",
  "feedback": "上一轮 judge feedback，或空字符串"
}

# 输出格式
只返回有效 JSON：
{
  "keyword": "string",
  "top_k": $default_top_k,
  "source_global_user_id": "string or omitted"
}
''').substitute(default_top_k=RAG_SEARCH_DEFAULT_TOP_K)
_llm_interface = LLInterface()
_generator_llm = LLInterface()
_judge_llm = LLInterface()
_generator_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="RAG_SUBAGENT_LLM",
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
    model=RAG_SUBAGENT_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=RAG_SUBAGENT_LLM_THINKING_ENABLED,
    ),
)

_JUDGE_PROMPT = '''\
你判断 `search_persistent_memory_keyword` 的结果是否解决当前槽位。

# 任务
- 判断当前结果是否足够解决槽位。
- 如果未解决，feedback 必须清楚说明下一次应如何修改 keyword。

# 生成步骤
1. 读取 `task`，识别需要的 memory evidence。
2. 检查 `result` 中的相关 durable memory、错误或空输出。
3. 只有结果足够解决槽位时，才返回 `resolved: true`。
4. 如果未解决，说明下一次 keyword 应更短、更宽、改用同义词，或使用不同来源过滤。

# 输入格式
{
  "task": "外层 RAG supervisor 给出的槽位描述",
  "result": "search_persistent_memory_keyword 的工具结果"
}

# 常见反馈方向
- Keyword 太长；缩成核心词。
- Keyword 太具体；使用更常见或更宽的表达。
- 没有匹配；使用同义词或移除额外修饰。
- 添加或移除 source-user filtering。
- 反馈说明使用中文；source text 保持原文。

# 输出格式
只返回有效 JSON：
{
  "resolved": true or false,
  "feedback": "string"
}
'''
_judge_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="RAG_SUBAGENT_LLM",
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
    model=RAG_SUBAGENT_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=RAG_SUBAGENT_LLM_THINKING_ENABLED,
    ),
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

    top_k = raw_args.get("top_k", RAG_SEARCH_DEFAULT_TOP_K)
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
        args["top_k"] = min(top_k, RAG_SEARCH_MAX_TOP_K)
    else:
        args["top_k"] = RAG_SEARCH_DEFAULT_TOP_K

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
    llm_context = project_runtime_context_for_llm(context)
    human_message = HumanMessage(
        content=json.dumps(
            {"task": task, "context": llm_context, "feedback": feedback},
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _generator_llm.ainvoke([system_prompt, human_message], config=_generator_llm_config)
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


def _apply_runtime_constraints(
    args: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Apply trusted source-memory filters after keyword generation."""

    constrained = apply_source_memory_runtime_constraints(args, context=context)
    return constrained


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest keyword memory-search result resolves the slot.

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
    response = await _judge_llm.ainvoke([system_prompt, human_message], config=_judge_llm_config)
    verdict = parse_llm_json_output(response.content)
    if not isinstance(verdict, dict):
        return_value = False, "judge 输出无效；使用更短或更常见的 keyword。"
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
            args = _apply_runtime_constraints(args, context)
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
