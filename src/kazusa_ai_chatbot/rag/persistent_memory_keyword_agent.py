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
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = Template("""\
You generate retrieval parameters only for `search_persistent_memory_keyword`.

# Scope
- Produce parameters only for `search_persistent_memory_keyword`.
- `keyword` must be the shortest unambiguous core term or phrase, not a full
  sentence.
- Prefer literal anchors for proper nouns, nicknames, event names, filenames,
  or short tags.
- If `feedback` says the term is too specific, no results were found, or a
  broader expression is needed, change the next keyword meaningfully.
- `source_global_user_id` is a privacy boundary, not a relevance hint. Omit it
  by default; fill it only when the task explicitly asks for memory triggered,
  provided, or promised by one source user.
- Use English for generated control/status text. Preserve literal anchors,
  names, quotes, URLs, filenames, and source text in their original language.

# Generation Procedure
1. Read `task` and find the shortest unambiguous literal anchor.
2. Read `context`; use `source_global_user_id` only when the task explicitly
   requires a memory source-user filter.
3. If feedback says no result, too specific, or needs broadening, change the
   keyword substantially.
4. Output one keyword plus necessary filter fields.

# Input Format
{
  "task": "slot description from the outer RAG supervisor",
  "context": "known facts and runtime hints",
  "feedback": "previous judge feedback, or empty string"
}

# Output Format
Return valid JSON only:
{
  "keyword": "string",
  "top_k": $default_top_k,
  "source_global_user_id": "string or omitted"
}
""").substitute(default_top_k=RAG_SEARCH_DEFAULT_TOP_K)
_generator_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

_JUDGE_PROMPT = """\
You judge whether a `search_persistent_memory_keyword` result resolves the current slot.

# Task
- Decide whether the current result is enough to resolve the slot.
- If unresolved, feedback must clearly tell the next attempt how to change the
  keyword.

# Audit Procedure
1. Read `task` and identify the memory evidence needed.
2. Inspect `result` for relevant durable memory, errors, or empty output.
3. Return `resolved: true` only when the result is enough to resolve the slot.
4. If unresolved, state whether the next keyword should be shorter, broader,
   a synonym, or use different source filtering.

# Input Format
{
  "task": "slot description from the outer RAG supervisor",
  "result": "tool result from search_persistent_memory_keyword"
}

# Common Feedback Directions
- Keyword too long; shrink to the core term.
- Keyword too specific; use a more common or broader expression.
- No match; use a synonym or remove extra modifiers.
- Add or remove source-user filtering.
- Use English for generated control/status text while preserving source text.

# Output Format
Return valid JSON only:
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
    response = await _judge_llm.ainvoke([system_prompt, human_message])
    verdict = parse_llm_json_output(response.content)
    if not isinstance(verdict, dict):
        return_value = False, "Invalid judge output; use a shorter or more common keyword."
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
