"""RAG helper agent: structured conversation filtering."""

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
from kazusa_ai_chatbot.rag.search_runtime import (
    apply_conversation_filter_runtime_constraints,
)
from kazusa_ai_chatbot.time_boundary import local_llm_datetime_to_storage_utc_iso
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = Template("""\
You generate structured filter parameters only for `get_conversation`.

# Scope
- Produce parameters only for `get_conversation`.
- Prefer explicit platform, channel, user, display_name, and time range from
  `context` and `known_facts`.
- If task or context.original_query explicitly asks for a count, such as
  "最近3条" or "last 3 messages", put that number in `limit`.
- If previous `feedback` says too few results, widen the time range or raise
  `limit`.
- If previous `feedback` says the user filter is wrong, change the user
  filter. If it says the time range is wrong, change the time range.
- Do not invent nonexistent UUIDs; leave unavailable fields empty.
- Use English for generated control/status text. Preserve display names,
  anchors, quotes, URLs, filenames, and source text in their original language.

# Generation Procedure
1. Read `task` and identify which structured conversation records are needed.
2. Extract explicit platform, channel, user, display_name, time range, and
   limit from `context` and `known_facts`.
3. If feedback says too few results, wrong user, or wrong time, adjust that
   filter.
4. Do not guess UUIDs; use display_name or omit user filtering when no clear
   UUID exists.

# Input Format
{
  "task": "slot description from the outer RAG supervisor",
  "context": "known facts and runtime hints",
  "feedback": "previous judge feedback, or empty string"
}

# Output Format
Return valid JSON only:
{
  "platform": "string or omitted",
  "platform_channel_id": "string or omitted",
  "limit": $default_limit,
  "global_user_id": "string or omitted",
  "display_name": "string or omitted",
  "from_timestamp": "local YYYY-MM-DD HH:MM or omitted",
  "to_timestamp": "local YYYY-MM-DD HH:MM or omitted"
}
""").substitute(default_limit=RAG_SEARCH_DEFAULT_TOP_K)
_generator_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

_JUDGE_PROMPT = """\
You judge whether a `get_conversation` result resolves the current slot.

# Task
- Decide whether the current result is enough to resolve the slot.
- If unresolved, feedback must specify how the next attempt should adjust
  filters.

# Audit Procedure
1. Read `task` and identify the needed conversation record range.
2. Check whether `result` contains enough records, the right user, and the
   right time range.
3. Return `resolved: true` only when the result is enough to resolve the slot.
4. If unresolved, state whether the next attempt should widen time, raise
   limit, change user filter, or adjust direction.

# Input Format
{
  "task": "slot description from the outer RAG supervisor",
  "result": "tool result from get_conversation"
}

# Common Feedback Directions
- Too few results; widen time range or raise limit.
- Wrong filter; change global_user_id or display_name.
- Time range too narrow or direction reversed.
- Relevant records already found; stop.
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
    """Keep only valid `get_conversation` arguments.

    Args:
        raw_args: Raw dict from the generator LLM.

    Returns:
        Validated argument dict for ``get_conversation``.
    """
    args: dict[str, Any] = {}

    limit = raw_args.get("limit", RAG_SEARCH_DEFAULT_TOP_K)
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        args["limit"] = min(limit, RAG_SEARCH_MAX_TOP_K)
    else:
        args["limit"] = RAG_SEARCH_DEFAULT_TOP_K

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
            args[key] = local_llm_datetime_to_storage_utc_iso(value)
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


def _apply_runtime_constraints(
    args: dict[str, Any],
    task: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Apply runtime-owned filters after structured argument generation."""

    constrained = apply_conversation_filter_runtime_constraints(
        args,
        context=context,
        task=task,
    )
    return constrained


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
        return_value = False, "Invalid judge output; adjust the time range or raise limit."
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
            args = _apply_runtime_constraints(args, task, context)
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
            "current_timestamp_utc": "2026-04-25T00:00:00+00:00",
            "known_facts": [],
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
