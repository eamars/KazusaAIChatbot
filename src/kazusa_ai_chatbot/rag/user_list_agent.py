"""RAG helper agent: enumerate users by display-name predicates."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import RAG_SUBAGENT_LLM_API_KEY, RAG_SUBAGENT_LLM_BASE_URL, RAG_SUBAGENT_LLM_MODEL
from kazusa_ai_chatbot.db import list_users_by_display_name
from kazusa_ai_chatbot.rag.cache2_policy import (
    USER_LIST_CACHE_NAME,
    build_user_list_cache_key,
    build_user_list_dependencies,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.prompt_projection import project_runtime_context_for_llm
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_EXTRACTOR_PROMPT = """\
You are a parameter extractor for `list_users_by_display_name`.

# Capability
This agent enumerates users. It is for questions like:
- list users whose display names end with a character
- find users whose display names contain a literal fragment
- list known users or observed conversation participants matching a name pattern

# Source selection
- Use source="user_profiles" for known/profiled/registered users.
- Use source="conversation_participants" for users observed speaking in chat history,
  especially when the task mentions a channel, recent activity, or speakers.
- Use source="both" only when the task clearly asks for all users and does not
  distinguish between profiled users and observed chat participants.

# Display-name operators
- "ends_with": display name ends with the value.
- "starts_with": display name starts with the value.
- "equals": display name exactly equals the value.
- "contains": display name contains the value.

# Generation Procedure
1. Read `task` and identify the requested user source: profiled users, conversation participants, or both.
2. Extract the literal display-name fragment and choose the matching operator.
3. Read `context` only for runtime hints such as platform or channel.
4. Preserve explicit requested count in `limit`; otherwise keep the default.

# Input Format
{
  "task": "slot description from the outer RAG supervisor",
  "context": "known facts and runtime hints"
}

# Output Format
Return valid JSON only:
{
  "source": "user_profiles | conversation_participants | both",
  "display_name_operator": "equals | contains | starts_with | ends_with",
  "display_name_value": "literal string",
  "limit": 20
}
"""

_extractor_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

_SOURCES = {"user_profiles", "conversation_participants", "both"}
_DISPLAY_NAME_OPERATORS = {"equals", "contains", "starts_with", "ends_with"}


def _normalize_limit(raw_limit: object) -> int:
    """Normalize an LLM-provided limit into a bounded positive integer.

    Args:
        raw_limit: Limit value from LLM JSON.

    Returns:
        Integer limit clamped to a small enumeration-safe range.
    """
    if isinstance(raw_limit, int) and not isinstance(raw_limit, bool):
        return_value = max(1, min(raw_limit, 50))
        return return_value
    return 20


def _normalize_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM extractor output into safe user-list arguments.

    Args:
        raw_args: Parsed JSON object from the extractor LLM.

    Returns:
        Dict containing source, display_name_operator, display_name_value, and limit.
    """
    source = text_or_empty(raw_args.get("source")) or "user_profiles"
    if source not in _SOURCES:
        source = "user_profiles"

    operator = text_or_empty(raw_args.get("display_name_operator")) or "contains"
    if operator not in _DISPLAY_NAME_OPERATORS:
        operator = "contains"

    return_value = {
        "source": source,
        "display_name_operator": operator,
        "display_name_value": text_or_empty(raw_args.get("display_name_value")),
        "limit": _normalize_limit(raw_args.get("limit", 20)),
    }
    return return_value


async def _extract_user_list_args(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Extract constrained user-list parameters from a slot description.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Runtime hints such as platform and channel.

    Returns:
        Normalized arguments for the user-list DB helper.
    """
    system_prompt = SystemMessage(content=_EXTRACTOR_PROMPT)
    llm_context = project_runtime_context_for_llm(context)
    human_message = HumanMessage(
        content=json.dumps(
            {"task": task, "context": llm_context},
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _extractor_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)
    if not isinstance(result, dict):
        return_value = _normalize_args({})
        return return_value
    return_value = _normalize_args(result)
    return return_value


class UserListAgent(BaseRAGHelperAgent):
    """RAG helper agent that enumerates users matching a display-name predicate.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="user_list_agent",
            cache_name=USER_LIST_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Enumerate users matching a display-name predicate.

        Args:
            task: Slot description containing the user enumeration request.
            context: Runtime hints supplying platform and channel filters.
            max_attempts: Unused; kept for interface compatibility.

        Returns:
            Dict with resolved (bool), result payload, and attempts count.
        """
        del max_attempts

        args = await _extract_user_list_args(task, context)
        display_name_value = args["display_name_value"]
        if not display_name_value:
            return_value = self.with_cache_status(
                {
                    "resolved": False,
                    "result": {
                        "error": "display_name_value is required for user enumeration.",
                        "args": args,
                    },
                    "attempts": 1,
                },
                hit=False,
                reason="skipped_missing_display_name_value",
            )
            return return_value

        cache_key = build_user_list_cache_key(args, context)
        cached = await self.read_cache(cache_key)
        if cached is not None:
            return_value = self.with_cache_status(
                {"resolved": True, "result": cached, "attempts": 0},
                hit=True,
                reason="hit",
                cache_key=cache_key,
            )
            return return_value

        platform = str(context.get("platform") or "").strip() or None
        platform_channel_id = str(context.get("platform_channel_id") or "").strip() or None
        users = await list_users_by_display_name(
            value=display_name_value,
            operator=args["display_name_operator"],
            source=args["source"],
            platform=platform,
            platform_channel_id=platform_channel_id,
            limit=args["limit"],
        )

        result = {"users": users}
        if users:
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_user_list_dependencies(args, context),
                metadata={"source": args.get("source", "user_profiles")},
            )

        return_value = self.with_cache_status(
            {
                "resolved": bool(users),
                "result": result,
                "attempts": 1,
            },
            hit=False,
            reason="miss_stored" if users else "miss_unresolved",
            cache_key=cache_key,
        )
        return return_value


async def _test_main() -> None:
    """Run a manual smoke check for UserListAgent."""
    agent = UserListAgent()
    result = await agent.run(
        task="列出用户名为“小”开头的用户",
        context={"platform": "qq"},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    result = await agent.run(
        task="列出用户名为“大”开头的用户",
        context={"platform": "qq"},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    result = await agent.run(
        task="列出用户名为“小”开头的用户",
        context={"platform": "qq"},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
