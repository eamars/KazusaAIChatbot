"""Inner-loop agent for enumerating users by display-name predicates."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.db import list_users_by_display_name
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

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

# Output format
Return valid JSON only:
{
  "source": "user_profiles | conversation_participants | both",
  "display_name_operator": "equals | contains | starts_with | ends_with",
  "display_name_value": "literal string",
  "limit": 20
}
"""

_extractor_llm = get_llm(temperature=0.0, top_p=1.0)

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
        return max(1, min(raw_limit, 50))
    return 20


def _normalize_args(raw_args: dict) -> dict:
    """Normalize LLM extractor output into safe user-list arguments.

    Args:
        raw_args: Parsed JSON object from the extractor LLM.

    Returns:
        Dict containing source, display_name_operator, display_name_value, and limit.
    """
    source = str(raw_args.get("source", "user_profiles")).strip()
    if source not in _SOURCES:
        source = "user_profiles"

    operator = str(raw_args.get("display_name_operator", "contains")).strip()
    if operator not in _DISPLAY_NAME_OPERATORS:
        operator = "contains"

    return {
        "source": source,
        "display_name_operator": operator,
        "display_name_value": str(raw_args.get("display_name_value", "")).strip(),
        "limit": _normalize_limit(raw_args.get("limit", 20)),
    }


async def _extract_user_list_args(task: str, context: dict) -> dict:
    """Extract constrained user-list parameters from a slot description.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Runtime hints such as platform and channel.

    Returns:
        Normalized arguments for the user-list DB helper.
    """
    system_prompt = SystemMessage(content=_EXTRACTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps({"task": task, "context": context}, ensure_ascii=False, default=str)
    )
    response = await _extractor_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(str(response.content))
    if not isinstance(result, dict):
        return _normalize_args({})
    return _normalize_args(result)


async def user_list_agent(
    task: str,
    context: dict,
    max_attempts: int = 3,
) -> dict:
    """Enumerate users matching a display-name predicate.

    Args:
        task: Slot description containing the user enumeration request.
        context: Runtime hints supplying platform and channel filters.
        max_attempts: Unused; kept for supervisor2 agent interface compatibility.

    Returns:
        Dict with resolved (bool), result payload, and attempts count.
    """
    del max_attempts

    args = await _extract_user_list_args(task, context)
    display_name_value = args["display_name_value"]
    if not display_name_value:
        return {
            "resolved": False,
            "result": {
                "error": "display_name_value is required for user enumeration.",
                "args": args,
            },
            "attempts": 1,
        }

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

    return {
        "resolved": bool(users),
        "result": {
            "users": users,
            # "query": {
            #     "source": args["source"],
            #     "display_name_operator": args["display_name_operator"],
            #     "display_name_value": display_name_value,
            #     "platform": platform,
            #     "platform_channel_id": platform_channel_id,
            #     "limit": args["limit"],
            # },
        },
        "attempts": 1,
    }


async def test_main() -> None:
    """Run a manual smoke check for the user-list agent."""
    result = await user_list_agent(
        task="User-list: list known users whose display names end with '子'",
        context={
            "platform": "qq",
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_main())
