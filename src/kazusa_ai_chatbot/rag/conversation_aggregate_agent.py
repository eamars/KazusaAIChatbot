"""RAG helper agent: factual conversation-history aggregates."""

from __future__ import annotations

import datetime
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import RAG_SUBAGENT_LLM_API_KEY, RAG_SUBAGENT_LLM_BASE_URL, RAG_SUBAGENT_LLM_MODEL
from kazusa_ai_chatbot.db import aggregate_conversation_by_user
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

_EXTRACTOR_PROMPT = """\
You are a parameter extractor for `aggregate_conversation_by_user`.

# Capability
This agent computes factual aggregates over conversation history. It fetches
evidence and performs simple math only. It must not provide opinions,
relationship judgments, motives, or persona interpretation.

# Supported aggregate
- Count messages grouped by user, optionally filtered by a literal keyword,
  known user, channel, and time window.

# Use cases
- Who spoke the most recently?
- How many messages did a resolved user send?
- Who mentioned a literal term most often?
- Which users talked about a known exact keyword?

# Output format
Return valid JSON only:
{
  "aggregate": "message_count_by_user",
  "keyword": "literal string or empty",
  "time_window": "recent | today | yesterday | all",
  "limit": 10
}
"""

_extractor_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)
_TIME_WINDOWS = {"recent", "today", "yesterday", "all"}


def _normalize_limit(raw_limit: object) -> int:
    """Normalize an LLM-provided limit into a bounded positive integer.

    Args:
        raw_limit: Limit value from parsed LLM JSON.

    Returns:
        Integer limit clamped to the aggregate result range.
    """
    if isinstance(raw_limit, int) and not isinstance(raw_limit, bool):
        return max(1, min(raw_limit, 50))
    return 10


def _normalize_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Normalize extractor output into safe aggregate arguments.

    Args:
        raw_args: Parsed JSON object from the extractor LLM.

    Returns:
        Dict containing aggregate, keyword, time_window, and limit.
    """
    time_window = str(raw_args.get("time_window", "recent")).strip()
    if time_window not in _TIME_WINDOWS:
        time_window = "recent"

    return {
        "aggregate": "message_count_by_user",
        "keyword": str(raw_args.get("keyword", "")).strip(),
        "time_window": time_window,
        "limit": _normalize_limit(raw_args.get("limit", 10)),
    }


async def _extract_aggregate_args(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Extract constrained aggregate parameters from a slot description.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Runtime hints and known facts collected by supervisor2.

    Returns:
        Normalized aggregate arguments.
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


def _parse_current_timestamp(context: dict[str, Any]) -> datetime.datetime:
    """Parse current timestamp from context, falling back to current UTC time.

    Args:
        context: Runtime context passed to the agent.

    Returns:
        Timezone-aware UTC datetime.
    """
    raw_timestamp = str(context.get("current_timestamp") or "").strip()
    if raw_timestamp:
        try:
            parsed = datetime.datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.datetime.now(datetime.timezone.utc)
    else:
        parsed = datetime.datetime.now(datetime.timezone.utc)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def _time_bounds(
    time_window: str, context: dict[str, Any]
) -> tuple[str | None, str | None]:
    """Convert a coarse time-window label into ISO timestamp bounds.

    Args:
        time_window: One of recent, today, yesterday, or all.
        context: Runtime context containing current_timestamp when available.

    Returns:
        Tuple of from_timestamp and to_timestamp strings, either of which may be None.
    """
    now = _parse_current_timestamp(context)
    if time_window == "all":
        return None, None
    if time_window == "recent":
        return (now - datetime.timedelta(days=7)).isoformat(), now.isoformat()

    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if time_window == "today":
        return start_of_today.isoformat(), now.isoformat()

    start_of_yesterday = start_of_today - datetime.timedelta(days=1)
    return start_of_yesterday.isoformat(), start_of_today.isoformat()


def _slot_number(task: str) -> int | None:
    """Extract a referenced slot number from a task description.

    Args:
        task: Slot description that may contain "slot N".

    Returns:
        Integer slot number, or None when no reference is present.
    """
    match = re.search(r"slot\s+(\d+)", task, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _global_user_id_from_known_fact(fact: dict[str, Any]) -> str:
    """Extract a global_user_id from a known fact payload.

    Args:
        fact: One entry from supervisor2 known_facts.

    Returns:
        UUID string when present, otherwise an empty string.
    """
    raw_result = fact.get("raw_result")
    if isinstance(raw_result, dict):
        value = raw_result.get("global_user_id")
        if value:
            return str(value)
    return ""


def _resolved_global_user_id(task: str, context: dict[str, Any]) -> str | None:
    """Resolve an optional user filter from context or referenced known facts.

    Args:
        task: Slot description selected by the dispatcher.
        context: Agent context containing known_facts and runtime filters.

    Returns:
        Resolved global_user_id, or None if the aggregate should include all users.
    """
    context_user_id = str(context.get("global_user_id") or "").strip()
    if context_user_id:
        return context_user_id

    slot_number = _slot_number(task)
    if slot_number is None:
        return None

    known_facts = context.get("known_facts", [])
    if not isinstance(known_facts, list) or slot_number < 1 or slot_number > len(known_facts):
        return None

    user_id = _global_user_id_from_known_fact(known_facts[slot_number - 1])
    return user_id or None


class ConversationAggregateAgent(BaseRAGHelperAgent):
    """RAG helper agent that computes factual aggregates over conversation history.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="conversation_aggregate_agent",
            cache_name="",
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Compute factual aggregates over conversation history.

        Args:
            task: Slot description containing the aggregate request.
            context: Runtime hints supplying platform, channel, timestamp, and known facts.
            max_attempts: Unused; kept for interface compatibility.

        Returns:
            Dict with resolved (bool), result payload, and attempts count.
        """
        del max_attempts

        args = await _extract_aggregate_args(task, context)
        from_timestamp, to_timestamp = _time_bounds(args["time_window"], context)
        platform = str(context.get("platform") or "").strip() or None
        platform_channel_id = str(context.get("platform_channel_id") or "").strip() or None
        global_user_id = _resolved_global_user_id(task, context)
        keyword = args["keyword"] or None

        result = await aggregate_conversation_by_user(
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=global_user_id,
            keyword=keyword,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            limit=args["limit"],
        )

        return self.with_cache_status(
            {
                "resolved": bool(result["rows"]),
                "result": {
                    "aggregate": args["aggregate"],
                    "time_window": args["time_window"],
                    **result,
                },
                "attempts": 1,
            },
            hit=False,
            reason="agent_not_cacheable",
        )


async def _test_main() -> None:
    """Run a manual smoke check for ConversationAggregateAgent."""
    agent = ConversationAggregateAgent()
    result = await agent.run(
        task="Conversation-aggregate: count recent messages by user",
        context={"platform": "qq"},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
