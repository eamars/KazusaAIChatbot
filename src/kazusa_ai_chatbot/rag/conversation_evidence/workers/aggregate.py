
"""RAG helper agent: factual conversation-history aggregates."""

from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    RAG_SUBAGENT_LLM_MODEL,
    RAG_SUBAGENT_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.db import aggregate_conversation_by_user
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.prompt_projection import project_runtime_context_for_llm
from kazusa_ai_chatbot.time_boundary import (
    local_date_bounds_to_storage_utc_iso,
    parse_storage_utc_datetime,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

_EXTRACTOR_PROMPT = '''\
你是 `aggregate_conversation_by_user` 的参数抽取器。

# 能力边界
本代理只计算聊天历史里的事实聚合。它可以取证并做简单计数，
但不能输出观点、关系判断、动机判断或角色人格解释。

# 支持的聚合
- 按用户统计消息数，可按字面关键词、已知用户、频道和时间窗口过滤。

# 典型用途
- 最近谁发言最多？
- 已解析用户发了多少条消息？
- 谁最常提到某个字面词？
- 哪些用户聊过某个已知精确关键词？

# 生成步骤
1. 读取 `task`，判断它是否需要消息数量聚合。
2. 读取 `context` 中的已知用户、频道和当前时间提示。
3. 只有任务要求统计某个精确提及时，才填写字面 `keyword`。
4. 选择与任务匹配的最窄受支持 `time_window`。
5. 将用户明确要求的数量保留在 `limit` 中，下游会再做上限校验。

# 输入格式
{
  "task": "外层 RAG supervisor 给出的槽位描述",
  "context": "已知事实和运行时提示"
}

# 输出格式
只返回有效 JSON：
{
  "aggregate": "message_count_by_user",
  "keyword": "字面字符串或空字符串",
  "time_window": "recent | today | yesterday | all",
  "limit": 10
}
'''

_llm_interface = LLInterface()
_extractor_llm = LLInterface()
_extractor_llm_config = LLMCallConfig(
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
_TIME_WINDOWS = {"recent", "today", "yesterday", "all"}


def _normalize_limit(raw_limit: object) -> int:
    """Normalize an LLM-provided limit into a bounded positive integer.

    Args:
        raw_limit: Limit value from parsed LLM JSON.

    Returns:
        Integer limit clamped to the aggregate result range.
    """
    if isinstance(raw_limit, int) and not isinstance(raw_limit, bool):
        return_value = max(1, min(raw_limit, 50))
        return return_value
    return 10


def _normalize_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Normalize extractor output into safe aggregate arguments.

    Args:
        raw_args: Parsed JSON object from the extractor LLM.

    Returns:
        Dict containing aggregate, keyword, time_window, and limit.
    """
    time_window = text_or_empty(raw_args.get("time_window")) or "recent"
    if time_window not in _TIME_WINDOWS:
        time_window = "recent"

    return_value = {
        "aggregate": "message_count_by_user",
        "keyword": text_or_empty(raw_args.get("keyword")),
        "time_window": time_window,
        "limit": _normalize_limit(raw_args.get("limit", 10)),
    }
    return return_value


async def _extract_aggregate_args(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Extract constrained aggregate parameters from a slot description.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Runtime hints and known facts collected by supervisor2.

    Returns:
        Normalized aggregate arguments.
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
    response = await _extractor_llm.ainvoke([system_prompt, human_message], config=_extractor_llm_config)
    result = parse_llm_json_output(response.content)
    if not isinstance(result, dict):
        return_value = _normalize_args({})
        return return_value
    return_value = _normalize_args(result)
    return return_value


def _parse_current_timestamp_utc(context: dict[str, Any]) -> datetime.datetime:
    """Parse the current storage UTC timestamp from runtime context.

    Args:
        context: Runtime context passed to the agent.

    Returns:
        Timezone-aware UTC datetime.
    """
    current_timestamp_utc = text_or_empty(context.get("current_timestamp_utc"))
    return_value = parse_storage_utc_datetime(current_timestamp_utc)
    return return_value


def _time_bounds(
    time_window: str, context: dict[str, Any]
) -> tuple[str | None, str | None]:
    """Convert a coarse time-window label into ISO timestamp bounds.

    Args:
        time_window: One of recent, today, yesterday, or all.
        context: Runtime context containing current_timestamp_utc when available.

    Returns:
        Tuple of from_timestamp and to_timestamp strings, either of which may be None.
    """
    if time_window == "all":
        return_value = None, None
        return return_value

    now_utc = _parse_current_timestamp_utc(context)
    if time_window == "recent":
        return_value = (
            (now_utc - datetime.timedelta(days=7)).isoformat(),
            now_utc.isoformat(),
        )
        return return_value

    local_time_context = context.get("local_time_context")
    if isinstance(local_time_context, dict):
        local_date_str = str(
            local_time_context.get("current_local_datetime", "")
        ).split(" ")[0]
    else:
        local_date_str = ""

    if time_window in ("today", "yesterday") and not local_date_str:
        return_value = None, None
        return return_value

    if time_window == "today" and local_date_str:
        today_start, _ = local_date_bounds_to_storage_utc_iso(local_date_str)
        return_value = today_start, now_utc.isoformat()
        return return_value
    if time_window == "yesterday" and local_date_str:
        try:
            local_date = datetime.date.fromisoformat(local_date_str)
        except ValueError:
            pass
        else:
            yesterday_str = (local_date - datetime.timedelta(days=1)).isoformat()
            yesterday_start, yesterday_end = local_date_bounds_to_storage_utc_iso(
                yesterday_str
            )
            return_value = yesterday_start, yesterday_end
            return return_value
    return_value = None, None
    return return_value


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
    return_value = int(match.group(1))
    return return_value


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
            return_value = str(value)
            return return_value
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
    return_value = user_id or None
    return return_value


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

        return_value = self.with_cache_status(
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
        return return_value


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
