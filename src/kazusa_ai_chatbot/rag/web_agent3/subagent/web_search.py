"""Search source subagent for web_agent3."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.config import SEARXNG_URL
import kazusa_ai_chatbot.rag.web_agent3.searxng_tools as searxng_tools
from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision

SOURCE = "web_search"
SUPPORTED_ACTIONS = ("search",)
DESCRIPTION = '''普通网页搜索。
生成 query 时：
- search: 生成普通网页搜索词。无结果时优先去掉过窄日期、年份和堆叠约束，再改用近义词、英文翻译、精确匹配、site: 限制或排除词。
- 搜索结果 snippet 只是线索，正文读取才是强证据。
- 涉及最新、最近或当前信息时，不要因为 reference_time 自动把当前日期、年份或未来日期写入 search query；只有用户明确要求某一天、某月或某个日期范围时才加入日期。
- 来源核实时，优先搜索任务领域里权威、稳定、可复核的来源路径，不要先使用过窄的时间过滤。
'''


def is_enabled() -> bool:
    """Return whether the search source has a configured backend."""
    enabled = bool(SEARXNG_URL)
    return enabled


async def execute(decision: _RouterDecision) -> Any:
    """Execute one web-search decision.

    Args:
        decision: Router decision with the search text in `query`.

    Returns:
        Raw search tool output.
    """
    if decision.action != "search":
        result = {
            "status": "error",
            "source": decision.source,
            "action": decision.action,
            "query": decision.query,
            "message": "web_search supports search actions only.",
        }
        return result

    result = await searxng_tools.web_search.ainvoke({"query": decision.query})
    return result
