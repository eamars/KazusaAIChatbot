"""Generic web source subagent backed by the existing SearXNG facility."""

from __future__ import annotations

from typing import Any

import kazusa_ai_chatbot.rag.web_agent3.searxng_tools as searxng_tools
from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision

SOURCE = "generic"
DESCRIPTION = '''普通网页搜索和 URL 读取。
生成 query 时：
- search: 生成普通网页搜索词。无结果时优先去掉过窄日期、年份和堆叠约束，再改用近义词、英文翻译、精确匹配、site: 限制或排除词。
- read: query 应尽量只包含一个明确 HTTP(S) URL；如果来自搜索结果，读取最相关的链接。
- 搜索结果 snippet 只是线索，正文读取才是强证据。
- 涉及最新、最近或当前信息时，不要因为 reference_time 自动把当前日期、年份或未来日期写入 search query；只有用户明确要求某一天、某月或某个日期范围时才加入日期。
- 来源核实时，优先搜索任务领域里权威、稳定、可复核的来源路径，不要先使用过窄的时间过滤。
'''


async def execute(decision: _RouterDecision) -> Any:
    """Execute a generic web search/read/stop decision.

    Args:
        decision: Router decision with the query passed through unchanged.

    Returns:
        Raw SearXNG tool output or a stop observation.
    """
    if decision.action == "search":
        result = await searxng_tools.web_search.ainvoke({"query": decision.query})
        return result

    if decision.action == "read":
        result = await searxng_tools.web_url_read.ainvoke({"url": decision.query})
        return result

    result = {
        "status": "stopped",
        "message": "Router stopped without another web action.",
    }
    return result
