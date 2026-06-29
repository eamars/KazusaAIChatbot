"""Direct URL-read source subagent for web_agent3."""

from __future__ import annotations

from typing import Any

import kazusa_ai_chatbot.rag.web_agent3.searxng_tools as searxng_tools
from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision

SOURCE = "web_read"
SUPPORTED_ACTIONS = ("read",)
DESCRIPTION = '''普通网页 URL 读取。
生成 query 时：
- read: query 应尽量只包含一个明确 HTTP(S) URL；如果来自搜索结果，读取最相关的链接。
- 搜索结果 snippet 只是线索，正文读取才是强证据。
- 如果读取为空、不可用、跳转无关或内容不含目标事实，最终证据包必须说明该路径未确认。
'''


async def execute(decision: _RouterDecision) -> Any:
    """Execute one direct URL-read decision.

    Args:
        decision: Router decision with the target URL in `query`.

    Returns:
        Raw URL-reader output.
    """
    if decision.action != "read":
        result = {
            "status": "error",
            "source": decision.source,
            "action": decision.action,
            "query": decision.query,
            "message": "web_read supports read actions only.",
        }
        return result

    result = await searxng_tools.web_url_read.ainvoke({"url": decision.query})
    return result
