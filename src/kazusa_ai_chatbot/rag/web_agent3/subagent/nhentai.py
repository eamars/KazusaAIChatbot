"""nHentai source subagent placeholder."""

from __future__ import annotations

import logging

from kazusa_ai_chatbot.rag.web_agent3.contracts import (
    _DUMMY_PROVIDER_FIXME,
    _RouterDecision,
)

logger = logging.getLogger(__name__)

SOURCE = "nhentai"
DESCRIPTION = '''nHentai 图库编号（6或7位神秘代码）、页面或链接来源。
生成 query 时：
- search: 保留用户给出的数字编号、页面 URL 或 API 文档目标。
- read: 保留原始数字编号、页面 URL 或用户给出的目标字符串。
'''


async def execute(decision: _RouterDecision) -> dict[str, str]:
    """Return the current no-data observation for nHentai source work."""
    logger.debug(
        f"web_agent3 nhentai placeholder has no data: {_DUMMY_PROVIDER_FIXME}"
    )
    result = {
        "status": "no_search_data",
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "message": "Source subagent placeholder has no search data.",
    }
    return result
