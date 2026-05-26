"""Bilibili source subagent placeholder."""

from __future__ import annotations

import logging

from kazusa_ai_chatbot.rag.web_agent3.contracts import (
    _DUMMY_PROVIDER_FIXME,
    _RouterDecision,
)

logger = logging.getLogger(__name__)

SOURCE = "bilibili"
DESCRIPTION = '''Bilibili 视频、页面、编号或链接来源。
生成 query 时：
- search: 保留用户给出的标题、UP 主、BV/AV 号、关键词或 Bilibili URL。
- read: 保留原始 BV/AV 号、页面 URL 或用户给出的目标字符串。
'''


async def execute(decision: _RouterDecision) -> dict[str, str]:
    """Return the current no-data observation for Bilibili source work."""
    logger.debug(
        f"web_agent3 bilibili placeholder has no data: {_DUMMY_PROVIDER_FIXME}"
    )
    result = {
        "status": "no_search_data",
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "message": "Source subagent placeholder has no search data.",
    }
    return result
