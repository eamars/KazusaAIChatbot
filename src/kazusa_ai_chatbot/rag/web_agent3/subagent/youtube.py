"""YouTube source subagent with temporary generic-web execution."""

from __future__ import annotations

import logging
from typing import Any

import kazusa_ai_chatbot.rag.web_agent3.subagent.generic as generic_subagent
from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision

logger = logging.getLogger(__name__)

SOURCE = "youtube"
DESCRIPTION = '''YouTube 视频、频道、页面或链接来源。
生成 query 时：
- search: 保留用户给出的视频标题、频道、关键词、video id 或 YouTube URL。
- read: 保留原始 YouTube URL、video id 或用户给出的目标字符串。
'''
_GENERIC_WEB_FALLBACK_FIXME = (
    "FIXME(web_agent3): replace temporary generic web fallback with a "
    "YouTube provider implementation inside this subagent."
)


async def execute(decision: _RouterDecision) -> Any:
    """Execute YouTube work through generic web search/read until API support."""
    logger.debug(
        f"web_agent3 youtube uses generic web fallback: "
        f"{_GENERIC_WEB_FALLBACK_FIXME}"
    )
    generic_decision = _RouterDecision(
        action=decision.action,
        source="generic",
        query=decision.query,
    )
    result = await generic_subagent.execute(generic_decision)
    return result
