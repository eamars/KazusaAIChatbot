"""Bilibili source subagent with temporary generic-web execution."""

from __future__ import annotations

import logging
from typing import Any

import kazusa_ai_chatbot.rag.web_agent3.subagent.generic as generic_subagent
from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision

logger = logging.getLogger(__name__)

SOURCE = "bilibili"
DESCRIPTION = '''Bilibili 视频、页面、编号或链接来源。
生成 query 时：
- search: 保留用户给出的标题、UP 主、BV/AV 号、关键词或 Bilibili URL。
- read: 保留原始 BV/AV 号、页面 URL 或用户给出的目标字符串。
'''
_GENERIC_WEB_FALLBACK_FIXME = (
    "FIXME(web_agent3): replace temporary generic web fallback with a "
    "Bilibili provider implementation inside this subagent."
)


async def execute(decision: _RouterDecision) -> Any:
    """Execute Bilibili work through generic web search/read until API support."""
    logger.debug(
        f"web_agent3 bilibili uses generic web fallback: "
        f"{_GENERIC_WEB_FALLBACK_FIXME}"
    )
    generic_decision = _RouterDecision(
        action=decision.action,
        source="generic",
        query=decision.query,
    )
    result = await generic_subagent.execute(generic_decision)
    return result
