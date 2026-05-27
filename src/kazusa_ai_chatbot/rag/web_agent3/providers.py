"""Source decision executor for web_agent3."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.rag.web_agent3 import subagent as _source_subagent_package
from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision


async def _execute_source_decision(decision: _RouterDecision) -> Any:
    """Dispatch one router decision to the selected source subagent.

    Args:
        decision: Minimal router decision containing only action, source, and
            query.

    Returns:
        Raw observation from the selected source subagent.
    """
    source_subagent = _source_subagent_package._SUBAGENTS[decision.source]
    result = await source_subagent.execute(decision)
    return result
