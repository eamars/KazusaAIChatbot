"""History-evidence recall collector."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.db import get_conversation_history
from kazusa_ai_chatbot.rag.recall.contracts import _candidate, _entry_text

class HistoryEvidenceCollector:
    """Collect bounded transcript proof for exact history or conflicts."""

    async def collect(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """Read recent conversation history and convert rows to candidates."""

        rows = await get_conversation_history(
            platform=context["platform"],
            platform_channel_id=context["platform_channel_id"],
            global_user_id=context["global_user_id"],
            limit=20,
        )
        candidates: list[dict[str, str]] = []
        for row in rows[:5]:
            claim = _entry_text(row)
            if not claim:
                continue
            candidates.append(
                _candidate(
                    source="conversation_history",
                    claim=claim,
                    temporal_scope="historical_proof",
                    lifecycle_status="historical",
                    evidence_time=row.get("timestamp", ""),
                    authority="primary_for_exact_wording",
                )
            )
        return candidates
