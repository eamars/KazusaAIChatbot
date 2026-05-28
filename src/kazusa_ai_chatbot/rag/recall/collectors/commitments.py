"""Active-commitment recall collector."""

from __future__ import annotations

from kazusa_ai_chatbot.db import (
    UserMemoryUnitStatus,
    UserMemoryUnitType,
    query_user_memory_units,
)
from kazusa_ai_chatbot.rag.recall.contracts import _candidate, _event_claim
from kazusa_ai_chatbot.utils import text_or_empty

class ActiveCommitmentCollector:
    """Collect active durable commitments for the current user."""

    async def collect(self, global_user_id: str) -> list[dict[str, str]]:
        """Read active commitment memory units and convert them to candidates."""

        units = await query_user_memory_units(
            global_user_id,
            unit_types=[UserMemoryUnitType.ACTIVE_COMMITMENT],
            statuses=[UserMemoryUnitStatus.ACTIVE],
            limit=6,
        )
        candidates: list[dict[str, str]] = []
        for unit in units:
            if unit.get("status") != UserMemoryUnitStatus.ACTIVE:
                continue
            claim = text_or_empty(unit.get("fact"))
            if not claim:
                continue
            evidence_time = (
                text_or_empty(unit.get("updated_at"))
                or text_or_empty(unit.get("last_seen_at"))
            )
            candidates.append(
                _candidate(
                    source="user_memory_units",
                    claim=claim,
                    temporal_scope="durable_ongoing",
                    lifecycle_status="active",
                    evidence_time=evidence_time,
                    authority="primary_for_durable_commitment",
                )
            )
        return candidates[:6]
