"""Tests proving legacy memory writers route through memory evolution APIs."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.db import memory as db_memory_module
from kazusa_ai_chatbot.db.schemas import build_memory_doc
from kazusa_ai_chatbot.memory_evolution.models import MemoryAuthority


@pytest.mark.asyncio
async def test_save_memory_wraps_insert_memory_unit_with_evolving_ids() -> None:
    """The legacy helper no longer writes directly to ``db.memory``."""
    insert_memory_unit = AsyncMock(return_value={})
    with patch(
        "kazusa_ai_chatbot.db.memory.insert_memory_unit",
        insert_memory_unit,
    ):
        doc = build_memory_doc(
            memory_name="Manual memory",
            content="Manual content",
            source_global_user_id="",
            memory_type="fact",
            source_kind="seeded_manual",
            confidence_note="manual",
        )

        await db_memory_module.save_memory(
            doc,
            "2026-05-05T00:00:00+00:00",
        )

    document = insert_memory_unit.await_args.kwargs["document"]
    assert document["memory_unit_id"].startswith("manual_")
    assert document["lineage_id"] == document["memory_unit_id"]
    assert document["version"] == 1
    assert document["authority"] == MemoryAuthority.MANUAL
    assert document["timestamp"] == "2026-05-05T00:00:00+00:00"
    assert "embedding" not in document


@pytest.mark.asyncio
async def test_save_memory_marks_extracted_sources_as_promoted() -> None:
    """Legacy conversation-extracted writes are not tagged as manual memory."""
    insert_memory_unit = AsyncMock(return_value={})
    with patch(
        "kazusa_ai_chatbot.db.memory.insert_memory_unit",
        insert_memory_unit,
    ):
        doc = build_memory_doc(
            memory_name="Extracted memory",
            content="Extracted content",
            source_global_user_id="user-1",
            memory_type="fact",
            source_kind="conversation_extracted",
            confidence_note="conversation",
        )

        await db_memory_module.save_memory(
            doc,
            "2026-05-05T00:00:00+00:00",
        )

    document = insert_memory_unit.await_args.kwargs["document"]
    assert document["authority"] == MemoryAuthority.REFLECTION_PROMOTED
