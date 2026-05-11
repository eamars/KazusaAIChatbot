"""Tests for prompt-facing promoted reflection context."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.reflection_cycle import context as context_module


@pytest.mark.asyncio
async def test_reflection_context_returns_empty_when_no_promoted_lanes(
    monkeypatch,
) -> None:
    """Empty promoted lanes should produce no prompt-facing context."""

    find_active = AsyncMock(return_value=[])
    monkeypatch.setattr(context_module, "find_active_memory_units", find_active)
    monkeypatch.setattr(
        context_module,
        "build_global_character_growth_context",
        AsyncMock(return_value={}),
    )

    result = await context_module.build_promoted_reflection_context()

    assert result == {}
    assert find_active.await_count == 2
    for call in find_active.await_args_list:
        assert call.kwargs["query"]["source_kind"] == "reflection_inferred"
        assert call.kwargs["query"]["source_global_user_id"] == ""
        assert call.kwargs["limit"] == 3


@pytest.mark.asyncio
async def test_reflection_context_projects_only_promoted_memory_lanes(
    monkeypatch,
) -> None:
    """Enabled context should query the two approved reflection memory lanes."""

    async def _find_active_memory_units(*, query, limit):
        assert query["source_kind"] == "reflection_inferred"
        assert query["source_global_user_id"] == ""
        assert limit == 3
        if query["memory_type"] == "fact":
            return [
                (
                    -1.0,
                    {
                        "memory_name": "World rule",
                        "content": "Shared lore content.",
                        "memory_type": "fact",
                        "updated_at": "2026-05-04T10:00:00+00:00",
                        "confidence_note": "reflection",
                    },
                )
            ]
        return [
            (
                -1.0,
                {
                    "memory_name": "Response habit",
                    "content": "Stay concrete in future responses.",
                    "memory_type": "defense_rule",
                    "updated_at": "2026-05-05T10:00:00+00:00",
                    "confidence_note": "reflection",
                },
            )
        ]

    monkeypatch.setattr(
        context_module,
        "find_active_memory_units",
        _find_active_memory_units,
    )
    monkeypatch.setattr(
        context_module,
        "build_global_character_growth_context",
        AsyncMock(return_value={}),
    )

    result = await context_module.build_promoted_reflection_context()

    assert result["promoted_lore"][0]["memory_type"] == "fact"
    assert result["promoted_self_guidance"][0]["memory_type"] == "defense_rule"
    assert result["source_dates"] == ["2026-05-04", "2026-05-05"]
    assert "raw_hourly" not in str(result)
    assert "source_message_refs" not in str(result)
