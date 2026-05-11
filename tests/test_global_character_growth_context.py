"""Runtime context tests for global character growth."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.global_character_growth import context as growth_context
from kazusa_ai_chatbot.reflection_cycle import context as reflection_context


@pytest.mark.asyncio
async def test_global_growth_context_returns_empty_when_no_promoted_traits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unpromoted or missing traits should produce no prompt context."""

    monkeypatch.setattr(
        growth_context.growth_store,
        "list_prompt_visible_growth_traits",
        AsyncMock(return_value=[]),
    )

    result = await growth_context.build_global_character_growth_context(limit=3)

    assert result == {}


@pytest.mark.asyncio
async def test_global_growth_context_projects_active_promoted_traits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt-visible projection should be compact and source-detail-free."""

    monkeypatch.setattr(
        growth_context.growth_store,
        "list_prompt_visible_growth_traits",
        AsyncMock(return_value=[
            _trait("clarity", "保持关心可见，但不要催促同意。"),
            _trait("guarded_care", "退开前先提供修复机会。"),
        ]),
    )

    result = await growth_context.build_global_character_growth_context(limit=3)

    assert list(result) == ["promoted_global_growth", "retrieval_notes"]
    assert result["promoted_global_growth"] == [
        {
            "growth_axis": "clarity",
            "guidance": "保持关心可见，但不要催促同意。",
            "maturity": "promoted",
            "updated_at": "2026-05-05",
        },
        {
            "growth_axis": "guarded_care",
            "guidance": "退开前先提供修复机会。",
            "maturity": "promoted",
            "updated_at": "2026-05-05",
        },
    ]
    assert "source_memory_unit_ids" not in str(result)


@pytest.mark.asyncio
async def test_reflection_context_merges_promoted_global_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing promoted reflection context should carry global growth too."""

    async def _project_lane(*, memory_type: str, limit: int) -> list[dict]:
        del memory_type, limit
        return []

    monkeypatch.setattr(reflection_context, "_project_lane", _project_lane)
    monkeypatch.setattr(
        reflection_context,
        "build_global_character_growth_context",
        AsyncMock(return_value={
            "promoted_global_growth": [{
                "growth_axis": "clarity",
                "guidance": "保持关心可见，但不要催促同意。",
                "maturity": "promoted",
                "updated_at": "2026-05-05T10:00:00+00:00",
            }],
            "retrieval_notes": [
                "Only active promoted global character-growth traits are included.",
            ],
        }),
    )

    result = await reflection_context.build_promoted_reflection_context()

    assert result["promoted_global_growth"][0]["growth_axis"] == "clarity"
    assert any("global character-growth" in note for note in result["retrieval_notes"])


def _trait(growth_axis: str, guidance: str) -> dict:
    """Build a prompt-visible trait fixture."""

    return {
        "trait_id": f"trait-{growth_axis}",
        "status": "active",
        "growth_axis": growth_axis,
        "guidance": guidance,
        "maturity_band": "promoted",
        "strength": 0.8,
        "updated_at": "2026-05-05T10:00:00+00:00",
        "source_memory_unit_ids": ["memory-1"],
    }
