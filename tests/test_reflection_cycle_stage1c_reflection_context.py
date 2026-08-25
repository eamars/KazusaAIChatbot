"""Tests for prompt-facing promoted reflection context."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.reflection_cycle import context as context_module


def _certified_memory_row(
    *,
    memory_type: str,
    memory_unit_id: str,
) -> dict[str, object]:
    return {
        "memory_name": "Certified memory",
        "content": "A bounded certified semantic memory.",
        "memory_unit_id": memory_unit_id,
        "memory_type": memory_type,
        "source_kind": "reflection_inferred",
        "source_global_user_id": "",
        "authority": "reflection_promoted",
        "status": "active",
        "privacy_review": {
            "global_applicability": "global",
            "target_specific_meaning_removed": True,
            "affects_identity_or_boundaries": False,
            "private_detail_risk": "low",
            "user_details_removed": True,
            "boundary_assessment": "The meaning is deidentified and global.",
            "reviewer": "automated_llm",
        },
        "updated_at": "2026-05-04T10:00:00+00:00",
        "embedding": [0.1, 0.2],
        "evidence_refs": [{"source": "reflection_cycle"}],
        "source_message_refs": ["private-source-ref"],
        "raw_reflection_content": "raw reflection output",
    }


@pytest.mark.asyncio
async def test_reflection_context_returns_empty_when_no_promoted_lanes(
    monkeypatch,
) -> None:
    """Empty promoted lanes should produce no prompt-facing context."""

    find_active = AsyncMock(return_value=[])
    monkeypatch.setattr(context_module, "find_active_memory_units", find_active)

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
            fact = _certified_memory_row(
                memory_type="fact",
                memory_unit_id="fact-unit",
            )
            fact.update({
                "memory_name": "World rule",
                "content": "Shared lore content.",
                "updated_at": "2026-05-04T10:00:00+00:00",
                "confidence_note": "reflection",
            })
            return [
                (
                    -1.0,
                    fact,
                )
            ]
        guidance = _certified_memory_row(
            memory_type="defense_rule",
            memory_unit_id="guidance-unit",
        )
        guidance.update({
            "memory_name": "Response habit",
            "content": "Stay concrete in future responses.",
            "updated_at": "2026-05-05T10:00:00+00:00",
            "confidence_note": "reflection",
        })
        return [
            (
                -1.0,
                guidance,
            )
        ]

    monkeypatch.setattr(
        context_module,
        "find_active_memory_units",
        _find_active_memory_units,
    )

    result = await context_module.build_promoted_reflection_context()

    assert result["promoted_lore"][0]["memory_type"] == "fact"
    assert result["promoted_self_guidance"][0]["memory_type"] == "defense_rule"
    assert result["source_dates"] == ["2026-05-04", "2026-05-05"]
    assert "raw_hourly" not in str(result)
    assert "source_message_refs" not in str(result)


@pytest.mark.asyncio
async def test_reflection_context_preserves_typed_authority_certificate(
    monkeypatch,
) -> None:
    """Preserve certified metadata while excluding source-only fields."""

    async def _find_active_memory_units(*, query, limit):
        memory_type = query["memory_type"]
        return [
            (
                -1.0,
                _certified_memory_row(
                    memory_type=memory_type,
                    memory_unit_id=f"{memory_type}-unit",
                ),
            )
        ]

    monkeypatch.setattr(
        context_module,
        "find_active_memory_units",
        _find_active_memory_units,
    )

    result = await context_module.build_promoted_reflection_context()

    for lane_name in ("promoted_lore", "promoted_self_guidance"):
        row = result[lane_name][0]
        assert row["memory_unit_id"]
        assert row["source_kind"] == "reflection_inferred"
        assert row["source_global_user_id"] == ""
        assert row["authority"] == "reflection_promoted"
        assert row["status"] == "active"
        assert row["scope_type"] == "global"
        assert set(row["privacy_review"]) == {
            "global_applicability",
            "target_specific_meaning_removed",
            "affects_identity_or_boundaries",
            "private_detail_risk",
            "user_details_removed",
            "boundary_assessment",
            "reviewer",
        }
        assert "embedding" not in row
        assert "evidence_refs" not in row
        assert "source_message_refs" not in row
        assert "raw_reflection_content" not in row
