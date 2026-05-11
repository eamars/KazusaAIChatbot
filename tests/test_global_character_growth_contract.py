"""Contract tests for global character growth from reflection."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect

import pytest

import kazusa_ai_chatbot.global_character_growth as growth
from kazusa_ai_chatbot.config import GLOBAL_CHARACTER_GROWTH_PASS_ENABLED
from kazusa_ai_chatbot.global_character_growth import models, projection


def test_public_package_exports_only_approved_entrypoints() -> None:
    """The package facade should expose only the approved public API."""

    assert growth.__all__ == [
        "run_global_character_growth_pass",
        "build_global_character_growth_context",
    ]
    assert inspect.iscoroutinefunction(growth.run_global_character_growth_pass)
    assert inspect.iscoroutinefunction(growth.build_global_character_growth_context)


def test_global_growth_pass_flag_defaults_on() -> None:
    """Character growth is a default-on background pass."""

    assert GLOBAL_CHARACTER_GROWTH_PASS_ENABLED is True


def test_stable_drift_constants_are_provisional_defaults() -> None:
    """Stable drift constants should remain centralized and deliberate."""

    assert models.PREVIOUS_STRENGTH_WEIGHT == pytest.approx(0.85)
    assert models.EVIDENCE_STRENGTH_WEIGHT == pytest.approx(0.15)
    assert models.MAX_DAILY_STRENGTH_DELTA == pytest.approx(0.18)
    assert models.OBSERVED_STRENGTH_CEILING == pytest.approx(0.25)
    assert models.EMERGING_STRENGTH_CEILING == pytest.approx(0.50)
    assert models.STABILIZING_STRENGTH_CEILING == pytest.approx(0.75)


def test_candidate_prompt_payload_caps_inputs_and_projects_shape() -> None:
    """Prompt payloads should be compact and source-card based."""

    memory_rows = [_memory_doc(index) for index in range(90)]
    trait_rows = [_trait_doc(index, maturity_band="promoted") for index in range(16)]
    malformed_trait = _trait_doc(99, maturity_band="promoted")
    del malformed_trait["status"]
    trait_rows.append(malformed_trait)

    payload = projection.build_candidate_prompt_payload(
        memory_rows=memory_rows,
        current_trait_rows=trait_rows,
        limit=80,
    )

    assert payload["evaluation_mode"] == "global_character_growth_v1"
    assert payload["prompt_version"] == "global_character_growth_candidate_v1"
    assert len(payload["memory_cards"]) == 80
    assert len(payload["current_global_growth_traits"]) == 12
    projected_ids = {
        trait["trait_id"]
        for trait in payload["current_global_growth_traits"]
    }
    assert "trait-99" not in projected_ids
    assert len(payload["allowed_growth_axes"]) == 8
    assert payload["candidate_limits"] == {
        "max_candidates": 4,
        "max_source_cards_per_candidate": 8,
    }
    first_card = payload["memory_cards"][0]
    assert len(first_card["content"]) <= 420
    assert len(first_card["confidence_note"]) <= 120
    assert len(first_card["character_local_dates"]) <= 8
    assert len(first_card["source_reflection_run_ids"]) <= 8


def test_input_quality_diagnostics_are_human_auditable() -> None:
    """Run records need enough input diagnostics to explain sparse output."""

    missing_status = _memory_doc(3, updated_at="2026-05-04T10:00:00+00:00")
    del missing_status["status"]
    rows = [
        _memory_doc(1, updated_at="2026-05-01T10:00:00+00:00"),
        _memory_doc(2, updated_at="2026-05-03T10:00:00+00:00"),
        missing_status,
        {
            "memory_unit_id": "not-reflection",
            "memory_name": "Wrong source",
            "content": "Not eligible.",
            "memory_type": "fact",
            "source_kind": "manual",
            "authority": "manual",
            "source_global_user_id": "",
            "status": "active",
            "updated_at": "2026-05-03T10:00:00+00:00",
        },
    ]

    cards, diagnostics = projection.build_memory_cards(rows, limit=80)

    assert len(cards) == 2
    assert diagnostics["raw_memory_rows"] == 4
    assert diagnostics["eligible_memory_cards"] == 2
    assert diagnostics["unique_source_dates"] == 2
    assert diagnostics["source_date_span_days"] == 2
    assert diagnostics["dropped_rows"] == {
        "not_active": 1,
        "not_reflection_promoted": 1,
    }


def test_input_quality_diagnostics_ignore_malformed_source_dates() -> None:
    """Bad upstream timestamps should not crash the growth projection."""

    rows = [_memory_doc(1, updated_at="not-a-date")]

    cards, diagnostics = projection.build_memory_cards(rows, limit=80)

    assert len(cards) == 1
    assert cards[0]["character_local_dates"] == []
    assert diagnostics["unique_source_dates"] == 0
    assert diagnostics["source_date_span_days"] == 0


def test_shadow_projection_is_log_only_and_bounded() -> None:
    """Emerging traits may be logged for review but not prompted."""

    planned_updates = [
        {
            "trait_id": f"trait-{index}",
            "growth_axis": "clarity",
            "guidance": f"Guidance {index}",
            "strength": 0.25 + index / 100,
            "maturity_band": "emerging",
            "status": "active",
        }
        for index in range(8)
    ]

    shadow = projection.build_shadow_projection(planned_updates)

    assert len(shadow) == 5
    assert shadow[0] == {
        "growth_axis": "clarity",
        "guidance": "Guidance 0",
        "maturity": "emerging",
        "prompt_visible_now": False,
        "review_note": "Not prompt-visible until maturity is promoted.",
    }
    assert "strength" not in shadow[0]
    assert "trait_id" not in shadow[0]


def test_runtime_context_projects_promoted_active_traits_only() -> None:
    """Prompt context must exclude source ids, strengths, and unpromoted rows."""

    rows = [
        _trait_doc(1, maturity_band="promoted", strength=0.8),
        _trait_doc(2, maturity_band="stabilizing", strength=0.6),
        _trait_doc(3, maturity_band="promoted", strength=0.9, status="superseded"),
        _trait_doc(4, maturity_band="promoted", strength=0.95),
    ]

    context = projection.project_runtime_context(rows, limit=3)

    assert list(context) == ["promoted_global_growth", "retrieval_notes"]
    assert len(context["promoted_global_growth"]) == 2
    rendered = str(context)
    assert "strength" not in rendered
    assert "source_memory_unit_ids" not in rendered
    assert "trait_id" not in rendered


@pytest.mark.asyncio
async def test_dry_run_with_trait_writes_is_rejected() -> None:
    """Dry-run mode must not accept write enablement."""

    with pytest.raises(ValueError, match="enable_trait_writes"):
        await growth.run_global_character_growth_pass(
            character_local_date="2026-05-10",
            dry_run=True,
            enable_trait_writes=True,
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )


def _memory_doc(index: int, *, updated_at: str = "2026-05-01T10:00:00+00:00") -> dict:
    """Build a reflection-promoted memory row fixture."""

    return {
        "memory_unit_id": f"memory-{index}",
        "memory_name": f"Memory {index}",
        "content": "Repeated global communication pattern. " * 40,
        "memory_type": "defense_rule",
        "source_kind": "reflection_inferred",
        "authority": "reflection_promoted",
        "source_global_user_id": "",
        "status": "active",
        "evidence_refs": [
            {
                "reflection_run_id": f"run-{index}-{run_index}",
                "captured_at": updated_at,
            }
            for run_index in range(10)
        ],
        "confidence_note": "stable support " * 20,
        "updated_at": updated_at,
    }


def _trait_doc(
    index: int,
    *,
    maturity_band: str,
    strength: float = 0.8,
    status: str = "active",
) -> dict:
    """Build a global-growth trait row fixture."""

    return {
        "trait_id": f"trait-{index}",
        "lineage_id": f"lineage-{index}",
        "status": status,
        "growth_axis": "clarity",
        "trait_name": f"Trait {index}",
        "guidance": f"Keep guidance {index} compact and general.",
        "strength": strength,
        "maturity_band": maturity_band,
        "updated_at": "2026-05-05T10:00:00+00:00",
        "source_memory_unit_ids": [f"memory-{index}"],
    }
