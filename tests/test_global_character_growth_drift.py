"""Stable-drift tests for global character growth traits."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.global_character_growth import drift


def test_three_confirming_days_remain_emerging_not_prompt_visible() -> None:
    """The POC path should remain slow after three confirming days."""

    strength = drift.calculate_stable_strength(
        previous_strength=0.0,
        evidence_strength=0.94,
        confirming_day_count=3,
    )

    assert strength == pytest.approx(0.363, abs=0.001)
    assert drift.maturity_band_for_strength(strength) == "emerging"


def test_maturity_band_thresholds() -> None:
    """Only promoted strength reaches prompt visibility."""

    assert drift.maturity_band_for_strength(0.249) == "observed"
    assert drift.maturity_band_for_strength(0.25) == "emerging"
    assert drift.maturity_band_for_strength(0.50) == "stabilizing"
    assert drift.maturity_band_for_strength(0.75) == "promoted"
    assert drift.is_prompt_visible({
        "status": "active",
        "maturity_band": "promoted",
    }) is True
    assert drift.is_prompt_visible({
        "status": "active",
        "maturity_band": "stabilizing",
    }) is False


def test_evidence_strength_is_lower_for_non_stable_candidates() -> None:
    """Weaker candidates can be tracked without promoting quickly."""

    stable = drift.evidence_strength_for_candidate({
        "support_level": "stable",
        "confidence": "high",
    })
    emerging = drift.evidence_strength_for_candidate({
        "support_level": "emerging",
        "confidence": "medium",
    })
    insufficient = drift.evidence_strength_for_candidate({
        "support_level": "insufficient",
        "confidence": "low",
    })

    assert stable == pytest.approx(0.94)
    assert 0 < insufficient < emerging < stable


def test_plan_new_trait_update_records_source_history() -> None:
    """Trait updates should keep audit source ids out of prompt projection."""

    updates = drift.plan_trait_updates(
        existing_trait_rows=[],
        accepted_candidates=[_candidate()],
        now_iso="2026-05-11T00:00:00+00:00",
    )

    assert len(updates) == 1
    update = updates[0]
    assert update["action"] == "insert"
    trait = update["trait"]
    assert trait["status"] == "active"
    assert trait["maturity_band"] == "emerging"
    assert trait["strength"] == pytest.approx(0.363, abs=0.001)
    assert trait["evidence_count"] == 2
    assert trait["source_memory_unit_ids"] == ["memory-1", "memory-2"]
    assert trait["source_reflection_run_ids"] == ["run-1", "run-2"]


def test_plan_existing_trait_update_uses_new_dates_only() -> None:
    """Repeated same-day evidence should not inflate strength twice."""

    existing = [{
        "trait_id": "trait-existing",
        "lineage_id": "lineage-existing",
        "status": "active",
        "growth_axis": "clarity",
        "trait_name": "Care with boundaries",
        "guidance": "Keep care visible while preserving consent.",
        "strength": 0.25,
        "maturity_band": "emerging",
        "supporting_dates": ["2026-05-01"],
        "source_memory_unit_ids": ["memory-1"],
        "source_reflection_run_ids": ["run-1"],
        "source_candidate_ids": ["candidate-old"],
        "evidence_count": 1,
        "version": 1,
        "supersedes_trait_ids": [],
        "merged_from_trait_ids": [],
        "first_observed_date": "2026-05-01",
        "last_observed_date": "2026-05-01",
        "created_at": "2026-05-01T00:00:00+00:00",
        "updated_at": "2026-05-01T00:00:00+00:00",
    }]
    candidate = _candidate(supporting_dates=["2026-05-01", "2026-05-03"])

    updates = drift.plan_trait_updates(
        existing_trait_rows=existing,
        accepted_candidates=[candidate],
        now_iso="2026-05-11T00:00:00+00:00",
    )

    trait = updates[0]["trait"]
    assert updates[0]["action"] == "update"
    assert trait["trait_id"] == "trait-existing"
    assert trait["supporting_dates"] == ["2026-05-01", "2026-05-03"]
    assert trait["version"] == 2
    assert trait["strength"] > 0.25
    assert trait["strength"] < 0.40


def _candidate(
    *,
    supporting_dates: list[str] | None = None,
) -> dict:
    """Build one accepted candidate fixture."""

    return {
        "candidate_id": "gcc_candidate",
        "growth_axis": "clarity",
        "trait_name": "Care with boundaries",
        "guidance": "Keep care visible while preserving consent.",
        "source_card_ids": ["card-1", "card-2"],
        "supporting_dates": supporting_dates or [
            "2026-05-01",
            "2026-05-02",
            "2026-05-03",
        ],
        "source_memory_unit_ids": ["memory-1", "memory-2"],
        "source_reflection_run_ids": ["run-1", "run-2"],
        "support_level": "stable",
        "confidence": "high",
        "evidence_strength": 0.94,
    }
