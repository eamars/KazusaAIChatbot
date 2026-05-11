"""Stable drift planning for global character growth traits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from hashlib import sha256
from typing import Any

from kazusa_ai_chatbot.global_character_growth.models import (
    DUPLICATE_OVERLAP_THRESHOLD,
    EMERGING_EVIDENCE_STRENGTH,
    EMERGING_STRENGTH_CEILING,
    EVIDENCE_STRENGTH_WEIGHT,
    FULL_EVIDENCE_STRENGTH,
    LOW_EVIDENCE_STRENGTH,
    MAX_DAILY_STRENGTH_DELTA,
    OBSERVED_STRENGTH_CEILING,
    PREVIOUS_STRENGTH_WEIGHT,
    STABILIZING_STRENGTH_CEILING,
    AcceptedCandidate,
    MaturityBand,
    TraitUpdate,
)


def calculate_stable_strength(
    *,
    previous_strength: float,
    evidence_strength: float,
    confirming_day_count: int,
) -> float:
    """Apply the provisional EMA drift formula for confirming days."""

    strength = _clamp(previous_strength)
    evidence = _clamp(evidence_strength)
    for _ in range(max(confirming_day_count, 0)):
        raw_strength = (
            strength * PREVIOUS_STRENGTH_WEIGHT
            + evidence * EVIDENCE_STRENGTH_WEIGHT
        )
        applied_delta = min(raw_strength - strength, MAX_DAILY_STRENGTH_DELTA)
        strength = _clamp(strength + applied_delta)
    return strength


def maturity_band_for_strength(strength: float) -> MaturityBand:
    """Return the semantic maturity band for a numeric strength."""

    clamped = _clamp(strength)
    if clamped < OBSERVED_STRENGTH_CEILING:
        return_value: MaturityBand = "observed"
        return return_value
    if clamped < EMERGING_STRENGTH_CEILING:
        return_value = "emerging"
        return return_value
    if clamped < STABILIZING_STRENGTH_CEILING:
        return_value = "stabilizing"
        return return_value
    return_value = "promoted"
    return return_value


def evidence_strength_for_candidate(candidate: Mapping[str, Any]) -> float:
    """Map candidate support labels to provisional evidence strength."""

    support_level = str(candidate.get("support_level", "insufficient"))
    confidence = str(candidate.get("confidence", "low"))
    if support_level == "stable" and confidence == "high":
        return_value = FULL_EVIDENCE_STRENGTH
        return return_value
    if support_level in {"stable", "emerging"} and confidence in {"high", "medium"}:
        return_value = EMERGING_EVIDENCE_STRENGTH
        return return_value
    return_value = LOW_EVIDENCE_STRENGTH
    return return_value


def plan_trait_updates(
    *,
    existing_trait_rows: Sequence[Mapping[str, Any]],
    accepted_candidates: Sequence[AcceptedCandidate],
    now_iso: str,
) -> list[TraitUpdate]:
    """Plan insert/update rows for validated global-growth candidates."""

    updates: list[TraitUpdate] = []
    for candidate in accepted_candidates:
        matching_trait = _matching_trait(candidate, existing_trait_rows)
        if matching_trait is None:
            updates.append({
                "action": "insert",
                "trait": _new_trait(candidate, now_iso),
            })
            continue
        planned_update = _updated_trait(matching_trait, candidate, now_iso)
        if planned_update is not None:
            updates.append({
                "action": "update",
                "trait": planned_update,
            })
    return updates


def is_prompt_visible(trait: Mapping[str, Any]) -> bool:
    """Return whether a trait is eligible for runtime prompt projection."""

    return_value = (
        str(trait.get("status", "")) == "active"
        and str(trait.get("maturity_band", "")) == "promoted"
    )
    return return_value


def _new_trait(candidate: AcceptedCandidate, now_iso: str) -> dict:
    supporting_dates = sorted(set(candidate["supporting_dates"]))
    strength = calculate_stable_strength(
        previous_strength=0.0,
        evidence_strength=float(candidate["evidence_strength"]),
        confirming_day_count=len(supporting_dates),
    )
    trait_id = _trait_id(candidate)
    trait = {
        "_id": trait_id,
        "trait_id": trait_id,
        "lineage_id": trait_id,
        "status": "active",
        "growth_axis": candidate["growth_axis"],
        "trait_name": candidate["trait_name"],
        "guidance": candidate["guidance"],
        "strength": strength,
        "maturity_band": maturity_band_for_strength(strength),
        "first_observed_date": supporting_dates[0] if supporting_dates else "",
        "last_observed_date": supporting_dates[-1] if supporting_dates else "",
        "supporting_dates": supporting_dates,
        "source_memory_unit_ids": _unique_sorted(candidate["source_memory_unit_ids"]),
        "source_reflection_run_ids": _unique_sorted(
            candidate["source_reflection_run_ids"],
        ),
        "source_candidate_ids": [candidate["candidate_id"]],
        "evidence_count": len(_unique_sorted(candidate["source_memory_unit_ids"])),
        "version": 1,
        "supersedes_trait_ids": [],
        "merged_from_trait_ids": [],
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    return trait


def _updated_trait(
    existing: Mapping[str, Any],
    candidate: AcceptedCandidate,
    now_iso: str,
) -> dict | None:
    previous_dates = [str(value) for value in existing.get("supporting_dates", [])]
    candidate_dates = sorted(set(candidate["supporting_dates"]))
    new_dates = [
        source_date for source_date in candidate_dates
        if source_date not in previous_dates
    ]
    all_dates = sorted(set(previous_dates + candidate_dates))
    if not all_dates:
        return_value = None
        return return_value
    previous_strength = float(existing.get("strength", 0.0))
    evidence_strength = float(candidate["evidence_strength"])
    strength = calculate_stable_strength(
        previous_strength=previous_strength,
        evidence_strength=evidence_strength,
        confirming_day_count=len(new_dates),
    )
    trait = dict(existing)
    trait["strength"] = strength
    trait["maturity_band"] = maturity_band_for_strength(strength)
    trait["last_observed_date"] = all_dates[-1]
    trait["supporting_dates"] = all_dates
    trait["source_memory_unit_ids"] = _unique_sorted(
        list(existing.get("source_memory_unit_ids", []))
        + candidate["source_memory_unit_ids"],
    )
    trait["source_reflection_run_ids"] = _unique_sorted(
        list(existing.get("source_reflection_run_ids", []))
        + candidate["source_reflection_run_ids"],
    )
    trait["source_candidate_ids"] = _unique_sorted(
        list(existing.get("source_candidate_ids", []))
        + [candidate["candidate_id"]],
    )
    trait["evidence_count"] = len(trait["source_memory_unit_ids"])
    trait["version"] = int(existing.get("version", 1)) + 1
    trait["updated_at"] = now_iso
    return trait


def _matching_trait(
    candidate: AcceptedCandidate,
    existing_trait_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for trait in existing_trait_rows:
        if str(trait.get("status", "")) != "active":
            continue
        if str(trait.get("growth_axis", "")) != candidate["growth_axis"]:
            continue
        overlap = SequenceMatcher(
            None,
            _normalize_text(str(trait.get("guidance", ""))),
            _normalize_text(candidate["guidance"]),
        ).ratio()
        if overlap >= DUPLICATE_OVERLAP_THRESHOLD:
            return trait
    return_value = None
    return return_value


def _trait_id(candidate: AcceptedCandidate) -> str:
    seed = "|".join([
        candidate["growth_axis"],
        _normalize_text(candidate["guidance"]),
        ",".join(_unique_sorted(candidate["source_memory_unit_ids"])),
    ])
    digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
    return_value = f"gct_{digest}"
    return return_value


def _normalize_text(value: str) -> str:
    normalized = " ".join(value.lower().split())
    return normalized


def _unique_sorted(values: Sequence[str]) -> list[str]:
    return_value = sorted({str(value) for value in values if str(value)})
    return return_value


def _clamp(value: float) -> float:
    return_value = max(0.0, min(1.0, value))
    return return_value
