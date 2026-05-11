"""Prompt-safe projections for global character growth."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from kazusa_ai_chatbot.global_character_growth.models import (
    ALLOWED_GROWTH_AXES,
    EVALUATION_MODE,
    MAX_ACCEPTED_CANDIDATES,
    MAX_CARD_CONFIDENCE_NOTE_CHARS,
    MAX_CARD_CONTENT_CHARS,
    MAX_CARD_DATES,
    MAX_CARD_REFLECTION_RUN_IDS,
    MAX_CURRENT_TRAITS,
    MAX_CURRENT_TRAIT_GUIDANCE_CHARS,
    MAX_MEMORY_CARDS,
    MAX_SOURCE_CARDS_PER_CANDIDATE,
    PROMPT_VERSION,
    RUNTIME_CONTEXT_LIMIT,
    SHADOW_PROJECTION_LIMIT,
    CandidatePromptPayload,
    GlobalCharacterGrowthContext,
    InputQualityDiagnostics,
    MemoryCard,
)


def build_candidate_prompt_payload(
    *,
    memory_rows: Sequence[Mapping[str, Any]],
    current_trait_rows: Sequence[Mapping[str, Any]],
    limit: int = MAX_MEMORY_CARDS,
) -> CandidatePromptPayload:
    """Build the bounded JSON payload for candidate generation."""

    memory_cards, _ = build_memory_cards(memory_rows, limit=limit)
    trait_summaries = project_current_traits(current_trait_rows)
    payload: CandidatePromptPayload = {
        "evaluation_mode": EVALUATION_MODE,
        "prompt_version": PROMPT_VERSION,
        "memory_cards": memory_cards,
        "current_global_growth_traits": trait_summaries,
        "candidate_limits": {
            "max_candidates": MAX_ACCEPTED_CANDIDATES,
            "max_source_cards_per_candidate": MAX_SOURCE_CARDS_PER_CANDIDATE,
        },
        "allowed_growth_axes": list(ALLOWED_GROWTH_AXES),
    }
    return payload


def build_memory_cards(
    memory_rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = MAX_MEMORY_CARDS,
) -> tuple[list[MemoryCard], InputQualityDiagnostics]:
    """Project reflection-promoted memory rows into source cards."""

    capped_limit = min(max(limit, 0), MAX_MEMORY_CARDS)
    raw_memory_rows = len(memory_rows)
    cards: list[MemoryCard] = []
    dropped_rows: dict[str, int] = {}
    all_dates: list[str] = []
    for row in memory_rows:
        drop_reason = _drop_reason(row)
        if drop_reason:
            dropped_rows[drop_reason] = dropped_rows.get(drop_reason, 0) + 1
            continue
        if len(cards) >= capped_limit:
            dropped_rows["limit_exceeded"] = dropped_rows.get("limit_exceeded", 0) + 1
            continue
        card = _memory_card_from_row(row)
        cards.append(card)
        all_dates.extend(card["character_local_dates"])

    unique_dates = sorted(set(all_dates))
    date_span_days = _date_span_days(unique_dates)
    diagnostics: InputQualityDiagnostics = {
        "raw_memory_rows": raw_memory_rows,
        "eligible_memory_cards": len(cards),
        "unique_source_dates": len(unique_dates),
        "source_date_span_days": date_span_days,
        "promotion_density": _promotion_density(len(cards), len(unique_dates)),
        "dropped_rows": dropped_rows,
        "quality_notes": _quality_notes(len(cards), len(unique_dates), dropped_rows),
    }
    return_value = (cards, diagnostics)
    return return_value


def project_current_traits(
    trait_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Project current traits into candidate-generation summaries."""

    projected: list[dict[str, str]] = []
    for row in trait_rows:
        if len(projected) >= MAX_CURRENT_TRAITS:
            break
        if str(row.get("status", "")) != "active":
            continue
        projected.append({
            "trait_id": str(row.get("trait_id", "")),
            "growth_axis": str(row.get("growth_axis", "")),
            "guidance": _truncate(
                str(row.get("guidance", "")),
                MAX_CURRENT_TRAIT_GUIDANCE_CHARS,
            ),
            "maturity_band": str(row.get("maturity_band", "")),
        })
    return projected


def build_shadow_projection(trait_rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Build log-only guidance for recently observed trait updates."""

    shadow_rows: list[dict] = []
    for row in trait_rows:
        raw_trait = row.get("trait")
        trait = raw_trait if isinstance(raw_trait, dict) else row
        if not isinstance(trait, Mapping):
            continue
        if str(trait.get("status", "")) != "active":
            continue
        maturity = str(trait.get("maturity_band", ""))
        if maturity not in {"emerging", "stabilizing", "promoted"}:
            continue
        prompt_visible = (
            str(trait.get("status", "")) == "active"
            and maturity == "promoted"
        )
        review_note = (
            "Prompt-visible now."
            if prompt_visible
            else "Not prompt-visible until maturity is promoted."
        )
        shadow_rows.append({
            "growth_axis": str(trait.get("growth_axis", "")),
            "guidance": str(trait.get("guidance", "")),
            "maturity": maturity,
            "prompt_visible_now": prompt_visible,
            "review_note": review_note,
        })
        if len(shadow_rows) >= SHADOW_PROJECTION_LIMIT:
            break
    return shadow_rows


def project_runtime_context(
    trait_rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = RUNTIME_CONTEXT_LIMIT,
) -> GlobalCharacterGrowthContext:
    """Project active promoted traits for L2 prompt use."""

    promoted_rows: list[dict] = []
    capped_limit = min(max(limit, 0), RUNTIME_CONTEXT_LIMIT)
    for row in trait_rows:
        if len(promoted_rows) >= capped_limit:
            break
        if str(row.get("status", "")) != "active":
            continue
        if str(row.get("maturity_band", "")) != "promoted":
            continue
        promoted_rows.append({
            "growth_axis": str(row.get("growth_axis", "")),
            "guidance": str(row.get("guidance", "")),
            "maturity": "promoted",
            "updated_at": _date_prefix(str(row.get("updated_at", ""))),
        })
    if not promoted_rows:
        return_value: GlobalCharacterGrowthContext = {}
        return return_value
    context: GlobalCharacterGrowthContext = {
        "promoted_global_growth": promoted_rows,
        "retrieval_notes": [
            "Only active promoted global character-growth traits are included.",
        ],
    }
    return context


def _memory_card_from_row(row: Mapping[str, Any]) -> MemoryCard:
    memory_unit_id = str(row.get("memory_unit_id", ""))
    evidence_refs = row.get("evidence_refs", [])
    source_dates = _source_dates(row, evidence_refs)
    source_run_ids = _source_reflection_run_ids(evidence_refs)
    card: MemoryCard = {
        "source_card_id": f"card-{memory_unit_id}",
        "memory_unit_id": memory_unit_id,
        "memory_name": str(row.get("memory_name", "")),
        "memory_type": str(row.get("memory_type", "")),
        "content": _truncate(str(row.get("content", "")), MAX_CARD_CONTENT_CHARS),
        "character_local_dates": source_dates[:MAX_CARD_DATES],
        "source_reflection_run_ids": source_run_ids[:MAX_CARD_REFLECTION_RUN_IDS],
        "confidence_note": _truncate(
            str(row.get("confidence_note", "")),
            MAX_CARD_CONFIDENCE_NOTE_CHARS,
        ),
    }
    return card


def _drop_reason(row: Mapping[str, Any]) -> str:
    if str(row.get("source_kind", "")) != "reflection_inferred":
        return "not_reflection_promoted"
    if str(row.get("authority", "")) != "reflection_promoted":
        return "not_reflection_promoted"
    if str(row.get("source_global_user_id", "")) != "":
        return "not_global"
    if str(row.get("status", "")) != "active":
        return "not_active"
    if not str(row.get("memory_unit_id", "")):
        return "missing_memory_unit_id"
    if not str(row.get("content", "")):
        return "empty_content"
    return_value = ""
    return return_value


def _source_dates(row: Mapping[str, Any], evidence_refs: object) -> list[str]:
    dates: list[str] = []
    if isinstance(evidence_refs, Sequence) and not isinstance(evidence_refs, str):
        for ref in evidence_refs:
            if not isinstance(ref, Mapping):
                continue
            captured_at = str(ref.get("captured_at", ""))
            source_date = _date_prefix(captured_at)
            if source_date and source_date not in dates:
                dates.append(source_date)
    updated_date = _date_prefix(str(row.get("updated_at", "")))
    if updated_date and updated_date not in dates:
        dates.append(updated_date)
    return dates


def _source_reflection_run_ids(evidence_refs: object) -> list[str]:
    run_ids: list[str] = []
    if isinstance(evidence_refs, Sequence) and not isinstance(evidence_refs, str):
        for ref in evidence_refs:
            if not isinstance(ref, Mapping):
                continue
            run_id = str(ref.get("reflection_run_id", ""))
            if run_id and run_id not in run_ids:
                run_ids.append(run_id)
    return run_ids


def _date_prefix(value: str) -> str:
    if len(value) < 10:
        return_value = ""
        return return_value
    candidate = value[:10]
    try:
        date.fromisoformat(candidate)
    except ValueError:
        return_value = ""
        return return_value
    return_value = candidate
    return return_value


def _date_span_days(source_dates: Sequence[str]) -> int:
    if len(source_dates) < 2:
        return_value = 0
        return return_value
    parsed_dates = [date.fromisoformat(source_date) for source_date in source_dates]
    return_value = (max(parsed_dates) - min(parsed_dates)).days
    return return_value


def _promotion_density(eligible_cards: int, unique_dates: int) -> str:
    if eligible_cards == 0:
        return_value = "none"
        return return_value
    if eligible_cards < 4 or unique_dates < 3:
        return_value = "sparse"
        return return_value
    return_value = "adequate"
    return return_value


def _quality_notes(
    eligible_cards: int,
    unique_dates: int,
    dropped_rows: Mapping[str, int],
) -> list[str]:
    notes: list[str] = []
    if eligible_cards == 0:
        notes.append("No eligible reflection-promoted memory cards.")
    elif unique_dates < 3:
        notes.append("Eligible memory spans fewer than three source dates.")
    if dropped_rows:
        notes.append("Some memory rows were dropped by deterministic input gates.")
    return notes


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return_value = value
        return return_value
    return_value = value[:limit].rstrip()
    return return_value
