"""Runner facade for global character growth passes."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.config import GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET
from kazusa_ai_chatbot.db import global_character_growth as growth_store
from kazusa_ai_chatbot.global_character_growth.drift import plan_trait_updates
from kazusa_ai_chatbot.global_character_growth.llm import (
    count_candidate_generation_prompt_chars,
    generate_growth_candidates,
)
from kazusa_ai_chatbot.global_character_growth.models import (
    CandidatePromptPayload,
    MAX_CURRENT_TRAITS,
    MAX_MEMORY_CARDS,
    PROMPT_VERSION,
    GlobalCharacterGrowthRunResult,
    PromptBudgetDiagnostics,
    RUN_KIND,
)
from kazusa_ai_chatbot.global_character_growth.projection import (
    build_budgeted_candidate_prompt_payload,
    build_shadow_projection,
)
from kazusa_ai_chatbot.global_character_growth.validation import (
    validate_candidate_response,
)
from kazusa_ai_chatbot.memory_evolution import find_active_memory_units


logger = logging.getLogger(__name__)


async def run_global_character_growth_pass(
    *,
    character_local_date: str | None,
    dry_run: bool,
    enable_trait_writes: bool,
    limit: int = 80,
    now: datetime | None = None,
) -> GlobalCharacterGrowthRunResult:
    """Run a global character-growth pass."""

    if dry_run and enable_trait_writes:
        raise ValueError("enable_trait_writes cannot be true during dry_run")
    if not dry_run and not enable_trait_writes:
        raise ValueError("enable_trait_writes is required for apply mode")
    run_now = now or datetime.now(timezone.utc)
    local_date = character_local_date or run_now.date().isoformat()
    run_id = build_run_id(
        character_local_date=local_date,
        dry_run=dry_run,
        now=run_now,
    )
    run_created_at = run_now.astimezone(timezone.utc).isoformat()
    failure_input_quality: dict[str, Any] = {
        "raw_memory_rows": 0,
        "eligible_memory_cards": 0,
        "unique_source_dates": 0,
        "source_date_span_days": 0,
        "promotion_density": "none",
        "dropped_rows": {},
        "quality_notes": ["Global character growth pass failed."],
    }
    failure_prompt_budget = _zero_prompt_budget_diagnostics()
    failure_current_traits: list[dict] = []
    failure_memory_cards: list[dict] = []
    try:
        memory_results = await find_active_memory_units(
            query={
                "source_kind": "reflection_inferred",
                "authority": "reflection_promoted",
                "source_global_user_id": "",
            },
            limit=limit,
        )
        memory_rows = [document for _, document in memory_results]
        current_traits = await growth_store.list_active_growth_traits(
            limit=MAX_CURRENT_TRAITS,
        )
        failure_current_traits = current_traits

        def prompt_char_counter(payload: CandidatePromptPayload) -> int:
            count = count_candidate_generation_prompt_chars(payload=payload)
            return count

        prompt_payload, input_quality, prompt_budget = (
            build_budgeted_candidate_prompt_payload(
                memory_rows=memory_rows,
                current_trait_rows=current_traits,
                prompt_char_budget=GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET,
                prompt_char_counter=prompt_char_counter,
                limit=min(limit, MAX_MEMORY_CARDS),
            )
        )
        memory_cards = prompt_payload["memory_cards"]
        if input_quality["eligible_memory_cards"] == 0:
            prompt_budget = _zero_prompt_budget_diagnostics()
        failure_input_quality = input_quality
        failure_prompt_budget = prompt_budget
        failure_memory_cards = memory_cards
        if not memory_cards:
            summary = "No eligible reflection-promoted memory cards."
            if input_quality["eligible_memory_cards"] > 0:
                summary = (
                    "No eligible reflection-promoted memory cards after prompt budget."
                )
            run_doc = _run_document(
                run_id=run_id,
                status="skipped",
                dry_run=dry_run,
                created_at=run_created_at,
                character_local_date=local_date,
                input_quality=input_quality,
                prompt_budget=prompt_budget,
                current_traits=current_traits,
                memory_cards=memory_cards,
                accepted_candidates=[],
                rejected_candidates=[],
                trait_updates=[],
                shadow_projection=[],
                validation_warnings=input_quality["quality_notes"],
                raw_llm_output="",
                summary=summary,
                error="",
            )
            await growth_store.insert_growth_run_document(run_doc)
            result = _result_from_run_doc(run_doc)
            return result

        parsed_response = await generate_growth_candidates(payload=prompt_payload)
        raw_llm_output = str(parsed_response.pop("_raw_output", ""))
        validated = validate_candidate_response(
            parsed_response=parsed_response,
            memory_cards=memory_cards,
            current_trait_rows=current_traits,
        )
        trait_updates = plan_trait_updates(
            existing_trait_rows=current_traits,
            accepted_candidates=validated["accepted_candidates"],
            now_iso=run_created_at,
        )
        shadow_projection = build_shadow_projection(trait_updates)
        if not dry_run and trait_updates:
            await growth_store.upsert_growth_trait_documents([
                update["trait"]
                for update in trait_updates
            ])
        status = "dry_run" if dry_run else "applied"
        run_doc = _run_document(
            run_id=run_id,
            status=status,
            dry_run=dry_run,
            created_at=run_created_at,
            character_local_date=local_date,
            input_quality=input_quality,
            prompt_budget=prompt_budget,
            current_traits=current_traits,
            memory_cards=memory_cards,
            accepted_candidates=validated["accepted_candidates"],
            rejected_candidates=validated["rejected_candidates"],
            trait_updates=trait_updates,
            shadow_projection=shadow_projection,
            validation_warnings=validated["validation_warnings"],
            raw_llm_output=raw_llm_output or json.dumps(parsed_response, ensure_ascii=False),
            summary=str(parsed_response.get("summary", "")),
            error="",
        )
        await growth_store.insert_growth_run_document(run_doc)
        result = _result_from_run_doc(run_doc)
        return result
    except Exception as exc:
        logger.exception(f"Global character growth pass failed: {exc}")
        run_doc = _run_document(
            run_id=run_id,
            status="failed",
            dry_run=dry_run,
            created_at=run_created_at,
            character_local_date=local_date,
            input_quality=failure_input_quality,
            prompt_budget=failure_prompt_budget,
            current_traits=failure_current_traits,
            memory_cards=failure_memory_cards,
            accepted_candidates=[],
            rejected_candidates=[],
            trait_updates=[],
            shadow_projection=[],
            validation_warnings=["global_character_growth_failed"],
            raw_llm_output="",
            summary="",
            error=str(exc),
        )
        await growth_store.insert_growth_run_document(run_doc)
        result = _result_from_run_doc(run_doc)
        return result


def build_run_id(
    *,
    character_local_date: str,
    dry_run: bool,
    now: datetime,
) -> str:
    """Build a stable run id for one invocation."""

    mode = "dry_run" if dry_run else "apply"
    timestamp = now.astimezone(timezone.utc).isoformat()
    return_value = f"global_character_growth:{character_local_date}:{mode}:{timestamp}"
    return return_value


def _run_document(
    *,
    run_id: str,
    status: str,
    dry_run: bool,
    created_at: str,
    character_local_date: str,
    input_quality: dict[str, Any],
    prompt_budget: PromptBudgetDiagnostics,
    current_traits: list[dict],
    memory_cards: list[dict],
    accepted_candidates: list[dict],
    rejected_candidates: list[dict],
    trait_updates: list[dict],
    shadow_projection: list[dict],
    validation_warnings: list[str],
    raw_llm_output: str,
    summary: str,
    error: str,
) -> dict[str, Any]:
    """Build a durable run document."""

    source_memory_unit_ids = _unique_sorted([
        card["memory_unit_id"]
        for card in memory_cards
    ])
    source_reflection_run_ids = _unique_sorted([
        run_id_value
        for card in memory_cards
        for run_id_value in card["source_reflection_run_ids"]
    ])
    run_doc = {
        "_id": run_id,
        "run_id": run_id,
        "run_kind": RUN_KIND,
        "status": status,
        "dry_run": dry_run,
        "prompt_version": PROMPT_VERSION,
        "created_at": created_at,
        "updated_at": created_at,
        "character_local_date": character_local_date,
        "input_counts": {
            "raw_memory_rows": input_quality["raw_memory_rows"],
            "eligible_memory_cards": input_quality["eligible_memory_cards"],
            "current_traits": len(current_traits),
        },
        "input_quality": {
            "promotion_density": input_quality["promotion_density"],
            "eligible_date_count": input_quality["unique_source_dates"],
            "date_span_days": input_quality["source_date_span_days"],
            "dropped_memory_cards_by_reason": input_quality["dropped_rows"],
            "quality_notes": input_quality["quality_notes"],
        },
        "prompt_budget": prompt_budget,
        "source_memory_unit_ids": source_memory_unit_ids,
        "source_reflection_run_ids": source_reflection_run_ids,
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": rejected_candidates,
        "trait_updates": trait_updates,
        "shadow_projection": shadow_projection,
        "validation_warnings": validation_warnings,
        "raw_llm_output": raw_llm_output,
        "summary": summary,
        "error": error,
    }
    return run_doc


def _result_from_run_doc(run_doc: dict[str, Any]) -> GlobalCharacterGrowthRunResult:
    promoted_trait_count = 0
    for update in run_doc["trait_updates"]:
        trait = update.get("trait", {})
        if not isinstance(trait, dict):
            continue
        if trait.get("maturity_band") == "promoted":
            promoted_trait_count += 1

    result: GlobalCharacterGrowthRunResult = {
        "run_id": str(run_doc["run_id"]),
        "run_kind": RUN_KIND,
        "status": str(run_doc["status"]),
        "dry_run": bool(run_doc["dry_run"]),
        "eligible_memory_cards": int(run_doc["input_counts"]["eligible_memory_cards"]),
        "accepted_candidate_count": len(run_doc["accepted_candidates"]),
        "rejected_candidate_count": len(run_doc["rejected_candidates"]),
        "trait_update_count": len(run_doc["trait_updates"]),
        "promoted_trait_count": promoted_trait_count,
        "shadow_projection_count": len(run_doc["shadow_projection"]),
        "input_quality_density": str(run_doc["input_quality"]["promotion_density"]),
        "dropped_memory_cards_for_prompt_budget": int(
            run_doc["prompt_budget"]["dropped_memory_cards_for_prompt_budget"],
        ),
        "rendered_prompt_chars_after_budget": int(
            run_doc["prompt_budget"]["rendered_prompt_chars_after_budget"],
        ),
        "warning_count": len(run_doc["validation_warnings"]),
    }
    return result


def _zero_prompt_budget_diagnostics() -> PromptBudgetDiagnostics:
    """Build zeroed prompt-budget diagnostics for pre-budget exits."""

    prompt_budget: PromptBudgetDiagnostics = {
        "prompt_char_budget": GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET,
        "rendered_prompt_chars_before_budget": 0,
        "rendered_prompt_chars_after_budget": 0,
        "memory_cards_before_prompt_budget": 0,
        "memory_cards_after_prompt_budget": 0,
        "dropped_memory_cards_for_prompt_budget": 0,
        "prompt_budget_status": "within_budget",
    }
    return prompt_budget


def _unique_sorted(values: list[str]) -> list[str]:
    return_value = sorted({str(value) for value in values if str(value)})
    return return_value
