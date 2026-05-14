"""Patched orchestration tests for global character growth runner."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.global_character_growth import models as growth_models
from kazusa_ai_chatbot.global_character_growth import runner as runner_module


@pytest.mark.asyncio
async def test_runner_dry_run_records_run_without_trait_writes(monkeypatch) -> None:
    """Dry-run should call the LLM and record audit data without mutating traits."""

    recorded_runs = []
    monkeypatch.setattr(
        runner_module,
        "find_active_memory_units",
        AsyncMock(return_value=[
            (-1.0, _memory_doc("memory-1", "2026-05-01")),
            (-1.0, _memory_doc("memory-2", "2026-05-03")),
        ]),
    )
    monkeypatch.setattr(
        runner_module.growth_store,
        "list_active_growth_traits",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runner_module,
        "generate_growth_candidates",
        AsyncMock(return_value={
            "candidate_deltas": [_candidate()],
            "summary": "one candidate",
        }),
    )

    async def _insert_run(document: dict) -> None:
        recorded_runs.append(document)

    write_traits = AsyncMock()
    monkeypatch.setattr(
        runner_module.growth_store,
        "insert_growth_run_document",
        _insert_run,
    )
    monkeypatch.setattr(
        runner_module.growth_store,
        "upsert_growth_trait_documents",
        write_traits,
    )

    result = await runner_module.run_global_character_growth_pass(
        character_local_date="2026-05-10",
        dry_run=True,
        enable_trait_writes=False,
        now=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )

    assert result["status"] == "dry_run"
    assert result["dry_run"] is True
    assert result["accepted_candidate_count"] == 1
    assert result["trait_update_count"] == 1
    assert result["shadow_projection_count"] == 1
    assert recorded_runs[0]["status"] == "dry_run"
    assert recorded_runs[0]["input_counts"]["eligible_memory_cards"] == 2
    assert recorded_runs[0]["shadow_projection"]
    write_traits.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_apply_requires_explicit_trait_write_enablement() -> None:
    """Manual apply mode must not silently mutate traits."""

    with pytest.raises(ValueError, match="enable_trait_writes"):
        await runner_module.run_global_character_growth_pass(
            character_local_date="2026-05-10",
            dry_run=False,
            enable_trait_writes=False,
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )


@pytest.mark.asyncio
async def test_runner_apply_writes_traits_and_run_document(monkeypatch) -> None:
    """Apply mode writes only the new trait ledger and run record."""

    recorded_runs = []
    write_traits = AsyncMock()
    monkeypatch.setattr(
        runner_module,
        "find_active_memory_units",
        AsyncMock(return_value=[
            (-1.0, _memory_doc("memory-1", "2026-05-01")),
            (-1.0, _memory_doc("memory-2", "2026-05-03")),
        ]),
    )
    monkeypatch.setattr(
        runner_module.growth_store,
        "list_active_growth_traits",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runner_module,
        "generate_growth_candidates",
        AsyncMock(return_value={
            "candidate_deltas": [_candidate()],
            "summary": "one candidate",
        }),
    )

    async def _insert_run(document: dict) -> None:
        recorded_runs.append(document)

    monkeypatch.setattr(
        runner_module.growth_store,
        "insert_growth_run_document",
        _insert_run,
    )
    monkeypatch.setattr(
        runner_module.growth_store,
        "upsert_growth_trait_documents",
        write_traits,
    )

    result = await runner_module.run_global_character_growth_pass(
        character_local_date="2026-05-10",
        dry_run=False,
        enable_trait_writes=True,
        now=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )

    assert result["status"] == "applied"
    assert recorded_runs[0]["status"] == "applied"
    write_traits.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_skips_when_no_eligible_memory(monkeypatch) -> None:
    """Sparse upstream reflection memory should become an auditable skip."""

    recorded_runs = []
    monkeypatch.setattr(
        runner_module,
        "find_active_memory_units",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runner_module.growth_store,
        "list_active_growth_traits",
        AsyncMock(return_value=[]),
    )
    generate = AsyncMock()
    monkeypatch.setattr(runner_module, "generate_growth_candidates", generate)

    async def _insert_run(document: dict) -> None:
        recorded_runs.append(document)

    monkeypatch.setattr(
        runner_module.growth_store,
        "insert_growth_run_document",
        _insert_run,
    )

    result = await runner_module.run_global_character_growth_pass(
        character_local_date="2026-05-10",
        dry_run=True,
        enable_trait_writes=False,
        now=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )

    assert result["status"] == "skipped"
    assert result["warning_count"] >= 1
    assert recorded_runs[0]["status"] == "skipped"
    assert recorded_runs[0]["input_counts"]["eligible_memory_cards"] == 0
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics(
    monkeypatch,
) -> None:
    """The LLM should receive only the final budgeted candidate payload."""

    recorded_runs = []
    budget_call_count = 0
    prompt_payload = _candidate_prompt_payload([
        _memory_card("memory-1", "2026-05-01"),
        _memory_card("memory-2", "2026-05-03"),
    ])
    input_quality = _input_quality(eligible_memory_cards=2)
    prompt_budget = _prompt_budget(
        before_chars=5000,
        after_chars=3000,
        before_cards=3,
        after_cards=2,
        dropped_cards=1,
        status="trimmed_to_budget",
    )
    monkeypatch.setattr(
        runner_module,
        "find_active_memory_units",
        AsyncMock(return_value=[
            (-1.0, _memory_doc("memory-1", "2026-05-01")),
            (-1.0, _memory_doc("memory-2", "2026-05-03")),
        ]),
    )
    monkeypatch.setattr(
        runner_module.growth_store,
        "list_active_growth_traits",
        AsyncMock(return_value=[]),
    )

    def build_budgeted_candidate_prompt_payload(**kwargs):
        nonlocal budget_call_count

        budget_call_count += 1
        assert kwargs["prompt_char_budget"] == (
            runner_module.GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET
        )
        assert callable(kwargs["prompt_char_counter"])
        return_value = (prompt_payload, input_quality, prompt_budget)
        return return_value

    generate = AsyncMock(return_value={
        "candidate_deltas": [_candidate()],
        "summary": "one candidate",
    })
    monkeypatch.setattr(
        runner_module,
        "build_budgeted_candidate_prompt_payload",
        build_budgeted_candidate_prompt_payload,
        raising=False,
    )
    monkeypatch.setattr(runner_module, "generate_growth_candidates", generate)

    async def _insert_run(document: dict) -> None:
        recorded_runs.append(document)

    monkeypatch.setattr(
        runner_module.growth_store,
        "insert_growth_run_document",
        _insert_run,
    )
    monkeypatch.setattr(
        runner_module.growth_store,
        "upsert_growth_trait_documents",
        AsyncMock(),
    )

    result = await runner_module.run_global_character_growth_pass(
        character_local_date="2026-05-10",
        dry_run=True,
        enable_trait_writes=False,
        now=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )

    assert budget_call_count == 1
    generate.assert_awaited_once_with(payload=prompt_payload)
    assert recorded_runs[0]["prompt_budget"] == prompt_budget
    assert result["dropped_memory_cards_for_prompt_budget"] == 1
    assert result["rendered_prompt_chars_after_budget"] == 3000


@pytest.mark.asyncio
async def test_runner_skips_llm_when_prompt_budget_drops_all_cards(
    monkeypatch,
) -> None:
    """Prompt budgeting can turn an otherwise eligible run into a skip."""

    recorded_runs = []
    budget_call_count = 0
    prompt_payload = _candidate_prompt_payload([])
    input_quality = _input_quality(eligible_memory_cards=2)
    prompt_budget = _prompt_budget(
        before_chars=5000,
        after_chars=4100,
        before_cards=2,
        after_cards=0,
        dropped_cards=2,
        status="empty_after_budget",
    )
    monkeypatch.setattr(
        runner_module,
        "find_active_memory_units",
        AsyncMock(return_value=[
            (-1.0, _memory_doc("memory-1", "2026-05-01")),
            (-1.0, _memory_doc("memory-2", "2026-05-03")),
        ]),
    )
    monkeypatch.setattr(
        runner_module.growth_store,
        "list_active_growth_traits",
        AsyncMock(return_value=[]),
    )

    def build_budgeted_candidate_prompt_payload(**_kwargs):
        nonlocal budget_call_count

        budget_call_count += 1
        return_value = (prompt_payload, input_quality, prompt_budget)
        return return_value

    generate = AsyncMock(return_value={
        "candidate_deltas": [],
        "summary": "no candidate",
    })
    monkeypatch.setattr(
        runner_module,
        "build_budgeted_candidate_prompt_payload",
        build_budgeted_candidate_prompt_payload,
        raising=False,
    )
    monkeypatch.setattr(runner_module, "generate_growth_candidates", generate)

    async def _insert_run(document: dict) -> None:
        recorded_runs.append(document)

    monkeypatch.setattr(
        runner_module.growth_store,
        "insert_growth_run_document",
        _insert_run,
    )

    result = await runner_module.run_global_character_growth_pass(
        character_local_date="2026-05-10",
        dry_run=True,
        enable_trait_writes=False,
        now=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )

    assert budget_call_count == 1
    generate.assert_not_awaited()
    assert result["status"] == "skipped"
    assert recorded_runs[0]["status"] == "skipped"
    assert recorded_runs[0]["summary"] == (
        "No eligible reflection-promoted memory cards after prompt budget."
    )
    assert recorded_runs[0]["prompt_budget"] == prompt_budget
    assert result["dropped_memory_cards_for_prompt_budget"] == 2
    assert result["rendered_prompt_chars_after_budget"] == 4100


@pytest.mark.asyncio
async def test_runner_records_failed_llm_attempt(monkeypatch) -> None:
    """LLM failures should leave a failure run document for operators."""

    recorded_runs = []
    monkeypatch.setattr(
        runner_module,
        "find_active_memory_units",
        AsyncMock(return_value=[
            (-1.0, _memory_doc("memory-1", "2026-05-01")),
            (-1.0, _memory_doc("memory-2", "2026-05-03")),
        ]),
    )
    monkeypatch.setattr(
        runner_module.growth_store,
        "list_active_growth_traits",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runner_module,
        "generate_growth_candidates",
        AsyncMock(side_effect=RuntimeError("llm unavailable")),
    )

    async def _insert_run(document: dict) -> None:
        recorded_runs.append(document)

    monkeypatch.setattr(
        runner_module.growth_store,
        "insert_growth_run_document",
        _insert_run,
    )

    result = await runner_module.run_global_character_growth_pass(
        character_local_date="2026-05-10",
        dry_run=True,
        enable_trait_writes=False,
        now=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )

    assert result["status"] == "failed"
    assert "llm unavailable" in recorded_runs[0]["error"]
    assert recorded_runs[0]["status"] == "failed"
    assert recorded_runs[0]["input_counts"]["eligible_memory_cards"] == 2
    assert recorded_runs[0]["prompt_budget"]["memory_cards_before_prompt_budget"] == 2
    assert (
        recorded_runs[0]["prompt_budget"]["rendered_prompt_chars_before_budget"] > 0
    )
    assert result["rendered_prompt_chars_after_budget"] > 0


def test_run_id_is_deterministic_for_same_inputs() -> None:
    """Run ids should be stable for replay and trace inspection."""

    first = runner_module.build_run_id(
        character_local_date="2026-05-10",
        dry_run=True,
        now=datetime(2026, 5, 11, 1, 2, 3, tzinfo=timezone.utc),
    )
    second = runner_module.build_run_id(
        character_local_date="2026-05-10",
        dry_run=True,
        now=datetime(2026, 5, 11, 1, 2, 3, tzinfo=timezone.utc),
    )

    assert first == second
    assert first.startswith("global_character_growth:")


def _memory_doc(memory_id: str, source_date: str) -> dict:
    """Build one reflection-promoted memory row fixture."""

    return {
        "memory_unit_id": memory_id,
        "memory_name": f"Memory {memory_id}",
        "content": "Repeated general communication pattern.",
        "memory_type": "defense_rule",
        "source_kind": "reflection_inferred",
        "authority": "reflection_promoted",
        "source_global_user_id": "",
        "status": "active",
        "evidence_refs": [{
            "reflection_run_id": f"run-{memory_id}",
            "captured_at": f"{source_date}T10:00:00+00:00",
        }],
        "confidence_note": "stable support",
        "updated_at": f"{source_date}T10:00:00+00:00",
    }


def _memory_card(memory_id: str, source_date: str) -> dict:
    """Build one prompt-visible memory card fixture."""

    return {
        "source_card_id": f"card-{memory_id}",
        "memory_unit_id": memory_id,
        "memory_name": f"Memory {memory_id}",
        "memory_type": "defense_rule",
        "content": "Repeated general communication pattern.",
        "character_local_dates": [source_date],
        "source_reflection_run_ids": [f"run-{memory_id}"],
        "confidence_note": "stable support",
    }


def _candidate_prompt_payload(memory_cards: list[dict]) -> dict:
    """Build one candidate prompt payload fixture."""

    return {
        "evaluation_mode": growth_models.EVALUATION_MODE,
        "prompt_version": growth_models.PROMPT_VERSION,
        "memory_cards": memory_cards,
        "current_global_growth_traits": [],
        "candidate_limits": {
            "max_candidates": growth_models.MAX_ACCEPTED_CANDIDATES,
            "max_source_cards_per_candidate": (
                growth_models.MAX_SOURCE_CARDS_PER_CANDIDATE
            ),
        },
        "allowed_growth_axes": list(growth_models.ALLOWED_GROWTH_AXES),
    }


def _input_quality(*, eligible_memory_cards: int) -> dict:
    """Build input-quality diagnostics for runner tests."""

    return {
        "raw_memory_rows": eligible_memory_cards,
        "eligible_memory_cards": eligible_memory_cards,
        "unique_source_dates": eligible_memory_cards,
        "source_date_span_days": eligible_memory_cards,
        "promotion_density": "sparse",
        "dropped_rows": {},
        "quality_notes": [],
    }


def _prompt_budget(
    *,
    before_chars: int,
    after_chars: int,
    before_cards: int,
    after_cards: int,
    dropped_cards: int,
    status: str,
) -> dict:
    """Build prompt-budget diagnostics for runner tests."""

    return {
        "prompt_char_budget": 32000,
        "rendered_prompt_chars_before_budget": before_chars,
        "rendered_prompt_chars_after_budget": after_chars,
        "memory_cards_before_prompt_budget": before_cards,
        "memory_cards_after_prompt_budget": after_cards,
        "dropped_memory_cards_for_prompt_budget": dropped_cards,
        "prompt_budget_status": status,
    }


def _candidate() -> dict:
    """Build one LLM candidate fixture."""

    return {
        "candidate_action": "observe_trait",
        "growth_axis": "clarity",
        "trait_name": "关心与边界",
        "guidance": "保持关心可见，同时保留清晰同意空间。",
        "source_card_ids": ["card-memory-1", "card-memory-2"],
        "supporting_dates": ["2026-05-01", "2026-05-03"],
        "scope_assessment": "global",
        "support_level": "stable",
        "confidence": "high",
        "private_detail_risk": "low",
        "novelty_reason": "new repeated pattern",
        "stability_reason": "seen on multiple days",
        "rejection_reason": "",
    }
