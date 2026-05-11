"""Patched orchestration tests for global character growth runner."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

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
