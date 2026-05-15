"""Deterministic tests for global reflection promotion."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.memory_evolution.models import (
    MemoryAuthority,
    MemorySourceKind,
    MemoryStatus,
)
from kazusa_ai_chatbot.reflection_cycle import promotion as promotion_module


@pytest.fixture(autouse=True)
def _mock_character_profile(monkeypatch) -> None:
    """Give promotion prompts a deterministic active character profile name."""

    monkeypatch.setattr(
        promotion_module,
        "get_character_profile",
        AsyncMock(return_value={"name": "杏山千纱 (Kyōyama Kazusa)"}),
    )


def test_global_promotion_prompt_has_required_contract_sections() -> None:
    """Prompt render should expose the pinned promotion contract sections."""

    payload = _promotion_payload()
    prompt = promotion_module.build_global_promotion_prompt(
        payload,
        character_name="杏山千纱 (Kyōyama Kazusa)",
    )

    for header in (
        "# 任务",
        "# 核心任务",
        "# 语言政策",
        "# 记忆视角契约",
        "# 生成步骤",
        "# 输入格式",
        "# 输出格式",
        "# 禁止事项",
    ):
        assert header in prompt.system_prompt
    assert "evaluation_mode" in prompt.human_prompt
    assert "channel_daily_syntheses" in prompt.human_prompt
    assert "evidence_cards" in prompt.human_prompt
    assert "promotion_limits" in prompt.human_prompt
    assert "promotion_decisions" in prompt.system_prompt
    assert "杏山千纱 (Kyōyama Kazusa)" in prompt.system_prompt
    assert "character_profile" not in prompt.system_prompt
    assert "active_character" not in prompt.system_prompt
    assert "active_character" not in prompt.human_prompt
    assert "source_utterance" in prompt.human_prompt


def test_promotion_validation_rejects_private_or_boundary_unsafe_rows() -> None:
    """Unsafe promotion rows should produce deterministic validation warnings."""

    unsafe = _decision("lore")
    unsafe["privacy_review"] = {
        "private_detail_risk": "high",
        "user_details_removed": False,
        "boundary_assessment": "unsafe",
        "reviewer": "automated_llm",
    }
    unsafe["boundary_assessment"] = {
        "verdict": "blocked",
        "affects_identity_or_boundaries": True,
        "reason": "unsafe",
    }

    warnings = promotion_module.validate_promotion_decisions([unsafe])

    assert any("user details" in warning for warning in warnings)
    assert any("private_detail_risk" in warning for warning in warnings)
    assert any("boundary verdict" in warning for warning in warnings)


def test_evidence_cards_preserve_low_privacy_risk_for_negated_notes() -> None:
    """Evidence card privacy risk should not be keyword-classified in code."""

    hourly_doc = _hourly_doc()
    hourly_doc["output"]["privacy_notes"] = ['无明显隐私风险']

    cards = promotion_module._evidence_cards_from_hourly_doc(hourly_doc)

    assert cards[0]["private_detail_risk"] == "low"


@pytest.mark.asyncio
async def test_global_promotion_skips_existing_succeeded_run_without_llm_or_writes(
    monkeypatch,
) -> None:
    """A succeeded daily global promotion run should not replay write work."""

    character_local_date = "2026-05-04"
    global_run_id = promotion_module.repository.daily_global_promotion_run_id(
        character_local_date=character_local_date,
        prompt_version=promotion_module.GLOBAL_PROMOTION_PROMPT_VERSION,
    )
    existing_run = _global_run_doc(status="succeeded", run_id=global_run_id)
    daily_channel_runs = AsyncMock(return_value=[_daily_doc()])
    run_promotion_llm = AsyncMock(
        return_value={"promotion_decisions": [_decision("lore")]},
    )
    find_memory = AsyncMock(return_value=[])
    insert_memory = AsyncMock(return_value=_stored_memory_unit("unit-1"))
    supersede_memory = AsyncMock()
    merge_memory = AsyncMock()
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=existing_run),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        daily_channel_runs,
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        run_promotion_llm,
    )
    monkeypatch.setattr(promotion_module, "find_active_memory_units", find_memory)
    monkeypatch.setattr(promotion_module, "insert_memory_unit", insert_memory)
    monkeypatch.setattr(promotion_module, "supersede_memory_unit", supersede_memory)
    monkeypatch.setattr(promotion_module, "merge_memory_units", merge_memory)
    monkeypatch.setattr(promotion_module.repository, "upsert_run", AsyncMock())

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date=character_local_date,
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.skipped_count == 1
    assert result.succeeded_count == 0
    assert result.run_ids == [global_run_id]
    assert result.defer_reason == "daily global promotion already succeeded"
    daily_channel_runs.assert_not_awaited()
    run_promotion_llm.assert_not_awaited()
    find_memory.assert_not_awaited()
    insert_memory.assert_not_awaited()
    supersede_memory.assert_not_awaited()
    merge_memory.assert_not_awaited()


@pytest.mark.parametrize("status", ["skipped", "failed", "dry_run"])
@pytest.mark.asyncio
async def test_global_promotion_retries_existing_skipped_failed_and_dry_run_rows(
    monkeypatch,
    status,
) -> None:
    """Only succeeded daily global promotion rows should block retry."""

    character_local_date = "2026-05-04"
    global_run_id = promotion_module.repository.daily_global_promotion_run_id(
        character_local_date=character_local_date,
        prompt_version=promotion_module.GLOBAL_PROMOTION_PROMPT_VERSION,
    )
    persisted = []

    async def _reflection_run_by_id(run_id):
        if run_id == global_run_id:
            existing = _global_run_doc(status=status, run_id=global_run_id)
            return existing
        hourly_doc = _hourly_doc()
        return hourly_doc

    async def _upsert(document):
        persisted.append(document)

    run_promotion_llm = AsyncMock(
        return_value={"promotion_decisions": [_decision("lore")]},
    )
    insert_memory = AsyncMock(return_value=_stored_memory_unit("unit-1"))
    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(side_effect=_reflection_run_by_id),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        run_promotion_llm,
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(promotion_module, "insert_memory_unit", insert_memory)
    monkeypatch.setattr(promotion_module.repository, "upsert_run", _upsert)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date=character_local_date,
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.succeeded_count == 1
    assert persisted[-1]["status"] == "succeeded"
    run_promotion_llm.assert_awaited_once()
    insert_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_global_promotion_persists_skipped_when_memory_writes_disabled(
    monkeypatch,
) -> None:
    """Prompt-only apply runs must not persist as successful memory writes."""

    character_local_date = "2026-05-04"
    global_run_id = promotion_module.repository.daily_global_promotion_run_id(
        character_local_date=character_local_date,
        prompt_version=promotion_module.GLOBAL_PROMOTION_PROMPT_VERSION,
    )
    persisted = []

    async def _reflection_run_by_id(run_id):
        if run_id == global_run_id:
            return_value = None
            return return_value
        hourly_doc = _hourly_doc()
        return hourly_doc

    async def _upsert(document):
        persisted.append(document)

    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(side_effect=_reflection_run_by_id),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        AsyncMock(return_value={"promotion_decisions": [_decision("lore")]}),
    )
    monkeypatch.setattr(promotion_module.repository, "upsert_run", _upsert)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date=character_local_date,
        dry_run=False,
        enable_memory_writes=False,
    )

    assert result.skipped_count == 1
    assert result.defer_reason == "memory writes disabled"
    assert persisted[-1]["status"] == "skipped"


@pytest.mark.asyncio
async def test_global_promotion_records_failed_write_phase_without_worker_crash(
    monkeypatch,
) -> None:
    """Unexpected write failures should become failed promotion results."""

    character_local_date = "2026-05-04"
    global_run_id = promotion_module.repository.daily_global_promotion_run_id(
        character_local_date=character_local_date,
        prompt_version=promotion_module.GLOBAL_PROMOTION_PROMPT_VERSION,
    )
    persisted = []

    async def _reflection_run_by_id(run_id):
        if run_id == global_run_id:
            return_value = None
            return return_value
        hourly_doc = _hourly_doc()
        return hourly_doc

    async def _upsert(document):
        persisted.append(document)

    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(side_effect=_reflection_run_by_id),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        AsyncMock(return_value={"promotion_decisions": [_decision("lore")]}),
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        promotion_module,
        "insert_memory_unit",
        AsyncMock(side_effect=RuntimeError("unexpected memory failure")),
    )
    monkeypatch.setattr(promotion_module.repository, "upsert_run", _upsert)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date=character_local_date,
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.failed_count == 1
    assert "unexpected memory failure" in result.defer_reason
    assert persisted[-1]["status"] == "failed"
    assert "unexpected memory failure" in persisted[-1]["error"]


@pytest.mark.asyncio
async def test_global_promotion_skips_memory_write_when_scores_are_unavailable(
    monkeypatch,
) -> None:
    """Malformed score rows should defer writes and leave memory APIs unused."""

    persisted = []
    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=_hourly_doc()),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        AsyncMock(return_value={"promotion_decisions": [_decision("lore")]}),
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[{"score": 0.5}]),
    )
    monkeypatch.setattr(promotion_module, "insert_memory_unit", AsyncMock())
    monkeypatch.setattr(promotion_module, "supersede_memory_unit", AsyncMock())
    monkeypatch.setattr(promotion_module, "merge_memory_units", AsyncMock())

    async def _upsert(document):
        persisted.append(document)

    monkeypatch.setattr(promotion_module.repository, "upsert_run", _upsert)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date="2026-05-04",
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.deferred is True
    assert result.defer_reason == "memory search returned malformed score rows"
    assert persisted[-1]["status"] == "skipped"
    promotion_module.insert_memory_unit.assert_not_awaited()
    promotion_module.supersede_memory_unit.assert_not_awaited()
    promotion_module.merge_memory_units.assert_not_awaited()


@pytest.mark.asyncio
async def test_promotion_logs_info_for_memory_mutation_and_debug_for_evidence(
    monkeypatch,
    caplog,
) -> None:
    """Promotion logs should split operator summary from supporting details."""

    stored = {
        "memory_unit_id": "unit-1",
        "lineage_id": "unit-1",
        "memory_type": "fact",
        "memory_name": "频道规则",
        "content": "角色确认群规应保持事实性，不应写成用户画像。",
        "source_global_user_id": "",
        "source_kind": MemorySourceKind.REFLECTION_INFERRED,
        "authority": MemoryAuthority.REFLECTION_PROMOTED,
        "status": MemoryStatus.ACTIVE,
    }
    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=_hourly_doc()),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        AsyncMock(return_value={"promotion_decisions": [_decision("lore")]}),
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        promotion_module,
        "insert_memory_unit",
        AsyncMock(return_value=stored),
    )
    monkeypatch.setattr(promotion_module.repository, "upsert_run", AsyncMock())
    caplog.set_level(logging.DEBUG, logger=promotion_module.__name__)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date="2026-05-04",
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.succeeded_count == 1
    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert any(
        "Reflection promotion memory mutation" in message
        for message in info_messages
    )
    assert any("频道规则" in message for message in info_messages)
    assert any("top_score" in message for message in debug_messages)
    assert all(
        "global-user" not in message
        for message in info_messages + debug_messages
    )
    assert all(
        "platform_user_id" not in message
        for message in info_messages + debug_messages
    )


@pytest.mark.asyncio
async def test_memory_write_lock_defers_promotion(monkeypatch) -> None:
    """Memory write lock contention should defer promotion."""

    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=_hourly_doc()),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        AsyncMock(return_value={"promotion_decisions": [_decision("lore")]}),
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        promotion_module,
        "insert_memory_unit",
        AsyncMock(side_effect=RuntimeError("memory write or reset is already running")),
    )
    monkeypatch.setattr(promotion_module.repository, "upsert_run", AsyncMock())

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date="2026-05-04",
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.deferred is True
    assert result.defer_reason == "memory write or reset is already running"


@pytest.mark.asyncio
async def test_promotion_skips_active_replacement_replay(monkeypatch) -> None:
    """Replaying an already-active replacement must not supersede itself."""

    character_local_date = "2026-05-04"
    global_run_id = promotion_module.repository.daily_global_promotion_run_id(
        character_local_date=character_local_date,
        prompt_version=promotion_module.GLOBAL_PROMOTION_PROMPT_VERSION,
    )
    decision = _decision("lore")
    deterministic_doc = promotion_module._memory_document_for_decision(
        decision=decision,
        character_local_date=character_local_date,
        global_run_id=global_run_id,
        source_unit_ids=[],
        source_lineage_ids=[],
        mutation_action="insert",
    )
    active_replacement = {
        **deterministic_doc,
        "lineage_id": "existing-lineage",
    }
    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=_hourly_doc()),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        AsyncMock(return_value={"promotion_decisions": [decision]}),
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[(0.95, active_replacement)]),
    )
    monkeypatch.setattr(promotion_module.repository, "upsert_run", AsyncMock())
    monkeypatch.setattr(
        promotion_module,
        "supersede_memory_unit",
        AsyncMock(
            side_effect=ValueError(
                "replacement memory_unit_id already exists",
            ),
        ),
    )

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date=character_local_date,
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.succeeded_count == 0
    assert result.skipped_count == 1
    assert result.memory_mutations == []
    assert any(
        "replacement already active" in warning
        for warning in result.validation_warnings
    )
    promotion_module.supersede_memory_unit.assert_not_awaited()


@pytest.mark.asyncio
async def test_promotion_skips_duplicate_replacement_id_from_different_source(
    monkeypatch,
) -> None:
    """Duplicate replacement ids should not crash when the source differs."""

    character_local_date = "2026-05-04"
    global_run_id = promotion_module.repository.daily_global_promotion_run_id(
        character_local_date=character_local_date,
        prompt_version=promotion_module.GLOBAL_PROMOTION_PROMPT_VERSION,
    )
    decision = _decision("lore")
    deterministic_doc = promotion_module._memory_document_for_decision(
        decision=decision,
        character_local_date=character_local_date,
        global_run_id=global_run_id,
        source_unit_ids=[],
        source_lineage_ids=[],
        mutation_action="insert",
    )
    active_source = {
        **deterministic_doc,
        "memory_unit_id": "different-active-source",
        "lineage_id": "different-lineage",
    }
    persisted = []

    async def _reflection_run_by_id(run_id):
        if run_id == global_run_id:
            return_value = None
            return return_value
        hourly_doc = _hourly_doc()
        return hourly_doc

    async def _upsert(document):
        persisted.append(document)

    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(side_effect=_reflection_run_by_id),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        AsyncMock(return_value={"promotion_decisions": [decision]}),
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[(0.95, active_source)]),
    )
    monkeypatch.setattr(
        promotion_module,
        "supersede_memory_unit",
        AsyncMock(
            side_effect=ValueError("replacement memory_unit_id already exists"),
        ),
    )
    monkeypatch.setattr(promotion_module.repository, "upsert_run", _upsert)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date=character_local_date,
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.succeeded_count == 0
    assert result.skipped_count == 1
    assert result.memory_mutations == []
    assert persisted[-1]["status"] == "skipped"
    assert any(
        "replacement memory_unit_id already exists" in warning
        for warning in result.validation_warnings
    )
    promotion_module.supersede_memory_unit.assert_awaited_once()


@pytest.mark.asyncio
async def test_primary_interaction_busy_defers_before_memory_write(
    monkeypatch,
) -> None:
    """A busy probe should prevent promotion memory writes after LLM output."""

    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=_hourly_doc()),
    )
    run_promotion_llm = AsyncMock(
        return_value={"promotion_decisions": [_decision("lore")]},
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        run_promotion_llm,
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(promotion_module, "insert_memory_unit", AsyncMock())
    monkeypatch.setattr(promotion_module.repository, "upsert_run", AsyncMock())

    busy_checks = iter([False, True])

    def _busy_probe() -> bool:
        is_busy = next(busy_checks)
        return is_busy

    result = await promotion_module._run_global_reflection_promotion(
        character_local_date="2026-05-04",
        dry_run=False,
        enable_memory_writes=True,
        is_primary_interaction_busy=_busy_probe,
    )

    assert result.deferred is True
    assert result.defer_reason == "primary interaction busy"
    run_promotion_llm.assert_awaited_once()
    promotion_module.find_active_memory_units.assert_not_awaited()
    promotion_module.insert_memory_unit.assert_not_awaited()


@pytest.mark.asyncio
async def test_promotion_uses_repository_evidence_refs_not_llm_refs(
    monkeypatch,
) -> None:
    """Memory writes should derive evidence refs from stored reflection runs."""

    captured_docs = []
    malicious_decision = _decision("lore")
    malicious_decision["evidence_refs"] = [
        {
            "reflection_run_id": "llm-made-up",
            "scope_ref": "made-up-scope",
            "captured_at": "2026-05-04T00:00:00+00:00",
            "source": "reflection_cycle",
        }
    ]
    stored = {
        "memory_unit_id": "unit-1",
        "lineage_id": "unit-1",
        "memory_type": "fact",
        "memory_name": "频道规则",
        "content": "角色确认群规应保持事实性，不应写成用户画像。",
        "source_global_user_id": "",
        "source_kind": MemorySourceKind.REFLECTION_INFERRED,
        "authority": MemoryAuthority.REFLECTION_PROMOTED,
        "status": MemoryStatus.ACTIVE,
    }

    async def _insert_memory_unit(document):
        captured_docs.append(document)
        return stored

    monkeypatch.setattr(
        promotion_module.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[_daily_doc()]),
    )
    monkeypatch.setattr(
        promotion_module.repository,
        "reflection_run_by_id",
        AsyncMock(return_value=_hourly_doc()),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        AsyncMock(return_value={"promotion_decisions": [malicious_decision]}),
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        promotion_module,
        "insert_memory_unit",
        AsyncMock(side_effect=_insert_memory_unit),
    )
    monkeypatch.setattr(promotion_module.repository, "upsert_run", AsyncMock())

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date="2026-05-04",
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.succeeded_count == 1
    assert captured_docs
    evidence_refs = captured_docs[0]["evidence_refs"]
    assert evidence_refs[0]["reflection_run_id"] == "hourly-run-1"
    assert evidence_refs[0]["scope_ref"] == "scope-1"
    assert evidence_refs[0]["captured_at"] == "2026-05-04T10:00:00+00:00"
    assert "llm-made-up" not in {
        evidence_ref["reflection_run_id"]
        for evidence_ref in evidence_refs
    }


def _promotion_payload() -> promotion_module.GlobalPromotionPromptPayload:
    """Build a minimal prompt payload fixture."""

    payload = {
        "evaluation_mode": "daily_global_promotion",
        "character_local_date": "2026-05-04",
        "character_time_zone": "Pacific/Auckland",
        "channel_daily_syntheses": [
            {
                "daily_run_id": "daily-run-1",
                "scope_ref": "scope-1",
                "channel_type": "group",
                "character_local_date": "2026-05-04",
                "confidence": "high",
                "day_summary": (
                    "角色确认频道的固定设定应作为公共事实记录，"
                    "同时要求写记忆时区分频道事实和用户画像。"
                ),
                "cross_hour_topics": ["频道固定设定", "记忆撰写规范"],
                "conversation_quality_patterns": ["事实性表达", "用户画像隔离"],
                "privacy_risk_labels": ["无明显风险"],
                "validation_warning_labels": [],
            }
        ],
        "evidence_cards": [
            {
                "evidence_card_id": "evidence-1",
                "source_reflection_run_ids": ["hourly-run-1"],
                "scope_ref": "scope-1",
                "channel_type": "group",
                "character_local_date": "2026-05-04",
                "captured_at": "2026-05-04T10:00:00+00:00",
                "active_character_utterance": "这个频道的固定设定应写成公共频道事实。",
                "sanitized_observation": "角色确认频道固定设定属于公共事实。",
                "supports": ["lore"],
                "private_detail_risk": "low",
            },
            {
                "evidence_card_id": "evidence-2",
                "source_reflection_run_ids": ["hourly-run-1"],
                "scope_ref": "scope-1",
                "channel_type": "group",
                "character_local_date": "2026-05-04",
                "captured_at": "2026-05-04T10:05:00+00:00",
                "active_character_utterance": "以后写记忆时，不要把频道事实写成用户画像。",
                "sanitized_observation": "角色给出未来记忆撰写的行为规则。",
                "supports": ["self_guidance"],
                "private_detail_risk": "low",
            }
        ],
        "promotion_limits": {
            "max_lore": 1,
            "max_self_guidance": 1,
            "max_total_decisions": 2,
        },
        "review_questions": ["哪些内容可晋升？"],
    }
    return payload


def _decision(lane: str) -> promotion_module.ReflectionPromotionDecision:
    """Build a promotion decision fixture."""

    memory_type = promotion_module.PROMOTION_LANE_MEMORY_TYPE[lane]
    decision = {
        "lane": lane,
        "decision": "promote_new",
        "selected_candidate_id": "candidate-1",
        "sanitized_memory_name": "频道规则",
        "sanitized_content": "角色确认群规应保持事实性，不应写成用户画像。",
        "memory_type": memory_type,
        "authority": MemoryAuthority.REFLECTION_PROMOTED,
        "signal_strength": "high",
        "character_agreement": "spoken",
        "boundary_assessment": {
            "verdict": "acceptable",
            "affects_identity_or_boundaries": False,
            "reason": "不涉及身份或亲密边界。",
        },
        "privacy_review": {
            "private_detail_risk": "low",
            "user_details_removed": True,
            "boundary_assessment": "可接受。",
            "reviewer": "automated_llm",
        },
        "evidence_refs": [
            {
                "reflection_run_id": "hourly-run-1",
                "scope_ref": "scope-1",
                "captured_at": "2026-05-04T10:00:00+00:00",
                "source": "reflection_cycle",
            }
        ],
    }
    return decision


def _daily_doc() -> dict:
    """Build a daily-channel run document fixture."""

    doc = {
        "run_id": "daily-run-1",
        "run_kind": "daily_channel",
        "status": "succeeded",
        "prompt_version": "readonly_reflection_v1",
        "attempt_count": 1,
        "scope": {
            "scope_ref": "scope-1",
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
        },
        "character_local_date": "2026-05-04",
        "source_message_refs": [],
        "source_reflection_run_ids": ["hourly-run-1"],
        "output": {
            "day_summary": "角色确认频道固定设定是公共事实，并要求记忆写作避免用户画像化。",
            "cross_hour_topics": ["频道固定设定", "记忆撰写规范"],
            "conversation_quality_patterns": ["事实性表达", "用户画像隔离"],
            "privacy_risks": ["无明显风险"],
            "confidence": "high",
        },
        "promotion_decisions": [],
        "validation_warnings": [],
        "error": "",
    }
    return doc


def _hourly_doc() -> dict:
    """Build an hourly reflection run document fixture."""

    doc = {
        "run_id": "hourly-run-1",
        "run_kind": "hourly_slot",
        "status": "succeeded",
        "prompt_version": "readonly_reflection_v1",
        "attempt_count": 1,
        "scope": {
            "scope_ref": "scope-1",
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
        },
        "character_local_date": "2026-05-04",
        "hour_start": "2026-05-04T10:00:00+00:00",
        "hour_end": "2026-05-04T11:00:00+00:00",
        "source_message_refs": [],
        "source_reflection_run_ids": [],
        "output": {
            "topic_summary": "角色确认频道固定设定属于公共事实。",
            "conversation_quality_feedback": [
                "角色给出未来记忆撰写的行为规则。",
            ],
            "active_character_utterances": [
                "这个频道的固定设定应写成公共频道事实。",
            ],
            "privacy_notes": [],
        },
        "promotion_decisions": [],
        "validation_warnings": [],
        "created_at": "2026-05-04T10:00:00+00:00",
        "updated_at": "2026-05-04T10:00:00+00:00",
    }
    return doc


def _global_run_doc(*, status: str, run_id: str) -> dict:
    """Build a daily global promotion run document fixture."""

    doc = {
        "run_id": run_id,
        "run_kind": "daily_global_promotion",
        "status": status,
        "prompt_version": promotion_module.GLOBAL_PROMOTION_PROMPT_VERSION,
        "attempt_count": 1,
        "scope": {
            "scope_ref": "daily_global",
            "platform": "system",
            "platform_channel_id": "global",
            "channel_type": "system",
        },
        "character_local_date": "2026-05-04",
        "source_message_refs": [],
        "source_reflection_run_ids": ["daily-run-1"],
        "output": {"promotion_decisions": [_decision("lore")]},
        "promotion_decisions": [_decision("lore")],
        "validation_warnings": [],
        "error": "",
        "created_at": "2026-05-05T05:00:00+00:00",
        "updated_at": "2026-05-05T05:00:00+00:00",
    }
    return doc


def _stored_memory_unit(memory_unit_id: str) -> dict:
    """Build a stored memory-unit result fixture."""

    doc = {
        "memory_unit_id": memory_unit_id,
        "lineage_id": memory_unit_id,
        "memory_type": "fact",
        "memory_name": "stored memory",
        "content": "stored content",
        "source_global_user_id": "",
        "source_kind": MemorySourceKind.REFLECTION_INFERRED,
        "authority": MemoryAuthority.REFLECTION_PROMOTED,
        "status": MemoryStatus.ACTIVE,
    }
    return doc
