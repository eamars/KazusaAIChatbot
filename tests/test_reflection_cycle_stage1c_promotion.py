"""Deterministic tests for global reflection promotion."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.memory_evolution.models import (
    MemoryAuthority,
    MemorySourceKind,
    MemoryStatus,
)
from kazusa_ai_chatbot.reflection_cycle import promotion as promotion_module
from kazusa_ai_chatbot.reflection_cycle import repository, selector


@pytest.fixture(autouse=True)
def _mock_character_profile(monkeypatch) -> None:
    """Give promotion prompts a deterministic active character profile name."""

    monkeypatch.setattr(
        promotion_module,
        "get_character_profile",
        AsyncMock(return_value={"name": "杏山千纱 (Kyōyama Kazusa)"}),
    )


@pytest.fixture(autouse=True)
def _mock_promotion_review(monkeypatch) -> None:
    """Give deterministic promotion runs an independent accepted review."""

    async def _review(prompt):
        payload = json.loads(prompt.human_prompt)
        reviews = [
            {
                "selected_candidate_id": candidate["selected_candidate_id"],
                "decision": "accept",
                **_review_certificate(),
            }
            for candidate in payload["candidates"]
        ]
        return {"reviews": reviews}

    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_review_llm",
        _review,
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
        "global_applicability": "global",
        "target_specific_meaning_removed": True,
        "affects_identity_or_boundaries": True,
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


def test_promotion_evidence_carries_source_privacy_notes_without_assuming_low_risk() -> None:
    """Missing source privacy assessment remains explicitly unreviewed."""

    hourly_doc = _hourly_doc()
    hourly_doc["output"]["privacy_notes"] = ["来源隐私评估未确定"]

    cards = promotion_module._evidence_cards_from_hourly_doc(hourly_doc)

    assert cards[0]["private_detail_risk"] == "unreviewed"
    assert cards[0]["source_privacy_notes"] == ["来源隐私评估未确定"]


def test_hourly_evidence_cards_respect_serialized_cap_and_preserve_privacy_evidence() -> None:
    """Hourly cards stay bounded without discarding source privacy evidence."""

    for privacy_notes in (
        [],
        ["来源隐私评估未确定"],
        ["来源隐私说明-" + ("x" * 120)] * 3,
    ):
        hourly_doc = _canonical_max_hourly_doc()
        hourly_doc["output"]["privacy_notes"] = privacy_notes

        cards = promotion_module._evidence_cards_from_hourly_doc(hourly_doc)

        assert cards
        card = cards[0]
        serialized_card = json.dumps(
            card,
            ensure_ascii=False,
            sort_keys=True,
        )
        assert len(serialized_card) <= (
            promotion_module.PROMOTION_MAX_EVIDENCE_CARD_CHARS
        )
        source_run_id = hourly_doc["run_id"]
        assert card["evidence_card_id"] == f"evidence_{source_run_id}"
        assert card["source_reflection_run_ids"] == [source_run_id]
        assert card["scope_ref"] == hourly_doc["scope"]["scope_ref"]
        assert card["channel_type"] == "group"
        assert card["supports"] == ["lore", "self_guidance"]
        if privacy_notes:
            assert card["source_privacy_notes"]
            assert any(
                len(note.replace("...", "")) >= 4
                for note in card["source_privacy_notes"]
            )


def test_channel_cards_respect_serialized_cap_for_readable_fields() -> None:
    """The shared card cap also covers daily-channel readable fields."""

    daily_doc = _daily_doc()
    daily_doc["output"]["day_summary"] = "摘要" * 300
    daily_doc["output"]["cross_hour_topics"] = ["主题" * 60] * 3
    daily_doc["output"]["conversation_quality_patterns"] = [
        "模式" * 60
    ] * 3
    daily_doc["output"]["privacy_risks"] = ["风险" * 60] * 3
    daily_doc["validation_warnings"] = ["警告" * 60] * 3

    cards = promotion_module._channel_daily_cards([daily_doc])

    assert cards
    card = cards[0]
    assert len(json.dumps(card, ensure_ascii=False, sort_keys=True)) <= (
        promotion_module.PROMOTION_MAX_CHANNEL_CARD_CHARS
    )
    assert card["daily_run_id"] == daily_doc["run_id"]
    assert card["confidence"] == "high"
    assert card["day_summary"]
    assert card["cross_hour_topics"]


def test_oversized_untrimmable_card_fails_closed() -> None:
    """An envelope that cannot fit without changing identity fails closed."""

    oversized_card = {
        "evidence_card_id": "evidence_" + ("x" * 800),
        "source_reflection_run_ids": ["reflection_run_1"],
        "scope_ref": "scope_1",
        "channel_type": "group",
        "character_local_date": "2026-05-04",
        "captured_at": "2026-05-04 22:00",
        "active_character_utterance": "可读文本",
        "sanitized_observation": "可读观察",
        "supports": ["lore", "self_guidance"],
        "source_privacy_notes": ["来源说明"],
        "private_detail_risk": "unreviewed",
    }

    with pytest.raises(ValueError, match="cannot fit"):
        promotion_module._cap_serialized_card(
            oversized_card,
            promotion_module.PROMOTION_MAX_EVIDENCE_CARD_CHARS,
        )


@pytest.mark.asyncio
async def test_live_harness_persists_artifact_before_review_error(
    monkeypatch,
    tmp_path,
) -> None:
    """The live harness writes raw stages before propagating review errors."""

    from tests import test_reflection_cycle_stage1c_promotion_live_llm as live_module

    promoter_response = SimpleNamespace(
        content=json.dumps(
            {"promotion_decisions": [_decision("lore")]},
            ensure_ascii=False,
        ),
    )
    reviewer_response = SimpleNamespace(
        content=json.dumps({"reviews": []}, ensure_ascii=False),
    )
    monkeypatch.setattr(
        live_module._llm_interface,
        "ainvoke",
        AsyncMock(return_value=promoter_response),
    )
    monkeypatch.setattr(
        promotion_module._global_promotion_review_llm,
        "ainvoke",
        AsyncMock(return_value=reviewer_response),
    )
    captured: dict[str, Any] = {}

    def _write_trace(test_name, case_id, payload):
        captured["test_name"] = test_name
        captured["case_id"] = case_id
        captured["payload"] = payload
        trace_path = tmp_path / "malformed-review.json"
        trace_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return trace_path

    monkeypatch.setattr(live_module, "write_llm_trace", _write_trace)

    with pytest.raises(promotion_module.GlobalPromotionReviewContractError):
        await live_module._run_case(
            "harness_malformed_review",
            _promotion_payload(),
        )

    artifact = captured["payload"]
    assert artifact["rendered_prompt"]
    assert artifact["promoter_prompt"]["system_prompt"] == (
        artifact["rendered_prompt"]
    )
    assert artifact["promoter_prompt"]["human_prompt"]
    assert artifact["input_payload"]["evidence_cards"]
    assert artifact["raw_output"]
    assert artifact["parsed_output"]["promotion_decisions"]
    assert artifact["review_raw_output"]
    assert artifact["review_output"] == {"reviews": []}
    assert artifact["review_call_count"] == 1
    assert artifact["final_exception"]["type"] == (
        "GlobalPromotionReviewContractError"
    )
    assert artifact["final_exception"]["stage"] == "review_contract"
    assert (tmp_path / "malformed-review.json").exists()


def test_reviewer_prompt_max_shape_stays_within_budget_without_truncation_warning() -> None:
    """The independent reviewer receives bounded, lane-matched evidence."""

    payload = _promotion_payload()
    evidence_cards = []
    for index in range(promotion_module.PROMOTION_MAX_EVIDENCE_CARDS):
        scope_ref = selector.build_scope_ref(
            "qq",
            f"review-channel-{index}",
            "group",
        )
        source_run_id = repository.hourly_run_id(
            scope_ref=scope_ref,
            hour_start=(
                f"2026-05-{4 + index // 24:02d}T"
                f"{index % 24:02d}:00:00+00:00"
            ),
        )
        evidence_cards.append(
            {
                "evidence_card_id": f"evidence_{source_run_id}",
                "source_reflection_run_ids": [source_run_id],
                "scope_ref": scope_ref,
                "channel_type": "group",
                "character_local_date": "2026-05-04",
                "captured_at": "2026-05-04 22:00",
                "active_character_utterance": "u" * 180,
                "sanitized_observation": "o" * 180,
                "supports": ["lore", "self_guidance"],
                "source_privacy_notes": ["p" * 120] * 3,
                "private_detail_risk": "unreviewed",
            },
        )
    payload["evidence_cards"] = evidence_cards
    candidates = []
    for index, lane in enumerate(("lore", "self_guidance")):
        decision = _decision(lane)
        decision["selected_candidate_id"] = f"candidate-{index}"
        decision["sanitized_memory_name"] = "n" * 300
        decision["sanitized_content"] = "m" * 1000
        candidates.append(decision)

    review_payload = promotion_module._promotion_review_payload(
        candidates,
        payload,
    )
    review_prompt = promotion_module.build_global_promotion_review_prompt(
        review_payload,
    )

    assert review_prompt.prompt_chars <= (
        promotion_module.GLOBAL_PROMOTION_REVIEW_PROMPT_MAX_CHARS
    )
    assert review_prompt.validation_warnings == []
    assert len(review_payload["evidence_cards"]) <= (
        promotion_module.PROMOTION_MAX_REVIEW_EVIDENCE_CARDS
    )
    assert all(
        set(card["supports"]) & {"lore", "self_guidance"}
        for card in review_payload["evidence_cards"]
    )
    assert {"lore", "self_guidance"} <= {
        lane
        for card in review_payload["evidence_cards"]
        for lane in card["supports"]
    }
    assert [
        candidate["sanitized_content"]
        for candidate in review_payload["candidates"]
    ] == ["m" * 1000, "m" * 1000]


@pytest.mark.asyncio
async def test_reflection_memory_write_requires_independent_global_scope_review(
    monkeypatch,
) -> None:
    """A promoter certificate alone cannot admit a global memory write."""

    decision = _decision("self_guidance")
    reviewer_certificate = _review_certificate(
        target_specific_meaning_removed=False,
    )

    async def _review_llm(prompt):
        payload = json.loads(prompt.human_prompt)
        candidate_id = payload["candidates"][0]["selected_candidate_id"]
        return {
            "reviews": [{
                "selected_candidate_id": candidate_id,
                "decision": "reject",
                **reviewer_certificate,
            }],
        }

    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_review_llm",
        _review_llm,
    )

    reviewed, warnings = await promotion_module._review_promotion_candidates(
        decisions=[decision],
        payload=_promotion_payload(),
    )

    assert warnings == []
    assert reviewed[0]["review_decision"] == "reject"
    assert reviewed[0]["review_admitted"] is False
    assert reviewed[0]["privacy_review"]["global_applicability"] == "global"
    assert reviewed[0]["reviewer_privacy_review"] == reviewer_certificate


@pytest.mark.asyncio
async def test_identity_or_boundary_candidate_never_writes_self_guidance(
    monkeypatch,
) -> None:
    """Reviewer identity or boundary scope blocks self-guidance persistence."""

    decision = _decision("self_guidance")
    reviewer_certificate = _review_certificate(
        affects_identity_or_boundaries=True,
    )

    async def _review_llm(prompt):
        payload = json.loads(prompt.human_prompt)
        candidate_id = payload["candidates"][0]["selected_candidate_id"]
        return {
            "reviews": [{
                "selected_candidate_id": candidate_id,
                "decision": "accept",
                **reviewer_certificate,
            }],
        }

    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_review_llm",
        _review_llm,
    )
    insert_mock = AsyncMock()
    monkeypatch.setattr(promotion_module, "insert_memory_unit", insert_mock)

    reviewed, warnings = await promotion_module._review_promotion_candidates(
        decisions=[decision],
        payload=_promotion_payload(),
    )
    write_result = await promotion_module._write_validated_promotion_decisions(
        decisions=reviewed,
        character_local_date="2026-05-04",
        global_run_id="global-run-1",
    )

    assert warnings == []
    assert write_result["mutations"] == []
    assert reviewed[0]["review_decision"] == "accept"
    assert reviewed[0]["review_admitted"] is False
    insert_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_promotion_review_is_candidate_gated_and_batched(monkeypatch) -> None:
    """The reviewer runs once for mutating candidates and zero times otherwise."""

    review_calls = 0

    async def _review_llm(prompt):
        nonlocal review_calls
        review_calls += 1
        payload = json.loads(prompt.human_prompt)
        return {
            "reviews": [
                {
                    "selected_candidate_id": candidate[
                        "selected_candidate_id"
                    ],
                    "decision": "accept",
                    **_review_certificate(),
                }
                for candidate in payload["candidates"]
            ],
        }

    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_review_llm",
        _review_llm,
    )

    no_write_decisions = [_decision("lore")]
    no_write_decisions[0]["decision"] = "no_action"
    reviewed_no_write, no_write_warnings = (
        await promotion_module._review_promotion_candidates(
            decisions=no_write_decisions,
            payload=_promotion_payload(),
        )
    )
    assert review_calls == 0
    assert no_write_warnings == []
    assert reviewed_no_write[0]["decision"] == "no_action"

    second_decision = _decision("self_guidance")
    second_decision["selected_candidate_id"] = "candidate-2"
    reviewed, warnings = await promotion_module._review_promotion_candidates(
        decisions=[_decision("lore"), second_decision],
        payload=_promotion_payload(),
    )
    assert review_calls == 1
    assert warnings == []
    assert all(decision["review_admitted"] for decision in reviewed)


@pytest.mark.asyncio
async def test_promotion_reviewer_input_excludes_promoter_certificate_and_rewrite_fields(
    monkeypatch,
) -> None:
    """Independent review receives meaning and evidence without judgments."""

    captured: dict[str, Any] = {}

    async def _review_llm(prompt):
        captured["payload"] = json.loads(prompt.human_prompt)
        candidate_id = captured["payload"]["candidates"][0][
            "selected_candidate_id"
        ]
        return {
            "reviews": [{
                "selected_candidate_id": candidate_id,
                "decision": "accept",
                **_review_certificate(),
            }],
        }

    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_review_llm",
        _review_llm,
    )
    decision = _decision("lore")
    original_meaning = {
        key: decision[key]
        for key in (
            "lane",
            "memory_type",
            "sanitized_memory_name",
            "sanitized_content",
        )
    }

    reviewed, warnings = await promotion_module._review_promotion_candidates(
        decisions=[decision],
        payload=_promotion_payload(),
    )

    candidate_payload = captured["payload"]["candidates"][0]
    assert set(candidate_payload) == {
        "selected_candidate_id",
        "lane",
        "memory_type",
        "sanitized_memory_name",
        "sanitized_content",
    }
    serialized_payload = json.dumps(captured["payload"], ensure_ascii=False)
    assert "privacy_review" not in serialized_payload
    assert "boundary_assessment" not in serialized_payload
    assert "global_applicability" not in serialized_payload
    assert warnings == []
    assert {
        key: reviewed[0][key]
        for key in original_meaning
    } == original_meaning


def test_promotion_review_contract_rejects_rewrite_fields_and_unknown_ids() -> None:
    """The closed reviewer response cannot rewrite or add candidates."""

    candidate = _decision("lore")
    review = {
        "selected_candidate_id": candidate["selected_candidate_id"],
        "decision": "accept",
        **_review_certificate(),
        "sanitized_content": "rewritten content",
    }

    errors = promotion_module._promotion_review_contract_errors(
        {"reviews": [review]},
        [candidate],
    )

    assert any("invalid key set" in error for error in errors)


@pytest.mark.asyncio
async def test_malformed_promotion_review_fails_closed_before_similarity_or_write(
    monkeypatch,
) -> None:
    """Malformed review coverage blocks similarity search and persistence."""

    persisted: list[dict[str, Any]] = []

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
        AsyncMock(return_value=_hourly_doc()),
    )
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_llm",
        AsyncMock(return_value={"promotion_decisions": [_decision("lore")]}),
    )
    reviewer_mock = AsyncMock(return_value={"reviews": []})
    monkeypatch.setattr(
        promotion_module,
        "run_global_promotion_review_llm",
        reviewer_mock,
    )
    find_mock = AsyncMock(return_value=[])
    insert_mock = AsyncMock()
    monkeypatch.setattr(promotion_module, "find_active_memory_units", find_mock)
    monkeypatch.setattr(promotion_module, "insert_memory_unit", insert_mock)
    monkeypatch.setattr(promotion_module.repository, "upsert_run", _upsert)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date="2026-05-04",
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.failed_count == 1
    reviewer_mock.assert_awaited_once()
    find_mock.assert_not_awaited()
    insert_mock.assert_not_awaited()
    assert persisted[-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_promotion_persists_actual_reviewer_certificate_and_audit(
    monkeypatch,
) -> None:
    """Writes store reviewer evidence while retaining both audit certificates."""

    decision = _decision("lore")
    reviewer_certificate = _review_certificate(
        reason="最终审阅确认候选含义已去除特定对象影响。",
    )
    persisted: list[dict[str, Any]] = []
    captured_docs: list[dict[str, Any]] = []

    async def _review_llm(prompt):
        payload = json.loads(prompt.human_prompt)
        return {
            "reviews": [{
                "selected_candidate_id": payload["candidates"][0][
                    "selected_candidate_id"
                ],
                "decision": "accept",
                **reviewer_certificate,
            }],
        }

    async def _upsert(document):
        persisted.append(document)

    async def _insert(*, document):
        captured_docs.append(document)
        return _stored_memory_unit("reviewed-unit")

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
        "run_global_promotion_review_llm",
        _review_llm,
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(promotion_module, "insert_memory_unit", _insert)
    monkeypatch.setattr(promotion_module.repository, "upsert_run", _upsert)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date="2026-05-04",
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.succeeded_count == 1
    assert captured_docs[0]["privacy_review"]["boundary_assessment"] == (
        reviewer_certificate["reason"]
    )
    final_decision = persisted[-1]["promotion_decisions"][0]
    assert final_decision["promoter_privacy_review"][
        "global_applicability"
    ] == "global"
    assert final_decision["reviewer_privacy_review"] == reviewer_certificate
    assert final_decision["review_decision"] == "accept"


def test_global_promotion_prompt_version_cuts_over_to_independent_review() -> None:
    """The canonical version identifies the new promoter/reviewer contract."""

    assert promotion_module.GLOBAL_PROMOTION_PROMPT_VERSION != (
        "reflection_global_promotion_v1"
    )
    assert promotion_module.GLOBAL_PROMOTION_PROMPT_VERSION.endswith("_v2")
    assert promotion_module.GLOBAL_PROMOTION_REVIEW_PROMPT_VERSION.endswith(
        "_v1"
    )


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
async def test_invalid_promotion_contract_requests_complete_replacement(
    monkeypatch,
) -> None:
    """A wrong lane memory type must regenerate before any memory write."""

    invalid = _decision("self_guidance")
    invalid["memory_type"] = "fact"
    valid = _decision("self_guidance")
    persisted = []

    async def _upsert(document):
        persisted.append(document)

    run_promotion_llm = AsyncMock(side_effect=[
        {"promotion_decisions": [invalid]},
        {"promotion_decisions": [valid]},
    ])
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
        run_promotion_llm,
    )
    monkeypatch.setattr(
        promotion_module,
        "find_active_memory_units",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        promotion_module,
        "insert_memory_unit",
        AsyncMock(return_value=_stored_memory_unit("unit-repaired")),
    )
    monkeypatch.setattr(promotion_module.repository, "upsert_run", _upsert)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date="2026-05-04",
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.succeeded_count == 1
    assert run_promotion_llm.await_count == 2
    assert persisted[-1]["attempt_count"] == 2
    assert persisted[-1]["status"] == "succeeded"
    assert persisted[-1]["promotion_decisions"][0]["memory_type"] == (
        "defense_rule"
    )


@pytest.mark.asyncio
async def test_invalid_promotion_contract_exhaustion_fails_without_write(
    monkeypatch,
) -> None:
    """Three invalid replacements persist a typed failure and no candidate."""

    invalid = _decision("self_guidance")
    invalid["memory_type"] = "fact"
    persisted = []

    async def _upsert(document):
        persisted.append(document)

    run_promotion_llm = AsyncMock(
        return_value={"promotion_decisions": [invalid]},
    )
    insert_memory = AsyncMock()
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
        run_promotion_llm,
    )
    monkeypatch.setattr(promotion_module, "insert_memory_unit", insert_memory)
    monkeypatch.setattr(promotion_module.repository, "upsert_run", _upsert)

    result = await promotion_module.run_global_reflection_promotion(
        character_local_date="2026-05-04",
        dry_run=False,
        enable_memory_writes=True,
    )

    assert result.failed_count == 1
    assert run_promotion_llm.await_count == 3
    assert persisted[-1]["attempt_count"] == 3
    assert persisted[-1]["status"] == "failed"
    assert "contract" in persisted[-1]["error"].lower()
    insert_memory.assert_not_awaited()


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
            "captured_at": "2026-05-04 12:00",
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
    assert evidence_refs[0]["captured_at"] == "2026-05-04 22:00"
    assert "llm-made-up" not in {
        evidence_ref["reflection_run_id"]
        for evidence_ref in evidence_refs
    }


def _promotion_payload() -> promotion_module.GlobalPromotionPromptPayload:
    """Build a minimal prompt payload fixture."""

    payload = {
        "evaluation_mode": "daily_global_promotion",
        "character_local_date": "2026-05-04",
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
                "captured_at": "2026-05-04 22:00",
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
                "captured_at": "2026-05-04 22:05",
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
            "global_applicability": "global",
            "target_specific_meaning_removed": True,
            "affects_identity_or_boundaries": False,
            "private_detail_risk": "low",
            "user_details_removed": True,
            "boundary_assessment": "可接受。",
            "reviewer": "automated_llm",
        },
        "evidence_refs": [
            {
                "reflection_run_id": "hourly-run-1",
                "scope_ref": "scope-1",
                "captured_at": "2026-05-04 22:00",
                "source": "reflection_cycle",
            }
        ],
    }
    return decision


def _review_certificate(**overrides: Any) -> dict[str, Any]:
    """Build an independent reviewer certificate fixture."""

    certificate: dict[str, Any] = {
        "global_applicability": "global",
        "target_specific_meaning_removed": True,
        "affects_identity_or_boundaries": False,
        "private_detail_risk": "low",
        "user_details_removed": True,
        "reason": "独立审阅确认范围与隐私条件满足。",
    }
    certificate.update(overrides)
    return certificate


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


def _canonical_max_hourly_doc() -> dict:
    """Build a canonical-shape hourly document at readable-field maxima."""

    scope_ref = selector.build_scope_ref(
        "qq",
        "canonical-channel-480386272",
        "group",
    )
    run_id = repository.hourly_run_id(
        scope_ref=scope_ref,
        hour_start="2026-05-04T10:00:00+00:00",
    )
    doc = _hourly_doc()
    doc["run_id"] = run_id
    doc["scope"]["scope_ref"] = scope_ref
    doc["scope"]["platform_channel_id"] = "canonical-channel-480386272"
    doc["output"]["topic_summary"] = "主题" * 90
    doc["output"]["conversation_quality_feedback"] = [
        "反馈" * 60,
        "质量" * 60,
    ]
    doc["output"]["active_character_utterances"] = ["发言" * 90]
    doc["output"]["privacy_notes"] = ["隐私说明" * 60] * 3
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
