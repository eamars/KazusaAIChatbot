"""Deterministic contract tests for the scheduled speech content gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    validate_scheduled_speech_semantic_verdict,
)
from kazusa_ai_chatbot.nodes import dialog_agent
from kazusa_ai_chatbot.self_cognition import models
from kazusa_ai_chatbot.self_cognition import worker


def _verdict(**overrides: str) -> dict[str, str]:
    """Build one closed scheduled semantic verdict."""

    verdict = {
        "schema_version": "scheduled_speech_semantic_verdict.v1",
        "time_claim_alignment": "aligned",
        "objective_alignment": "aligned",
        "source_grounding": "current_authority",
        "audience_alignment": "aligned",
        "execution_claim": "aligned",
    }
    verdict.update(overrides)
    return verdict


def test_gate_verdict_schema_has_only_closed_semantic_dimensions() -> None:
    """Every semantic dimension accepts only its closed vocabulary."""

    closed_values = {
        "time_claim_alignment": {
            "aligned",
            "no_claim",
            "premature",
            "contradictory",
            "unavailable",
        },
        "objective_alignment": {
            "aligned",
            "scope_expansion",
            "contradiction",
            "unsupported",
            "unavailable",
        },
        "source_grounding": {
            "current_authority",
            "historical_only",
            "unsupported",
            "unavailable",
        },
        "audience_alignment": {"aligned", "mismatch", "unavailable"},
        "execution_claim": {
            "aligned",
            "premature",
            "false",
            "unavailable",
        },
    }

    for dimension, allowed in closed_values.items():
        for value in allowed:
            validated = validate_scheduled_speech_semantic_verdict(
                _verdict(**{dimension: value})
            )
            assert validated[dimension] == value
        with pytest.raises(CognitionContractError):
            validate_scheduled_speech_semantic_verdict(
                _verdict(**{dimension: "open_value"})
            )

    with pytest.raises(CognitionContractError, match="schema"):
        validate_scheduled_speech_semantic_verdict(
            _verdict(schema_version="open.v1")
        )
    with pytest.raises(CognitionContractError, match="not exact"):
        validate_scheduled_speech_semantic_verdict(
            {
                **_verdict(),
                "free_form_reason": "unsupported",
            }
        )


@pytest.mark.asyncio
async def test_gate_structural_failure_suppresses_after_bounded_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed evaluator output fails closed after the bounded repair cap."""

    invalid_responses = [
        SimpleNamespace(content="not-json"),
        SimpleNamespace(content='{"schema_version": "wrong"}'),
        SimpleNamespace(content="[]"),
    ]
    call_count = 0

    async def fake_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        nonlocal call_count
        del messages, config
        response = invalid_responses[call_count]
        call_count += 1
        return response

    monkeypatch.setattr(
        dialog_agent._scheduled_speech_evaluator_llm,
        "ainvoke",
        fake_ainvoke,
    )

    result = await dialog_agent.evaluate_scheduled_future_speech_content(
        candidate_text="十点整！时间到啦！",
        semantic_objective="在约定时间开始补偿考核。",
        authorized_content_summary="在约定时间开始补偿考核。",
        authorized_detail_refs=[
            {
                "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
                "provenance_role": "current_event",
            }
        ],
        audience_kind="group",
        local_due_datetime="2026-05-10 22:00",
        due_identity_facts={"due_reached": True},
    )

    assert result["status"] == "unavailable"
    assert result["verdict"] is None
    assert result["attempt_count"] == dialog_agent.DIALOG_VERIFIER_ATTEMPT_LIMIT
    assert call_count == dialog_agent.DIALOG_VERIFIER_ATTEMPT_LIMIT

    accepted, gate_codes = worker.evaluate_scheduled_content_gate(
        authority_missing=False,
        authority_invalid=False,
        trigger_identity_ok=True,
        due_reached=True,
        candidate_present=True,
        verdict=result["verdict"],
    )
    assert accepted is False
    assert gate_codes == ["scheduled_evaluator_contract_error"]


def test_scheduled_gate_disposition_constants_are_closed() -> None:
    """Gate disposition metadata stays within the closed model contract."""

    assert models.SCHEDULED_GATE_DISPOSITION_VALUES == {
        "accepted",
        "suppressed",
        "not_evaluated",
    }
    gate_result: models.SelfCognitionScheduledGateResult = {
        "schema_version": models.SCHEDULED_GATE_RESULT_SCHEMA_VERSION,
        "disposition": models.SCHEDULED_GATE_DISPOSITION_SUPPRESSED,
        "gate_codes": ["scheduled_objective_mismatch"],
        "evaluator_attempt_count": 1,
    }
    assert gate_result["gate_codes"] == ["scheduled_objective_mismatch"]
