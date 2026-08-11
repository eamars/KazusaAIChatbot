"""Replay observed semantic families through the live LLM boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import time_ns
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContextLimitError,
    CognitionExecutionError,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    reset_validation_capture,
    validation_capture_snapshot,
    write_diagnostic_artifact,
    write_validation_capture,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _cognition_elapsed_seconds,
    _episode_updated_at,
    _fact_without_producer,
    _native_relationship_context,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    SEMANTIC_APPRAISAL_PROMPT,
    SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP,
    _appraise_semantic_item,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_state_update,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_TRACE_STEPS_PATH = (
    _ROOT
    / "test_artifacts"
    / "diagnostics"
    / "llm_trace_steps_2026-08-04.json"
)
_CURRENT_RUN_TRACE_PATH = (
    _ROOT
    / "test_artifacts"
    / "diagnostics"
    / (
        "cognition_v2_run_llmtrace_fab989d622da48a89c6e5566e2121251_"
        "20260805.json"
    )
)
_A1A573_TRACE_PATH = (
    _ROOT
    / "test_artifacts"
    / "diagnostics"
    / "cognition_trace_a1a573b590a3494786c4edebdee55342.json"
)
_ARTIFACT_ROOT = (
    _ROOT
    / "test_artifacts"
    / "cognition_core_v2_trace_failure_modes"
    / "semantic"
)
_RAW_ARTIFACT_ROOT = _ARTIFACT_ROOT / "raw"

_SEMANTIC_CASES: dict[str, dict[str, object]] = {
    "semantic_delta_path_not_owned": {
        "trace_id": "llmtrace_198a113af3b843e4b334345ff1c0620f",
        "stage_name": (
            "semantic_appraisal.q:relationship_social.item_1"
        ),
        "attempt_index": 1,
        "question_id": "q:relationship_social",
        "expected_error": "semantic delta path",
    },
    "selected_evidence_unknown_handle": {
        "trace_id": "llmtrace_4d896d405f8140a6ac4b731cc8d1ad4d",
        "stage_name": "semantic_appraisal.q:relationship_social.item_1",
        "attempt_index": 1,
        "question_id": "q:relationship_social",
        "expected_error": "selected evidence contains",
    },
    "terminal_event_transition_rejected": {
        "trace_id": "llmtrace_6e00180ecaf54289a5b21153a3ff3d6c",
        "stage_name": (
            "semantic_appraisal.q:goal_threat_outcome.item_1"
        ),
        "attempt_index": 1,
        "question_id": "q:goal_threat_outcome",
        "expected_error": "terminal event cannot transition",
    },
    "candidate_origin_evidence_missing": {
        "trace_id": "llmtrace_7f6bf91853df403d95ec8c4e767e22ef",
        "stage_name": (
            "semantic_appraisal.q:epistemic_comparison_memory.item_1.repair"
        ),
        "attempt_index": 2,
        "question_id": "q:epistemic_comparison_memory",
        "expected_error": "causal candidates must cite originating evidence",
    },
    "semantic_role_value_invalid": {
        "trace_id": "llmtrace_f75c08f10903487e881cbdceb7a94d00",
        "stage_name": "semantic_appraisal.q:moral_identity.item_1",
        "attempt_index": 1,
        "question_id": "q:moral_identity",
        "expected_error": "semantic role value is invalid",
    },
    "current_run_event_agency_role_value_invalid": {
        "trace_id": "llmtrace_fab989d622da48a89c6e5566e2121251",
        "trace_path": _CURRENT_RUN_TRACE_PATH,
        "stage_name": "semantic_appraisal.q:event_agency.item_1",
        "attempt_index": 1,
        "question_id": "q:event_agency",
        "expected_error": "semantic role value is invalid",
    },
    "a1a573_goal_threat_unowned_knowledge_gap_path": {
        "trace_id": "llmtrace_93482f08e4a74aa5af90adc6e6f5918a",
        "trace_path": _A1A573_TRACE_PATH,
        "stage_name": (
            "semantic_appraisal.q:goal_threat_outcome.item_1"
        ),
        "attempt_index": 1,
        "question_id": "q:goal_threat_outcome",
        "expected_error": "semantic delta path",
        "replay_historical_candidate": True,
        "require_repair_call": True,
    },
    "resolved_knowledge_gap_transition_rejected": {
        "trace_id": "llmtrace_b2935ecd4361456a9bfb10deeaa790b1",
        "stage_name": (
            "semantic_appraisal.q:goal_threat_outcome.item_1"
        ),
        "attempt_index": 1,
        "question_id": "q:goal_threat_outcome",
        "expected_error": "resolved knowledge gap cannot transition",
    },
    "selected_roles_unknown_handle": {
        "trace_id": "llmtrace_31ae8b9c1d9e4afcb0b7e7bcd33c72e0",
        "stage_name": "semantic_appraisal.q:event_agency.item_1",
        "attempt_index": 1,
        "question_id": "q:event_agency",
        "expected_error": "selected roles contains unknown handles",
    },
    "semantic_proposition_subject_kind_mismatch": {
        "trace_id": "llmtrace_3bfd68d36df6471daa96f197ca078fd9",
        "stage_name": (
            "semantic_appraisal.q:goal_threat_outcome.item_1.repair"
        ),
        "attempt_index": 2,
        "question_id": "q:goal_threat_outcome",
        "expected_error": (
            "semantic proposition kind requires subject kind"
        ),
    },
    "semantic_proposition_object_handle_not_permitted": {
        "trace_id": "llmtrace_2a4a4b5f1d6c492cb69800b7f7558f1c",
        "stage_name": "semantic_appraisal.q:relationship_social.item_1",
        "attempt_index": 1,
        "question_id": "q:relationship_social",
        "expected_error": (
            "semantic proposition object handle is not permitted"
        ),
    },
    "delta_reason_invalid": {
        "trace_id": "llmtrace_90bf8dda523c4f54befc4480991aafbd",
        "stage_name": (
            "semantic_appraisal.q:epistemic_comparison_memory.item_1"
        ),
        "attempt_index": 1,
        "question_id": "q:epistemic_comparison_memory",
        "expected_error": "reason must be non-empty text up to 300 characters",
    },
    "semantic_delta_type_invalid": {
        "trace_id": "llmtrace_9362f66e63af48be85be587c9d858808",
        "stage_name": (
            "semantic_appraisal.q:goal_threat_outcome.item_1.repair"
        ),
        "attempt_index": 2,
        "question_id": "q:goal_threat_outcome",
        "expected_error": (
            "semantic delta must be a JSON integer from -40 through 40"
        ),
    },
    "semantic_micro_appraisal_fields_not_exact": {
        "trace_id": "llmtrace_3faf12a8ac244089b5c538671e6e79dd",
        "stage_name": "semantic_appraisal.q:existential_drive.item_2",
        "attempt_index": 1,
        "question_id": "q:existential_drive",
        "expected_error": (
            "semantic micro-appraisal fields must be exactly question_id"
        ),
    },
}


class _HistoricalFirstThenLiveLLM:
    """Replay one preserved invalid candidate before a real repair call."""

    def __init__(self, delegate: Any, historical_response: str) -> None:
        self.delegate = delegate
        self.historical_response = historical_response
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object,
        **kwargs: object,
    ) -> Any:
        """Return the preserved first candidate, then call the real model."""

        if not self.calls:
            response = SimpleNamespace(content=self.historical_response)
            response_source = "preserved_historical_candidate"
        else:
            response = await self.delegate.ainvoke(
                messages,
                *args,
                config=config,
                **kwargs,
            )
            response_source = "live_model"
        self.calls.append({
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(response.content),
            "response_source": response_source,
        })
        return response


def _load_case_capsule(
    case: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Load one preserved capsule and its representative failed attempt."""

    trace_path = case.get("trace_path", _TRACE_STEPS_PATH)
    if not isinstance(trace_path, Path):
        trace_path = _TRACE_STEPS_PATH
    if not trace_path.exists():
        raise AssertionError(
            f"captured trace export is missing: {trace_path}"
        )
    export = json.loads(trace_path.read_text(encoding="utf-8"))
    capsules: list[Mapping[str, Any]] = []
    documents = export.get("documents")
    if isinstance(documents, list):
        for row in documents:
            if isinstance(row, Mapping) and isinstance(
                row.get("capsule"),
                Mapping,
            ):
                capsules.append(row["capsule"])
    direct_capsules = export.get("cognition_failure_capsules")
    if isinstance(direct_capsules, list):
        capsules.extend(
            capsule
            for capsule in direct_capsules
            if isinstance(capsule, Mapping)
        )
    for capsule in capsules:
        if capsule.get("trace_id") != case["trace_id"]:
            continue
        for attempt in capsule.get("attempts", []):
            if not isinstance(attempt, dict):
                continue
            if attempt.get("stage_name") != case["stage_name"]:
                continue
            if attempt.get("attempt_index") != case["attempt_index"]:
                continue
            validation_error = str(attempt.get("validation_error") or "")
            if not validation_error:
                continue
            if str(case["expected_error"]) not in validation_error:
                continue
            base_stage_name = str(case["stage_name"]).split(
                ".repair",
                maxsplit=1,
            )[0]
            for base_attempt in capsule.get("attempts", []):
                if not isinstance(base_attempt, dict):
                    continue
                if base_attempt.get("stage_name") != base_stage_name:
                    continue
                messages = base_attempt.get("messages")
                if not isinstance(messages, list):
                    continue
                system_messages = [
                    message.get("content")
                    for message in messages
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "system"
                    )
                ]
                human_messages = [
                    message.get("content")
                    for message in messages
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "human"
                    )
                ]
                if (
                    system_messages
                    and isinstance(system_messages[0], str)
                    and human_messages
                    and isinstance(human_messages[0], str)
                ):
                    return (
                        capsule,
                        attempt,
                        system_messages[0],
                        human_messages[0],
                    )
    raise AssertionError(
        "representative semantic failure attempt is missing from export"
    )


def _build_appraisal_context(
    input_payload: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    list[Mapping[str, Any]],
]:
    """Rebuild the facade's deterministic pre-appraisal context only."""

    payload = validate_cognition_core_input(input_payload)
    previous_state = validate_cognition_state(payload["mutable_state"])
    updated_at = _episode_updated_at(payload["episode"])
    elapsed_seconds = _cognition_elapsed_seconds(previous_state, updated_at)
    fact_pairs = [
        (fact["producer"], _fact_without_producer(fact))
        for fact in payload["direct_facts"]
    ]
    relationship_context = _native_relationship_context(
        payload.get("relationship_context"),
    )
    preliminary_state = apply_state_update(
        previous_state,
        direct_facts=fact_pairs,
        elapsed_seconds=elapsed_seconds,
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=relationship_context,
    )
    preliminary_state = create_deterministic_goals(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        relationship_context=relationship_context,
        evidence=payload["evidence"],
        updated_at=updated_at,
    )
    preliminary_state = validate_cognition_state(preliminary_state)
    projection = project_state_for_prompt(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context",
        ),
        evidence=payload["evidence"],
    )
    questions = plan_semantic_questions(
        payload["evidence"],
        preliminary_state,
        projection.handle_to_ref,
    )
    return payload, preliminary_state, projection, questions


def _normalize_archival_input(
    input_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore the trusted field absent from a subset of early captures."""

    normalized = deepcopy(dict(input_payload))
    evidence = normalized.get("evidence")
    if not isinstance(evidence, list):
        return normalized
    for row in evidence:
        if not isinstance(row, dict):
            continue
        evidence_ref = row.get("evidence_ref")
        if (
            isinstance(evidence_ref, dict)
            and evidence_ref.get("source_kind") == "promoted_memory"
            and "memory_scope" not in row
        ):
            row["memory_scope"] = "shared_character_or_world"
    return normalized


async def _run_semantic_case(case_id: str) -> None:
    """Run one case through current generation and validation."""

    case = _SEMANTIC_CASES[case_id]
    (
        capsule,
        representative,
        historical_system_prompt,
        historical_payload_text,
    ) = _load_case_capsule(case)
    input_payload = capsule.get("input_payload")
    if not isinstance(input_payload, dict):
        raise AssertionError("failure capsule input payload is not an object")
    normalized_input = _normalize_archival_input(input_payload)
    payload, preliminary_state, projection, questions = (
        _build_appraisal_context(normalized_input)
    )
    question_id = str(case["question_id"])
    matching_questions = [
        question
        for question in questions
        if question.get("question_id") == question_id
    ]
    assert len(matching_questions) == 1
    reset_validation_capture(
        f"trace_failure_modes_{case_id}_{time_ns()}"
    )
    caught_error: CognitionExecutionError | CognitionContextLimitError | None = (
        None
    )
    base_services = build_cognition_core_services()
    historical_llm: _HistoricalFirstThenLiveLLM | None = None
    services = base_services
    if case.get("replay_historical_candidate"):
        historical_response = representative.get("raw_response_text")
        if not isinstance(historical_response, str):
            raise AssertionError(
                "historical semantic candidate is not text"
            )
        historical_llm = _HistoricalFirstThenLiveLLM(
            base_services.llm,
            historical_response,
        )
        services = replace(base_services, llm=historical_llm)
    question = matching_questions[0]
    evidence_by_handle: dict[str, dict[str, str]] = {}
    for row in payload["evidence"]:
        if row["evidence_handle"] not in question["evidence_handles"]:
            continue
        projected_row: dict[str, str] = {
            "handle": row["evidence_handle"],
            "semantic_text": row["semantic_text"],
            "source_kind": row["evidence_ref"]["source_kind"],
        }
        if "memory_scope" in row:
            projected_row["memory_scope"] = row["memory_scope"]
        evidence_by_handle[row["evidence_handle"]] = projected_row
    config = getattr(
        services,
        f"appraisal_{question['question_kind']}_config",
    )
    historical_payload = json.loads(historical_payload_text)
    historical_question = historical_payload["question"]
    item_index = int(
        str(case["stage_name"]).split(".item_", maxsplit=1)[1].split(
        ".",
        maxsplit=1,
        )[0]
    )
    try:
        await _appraise_semantic_item(
            question=question,
            item_question=question,
            evidence=payload["evidence"],
            evidence_by_handle=evidence_by_handle,
            projection=projection,
            validation_state=preliminary_state,
            accepted_result=None,
            services=services,
            config=config,
            system_message=SystemMessage(content=SEMANTIC_APPRAISAL_PROMPT),
            payload_text=historical_payload_text,
            repair_allowed_values={
                "handle_field_domains": historical_question[
                    "handle_field_domains"
                ],
                "candidate_origin_evidence": historical_question[
                    "candidate_origin_evidence"
                ],
                "permitted_delta_path_domains": historical_question[
                    "permitted_delta_path_domains"
                ],
            },
            item_index=item_index,
        )
    except CognitionExecutionError as exc:
        caught_error = exc
    except CognitionContextLimitError as exc:
        caught_error = exc
    capture = validation_capture_snapshot()
    assert capture is not None
    raw_capture_path = write_validation_capture(
        artifact_root=_RAW_ARTIFACT_ROOT
    )
    artifact = {
        "schema_version": "trace_failure_mode_live_replay.v1",
        "case_id": case_id,
        "source_trace_id": case["trace_id"],
        "source_stage_name": case["stage_name"],
        "source_attempt_index": case["attempt_index"],
        "replay_mode": (
            "preserved_historical_candidate_then_live_repair"
            if historical_llm is not None
            else "current_system_prompt_historical_payload"
        ),
        "current_system_prompt_chars": len(SEMANTIC_APPRAISAL_PROMPT),
        "historical_system_prompt_chars": len(historical_system_prompt),
        "historical_validation_error": representative.get(
            "validation_error"
        ),
        "observed_error": (
            {
                "error_code": getattr(caught_error, "error_code", None),
                "attempt_count": getattr(caught_error, "attempt_count", None),
                "message": str(caught_error),
            }
            if caught_error is not None
            else None
        ),
        "model_calls": (
            historical_llm.calls if historical_llm is not None else []
        ),
        "capture": capture,
        "raw_capture_path": str(raw_capture_path),
    }
    write_diagnostic_artifact(
        f"{case_id}_{time_ns()}",
        artifact,
        artifact_root=_ARTIFACT_ROOT,
    )
    stages = capture["stages"]
    assert isinstance(stages, list)
    failed_stages = [
        stage
        for stage in stages
        if isinstance(stage, Mapping)
        and stage.get("parse_status") == "failed"
    ]
    if case.get("require_repair_call"):
        assert historical_llm is not None
        assert len(historical_llm.calls) >= 2, (
            "semantic repair boundary was not reached; "
            f"raw_capture={raw_capture_path}"
        )
        historical_error = str(
            representative.get("validation_error") or ""
        )
        assert "; permitted paths:" in historical_error
        repair_messages = historical_llm.calls[1]["messages"]
        repair_size = sum(
            len(str(message["content"])) for message in repair_messages
        )
        assert repair_size <= SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP
        repair_payload = json.loads(repair_messages[-1]["content"])
        assert set(repair_payload) == {
            "repair_instruction",
            "contract_error",
            "allowed_values",
        }
        contract_error = str(repair_payload["contract_error"])
        assert "knowledge_gaps.k7.uncertainty" in contract_error
        assert "permitted paths:" not in contract_error
        assert (
            "permitted_delta_path_domains"
            in repair_payload["allowed_values"]
        )
        assert any(
            stage.get("error") == historical_error
            for stage in stages
            if isinstance(stage, Mapping)
        )
    if not failed_stages:
        assert caught_error is None
        return
    assert all(stage.get("raw_output") for stage in failed_stages)
    successful_stages = [
        stage
        for stage in stages
        if isinstance(stage, Mapping)
        and stage.get("parse_status") == "succeeded"
    ]
    if caught_error is None and successful_stages:
        return
    expected_error = str(case["expected_error"])
    observed_errors = [
        str(stage.get("error") or "") for stage in failed_stages
    ]
    if not any(expected_error in error for error in observed_errors):
        pytest.fail(
            "current live model produced a different terminal contract "
            f"failure: {observed_errors}"
        )
    if capsule.get("outcome") == "terminal_failure":
        assert caught_error is not None
        assert (
            caught_error.error_code
            == "semantic_appraisal_contract_exhausted"
        )
        assert expected_error in str(caught_error)


async def test_semantic_delta_path_not_owned_live_llm() -> None:
    """Reproduce the captured relationship delta-path exhaustion."""

    await _run_semantic_case("semantic_delta_path_not_owned")


async def test_selected_evidence_unknown_handle_live_llm() -> None:
    """Reproduce the captured selected-evidence handle exhaustion."""

    await _run_semantic_case("selected_evidence_unknown_handle")


async def test_terminal_event_transition_rejected_live_llm() -> None:
    """Reproduce the captured terminal-event transition exhaustion."""

    await _run_semantic_case("terminal_event_transition_rejected")


async def test_candidate_origin_evidence_missing_live_llm() -> None:
    """Reproduce the captured candidate-origin evidence exhaustion."""

    await _run_semantic_case("candidate_origin_evidence_missing")


async def test_semantic_role_value_invalid_live_llm() -> None:
    """Reproduce the captured semantic-role exhaustion."""

    await _run_semantic_case("semantic_role_value_invalid")


async def test_current_run_event_agency_role_value_invalid_live_llm() -> None:
    """Replay the current run's event-agency role-value failure."""

    await _run_semantic_case(
        "current_run_event_agency_role_value_invalid"
    )


async def test_a1a573_goal_threat_unowned_path_live_llm() -> None:
    """Replay the unowned knowledge-gap path from the plan evidence."""

    await _run_semantic_case(
        "a1a573_goal_threat_unowned_knowledge_gap_path"
    )


async def test_resolved_knowledge_gap_transition_rejected_live_llm() -> None:
    """Reproduce the captured resolved-gap transition exhaustion."""

    await _run_semantic_case("resolved_knowledge_gap_transition_rejected")


async def test_selected_roles_unknown_handle_live_llm() -> None:
    """Reproduce the captured selected-role handle exhaustion."""

    await _run_semantic_case("selected_roles_unknown_handle")


async def test_semantic_proposition_subject_kind_mismatch_live_llm() -> None:
    """Reproduce the captured semantic subject-kind exhaustion."""

    await _run_semantic_case("semantic_proposition_subject_kind_mismatch")


async def test_semantic_proposition_object_handle_not_permitted_live_llm(
) -> None:
    """Reproduce the captured semantic object-handle exhaustion."""

    await _run_semantic_case(
        "semantic_proposition_object_handle_not_permitted"
    )


async def test_delta_reason_invalid_live_llm() -> None:
    """Reproduce the captured semantic reason exhaustion."""

    await _run_semantic_case("delta_reason_invalid")


async def test_semantic_delta_type_invalid_live_llm() -> None:
    """Reproduce the captured semantic delta-type exhaustion."""

    await _run_semantic_case("semantic_delta_type_invalid")


async def test_semantic_micro_appraisal_fields_not_exact_live_llm() -> None:
    """Reproduce the captured singular semantic-item exhaustion."""

    await _run_semantic_case("semantic_micro_appraisal_fields_not_exact")


async def test_captured_trace_8d0d4295_capacity_path_live_llm() -> None:
    """Reserve the one-at-a-time live capacity replay gate."""

    pytest.skip(
        "supplemental live capacity replay requires explicit live-gate "
        "enablement after deterministic evidence review"
    )


async def test_captured_trace_9164e957_capacity_path_live_llm() -> None:
    """Reserve the second one-at-a-time live capacity replay gate."""

    pytest.skip(
        "supplemental live capacity replay requires explicit live-gate "
        "enablement after deterministic evidence review"
    )
