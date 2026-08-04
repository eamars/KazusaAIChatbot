"""Reproduce captured semantic-appraisal exhaustion with a live LLM."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from time import time_ns
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
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
    appraise_semantic_question,
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
_CAPTURED_TRACE_PATH = (
    _ROOT
    / 'test_artifacts'
    / 'diagnostics'
    / 'target_2304_llm_trace.json'
)
_ARTIFACT_ROOT = (
    _ROOT
    / 'test_artifacts'
    / 'cognition_core_v2_semantic_appraisal_exhaustion'
)
_RAW_ARTIFACT_ROOT = _ARTIFACT_ROOT / 'raw'
_EXPECTED_ERROR = (
    'causal candidates must cite originating evidence: ce1->e1'
)
_EXPECTED_ERROR_CODE = 'semantic_appraisal_contract_exhausted'
_EXPECTED_ATTEMPTS = 2
_TARGET_QUESTION_IDS = frozenset({
    'q:moral_identity',
    'q:goal_threat_outcome',
})


def _load_captured_input() -> tuple[dict[str, Any], str]:
    """Load the exact input payload from the preserved production trace."""

    if not _CAPTURED_TRACE_PATH.exists():
        raise AssertionError(
            f'captured production trace is missing: {_CAPTURED_TRACE_PATH}'
        )
    trace_bytes = _CAPTURED_TRACE_PATH.read_bytes()
    trace = json.loads(trace_bytes.decode('utf-8'))
    capsules = [
        step['capsule']
        for step in trace.get('llm_trace_steps', [])
        if (
            isinstance(step, Mapping)
            and step.get('stage_name') == 'cognition_failure_capsule'
            and isinstance(step.get('capsule'), Mapping)
        )
    ]
    if len(capsules) != 1:
        raise AssertionError(
            'the captured trace must contain exactly one failure capsule'
        )
    input_payload = capsules[0].get('input_payload')
    if not isinstance(input_payload, dict):
        raise AssertionError(
            'the captured failure capsule input is not an object'
        )
    return input_payload, hashlib.sha256(trace_bytes).hexdigest()


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
    previous_state = validate_cognition_state(payload['mutable_state'])
    updated_at = _episode_updated_at(payload['episode'])
    elapsed_seconds = _cognition_elapsed_seconds(previous_state, updated_at)
    fact_pairs = [
        (fact['producer'], _fact_without_producer(fact))
        for fact in payload['direct_facts']
    ]
    relationship_context = _native_relationship_context(
        payload.get('relationship_context'),
    )
    preliminary_state = apply_state_update(
        previous_state,
        direct_facts=fact_pairs,
        elapsed_seconds=elapsed_seconds,
        updated_at=updated_at,
        character_constraints=payload['character_constraints'],
        relationship_context=relationship_context,
    )
    preliminary_state = create_deterministic_goals(
        preliminary_state,
        character_constraints=payload['character_constraints'],
        relationship_context=relationship_context,
        evidence=payload['evidence'],
        updated_at=updated_at,
    )
    preliminary_state = validate_cognition_state(preliminary_state)
    projection = project_state_for_prompt(
        preliminary_state,
        character_constraints=payload['character_constraints'],
        character_identity_context=payload['character_identity_context'],
        relationship_context=payload.get('relationship_context'),
        character_operational_context=payload.get(
            'character_operational_context',
        ),
        evidence=payload['evidence'],
    )
    questions = plan_semantic_questions(
        payload['evidence'],
        preliminary_state,
        projection.handle_to_ref,
    )
    return payload, preliminary_state, projection, questions


def _candidate_origin_is_omitted(parsed_output: object) -> bool:
    """Identify the captured candidate-origin omission in one parsed output."""

    if not isinstance(parsed_output, Mapping):
        return False
    for field_name in ('propositions', 'deltas'):
        rows = parsed_output.get(field_name)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            structural_text = ' '.join(
                str(row.get(key, ''))
                for key in (
                    'subject_handle',
                    'object_handle',
                    'target_path',
                )
            )
            if 'ce1' not in structural_text:
                continue
            evidence_handles = row.get('evidence_handles')
            if (
                isinstance(evidence_handles, list)
                and 'e1' not in evidence_handles
            ):
                return True
    return False


def _persist_case_evidence(
    *,
    case_id: str,
    source_sha256: str,
    input_payload: Mapping[str, Any],
    question: Mapping[str, Any],
    error: CognitionExecutionError | None,
    capture: Mapping[str, Any],
) -> None:
    """Persist raw model evidence and a structured case envelope."""

    raw_path = write_validation_capture(artifact_root=_RAW_ARTIFACT_ROOT)
    write_diagnostic_artifact(
        f'{case_id}_{time_ns()}',
        {
            'case_id': case_id,
            'source_trace': {
                'path': str(_CAPTURED_TRACE_PATH),
                'sha256': source_sha256,
            },
            'input_payload': dict(input_payload),
            'question': dict(question),
            'expected': {
                'error_code': _EXPECTED_ERROR_CODE,
                'error': _EXPECTED_ERROR,
                'attempt_count': _EXPECTED_ATTEMPTS,
            },
            'observed_exception': (
                {
                    'type': type(error).__name__,
                    'message': str(error),
                    'error_code': error.error_code,
                    'attempt_count': error.attempt_count,
                }
                if error is not None
                else None
            ),
            'raw_capture_path': str(raw_path),
            'validation_capture': dict(capture),
        },
        artifact_root=_ARTIFACT_ROOT,
    )


async def _run_exact_exhaustion_case(question_id: str) -> None:
    """Run one captured appraisal family against the real model boundary."""

    input_payload, source_sha256 = _load_captured_input()
    payload, preliminary_state, projection, questions = (
        _build_appraisal_context(input_payload)
    )
    matching_questions = [
        question
        for question in questions
        if question.get('question_id') == question_id
    ]
    assert len(matching_questions) == 1
    question = matching_questions[0]
    assert question_id in _TARGET_QUESTION_IDS
    case_id = (
        'qq_54369546_message_1538018034_'
        f'{question_id.replace(":", "_")}'
    )
    reset_validation_capture(case_id)

    caught_error: CognitionExecutionError | None = None
    try:
        await appraise_semantic_question(
            question,
            payload['evidence'],
            projection,
            build_cognition_core_services(),
            validation_state=preliminary_state,
        )
    except CognitionExecutionError as exc:
        caught_error = exc

    capture = validation_capture_snapshot()
    assert capture is not None
    _persist_case_evidence(
        case_id=case_id,
        source_sha256=source_sha256,
        input_payload=input_payload,
        question=question,
        error=caught_error,
        capture=capture,
    )

    if caught_error is None:
        pytest.xfail(
            "current live model did not reproduce the captured exhausted "
            f"case; inspect {_ARTIFACT_ROOT}"
        )
    assert caught_error is not None
    assert caught_error.error_code == _EXPECTED_ERROR_CODE
    assert caught_error.attempt_count == _EXPECTED_ATTEMPTS
    stages = capture['stages']
    assert isinstance(stages, list)
    expected_stage_ids = [
        f'semantic_appraisal:{question_id}:item_1',
        f'semantic_appraisal:{question_id}:item_1:repair_1',
    ]
    observed_stages = [
        stage
        for stage in stages
        if isinstance(stage, Mapping)
        and stage.get('stage_id') in expected_stage_ids
    ]
    assert [stage['stage_id'] for stage in observed_stages] == (
        expected_stage_ids
    )
    assert len(observed_stages) == _EXPECTED_ATTEMPTS
    for stage in observed_stages:
        assert stage['parse_status'] == 'failed'
        assert stage['error'] == _EXPECTED_ERROR
        assert isinstance(stage['raw_output'], str)
        assert stage['raw_output']
        assert _candidate_origin_is_omitted(stage['parsed_output'])


async def test_moral_identity_contract_exhaustion_live_llm() -> None:
    """The captured moral appraisal reproduces its validation exhaustion."""

    await _run_exact_exhaustion_case('q:moral_identity')


async def test_goal_threat_outcome_contract_exhaustion_live_llm() -> None:
    """The captured goal appraisal reproduces its validation exhaustion."""

    await _run_exact_exhaustion_case('q:goal_threat_outcome')
