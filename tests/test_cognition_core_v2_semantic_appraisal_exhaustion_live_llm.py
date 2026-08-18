"""Replay captured semantic-appraisal boundary dispositions with a live LLM."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _cognition_elapsed_seconds,
    _episode_updated_at,
    _fact_without_producer,
    _native_relationship_context,
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
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v2_appraisal_replay_harness import (
    replay_appraisal_through_public_boundary,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_EXACT_BOUNDARY_CASES = {
    'q:moral_identity': {
        'case_id': 'capacity_8d0d4295_moral_identity_origin_binding',
        'trace_path': (
            _ROOT
            / 'test_artifacts'
            / 'diagnostics'
            / 'llmtrace_8d0d42952b76450c9e1dc32574f9fd44_replay_export.json'
        ),
        'trace_id': 'llmtrace_8d0d42952b76450c9e1dc32574f9fd44',
    },
    'q:goal_threat_outcome': {
        'case_id': 'capacity_9164e957_goal_threat_origin_binding',
        'trace_path': (
            _ROOT
            / 'test_artifacts'
            / 'diagnostics'
            / 'llmtrace_9164e957298e4cffb68db7911bcd28b1_replay_export.json'
        ),
        'trace_id': 'llmtrace_9164e957298e4cffb68db7911bcd28b1',
    },
}
_EXPECTED_ERROR = (
    'causal candidates must cite originating evidence: ce1->e1'
)
_NEAR_CAP_CURRENT_ERROR_FRAGMENTS = (
    'semantic delta path',
    'selected roles contains unknown handles',
    'causal candidates must cite originating evidence',
)
_NEAR_CAP_CASES: tuple[dict[str, object], ...] = (
    {
        'case_id': 'a1a573_near_cap_boundary_termination',
        'trace_id': 'llmtrace_93482f08e4a74aa5af90adc6e6f5918a',
        'trace_path': (
            _ROOT
            / 'test_artifacts'
            / 'diagnostics'
            / 'cognition_trace_a1a573b590a3494786c4edebdee55342.json'
        ),
        'stage_name': 'semantic_appraisal.q:goal_threat_outcome.item_1',
        'question_id': 'q:goal_threat_outcome',
    },
    {
        'case_id': 'caad1a_near_cap_boundary_termination',
        'trace_id': 'llmtrace_caad1a9370cf4d859e8ea6233f1e473d',
        'trace_path': (
            _ROOT
            / 'test_artifacts'
            / 'diagnostics'
            / (
                'postdraft_goal_bid_failure_llmtrace_'
                'caad1a9370cf4d859e8ea6233f1e473d.json'
            )
        ),
        'stage_name': 'semantic_appraisal.q:goal_threat_outcome.item_1',
        'question_id': 'q:goal_threat_outcome',
    },
    {
        'case_id': 'df6eb4_near_cap_boundary_termination',
        'trace_id': 'llmtrace_df6eb45b1bfc405fa0e781baa7ce8d76',
        'trace_path': (
            _ROOT
            / 'test_artifacts'
            / 'diagnostics'
            / (
                'postdraft_goal_bid_failure_llmtrace_'
                'df6eb45b1bfc405fa0e781baa7ce8d76.json'
            )
        ),
        'stage_name': 'semantic_appraisal.q:goal_threat_outcome.item_1',
        'question_id': 'q:goal_threat_outcome',
    },
)


def _load_captured_input(
    case: Mapping[str, object],
) -> tuple[dict[str, Any], str, str, Path]:
    """Load one preserved input and a real semantic candidate for mutation."""

    trace_path = case.get('trace_path')
    trace_id = case.get('trace_id')
    if not isinstance(trace_path, Path) or not isinstance(trace_id, str):
        raise AssertionError('exact boundary trace identity is invalid')
    if not trace_path.exists():
        raise AssertionError(
            f'captured production trace is missing: {trace_path}'
        )
    trace_bytes = trace_path.read_bytes()
    trace = json.loads(trace_bytes.decode('utf-8'))
    capsules = [
        capsule
        for capsule in trace.get('cognition_failure_capsules', [])
        if isinstance(capsule, Mapping) and capsule.get('trace_id') == trace_id
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
    attempts = [
        attempt
        for attempt in capsules[0].get('attempts', [])
        if (
            isinstance(attempt, Mapping)
            and attempt.get('stage_name')
            == 'semantic_appraisal.q:goal_threat_outcome.item_1'
            and isinstance(attempt.get('raw_response_text'), str)
        )
    ]
    if len(attempts) != 1:
        raise AssertionError('the captured trace lacks one goal appraisal')
    return (
        input_payload,
        hashlib.sha256(trace_bytes).hexdigest(),
        str(attempts[0]['raw_response_text']),
        trace_path,
    )


def _build_origin_binding_candidate(
    raw_response: str,
    question: Mapping[str, Any],
) -> str:
    """Turn a preserved candidate into a bounded origin-binding failure."""

    parsed = parse_llm_json_output(
        raw_response,
        deterministic_only=True,
    )
    if not isinstance(parsed, dict):
        raise AssertionError('preserved goal candidate is not an object')
    paths = [
        path
        for path in question['permitted_delta_paths']
        if '.ce1.' in path
    ]
    if not paths:
        raise AssertionError('goal question has no ce1-owned delta path')
    evidence_handles = [
        handle
        for handle in question['evidence_handles']
        if handle != 'e1'
    ]
    if not evidence_handles:
        raise AssertionError('goal question has no alternate evidence handle')
    parsed['question_id'] = question['question_id']
    parsed['proposition'] = None
    parsed['delta'] = {
        'target_path': paths[0],
        'delta': 0,
        'evidence_handles': [evidence_handles[0]],
        'reason': '受支持证据需要通过当前语义边界进行有限验证。',
    }
    return json.dumps(parsed, ensure_ascii=False)


def _load_near_cap_input(
    case: Mapping[str, object],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Load one preserved near-cap input and its historical attempt."""

    trace_path = case.get('trace_path')
    trace_id = case.get('trace_id')
    stage_name = case.get('stage_name')
    if not isinstance(trace_path, Path):
        raise AssertionError('near-cap trace path is invalid')
    if not isinstance(trace_id, str) or not isinstance(stage_name, str):
        raise AssertionError('near-cap trace identity is invalid')
    if not trace_path.exists():
        raise AssertionError(
            f'near-cap production trace is missing: {trace_path}'
        )
    trace_bytes = trace_path.read_bytes()
    trace = json.loads(trace_bytes.decode('utf-8'))
    capsules = [
        capsule
        for capsule in trace.get('cognition_failure_capsules', [])
        if (
            isinstance(capsule, Mapping)
            and capsule.get('trace_id') == trace_id
        )
    ]
    if len(capsules) != 1:
        raise AssertionError(
            'the near-cap trace must contain exactly one matching capsule'
        )
    capsule = capsules[0]
    input_payload = capsule.get('input_payload')
    if not isinstance(input_payload, dict):
        raise AssertionError('near-cap failure input is not an object')
    attempts = [
        attempt
        for attempt in capsule.get('attempts', [])
        if (
            isinstance(attempt, Mapping)
            and attempt.get('stage_name') == stage_name
        )
    ]
    if len(attempts) != 1:
        raise AssertionError(
            'the near-cap trace must contain one initial goal appraisal'
        )
    attempt = attempts[0]
    return input_payload, hashlib.sha256(trace_bytes).hexdigest(), {
        'stage_name': attempt['stage_name'],
        'validation_error': str(attempt.get('validation_error') or ''),
        'raw_response_text': str(attempt.get('raw_response_text') or ''),
        'message_lengths': [
            len(str(message.get('content') or ''))
            for message in attempt.get('messages', [])
            if isinstance(message, Mapping)
        ],
    }


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
        scene_context=payload.get('scene_context'),
        evidence=payload['evidence'],
    )
    questions = plan_semantic_questions(
        payload['evidence'],
        preliminary_state,
        projection.handle_to_ref,
    )
    return payload, preliminary_state, projection, questions


async def _run_exact_boundary_case(
    question_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay an exact boundary candidate through the public facade."""

    case = _EXACT_BOUNDARY_CASES[question_id]
    input_payload, source_sha256, raw_response, source_path = (
        _load_captured_input(case)
    )
    _, _, _, questions = _build_appraisal_context(input_payload)
    question = next(
        question for question in questions
        if question.get("question_id") == question_id
    )
    preserved = _build_origin_binding_candidate(raw_response, question)
    before = parse_llm_json_output(raw_response, deterministic_only=True)
    after = parse_llm_json_output(preserved, deterministic_only=True)
    controlled_mutation = {
        "classification": "controlled_candidate_mutation",
        "source": "preserved_historical_candidate",
        "json_pointers": [
            {
                "pointer": "/proposition",
                "before": before.get("proposition")
                if isinstance(before, Mapping)
                else None,
                "after": after.get("proposition")
                if isinstance(after, Mapping)
                else None,
            },
            {
                "pointer": "/delta",
                "before": before.get("delta")
                if isinstance(before, Mapping)
                else None,
                "after": after.get("delta")
                if isinstance(after, Mapping)
                else None,
            },
        ],
        "first_attempt_performance_evidence": False,
    }
    replay = await replay_appraisal_through_public_boundary(
        input_payload=input_payload,
        question=question,
        first_response_text=preserved,
        case_id=(
            "exact_boundary_"
            f"{question_id.replace(':', '_')}"
        ),
        expected_error_fragments=(_EXPECTED_ERROR,),
        source_trace_id=str(case["trace_id"]),
        source_path=source_path,
        source_sha256=source_sha256,
        candidate_classification="controlled_candidate_mutation",
        controlled_mutation=controlled_mutation,
        require_repair_call=False,
        monkeypatch=monkeypatch,
    )
    assert replay["artifact_path"].exists()


async def test_moral_identity_boundary_terminates_family_live_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The captured moral candidate terminates its appraisal family."""

    await _run_exact_boundary_case('q:moral_identity', monkeypatch)


async def test_goal_threat_outcome_boundary_terminates_family_live_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The captured goal candidate terminates its appraisal family."""

    await _run_exact_boundary_case('q:goal_threat_outcome', monkeypatch)


async def _run_near_cap_case(
    case: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay one near-cap boundary candidate through the public facade."""

    input_payload, source_sha256, historical = _load_near_cap_input(case)
    _, _, _, questions = _build_appraisal_context(input_payload)
    question_id = case.get("question_id")
    if not isinstance(question_id, str):
        raise AssertionError("near-cap question id is invalid")
    question = next(
        question for question in questions
        if question.get("question_id") == question_id
    )
    source_path = case.get("trace_path")
    if not isinstance(source_path, Path):
        raise AssertionError("near-cap source path is invalid")
    replay = await replay_appraisal_through_public_boundary(
        input_payload=input_payload,
        question=question,
        first_response_text=str(historical["raw_response_text"]),
        case_id=str(case["case_id"]),
        expected_error_fragments=_NEAR_CAP_CURRENT_ERROR_FRAGMENTS,
        source_trace_id=str(case["trace_id"]),
        source_path=source_path,
        source_sha256=source_sha256,
        candidate_classification="preserved_failed_candidate",
        require_repair_call=False,
        monkeypatch=monkeypatch,
    )
    assert replay["artifact_path"].exists()


async def test_a1a573_near_cap_boundary_terminates_family_live_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original near-cap candidate terminates without repair."""

    await _run_near_cap_case(_NEAR_CAP_CASES[0], monkeypatch)


async def test_caad1a_near_cap_boundary_terminates_family_live_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first near-cap recurrence terminates without repair."""

    await _run_near_cap_case(_NEAR_CAP_CASES[1], monkeypatch)


async def test_df6eb4_near_cap_boundary_terminates_family_live_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second near-cap recurrence terminates without repair."""

    await _run_near_cap_case(_NEAR_CAP_CASES[2], monkeypatch)
