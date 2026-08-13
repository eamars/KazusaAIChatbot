"""Real-model probes for the residual semantic-Appraisal contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
import sys
from time import perf_counter
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContextLimitError,
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    SEMANTIC_APPRAISAL_PROMPT,
    SEMANTIC_APPRAISAL_PROMPT_CAP,
    appraise_semantic_question,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v2_test_helpers import canonical_identity_context
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


_TRACE_SUITE = 'cognition_core_v2_appraisal_contract_live_llm'


class _CapturingLLM:
    """Capture each real semantic-Appraisal request and response."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object,
        **kwargs: object,
    ) -> Any:
        """Invoke the configured route while retaining inspectable evidence."""

        started_at = perf_counter()
        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        self.calls.append({
            'messages': [
                {
                    'type': type(message).__name__,
                    'content': str(getattr(message, 'content', '')),
                }
                for message in messages
            ],
            'raw_output': str(response.content),
            'duration_ms': round((perf_counter() - started_at) * 1000, 3),
            'route': {
                'route_name': str(getattr(config, 'route_name', '')),
                'model': str(getattr(config, 'model', '')),
            },
        })
        return response


def _character_constraints() -> dict[str, Any]:
    """Build the fixed character projection required by appraisal prompts."""

    state = build_character_production_state(
        updated_at='2026-08-11T00:00:00Z',
    )
    return {
        'drives': state['drives'],
        'standards': state['standards'],
        'meaning_state': state['meaning_state'],
        'personality_judgment': {
            'logic': 'evidence-led judgment',
            'defense': 'protect character agency',
            'quirks': 'warm directness',
            'taboos': 'inventing unsupported facts',
        },
    }


def _build_case(
    *,
    case_id: str,
    permitted_delta_paths: list[str],
    semantic_text: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Any,
    dict[str, Any],
]:
    """Build one small production-shaped relationship appraisal boundary."""

    updated_at = '2026-08-11T00:00:00Z'
    state = build_acquaintance_user_state(
        global_user_id=f'appraisal-live-{case_id}',
        updated_at=updated_at,
    )
    evidence = [{
        'evidence_handle': 'e1',
        'evidence_ref': {
            'source_kind': 'episode',
            'source_id': f'episode:appraisal-live:{case_id}',
            'occurred_at': updated_at,
            'semantic_summary': semantic_text,
        },
        'semantic_text': semantic_text,
        'visible_to': ['q:relationship_social'],
    }]
    projection = project_state_for_prompt(
        state,
        character_constraints=_character_constraints(),
        character_identity_context=canonical_identity_context(),
        evidence=evidence,
    )
    question = {
        'question_id': 'q:relationship_social',
        'question_kind': 'relationship_social',
        'semantic_question': (
            '判断当前用户本轮表达对关系互动意味着什么，只使用当前证据。'
        ),
        'evidence_handles': ['e1'],
        'permitted_role_handles': ['r1', 'current_user', 'self'],
        'permitted_role_assignment_handles': [
            'r1',
            'current_user',
            'self',
        ],
        'permitted_delta_paths': permitted_delta_paths,
        'dependencies': [],
    }
    return state, evidence, projection, question


async def _run_case(
    *,
    case_id: str,
    permitted_delta_paths: list[str],
    semantic_text: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], object]:
    """Run one real appraisal family and persist its model-boundary evidence."""

    state, evidence, projection, question = _build_case(
        case_id=case_id,
        permitted_delta_paths=permitted_delta_paths,
        semantic_text=semantic_text,
    )
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    result: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    try:
        result = await appraise_semantic_question(
            question,
            evidence,
            projection,
            services,
            validation_state=state,
        )
    except (CognitionExecutionError, CognitionContextLimitError) as exc:
        failure = {
            'error_class': type(exc).__name__,
            'error_code': getattr(exc, 'error_code', None),
            'attempt_count': getattr(exc, 'attempt_count', None),
            'message': str(exc),
        }
    trace_path = write_llm_trace(
        _TRACE_SUITE,
        case_id,
        {
            'case_id': case_id,
            'question': question,
            'evidence': evidence,
            'model_calls': capturing_llm.calls,
            'result': result,
            'failure': failure,
            'agent_review': {
                'hard_gate': 'result must be structurally valid',
                'behavior_verdict': (
                    'real model output is inspected by the assertions in this '
                    'node'
                ),
            },
        },
    )
    assert capturing_llm.calls, (
        f'Appraisal made no live model call; trace={trace_path}'
    )
    assert failure is None, (
        f'Appraisal contract failed before completion: {failure}; '
        f'trace={trace_path}'
    )
    assert result is not None
    return result, capturing_llm.calls, trace_path


def _initial_question_payload(call: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the initial human payload from one captured model call."""

    messages = call.get('messages')
    if not isinstance(messages, list) or not messages:
        raise AssertionError('captured Appraisal messages are missing')
    payload = json.loads(str(messages[-1]['content']))
    if not isinstance(payload, dict):
        raise AssertionError('captured Appraisal payload is not an object')
    return payload


def _expand_path_domains(domains: object) -> set[str]:
    """Expand grouped path domains into their exact canonical path strings."""

    if not isinstance(domains, list):
        raise AssertionError('Appraisal path domains are not a list')
    paths: set[str] = set()
    for domain in domains:
        if not isinstance(domain, Mapping):
            raise AssertionError('Appraisal path domain is not an object')
        state_field = domain.get('state_field')
        handles = domain.get('handles')
        axes = domain.get('axes')
        if (
            not isinstance(state_field, str)
            or not isinstance(handles, list)
            or not isinstance(axes, list)
        ):
            raise AssertionError('Appraisal path domain has invalid fields')
        paths.update(
            f'{state_field}.{handle}.{axis}'
            for handle in handles
            for axis in axes
        )
    return paths


async def test_live_appraisal_projects_exact_paths_without_crossing_domains(
) -> None:
    """The real prompt must expose cap-safe exact paths and separate handles."""

    permitted_paths = [
        'relationship.r1.familiarity',
        'relationship.r1.positive_regard',
        'relationship.r1.trust',
        'relationship.r1.attachment',
        'relationship.r1.desired_closeness',
        'relationship.r1.perceived_closeness',
        'relationship.r1.care',
        'relationship.r1.boundary_safety',
        'relationship.r1.exclusivity',
        'relationship.r1.unresolved_injury',
        'relationship.r1.salience',
    ]
    _, calls, trace_path = await _run_case(
        case_id='exact_path_domains',
        permitted_delta_paths=permitted_paths,
        semantic_text=(
            '当前用户认真回应了角色的边界表达，关系中的信任与亲近感出现了'
            '一个需要被当前角色判断的细微变化。'
        ),
    )
    initial_payload = _initial_question_payload(calls[0])
    question_payload = initial_payload['question']
    assert question_payload['permitted_target_paths'] == permitted_paths
    assert _expand_path_domains(
        question_payload['permitted_delta_path_domains']
    ) == set(permitted_paths)
    handle_domains = question_payload['handle_field_domains']
    assert set(handle_domains['evidence_handles']) == {'e1'}
    assert 'e1' not in set(handle_domains['subject_handle'])
    assert 'e1' not in set(handle_domains['object_handle'])
    assert 'e1' not in set(handle_domains['entity_handle'])
    assert set(handle_domains['subject_handle']) == {
        'r1',
        'current_user',
        'self',
    }
    assert set(handle_domains['entity_handle']) == {
        'r1',
        'current_user',
        'self',
    }
    assert set(handle_domains['entity_handle']) == set(
        question_payload['permitted_role_assignment_handles']
    )
    prompt_chars = sum(
        len(str(message['content']))
        for message in calls[0]['messages']
    )
    assert prompt_chars <= SEMANTIC_APPRAISAL_PROMPT_CAP
    assert len(calls[0]['messages'][0]['content']) == len(
        SEMANTIC_APPRAISAL_PROMPT
    )
    assert trace_path.exists()


async def test_live_appraisal_emits_singular_nullable_micro_item() -> None:
    """The first real response must follow the singular micro-item contract."""

    permitted_paths = [
        'relationship.r1.perceived_closeness',
    ]
    result, calls, trace_path = await _run_case(
        case_id='singular_nullable_micro_item',
        permitted_delta_paths=permitted_paths,
        semantic_text=(
            '当前用户明确表示愿意尊重当前角色提出的节奏边界，当前角色需要'
            '判断这是否支持关系亲近感的小幅变化。'
        ),
    )
    initial_payload = _initial_question_payload(calls[0])
    micro_contract = initial_payload['question']['micro_appraisal']
    assert micro_contract['maximum_propositions'] == 1
    assert micro_contract['maximum_deltas'] == 1
    assert micro_contract['empty_lists_end_family'] is True
    parsed = parse_llm_json_output(
        calls[0]['raw_output'],
        deterministic_only=True,
    )
    assert isinstance(parsed, dict)
    assert set(parsed) == {'question_id', 'proposition', 'delta'}
    assert parsed['question_id'] == 'q:relationship_social'
    assert parsed['proposition'] is None or isinstance(
        parsed['proposition'],
        Mapping,
    )
    assert parsed['delta'] is None or isinstance(parsed['delta'], Mapping)
    if isinstance(parsed['delta'], Mapping):
        assert parsed['delta']['target_path'] in permitted_paths
    assert result['question_id'] == 'q:relationship_social'
    assert trace_path.exists()
