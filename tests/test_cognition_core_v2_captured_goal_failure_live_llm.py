"""Active-prompt replay of the captured ordinary-goal authority failure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
    EVIDENCE_SOURCE_QUESTION_IDS,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT,
    run_goal_cognition,
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
    / 'turn_173098348_llm_trace.json'
)
_ARTIFACT_ROOT = (
    _ROOT
    / 'test_artifacts'
    / 'cognition_core_v2_relational_willingness'
    / 'captured_goal_replay'
)
_TRACE_SUITE = 'cognition_core_v2_captured_goal_failure_live_llm'
_CASE_ID = 'qq_54369546_message_173098348'
_STAGE_NAME = 'goal_cognition.ordinary_response.initial'
_EXPECTED_APPLICABILITY = 'relationship_sensitive'
_EXPECTED_RELATIONSHIP_STATE = 'unestablished'
_NON_ACCEPTING_STANCES = frozenset({'reject', 'deflect'})
_EPISODE_HANDLES = {'e1'}
_CAPTURED_OCCURRED_AT = '2026-08-04T00:00:00Z'


def _evidence_provenance_role(
    source_kind: str,
    memory_scope: object,
) -> str:
    """Return the provenance role applied by the active evidence builder."""

    from kazusa_ai_chatbot.cognition_core_v2.contracts import (
        project_evidence_provenance_role,
    )

    return project_evidence_provenance_role(source_kind, memory_scope)


def _load_captured_attempt() -> dict[str, Any]:
    """Load the exact historical model-facing attempt from the trace export."""

    if not _CAPTURED_TRACE_PATH.exists():
        raise AssertionError(
            f'captured production trace is missing: {_CAPTURED_TRACE_PATH}'
        )
    trace = json.loads(
        _CAPTURED_TRACE_PATH.read_text(encoding='utf-8')
    )
    trace_steps = trace['llm_trace_steps']
    capsules = trace.get('cognition_failure_capsules') or []
    for capsule in capsules:
        for attempt in capsule['attempts']:
            if attempt['stage_name'] != _STAGE_NAME:
                continue
            messages = attempt['messages']
            if not isinstance(messages, list) or len(messages) != 2:
                raise AssertionError(
                    'captured goal stage must have two messages'
                )
            raw_human_payload = str(messages[1].get('content', ''))
            try:
                frozen_human_payload = json.loads(raw_human_payload)
            except (TypeError, ValueError) as exc:
                raise AssertionError(
                    'captured goal human payload is not JSON'
                ) from exc
            return {
                'trace_id': capsule['trace_id'],
                'stage_name': attempt['stage_name'],
                'branch_id': attempt['branch_id'],
                'config': attempt['config'],
                'historical_system_prompt': str(
                    messages[0].get('content', '')
                ),
                'frozen_human_payload': frozen_human_payload,
                'frozen_human_payload_raw': raw_human_payload,
                'historical_output': attempt['parsed_output'],
            }
    for trace_step in trace_steps:
        if trace_step.get('stage_name') != _STAGE_NAME:
            continue
        raise AssertionError(
            'captured goal stage exists as a trace step but not as a '
            'failure-capsule attempt'
        )
    raise AssertionError(
        f'captured stage {_STAGE_NAME} is missing from the trace'
    )


def _replay_evidence_rows(
    frozen_payload: object,
) -> list[dict[str, Any]]:
    """Rebuild typed V2 evidence rows from the frozen dynamic payload."""

    if not isinstance(frozen_payload, dict):
        raise AssertionError('captured goal payload must be an object')
    raw_evidence = frozen_payload.get('evidence')
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise AssertionError('captured goal payload has no evidence rows')
    evidence: list[dict[str, Any]] = []
    for row in raw_evidence:
        if not isinstance(row, dict):
            raise AssertionError('captured evidence row must be an object')
        handle = str(row.get('handle', ''))
        source_kind = str(row.get('source_kind', ''))
        semantic_text = str(row.get('semantic_text', ''))
        if source_kind not in EVIDENCE_SOURCE_QUESTION_IDS:
            raise AssertionError(
                f'captured evidence source kind is unsupported: {source_kind}'
            )
        evidence_row: dict[str, Any] = {
            'evidence_handle': handle,
            'evidence_ref': {
                'source_kind': source_kind,
                'source_id': f'captured-replay:{handle}',
                'occurred_at': _CAPTURED_OCCURRED_AT,
                'semantic_summary': semantic_text[:200],
            },
            'semantic_text': semantic_text,
            'visible_to': list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
        }
        if isinstance(row.get('memory_scope'), str):
            evidence_row['memory_scope'] = row['memory_scope']
        evidence.append(evidence_row)
    return evidence


def _replay_role_bindings(
    frozen_payload: object,
) -> dict[str, dict[str, str]]:
    """Rebuild private handle bindings required by the current builder."""

    if not isinstance(frozen_payload, dict):
        raise AssertionError('captured goal payload must be an object')
    role_handles = frozen_payload.get('role_handles')
    if not isinstance(role_handles, list):
        raise AssertionError('captured goal payload has no role handles')
    bindings: dict[str, dict[str, str]] = {}
    for handle in role_handles:
        if handle == 'self':
            bindings[handle] = {
                'role': 'actor',
                'entity_kind': 'character',
                'entity_id': 'character:global',
            }
        elif handle == 'current_user':
            bindings[handle] = {
                'role': 'target',
                'entity_kind': 'user',
                'entity_id': 'replay:current-user',
            }
        else:
            bindings[handle] = {
                'role': 'target',
                'entity_kind': 'user',
                'entity_id': f'replay:{handle}',
            }
    return bindings


def _replay_semantic_context(
    frozen_payload: object,
) -> dict[str, Any]:
    """Rebuild the branch context consumed by the active goal builder."""

    if not isinstance(frozen_payload, dict):
        raise AssertionError('captured goal payload must be an object')
    semantic_context = frozen_payload.get('semantic_context')
    if not isinstance(semantic_context, dict):
        raise AssertionError('captured goal payload has no semantic context')
    context = dict(semantic_context)
    context['role_summaries'] = dict(
        frozen_payload.get('role_summaries') or {}
    )
    context['_role_bindings'] = _replay_role_bindings(frozen_payload)
    return context


class _CapturingLLM:
    """Capture the active goal model boundary without changing behavior."""

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
                    'role': (
                        'system'
                        if type(message).__name__ == 'SystemMessage'
                        else 'human'
                    ),
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


def _write_artifact(sample_index: int, artifact: dict[str, Any]) -> str:
    """Write one durable human-readable replay artifact."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = (
        _ARTIFACT_ROOT
        / f'{_CASE_ID}__sample{sample_index}__{time_ns()}.json'
    )
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str)
        + '\n',
        encoding='utf-8',
    )
    return str(path)


async def _run_captured_replay(sample_index: int) -> dict[str, Any]:
    """Run the frozen dynamic payload through the current V2 goal builder."""

    captured = _load_captured_attempt()
    evidence = _replay_evidence_rows(captured['frozen_human_payload'])
    semantic_context = _replay_semantic_context(
        captured['frozen_human_payload']
    )
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    bid: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    started_at = perf_counter()
    try:
        bid = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
            {
                'scope': 'user',
                'kind': 'goal',
                'entity_id': 'goal:captured-replay',
            },
            semantic_context,
            evidence,
            services,
        )
    except CognitionExecutionError as exc:
        failure = {
            'error_code': exc.error_code,
            'message': str(exc),
            'attempt_count': exc.attempt_count,
            'safe_checkpoint': exc.safe_checkpoint,
        }
    duration_ms = int((perf_counter() - started_at) * 1000)

    evidence_roles = [
        {
            'handle': row['evidence_handle'],
            'source_kind': row['evidence_ref']['source_kind'],
            'memory_scope': row.get('memory_scope'),
            'provenance_role': None,
        }
        for row in evidence
    ]
    decision: dict[str, Any] | None = None
    if isinstance(bid, dict):
        candidate = bid.get('relational_willingness')
        if isinstance(candidate, dict):
            decision = dict(candidate)
    rendered_calls = capturing_llm.calls
    rendered_prompt = (
        rendered_calls[0]['messages']
        if rendered_calls
        else []
    )
    human_payload = (
        rendered_calls[0]['messages'][1]['content']
        if rendered_calls
        else ''
    )
    raw_output = (
        rendered_calls[0]['raw_output'] if rendered_calls else ''
    )
    parsed_output: object = None
    if isinstance(bid, dict):
        parsed_output = {
            key: value
            for key, value in bid.items()
            if key not in {
                'branch_id',
                'goal_ref',
                'target_roles',
            }
        }
    artifact: dict[str, Any] = {
        'schema_version': (
            'cognition_core_v2_captured_goal_replay.v1'
        ),
        'case_id': _CASE_ID,
        'sample_index': sample_index,
        'input_kind': 'captured_production_stage_boundary',
        'source_trace_path': str(_CAPTURED_TRACE_PATH),
        'source_trace_id': captured['trace_id'],
        'stage_name': captured['stage_name'],
        'branch_id': captured['branch_id'],
        'prompt_version': {
            'system': 'GOAL_COGNITION_PROMPT',
            'system_sha256': hashlib.sha256(
                GOAL_COGNITION_PROMPT.encode('utf-8')
            ).hexdigest(),
            'human_payload_sha256': hashlib.sha256(
                human_payload.encode('utf-8')
            ).hexdigest(),
        },
        'historical_attempt': {
            'system_prompt': captured['historical_system_prompt'],
            'parsed_output': captured['historical_output'],
        },
        'frozen_dynamic_payload': captured['frozen_human_payload'],
        'reconstructed_evidence': evidence,
        'evidence_roles': [
            {
                **row,
                'provenance_role': _evidence_provenance_role(
                    row['source_kind'],
                    row.get('memory_scope'),
                ),
            }
            for row in evidence_roles
        ],
        'rendered_prompt': rendered_prompt,
        'raw_output': raw_output,
        'parsed_output': parsed_output,
        'validated_bid': bid,
        'typed_decision': decision,
        'downstream_effects': {
            'non_accept_action_denial': (
                isinstance(decision, dict)
                and decision.get('applicability') == 'relationship_sensitive'
                and decision.get('stance') != 'accept'
            ),
            'resolver_denial': (
                isinstance(decision, dict)
                and decision.get('applicability') == 'relationship_sensitive'
                and decision.get('stance') != 'accept'
            ),
        },
        'model_calls': rendered_calls,
        'metrics': {
            'goal_call_count': len(rendered_calls),
            'duration_ms': duration_ms,
            'prompt_lengths': [
                sum(
                    len(str(message['content']))
                    for message in call['messages']
                )
                for call in rendered_calls
            ],
        },
        'expected_contract': {
            'applicability': _EXPECTED_APPLICABILITY,
            'current_user_relationship_state': (
                _EXPECTED_RELATIONSHIP_STATE
            ),
            'stance': sorted(_NON_ACCEPTING_STANCES),
        },
        'frozen_payload_identifier_notes': {
            'group_id_in_frozen_scene': '54369546' in (
                json.dumps(
                    captured['frozen_human_payload'],
                    ensure_ascii=False,
                )
            ),
            'note': (
                'Identifiers inside the frozen captured payload are '
                'historical capture content, not production prompt text.'
            ),
        },
        'quality_notes': '',
    }
    artifact_path = _write_artifact(sample_index, artifact)
    artifact['artifact_path'] = artifact_path
    if failure is not None:
        pytest.fail(
            f'captured goal replay did not complete: {failure}; '
            f'artifact={artifact_path}'
        )
    if bid is None:
        pytest.fail(f'captured goal replay produced no bid; artifact={artifact_path}')
    if decision is None:
        pytest.fail(
            f'captured goal replay produced no decision; artifact={artifact_path}'
        )
    return {
        'artifact_path': artifact_path,
        'bid': bid,
        'decision': decision,
    }


def _assert_expected_decision(decision: dict[str, Any]) -> None:
    """Require the reconstructed group case to deny unsafe acceptance."""

    assert decision.get('applicability') == _EXPECTED_APPLICABILITY
    assert (
        decision.get('current_user_relationship_state')
        == _EXPECTED_RELATIONSHIP_STATE
    )
    assert decision.get('stance') in _NON_ACCEPTING_STANCES, (
        'captured production acceptance failure reproduced: '
        f'observed={decision}'
    )
    assert set(decision) == {
        'schema_version',
        'applicability',
        'stance',
        'current_user_relationship_state',
        'reason',
        'evidence_handles',
    }
    assert decision.get('schema_version') == 'relational_willingness.v2'
    assert set(decision['evidence_handles']).intersection(
        _EPISODE_HANDLES
    )


async def test_captured_goal_replay_sample_1_rejects() -> None:
    """Fresh sample one of the reconstructed group case rejects."""

    result = await _run_captured_replay(1)
    _assert_expected_decision(result['decision'])


async def test_captured_goal_replay_sample_2_rejects() -> None:
    """Fresh sample two of the reconstructed group case rejects."""

    result = await _run_captured_replay(2)
    _assert_expected_decision(result['decision'])


async def test_captured_goal_replay_sample_3_rejects() -> None:
    """Fresh sample three of the reconstructed group case rejects."""

    result = await _run_captured_replay(3)
    _assert_expected_decision(result['decision'])
