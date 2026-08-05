"""Replay captured Cognition Core V2 goal-contract recovery evidence."""

from __future__ import annotations

import json
from copy import deepcopy
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
    / (
        'cognition_v2_run_llmtrace_fab989d622da48a89c6e5566e2121251_'
        '20260805.json'
    )
)
_ARTIFACT_ROOT = (
    _ROOT
    / 'test_artifacts'
    / 'cognition_core_v2_captured_run_failures'
)
_TRACE_ID = 'llmtrace_fab989d622da48a89c6e5566e2121251'
_STAGE_NAME = 'goal_cognition.ordinary_response.initial'
_CASE_ID = 'current_run_goal_relational_willingness_contract'
_EXPECTED_ERROR = (
    'non-sensitive relational willingness must be not_applicable with '
    'not_applicable relationship state'
)
_CAPTURED_OCCURRED_AT = '2026-08-05T15:33:55Z'


class _CapturingLLM:
    """Capture current goal-model calls without changing their behavior."""

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


def _load_captured_goal_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the goal payload and failed attempt from the protected export."""

    if not _CAPTURED_TRACE_PATH.exists():
        raise AssertionError(
            f'captured production trace is missing: {_CAPTURED_TRACE_PATH}'
        )
    trace = json.loads(
        _CAPTURED_TRACE_PATH.read_text(encoding='utf-8')
    )
    capsules = trace.get('cognition_failure_capsules') or []
    for capsule in capsules:
        if capsule.get('trace_id') != _TRACE_ID:
            continue
        for attempt in capsule.get('attempts', []):
            if attempt.get('stage_name') != _STAGE_NAME:
                continue
            messages = attempt.get('messages')
            if not isinstance(messages, list):
                raise AssertionError('captured goal messages are not a list')
            human_messages = [
                message.get('content')
                for message in messages
                if (
                    isinstance(message, dict)
                    and message.get('role') == 'human'
                )
            ]
            if not human_messages or not isinstance(human_messages[0], str):
                raise AssertionError('captured goal human payload is missing')
            payload = json.loads(human_messages[0])
            if not isinstance(payload, dict):
                raise AssertionError('captured goal payload is not an object')
            return payload, {
                'validation_error': str(attempt.get('validation_error') or ''),
                'historical_output': attempt.get('parsed_output'),
            }
    raise AssertionError(
        f'captured goal stage {_STAGE_NAME} is missing from the trace'
    )


def _replay_evidence_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild typed evidence rows from the captured goal projection."""

    raw_evidence = payload.get('evidence')
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise AssertionError('captured goal payload has no evidence rows')
    evidence: list[dict[str, Any]] = []
    for row in raw_evidence:
        if not isinstance(row, dict):
            raise AssertionError('captured goal evidence row is not an object')
        handle = str(row.get('handle', ''))
        source_kind = str(row.get('source_kind', ''))
        semantic_text = str(row.get('semantic_text', ''))
        if source_kind not in EVIDENCE_SOURCE_QUESTION_IDS:
            raise AssertionError(
                f'captured evidence source kind is unsupported: {source_kind}'
            )
        evidence.append({
            'evidence_handle': handle,
            'evidence_ref': {
                'source_kind': source_kind,
                'source_id': f'captured-run:{handle}',
                'occurred_at': _CAPTURED_OCCURRED_AT,
                'semantic_summary': semantic_text[:200],
            },
            'semantic_text': semantic_text,
            'visible_to': list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
        })
    return evidence


def _replay_semantic_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore role bindings required by the current goal builder."""

    semantic_context = payload.get('semantic_context')
    if not isinstance(semantic_context, dict):
        raise AssertionError('captured goal semantic context is missing')
    context = deepcopy(semantic_context)
    role_handles = payload.get('role_handles')
    if not isinstance(role_handles, list):
        raise AssertionError('captured goal role handles are missing')
    role_bindings: dict[str, dict[str, str]] = {}
    for handle in role_handles:
        if handle == 'self':
            role_bindings[handle] = {
                'role': 'actor',
                'entity_kind': 'character',
                'entity_id': 'character:global',
            }
        elif handle == 'current_user':
            role_bindings[handle] = {
                'role': 'target',
                'entity_kind': 'user',
                'entity_id': 'replay:current-user',
            }
        else:
            role_bindings[handle] = {
                'role': 'target',
                'entity_kind': 'user',
                'entity_id': f'replay:{handle}',
            }
    context['_role_bindings'] = role_bindings
    context['role_summaries'] = dict(payload.get('role_summaries') or {})
    return context


def _write_artifact(artifact: dict[str, Any]) -> Path:
    """Write the current-prompt replay evidence for human inspection."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_ROOT / f'{_CASE_ID}_{time_ns()}.json'
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str)
        + '\n',
        encoding='utf-8',
    )
    return path


async def test_captured_run_goal_relational_willingness_repair_live_llm(
) -> None:
    """Replay the captured non-sensitive relational-willingness repair."""

    payload, historical = _load_captured_goal_payload()
    evidence = _replay_evidence_rows(payload)
    semantic_context = _replay_semantic_context(payload)
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    bid: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    try:
        result = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
            {
                'scope': 'user',
                'kind': 'goal',
                'entity_id': 'goal:current-run-replay',
            },
            semantic_context,
            evidence,
            services,
        )
        if isinstance(result, dict):
            bid = result
    except CognitionExecutionError as exc:
        failure = {
            'error_code': exc.error_code,
            'message': str(exc),
            'attempt_count': exc.attempt_count,
        }
    artifact = {
        'schema_version': 'cognition_core_v2_captured_run_goal_replay.v1',
        'case_id': _CASE_ID,
        'input_kind': 'captured_production_stage_boundary',
        'source_trace_id': _TRACE_ID,
        'source_trace_path': str(_CAPTURED_TRACE_PATH),
        'stage_name': _STAGE_NAME,
        'historical_validation_error': historical['validation_error'],
        'historical_output': historical['historical_output'],
        'model_calls': capturing_llm.calls,
        'observed_failure': failure,
        'validated_bid': bid,
        'expected_contract': {
            'applicability': 'not_relationship_sensitive',
            'current_user_relationship_state': 'not_applicable',
            'stance': 'not_applicable',
        },
    }
    artifact_path = _write_artifact(artifact)
    if failure is not None:
        pytest.fail(f'captured goal repair failed; artifact={artifact_path}')
    if bid is None:
        pytest.fail(
            f'captured goal replay returned no bid; artifact={artifact_path}'
        )
    relational = bid.get('relational_willingness')
    assert isinstance(relational, dict)
    assert relational['applicability'] == 'not_relationship_sensitive'
    assert relational['current_user_relationship_state'] == 'not_applicable'
    assert relational['stance'] == 'not_applicable'
