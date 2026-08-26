"""Live regression gate for the cognition P-stage response-goal contract."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    CanonicalContractError,
    bind_protected_chain_records,
    reset_protected_chain_records,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)

pytestmark = pytest.mark.live_llm

_ARTIFACT_DIRECTORY = Path(
    'test_artifacts/diagnostics/cognition_v3_response_goal_contract_live_llm'
)
_CAPTURED_TRACE_PATH = Path('test_artifacts/diagnostics/trace_08-46.json')
_REPLAY_ATTEMPTS = 3


def _captured_failure_shape() -> dict[str, object]:
    """Load the exact P packet from the protected production trace export."""

    assert _CAPTURED_TRACE_PATH.is_file(), (
        f'captured trace is unavailable: {_CAPTURED_TRACE_PATH}'
    )
    with _CAPTURED_TRACE_PATH.open(encoding='utf-8') as handle:
        trace = json.load(handle)
    p_step = next(
        step
        for step in trace['llm_trace_steps']
        if step['stage_name'] == 'cognition_core_v3.P'
    )
    human_message = next(
        message
        for message in p_step['raw_messages']
        if message['role'] == 'human'
    )
    packet = json.loads(human_message['content'])
    return packet


def _write_artifact(value: dict[str, Any]) -> Path:
    """Persist one inspectable live replay artifact as UTF-8 JSON."""

    _ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_DIRECTORY / f'response_goal_{time.time_ns()}.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    return path


async def test_live_captured_p_response_goal_is_bounded_text() -> None:
    """Require repeated captured-shape P outputs to satisfy the text contract."""

    packet = _captured_failure_shape()
    services = build_cognition_core_services()
    trace_token = bind_protected_chain_records(
        run_id=f'p-response-goal-live-{time.time_ns()}',
        source_kind='captured_p_response_goal_contract_live_test',
    )
    outputs: list[dict[str, object]] = []
    validation_errors: list[str] = []
    try:
        for _ in range(_REPLAY_ATTEMPTS):
            parsed = await facade_module._call_once(
                services=services,
                stage='P',
                packet=packet,
            )
            outputs.append(parsed)
            try:
                facade_module._validate_plan(
                    parsed,
                    self_cognition=False,
                    capabilities=packet['capabilities'],
                )
            except CanonicalContractError as exc:
                validation_errors.append(str(exc))
    finally:
        protected_records = list(snapshot_protected_chain_records())
        reset_protected_chain_records(trace_token)
        artifact = {
            'schema': 'cognition_v3_response_goal_contract_live_test.v1',
            'case_id': 'captured_p_response_goal_object_copy',
            'input_kind': 'protected_captured_failure',
            'behavior_contract': (
                'P returns one non-empty response_goal string for downstream '
                'surface generation.'
            ),
            'packet': packet,
            'route': services.chain_lane.route_name,
            'model': services.chain_lane.model,
            'outputs': outputs,
            'validation_errors': validation_errors,
            'protected_records': protected_records,
        }
        artifact_path = _write_artifact(artifact)
        print(f'live cognition artifact: {artifact_path}')

    assert len(outputs) == _REPLAY_ATTEMPTS
    assert validation_errors == []
    assert all(
        isinstance(output['response_goal'], str)
        and output['response_goal'].strip()
        for output in outputs
    )
