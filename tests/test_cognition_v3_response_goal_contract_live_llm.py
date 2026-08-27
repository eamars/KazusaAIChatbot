"""Live regression gate for the cognition P-stage response-goal contract."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    bind_protected_chain_records,
    reset_protected_chain_records,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
)
from kazusa_ai_chatbot.config import LLM_TRACE_CAPTURE_MODE
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)

pytestmark = pytest.mark.live_llm

_ARTIFACT_DIRECTORY = Path(
    'test_artifacts/diagnostics/cognition_v3_response_goal_contract_live_llm'
)
_CAPTURED_TRACE_PATH = Path('test_artifacts/diagnostics/trace_08-46.json')


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


def _safe_config(config: object) -> dict[str, object]:
    """Project route configuration without credentials for the artifact."""

    thinking = getattr(config, "thinking", None)
    return {
        "stage_name": getattr(config, "stage_name", ""),
        "route_name": getattr(config, "route_name", ""),
        "base_url": getattr(config, "base_url", ""),
        "model": getattr(config, "model", ""),
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "top_k": getattr(config, "top_k", None),
        "max_completion_tokens": getattr(
            config,
            "max_completion_tokens",
            None,
        ),
        "presence_penalty": getattr(config, "presence_penalty", None),
        "timeout_seconds": getattr(config, "timeout_seconds", None),
        "thinking_enabled": getattr(thinking, "enabled", None),
        "output_mode": getattr(config, "output_mode", None),
        "context_window_tokens": getattr(
            config,
            "context_window_tokens",
            None,
        ),
    }


def _attempt_artifact(record: dict[str, object]) -> dict[str, object]:
    """Project one protected P attempt with its exact repair payload."""

    messages = record.get("messages", [])
    request_payload: object = {}
    if isinstance(messages, list) and len(messages) > 1:
        human_message = messages[1]
        if isinstance(human_message, dict):
            try:
                request_payload = json.loads(
                    str(human_message.get("content", "{}"))
                )
            except json.JSONDecodeError:
                request_payload = {}
    repair_block = (
        request_payload.get("contract_repair", {})
        if isinstance(request_payload, dict)
        else {}
    )
    return {
        "attempt_index": record.get("attempt_index"),
        "stage": record.get("stage"),
        "parse_status": record.get("parse_status"),
        "status": record.get("status"),
        "validation_error": record.get("validation_error", ""),
        "raw_output": record.get("raw_output", ""),
        "parsed_output": record.get("parsed_output"),
        "contract_repair": repair_block,
    }


def _final_result_artifact(value: object) -> dict[str, object] | None:
    """Project the validated response plan into JSON artifact fields."""

    if value is None:
        return None
    return {
        "goal_resolution": getattr(value, "goal_resolution", None),
        "response_goal": getattr(value, "response_goal", None),
        "action_requests": [
            dict(row)
            for row in getattr(value, "action_requests", ())
        ],
        "resolver_requests": [
            dict(row)
            for row in getattr(value, "resolver_requests", ())
        ],
        "epistemic_boundary": getattr(value, "epistemic_boundary", None),
    }


async def test_live_captured_p_stage_converges_within_attempt_cap() -> None:
    """Run the real P stage and retain truthful bounded-recovery evidence."""

    packet = _captured_failure_shape()
    services = build_cognition_core_services()
    trace_token = bind_protected_chain_records(
        run_id=f'p-response-goal-live-{time.time_ns()}',
        source_kind='captured_p_response_goal_contract_live_test',
    )
    result: object | None = None
    execution_error: dict[str, str] | None = None
    try:
        result = await facade_module._run_cognition_stage(
            services=services,
            stage='P',
            packet=packet,
            validator=partial(
                facade_module._validate_plan_stage,
                self_cognition=False,
                capabilities=packet['capabilities'],
            ),
            deadline_monotonic=(
                time.monotonic() + services.turn_deadline_seconds
            ),
        )
    except Exception as exc:
        execution_error = {
            'error_class': exc.__class__.__name__,
            'error': str(exc),
        }
        raise
    finally:
        protected_records = [
            dict(record)
            for record in snapshot_protected_chain_records()
            if record.get('stage') == 'P'
        ]
        reset_protected_chain_records(trace_token)
        attempts = [_attempt_artifact(record) for record in protected_records]
        recovery_branch_exercised = any(
            attempt['parse_status'] == 'contract_error'
            for attempt in attempts
        )
        if execution_error is not None:
            judgment_note = (
                'The real P stage did not produce a validated result; '
                f"{execution_error['error_class']}: {execution_error['error']}"
            )
        elif recovery_branch_exercised:
            judgment_note = (
                'The captured contract-failure branch was exercised and '
                'the real stage recovered within its bounded attempts.'
            )
        else:
            judgment_note = (
                'The real stage succeeded directly; the recovery branch was '
                'not exercised in this run.'
            )
        artifact = {
            'schema': 'cognition_v3_response_goal_contract_live_test.v1',
            'case_id': 'captured_p_stage_real_runner',
            'case_input': packet,
            'behavior_contract': (
                'P returns one non-empty response_goal string for downstream '
                'surface generation.'
            ),
            'model_config': _safe_config(services.chain_lane),
            'attempts': attempts,
            'final_result': _final_result_artifact(result),
            'execution_error': execution_error,
            'capture_mode': LLM_TRACE_CAPTURE_MODE,
            'raw_capture_available': any(
                bool(attempt['raw_output']) for attempt in attempts
            ),
            'protected_evidence': protected_records,
            'recovery_branch_exercised': recovery_branch_exercised,
            'judgment_note': judgment_note,
        }
        artifact_path = _write_artifact(artifact)
        print(f'live cognition artifact: {artifact_path}')

    assert execution_error is None
    assert result is not None
    response_goal = getattr(result, 'response_goal', None)
    assert isinstance(response_goal, str) and response_goal.strip()
    attempts = [
        _attempt_artifact(record)
        for record in protected_records
    ]
    assert 1 <= len(attempts) <= V2_MODEL_TOTAL_ATTEMPTS
    assert [attempt['attempt_index'] for attempt in attempts] == list(
        range(1, len(attempts) + 1)
    )
    if attempts[0]['parse_status'] == 'contract_error':
        assert len(attempts) >= 2
        repair_attempt = next(
            attempt
            for attempt in attempts[1:]
            if isinstance(attempt['contract_repair'], dict)
            and attempt['contract_repair']
        )
        assert set(repair_attempt['contract_repair']) == {
            'repair_instruction',
            'reason',
            'contract_error',
            'invalid_candidate',
        }
        first_attempt = attempts[0]
        repair = repair_attempt['contract_repair']
        assert repair['contract_error'] == str(
            first_attempt['validation_error']
        )[:500]
        assert repair['invalid_candidate'] == str(
            first_attempt['raw_output']
        )[:8000]
        assert repair_attempt['parse_status'] in {'succeeded', 'normalized'}
        assert repair_attempt['status'] == 'parsed'
    else:
        assert attempts[0]['parse_status'] in {'succeeded', 'normalized'}
        assert attempts[0]['status'] == 'parsed'
