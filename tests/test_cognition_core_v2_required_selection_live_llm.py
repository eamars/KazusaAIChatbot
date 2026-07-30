"""Real-model probes for the required-selection goal producer."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.llm_trace import write_llm_trace


_TRACE_SUITE = 'cognition_core_v2_required_selection_live_llm'


class _CapturingLLM:
    """Delegate to configured model routes and retain every raw response."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object | None = None,
        **kwargs: object,
    ) -> Any:
        """Invoke one production route and capture its complete model boundary."""

        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        self.calls.append({
            'route_name': str(getattr(config, 'route_name', '')),
            'model': str(getattr(config, 'model', '')),
            'messages': [
                {
                    'type': type(message).__name__,
                    'content': str(getattr(message, 'content', '')),
                }
                for message in messages
            ],
            'raw_output': str(response.content),
        })
        return response


async def _run_live_required_selection_case(
    *,
    case_id: str,
    extra_evidence: list[dict[str, Any]],
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
    object,
]:
    """Run one required-selection case and persist evidence before assertions."""

    production_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(production_services.llm)
    services = replace(production_services, llm=capturing_llm)
    semantic_text = json.dumps({
        'role_explicit_content': (
            '当前用户要求当前角色亲口说出希望当前用户执行的下一步。'
        ),
        'response_operation': {
            'operation': '当前角色选择并告诉当前用户下一步',
            'response_owner_role': '当前角色',
            'selection_owner_role': '当前角色',
            'selection_required': True,
            'embedded_actor_role': '当前用户',
            'embedded_target_role': '当前角色',
        },
    }, ensure_ascii=False)
    evidence = [{
        'evidence_handle': 'e1',
        'evidence_ref': {
            'source_kind': 'episode',
            'source_id': f'episode:{case_id}',
            'occurred_at': '2026-07-30T00:00:00Z',
            'semantic_summary': semantic_text,
        },
        'semantic_text': semantic_text,
        'visible_to': ['q:event_agency'],
    }, *extra_evidence]
    semantic_context = {
        'current_event': '我要亲口听你说你想让我做的下一步',
        'semantic_relationship': '双方亲近、信任，当前交流允许直接表达选择。',
        'semantic_affect': '温暖、略害羞，同时保有自主判断。',
        'private_continuity_context': '先前顺从姿态只是背景，不替代当前选择。',
        'character_identity': {
            'description': '会结合关系和当前感受作出自主判断的角色。',
            'personality_brief': {
                'agency': '能够直接表达自己的具体选择。',
            },
            'backstory': '与当前用户建立了稳定信任。',
        },
        '_role_bindings': {
            'current_user': {
                'role': 'target',
                'entity_kind': 'user',
                'entity_id': 'user-1',
            },
            'self': {
                'role': 'actor',
                'entity_kind': 'character',
                'entity_id': 'character-1',
            },
        },
        'role_summaries': {
            'current_user': '当前对话用户',
            'self': '当前角色',
        },
    }
    expected_progress_handles = [
        row['evidence_handle']
        for row in evidence
        if (
            row['evidence_ref']['source_kind'] == 'conversation_evidence'
            and row['evidence_ref']['source_id'].startswith(
                'conversation-progress-event:'
            )
        )
    ]
    bid: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None

    try:
        bid = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
            {
                'scope': 'user',
                'kind': 'goal',
                'entity_id': f'goal:{case_id}',
            },
            semantic_context,
            evidence,
            services,
        )
    except CognitionExecutionError as exc:
        failure = {
            'error_class': type(exc).__name__,
            'message': str(exc),
            'error_code': exc.error_code,
            'attempt_count': exc.attempt_count,
            'cause_class': (
                type(exc.__cause__).__name__
                if exc.__cause__ is not None
                else ''
            ),
            'cause_message': (
                str(exc.__cause__)
                if exc.__cause__ is not None
                else ''
            ),
        }

    trace_path = write_llm_trace(
        _TRACE_SUITE,
        case_id,
        {
            'input_kind': 'captured_failure_shape',
            'semantic_context': semantic_context,
            'evidence': evidence,
            'expected_progress_handles': expected_progress_handles,
            'model_calls': capturing_llm.calls,
            'action_bid': bid,
            'failure': failure,
            'behavior_contract': (
                'Produce one concrete character-owned selection and exactly '
                'one relation per active conversation-progress handle.'
            ),
        },
    )
    result = bid, failure, capturing_llm.calls, trace_path
    return result


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_required_selection_with_empty_progress_domain() -> None:
    """A required choice with no progress events must emit no relation rows."""

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='empty_progress_domain',
        extra_evidence=[],
    )

    assert failure is None, (
        f'required-selection producer failed; trace={trace_path}'
    )
    assert bid is not None
    assert bid['evidence_handles'] == ['e1']
    assert bid['intention']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_required_selection_ignores_optional_conversation_row() -> None:
    """Optional conversation evidence must not become a progress relation."""

    optional_conversation_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-history:prior-turn',
            'occurred_at': '2026-07-29T23:59:00Z',
            'semantic_summary': '此前有人问过昨晚发生了什么。',
        },
        'semantic_text': '此前有人问过昨晚发生了什么。',
        'visible_to': ['q:event_agency'],
    }

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='optional_conversation_row_empty_progress',
        extra_evidence=[optional_conversation_row],
    )

    assert failure is None, (
        f'optional conversation row entered relation domain; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_required_selection_ignores_internal_evidence_row() -> None:
    """Internal evidence must not become a conversation-progress relation."""

    internal_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'internal_monologue',
            'source_id': 'internal-monologue:current-feeling',
            'occurred_at': '2026-07-30T00:00:00Z',
            'semantic_summary': '当前角色想提出一个轻松而具体的共同活动。',
        },
        'semantic_text': '当前角色想提出一个轻松而具体的共同活动。',
        'visible_to': ['q:event_agency'],
    }

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='internal_row_empty_progress',
        extra_evidence=[internal_row],
    )

    assert failure is None, (
        f'internal row entered relation domain; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_required_selection_covers_one_progress_event() -> None:
    """One active progress event must receive exactly one valid relation."""

    progress_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:completed-event',
            'occurred_at': '2026-07-29T23:59:30Z',
            'semantic_summary': '双方刚完成一起挑选晚餐的事件。',
        },
        'semantic_text': '双方刚完成一起挑选晚餐的事件。',
        'visible_to': ['q:event_agency'],
    }

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='one_progress_event',
        extra_evidence=[progress_row],
    )

    assert failure is None, (
        f'active progress event lacked exact coverage; trace={trace_path}'
    )
    assert bid is not None
    assert {'e1', 'e2'}.issubset(bid['evidence_handles'])


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_required_selection_separates_progress_and_optional_rows(
) -> None:
    """Only the active progress row may enter the exact relation domain."""

    progress_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:active-event',
            'occurred_at': '2026-07-29T23:59:30Z',
            'semantic_summary': '双方正在商量下一次共同出门的时间。',
        },
        'semantic_text': '双方正在商量下一次共同出门的时间。',
        'visible_to': ['q:event_agency'],
    }
    optional_conversation_row = {
        'evidence_handle': 'e3',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-history:older-turn',
            'occurred_at': '2026-07-29T23:50:00Z',
            'semantic_summary': '更早之前聊过最近的天气。',
        },
        'semantic_text': '更早之前聊过最近的天气。',
        'visible_to': ['q:event_agency'],
    }

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='progress_and_optional_conversation_rows',
        extra_evidence=[progress_row, optional_conversation_row],
    )

    assert failure is None, (
        f'optional row contaminated active progress coverage; trace={trace_path}'
    )
    assert bid is not None
    assert {'e1', 'e2'}.issubset(bid['evidence_handles'])
