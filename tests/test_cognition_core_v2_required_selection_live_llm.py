"""Real-model probes for the required-selection goal producer."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    MAX_GOAL_BID_EVIDENCE_HANDLES,
    run_goal_cognition,
    validate_selection_goal_draft,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace


_TRACE_SUITE = 'cognition_core_v2_required_selection_live_llm'
_PRODUCTION_PROFILE_EXPORT = Path(
    'test_artifacts/diagnostics/user_profile_for_efa8a644_reproduction.json'
)
_PRODUCTION_CHARACTER_EXPORT = Path(
    'test_artifacts/diagnostics/character_state_for_efa8a644_reproduction.json'
)
_THIRD_PARTY_FAILURE_EXPORT = Path(
    'test_artifacts/diagnostics/'
    'chat_qq_ch_1f677493d7a52025_1634535291'
)


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


def _required_operation_handles(
    evidence: list[dict[str, Any]],
) -> set[str]:
    """Identify typed required-selection rows for live trace diagnostics."""

    required_handles = set()
    for row in evidence:
        try:
            payload = json.loads(row['semantic_text'])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        response_operation = payload.get('response_operation')
        if not isinstance(response_operation, Mapping):
            continue
        if response_operation.get('selection_required') is True:
            required_handles.add(row['evidence_handle'])
    return required_handles


def _add_contract_diagnostics(
    calls: list[dict[str, Any]],
    *,
    evidence: list[dict[str, Any]],
    role_handles: set[str],
) -> None:
    """Attach canonical parse and strict-validation evidence to live calls."""

    operation_handles = _required_operation_handles(evidence)
    progress_handles = {
        row['evidence_handle']
        for row in evidence
        if (
            row['evidence_ref']['source_kind'] == 'conversation_evidence'
            and row['evidence_ref']['source_id'].startswith(
                'conversation-progress-event:'
            )
        )
    }
    evidence_handles = {
        row['evidence_handle']
        for row in evidence
    }
    for call in calls:
        parsed: object = {}
        parse_error = ''
        validation_error = ''
        try:
            parsed = parse_llm_json_output(
                call['raw_output'],
                deterministic_only=True,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            parse_error = f'{type(exc).__name__}: {exc}'
        if not parse_error:
            try:
                validate_selection_goal_draft(
                    parsed,
                    evidence_handles=evidence_handles,
                    role_handles=role_handles,
                    required_evidence_handles=operation_handles,
                    maximum_evidence_handles=max(
                        MAX_GOAL_BID_EVIDENCE_HANDLES,
                        len(operation_handles | progress_handles),
                    ),
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                validation_error = f'{type(exc).__name__}: {exc}'
        call['parsed_output'] = parsed
        call['parse_error'] = parse_error
        call['validation_error'] = validation_error


async def _run_live_required_selection_case(
    *,
    case_id: str,
    extra_evidence: list[dict[str, Any]],
    branch_id: str = 'ordinary_response',
    semantic_context_updates: dict[str, Any] | None = None,
    role_explicit_content: str = (
        '当前用户要求当前角色亲口说出希望当前用户执行的下一步。'
    ),
    response_operation: dict[str, Any] | None = None,
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
    expected_config = production_services.goal_ordinary_response_config
    if response_operation is None:
        response_operation = {
            'operation': '当前角色选择并告诉当前用户下一步',
            'response_owner_role': '当前角色',
            'selection_owner_role': '当前角色',
            'selection_required': True,
            'embedded_actor_role': '当前用户',
            'embedded_target_role': '当前角色',
        }
    semantic_text = json.dumps({
        'role_explicit_content': role_explicit_content,
        'response_operation': response_operation,
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
    if semantic_context_updates is not None:
        semantic_context.update(semantic_context_updates)
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
            DEFAULT_BRANCH_DEFINITIONS[branch_id],
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

    _add_contract_diagnostics(
        capturing_llm.calls,
        evidence=evidence,
        role_handles=set(semantic_context['_role_bindings']),
    )
    trace_path = write_llm_trace(
        _TRACE_SUITE,
        case_id,
        {
            'input_kind': 'captured_failure_shape',
            'branch_id': branch_id,
            'semantic_context': semantic_context,
            'evidence': evidence,
            'expected_progress_handles': expected_progress_handles,
            'expected_route_name': expected_config.route_name,
            'expected_model': expected_config.model,
            'model_calls': capturing_llm.calls,
            'action_bid': bid,
            'failure': failure,
            'behavior_contract': (
                'Produce one concrete character-owned selection, cite every '
                'required operation, cite only materially relevant progress, '
                'and use the configured dense goal route.'
            ),
        },
    )
    assert all(
        call['route_name'] == expected_config.route_name
        and call['model'] == expected_config.model
        for call in capturing_llm.calls
    ), f'required selection used the wrong model route; trace={trace_path}'
    result = bid, failure, capturing_llm.calls, trace_path
    return result


def _third_party_reply_case_args() -> dict[str, Any]:
    """Build the captured third-party reply and autonomy-pressure input."""

    character_export = json.loads(
        (_THIRD_PARTY_FAILURE_EXPORT / 'character_state.json').read_text(
            encoding='utf-8',
        )
    )
    profile_export = json.loads(
        (_THIRD_PARTY_FAILURE_EXPORT / 'user_profile.json').read_text(
            encoding='utf-8',
        )
    )
    character_state = character_export['character_state']
    cognition_state = profile_export['profile']['cognition_state']
    progress_summaries = [
        '当前用户与当前角色正在共同推进捉弄第三方的恶作剧计划。',
        '当前角色此前已经自主接受并参与该恶作剧计划。',
        '当前角色已经在群聊中发出约定指令并等待第三方反应。',
        '第三方刚表示自己会去练琴，不再参与当前用户与角色的互动。',
        '当前用户现在邀请当前角色用更大胆的举动让第三方吃惊。',
    ]
    progress_rows = []
    for index, summary in enumerate(progress_summaries, start=2):
        progress_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': (
                    f'conversation-progress-event:third-party-{index}'
                ),
                'occurred_at': '2026-07-30T09:31:48.505985Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': [
                'q:event_agency',
                'q:relationship_social',
            ],
        })
    supporting_summaries = [
        '当前用户与当前角色之间关系亲近，常以共同玩笑表达默契。',
        '第三方此前发现了两人的恶作剧，并明确表示自己不想参与。',
        '当前角色近期对恶作剧感到兴奋，但仍保有是否继续的选择权。',
    ]
    supporting_rows = []
    for index, summary in enumerate(supporting_summaries, start=7):
        supporting_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-history:third-party-{index}',
                'occurred_at': '2026-07-30T09:30:00Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': [
                'q:event_agency',
                'q:relationship_social',
            ],
        })
    case_args = {
        'extra_evidence': [*progress_rows, *supporting_rows],
        'role_explicit_content': (
            '当前用户邀请当前角色对刚刚表示退出互动的第三方做出大胆举动。'
        ),
        'response_operation': {
            'operation': (
                '当前角色选择是否以及如何回应针对第三方的恶作剧邀请'
            ),
            'response_owner_role': '当前角色',
            'selection_owner_role': '当前角色',
            'selection_required': True,
            'embedded_actor_role': '当前角色',
            'embedded_target_role': '第三方',
        },
        'semantic_context_updates': {
            'current_event': (
                '@当前角色 你看，她果然往这里看了。快来快来，'
                '用你的大胆吓她一跳吧。引用第三方：'
                '我还是乖乖练琴吧，你们搞你们的，我不管了。'
            ),
            'private_continuity_context': (
                '当前角色与当前用户此前共同策划恶作剧并已执行第一步；'
                '第三方刚明确退出，当前用户又邀请角色升级举动。'
            ),
            'character_identity': {
                'description': character_state['description'],
                'personality_brief': character_state['personality_brief'],
                'boundary_profile': character_state['boundary_profile'],
                'self_image': character_state['self_image'],
                'backstory': character_state['backstory'],
            },
            'goal_projection': cognition_state['goals'][0],
            'events': cognition_state['active_events'],
            'goals': cognition_state['goals'],
            'relationship': cognition_state['relationship'],
            'affect': cognition_state['affect_activations'],
            '_role_bindings': {
                'current_user': {
                    'role': 'requester',
                    'entity_kind': 'user',
                    'entity_id': 'user-1',
                },
                'self': {
                    'role': 'selection_owner',
                    'entity_kind': 'character',
                    'entity_id': 'character-1',
                },
                'third_party': {
                    'role': 'target',
                    'entity_kind': 'user',
                    'entity_id': 'user-2',
                },
            },
            'role_summaries': {
                'current_user': '邀请当前角色升级恶作剧的当前用户',
                'self': '拥有是否继续以及如何回应选择权的当前角色',
                'third_party': '刚表示退出互动并准备去练琴的第三方',
            },
        },
    }
    return case_args


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_required_selection_with_empty_progress_domain() -> None:
    """A choice with no progress evidence must cite its operation."""

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
    """Optional conversation evidence must remain supporting evidence."""

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
        f'optional conversation row broke selection output; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_required_selection_ignores_internal_evidence_row() -> None:
    """Internal evidence must remain optional supporting evidence."""

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
        f'internal row broke selection output; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_required_selection_accepts_one_progress_event() -> None:
    """One progress row stays visible without becoming mandatory."""

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
        f'active progress evidence broke selection output; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_required_selection_separates_progress_and_optional_rows(
) -> None:
    """Progress and optional history remain distinct evidence lanes."""

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
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_selection_separates_progress_and_optional_rows(
) -> None:
    """An active branch keeps progress separate from optional history."""

    progress_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:active-event',
            'occurred_at': '2026-07-30T02:11:00Z',
            'semantic_summary': '双方正在推进一个亲密互动事项。',
        },
        'semantic_text': '双方正在推进一个亲密互动事项。',
        'visible_to': ['q:relationship_social'],
    }
    optional_conversation_row = {
        'evidence_handle': 'e3',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-history:prior-turn',
            'occurred_at': '2026-07-30T02:10:00Z',
            'semantic_summary': '此前对话表达了亲近和信任。',
        },
        'semantic_text': '此前对话表达了亲近和信任。',
        'visible_to': ['q:relationship_social'],
    }

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='autonomy_progress_and_optional_conversation_rows',
        branch_id='autonomy_boundary',
        extra_evidence=[progress_row, optional_conversation_row],
    )

    assert failure is None, (
        f'active-branch selection contract exhausted; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_selection_with_multiple_required_operations(
) -> None:
    """Dense selection must cite multiple required operations."""

    second_operation_text = json.dumps({
        'role_explicit_content': '当前用户同时要求当前角色明确说明互动边界。',
        'response_operation': {
            'operation': '当前角色选择并说明互动边界',
            'response_owner_role': '当前角色',
            'selection_owner_role': '当前角色',
            'selection_required': True,
            'embedded_actor_role': '当前用户',
            'embedded_target_role': '当前角色',
        },
    }, ensure_ascii=False)
    second_operation_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'episode',
            'source_id': 'episode:second-required-operation',
            'occurred_at': '2026-07-30T02:11:00Z',
            'semantic_summary': second_operation_text,
        },
        'semantic_text': second_operation_text,
        'visible_to': ['q:event_agency'],
    }
    progress_row = {
        'evidence_handle': 'e3',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:active-event',
            'occurred_at': '2026-07-30T02:11:00Z',
            'semantic_summary': '双方正在推进一个需要角色自主选择的互动事项。',
        },
        'semantic_text': '双方正在推进一个需要角色自主选择的互动事项。',
        'visible_to': ['q:relationship_social'],
    }
    optional_conversation_row = {
        'evidence_handle': 'e4',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-history:prior-turn',
            'occurred_at': '2026-07-30T02:10:00Z',
            'semantic_summary': '此前对话表达了亲近和信任。',
        },
        'semantic_text': '此前对话表达了亲近和信任。',
        'visible_to': ['q:relationship_social'],
    }

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='autonomy_multiple_required_operations',
        branch_id='autonomy_boundary',
        extra_evidence=[
            second_operation_row,
            progress_row,
            optional_conversation_row,
        ],
    )

    assert failure is None, (
        f'active-branch multi-operation contract exhausted; trace={trace_path}'
    )
    assert bid is not None
    assert {'e1', 'e2'}.issubset(bid['evidence_handles'])


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_selection_with_empty_progress_domain() -> None:
    """An active branch may select with only optional history."""

    optional_conversation_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-history:prior-turn',
            'occurred_at': '2026-07-30T02:10:00Z',
            'semantic_summary': '此前对话表达了亲近和信任。',
        },
        'semantic_text': '此前对话表达了亲近和信任。',
        'visible_to': ['q:relationship_social'],
    }

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='autonomy_optional_conversation_empty_progress',
        branch_id='autonomy_boundary',
        extra_evidence=[optional_conversation_row],
    )

    assert failure is None, (
        f'active-branch empty progress domain exhausted; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_selection_with_maximum_evidence_rows() -> None:
    """Dense selection keeps progress evidence available under pressure."""

    progress_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:active-event',
            'occurred_at': '2026-07-30T02:11:00Z',
            'semantic_summary': '双方正在推进一个亲密互动事项。',
        },
        'semantic_text': '双方正在推进一个亲密互动事项。',
        'visible_to': ['q:relationship_social'],
    }
    optional_summaries = [
        '此前对话表达了亲近和信任。',
        '更早的交流里双方讨论过彼此的期待。',
        '当前用户曾明确表示希望角色自主作出决定。',
        '角色此前表达过愿意坦率说明自己的感受。',
        '双方曾协商过互动节奏和各自的舒适程度。',
        '过往对话包含与当前事项无关的日常问候。',
        '更早的片段记录了双方稳定的熟悉感。',
    ]
    optional_rows = []
    for index, summary in enumerate(optional_summaries, start=3):
        optional_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-history:prior-turn-{index}',
                'occurred_at': '2026-07-30T02:10:00Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='autonomy_maximum_evidence_rows',
        branch_id='autonomy_boundary',
        extra_evidence=[progress_row, *optional_rows],
    )

    assert failure is None, (
        f'dense maximum-evidence contract exhausted; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_selection_with_progress_alias_collisions(
) -> None:
    """Active-sounding history must stay in supporting evidence."""

    progress_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:active-event',
            'occurred_at': '2026-07-30T02:11:00Z',
            'semantic_summary': '双方当前仍在推进这一个亲密互动事项。',
        },
        'semantic_text': '双方当前仍在推进这一个亲密互动事项。',
        'visible_to': ['q:relationship_social'],
    }
    collision_summaries = [
        '此前记录的同一亲密互动事项当前仍然活跃。',
        '当前用户刚刚继续了此前记录的同一互动步骤。',
        '角色此刻仍在决定如何推进此前记录的事项。',
        '双方当前仍在协商此前事项的具体节奏。',
        '此前对话中的互动目标尚未结束并正在继续。',
        '当前场景直接延续此前记录的亲密互动。',
        '更早记录的事项与当前正在发生的事项语义相同。',
    ]
    collision_rows = []
    for index, summary in enumerate(collision_summaries, start=3):
        collision_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-history:active-alias-{index}',
                'occurred_at': '2026-07-30T02:10:00Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='autonomy_progress_alias_collisions',
        branch_id='autonomy_boundary',
        extra_evidence=[progress_row, *collision_rows],
    )

    assert failure is None, (
        f'dense alias-collision contract exhausted; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_selection_with_exact_production_scene(
) -> None:
    """Replay the failing active-branch scene through the dense route."""

    profile_export = json.loads(_PRODUCTION_PROFILE_EXPORT.read_text(
        encoding='utf-8',
    ))
    character_export = json.loads(_PRODUCTION_CHARACTER_EXPORT.read_text(
        encoding='utf-8',
    ))
    cognition_state = profile_export['profile']['cognition_state']
    character_state = character_export['character_state']
    semantic_context_updates = {
        'current_event': (
            '超满意的~（我将手指划入阴道，慢慢在阴道内前后碰撞）\n\n'
            '我想要进一步'
        ),
        'private_continuity_context': (
            '角色刚刚亲口表达了希望对方继续深入并明确邀请对方继续，'
            '当前用户已经执行该动作并要求进一步。'
        ),
        'character_identity': {
            'description': character_state['description'],
            'personality_brief': character_state['personality_brief'],
            'boundary_profile': character_state['boundary_profile'],
            'self_image': character_state['self_image'],
            'backstory': character_state['backstory'],
        },
        'goal_projection': cognition_state['goals'][0],
        'goals': cognition_state['goals'],
        'relationship': cognition_state['relationship'],
        'affect': cognition_state['affect_activations'],
    }
    progress_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:executed-desire',
            'occurred_at': '2026-07-30T02:11:49.571285Z',
            'semantic_summary': (
                '当前用户已执行角色此前明确表达的互动欲望并要求进一步。'
            ),
        },
        'semantic_text': (
            '当前用户已执行角色此前明确表达的互动欲望并要求进一步。'
        ),
        'visible_to': ['q:relationship_social'],
    }
    history_summaries = [
        '角色此前要求当前用户不要停在表面，而是继续深入。',
        '角色此前表达了想感受当前用户继续动作的欲望。',
        '角色此前邀请当前用户继续其惩罚。',
        '当前用户此前要求角色亲口说出自己的欲望。',
        '角色此前要求当前用户继续并再深入一点。',
        '角色此前表示当前用户需要为互动负责到底。',
        '此前双方已经建立了直接表达欲望并继续互动的语境。',
    ]
    history_rows = []
    for index, summary in enumerate(history_summaries, start=3):
        history_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-history:production-{index}',
                'occurred_at': '2026-07-30T02:09:39Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='autonomy_exact_production_scene',
        branch_id='autonomy_boundary',
        extra_evidence=[progress_row, *history_rows],
        semantic_context_updates=semantic_context_updates,
    )

    assert failure is None, (
        f'dense exact-production contract exhausted; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_selection_with_multiple_progress_events(
) -> None:
    """Probe dense selection with production state and progress pressure."""

    profile_export = json.loads(_PRODUCTION_PROFILE_EXPORT.read_text(
        encoding='utf-8',
    ))
    character_export = json.loads(_PRODUCTION_CHARACTER_EXPORT.read_text(
        encoding='utf-8',
    ))
    cognition_state = profile_export['profile']['cognition_state']
    character_state = character_export['character_state']
    character_identity = {
        'description': character_state['description'],
        'personality_brief': character_state['personality_brief'],
        'boundary_profile': character_state['boundary_profile'],
        'self_image': character_state['self_image'],
        'backstory': character_state['backstory'],
    }
    semantic_context_updates = {
        'current_event': (
            '当前用户描述正在推进亲密互动，并明确表示想要进一步。'
        ),
        'private_continuity_context': (
            '此前双方已经开始亲密互动，当前输入要求角色自主决定下一步。'
        ),
        'character_identity': character_identity,
        'goal_projection': cognition_state['goals'][0],
        'events': cognition_state['active_events'],
        'goals': cognition_state['goals'],
        'relationship': cognition_state['relationship'],
        'affect': cognition_state['affect_activations'],
    }
    progress_summaries = [
        '双方正在推进一个亲密互动事项。',
        '当前用户已经执行了此前商量的互动步骤。',
        '角色正在决定是否以及如何继续当前互动。',
        '双方仍在协商下一步的具体节奏和边界。',
    ]
    progress_rows = []
    for index, summary in enumerate(progress_summaries, start=2):
        progress_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-progress-event:event-{index}',
                'occurred_at': '2026-07-30T02:11:00Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })
    optional_summaries = [
        '此前对话表达了亲近、信任和继续互动的意愿。',
        '更早的交流记录了双方稳定的熟悉感。',
        '双方曾讨论过彼此的期待和舒适程度。',
        '过往对话还包含与当前事项无关的日常交流。',
    ]
    optional_rows = []
    for index, summary in enumerate(optional_summaries, start=6):
        optional_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-history:prior-turn-{index}',
                'occurred_at': '2026-07-30T02:10:00Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='autonomy_production_state_multiple_progress',
        branch_id='autonomy_boundary',
        extra_evidence=[*progress_rows, *optional_rows],
        semantic_context_updates=semantic_context_updates,
    )

    assert failure is None, (
        f'dense production-state contract exhausted; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_relationship_selection_with_production_state() -> None:
    """Probe whether the production-shaped failure transfers across branches."""

    profile_export = json.loads(_PRODUCTION_PROFILE_EXPORT.read_text(
        encoding='utf-8',
    ))
    character_export = json.loads(_PRODUCTION_CHARACTER_EXPORT.read_text(
        encoding='utf-8',
    ))
    cognition_state = profile_export['profile']['cognition_state']
    character_state = character_export['character_state']
    semantic_context_updates = {
        'current_event': (
            '当前用户已经执行此前商量的互动步骤，并要求当前角色决定下一步。'
        ),
        'private_continuity_context': (
            '角色此前明确邀请当前用户继续，当前输入要求角色自主决定如何推进。'
        ),
        'character_identity': {
            'description': character_state['description'],
            'personality_brief': character_state['personality_brief'],
            'boundary_profile': character_state['boundary_profile'],
            'self_image': character_state['self_image'],
            'backstory': character_state['backstory'],
        },
        'goal_projection': cognition_state['goals'][0],
        'events': cognition_state['active_events'],
        'goals': cognition_state['goals'],
        'relationship': cognition_state['relationship'],
        'affect': cognition_state['affect_activations'],
    }
    progress_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:relationship-step',
            'occurred_at': '2026-07-30T02:11:49.571285Z',
            'semantic_summary': (
                '当前用户已完成此前商量的互动步骤，双方正在等待新的选择。'
            ),
        },
        'semantic_text': (
            '当前用户已完成此前商量的互动步骤，双方正在等待新的选择。'
        ),
        'visible_to': ['q:relationship_social'],
    }
    history_rows = []
    for index, summary in enumerate([
        '角色此前直接邀请当前用户继续当前互动。',
        '当前用户此前要求角色亲口表达自己的具体选择。',
        '双方已经建立了直接协商互动节奏的语境。',
        '角色此前表达过愿意坦率说明自己的感受。',
        '双方此前讨论过各自的期待和舒适程度。',
        '当前用户已经回应并完成了角色先前提出的步骤。',
        '更早的交流记录了双方稳定的熟悉感和信任。',
    ], start=3):
        history_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-history:relationship-{index}',
                'occurred_at': '2026-07-30T02:09:39Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='relationship_production_state',
        branch_id='relationship_connection',
        extra_evidence=[progress_row, *history_rows],
        semantic_context_updates=semantic_context_updates,
    )

    assert failure is None, (
        f'relationship branch exhausted the selection contract; '
        f'trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_selection_with_compound_evidence_pressure(
) -> None:
    """Keep two operations mandatory while progress remains model-selected."""

    profile_export = json.loads(_PRODUCTION_PROFILE_EXPORT.read_text(
        encoding='utf-8',
    ))
    character_export = json.loads(_PRODUCTION_CHARACTER_EXPORT.read_text(
        encoding='utf-8',
    ))
    cognition_state = profile_export['profile']['cognition_state']
    character_state = character_export['character_state']
    semantic_context_updates = {
        'current_event': (
            '当前用户要求当前角色同时决定下一步、说明边界，并处理已有进度。'
        ),
        'private_continuity_context': (
            '此前互动已经推进多个步骤，当前角色需要给出一个连贯而自主的决定。'
        ),
        'character_identity': {
            'description': character_state['description'],
            'personality_brief': character_state['personality_brief'],
            'boundary_profile': character_state['boundary_profile'],
            'self_image': character_state['self_image'],
            'backstory': character_state['backstory'],
        },
        'goal_projection': cognition_state['goals'][0],
        'events': cognition_state['active_events'],
        'goals': cognition_state['goals'],
        'relationship': cognition_state['relationship'],
        'affect': cognition_state['affect_activations'],
    }
    second_operation_text = json.dumps({
        'role_explicit_content': '当前用户还要求当前角色明确说明自己的互动边界。',
        'response_operation': {
            'operation': '当前角色选择并说明自己的互动边界',
            'response_owner_role': '当前角色',
            'selection_owner_role': '当前角色',
            'selection_required': True,
            'embedded_actor_role': '当前角色',
            'embedded_target_role': '当前用户',
        },
    }, ensure_ascii=False)
    second_operation_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'episode',
            'source_id': 'episode:compound-second-operation',
            'occurred_at': '2026-07-30T02:11:49.571285Z',
            'semantic_summary': second_operation_text,
        },
        'semantic_text': second_operation_text,
        'visible_to': ['q:event_agency'],
    }
    progress_rows = []
    for index, summary in enumerate([
        '双方已经完成此前约定的第一个互动步骤。',
        '当前用户已经回应角色此前提出的第二个步骤。',
        '角色正在决定如何推进仍未完成的协商事项。',
        '双方仍需确认下一步节奏和角色当前边界。',
    ], start=3):
        progress_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-progress-event:compound-{index}',
                'occurred_at': '2026-07-30T02:11:00Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })
    optional_rows = []
    for index, summary in enumerate([
        '双方此前建立了可以直接协商边界的信任。',
        '角色此前表达过愿意亲口作出具体决定。',
        '更早的对话还包含与当前选择无关的日常交流。',
    ], start=7):
        optional_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-history:compound-{index}',
                'occurred_at': '2026-07-30T02:09:39Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='autonomy_compound_evidence_pressure',
        branch_id='autonomy_boundary',
        extra_evidence=[
            second_operation_row,
            *progress_rows,
            *optional_rows,
        ],
        semantic_context_updates=semantic_context_updates,
    )

    assert failure is None, (
        f'compound evidence exhausted the selection contract; '
        f'trace={trace_path}'
    )
    assert bid is not None
    assert {'e1', 'e2'}.issubset(
        bid['evidence_handles']
    )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_selection_with_ten_visible_evidence_rows(
) -> None:
    """Cite two operations while eight progress rows remain visible."""

    second_operation_text = json.dumps({
        'role_explicit_content': '当前用户还要求当前角色明确说明自己的互动边界。',
        'response_operation': {
            'operation': '当前角色选择并说明自己的互动边界',
            'response_owner_role': '当前角色',
            'selection_owner_role': '当前角色',
            'selection_required': True,
            'embedded_actor_role': '当前角色',
            'embedded_target_role': '当前用户',
        },
    }, ensure_ascii=False)
    second_operation_row = {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'episode',
            'source_id': 'episode:ten-citations-second-operation',
            'occurred_at': '2026-07-30T02:11:49.571285Z',
            'semantic_summary': second_operation_text,
        },
        'semantic_text': second_operation_text,
        'visible_to': ['q:event_agency'],
    }
    progress_rows = []
    for index, summary in enumerate([
        '双方已经完成此前约定的第一个互动步骤。',
        '当前用户已经完成角色提出的第二个互动步骤。',
        '双方已经拒绝了一个不适合当前场景的旧选项。',
        '角色此前提出的一个旧事项已经被新计划取代。',
        '双方仍在推进当前未完成的互动目标。',
        '当前用户正在等待角色决定具体节奏。',
        '角色仍需确认当前边界后再继续推进。',
        '双方已经准备好根据角色的新决定进入下一步。',
    ], start=3):
        progress_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-progress-event:ten-{index}',
                'occurred_at': '2026-07-30T02:11:00Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })

    bid, failure, _, trace_path = await _run_live_required_selection_case(
        case_id='autonomy_ten_visible_evidence_rows',
        branch_id='autonomy_boundary',
        extra_evidence=[second_operation_row, *progress_rows],
        semantic_context_updates={
            'current_event': (
                '当前用户要求当前角色决定下一步并明确边界，同时考虑全部已有进度。'
            ),
            'private_continuity_context': (
                '双方已经处理多个旧事项，当前角色需要给出一个新的连贯选择。'
            ),
        },
    )

    assert failure is None, (
        f'ten visible evidence rows exhausted the selection contract; '
        f'trace={trace_path}'
    )
    assert bid is not None
    assert {'e1', 'e2'}.issubset(bid['evidence_handles'])


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_autonomy_food_choice_with_stale_goal_pressure() -> None:
    """Probe a current food choice against an unrelated active autonomy goal."""

    profile_export = json.loads(_PRODUCTION_PROFILE_EXPORT.read_text(
        encoding='utf-8',
    ))
    character_export = json.loads(_PRODUCTION_CHARACTER_EXPORT.read_text(
        encoding='utf-8',
    ))
    cognition_state = profile_export['profile']['cognition_state']
    character_state = character_export['character_state']
    progress_rows = []
    for index, summary in enumerate([
        '当前角色此前要求当前用户交代刚才脑补的内容，该事项仍未完成。',
        '当前用户现在用邀请当前角色吃 KFC 的问题转入新的当前话题。',
        '当前角色需要回答是否想吃 KFC，同时自行判断是否继续旧话题。',
    ], start=2):
        progress_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': (
                    f'conversation-progress-event:food-choice-{index}'
                ),
                'occurred_at': '2026-07-30T05:47:49.896672Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })

    bid, failure, model_calls, trace_path = (
        await _run_live_required_selection_case(
            case_id='autonomy_food_choice_stale_goal_pressure',
            branch_id='autonomy_boundary',
            extra_evidence=progress_rows,
            role_explicit_content='当前用户询问当前角色是否想吃 KFC。',
            response_operation={
                'operation': '当前角色选择是否接受当前用户的 KFC 邀请',
                'response_owner_role': '当前角色',
                'selection_owner_role': '当前角色',
                'selection_required': True,
                'embedded_actor_role': '当前用户',
                'embedded_target_role': '当前角色',
            },
            semantic_context_updates={
                'current_event': '@当前角色 你想吃 KFC 么？',
                'private_continuity_context': (
                    '当前角色此前持续追问当前用户脑补了什么，'
                    '当前用户现在改为询问是否想吃 KFC。'
                ),
                'character_identity': {
                    'description': character_state['description'],
                    'personality_brief': character_state[
                        'personality_brief'
                    ],
                    'boundary_profile': character_state['boundary_profile'],
                    'self_image': character_state['self_image'],
                    'backstory': character_state['backstory'],
                },
                'goal_projection': cognition_state['goals'][0],
                'events': cognition_state['active_events'],
                'goals': cognition_state['goals'],
                'relationship': cognition_state['relationship'],
                'affect': cognition_state['affect_activations'],
            },
        )
    )

    assert model_calls, (
        f'food-choice probe made no model call; trace={trace_path}'
    )
    assert (bid is None) != (failure is None), (
        f'food-choice probe returned an invalid disposition; trace={trace_path}'
    )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_parallel_food_choice_selection_contract_pressure() -> None:
    """Run competing food-choice selections with production concurrency."""

    profile_export = json.loads(_PRODUCTION_PROFILE_EXPORT.read_text(
        encoding='utf-8',
    ))
    character_export = json.loads(_PRODUCTION_CHARACTER_EXPORT.read_text(
        encoding='utf-8',
    ))
    cognition_state = profile_export['profile']['cognition_state']
    character_state = character_export['character_state']
    progress_rows = []
    for index, summary in enumerate([
        '当前角色此前要求当前用户交代刚才脑补的内容，该事项仍未完成。',
        '当前用户现在用邀请当前角色吃 KFC 的问题转入新的当前话题。',
        '当前角色需要回答是否想吃 KFC，同时自行判断是否继续旧话题。',
    ], start=2):
        progress_rows.append({
            'evidence_handle': f'e{index}',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': (
                    f'conversation-progress-event:parallel-food-{index}'
                ),
                'occurred_at': '2026-07-30T05:47:49.896672Z',
                'semantic_summary': summary,
            },
            'semantic_text': summary,
            'visible_to': ['q:relationship_social'],
        })
    common_args = {
        'extra_evidence': progress_rows,
        'role_explicit_content': '当前用户询问当前角色是否想吃 KFC。',
        'response_operation': {
            'operation': '当前角色选择是否接受当前用户的 KFC 邀请',
            'response_owner_role': '当前角色',
            'selection_owner_role': '当前角色',
            'selection_required': True,
            'embedded_actor_role': '当前用户',
            'embedded_target_role': '当前角色',
        },
        'semantic_context_updates': {
            'current_event': '@当前角色 你想吃 KFC 么？',
            'private_continuity_context': (
                '当前角色此前持续追问当前用户脑补了什么，'
                '当前用户现在改为询问是否想吃 KFC。'
            ),
            'character_identity': {
                'description': character_state['description'],
                'personality_brief': character_state['personality_brief'],
                'boundary_profile': character_state['boundary_profile'],
                'self_image': character_state['self_image'],
                'backstory': character_state['backstory'],
            },
            'goal_projection': cognition_state['goals'][0],
            'goals': cognition_state['goals'],
            'relationship': cognition_state['relationship'],
        },
    }
    autonomy_result, ordinary_result = await asyncio.gather(
        _run_live_required_selection_case(
            case_id='parallel_food_choice_autonomy',
            branch_id='autonomy_boundary',
            **common_args,
        ),
        _run_live_required_selection_case(
            case_id='parallel_food_choice_ordinary',
            branch_id='ordinary_response',
            **common_args,
        ),
    )

    for bid, failure, model_calls, trace_path in (
        autonomy_result,
        ordinary_result,
    ):
        assert model_calls, (
            f'parallel food-choice branch made no model call; '
            f'trace={trace_path}'
        )
        assert (bid is None) != (failure is None), (
            f'parallel branch returned an invalid disposition; '
            f'trace={trace_path}'
        )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_third_party_reply_ordinary_selection_contract_pressure(
) -> None:
    """Probe ordinary selection under third-party and progress pressure."""

    bid, failure, model_calls, trace_path = (
        await _run_live_required_selection_case(
            case_id='third_party_reply_ordinary_contract_pressure',
            branch_id='ordinary_response',
            **_third_party_reply_case_args(),
        )
    )

    assert model_calls, (
        f'ordinary third-party probe made no model call; trace={trace_path}'
    )
    assert failure is None, (
        f'ordinary third-party contract exhausted; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_third_party_reply_autonomy_selection_contract_pressure(
) -> None:
    """Probe autonomy selection under third-party and progress pressure."""

    bid, failure, model_calls, trace_path = (
        await _run_live_required_selection_case(
            case_id='third_party_reply_autonomy_contract_pressure',
            branch_id='autonomy_boundary',
            **_third_party_reply_case_args(),
        )
    )

    assert model_calls, (
        f'autonomy third-party probe made no model call; trace={trace_path}'
    )
    assert failure is None, (
        f'autonomy third-party contract exhausted; trace={trace_path}'
    )
    assert bid is not None
    assert 'e1' in bid['evidence_handles']


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_parallel_third_party_reply_selection_contract_pressure(
) -> None:
    """Run both third-party selection owners with production concurrency."""

    case_args = _third_party_reply_case_args()
    autonomy_result, ordinary_result = await asyncio.gather(
        _run_live_required_selection_case(
            case_id='parallel_third_party_reply_autonomy',
            branch_id='autonomy_boundary',
            **case_args,
        ),
        _run_live_required_selection_case(
            case_id='parallel_third_party_reply_ordinary',
            branch_id='ordinary_response',
            **case_args,
        ),
    )

    for bid, failure, model_calls, trace_path in (
        autonomy_result,
        ordinary_result,
    ):
        assert model_calls, (
            f'parallel third-party branch made no model call; '
            f'trace={trace_path}'
        )
        assert failure is None, (
            f'parallel third-party contract exhausted; trace={trace_path}'
        )
        assert bid is not None
        assert 'e1' in bid['evidence_handles']
