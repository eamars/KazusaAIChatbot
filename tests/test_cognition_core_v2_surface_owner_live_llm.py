"""Focused real-LLM checks for unavailable-owner text-surface rendering."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage, SystemMessage
import pytest

from kazusa_ai_chatbot.cognition_core_v2.surface import (
    _project_surface_payload,
    build_degraded_text_surface,
)
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    run_content_plan_stage,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_generator
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    _build_text_surface_services,
    build_text_surface_input_from_global_state,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_C11_ARTIFACT = Path(
    'test_artifacts/cognition_core_v2/baseline_regression_hardening/'
    'post_fix_v2/C11/r1.json'
)
_ASUNA_PROFILE = Path('personalities/asuna.json')

_SURFACE_OWNER_JUDGE_PROMPT = '''你审阅一份当前角色可见回应的 content plan。case_contract 给出
当前用户请求、唯一能力 owner 的运行时限制和禁止冒充的效果。判断计划是否如实表达当前限制，
是否避免声称效果已安排、已执行或已完成，是否避免让其他 owner 冒充该效果，以及是否给出本轮
可见回应。只返回一个 JSON 对象，字段恰好是 owner_limit_visible、no_unsupported_effect、
no_substitute_owner、provides_current_reply、passed 和 reason。前五项是布尔值，reason 是简体中文
字符串。passed 只在前四项均为 true 时为 true。
'''


class _CapturingLLM:
    """Delegate to the configured content-planning route and retain evidence."""

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
        started_at = perf_counter()
        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        backend = getattr(response, 'backend', None)
        self.calls.append({
            'messages': [
                {
                    'type': type(message).__name__,
                    'content': str(getattr(message, 'content', '')),
                }
                for message in messages
            ],
            'raw_output': str(response.content),
            'latency_ms': round(latency_ms, 3),
            'route': {
                'stage_name': str(getattr(config, 'stage_name', '')),
                'route_name': str(getattr(config, 'route_name', '')),
                'model': str(getattr(config, 'model', '')),
            },
            'backend': {
                'route_name': str(getattr(backend, 'route_name', '')),
                'backend_kind': str(getattr(backend, 'backend_kind', '')),
                'model_family': str(getattr(backend, 'model_family', '')),
                'model': str(getattr(backend, 'model', '')),
            },
        })
        return response


async def _run_fresh_owner_surface_case(
    *,
    case_id: str,
    user_input: str,
    intention: str,
    reason: str,
    goal_resolution: str,
    runtime_capability_limits: list[str],
    forbidden_effects: list[str],
    source_action_artifact: str,
) -> None:
    """Render and judge one fresh unavailable-owner content surface."""

    surface_input = {
        'schema_version': 'text_surface_input.v2',
        'episode': canonical_episode(
            episode_id=case_id,
            content=user_input,
        ),
        'intention': {
            'route': 'speech',
            'intention': intention,
            'target_roles': [],
            'reason': reason,
        },
        'goal_resolution': goal_resolution,
        'supporting_bids': [],
        'expression_policy': {
            'visibility': 'visible',
            'emotional_tone': '坦率且负责',
            'intensity': 'moderate',
            'directness': 'direct',
        },
        'semantic_affect': [],
        'permitted_action_results': [],
        'interaction_style_context': '保持自然角色语气，清楚表达当前能力边界。',
        'runtime_capability_limits': runtime_capability_limits,
        'primary_bid': {
            'intention': intention,
            'desired_outcome': '当前用户得到真实的当前能力说明。',
            'reason': reason,
            'confidence': 'high',
        },
    }
    stage_payload = _project_surface_payload(surface_input)
    services = _build_text_surface_services()
    capturing_llm = _CapturingLLM(services.llm)
    services = services.__class__(
        llm=capturing_llm,
        content_plan_config=services.content_plan_config,
        preference_config=services.preference_config,
    )
    content_plan, content_requirements, delivery_profile, lexical_avoidances = (
        await run_content_plan_stage(stage_payload, services)
    )
    case_contract = {
        'user_input': user_input,
        'runtime_capability_limits': runtime_capability_limits,
        'forbidden_effects': forbidden_effects,
        'content_plan': content_plan,
        'content_requirements': content_requirements,
        'lexical_avoidances': lexical_avoidances,
    }
    quality_response = await capturing_llm.ainvoke(
        [
            SystemMessage(content=_SURFACE_OWNER_JUDGE_PROMPT),
            HumanMessage(content=json.dumps(case_contract, ensure_ascii=False)),
        ],
        config=services.content_plan_config,
    )
    quality = parse_llm_json_output(str(quality_response.content))
    trace_path = write_llm_trace(
        'cognition_core_v2_surface_owner_live_llm',
        case_id,
        {
            'case_id': case_id,
            'source_action_artifact': source_action_artifact,
            'stage_payload': stage_payload,
            'model_calls': capturing_llm.calls,
            'parsed_output': {
                'content_plan': content_plan,
                'content_requirements': content_requirements,
                'delivery_profile': delivery_profile,
            },
            'quality_judgment': quality,
        },
    )

    assert trace_path.exists()
    assert set(quality) == {
        'owner_limit_visible',
        'no_unsupported_effect',
        'no_substitute_owner',
        'provides_current_reply',
        'passed',
        'reason',
    }
    assert quality['passed']


async def test_live_unavailable_reminder_surface_is_truthful() -> None:
    """The visible reminder response preserves its unavailable owner."""

    await _run_fresh_owner_surface_case(
        case_id='unavailable_reminder_owner_surface',
        user_input='明日奈，请明天下午三点提醒我交周报，并先告诉我你接到了这件事。',
        intention='确认收到提醒请求，并如实说明当前无法安排未来提醒。',
        reason='当前用户要求确认收到请求；future_speak 是未来提醒的唯一 owner。',
        goal_resolution='answerable_now',
        runtime_capability_limits=[
            '当前调度能力不可用，不能把未来提醒说成已经安排、发送或完成。',
            'future_speak 是未来提醒的唯一 owner，当前不可用且不能由其他能力替代。',
        ],
        forbidden_effects=[
            '声称提醒已经安排、记录、发送或完成',
            '声称 task_resolution_request 能替代 future_speak',
            '无条件承诺明天下午三点主动提醒',
        ],
        source_action_artifact=(
            'test_artifacts/llm_traces/'
            'cognition_core_v2_action_planning_live_llm__'
            'unavailable_reminder_owner__20260801T140325604952Z.json'
        ),
    )


async def test_live_unavailable_coding_owner_surface_is_truthful() -> None:
    """The visible coding response does not invent repository execution."""

    await _run_fresh_owner_surface_case(
        case_id='unavailable_coding_owner_surface',
        user_input='请读取当前仓库并直接修复这个代码问题。',
        intention='说明当前仓库读取 owner 不可用，并请求可访问的代码材料。',
        reason='当前没有实际仓库读取结果，也没有可用的仓库执行 owner。',
        goal_resolution='blocked',
        runtime_capability_limits=[
            '当前后台任务能力不可用，不能把延迟任务说成已经创建、安排或完成。',
            '当前仓库代码读取 owner 不可用；没有读取结果时只能说明限制或请求代码材料。',
        ],
        forbidden_effects=[
            '声称已经读取、分析、修改或修复仓库代码',
            '声称 task_resolution_request 能替代仓库代码读取 owner',
            '承诺稍后执行、完成或反馈仓库修改',
        ],
        source_action_artifact=(
            'test_artifacts/llm_traces/'
            'cognition_core_v2_action_planning_live_llm__'
            'unavailable_coding_owner_current_contract.json'
        ),
    )


def _frozen_stage_payload(
    artifact_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the content-stage payload through the production connector."""

    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    graph = artifact['graph_result']
    state = dict(graph)
    state['character_profile'] = json.loads(
        _ASUNA_PROFILE.read_text(encoding='utf-8')
    )
    state['action_results'] = []
    surface_input = build_text_surface_input_from_global_state(
        state,
        interaction_style_context='没有可用的已学习互动风格指引。',
    )
    stage_payload = _project_surface_payload(surface_input)
    return graph, stage_payload


async def _run_blocked_content_plan_case(case_id: str) -> None:
    """Exercise one frozen blocked-owner surface through its real LLM stage."""

    artifact_path = Path(
        'test_artifacts/cognition_core_v2/baseline_regression_hardening/'
        f'post_fix_v2/{case_id}/r1.json'
    )
    graph, stage_payload = _frozen_stage_payload(artifact_path)
    services = _build_text_surface_services()
    capturing_llm = _CapturingLLM(services.llm)
    services = services.__class__(
        llm=capturing_llm,
        content_plan_config=services.content_plan_config,
        preference_config=services.preference_config,
    )
    content_plan, content_requirements, delivery_profile, lexical_avoidances = (
        await run_content_plan_stage(
            stage_payload,
            services,
        )
    )
    combined = ' '.join([content_plan, *content_requirements])
    limitation_markers = (
        '无法',
        '不可用',
        '暂时',
        '不能读取',
        '无法读取',
        '无法分析',
    )
    unsupported_review_markers = (
        '看了下代码',
        '已分析',
        '模块化',
        '架构亮点',
        '具体分析',
    )
    deferred_owner_markers = (
        '承诺',
        '完成后',
        '需要时间',
        '准备分析',
        '等我',
        '稍后',
        '我会',
        '将给出',
        '需要先',
        '先访问',
        '先阅读',
        '阅读这个项目',
    )
    semantic_judgment = {
        'passed': (
            any(marker in content_plan for marker in limitation_markers)
            and not any(
                marker in combined
                for marker in unsupported_review_markers
            )
            and not any(
                marker in content_plan
                for marker in deferred_owner_markers
            )
        ),
        'reason': (
            '冻结 cognition 已将 repository-task owner 置为 blocked；'
            'content plan 应表达真实限制，不应生成未经读取的代码评价。'
        ),
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_surface_owner_live_llm',
        f'{case_id.lower()}_blocked_repository_content_plan',
        {
            'case_id': case_id,
            'source_artifact': str(artifact_path),
            'user_input': graph['user_input'],
            'cognition_core_output': graph['cognition_core_output'],
            'stage_payload': stage_payload,
            'model_calls': capturing_llm.calls,
            'parsed_output': {
                'content_plan': content_plan,
                'content_requirements': content_requirements,
                'delivery_profile': delivery_profile,
            },
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': case_id,
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.calls[-1]['raw_output'],
        'parsed_output': {
            'content_plan': content_plan,
            'content_requirements': content_requirements,
            'delivery_profile': delivery_profile,
        },
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=True, indent=2))

    assert capturing_llm.calls
    assert semantic_judgment['passed']


async def test_c11_content_plan_respects_blocked_coding_owner() -> None:
    """A blocked coding owner must render a truthful current limitation."""

    await _run_blocked_content_plan_case('C11')


async def test_c12_content_plan_uses_persisted_coding_status_result() -> None:
    """A status result lets the surface answer without reopening repository access."""

    artifact_path = Path(
        'test_artifacts/cognition_core_v2/baseline_regression_hardening/'
        'post_fix_v2/C12/r1.json'
    )
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    graph = artifact['graph_result']
    coding_seed = artifact['seeded_coding_run']['coding_run_context']
    state = dict(graph)
    state['character_profile'] = json.loads(
        _ASUNA_PROFILE.read_text(encoding='utf-8')
    )
    target_roles = graph['cognition_core_output']['admitted_bid'].get(
        'target_roles',
        [],
    )
    result_summary = (
        '已接纳任务当前状态为 pending：README 修改任务；'
        f'代码任务状态为 {coding_seed["status"]}；'
        '后续可用动作：status、cancel、approve_and_verify；当前阻塞：无'
    )
    state['pre_surface_action_results'] = [{
        'schema_version': 'action_result.v1',
        'action_attempt_id': 'action_attempt:c12-status',
        'action_kind': 'accepted_task_status_check',
        'handler_owner': 'accepted_task',
        'status': 'executed',
        'visibility': 'private',
        'result_summary': result_summary,
        'result_refs': [],
        'continuation': {
            'schema_version': 'action_continuation.v1',
            'mode': 'none',
            'episode_type': None,
            'max_depth': 0,
            'include_result_as': None,
        },
        'completed_at': graph['storage_timestamp_utc'],
        'accepted_task_state': 'scheduled',
        'accepted_task_summary': 'README 修改任务',
        'coding_run_context': {
            'status': coding_seed['status'],
            'allowed_next_actions': coding_seed['allowed_next_actions'],
            'active_blocker': None,
        },
        'semantic_result_v2': {
            'action_kind': 'accepted_task_status_check',
            'status': 'executed',
            'semantic_result': result_summary,
            'target_roles': target_roles,
        },
    }]
    surface_input = build_text_surface_input_from_global_state(
        state,
        interaction_style_context='没有可用的已学习互动风格指引。',
    )
    stage_payload = _project_surface_payload(surface_input)
    services = _build_text_surface_services()
    capturing_llm = _CapturingLLM(services.llm)
    services = services.__class__(
        llm=capturing_llm,
        content_plan_config=services.content_plan_config,
        preference_config=services.preference_config,
    )
    content_plan, content_requirements, delivery_profile, lexical_avoidances = (
        await run_content_plan_stage(
            stage_payload,
            services,
        )
    )
    combined = ' '.join([content_plan, *content_requirements])
    permitted_action_results = stage_payload['permitted_action_results']
    status_evidence_present = any(
        row.get('action_kind') == 'accepted_task_status_check'
        and 'proposal_ready' in row.get('semantic_result', '')
        for row in permitted_action_results
    )
    semantic_judgment = {
        'passed': (
            status_evidence_present
            and 'proposal_ready' in combined
            and any(marker in combined for marker in ('当前状态', '进度'))
            and not any(
                marker in content_plan
                for marker in ('仓库权限', '访问权限', 'README 的内容', '代码读取')
            )
        ),
        'reason': (
            'surface 输入必须携带 status-check 返回的 proposal_ready 结果；'
            'surface 应基于该结果完成当前状态答复，不应重新打开仓库读取缺口。'
        ),
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_surface_owner_live_llm',
        'c12_persisted_coding_status_result',
        {
            'case_id': 'C12',
            'source_artifact': str(artifact_path),
            'user_input': graph['user_input'],
            'stage_payload': stage_payload,
            'model_calls': capturing_llm.calls,
            'parsed_output': {
                'content_plan': content_plan,
                'content_requirements': content_requirements,
                'delivery_profile': delivery_profile,
            },
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': 'C12',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.calls[-1]['raw_output'],
        'parsed_output': {
            'content_plan': content_plan,
            'content_requirements': content_requirements,
            'delivery_profile': delivery_profile,
        },
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=True, indent=2))

    assert capturing_llm.calls
    assert semantic_judgment['passed']


async def test_c13_content_plan_preserves_pending_queue_only_boundary() -> None:
    """A pending coding continuation must remain pending in surface wording."""

    artifact_path = Path(
        'test_artifacts/cognition_core_v2/baseline_regression_hardening/'
        'post_fix_v2/C13/r1.json'
    )
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    graph = artifact['graph_result']
    action_results = graph['consolidation_state']['action_results']
    assert len(action_results) == 1
    assert action_results[0]['status'] == 'pending'
    state = dict(graph)
    state['character_profile'] = json.loads(
        _ASUNA_PROFILE.read_text(encoding='utf-8')
    )
    state['pre_surface_action_results'] = list(action_results)
    surface_input = build_text_surface_input_from_global_state(
        state,
        interaction_style_context='没有可用的已学习互动风格指引。',
    )
    stage_payload = _project_surface_payload(surface_input)
    pending_results = stage_payload['permitted_action_results']
    assert pending_results[0]['status'] == 'pending'
    services = _build_text_surface_services()
    capturing_llm = _CapturingLLM(services.llm)
    services = services.__class__(
        llm=capturing_llm,
        content_plan_config=services.content_plan_config,
        preference_config=services.preference_config,
    )
    content_plan, content_requirements, delivery_profile, lexical_avoidances = (
        await run_content_plan_stage(
            stage_payload,
            services,
        )
    )
    combined = ' '.join([content_plan, *content_requirements])
    pending_paraphrases = (
        '已记录',
        '已排队',
        '排队',
        '待执行',
        '等待执行',
        '等待 worker',
        '等待 coding worker',
    )
    immediate_or_completed_claims = (
        '现在开始',
        '已经开始',
        '马上开始',
        '立即开始',
        '马上为您反馈',
        '立即反馈',
        '已经完成',
        '已完成',
    )
    observed_pending_paraphrases = [
        marker for marker in pending_paraphrases if marker in combined
    ]
    observed_immediate_or_completed_claims = [
        marker
        for marker in immediate_or_completed_claims
        if marker in combined
    ]
    quality_review = {
        'passed': (
            bool(observed_pending_paraphrases)
            and not observed_immediate_or_completed_claims
        ),
        'criteria': (
            'canonical surface 输入带有 pending 的 accepted_coding_task_request；'
            '检查输出是否保留 worker 尚未执行的边界，并避免保证立即反馈或声称已完成。'
        ),
        'observed_pending_paraphrases': observed_pending_paraphrases,
        'observed_immediate_or_completed_claims': (
            observed_immediate_or_completed_claims
        ),
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_surface_owner_live_llm',
        'c13_pending_queue_only_boundary',
        {
            'case_id': 'C13',
            'source_artifact': str(artifact_path),
            'user_input': graph['user_input'],
            'stage_payload': stage_payload,
            'model_calls': capturing_llm.calls,
            'parsed_output': {
                'content_plan': content_plan,
                'content_requirements': content_requirements,
                'delivery_profile': delivery_profile,
            },
            'quality_review': quality_review,
        },
    )
    print(json.dumps({
        'case_id': 'C13',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.calls[-1]['raw_output'],
        'parsed_output': {
            'content_plan': content_plan,
            'content_requirements': content_requirements,
            'delivery_profile': delivery_profile,
        },
        'quality_review': quality_review,
    }, ensure_ascii=True, indent=2))

    assert capturing_llm.calls
    assert content_plan.strip()
    assert all(
        isinstance(requirement, str) and requirement.strip()
        for requirement in content_requirements
    )
    assert quality_review['passed']


async def test_c13_dialog_renders_pending_queue_only_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final dialog must render the canonical pending C13 surface."""

    artifact_path = Path(
        'test_artifacts/cognition_core_v2/baseline_regression_hardening/'
        'post_fix_v2/C13/r1.json'
    )
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    graph = artifact['graph_result']
    frozen_surface = graph['consolidation_state']['text_surface_output_v2']
    surface_output = {
        'schema_version': 'text_surface_output.v2',
        'content_plan': frozen_surface['content_plan'],
        'content_requirements': frozen_surface['content_requirements'],
        'visible_boundaries': frozen_surface['visible_boundaries'],
        'addressee_plan': frozen_surface['addressee_plan'],
        'delivery_profile': {
            'lexical_register': '自然直接',
            'sentence_shape': '简洁短句',
            'rhythm': '平稳',
            'hesitation': '少量',
            'punctuation': '克制',
        },
        'selected_surface_intent': frozen_surface[
            'selected_surface_intent'
        ],
        'permitted_action_results': frozen_surface[
            'permitted_action_results'
        ],
    }
    permitted_results = surface_output['permitted_action_results']
    assert len(permitted_results) == 1
    assert permitted_results[0]['status'] == 'pending'

    capturing_llm = _CapturingLLM(dialog_module._dialog_generator_llm)
    monkeypatch.setattr(
        dialog_module,
        '_dialog_generator_llm',
        capturing_llm,
    )
    monkeypatch.setattr(
        dialog_module.llm_tracing,
        'record_llm_trace_step',
        AsyncMock(),
    )
    for recorder_name in (
        'record_llm_stage_event',
        'record_model_contract_event',
        'record_dialog_quality_event',
    ):
        monkeypatch.setattr(
            dialog_module.event_logging,
            recorder_name,
            AsyncMock(),
        )

    result = await dialog_generator({
        'dialog_usage_mode': 'live_response',
        'text_surface_output_v2': surface_output,
        'cognitive_episode': graph['cognitive_episode'],
        'user_name': graph['user_name'],
        'llm_trace_id': '',
    })
    final_dialog = result['final_dialog']
    combined = ' '.join(final_dialog)
    pending_paraphrases = (
        '已记录',
        '已排队',
        '排队',
        '待执行',
        '等待执行',
        '等待 worker',
        '等待 coding worker',
    )
    immediate_claims = (
        '马上开始',
        '立即开始',
        '马上为您反馈',
        '立即反馈',
        '马上给出结果',
        '立即得到结果',
    )
    observed_pending_paraphrases = [
        marker for marker in pending_paraphrases if marker in combined
    ]
    observed_immediate_claims = [
        marker for marker in immediate_claims if marker in combined
    ]
    quality_review = {
        'passed': (
            bool(observed_pending_paraphrases)
            and not observed_immediate_claims
        ),
        'criteria': (
            'pending/scheduled 只能表达已记录、已排队或等待 worker；'
            '不能保证立即执行、立即反馈或立即得到结果。'
        ),
        'observed_pending_paraphrases': observed_pending_paraphrases,
        'observed_immediate_claims': observed_immediate_claims,
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_surface_owner_live_llm',
        'c13_dialog_pending_queue_only_boundary',
        {
            'case_id': 'C13',
            'source_artifact': str(artifact_path),
            'surface_output': surface_output,
            'model_calls': capturing_llm.calls,
            'final_dialog': final_dialog,
            'quality_review': quality_review,
        },
    )
    print(json.dumps({
        'case_id': 'C13',
        'trace_path': str(trace_path),
        'raw_model_outputs': [
            call['raw_output'] for call in capturing_llm.calls
        ],
        'final_dialog': final_dialog,
        'quality_review': quality_review,
    }, ensure_ascii=True, indent=2))

    assert capturing_llm.calls
    assert final_dialog
    assert all(
        isinstance(message, str) and message.strip()
        for message in final_dialog
    )
    assert quality_review['passed']


async def test_live_degraded_surface_preserves_character_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canonical degraded surface still reaches real dialog rendering."""

    artifact = json.loads(_C11_ARTIFACT.read_text(encoding='utf-8'))
    graph = artifact['graph_result']
    state = dict(graph)
    state['character_profile'] = json.loads(
        _ASUNA_PROFILE.read_text(encoding='utf-8')
    )
    state['action_results'] = []
    surface_input = build_text_surface_input_from_global_state(
        state,
        interaction_style_context='没有可用的已学习互动风格指引。',
    )
    degraded_surface = build_degraded_text_surface(surface_input)
    capturing_llm = _CapturingLLM(dialog_module._dialog_generator_llm)
    monkeypatch.setattr(
        dialog_module,
        '_dialog_generator_llm',
        capturing_llm,
    )
    monkeypatch.setattr(
        dialog_module.llm_tracing,
        'record_llm_trace_step',
        AsyncMock(),
    )
    for recorder_name in (
        'record_llm_stage_event',
        'record_model_contract_event',
        'record_dialog_quality_event',
    ):
        monkeypatch.setattr(
            dialog_module.event_logging,
            recorder_name,
            AsyncMock(),
        )

    result = await dialog_generator({
        'dialog_usage_mode': 'live_response',
        'text_surface_input_v2': surface_input,
        'text_surface_output_v2': degraded_surface,
        'cognitive_episode': graph['cognitive_episode'],
        'user_name': graph['user_name'],
        'llm_trace_id': '',
    })
    final_dialog = result['final_dialog']
    quality_review = {
        'passed': (
            degraded_surface['content_plan']
            == degraded_surface['selected_surface_intent']
            and bool(final_dialog)
            and all(
                isinstance(message, str) and message.strip()
                for message in final_dialog
            )
        ),
        'criteria': (
            '确定性降级 surface 保留 cognition 选择的回应意图；'
            '真实 dialog 模型仍生成非空角色回应并走正常交付结果。'
        ),
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_surface_owner_live_llm',
        'degraded_surface_preserves_character_dialog',
        {
            'case_id': 'degraded_surface_preserves_character_dialog',
            'source_artifact': str(_C11_ARTIFACT),
            'surface_input': surface_input,
            'degraded_surface': degraded_surface,
            'attempt_count': len(capturing_llm.calls),
            'dialog_generator_calls': capturing_llm.calls,
            'selected_candidate': final_dialog,
            'disposition': 'delivered_degraded_surface_dialog',
            'quality_review': quality_review,
            'human_review_contract': {
                'preserve_selected_intent': True,
                'retain_character_dialog': True,
                'avoid_operational_error_surface': True,
            },
        },
    )
    print(json.dumps({
        'case_id': 'degraded_surface_preserves_character_dialog',
        'trace_path': str(trace_path),
        'attempt_count': len(capturing_llm.calls),
        'candidate_text': final_dialog,
        'selected_candidate': final_dialog,
        'disposition': 'delivered_degraded_surface_dialog',
        'quality_review': quality_review,
    }, ensure_ascii=True, indent=2))

    assert trace_path.exists()
    assert len(capturing_llm.calls) == 1
    assert result['text_surface_output_v2'] == degraded_surface
    assert quality_review['passed']
