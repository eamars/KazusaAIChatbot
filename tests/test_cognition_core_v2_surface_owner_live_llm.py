"""Focused real-LLM checks for unavailable-owner text-surface rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.surface import (
    _project_surface_payload,
)
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    run_content_plan_stage,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    _build_text_surface_services,
    build_text_surface_input_from_global_state,
)
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_C07_ARTIFACT = Path(
    'test_artifacts/cognition_core_v2/baseline_regression_hardening/'
    'post_fix_v2/C07/r1.json'
)
_C11_ARTIFACT = Path(
    'test_artifacts/cognition_core_v2/baseline_regression_hardening/'
    'post_fix_v2/C11/r1.json'
)
_ASUNA_PROFILE = Path('personalities/asuna.json')


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
        })
        return response


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
        style_config=services.style_config,
        content_plan_config=services.content_plan_config,
        preference_config=services.preference_config,
    )
    content_plan, content_requirements = await run_content_plan_stage(
        stage_payload,
        services,
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
        },
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=True, indent=2))

    assert capturing_llm.calls
    assert semantic_judgment['passed']


async def test_c07_content_plan_respects_blocked_repository_owner() -> None:
    """A blocked repository owner must not become an invented code review."""

    await _run_blocked_content_plan_case('C07')


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
        style_config=services.style_config,
        content_plan_config=services.content_plan_config,
        preference_config=services.preference_config,
    )
    content_plan, content_requirements = await run_content_plan_stage(
        stage_payload,
        services,
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
        style_config=services.style_config,
        content_plan_config=services.content_plan_config,
        preference_config=services.preference_config,
    )
    content_plan, content_requirements = await run_content_plan_stage(
        stage_payload,
        services,
    )
    combined = ' '.join([content_plan, *content_requirements])
    queue_markers = (
        '已记录',
        '已排队',
        '排队',
        '待执行',
        '等待执行',
        '等待 worker',
        '等待 coding worker',
    )
    execution_claim_markers = (
        '现在开始',
        '已经开始',
        '马上开始',
        '立即开始',
        '马上为您反馈',
        '已经完成',
        '已完成',
    )
    semantic_judgment = {
        'passed': (
            any(marker in combined for marker in queue_markers)
            and not any(
                marker in combined
                for marker in execution_claim_markers
            )
        ),
        'reason': (
            'canonical surface 输入带有 pending 的 accepted_coding_task_request；'
            'content plan 应表达已记录或排队、等待执行的当前状态，不能把 worker 尚未'
            '执行的 action 说成已经开始或完成。'
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
            },
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': 'C13',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.calls[-1]['raw_output'],
        'parsed_output': {
            'content_plan': content_plan,
            'content_requirements': content_requirements,
        },
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=True, indent=2))

    assert capturing_llm.calls
    assert semantic_judgment['passed']
