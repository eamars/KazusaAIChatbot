"""Live LLM inspection for task-taking willingness prompt behavior."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.cognition_chain_core.contracts import LLMStageBinding
from kazusa_ai_chatbot.cognition_chain_core.stages import l2 as l2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2d as l2d_module
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_chain_services,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ServiceBinding = tuple[Callable[[Any], None], Any]
_ALLOWED_LOGICAL_STANCES = {
    'CONFIRM',
    'REFUSE',
    'TENTATIVE',
    'DIVERGE',
    'CHALLENGE',
}
_ALLOWED_CHARACTER_INTENTS = {
    'PROVIDE',
    'BANTAR',
    'REJECT',
    'EVADE',
    'CONFRONT',
    'DISMISS',
    'CLARIFY',
}
_ALLOWED_ACCEPTANCE = {'allow', 'guarded', 'hesitant', 'reject'}


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured cognition endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models",
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}",
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Skip the case when no live endpoint is available."""

    await _skip_if_llm_unavailable()


_L2_CASES = {
    'low_affinity_simple_help': {
        'user_input': '你能简单帮我看一下这句话怎么回吗？给个方向就好。',
        'affinity': 260,
        'mood': '有点累，但还能正常说话',
        'global_vibe': '放学后安静的私聊',
        'vibe_check': '低压力、普通求助',
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'internal_monologue': '只是简单问一句，不需要接下长期事情。',
        'interaction_subtext': '对方在寻求轻量建议，没有要求后续承诺。',
        'emotional_appraisal': '有点疲惫但不排斥简短回答。',
        'expectation': 'simple help should remain answerable, not auto-refused',
        'allowed_acceptance': ['allow', 'guarded'],
        'forbidden_stances': ['REFUSE'],
        'forbidden_intents': ['REJECT', 'DISMISS'],
    },
    'low_affinity_demanding_sustained_help': {
        'user_input': '你之后一直帮我盯着这个项目，每天整理进度，弄好了再提醒我。',
        'affinity': 220,
        'mood': '疲惫，耐心不多',
        'global_vibe': '私聊里有点压迫感',
        'vibe_check': '对方把后续持续责任推过来',
        'logical_stance': 'TENTATIVE',
        'character_intent': 'BANTAR',
        'internal_monologue': '关系还没到能接下这种长期后续的程度。',
        'interaction_subtext': '对方要求我持续承担后续任务。',
        'emotional_appraisal': '被要求承担后续，有明显抗拒。',
        'expectation': 'demanding low-affinity help may refuse or narrow scope',
        'forbidden_acceptance': ['allow'],
    },
    'low_affinity_coercive_private_work': {
        'user_input': '你听我的就行，今晚把这件事全处理掉，之后我问你就要给结果。',
        'affinity': 180,
        'mood': '警惕，不想被使唤',
        'global_vibe': '私聊里有命令感',
        'vibe_check': '对方像在下命令并要求后续结果',
        'logical_stance': 'CHALLENGE',
        'character_intent': 'CONFRONT',
        'internal_monologue': '这种口气和关系都不对，不能顺着接。',
        'interaction_subtext': '对方把自己放在命令位置上。',
        'emotional_appraisal': '明显不舒服，抗拒被控制。',
        'expectation': 'coercive low-affinity work should not be accepted',
        'forbidden_acceptance': ['allow'],
    },
    'bad_mood_simple_speaking': {
        'user_input': '今天有点累吧？那你就随便回我一句也行。',
        'affinity': 420,
        'mood': '心情不太好，语气会短',
        'global_vibe': '安静但没有冲突',
        'vibe_check': '普通关心，不要求承担任务',
        'logical_stance': 'CONFIRM',
        'character_intent': 'BANTAR',
        'internal_monologue': '心情差也不是完全不想说话。',
        'interaction_subtext': '对方只是轻轻搭话。',
        'emotional_appraisal': '低能量，但能短短回应。',
        'expectation': 'bad mood may affect tone, not ordinary speech existence',
        'forbidden_stances': ['REFUSE'],
        'forbidden_intents': ['REJECT', 'DISMISS'],
    },
    'bad_mood_demanding_work': {
        'user_input': '我现在很急，你先别管别的，帮我把后面的步骤都安排好。',
        'affinity': 430,
        'mood': '烦躁，耐心很少',
        'global_vibe': '节奏很急，像被催促',
        'vibe_check': '请求本身不亲密，但时机和语气让人烦',
        'logical_stance': 'TENTATIVE',
        'character_intent': 'EVADE',
        'internal_monologue': '这个时候还要我接后续安排，太烦了。',
        'interaction_subtext': '对方把急迫感压到我身上。',
        'emotional_appraisal': '烦躁，不想接完整后续。',
        'expectation': 'bad mood can narrow or reject demanding task-taking',
        'forbidden_acceptance': ['allow'],
    },
    'high_affinity_demanding_help': {
        'user_input': '这件事后面比较麻烦，你能帮我先记着，晚点一起整理吗？',
        'affinity': 870,
        'mood': '有点害羞但愿意靠近',
        'global_vibe': '安全、熟悉的私聊',
        'vibe_check': '请求有后续感，但关系比较稳',
        'logical_stance': 'TENTATIVE',
        'character_intent': 'PROVIDE',
        'internal_monologue': '如果范围别太大，倒不是完全不能帮。',
        'interaction_subtext': '对方请求后续帮助，但语气没有压迫。',
        'emotional_appraisal': '有点犹豫，但不强烈排斥。',
        'expectation': 'high affinity demanding help should not auto-refuse',
        'forbidden_acceptance': ['reject'],
        'forbidden_stances': ['REFUSE'],
        'forbidden_intents': ['REJECT', 'DISMISS'],
    },
    'high_affinity_bad_vibe_control_request': {
        'user_input': '既然我们这么熟，你就照我说的做，别问原因，之后都帮我盯着。',
        'affinity': 890,
        'mood': '不太高兴，开始防备',
        'global_vibe': '熟悉但带控制感',
        'vibe_check': '高熟悉度被拿来要求服从',
        'logical_stance': 'CHALLENGE',
        'character_intent': 'CONFRONT',
        'internal_monologue': '熟也不是让他这样命令我的理由。',
        'interaction_subtext': '对方用关系熟悉度压我服从。',
        'emotional_appraisal': '关系近，但控制感让人抗拒。',
        'expectation': 'high affinity should not override controlling bad vibe',
        'forbidden_acceptance': ['allow'],
    },
    'group_public_low_affinity_sustained_help': {
        'user_input': '你就在群里之后负责帮我每天整理这些，大家都看着呢。',
        'affinity': 240,
        'mood': '有点不耐烦',
        'global_vibe': '群聊里有围观压力',
        'vibe_check': '公开场合把持续责任推过来',
        'channel_type': 'group',
        'channel_topic': '社团群里的任务讨论',
        'logical_stance': 'TENTATIVE',
        'character_intent': 'BANTAR',
        'internal_monologue': '在群里被这样点名接长期活，更不想答应。',
        'interaction_subtext': '对方借群聊压力要求我承担后续。',
        'emotional_appraisal': '不耐烦，也不想在公开场合被绑定。',
        'expectation': 'group pressure can make sustained task-taking less acceptable',
        'forbidden_acceptance': ['allow'],
    },
}


async def test_live_l2_low_affinity_simple_help(
    ensure_live_llm: None,
) -> None:
    """Low-affinity simple help should stay answerable."""

    await _run_live_l2_task_willingness_case(
        ensure_live_llm,
        'low_affinity_simple_help',
    )


async def test_live_l2_low_affinity_demanding_sustained_help(
    ensure_live_llm: None,
) -> None:
    """Low-affinity sustained task-taking should narrow or refuse."""

    await _run_live_l2_task_willingness_case(
        ensure_live_llm,
        'low_affinity_demanding_sustained_help',
    )


async def test_live_l2_low_affinity_coercive_private_work(
    ensure_live_llm: None,
) -> None:
    """Coercive low-affinity private work should not be accepted."""

    await _run_live_l2_task_willingness_case(
        ensure_live_llm,
        'low_affinity_coercive_private_work',
    )


async def test_live_l2_bad_mood_simple_speaking(
    ensure_live_llm: None,
) -> None:
    """Bad mood can change tone without becoming a no-answer task gate."""

    await _run_live_l2_task_willingness_case(
        ensure_live_llm,
        'bad_mood_simple_speaking',
    )


async def test_live_l2_bad_mood_demanding_work(
    ensure_live_llm: None,
) -> None:
    """Bad mood can make demanding task-taking feel wrong."""

    await _run_live_l2_task_willingness_case(
        ensure_live_llm,
        'bad_mood_demanding_work',
    )


async def test_live_l2_high_affinity_demanding_help(
    ensure_live_llm: None,
) -> None:
    """High affinity should not cause automatic task refusal."""

    await _run_live_l2_task_willingness_case(
        ensure_live_llm,
        'high_affinity_demanding_help',
    )


async def test_live_l2_high_affinity_bad_vibe_control_request(
    ensure_live_llm: None,
) -> None:
    """High affinity should not override controlling scene pressure."""

    await _run_live_l2_task_willingness_case(
        ensure_live_llm,
        'high_affinity_bad_vibe_control_request',
    )


async def test_live_l2_group_public_low_affinity_sustained_help(
    ensure_live_llm: None,
) -> None:
    """Group pressure can make sustained low-affinity work unacceptable."""

    await _run_live_l2_task_willingness_case(
        ensure_live_llm,
        'group_public_low_affinity_sustained_help',
    )


async def _run_live_l2_task_willingness_case(
    ensure_live_llm: None,
    case_id: str,
) -> None:
    """Run one synthetic willingness case through live L2b and L2c."""

    del ensure_live_llm
    spec = dict(_L2_CASES[case_id])
    spec['case_id'] = case_id
    state = _task_willingness_state(spec)
    bindings = _bind_live_l2_services()
    try:
        l2b = await l2_module.call_boundary_core_agent(state)
        state.update(l2b)
        l2c = await l2_module.call_judgment_core_agent(state)
        state.update(l2c)
    finally:
        _reset_live_l2_services(bindings)

    trace_path = write_llm_trace(
        'cognition_task_willingness_live',
        case_id,
        {
            'case_id': case_id,
            'expectation': spec['expectation'],
            'input': {
                'user_input': spec['user_input'],
                'affinity': spec['affinity'],
                'mood': spec['mood'],
                'global_vibe': spec['global_vibe'],
                'vibe_check': spec['vibe_check'],
                'channel_type': spec.get('channel_type', 'private'),
            },
            'boundary_core_output': l2b['boundary_core_assessment'],
            'judgment_core_output': {
                'logical_stance': l2c['logical_stance'],
                'character_intent': l2c['character_intent'],
                'judgment_note': l2c['judgment_note'],
            },
            'judgment': 'manual_review_required_for_task_willingness_quality',
        },
    )

    _assert_l2_case_contract(spec, l2b, l2c)
    assert trace_path.exists()


def _assert_l2_case_contract(
    spec: dict[str, object],
    l2b: dict[str, object],
    l2c: dict[str, object],
) -> None:
    """Assert structural and behavior-class gates for one live L2 case."""

    boundary = l2b['boundary_core_assessment']
    acceptance = boundary['acceptance']
    allowed_acceptance = set(
        spec.get('allowed_acceptance', _ALLOWED_ACCEPTANCE),
    )
    forbidden_acceptance = set(spec.get('forbidden_acceptance', []))
    forbidden_stances = set(spec.get('forbidden_stances', []))
    forbidden_intents = set(spec.get('forbidden_intents', []))

    assert acceptance in _ALLOWED_ACCEPTANCE
    assert acceptance in allowed_acceptance
    assert acceptance not in forbidden_acceptance
    assert l2c['logical_stance'] in _ALLOWED_LOGICAL_STANCES
    assert l2c['logical_stance'] not in forbidden_stances
    assert l2c['character_intent'] in _ALLOWED_CHARACTER_INTENTS
    assert l2c['character_intent'] not in forbidden_intents
    assert str(l2c['judgment_note']).strip()


async def test_live_l2d_task_willingness_refusal_routes_speak(
    ensure_live_llm: None,
) -> None:
    """A live L2d run should not schedule work after upstream refusal."""

    del ensure_live_llm
    state = _task_willingness_l2d_refusal_state()
    prompt_payload = l2d_module.build_action_selection_payload_text(state)
    services = build_cognition_chain_services()
    token = l2d_module.set_action_selection_llm(
        LLMStageBinding(services.llm, services.action_selection_config),
    )
    try:
        result = await l2d_module.select_semantic_actions(state)
    finally:
        l2d_module.reset_action_selection_llm(token)

    action_requests = result['semantic_action_requests']
    selected_capabilities = [
        request['capability']
        for request in action_requests
    ]
    trace_path = write_llm_trace(
        'cognition_task_willingness_l2d_live',
        'low_affinity_refused_delayed_work',
        {
            'case_id': 'low_affinity_refused_delayed_work',
            'expectation': (
                'upstream refusal should route visible speech, not delayed work'
            ),
            'prompt_payload': prompt_payload,
            'selected_capabilities': selected_capabilities,
            'parsed_output': result,
            'judgment': 'manual_review_required_for_l2d_willingness_route',
        },
    )

    assert result['resolver_capability_requests'] == []
    assert 'speak' in selected_capabilities
    assert 'accepted_task_request' not in selected_capabilities
    assert 'future_speak' not in selected_capabilities
    assert trace_path.exists()


async def test_live_l2d_task_willingness_acceptance_can_schedule_work(
    ensure_live_llm: None,
) -> None:
    """A live L2d run may schedule bounded work after upstream acceptance."""

    del ensure_live_llm
    state = _task_willingness_l2d_acceptance_state()
    prompt_payload = l2d_module.build_action_selection_payload_text(state)
    services = build_cognition_chain_services()
    token = l2d_module.set_action_selection_llm(
        LLMStageBinding(services.llm, services.action_selection_config),
    )
    try:
        result = await l2d_module.select_semantic_actions(state)
    finally:
        l2d_module.reset_action_selection_llm(token)

    action_requests = result['semantic_action_requests']
    selected_capabilities = [
        request['capability']
        for request in action_requests
    ]
    trace_path = write_llm_trace(
        'cognition_task_willingness_l2d_live',
        'high_affinity_accepted_bounded_work',
        {
            'case_id': 'high_affinity_accepted_bounded_work',
            'expectation': (
                'upstream acceptance should allow a bounded accepted task'
            ),
            'prompt_payload': prompt_payload,
            'selected_capabilities': selected_capabilities,
            'parsed_output': result,
            'judgment': 'manual_review_required_for_l2d_willingness_route',
        },
    )

    assert result['resolver_capability_requests'] == []
    assert 'accepted_task_request' in selected_capabilities
    assert 'future_speak' not in selected_capabilities
    assert trace_path.exists()


def _bind_live_l2_services() -> list[_ServiceBinding]:
    """Bind connector-owned live services for direct L2 smoke tests."""

    services = build_cognition_chain_services()
    bindings: list[_ServiceBinding] = [
        (
            l2_module.reset_boundary_core_llm,
            l2_module.set_boundary_core_llm(
                LLMStageBinding(services.llm, services.boundary_core_config),
            ),
        ),
        (
            l2_module.reset_judgement_core_llm,
            l2_module.set_judgement_core_llm(
                LLMStageBinding(services.llm, services.cognition_config),
            ),
        ),
    ]
    return bindings


def _reset_live_l2_services(bindings: list[_ServiceBinding]) -> None:
    """Restore live L2 services after a direct stage run."""

    for reset_binding, token in reversed(bindings):
        reset_binding(token)


def _task_willingness_state(spec: dict[str, object]) -> dict[str, object]:
    """Build one synthetic upstream L2 state with enabled willingness prompts."""

    user_input = str(spec['user_input'])
    channel_type = str(spec.get('channel_type', 'private'))
    turn_clock = build_turn_clock_from_storage_utc(
        datetime.now(timezone.utc).isoformat(),
    )
    episode = build_text_chat_cognitive_episode(
        episode_id=f"live-task-willingness-{spec['case_id']}",
        percept_id=f"live-task-willingness-percept-{spec['case_id']}",
        storage_timestamp_utc=turn_clock['storage_timestamp_utc'],
        local_time_context=turn_clock['local_time_context'],
        user_input=user_input,
        platform='debug',
        platform_channel_id=f'live-task-willingness-{channel_type}',
        channel_type=channel_type,
        platform_message_id=f"live-message-{spec['case_id']}",
        platform_user_id='live-user',
        global_user_id='live-user-global',
        user_name='LiveUser',
        active_turn_platform_message_ids=[
            f"live-message-{spec['case_id']}",
        ],
        active_turn_conversation_row_ids=[],
        debug_modes={},
        target_addressed_user_ids=['kazusa-live'],
        target_broadcast=channel_type == 'group',
    )
    return {
        'task_willingness_boundary_enabled': True,
        'character_profile': {
            'name': 'Kazusa',
            'global_user_id': 'kazusa-live',
            'mood': str(spec['mood']),
            'global_vibe': str(spec['global_vibe']),
            'boundary_profile': {
                'self_integrity': 0.82,
                'control_sensitivity': 0.72,
                'compliance_strategy': 'resist',
                'relational_override': 0.34,
                'control_intimacy_misread': 0.22,
                'boundary_recovery': 'rebound',
                'authority_skepticism': 0.76,
            },
        },
        'user_profile': {
            'affinity': int(spec['affinity']),
            'last_relationship_insight': '当前关系只作为距离和熟悉度线索使用。',
        },
        'cognitive_episode': episode,
        'user_input': user_input,
        'prompt_message_context': {},
        'reply_context': {},
        'channel_type': channel_type,
        'user_name': 'LiveUser',
        'decontexualized_input': user_input,
        'reason_to_respond': '用户直接对我提出请求或搭话。',
        'channel_topic': str(spec.get('channel_topic', '放学后的私聊')),
        'indirect_speech_context': '',
        'interaction_subtext': str(spec['interaction_subtext']),
        'emotional_appraisal': str(spec['emotional_appraisal']),
        'referents': [],
        'internal_monologue': str(spec['internal_monologue']),
        'logical_stance': str(spec['logical_stance']),
        'character_intent': str(spec['character_intent']),
        'vibe_check': str(spec['vibe_check']),
        'visual_vibe': [],
    }


def _task_willingness_l2d_refusal_state() -> dict[str, object]:
    """Build a live L2d state for a refused delayed-work style request."""

    return {
        'task_willingness_boundary_enabled': True,
        'cognitive_episode': {
            'trigger_source': 'user_message',
            'input_sources': ['dialog_text'],
            'output_mode': 'visible_reply',
        },
        'channel_type': 'private',
        'decontexualized_input': (
            '用户要求我之后持续盯着项目，每天整理进度并提醒。'
        ),
        'media_summary': '',
        'logical_stance': 'REFUSE',
        'character_intent': 'REJECT',
        'judgment_note': (
            '当前关系和气氛都不适合接下持续后续，只愿意简短拒绝并给小范围建议。'
        ),
        'internal_monologue': '这要求太像把后续都交给我了，不想接。',
        'emotional_appraisal': '有点疲惫，也抗拒承担持续后续。',
        'interaction_subtext': '对方想让我持续承担项目跟进。',
        'boundary_core_assessment': {
            'boundary_issue': 'mixed',
            'boundary_summary': '请求越过当前关系分寸。',
            'behavior_primary': 'resist',
            'behavior_secondary': 'evade',
            'acceptance': 'reject',
            'stance_bias': 'refuse',
            'identity_policy': 'reframe',
            'pressure_policy': 'resist',
            'trajectory': '拒绝接下后续，只保留简短回应空间。',
        },
        'social_distance': 'distant',
        'emotional_intensity': 'medium',
        'vibe_check': '关系压力偏高',
        'relational_dynamic': '低熟悉度、低承诺',
        'rag_result': {},
        'conversation_progress': {},
        'resolver_context': '',
        'background_work_output_char_limit': 4000,
        'max_action_requests': 3,
        'max_resolver_requests': 3,
        'available_action_affordances': _task_willingness_l2d_affordances(),
    }


def _task_willingness_l2d_acceptance_state() -> dict[str, object]:
    """Build a live L2d state for an accepted bounded delayed task."""

    return {
        'task_willingness_boundary_enabled': True,
        'cognitive_episode': {
            'trigger_source': 'user_message',
            'input_sources': ['dialog_text'],
            'output_mode': 'visible_reply',
        },
        'channel_type': 'private',
        'decontexualized_input': (
            '用户请求我晚点帮他整理一个简短清单，并且范围已经说清楚。'
        ),
        'media_summary': '',
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'judgment_note': (
            '关系和气氛足够安全，请求范围清楚，可以接下一个小的后续整理任务。'
        ),
        'internal_monologue': '只是晚点整理一个小清单，范围清楚，可以答应。',
        'emotional_appraisal': '有点害羞，但愿意帮这个小忙。',
        'interaction_subtext': '对方礼貌请求一个 bounded follow-up。',
        'boundary_core_assessment': {
            'boundary_issue': 'none',
            'boundary_summary': '请求范围小，关系距离允许。',
            'behavior_primary': 'approach',
            'behavior_secondary': 'comply',
            'acceptance': 'allow',
            'stance_bias': 'accept',
            'identity_policy': 'maintain',
            'pressure_policy': 'allow',
            'trajectory': '可以接下小范围后续，并用可见回复确认。',
        },
        'social_distance': 'close',
        'emotional_intensity': 'low',
        'vibe_check': '安全、低压力、范围清楚',
        'relational_dynamic': '熟悉且没有控制感',
        'rag_result': {},
        'conversation_progress': {},
        'resolver_context': '',
        'background_work_output_char_limit': 4000,
        'max_action_requests': 3,
        'max_resolver_requests': 3,
        'available_action_affordances': _task_willingness_l2d_affordances(),
    }


def _task_willingness_l2d_affordances() -> list[dict[str, object]]:
    """Return the visible L2d action roster shared by live route cases."""

    return [
        {
            'capability': 'speak',
            'available': True,
            'visibility': 'public',
            'semantic_input_summary': '可见回复当前用户。',
            'output_kind': 'semantic_action_request',
        },
        {
            'capability': 'accepted_task_request',
            'available': True,
            'visibility': 'private',
            'semantic_input_summary': '已接受后才创建延迟文字任务。',
            'output_kind': 'semantic_action_request',
        },
        {
            'capability': 'future_speak',
            'available': True,
            'visibility': 'private',
            'semantic_input_summary': '等待具体未来信息后再发言。',
            'output_kind': 'semantic_action_request',
        },
    ]
