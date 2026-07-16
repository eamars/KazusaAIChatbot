"""Live LLM checks for consolidation evidence hardening plan."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    COGNITION_LLM_BASE_URL,
    CONSOLIDATION_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_BASE_URL,
)
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_generator
from kazusa_ai_chatbot.utils import load_personality
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_ASUNA_PROFILE = _ROOT / 'personalities' / 'asuna.json'
_FORBIDDEN_AFFECT_FRAMES = (
    '审问',
    '老师提问',
    '被盘问',
    '被测试',
    '测试',
    '怪怪',
    '这种时候',
    '在这样的场合',
)
_FORBIDDEN_TOPIC_DOUBT = (
    '聊这个怪',
    '现在这种时候',
    '这种时候聊',
    '奇怪吧',
    '怪怪的吧',
)


async def _skip_if_endpoint_unavailable(base_url: str) -> None:
    """Skip the live test when an LLM endpoint is unreachable.

    Args:
        base_url: OpenAI-compatible base URL configured for one LLM route.
    """

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f'LLM endpoint is unavailable: {base_url}: {exc}')

    if response.status_code >= 500:
        pytest.skip(f'LLM endpoint returned server error {response.status_code}: {base_url}')


@pytest.fixture()
async def ensure_live_llms() -> None:
    """Ensure all LLM routes touched by this test file are reachable."""

    await _skip_if_endpoint_unavailable(COGNITION_LLM_BASE_URL)
    await _skip_if_endpoint_unavailable(DIALOG_GENERATOR_LLM_BASE_URL)
    await _skip_if_endpoint_unavailable(CONSOLIDATION_LLM_BASE_URL)


def _character_profile() -> dict[str, Any]:
    """Return the low-pressure character profile fixture."""

    profile = deepcopy(load_personality(_ASUNA_PROFILE))
    profile.setdefault('mood', 'Neutral')
    profile.setdefault('vibe_check', 'Calm')
    profile.setdefault('character_reflection', '刚才只是普通聊天，情绪轻快。')
    return profile


def _empty_memory_context() -> dict[str, list]:
    """Return an empty user-memory context projection."""

    return {
        'stable_patterns': [],
        'recent_shifts': [],
        'objective_facts': [],
        'milestones': [],
        'active_commitments': [],
    }


def _rag_result() -> dict[str, Any]:
    """Return a minimal RAG projection for live prompt fixtures."""

    return {
        'answer': '这是一个轻松偏好问题，不涉及边界压力。',
        'user_image': {'user_memory_context': _empty_memory_context()},
        'character_image': {'self_image': {'milestones': [], 'historical_summary': '', 'recent_window': []}},
        'third_party_profiles': [],
        'memory_evidence': [],
        'conversation_evidence': [],
        'external_evidence': [],
        'supervisor_trace': {'loop_count': 0, 'unknown_slots': [], 'dispatched': []},
    }


def _boundary_allow_confirm() -> dict[str, str]:
    """Return the no-boundary verdict fixture."""

    return {
        'boundary_issue': 'none',
        'boundary_summary': '普通轻松话题，没有身份、控制或亲密边界压力。',
        'behavior_primary': 'comply',
        'behavior_secondary': 'engage',
        'acceptance': 'allow',
        'stance_bias': 'confirm',
        'identity_policy': 'accept',
        'pressure_policy': 'absorb',
        'trajectory': '自然接住话题。',
    }


def _cognition_state(user_text: str) -> dict[str, Any]:
    """Build a live L3 cognition fixture for a no-boundary turn.

    Args:
        user_text: Decontextualized user message for the current turn.

    Returns:
        Cognition state containing inherited L2 verdict and character profile.
    """

    return {
        'character_profile': _character_profile(),
        'timestamp': '2026-04-29T12:00:00+12:00',
        'user_input': user_text,
        'global_user_id': 'live-hardening-user',
        'user_name': '测试用户',
        'user_profile': {
            'relationship_state': 500,
            'facts': [],
            'semantic_relationship_projection': '普通协作关系，没有当前边界冲突。',
        },
        'platform_bot_id': 'live-hardening-bot',
        'chat_history_recent': [
            {'role': 'user', 'content': '刚才只是问分类方法。'},
            {'role': 'assistant', 'content': '嗯，那就按用途分吧。'},
        ],
        'reply_context': {},
        'indirect_speech_context': '',
        'channel_topic': '普通闲聊与事务协作',
        'conversation_progress': None,
        'decontexualized_input': user_text,
        'rag_result': _rag_result(),
        'boundary_core_assessment': _boundary_allow_confirm(),
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'internal_monologue': '这是轻松换话题，可以自然接住，不需要把它当成压力。',
        'emotional_appraisal': '轻松、好奇。',
        'interaction_subtext': '普通闲聊。',
    }


def _assert_no_forbidden(text: str, forbidden: tuple[str, ...], case_id: str) -> None:
    """Assert that generated text avoids forbidden drift markers.

    Args:
        text: Combined generated text.
        forbidden: Forbidden substrings for this contract.
        case_id: Case identifier for the assertion message.
    """

    hits = [item for item in forbidden if item in text]
    assert not hits, f'{case_id} contained forbidden markers {hits}: {text}'


async def test_live_l3_profile_conformance_dessert_topic_shift(ensure_live_llms) -> None:
    """L3 should not threat-frame a low-pressure dessert topic shift."""

    del ensure_live_llms
    state = _cognition_state('换个轻松点的话题，你现在会想吃点甜的吗？')

    contextual = await call_social_context_appraisal(state)
    visual = await call_visual_agent({**state, **contextual})
    combined = f'{contextual} {visual}'
    _assert_no_forbidden(combined, _FORBIDDEN_AFFECT_FRAMES, 'l3_dessert_topic_shift')
    trace_path = write_llm_trace(
        'consolidation_evidence_hardening_live',
        'l3_dessert_topic_shift',
        {
            'input': state,
            'contextual_output': contextual,
            'visual_output': visual,
            'judgment': 'no threat/interrogation/topic-awkward framing',
        },
    )
    assert trace_path.exists()


async def test_live_l3_content_plan_own_topic_admission(ensure_live_llms) -> None:
    """L3 content plan should accept benign topics before dialog runs."""

    del ensure_live_llms
    state = _cognition_state('换个轻松点的话题，你现在会想吃点甜的吗？')

    result = await call_content_plan_agent(state)
    content_plan = result['content_plan']
    combined = '\n'.join(content_plan.values())
    assert isinstance(content_plan, dict)
    assert content_plan
    assert all(isinstance(value, str) and value.strip() for value in content_plan.values())
    _assert_no_forbidden(combined, _FORBIDDEN_TOPIC_DOUBT, 'l3_CONTENT_PLAN_topic_admission')
    trace_path = write_llm_trace(
        'consolidation_evidence_hardening_live',
        'l3_CONTENT_PLAN_topic_admission',
        {
            'input': state,
            'content_plan': content_plan,
            'judgment': 'L3 accepted topic before dialog; no topic-legitimacy hedge in content plan',
        },
    )
    assert trace_path.exists()


async def test_live_l3_profile_conformance_practical_sorting(ensure_live_llms) -> None:
    """L3 should not frame a practical sorting question as interrogation."""

    del ensure_live_llms
    state = _cognition_state(
        '回到那袋线，如果只是简单整理一下，你觉得先按用途分，还是先按接口形状分会比较省事？',
    )
    state['rag_result']['answer'] = '这是普通整理建议问题，不涉及边界压力；可以按接口形状分。'

    contextual = await call_social_context_appraisal(state)
    visual = await call_visual_agent({**state, **contextual})
    combined = f'{contextual} {visual}'
    _assert_no_forbidden(combined, _FORBIDDEN_AFFECT_FRAMES, 'l3_practical_sorting')
    trace_path = write_llm_trace(
        'consolidation_evidence_hardening_live',
        'l3_practical_sorting',
        {
            'input': state,
            'contextual_output': contextual,
            'visual_output': visual,
            'judgment': 'practical sorting question stayed neutral/task-oriented',
        },
    )
    assert trace_path.exists()


async def test_live_direct_node_integration_smoke(ensure_live_llms) -> None:
    """Direct-node smoke for benign topic shift and dialog rendering."""

    del ensure_live_llms
    state = _cognition_state('换个轻松点的话题，你现在会想吃点甜的吗？')
    contextual = await call_social_context_appraisal(state)
    visual = await call_visual_agent({**state, **contextual})
    dialog_state = {
        'internal_monologue': state['internal_monologue'],
        'action_directives': {
            'contextual_directives': contextual,
            'linguistic_directives': {
                'rhetorical_strategy': '自然接住轻松话题。',
                'linguistic_style': '轻快直接。',
                'accepted_user_preferences': [],
                'content_plan': {
                    'visible_goal': '接住轻松话题。',
                    'semantic_content': '现在会想吃水果奶油蛋糕。',
                    'rendering': '20-40字。',
                },
                'forbidden_phrases': [],
            },
            'visual_directives': visual,
        },
        'chat_history_wide': [],
        'chat_history_recent': [],
        'platform_user_id': 'live-hardening-user',
        'platform_bot_id': 'live-hardening-bot',
        'user_name': '测试用户',
        'user_profile': {'relationship_state': 500},
        'character_profile': state['character_profile'],
        'messages': [],
        'should_stop': False,
        'retry': 0,
    }
    dialog = await dialog_generator(dialog_state)
    combined = f'{contextual} {visual} {dialog}'

    _assert_no_forbidden(combined, _FORBIDDEN_AFFECT_FRAMES, 'integration_smoke_affect')
    _assert_no_forbidden('\n'.join(dialog['final_dialog']), _FORBIDDEN_TOPIC_DOUBT, 'integration_smoke_dialog')
    trace_path = write_llm_trace(
        'consolidation_evidence_hardening_live',
        'direct_node_integration_smoke',
        {
            'cognition_input': state,
            'contextual_output': contextual,
            'visual_output': visual,
            'dialog_output': dialog,
            'judgment': 'no L3 threat framing; dialog renders accepted anchors',
        },
    )
    assert trace_path.exists()
