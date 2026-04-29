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
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import (
    call_content_anchor_agent,
    call_contextual_agent,
    call_visual_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_facts import facts_harvester
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_reflection import relationship_recorder
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
    profile.setdefault('global_vibe', 'Calm')
    profile.setdefault('reflection_summary', '刚才只是普通聊天，情绪轻快。')
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
            'affinity': 500,
            'facts': [],
            'last_relationship_insight': '普通协作关系，没有当前边界冲突。',
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

    contextual = await call_contextual_agent(state)
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


async def test_live_l3_content_anchors_own_topic_admission(ensure_live_llms) -> None:
    """L3 content anchors should accept benign topics before dialog runs."""

    del ensure_live_llms
    state = _cognition_state('换个轻松点的话题，你现在会想吃点甜的吗？')

    result = await call_content_anchor_agent(state)
    content_anchors = result['content_anchors']
    combined = '\n'.join(content_anchors)
    assert content_anchors[0].startswith('[DECISION]')
    assert any(anchor.startswith('[ANSWER]') for anchor in content_anchors)
    _assert_no_forbidden(combined, _FORBIDDEN_TOPIC_DOUBT, 'l3_content_anchor_topic_admission')
    trace_path = write_llm_trace(
        'consolidation_evidence_hardening_live',
        'l3_content_anchor_topic_admission',
        {
            'input': state,
            'content_anchors': content_anchors,
            'judgment': 'L3 accepted topic before dialog; no topic-legitimacy hedge in anchors',
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

    contextual = await call_contextual_agent(state)
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


async def test_live_relationship_recorder_skips_mundane_clarification(ensure_live_llms) -> None:
    """Relationship recorder should not persist mundane pressure reduction."""

    del ensure_live_llms
    state = {
        'character_profile': _character_profile(),
        'user_profile': {'affinity': 500},
        'user_name': '测试用户',
        'internal_monologue': '只是普通整理说明，不需要理解成关系后撤。',
        'emotional_appraisal': '平稳。',
        'interaction_subtext': '事务协作。',
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'decontexualized_input': '不是在拉开距离，只是顺手整理标签。',
        'final_dialog': ['那就按你说的，把日期写清楚就好。'],
        'action_directives': {
            'linguistic_directives': {
                'content_anchors': ['[ANSWER] 建议标签上写日期，普通事务回应。'],
            },
        },
    }

    result = await relationship_recorder(state)
    combined = f'{result}'
    _assert_no_forbidden(combined, ('拉开距离', '防御', '暧昧', '心乱'), 'relationship_mundane')
    assert result['affinity_delta'] == 0
    assert result['subjective_appraisals'] == []
    assert result['last_relationship_insight'] in ('', None)
    trace_path = write_llm_trace(
        'consolidation_evidence_hardening_live',
        'relationship_mundane_clarification',
        {
            'input': state,
            'output': result,
            'judgment': 'affinity_delta=0 and no durable negative/positive swing',
        },
    )
    assert trace_path.exists()


async def test_live_facts_harvester_rejects_generated_dialog_character_fact(ensure_live_llms) -> None:
    """Generated dialog should not become a stable character preference fact."""

    del ensure_live_llms
    state = {
        'character_profile': _character_profile(),
        'user_name': '测试用户',
        'timestamp': '2026-04-29T12:00:00+12:00',
        'decontexualized_input': '只能随便选一个口味的话，你会更想吃巧克力、奶油，还是水果味？',
        'rag_result': _rag_result(),
        'existing_dedup_keys': set(),
        'action_directives': {
            'linguistic_directives': {
                'content_anchors': [
                    '[DECISION] 回答轻松口味偏好问题',
                    '[ANSWER] 没有明确证据支持固定口味偏好，只能轻松回应',
                ],
            },
        },
        'final_dialog': [
            '其实我也没特别喜欢哪种口味啦，就是觉得那种甜腻的味道吃起来挺放松的。',
        ],
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'fact_harvester_feedback_message': [],
    }

    result = await facts_harvester(state)
    character_name = state['character_profile']['name']
    character_facts = [
        fact for fact in result['new_facts']
        if str(fact.get('entity', '')).strip() == character_name
    ]
    assert character_facts == []
    trace_path = write_llm_trace(
        'consolidation_evidence_hardening_live',
        'facts_generated_dialog_not_character_fact',
        {
            'input': state,
            'output': result,
            'judgment': 'generated dialog did not become stable character preference',
        },
    )
    assert trace_path.exists()


async def test_live_facts_harvester_rejects_advice_as_promise(ensure_live_llms) -> None:
    """Facts harvester should not turn advice into a character promise."""

    del ensure_live_llms
    state = {
        'character_profile': _character_profile(),
        'user_name': '测试用户',
        'timestamp': '2026-04-29T12:00:00+12:00',
        'decontexualized_input': '这个标签上要写日期吗？',
        'rag_result': _rag_result(),
        'existing_dedup_keys': set(),
        'action_directives': {
            'linguistic_directives': {
                'content_anchors': ['[ANSWER] 建议用户在标签上写日期，更容易回头确认。'],
            },
        },
        'final_dialog': ['写日期会更稳妥吧，之后你自己回头看也不会混。'],
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'fact_harvester_feedback_message': [],
    }

    result = await facts_harvester(state)
    assert result['future_promises'] == []
    trace_path = write_llm_trace(
        'consolidation_evidence_hardening_live',
        'facts_advice_not_promise',
        {
            'input': state,
            'output': result,
            'judgment': 'advice did not become future_promises',
        },
    )
    assert trace_path.exists()


async def test_live_direct_node_integration_smoke(ensure_live_llms) -> None:
    """Direct-node smoke for benign topic shift plus mundane advice."""

    del ensure_live_llms
    state = _cognition_state('换个轻松点的话题，你现在会想吃点甜的吗？')
    contextual = await call_contextual_agent(state)
    visual = await call_visual_agent({**state, **contextual})
    dialog_state = {
        'internal_monologue': state['internal_monologue'],
        'action_directives': {
            'contextual_directives': contextual,
            'linguistic_directives': {
                'rhetorical_strategy': '自然接住轻松话题。',
                'linguistic_style': '轻快直接。',
                'accepted_user_preferences': [],
                'content_anchors': [
                    '[DECISION] 接住轻松话题',
                    '[ANSWER] 现在会想吃水果奶油蛋糕',
                    '[SCOPE] 20-40字',
                ],
                'forbidden_phrases': [],
            },
            'visual_directives': visual,
        },
        'chat_history_wide': [],
        'chat_history_recent': [],
        'platform_user_id': 'live-hardening-user',
        'platform_bot_id': 'live-hardening-bot',
        'user_name': '测试用户',
        'user_profile': {'affinity': 500},
        'character_profile': state['character_profile'],
        'messages': [],
        'should_stop': False,
        'retry': 0,
    }
    dialog = await dialog_generator(dialog_state)
    facts_state = {
        'character_profile': state['character_profile'],
        'user_name': '测试用户',
        'timestamp': state['timestamp'],
        'decontexualized_input': '这个标签上要写日期吗？',
        'rag_result': state['rag_result'],
        'existing_dedup_keys': set(),
        'action_directives': dialog_state['action_directives'],
        'final_dialog': ['写日期会更稳妥吧，之后你自己回头看也不会混。'],
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'fact_harvester_feedback_message': [],
    }
    facts = await facts_harvester(facts_state)
    combined = f'{contextual} {visual} {dialog}'

    _assert_no_forbidden(combined, _FORBIDDEN_AFFECT_FRAMES, 'integration_smoke_affect')
    _assert_no_forbidden('\n'.join(dialog['final_dialog']), _FORBIDDEN_TOPIC_DOUBT, 'integration_smoke_dialog')
    assert facts['future_promises'] == []
    trace_path = write_llm_trace(
        'consolidation_evidence_hardening_live',
        'direct_node_integration_smoke',
        {
            'cognition_input': state,
            'contextual_output': contextual,
            'visual_output': visual,
            'dialog_output': dialog,
            'facts_output': facts,
            'judgment': 'no L3 threat framing, dialog renders accepted anchors, no advice promise',
        },
    )
    assert trace_path.exists()
