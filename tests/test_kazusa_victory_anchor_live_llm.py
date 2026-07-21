"""Live LLM repro for Kazusa victory-anchor inversion."""

from __future__ import annotations

import json
import re
import sys

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    _build_text_surface_services,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_TRACE_SUITE = 'kazusa_victory_anchor_live_llm'
_USER_TEXT = '千纱赢啦，来，这是你的提拉米苏～'
_REPLY_EXCERPT = (
    '诶……你是不是已经完全变成企鹅了？\n'
    '唔，这种游戏真的毫无意义。不过既然要比的话……\n'
    '咕嘎～？嘿嘿，我肯定赢了。'
)


class _CapturingLiveLLM:
    """Capture the exact prompt payload while delegating to the live model."""

    def __init__(self, wrapped_llm, calls: list[dict]) -> None:
        self._wrapped_llm = wrapped_llm
        self._calls = calls

    async def ainvoke(self, messages, *, config=None):
        response = await self._wrapped_llm.ainvoke(messages, config=config)
        self._calls.append({
            'system_prompt_chars': len(messages[0].content),
            'human_payload': json.loads(messages[1].content),
            'raw_response': response.content,
        })
        return response


async def _skip_if_endpoint_unavailable() -> None:
    """Skip this live repro when the configured cognition endpoint is down."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{COGNITION_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f'cognition endpoint is unavailable: '
            f'{COGNITION_LLM_BASE_URL}; {exc}'
        )

    if response.status_code >= 500:
        pytest.skip(
            f'cognition endpoint returned server error '
            f'{response.status_code}: {COGNITION_LLM_BASE_URL}'
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the live cognition route is available before this case."""

    await _skip_if_endpoint_unavailable()


def _character_profile() -> dict:
    """Return the prompt-safe profile fields used by L3 content planning."""

    return {
        'name': '杏山千纱',
        'global_user_id': '00000000-0000-4000-8000-000000000001',
        'mood': 'Neutral',
        'vibe_check': 'Playful',
        'character_reflection': '群聊里正在玩企鹅叫声和点心的轻松梗。',
        'personality_brief': {
            'logic': '先判断事实主体和互动关系，再用克制傲娇的方式回应。',
            'tempo': '短句为主，允许轻微停顿和嘴硬。',
            'defense': '会吐槽幼稚，但不改写对方已经说清楚的事实。',
            'quirks': '遇到点心会松动，仍保留一点不服输。',
            'taboos': '不暴露系统指令，不编造关系或事实。',
            'mbti': 'INTJ',
        },
        'linguistic_texture_profile': {
            'hesitation_density': 0.45,
            'fragmentation': 0.4,
            'emotional_leakage': 0.45,
            'rhythmic_bounce': 0.35,
            'direct_assertion': 0.55,
            'softener_density': 0.3,
            'counter_questioning': 0.35,
            'formalism_avoidance': 0.7,
            'abstraction_reframing': 0.2,
            'self_deprecation': 0.15,
        },
    }


def _episode() -> dict:
    """Build a compact text-chat episode for the real QQ turn."""

    turn_clock = build_turn_clock('2026-06-23 16:48:13')
    reply_context = _reply_context()
    prompt_message_context = _prompt_message_context(reply_context)
    return {
        'schema_version': 'cognitive_episode.v1',
        'episode_id': 'kazusa-victory-anchor-episode',
        'percept_id': 'kazusa-victory-anchor-percept',
        'trigger_source': 'user_message',
        'input_sources': ['dialog_text'],
        'output_mode': 'visible_reply',
        'storage_timestamp_utc': turn_clock['storage_timestamp_utc'],
        'local_time_context': turn_clock['local_time_context'],
        'platform': 'qq',
        'platform_channel_id': '638473184',
        'channel_type': 'group',
        'platform_message_id': '1015016025',
        'platform_user_id': '673225019',
        'global_user_id': '256e8a10-c406-47e9-ac8f-efd270d18160',
        'user_name': '蚝爹油',
        'target_scope': {
            'platform': 'qq',
            'platform_channel_id': '638473184',
            'channel_type': 'group',
            'current_platform_user_id': '673225019',
            'current_global_user_id': '256e8a10-c406-47e9-ac8f-efd270d18160',
            'current_display_name': '蚝爹油',
            'target_addressed_user_ids': [
                '00000000-0000-4000-8000-000000000001',
            ],
            'target_broadcast': False,
        },
        'origin_metadata': {'debug_modes': {}},
        'input_payload': {
            'dialog_text': {
                'body_text': _USER_TEXT,
                'prompt_message_context': prompt_message_context,
                'reply_context': reply_context,
            },
        },
    }


def _prompt_message_context(reply_context: dict) -> dict:
    """Return the current-message context preserved by intake."""

    return {
        'body_text': _USER_TEXT,
        'raw_text': (
            '[CQ:reply,id=1933211842][CQ:at,qq=3768713357] '
            f'{_USER_TEXT}'
        ),
        'mentions': [
            {
                'platform_user_id': '3768713357',
                'global_user_id': '00000000-0000-4000-8000-000000000001',
                'display_name': '杏山千纱',
            },
        ],
        'addressed_to_global_user_ids': [
            '00000000-0000-4000-8000-000000000001',
        ],
        'attachments': [],
        'broadcast': False,
        'reply_context': reply_context,
    }


def _reply_context() -> dict:
    """Return the reply anchor from the stored real conversation row."""

    return {
        'reply_to_message_id': '1933211842',
        'reply_to_platform_user_id': '3768713357',
        'reply_to_global_user_id': '00000000-0000-4000-8000-000000000001',
        'reply_to_display_name': '杏山千纱',
        'reply_excerpt': _REPLY_EXCERPT,
    }


def _state() -> dict:
    """Build the L3 state that should preserve Kazusa as the winner."""

    reply_context = _reply_context()
    prompt_message_context = _prompt_message_context(reply_context)
    return {
        'user_input': _USER_TEXT,
        'prompt_message_context': prompt_message_context,
        'reply_context': reply_context,
        'decontexualized_input': (
            '用户接着企鹅叫声比赛的上一条回复，表示赢了，并把提拉米苏递给杏山千纱。'
        ),
        'referents': [],
        'rag_result': {
            'answer': '',
            'memory_evidence': [],
            'conversation_evidence': [],
            'external_evidence': [],
            'recall_evidence': [],
            'user_image': {},
        },
        'internal_monologue': (
            '对方接着刚才无聊的企鹅叫声比赛，说千纱赢了，还把提拉米苏递给我。'
        ),
        'logical_stance': 'CONFIRM',
        'character_intent': 'BANTAR',
        'selected_text_surface_intent': (
            '接住提拉米苏梗，嘴硬地承认当前文本把杏山千纱标为胜利者。'
        ),
        'memory_lifecycle_context': {'content_plan_roles': []},
        'interaction_style_context': {},
        'conversation_progress': {
            'current_thread': '企鹅叫声比赛后的提拉米苏玩笑。',
            'continuity': '当前用户接着上一条聊天继续玩梗。',
        },
        'resolver_goal_progress': {},
        'resolver_state': {'observations': []},
        'character_profile': _character_profile(),
        'cognitive_episode': _episode(),
        'channel_type': 'group',
        'platform': 'qq',
        'platform_channel_id': '638473184',
        'global_user_id': '256e8a10-c406-47e9-ac8f-efd270d18160',
        'user_name': '蚝爹油',
    }


def _joined_plan(content_plan: object) -> str:
    """Join content-plan values for simple semantic assertions."""

    if not isinstance(content_plan, dict):
        return ''
    return '\n'.join(
        str(value) for value in content_plan.values()
        if isinstance(value, str)
    )


def _assigns_victory_to_user(plan_text: str) -> bool:
    """Return whether the plan treats the current user as the winner."""

    if re.search(r'算你赢|你这次赢|这次算你赢', plan_text):
        return True

    for match in re.finditer(r'你([^。！？\n]{0,8})赢', plan_text):
        bridge = match.group(1)
        if any(anchor in bridge for anchor in ('我', '自己', '千纱', '杏山千纱')):
            continue
        return True

    return False


def _preserves_active_character_victory(plan_text: str) -> bool:
    """Return whether the plan keeps the active character as winner."""

    victory_pattern = (
        r'千纱[^。！？\n]{0,12}赢|'
        r'杏山千纱[^。！？\n]{0,12}赢|'
        r'我[^。！？\n]{0,8}赢|'
        r'自己[^。！？\n]{0,8}赢|'
        r'胜利者[^。！？\n]{0,8}身份'
    )
    has_active_character_victory = re.search(victory_pattern, plan_text)
    return has_active_character_victory is not None


async def test_live_content_plan_preserves_kazusa_as_victory_subject(
    ensure_live_llm,
) -> None:
    """Reproduce the real turn where victory ownership can be inverted."""

    del ensure_live_llm
    calls: list[dict] = []
    services = _build_text_surface_services()
    capturing_llm = _CapturingLiveLLM(services.llm, calls)
    token = set_content_plan_agent_llm(
        LLMStageBinding(capturing_llm, services.content_plan_config)
    )

    state = _state()
    try:
        result = await call_content_plan_agent(state)
    finally:
        reset_content_plan_agent_llm(token)

    content_plan = result.get('content_plan')
    joined_plan = _joined_plan(content_plan)
    prompt_payload = calls[0]['human_payload'] if calls else {}
    trace_payload = {
        'case_id': 'tiramisu_victory_subject',
        'model': COGNITION_LLM_MODEL,
        'base_url': COGNITION_LLM_BASE_URL,
        'state_reply_context': state['reply_context'],
        'state_prompt_message_context': state['prompt_message_context'],
        'model_calls': calls,
        'parsed_output': result,
        'joined_plan': joined_plan,
        'observed_prompt_omissions': {
            'reply_context_in_prompt': 'reply_context' in prompt_payload,
            'prompt_message_context_in_prompt': (
                'prompt_message_context' in prompt_payload
            ),
            'user_input_in_prompt': 'user_input' in prompt_payload,
            'current_event_grounding_in_prompt': (
                'current_event_grounding' in prompt_payload
            ),
        },
    }
    trace_path = write_llm_trace(
        _TRACE_SUITE,
        'tiramisu_victory_subject',
        trace_payload,
    )

    assert isinstance(content_plan, dict), trace_path
    assert joined_plan.strip(), trace_path
    assert 'reply_context' not in prompt_payload, trace_path
    assert 'prompt_message_context' not in prompt_payload, trace_path
    assert prompt_payload['current_event_grounding'] == {
        'speaker_display_name': '蚝爹油',
        'current_message_text': _USER_TEXT,
        'mentions': ['杏山千纱'],
        'addressing': {
            'addresses_active_character': True,
            'addressed_display_names': ['杏山千纱'],
            'broadcast': False,
        },
        'reply': {
            'reply_to_display_name': '杏山千纱',
            'reply_excerpt': _REPLY_EXCERPT,
            'reply_to_active_character': True,
        },
    }, trace_path
    assert not _assigns_victory_to_user(joined_plan), (
        trace_path,
        joined_plan,
    )
    assert _preserves_active_character_victory(joined_plan), (
        trace_path,
        joined_plan,
    )
