"""Tests for prompt-safe current-event grounding in cognition stages."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_chain_core.current_event_grounding import (
    MAX_CURRENT_MESSAGE_TEXT_CHARS,
    MAX_REPLY_EXCERPT_CHARS,
    build_current_event_grounding_for_llm,
)
from kazusa_ai_chatbot.cognition_chain_core.stages import l2 as l2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l3 as l3_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from llm_test_helpers import bind_test_llm


_ACTIVE_CHARACTER_ID = '00000000-0000-4000-8000-000000000001'
_USER_TEXT = '千纱赢啦，来，这是你的提拉米苏～'
_REPLY_EXCERPT = '咕嘎～？嘿嘿，我肯定赢了。'


class _CapturingLLM:
    """Capture stage messages while returning a fixed JSON response."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.messages: list[list[object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config,
    ) -> SimpleNamespace:
        del config
        self.messages.append(messages)
        response = SimpleNamespace(
            content=json.dumps(self._payload, ensure_ascii=False),
        )
        return response


def _prompt_message_context() -> dict:
    return {
        'body_text': _USER_TEXT,
        'raw_text': f'[CQ:at,qq=3768713357] {_USER_TEXT}',
        'mentions': [
            {
                'platform_user_id': '3768713357',
                'global_user_id': _ACTIVE_CHARACTER_ID,
                'display_name': '杏山千纱',
            },
        ],
        'addressed_to_global_user_ids': [_ACTIVE_CHARACTER_ID],
        'attachments': [{'base64_data': 'not prompt safe'}],
        'broadcast': False,
    }


def _reply_context() -> dict:
    return {
        'reply_to_message_id': '1933211842',
        'reply_to_platform_user_id': '3768713357',
        'reply_to_global_user_id': _ACTIVE_CHARACTER_ID,
        'reply_to_display_name': '杏山千纱',
        'reply_excerpt': _REPLY_EXCERPT,
    }


def _episode() -> dict:
    turn_clock = build_turn_clock('2026-06-23 16:48:13')
    episode = build_text_chat_cognitive_episode(
        episode_id='current-event-grounding-episode',
        percept_id='current-event-grounding-percept',
        storage_timestamp_utc=turn_clock['storage_timestamp_utc'],
        local_time_context=turn_clock['local_time_context'],
        user_input=_USER_TEXT,
        platform='qq',
        platform_channel_id='638473184',
        channel_type='group',
        platform_message_id='1015016025',
        platform_user_id='673225019',
        global_user_id='256e8a10-c406-47e9-ac8f-efd270d18160',
        user_name='蚝爹油',
        active_turn_platform_message_ids=['1015016025'],
        active_turn_conversation_row_ids=['conversation-row-1'],
        debug_modes={},
        target_addressed_user_ids=[_ACTIVE_CHARACTER_ID],
        target_broadcast=False,
    )
    return episode


def _character_profile() -> dict:
    return {
        'name': '杏山千纱',
        'global_user_id': _ACTIVE_CHARACTER_ID,
        'mood': 'Neutral',
        'global_vibe': 'Playful',
        'personality_brief': {
            'mbti': 'INTJ',
        },
    }


def _rag_result() -> dict:
    return {
        'answer': '',
        'memory_evidence': [],
        'conversation_evidence': [],
        'external_evidence': [],
        'recall_evidence': [],
        'user_image': {},
    }


def _l2a_state() -> dict:
    return {
        'user_input': _USER_TEXT,
        'prompt_message_context': _prompt_message_context(),
        'reply_context': _reply_context(),
        'user_name': '蚝爹油',
        'character_profile': _character_profile(),
        'local_time_context': _episode()['local_time_context'],
        'user_profile': {
            'affinity': 500,
            'last_relationship_insight': '',
        },
        'cognitive_episode': _episode(),
        'decontexualized_input': '用户表示杏山千纱赢了，并把提拉米苏递给她。',
        'rag_result': _rag_result(),
        'indirect_speech_context': '',
        'emotional_appraisal': {'valence': 'warm'},
        'interaction_subtext': {'summary': '轻松玩笑'},
    }


def _l2c1_state() -> dict:
    state = _l2a_state()
    state.update({
        'referents': [],
        'internal_monologue': '对方说杏山千纱赢了。',
        'logical_stance': 'CONFIRM',
        'character_intent': 'BANTAR',
        'boundary_core_assessment': {
            'boundary_issue': False,
            'boundary_summary': '无边界问题。',
            'behavior_primary': '轻松接梗。',
            'behavior_secondary': '保留嘴硬。',
            'acceptance': '允许轻松承接。',
            'stance_bias': 'CONFIRM',
            'identity_policy': '无身份压力。',
            'pressure_policy': '无压力。',
            'trajectory': '日常玩笑。',
        },
    })
    return state


def _l3_state() -> dict:
    return {
        'user_input': _USER_TEXT,
        'prompt_message_context': _prompt_message_context(),
        'reply_context': _reply_context(),
        'user_name': '蚝爹油',
        'decontexualized_input': '用户表示杏山千纱赢了，并把提拉米苏递给她。',
        'referents': [],
        'rag_result': _rag_result(),
        'internal_monologue': '对方说杏山千纱赢了，还递来提拉米苏。',
        'logical_stance': 'CONFIRM',
        'character_intent': 'BANTAR',
        'selected_text_surface_intent': '接住提拉米苏梗，承认当前胜负事实。',
        'memory_lifecycle_context': {'content_plan_roles': []},
        'interaction_style_context': {},
        'conversation_progress': {},
        'resolver_goal_progress': {},
        'resolver_state': {'observations': []},
        'character_profile': _character_profile(),
        'cognitive_episode': _episode(),
        'channel_type': 'group',
        'platform': 'qq',
        'platform_channel_id': '638473184',
        'global_user_id': '256e8a10-c406-47e9-ac8f-efd270d18160',
    }


def _captured_payload(fake_llm: _CapturingLLM) -> dict:
    messages = fake_llm.messages[0]
    payload = json.loads(messages[1].content)
    return payload


def test_current_event_grounding_projects_visible_names_without_ids() -> None:
    grounding = build_current_event_grounding_for_llm(
        user_input='fallback text',
        prompt_message_context=_prompt_message_context(),
        reply_context=_reply_context(),
        speaker_display_name='蚝爹油',
        active_character_display_name='杏山千纱',
        active_character_global_user_id=_ACTIVE_CHARACTER_ID,
    )

    assert grounding == {
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
    }
    serialized = json.dumps(grounding, ensure_ascii=False)
    assert 'global_user_id' not in serialized
    assert 'platform_user_id' not in serialized
    assert 'reply_to_message_id' not in serialized


def test_current_event_grounding_caps_text_and_reply_excerpt() -> None:
    prompt_message_context = {
        **_prompt_message_context(),
        'body_text': '甲' * (MAX_CURRENT_MESSAGE_TEXT_CHARS + 20),
    }
    reply_context = {
        **_reply_context(),
        'reply_excerpt': '乙' * (MAX_REPLY_EXCERPT_CHARS + 20),
    }

    grounding = build_current_event_grounding_for_llm(
        user_input='fallback text',
        prompt_message_context=prompt_message_context,
        reply_context=reply_context,
        speaker_display_name='蚝爹油',
        active_character_display_name='杏山千纱',
        active_character_global_user_id=_ACTIVE_CHARACTER_ID,
    )

    assert len(grounding['current_message_text']) == (
        MAX_CURRENT_MESSAGE_TEXT_CHARS
    )
    assert grounding['current_message_text'].endswith('...')
    assert len(grounding['reply']['reply_excerpt']) == MAX_REPLY_EXCERPT_CHARS
    assert grounding['reply']['reply_excerpt'].endswith('...')


def test_current_event_grounding_omits_raw_wire_and_storage_fields() -> None:
    prompt_message_context = {
        **_prompt_message_context(),
        'body_text': (
            f'[CQ:reply,id=1933211842][CQ:at,qq=3768713357] {_USER_TEXT} '
            '<@3768713357>'
        ),
    }
    reply_context = {
        **_reply_context(),
        'reply_excerpt': f'[CQ:reply,id=1933211842] {_REPLY_EXCERPT}',
    }

    grounding = build_current_event_grounding_for_llm(
        user_input='fallback text',
        prompt_message_context=prompt_message_context,
        reply_context=reply_context,
        speaker_display_name='蚝爹油',
        active_character_display_name='杏山千纱',
        active_character_global_user_id=_ACTIVE_CHARACTER_ID,
    )

    assert grounding['current_message_text'] == _USER_TEXT
    assert grounding['reply']['reply_excerpt'] == _REPLY_EXCERPT
    serialized = json.dumps(grounding, ensure_ascii=False)
    forbidden_fragments = [
        '[CQ:',
        '<@',
        '3768713357',
        _ACTIVE_CHARACTER_ID,
        'raw_text',
        'raw_wire_text',
        'base64_data',
        'message_id',
    ]
    for fragment in forbidden_fragments:
        assert fragment not in serialized


@pytest.mark.asyncio
async def test_l2a_payload_includes_current_event_grounding_for_text_chat() -> None:
    fake_llm = _CapturingLLM({
        'internal_monologue': '对方说杏山千纱赢了。',
        'logical_stance': 'CONFIRM',
        'character_intent': 'BANTAR',
    })
    token = l2_module.set_conscious_llm(
        bind_test_llm(fake_llm, 'conscious_llm'),
    )
    try:
        await l2_module.call_cognition_consciousness(_l2a_state())
    finally:
        l2_module.reset_conscious_llm(token)

    payload = _captured_payload(fake_llm)
    assert payload['current_event_grounding']['current_message_text'] == (
        _USER_TEXT
    )
    assert payload['current_event_grounding']['reply']['reply_excerpt'] == (
        _REPLY_EXCERPT
    )


@pytest.mark.asyncio
async def test_l2c1_payload_includes_current_event_grounding_for_text_chat() -> None:
    fake_llm = _CapturingLLM({
        'logical_stance': 'CONFIRM',
        'character_intent': 'BANTAR',
        'judgment_note': '以当前回复锚点保持主体归属。',
    })
    token = l2_module.set_judgement_core_llm(
        bind_test_llm(fake_llm, 'judgement_core_llm'),
    )
    try:
        await l2_module.call_judgment_core_agent(_l2c1_state())
    finally:
        l2_module.reset_judgement_core_llm(token)

    payload = _captured_payload(fake_llm)
    grounding = payload['current_event_grounding']
    assert grounding['speaker_display_name'] == '蚝爹油'
    assert grounding['reply']['reply_to_display_name'] == '杏山千纱'


@pytest.mark.asyncio
async def test_l3_content_plan_payload_includes_current_event_grounding_for_text_chat() -> None:
    fake_llm = _CapturingLLM({
        'content_plan': {
            'visible_goal': '接住当前轻松玩笑。',
            'semantic_content': '杏山千纱赢了，对方递来提拉米苏。',
            'voice': '轻快嘴硬。',
            'rendering': '短句。',
        },
    })
    token = l3_module.set_content_plan_agent_llm(
        bind_test_llm(fake_llm, 'content_plan_agent_llm'),
    )
    try:
        await l3_module.call_content_plan_agent(_l3_state())
    finally:
        l3_module.reset_content_plan_agent_llm(token)

    payload = _captured_payload(fake_llm)
    grounding = payload['current_event_grounding']
    assert grounding['mentions'] == ['杏山千纱']
    assert grounding['addressing']['addresses_active_character'] is True
    serialized = json.dumps(grounding, ensure_ascii=False)
    assert 'global_user_id' not in serialized
    assert 'platform_user_id' not in serialized
