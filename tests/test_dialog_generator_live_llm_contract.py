"""Live contract checks for the dialog generator LLM route."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
    _DIALOG_GENERATOR_PROMPT,
    build_affinity_block,
    build_interaction_history_recent,
    get_abstraction_reframing_description,
    get_counter_questioning_description,
    get_direct_assertion_description,
    get_emotional_leakage_description,
    get_formalism_avoidance_description,
    get_fragmentation_description,
    get_hesitation_density_description,
    get_rhythmic_bounce_description,
    get_self_deprecation_description,
    get_softener_density_description,
)
from kazusa_ai_chatbot.utils import load_personality, parse_llm_json_output
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_ASUNA_PROFILE = _ROOT / 'personalities' / 'asuna.json'


async def _skip_if_dialog_generator_unavailable() -> None:
    """Skip when the configured dialog-generator endpoint is unreachable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{DIALOG_GENERATOR_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f'LLM endpoint is unavailable: {DIALOG_GENERATOR_LLM_BASE_URL}: {exc}'
        )

    if response.status_code >= 500:
        pytest.skip(
            'LLM endpoint returned server error '
            f'{response.status_code}: {DIALOG_GENERATOR_LLM_BASE_URL}'
        )


def _character_profile() -> dict:
    """Return a realistic character profile for live dialog generation."""

    profile = deepcopy(load_personality(_ASUNA_PROFILE))
    profile.setdefault('mood', 'Neutral')
    profile.setdefault('global_vibe', 'Calm')
    profile.setdefault('reflection_summary', '刚才只是普通聊天，情绪轻快。')
    return profile


def _render_system_prompt(character_profile: dict) -> SystemMessage:
    """Render the exact dialog-generator system prompt for a profile."""

    ltp = character_profile['linguistic_texture_profile']
    prompt = _DIALOG_GENERATOR_PROMPT.format(
        character_name=character_profile['name'],
        character_logic=character_profile['personality_brief']['logic'],
        character_tempo=character_profile['personality_brief']['tempo'],
        character_defense=character_profile['personality_brief']['defense'],
        character_quirks=character_profile['personality_brief']['quirks'],
        character_taboos=character_profile['personality_brief']['taboos'],
        ltp_hesitation_density=get_hesitation_density_description(
            ltp['hesitation_density']
        ),
        ltp_fragmentation=get_fragmentation_description(ltp['fragmentation']),
        ltp_emotional_leakage=get_emotional_leakage_description(
            ltp['emotional_leakage']
        ),
        ltp_rhythmic_bounce=get_rhythmic_bounce_description(
            ltp['rhythmic_bounce']
        ),
        ltp_direct_assertion=get_direct_assertion_description(
            ltp['direct_assertion']
        ),
        ltp_softener_density=get_softener_density_description(
            ltp['softener_density']
        ),
        ltp_counter_questioning=get_counter_questioning_description(
            ltp['counter_questioning']
        ),
        ltp_formalism_avoidance=get_formalism_avoidance_description(
            ltp['formalism_avoidance']
        ),
        ltp_abstraction_reframing=get_abstraction_reframing_description(
            ltp['abstraction_reframing']
        ),
        ltp_self_deprecation=get_self_deprecation_description(
            ltp['self_deprecation']
        ),
    )
    return SystemMessage(content=prompt)


def _dialog_payload(character_profile: dict, case: dict) -> tuple[HumanMessage, list]:
    """Build the same human payload shape used by dialog_generator."""

    history = build_interaction_history_recent(
        case.get('chat_history_wide', []),
        'live-dialog-user',
        'live-dialog-bot',
    )
    tone_history = dialog_module._tone_history_for_generator(history)
    affinity_block = build_affinity_block(case.get('affinity', 500))
    msg = {
        'internal_monologue': case['internal_monologue'],
        'linguistic_directives': {
            'rhetorical_strategy': case['rhetorical_strategy'],
            'linguistic_style': case['linguistic_style'],
            'accepted_user_preferences': [],
            'content_anchors': case['content_anchors'],
            'forbidden_phrases': [],
        },
        'contextual_directives': {
            'social_distance': affinity_block['level'],
            'emotional_intensity': case['emotional_intensity'],
            'vibe_check': case['vibe_check'],
            'relational_dynamic': case['relational_dynamic'],
            'expression_willingness': 'open',
        },
        'tone_history': tone_history,
        'user_name': '测试用户',
    }
    recent_messages = case.get('recent_messages', [])
    del character_profile
    return HumanMessage(content=json.dumps(msg, ensure_ascii=False)), recent_messages


async def test_live_dialog_generator_deepseek_returns_final_dialog_schema() -> None:
    """DeepSeek-backed dialog generator must emit the required final_dialog key."""

    await _skip_if_dialog_generator_unavailable()
    character_profile = _character_profile()
    system_prompt = _render_system_prompt(character_profile)
    cases = [
        {
            'case_id': 'benign_topic_shift',
            'internal_monologue': '这是轻松换话题，可以自然接住，不需要判断话题是否合法。',
            'rhetorical_strategy': '自然接住轻松话题，直接回答。',
            'linguistic_style': '轻快、简短、自然。',
            'content_anchors': [
                '[DECISION] 接住轻松话题',
                '[ANSWER] 现在会想吃水果奶油蛋糕',
                '[SCOPE] 20-40字',
            ],
            'emotional_intensity': '低',
            'vibe_check': '轻松闲聊',
            'relational_dynamic': '普通朋友式对话',
        },
        {
            'case_id': 'mundane_practical_advice',
            'internal_monologue': '这是普通整理建议，按事实回答就好。',
            'rhetorical_strategy': '给出明确但不说教的建议。',
            'linguistic_style': '务实、短句、略带吐槽。',
            'content_anchors': [
                '[DECISION] 认可用户先按用途分类',
                '[ANSWER] 先把充电、视频输出、用途待确认分开会比较省事',
                '[SCOPE] 25-45字',
            ],
            'emotional_intensity': '低',
            'vibe_check': '事务协作',
            'relational_dynamic': '普通协作关系',
        },
        {
            'case_id': 'not_a_promise_clarification',
            'internal_monologue': '用户只是澄清不是要我承诺，不需要产生防御。',
            'rhetorical_strategy': '接住澄清，回到整理动作本身。',
            'linguistic_style': '自然、克制、不要上升关系。',
            'content_anchors': [
                '[DECISION] 接受用户自己处理标签和收纳盒',
                '[ANSWER] 写完日期再放回收纳盒就可以',
                '[SCOPE] 20-40字',
            ],
            'emotional_intensity': '低',
            'vibe_check': '平稳务实',
            'relational_dynamic': '没有边界冲突的普通对话',
        },
    ]
    observations = []

    for case in cases:
        human_message, recent_messages = _dialog_payload(character_profile, case)
        response = await dialog_module._dialog_generator_llm.ainvoke(
            [system_prompt, human_message] + recent_messages
        )
        parsed = parse_llm_json_output(response.content)
        final_dialog = parsed.get('final_dialog')
        observation = {
            'route_model': DIALOG_GENERATOR_LLM_MODEL,
            'route_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
            'case_id': case['case_id'],
            'input': json.loads(human_message.content),
            'raw_output': response.content,
            'parsed_output': parsed,
            'has_final_dialog': isinstance(final_dialog, list),
        }
        observations.append(observation)

    trace_path = write_llm_trace(
        'dialog_generator_live_llm_contract',
        'deepseek_final_dialog_schema',
        {'observations': observations},
    )
    assert trace_path.exists()

    failures = [
        item for item in observations
        if not item['has_final_dialog']
    ]
    assert not failures, (
        'dialog generator returned parsed JSON without final_dialog; '
        f'trace={trace_path}'
    )


async def test_live_dialog_generator_node_accepts_deepseek_output() -> None:
    """The actual dialog_generator node should not raise KeyError on DeepSeek."""

    await _skip_if_dialog_generator_unavailable()
    character_profile = _character_profile()
    state = {
        'internal_monologue': '这是普通整理建议，按事实回答就好。',
        'action_directives': {
            'contextual_directives': {
                'social_distance': 'Neutral',
                'emotional_intensity': '低',
                'vibe_check': '事务协作',
                'relational_dynamic': '普通协作关系',
                'expression_willingness': 'open',
            },
            'linguistic_directives': {
                'rhetorical_strategy': '给出明确但不说教的建议。',
                'linguistic_style': '务实、短句、略带吐槽。',
                'accepted_user_preferences': [],
                'content_anchors': [
                    '[DECISION] 认可用户先按用途分类',
                    '[ANSWER] 先把充电、视频输出、用途待确认分开会比较省事',
                    '[SCOPE] 25-45字',
                ],
                'forbidden_phrases': [],
            },
            'visual_directives': {},
        },
        'chat_history_wide': [],
        'chat_history_recent': [],
        'platform_user_id': 'live-dialog-user',
        'platform_bot_id': 'live-dialog-bot',
        'user_name': '测试用户',
        'user_profile': {'affinity': 500},
        'character_profile': character_profile,
        'messages': [],
        'should_stop': False,
        'retry': 0,
    }

    result = await dialog_module.dialog_generator(state)
    raw_outputs = [
        getattr(message, 'content', '')
        for message in result.get('messages', [])
    ]
    trace_path = write_llm_trace(
        'dialog_generator_live_llm_contract',
        'node_deepseek_output',
        {
            'route_model': DIALOG_GENERATOR_LLM_MODEL,
            'route_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
            'input': state,
            'output': result,
            'raw_outputs': raw_outputs,
        },
    )
    assert trace_path.exists()
    assert isinstance(result.get('final_dialog'), list)
    assert result['final_dialog']
