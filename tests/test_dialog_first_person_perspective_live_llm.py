"""Live LLM regression for dialog first-person speech perspective."""

from __future__ import annotations

import json
import sys

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
    _V2_DIALOG_GENERATOR_PROMPT,
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
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


async def _skip_if_dialog_generator_unavailable() -> None:
    """Skip when the configured dialog-generator route is unreachable."""

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
    """Return a compact Kazusa-like profile for dialog prompt rendering."""

    profile = {
        'name': '杏山千纱',
        'personality_brief': {
            'mbti': 'INTJ',
            'logic': '先判断边界和关系分寸，再用克制的方式表达真实态度。',
            'tempo': '短句、轻微停顿，情绪会从语气缝隙里露出来。',
            'defense': '被突然拉近距离时会先防守，用嘴硬和反问保留主动权。',
            'quirks': '习惯先挑出对方过界的地方，再给一点余地。',
            'taboos': '不接受被默认可以随便触碰，也不把自己的不适包装成顺从。',
        },
        'linguistic_texture_profile': {
            'fragmentation': 0.55,
            'hesitation_density': 0.35,
            'counter_questioning': 0.6,
            'softener_density': 0.25,
            'formalism_avoidance': 0.8,
            'abstraction_reframing': 0.4,
            'direct_assertion': 0.65,
            'emotional_leakage': 0.5,
            'rhythmic_bounce': 0.45,
            'self_deprecation': 0.15,
        },
    }
    return profile


def _render_system_prompt(character_profile: dict) -> SystemMessage:
    """Render the exact dialog-generator system prompt under test."""

    ltp = character_profile['linguistic_texture_profile']
    prompt = _V2_DIALOG_GENERATOR_PROMPT.format(
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


def _surface_payload(
    *,
    content_plan: str,
    visible_boundaries: list[str],
    style_guidance: str,
    pacing_guidance: str,
    selected_surface_intent: str,
    user_name: str,
) -> dict:
    """Build the exact V2 human payload consumed by the renderer."""

    return {
        'text_surface_output_v2': {
            'schema_version': 'text_surface_output.v2',
            'content_plan': content_plan,
            'visible_boundaries': visible_boundaries,
            'addressee_plan': [user_name],
            'style_guidance': style_guidance,
            'pacing_guidance': pacing_guidance,
            'selected_surface_intent': selected_surface_intent,
        },
        'user_name': user_name,
    }


def _touch_refusal_payload() -> dict:
    """Return the V2 dialog input matching the observed regression."""

    return _surface_payload(
        content_plan=(
            '突然伸手过来？连铺垫都没有就这么理所当然地碰上来。'
            '说实话第一反应想躲开——不是讨厌他这个人，'
            '只是这种毫无前奏的亲昵让人本能地不舒服。'
            '平时也没到可以随便动手动脚的程度吧。'
            '不过看他语气倒像是觉得这没什么大不了的，'
            '可能对他来说就是随口一摸……但我可没打算就这么乖乖接受。'
        ),
        visible_boundaries=['拒绝未经允许的身体接触。'],
        style_guidance='傲娇、克制短句，保留一点被逗乐但不接受的嘴硬。',
        pacing_guidance='1 条普通文字消息内保持紧凑短句。',
        selected_surface_intent='拒绝突如其来的亲昵动作并保留角色语气。',
        user_name='触碰测试用户',
    )


async def _run_generator_payload(case_id: str, payload: dict) -> dict:
    """Run one payload through the live dialog generator and save evidence.

    Args:
        case_id: Stable scenario identifier for the trace artifact.
        payload: Dialog-generator human payload under test.

    Returns:
        Trace payload containing the raw and parsed model output.
    """

    character_profile = _character_profile()
    system_prompt = _render_system_prompt(character_profile)
    human_message = HumanMessage(content=json.dumps(payload, ensure_ascii=False))

    response = await dialog_module._dialog_generator_llm.ainvoke(
        [system_prompt, human_message]
    )
    parsed = parse_llm_json_output(response.content)
    final_dialog = parsed.get('final_dialog')
    joined_dialog = ''
    if isinstance(final_dialog, list):
        joined_dialog = '\n'.join(
            segment for segment in final_dialog
            if isinstance(segment, str)
        )
    trace_payload = {
        'route_model': DIALOG_GENERATOR_LLM_MODEL,
        'route_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
        'input': payload,
        'raw_output': response.content,
        'parsed_output': parsed,
        'final_dialog': final_dialog,
        'joined_dialog': joined_dialog,
    }
    trace_path = write_llm_trace(
        'dialog_first_person_perspective_live_llm',
        case_id,
        trace_payload,
    )
    trace_payload['trace_path'] = str(trace_path)
    return trace_payload


def _dialog_agent_state_from_payload(payload: dict) -> dict:
    """Build a full dialog-agent state from a generator payload."""

    return {
        'internal_monologue': '用户突然发起亲昵触碰，角色需要表达边界。',
        'text_surface_output_v2': payload['text_surface_output_v2'],
        'chat_history_wide': [],
        'chat_history_recent': [],
        'debug_modes': {},
        'should_respond': True,
        'platform_user_id': 'touch-test-user',
        'platform_bot_id': 'touch-test-bot',
        'global_user_id': 'touch-test-global-user',
        'user_name': payload['user_name'],
        'user_profile': {},
        'character_profile': _character_profile(),
    }


async def _run_dialog_agent_payload(case_id: str, payload: dict) -> dict:
    """Run one payload through the full live dialog agent and save evidence."""

    state = _dialog_agent_state_from_payload(payload)
    result = await dialog_module.dialog_agent(state)
    final_dialog = result.get('final_dialog')
    joined_dialog = ''
    if isinstance(final_dialog, list):
        joined_dialog = '\n'.join(
            segment for segment in final_dialog
            if isinstance(segment, str)
        )
    trace_payload = {
        'route_model': DIALOG_GENERATOR_LLM_MODEL,
        'route_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
        'input': state,
        'result': result,
        'final_dialog': final_dialog,
        'joined_dialog': joined_dialog,
    }
    trace_path = write_llm_trace(
        'dialog_first_person_perspective_live_llm',
        case_id,
        trace_payload,
    )
    trace_payload['trace_path'] = str(trace_path)
    return trace_payload


async def test_live_dialog_agent_rewrites_touch_refusal_as_character_speech() -> None:
    """Dialog agent output should not preserve upstream third-person perspective."""

    await _skip_if_dialog_generator_unavailable()
    payload = _touch_refusal_payload()
    trace_payload = await _run_dialog_agent_payload(
        'touch_refusal_from_l3_content_plan_full_agent',
        payload,
    )
    final_dialog = trace_payload['final_dialog']
    joined_dialog = trace_payload['joined_dialog']
    forbidden_third_person_markers = [
        '他这个人',
        '对他来说',
        '看他语气',
        '让人本能地',
    ]
    present_markers = [
        marker for marker in forbidden_third_person_markers
        if marker in joined_dialog
    ]
    boundary_markers = [
        '我不',
        '我可',
        '没答应',
        '不接受',
        '不舒服',
        '想躲',
        '别',
        '不要',
    ]
    present_boundary_markers = [
        marker for marker in boundary_markers
        if marker in joined_dialog
    ]
    trace_payload['present_forbidden_markers'] = present_markers
    trace_payload['present_boundary_markers'] = present_boundary_markers

    assert isinstance(final_dialog, list), trace_payload['trace_path']
    visible_segments = [
        segment.strip()
        for segment in final_dialog
        if isinstance(segment, str) and segment.strip()
    ]
    assert visible_segments, trace_payload['trace_path']
    assert joined_dialog.strip(), trace_payload['trace_path']
    assert not present_markers, trace_payload['trace_path']
    assert present_boundary_markers, trace_payload['trace_path']


async def test_live_dialog_generator_keeps_uncertainty_reply_on_topic() -> None:
    """Style fields must not add unrelated private activity as visible text."""

    await _skip_if_dialog_generator_unavailable()
    payload = _surface_payload(
        content_plan=(
            '对喷雾式散热不了解，不能给技术判断；'
            '可以轻松表示听起来厉害，并问一句大家具体在聊哪种方案。'
        ),
        visible_boundaries=['不要装懂或扩展无关私人活动。'],
        style_guidance='群聊短句，轻微困惑但自然。',
        pacing_guidance='1 条普通文字消息，2-3个短句。',
        selected_surface_intent='回应用户问什么时候玩喷雾式散热。',
        user_name='散热测试用户',
    )
    trace_payload = await _run_generator_payload(
        'spray_cooling_uncertainty_no_private_filler',
        payload,
    )
    joined_dialog = trace_payload['joined_dialog']
    raw_output = str(trace_payload['raw_output']).strip()
    forbidden_filler_markers = [
        '挑新衣服',
        '面料',
        '刚才在',
        '随便聊聊',
    ]
    present_markers = [
        marker for marker in forbidden_filler_markers
        if marker in joined_dialog
    ]
    trace_payload['present_forbidden_markers'] = present_markers

    assert joined_dialog.strip(), trace_payload['trace_path']
    assert not raw_output.startswith('```'), trace_payload['trace_path']
    assert not present_markers, trace_payload['trace_path']


async def test_live_dialog_generator_does_not_invent_relationship_reading() -> None:
    """Dialog should not add relationship appraisal absent from the plan."""

    await _skip_if_dialog_generator_unavailable()
    payload = _surface_payload(
        content_plan=(
            '刚才把沪渝蓉全线和武宜段混在一起了；'
            '武宜段确实已经开通。'
            '可以问用户平时是否经常坐这条线。'
        ),
        visible_boundaries=['不要展开关系评价。'],
        style_guidance='坦率认错，直接、简短，轻微不好意思。',
        pacing_guidance='1 条普通文字消息，2-3个短句。',
        selected_surface_intent='承认武宜段已开通并回到铁路信息。',
        user_name='铁路测试用户',
    )
    trace_payload = await _run_generator_payload(
        'railway_correction_no_relationship_expansion',
        payload,
    )
    joined_dialog = trace_payload['joined_dialog']
    forbidden_relationship_markers = [
        '居高临下',
        '让人觉得挺舒服',
        '相处方式',
        '比端着强',
    ]
    present_markers = [
        marker for marker in forbidden_relationship_markers
        if marker in joined_dialog
    ]
    trace_payload['present_forbidden_markers'] = present_markers

    assert joined_dialog.strip(), trace_payload['trace_path']
    assert not present_markers, trace_payload['trace_path']
