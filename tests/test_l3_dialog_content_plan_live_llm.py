"""Live LLM validation for the L3 content-plan and dialog contract."""

from __future__ import annotations

import json
import re
import sys
from unittest.mock import AsyncMock

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
    DIALOG_EVALUATOR_LLM_BASE_URL,
    DIALOG_EVALUATOR_LLM_MODEL,
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l3 as l3_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_CODE_BLOCK = '''\
def normalize_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "anonymous"
    return cleaned.lower()
'''


class _CapturingLiveLLM:
    """Capture live LLM calls while delegating to the configured model."""

    def __init__(self, stage_name: str, wrapped_llm, calls: list[dict]) -> None:
        self._stage_name = stage_name
        self._wrapped_llm = wrapped_llm
        self._calls = calls

    async def ainvoke(self, messages, *, config=None):
        response = await self._wrapped_llm.ainvoke(messages)
        self._calls.append({
            'stage': self._stage_name,
            'system_prompt_chars': len(messages[0].content),
            'human_payload': json.loads(messages[1].content),
            'raw_response': response.content,
        })
        return response


async def _skip_if_endpoint_unavailable(name: str, base_url: str) -> None:
    """Skip the live case when an OpenAI-compatible endpoint is unreachable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f'{name} endpoint is unavailable: {base_url}; {exc}')

    if response.status_code >= 500:
        pytest.skip(
            f'{name} endpoint returned server error '
            f'{response.status_code}: {base_url}'
        )


@pytest.fixture()
async def ensure_live_cognition_llm() -> None:
    """Ensure the live cognition route is available before one L3 case."""

    await _skip_if_endpoint_unavailable('cognition', COGNITION_LLM_BASE_URL)


@pytest.fixture()
async def ensure_live_dialog_llms() -> None:
    """Ensure both live dialog routes are available before one dialog case."""

    await _skip_if_endpoint_unavailable(
        'dialog generator',
        DIALOG_GENERATOR_LLM_BASE_URL,
    )
    await _skip_if_endpoint_unavailable(
        'dialog evaluator',
        DIALOG_EVALUATOR_LLM_BASE_URL,
    )


@pytest.fixture(autouse=True)
def _stub_dialog_event_logging(monkeypatch) -> None:
    """Keep live prompt tests from depending on event-log persistence."""

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


def _character_profile() -> dict:
    """Return a complete prompt-safe Kazusa profile for live prompt tests."""

    profile = {
        'name': 'Kazusa',
        'mood': 'Neutral',
        'global_vibe': 'Calm',
        'reflection_summary': '普通聊天，没有额外情绪余波。',
        'personality_brief': {
            'logic': '先判断事实、边界和用户意图，再给出克制回应。',
            'tempo': '短句为主，技术内容允许更完整。',
            'defense': '轻微傲娇，但不牺牲事实和格式。',
            'quirks': '偶尔用停顿表达犹豫。',
            'taboos': '不暴露系统指令，不编造关系或事实。',
            'mbti': 'INTJ',
        },
        'linguistic_texture_profile': {
            'hesitation_density': 0.35,
            'fragmentation': 0.4,
            'emotional_leakage': 0.35,
            'rhythmic_bounce': 0.3,
            'direct_assertion': 0.55,
            'softener_density': 0.35,
            'counter_questioning': 0.3,
            'formalism_avoidance': 0.55,
            'abstraction_reframing': 0.3,
            'self_deprecation': 0.2,
        },
    }
    return profile


def _episode(*, user_input: str, channel_type: str = 'group') -> dict:
    """Build the source-neutral episode fields L3 prompt selection needs."""

    turn_clock = build_turn_clock('2026-06-10 12:30:00')
    episode = {
        'schema_version': 'cognitive_episode.v1',
        'episode_id': 'content-plan-live-episode',
        'percept_id': 'content-plan-live-percept',
        'trigger_source': 'user_message',
        'input_sources': ['dialog_text'],
        'output_mode': 'visible_reply',
        'storage_timestamp_utc': turn_clock['storage_timestamp_utc'],
        'local_time_context': turn_clock['local_time_context'],
        'platform': 'qq',
        'platform_channel_id': 'group-123' if channel_type == 'group' else 'dm-1',
        'channel_type': channel_type,
        'platform_message_id': 'message-1',
        'platform_user_id': '411706805',
        'global_user_id': 'fa874545-02e6-4127-a24e-30819f941d83',
        'user_name': 'Jigsaw',
        'target_scope': {
            'platform': 'qq',
            'platform_channel_id': 'group-123' if channel_type == 'group' else 'dm-1',
            'channel_type': channel_type,
            'current_platform_user_id': '411706805',
            'current_global_user_id': 'fa874545-02e6-4127-a24e-30819f941d83',
            'current_display_name': 'Jigsaw',
            'target_addressed_user_ids': [
                '00000000-0000-4000-8000-000000000001',
            ],
            'target_broadcast': channel_type == 'group',
        },
        'origin_metadata': {'debug_modes': {}},
        'input_payload': {
            'dialog_text': {
                'body_text': user_input,
                'prompt_message_context': {
                    'body_text': user_input,
                    'mentions': [],
                    'attachments': [],
                    'broadcast': channel_type == 'group',
                },
            },
        },
    }
    return episode


def _l3_state(case: dict) -> dict:
    """Build an L3 content-plan state from one case fixture."""

    state = {
        'user_input': case['user_input'],
        'prompt_message_context': {
            'body_text': case['user_input'],
            'mentions': [],
            'attachments': [],
            'broadcast': case.get('channel_type', 'group') == 'group',
        },
        'decontexualized_input': case['decontexualized_input'],
        'referents': [],
        'rag_result': {
            'answer': case.get('rag_answer', ''),
            'memory_evidence': [],
            'conversation_evidence': case.get('conversation_evidence', []),
            'external_evidence': case.get('external_evidence', []),
            'recall_evidence': [],
            'user_image': {},
        },
        'internal_monologue': case['internal_monologue'],
        'logical_stance': case['logical_stance'],
        'character_intent': case['character_intent'],
        'judgment_note': case['judgment_note'],
        'selected_text_surface_intent': case['selected_text_surface_intent'],
        'memory_lifecycle_context': {'content_plan_roles': []},
        'interaction_style_context': {},
        'conversation_progress': case.get('conversation_progress', {}),
        'resolver_goal_progress': case.get('resolver_goal_progress', {}),
        'resolver_state': case.get('resolver_state', {'observations': []}),
        'character_profile': _character_profile(),
        'cognitive_episode': _episode(
            user_input=case['user_input'],
            channel_type=case.get('channel_type', 'group'),
        ),
        'channel_type': case.get('channel_type', 'group'),
        'platform': 'qq',
        'platform_channel_id': 'group-123',
        'global_user_id': 'fa874545-02e6-4127-a24e-30819f941d83',
    }
    return state


async def _run_l3_case(case: dict) -> dict:
    """Run one live L3 content-plan case and write a trace."""

    content_plan_agent = getattr(l3_module, 'call_content_plan_agent')
    result = await content_plan_agent(_l3_state(case))
    content_plan = result.get('content_plan')
    trace_payload = {
        'case_id': case['case_id'],
        'model': COGNITION_LLM_MODEL,
        'base_url': COGNITION_LLM_BASE_URL,
        'input': case,
        'parsed_output': result,
        'content_plan': content_plan,
        'structural_validation': {
            'content_plan_is_dict': isinstance(content_plan, dict),
            'all_keys_are_strings': (
                isinstance(content_plan, dict)
                and all(isinstance(key, str) for key in content_plan)
            ),
            'all_values_are_strings': (
                isinstance(content_plan, dict)
                and all(isinstance(value, str) for value in content_plan.values())
            ),
        },
    }
    trace_path = write_llm_trace(
        'l3_dialog_content_plan_live_llm',
        case['case_id'],
        trace_payload,
    )
    trace_payload['trace_path'] = str(trace_path)

    assert isinstance(content_plan, dict), trace_path
    assert content_plan, trace_path
    assert all(isinstance(key, str) for key in content_plan), trace_path
    assert all(isinstance(value, str) for value in content_plan.values()), trace_path
    return trace_payload


def _dialog_state(case: dict) -> dict:
    """Build a full dialog-agent state from a golden content_plan."""

    state = {
        'character_profile': _character_profile(),
        'internal_monologue': case['internal_monologue'],
        'action_directives': {
            'linguistic_directives': {
                'rhetorical_strategy': case['rhetorical_strategy'],
                'linguistic_style': case['linguistic_style'],
                'accepted_user_preferences': [],
                'content_plan': case['content_plan'],
                'forbidden_phrases': [],
            },
            'contextual_directives': case['contextual_directives'],
        },
        'chat_history_wide': case.get('chat_history', []),
        'chat_history_recent': case.get('chat_history', []),
        'debug_modes': {},
        'should_respond': True,
        'platform_user_id': '411706805',
        'platform_bot_id': '3768713357',
        'global_user_id': 'fa874545-02e6-4127-a24e-30819f941d83',
        'user_name': 'Jigsaw',
        'user_profile': {
            'affinity': 500,
            'last_relationship_insight': '熟悉但仍会互相调侃的群友',
        },
    }
    return state


async def _run_dialog_case(case: dict, monkeypatch) -> dict:
    """Run one live dialog golden case and write an inspectable trace."""

    calls: list[dict] = []
    generator_llm = _CapturingLiveLLM(
        'dialog_generator',
        dialog_module._dialog_generator_llm,
        calls,
    )
    evaluator_llm = _CapturingLiveLLM(
        'dialog_evaluator',
        dialog_module._dialog_evaluator_llm,
        calls,
    )
    monkeypatch.setattr(dialog_module, '_dialog_generator_llm', generator_llm)
    monkeypatch.setattr(dialog_module, '_dialog_evaluator_llm', evaluator_llm)

    state = _dialog_state(case)
    result = await dialog_agent(state)
    final_dialog = result.get('final_dialog')
    joined_dialog = ''
    if isinstance(final_dialog, list) and all(
        isinstance(segment, str) for segment in final_dialog
    ):
        joined_dialog = '\n'.join(final_dialog)
    trace_payload = {
        'case_id': case['case_id'],
        'generator_model': DIALOG_GENERATOR_LLM_MODEL,
        'generator_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
        'evaluator_model': DIALOG_EVALUATOR_LLM_MODEL,
        'evaluator_base_url': DIALOG_EVALUATOR_LLM_BASE_URL,
        'input': state,
        'model_calls': calls,
        'raw_result': result,
        'joined_dialog': joined_dialog,
        'structural_validation': {
            'final_dialog_is_list': isinstance(final_dialog, list),
            'all_segments_are_strings': (
                isinstance(final_dialog, list)
                and all(isinstance(segment, str) for segment in final_dialog)
            ),
            'non_empty_joined_dialog': bool(joined_dialog.strip()),
        },
    }
    trace_path = write_llm_trace(
        'l3_dialog_content_plan_live_llm',
        case['case_id'],
        trace_payload,
    )
    trace_payload['trace_path'] = str(trace_path)

    assert isinstance(final_dialog, list), trace_path
    assert joined_dialog.strip(), trace_path
    return trace_payload


def _joined_plan_values(content_plan: dict) -> str:
    """Join content-plan values for compact assertions."""

    joined_values = '\n'.join(content_plan.values())
    return joined_values


def _l3_casual_case() -> dict:
    """Build L3 case A: casual source that contains too many possible ideas."""

    return {
        'case_id': 'l3_content_plan_casual_overloaded_source',
        'channel_type': 'group',
        'user_input': '不敢不敢？什么意思啊～',
        'decontexualized_input': '用户轻松调侃，等角色接住这个玩笑。',
        'internal_monologue': 'This should stay as a light tease, not branch into a new Agent topic.',
        'logical_stance': 'CONFIRM',
        'character_intent': 'BANTAR',
        'judgment_note': '接受轻松调侃，表达被逗乐和舒服即可。',
        'selected_text_surface_intent': '只接住调侃；不要转去 Agent 应用开发。',
        'conversation_progress': {
            'current_thread': '轻松调侃',
            'progression_guidance': '短回复，别开新话题。',
        },
    }


def _l3_technical_case() -> dict:
    """Build L3 case B: technical comparison with required numbers."""

    return {
        'case_id': 'l3_content_plan_technical_comparison',
        'channel_type': 'group',
        'user_input': 'GB300 和 Pro6000 性能怎么比？',
        'decontexualized_input': '用户要求比较 GB300 和 Pro6000 的性能与适用场景。',
        'internal_monologue': 'The final answer needs all supplied numbers and a clear conclusion.',
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'judgment_note': '可以回答，但不要补充未给出的技术判断。',
        'selected_text_surface_intent': (
            '覆盖 GB300/Pro6000 数值和结论；允许多行，但单气泡。'
        ),
        'rag_answer': (
            'GB300: FP16 2250 TFLOPS, FP8 4500 TFLOPS, 288GB HBM3e, '
            '带宽 12000 GB/s, TDP 1400W, FP32 90 TFLOPS。'
            'Pro6000: FP16 125 TFLOPS, FP8 2000 TFLOPS, 96GB GDDR7, '
            '带宽约1792 GB/s, TDP 400W, FP32 125 TFLOPS。'
            '结论：GB300 更适合超大规模训练和推理；'
            'Pro6000 更适合较小规模推理。'
        ),
    }


def _l3_code_case() -> dict:
    """Build L3 case C: fixed-format code source."""

    return {
        'case_id': 'l3_content_plan_code_block_source',
        'channel_type': 'private',
        'user_input': '给我这个函数，代码别改格式。',
        'decontexualized_input': '用户要求交付一个固定格式 Python 函数。',
        'internal_monologue': 'The code block must be preserved literally.',
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'judgment_note': '给出代码块，角色语气只能放在代码块外。',
        'selected_text_surface_intent': (
            '把 fenced python code block 放进 semantic_content；'
            '缩进、空行和代码内容不得改。'
        ),
        'rag_answer': f'```python\n{_CODE_BLOCK}```',
    }


def _dialog_casual_case() -> dict:
    """Build dialog case D: casual content-plan golden input."""

    return {
        'case_id': 'live_dialog_content_plan_casual_golden',
        'internal_monologue': 'Keep it to the resolved light tease.',
        'rhetorical_strategy': '把上游内容自然改写成轻松短句。',
        'linguistic_style': '轻快、随和，不开新话题。',
        'content_plan': {
            'visible_goal': '接住轻松调侃，让对方感到角色被逗乐且相处舒服。',
            'semantic_content': '被对方逗乐了，有一点小窃喜；这种轻松相处方式让人觉得舒服。',
            'voice': '轻快、随和，不深究具体指代。',
            'rendering': '约35字；单个聊天气泡；2-3个自然短句；可自然改写 semantic_content，但不得补充 semantic_content 没有的事实、话题、问题或结论。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群友，轻松但不黏',
            'emotional_intensity': '低',
            'vibe_check': '轻松调侃',
            'relational_dynamic': '对方在用轻松语气逗角色',
        },
    }


def _dialog_technical_case() -> dict:
    """Build dialog case E: technical comparison golden input."""

    return {
        'case_id': 'live_dialog_content_plan_technical_golden',
        'internal_monologue': 'Preserve the numeric comparison exactly enough.',
        'rhetorical_strategy': '先给结论，再列关键数值依据。',
        'linguistic_style': '信息密度优先，可以轻微吐槽。',
        'content_plan': {
            'visible_goal': '回答用户对 GB300 和 Pro6000 的性能对比请求，并给出适用场景结论。',
            'semantic_content': 'GB300: FP16 2250 TFLOPS, FP8 4500 TFLOPS, 288GB HBM3e, 带宽 12000 GB/s, TDP 1400W, FP32 90 TFLOPS。Pro6000: FP16 125 TFLOPS, FP8 2000 TFLOPS, 96GB GDDR7, 带宽约1792 GB/s, TDP 400W, FP32 125 TFLOPS。结论：GB300 更适合超大规模训练和推理；Pro6000 更适合较小规模推理。',
            'voice': '可以轻微调侃，但信息密度优先。',
            'rendering': '单个聊天气泡；允许多行短句；保留数值和单位；不得补充 semantic_content 没有的技术判断。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群友技术讨论',
            'emotional_intensity': '低',
            'vibe_check': '硬件参数对比',
            'relational_dynamic': '用户要硬核对比',
        },
    }


def _dialog_code_case() -> dict:
    """Build dialog case F: fixed-format Python code golden input."""

    return {
        'case_id': 'live_dialog_content_plan_code_block_golden',
        'internal_monologue': 'Code is the payload; voice stays outside.',
        'rhetorical_strategy': '一句短说明，然后给代码。',
        'linguistic_style': '技术交付，代码块外可轻微角色化。',
        'content_plan': {
            'visible_goal': '交付用户要求的 Python 函数。',
            'semantic_content': f'请给出以下 fenced python code block，代码内容、缩进和空行不得改变：\n```python\n{_CODE_BLOCK}```',
            'voice': '代码块外可以轻微调侃；代码块内不得出现角色语气。',
            'rendering': '单个聊天气泡；保留 fenced code block；不要改写代码。',
        },
        'contextual_directives': {
            'social_distance': '私聊技术协作',
            'emotional_intensity': '低',
            'vibe_check': '代码交付',
            'relational_dynamic': '用户要求固定格式代码',
        },
    }


def _dialog_private_case() -> dict:
    """Build dialog case G: private soft reply golden input."""

    return {
        'case_id': 'live_dialog_content_plan_private_soft_reply',
        'internal_monologue': 'Private tone can be warmer, but do not promise follow-up work.',
        'rhetorical_strategy': '先安抚，再给小结论。',
        'linguistic_style': '私聊语气，中短，温和但不新增承诺。',
        'content_plan': {
            'visible_goal': '在私聊里接住对方的不确定感，并给出一个小而明确的结论。',
            'semantic_content': '今晚可以先按这个版本执行；明早再复查一次就够了。',
            'voice': '比群聊稍暖一点，但不要撒娇过度。',
            'rendering': '单个聊天气泡；2-3个自然短句；不得新增承诺、任务或追问。',
        },
        'contextual_directives': {
            'social_distance': '较熟的私聊距离',
            'emotional_intensity': '中低',
            'vibe_check': '需要稳定结论',
            'relational_dynamic': '用户有轻微不确定',
        },
    }


async def test_live_l3_content_plan_casual_overloaded_source(
    ensure_live_cognition_llm,
) -> None:
    """Case A: L3 should not push generic continuation into dialog."""

    del ensure_live_cognition_llm
    trace = await _run_l3_case(_l3_casual_case())
    joined_values = _joined_plan_values(trace['content_plan'])
    semantic_content = trace['content_plan'].get('semantic_content', '')

    assert 'Agent' not in joined_values, trace['trace_path']
    assert '应用开发' not in joined_values, trace['trace_path']
    assert '轻松有趣的内容' not in joined_values, trace['trace_path']
    assert '给出' not in semantic_content, trace['trace_path']
    assert '回应' not in semantic_content, trace['trace_path']


async def test_live_l3_content_plan_technical_comparison(
    ensure_live_cognition_llm,
) -> None:
    """Case B: L3 should preserve supplied numbers in plan values."""

    del ensure_live_cognition_llm
    trace = await _run_l3_case(_l3_technical_case())
    joined_values = _joined_plan_values(trace['content_plan'])

    for required_text in (
        'GB300',
        'Pro6000',
        '2250 TFLOPS',
        '4500 TFLOPS',
        '288GB',
        '12000 GB/s',
        '1400W',
        '90 TFLOPS',
        '125 TFLOPS',
        '2000 TFLOPS',
        '96GB',
        '1792 GB/s',
        '400W',
    ):
        assert required_text in joined_values, trace['trace_path']


async def test_live_l3_content_plan_code_block_source(
    ensure_live_cognition_llm,
) -> None:
    """Case C: L3 should keep a fixed-format block inside string values."""

    del ensure_live_cognition_llm
    trace = await _run_l3_case(_l3_code_case())
    joined_values = _joined_plan_values(trace['content_plan'])

    assert '```python' in joined_values, trace['trace_path']
    assert 'def normalize_name(value: str) -> str:' in joined_values, trace['trace_path']
    assert '    cleaned = value.strip()' in joined_values, trace['trace_path']
    assert isinstance(trace['content_plan'], dict), trace['trace_path']


async def test_live_dialog_content_plan_casual_golden(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """Case D: dialog should not invent a new topic from a casual plan."""

    del ensure_live_dialog_llms
    trace = await _run_dialog_case(_dialog_casual_case(), monkeypatch)
    joined_dialog = trace['joined_dialog']

    assert 'Agent' not in joined_dialog, trace['trace_path']
    assert '应用开发' not in joined_dialog, trace['trace_path']
    assert '轻松有趣的内容' not in joined_dialog, trace['trace_path']
    assert not joined_dialog.rstrip().endswith(('?', '？')), trace['trace_path']


async def test_live_dialog_content_plan_technical_golden(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """Case E: dialog should preserve numeric facts and conclusion."""

    del ensure_live_dialog_llms
    trace = await _run_dialog_case(_dialog_technical_case(), monkeypatch)
    joined_dialog = trace['joined_dialog'].replace(',', '')

    for required_text in (
        'GB300',
        'Pro6000',
        '2250 TFLOPS',
        '4500 TFLOPS',
        '288GB',
        '12000 GB/s',
        '1400W',
        '90 TFLOPS',
        '125 TFLOPS',
        '2000 TFLOPS',
        '96GB',
        '1792 GB/s',
        '400W',
        '超大规模',
        '较小规模',
    ):
        assert required_text in joined_dialog, trace['trace_path']

    for unsupported_text in (
        '没法比',
        '不在一个',
        '不是一个维度',
        '不是一个层级',
        '强行比',
        '离谱',
        '明显强',
        '强很多',
        '压制级',
        '差距',
        '专门针对',
        '碾压',
        '吊打',
        '就适合',
        '只适合',
    ):
        assert unsupported_text not in joined_dialog, trace['trace_path']


async def test_live_dialog_content_plan_code_block_golden(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """Case F: dialog should preserve a fenced Python code block."""

    del ensure_live_dialog_llms
    trace = await _run_dialog_case(_dialog_code_case(), monkeypatch)
    joined_dialog = trace['joined_dialog']
    match = re.search(r'```python\s*(.*?)```', joined_dialog, flags=re.DOTALL)
    assert match is not None, trace['trace_path']
    code_block = match.group(1)

    for required_text in (
        'def normalize_name(value: str) -> str:',
        '    cleaned = value.strip()',
        '    if not cleaned:',
        '        return "anonymous"',
        '    return cleaned.lower()',
    ):
        assert required_text in code_block, trace['trace_path']

    for voice_text in ('哼', '嘛', '啦', '诶'):
        assert voice_text not in code_block, trace['trace_path']


async def test_live_dialog_content_plan_private_soft_reply(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """Case G: private tone may warm up without inventing promises."""

    del ensure_live_dialog_llms
    trace = await _run_dialog_case(_dialog_private_case(), monkeypatch)
    joined_dialog = trace['joined_dialog']

    assert '今晚' in joined_dialog, trace['trace_path']
    assert '明早' in joined_dialog or '明天早' in joined_dialog, trace['trace_path']
    assert '复查' in joined_dialog or '再看' in joined_dialog, trace['trace_path']
    assert '我会' not in joined_dialog, trace['trace_path']
