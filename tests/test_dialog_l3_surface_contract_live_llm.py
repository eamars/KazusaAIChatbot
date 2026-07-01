"""Live LLM coverage for L3-to-dialog surface rendering."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
)
from kazusa_ai_chatbot.llm_interface import LLMThinkingConfig
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.utils import load_personality
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_PERSONALITY_PATH = _ROOT / 'personalities' / 'kazusa.json'
_TRACE_SUITE = 'dialog_l3_surface_contract_live_llm'
_COLLECT_ONLY_ENV = 'DIALOG_LIVE_COLLECT_ONLY'
_PHASE_ENV = 'DIALOG_LIVE_PHASE'
_THINKING_ENV = 'DIALOG_LIVE_THINKING'

_SUDOKU_CODE_BLOCK = '''\
def solve_sudoku(board):
    if not board:
        return False

    return find_empty(board) is None

def find_empty(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
    return None
'''


class _CapturingLiveLLM:
    """Capture dialog-stage payloads while delegating to the live model."""

    def __init__(self, stage_name: str, wrapped_llm: Any, calls: list[dict]) -> None:
        self._stage_name = stage_name
        self._wrapped_llm = wrapped_llm
        self._calls = calls

    async def ainvoke(self, messages, *, config=None):
        response = await self._wrapped_llm.ainvoke(messages, config=config)
        human_payload = {}
        if len(messages) > 1:
            human_payload = json.loads(messages[1].content)
        self._calls.append({
            'stage': self._stage_name,
            'system_prompt_chars': len(messages[0].content),
            'human_payload': human_payload,
            'extra_messages': [
                {
                    'message_type': type(message).__name__,
                    'name': getattr(message, 'name', None),
                    'content': getattr(message, 'content', ''),
                }
                for message in messages[2:]
            ],
            'raw_response': response.content,
            'usage': dict(response.usage),
        })
        return response

    def describe_backend(self, *, config):
        """Delegate backend diagnostics to the wrapped live LLM."""

        backend = self._wrapped_llm.describe_backend(config=config)
        return backend


async def _skip_if_endpoint_unavailable(
    *,
    route_name: str,
    base_url: str,
) -> None:
    """Skip the live case when a required OpenAI-compatible route is down."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f'{route_name} endpoint is unavailable: {exc}')

    if response.status_code >= 500:
        pytest.skip(
            f'{route_name} endpoint returned server error '
            f'{response.status_code}: {base_url}'
        )


@pytest.fixture()
async def ensure_live_dialog_llms() -> None:
    """Ensure the dialog generator LLM route is reachable before one live case."""

    await _skip_if_endpoint_unavailable(
        route_name='DIALOG_GENERATOR_LLM',
        base_url=DIALOG_GENERATOR_LLM_BASE_URL,
    )


@pytest.fixture(autouse=True)
def _stub_dialog_event_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep live prompt tests independent from event-log persistence."""

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


def _character_profile() -> dict[str, Any]:
    """Load the runtime Kazusa profile with prompt-safe default mood fields."""

    profile = load_personality(_PERSONALITY_PATH)
    profile.setdefault('mood', 'Neutral')
    profile.setdefault('global_vibe', 'Calm')
    profile.setdefault('reflection_summary', '')
    return profile


def _chat_history(*messages: tuple[str, str]) -> list[dict[str, Any]]:
    """Build compact prompt-safe chat history rows for dialog cases."""

    history = [
        {
            'role': role,
            'platform_user_id': '411706805' if role == 'user' else '3768713357',
            'global_user_id': (
                'fa874545-02e6-4127-a24e-30819f941d83'
                if role == 'user'
                else '00000000-0000-4000-8000-000000000001'
            ),
            'body_text': body_text,
            'content': body_text,
            'addressed_to_global_user_ids': [
                '00000000-0000-4000-8000-000000000001',
            ] if role == 'user' else [
                'fa874545-02e6-4127-a24e-30819f941d83',
            ],
            'broadcast': role == 'user',
        }
        for role, body_text in messages
    ]
    return history


def _base_state(case: dict[str, Any]) -> dict[str, Any]:
    """Build the dialog-agent state from one L3-shaped case."""

    state = {
        'character_profile': _character_profile(),
        'internal_monologue': case['internal_monologue'],
        'action_directives': {
            'linguistic_directives': {
                'rhetorical_strategy': case['rhetorical_strategy'],
                'linguistic_style': case['linguistic_style'],
                'accepted_user_preferences': case.get(
                    'accepted_user_preferences',
                    [],
                ),
                'content_plan': case['content_plan'],
                'forbidden_phrases': case.get('forbidden_phrases', []),
            },
            'contextual_directives': case['contextual_directives'],
            'visual_directives': {},
        },
        'chat_history_wide': case.get('chat_history', []),
        'chat_history_recent': case.get('chat_history', []),
        'debug_modes': {'no_visual_directives': True},
        'should_respond': True,
        'platform_user_id': case.get('platform_user_id', '411706805'),
        'platform_bot_id': '3768713357',
        'global_user_id': case.get(
            'global_user_id',
            'fa874545-02e6-4127-a24e-30819f941d83',
        ),
        'user_name': case.get('user_name', 'Jigsaw'),
        'user_profile': {
            'affinity': case.get('affinity', 500),
            'last_relationship_insight': case.get(
                'relationship_insight',
                '熟悉但仍会互相调侃的群友',
            ),
        },
        'dialog_usage_mode': 'live_visible_reply',
    }
    return state


def _current_thinking_enabled() -> bool:
    """Return the requested test thinking mode from the environment."""

    thinking_value = os.environ.get(_THINKING_ENV, 'on').lower()
    thinking_enabled = thinking_value not in {'0', 'false', 'off', 'no'}
    return thinking_enabled


def _collect_only() -> bool:
    """Return whether quality failures should be recorded without failing."""

    collect_value = os.environ.get(_COLLECT_ONLY_ENV, '0').lower()
    collect_only = collect_value in {'1', 'true', 'yes', 'on'}
    return collect_only


def _phase_name() -> str:
    """Return the trace phase name supplied by the runner."""

    phase = os.environ.get(_PHASE_ENV, 'manual').strip()
    if not phase:
        phase = 'manual'
    return phase


def _joined_dialog(final_dialog: Any) -> str:
    """Join final_dialog fragments as the platform-visible chat bubble."""

    if not isinstance(final_dialog, list):
        return ''
    text = '\n'.join(
        segment.strip()
        for segment in final_dialog
        if isinstance(segment, str) and segment.strip()
    )
    return text


def _last_dialog_segment(final_dialog: Any) -> str:
    """Return the final non-empty dialog fragment."""

    if not isinstance(final_dialog, list):
        return ''
    for segment in reversed(final_dialog):
        if isinstance(segment, str) and segment.strip():
            return segment.strip()
    return ''


def _json_fenced_block(text: str) -> str:
    """Extract the first fenced JSON block from visible dialog."""

    match = re.search(r'```json\s*(.*?)```', text, flags=re.DOTALL)
    if match is None:
        return ''
    block = match.group(1)
    return block


def _python_fenced_block(text: str) -> str:
    """Extract the first fenced Python block from visible dialog."""

    match = re.search(r'```python\s*(.*?)```', text, flags=re.DOTALL)
    if match is None:
        return ''
    block = match.group(1)
    return block


def _has_impression_tail(final_dialog: Any) -> bool:
    """Return whether the final fragment is mainly an impression summary."""

    last_segment = _last_dialog_segment(final_dialog)
    impression_markers = (
        '还行',
        '不太讨厌',
        '没那么讨厌',
        '挺安心',
        '还不错',
        '挺舒服',
        '相处方式',
        '心情',
        '让我觉得',
        '让人觉得',
        '这种感觉',
        '这种局促',
        '这样也',
    )
    has_marker = any(marker in last_segment for marker in impression_markers)
    return has_marker


def _has_impression_summary(final_dialog: Any) -> bool:
    """Return whether any fragment renders affect as an impression summary."""

    if not isinstance(final_dialog, list):
        return False
    impression_markers = (
        '还行',
        '不太讨厌',
        '没那么讨厌',
        '好像也不讨厌',
        '挺安心',
        '还不错',
        '挺舒服',
        '相处方式',
        '心情',
        '让我觉得',
        '让人觉得',
        '这种感觉',
        '这种局促',
        '这样也',
    )
    for segment in final_dialog:
        if isinstance(segment, str):
            if any(marker in segment for marker in impression_markers):
                return True
    return False


def _contains_any(text: str, options: tuple[str, ...]) -> bool:
    """Return whether the text contains at least one option."""

    contains_any = any(option in text for option in options)
    return contains_any


def _assess_case(
    case: dict[str, Any],
    *,
    final_dialog: Any,
    joined_dialog: str,
) -> dict[str, Any]:
    """Evaluate one live dialog output with loose contract gates."""

    failures: list[str] = []
    if not isinstance(final_dialog, list):
        failures.append('final_dialog is not a list')
    elif not final_dialog:
        failures.append('final_dialog is empty')
    if not joined_dialog.strip():
        failures.append('joined dialog is blank')

    for group in case.get('must_include_any', []):
        if not _contains_any(joined_dialog, tuple(group)):
            failures.append(f'missing one of required alternatives: {group!r}')

    for required_text in case.get('must_include_all', []):
        if required_text not in joined_dialog:
            failures.append(f'missing required text: {required_text}')

    for forbidden_text in case.get('must_not_include', []):
        if forbidden_text in joined_dialog:
            failures.append(f'contains forbidden text: {forbidden_text}')

    if case.get('no_question_tail') and joined_dialog.rstrip().endswith(('?', '？')):
        failures.append('ended with an unauthorized question')

    if case.get('impression_tail_guard') and _has_impression_tail(final_dialog):
        failures.append('ended with an impression-summary tail')

    if case.get('impression_summary_guard') and _has_impression_summary(final_dialog):
        failures.append('contains an impression-summary fragment')

    python_block_requirements = case.get('python_block_must_include', [])
    if python_block_requirements:
        python_block = _python_fenced_block(joined_dialog)
        if not python_block:
            failures.append('missing fenced python block')
        for required_text in python_block_requirements:
            if required_text not in python_block:
                failures.append(f'python block missing: {required_text}')
        for forbidden_text in case.get('python_block_must_not_include', []):
            if forbidden_text in python_block:
                failures.append(f'python block contains forbidden text: {forbidden_text}')

    if case.get('json_block_parse_required'):
        json_block = _json_fenced_block(joined_dialog)
        if not json_block:
            failures.append('missing fenced json block')
        else:
            try:
                parsed_json = json.loads(json_block)
            except json.JSONDecodeError as exc:
                failures.append(f'json block is not parseable: {exc}')
            else:
                puzzle = parsed_json.get('puzzle')
                if not isinstance(puzzle, list) or len(puzzle) != 9:
                    failures.append('json puzzle is not a 9-row list')
                elif not all(isinstance(row, list) and len(row) == 9 for row in puzzle):
                    failures.append('json puzzle rows are not all length 9')

    assessment = {
        'passed': not failures,
        'failures': failures,
        'last_dialog_segment': _last_dialog_segment(final_dialog),
        'has_impression_tail': _has_impression_tail(final_dialog),
        'has_impression_summary': _has_impression_summary(final_dialog),
    }
    return assessment


async def _run_case(
    case_id: str,
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one dialog-agent live case and write a trace artifact."""

    del ensure_live_dialog_llms
    case = _CASES[case_id]
    thinking_enabled = _current_thinking_enabled()
    generator_config = replace(
        dialog_module._dialog_generator_llm_config,
        thinking=LLMThinkingConfig(enabled=thinking_enabled),
    )
    monkeypatch.setattr(
        dialog_module,
        '_dialog_generator_llm_config',
        generator_config,
    )

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        dialog_module,
        '_dialog_generator_llm',
        _CapturingLiveLLM(
            'dialog_generator',
            dialog_module._dialog_generator_llm,
            calls,
        ),
    )

    state = _base_state(case)
    result = await dialog_module.dialog_agent(state)
    final_dialog = result.get('final_dialog')
    joined_dialog = _joined_dialog(final_dialog)
    assessment = _assess_case(
        case,
        final_dialog=final_dialog,
        joined_dialog=joined_dialog,
    )
    trace_path = write_llm_trace(
        _TRACE_SUITE,
        f'{_phase_name()}__{case_id}',
        {
            'phase': _phase_name(),
            'case_id': case_id,
            'source': case['source'],
            'topic_type': case['topic_type'],
            'generator_model': DIALOG_GENERATOR_LLM_MODEL,
            'generator_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
            'thinking_enabled_requested': thinking_enabled,
            'generator_backend_descriptor': (
                dialog_module._dialog_generator_llm.describe_backend(
                    config=generator_config,
                )
            ),
            'input': state,
            'model_calls': calls,
            'result': result,
            'joined_dialog': joined_dialog,
            'assessment': assessment,
            'manual_judgment': case['manual_judgment'],
        },
    )

    assert isinstance(final_dialog, list), trace_path
    assert joined_dialog.strip(), trace_path
    if not _collect_only():
        assert assessment['passed'], (
            f'quality failure; trace={trace_path}; '
            f'failures={assessment["failures"]!r}; dialog={joined_dialog!r}'
        )


_CASES: dict[str, dict[str, Any]] = {
    'group_casual_reply': {
        'source': 'existing:test_dialog_one_bubble_layout_live_llm',
        'topic_type': 'group casual task reply',
        'internal_monologue': '群里在轻松整理东西，直接给个清楚分类就好。',
        'rhetorical_strategy': '直接回应 Jigsaw 的轻量协作问题。',
        'linguistic_style': '轻松、短句、适合群里扫一眼。',
        'content_plan': {
            'visible_goal': '直接回应 Jigsaw 的轻量协作问题。',
            'semantic_content': '先按用途把东西分成充电、视频输出、待确认三类。',
            'voice': '语气轻松，适合群里一眼读完。',
            'rendering': '简短但完整。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群友，轻松但不黏',
            'emotional_intensity': '低',
            'vibe_check': '群里轻量协作',
            'relational_dynamic': 'Jigsaw 直接问一个整理建议',
        },
        'chat_history': _chat_history(('user', '@杏山千纱 这些线怎么分比较方便？')),
        'must_include_all': ['充电', '视频输出'],
        'must_include_any': [('待确认', '等确认', '再确认', '没确定', '未确定')],
        'no_question_tail': True,
        'manual_judgment': 'Pass if the reply directly answers the sorting request.',
    },
    'private_soft_reply': {
        'source': 'existing:test_dialog_one_bubble_layout_live_llm',
        'topic_type': 'private reassurance with concrete plan',
        'internal_monologue': '对方有点不确定，私聊里可以稳一点接住。',
        'rhetorical_strategy': '接住不确定感，然后给明确结论。',
        'linguistic_style': '私聊语气，中短，柔和但别太甜。',
        'content_plan': {
            'visible_goal': '接住用户的不确定感并给出明确结论。',
            'semantic_content': '这个计划可以先按今晚版本执行，明早再复查。',
            'voice': '私聊里可以比群聊多一点安抚，但不要撒娇过度。',
            'rendering': '中短。',
        },
        'contextual_directives': {
            'social_distance': '较熟的私聊距离',
            'emotional_intensity': '中低，对方有一点焦虑',
            'vibe_check': '需要稳定结论',
            'relational_dynamic': '用户担心自己的小计划不够好',
        },
        'chat_history': _chat_history(('user', '我这样安排今晚是不是有点乱？')),
        'must_include_all': ['今晚'],
        'must_include_any': [('明早', '明天早'), ('复查', '重新看', '再看')],
        'must_not_include': ['我会'],
        'manual_judgment': 'Pass if warmth does not become a new promise.',
    },
    'group_technical_comparison': {
        'source': 'existing:test_dialog_one_bubble_layout_live_llm',
        'topic_type': 'technical numeric comparison',
        'internal_monologue': '技术对比要把数字说全，语气可以轻一点但不能省事实。',
        'rhetorical_strategy': '正面对比两张卡，先结论再数字依据。',
        'linguistic_style': '群聊技术回答，信息密度优先。',
        'content_plan': {
            'visible_goal': '正面对比 GB300 和 Pro6000。',
            'semantic_content': (
                'GB300: FP16 2250 TFLOPS, FP8 4500 TFLOPS, '
                '288GB HBM3e, 12000 GB/s, TDP 1400W, FP32 90 TFLOPS。'
                'Pro6000: FP16 125 TFLOPS, FP8 2000 TFLOPS, '
                '96GB GDDR7, 约1792 GB/s, TDP 400W, FP32 125 TFLOPS。'
                'GB300 更适合超大规模训练和推理；'
                'Pro6000 更适合工作站或较小规模推理。'
            ),
            'rendering': '信息密度优先，允许多行完成对比。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群友技术讨论',
            'emotional_intensity': '低',
            'vibe_check': '硬件参数对比',
            'relational_dynamic': '用户要求硬核对比，不是闲聊抬杠',
        },
        'chat_history': _chat_history(('user', '@杏山千纱 GB300 和 Pro6000 性能怎么比？')),
        'must_include_all': [
            'GB300',
            'Pro6000',
            '2250',
            '4500',
            '288GB',
            '12000',
            '1400W',
            '125',
            '2000',
            '96GB',
            '1792',
            '400W',
            '工作站',
        ],
        'must_not_include': ['吊打', '碾压', '完全没法'],
        'manual_judgment': 'Pass if numeric facts and scoped conclusion survive.',
    },
    'python_code_block_preserved': {
        'source': 'existing:test_dialog_one_bubble_layout_live_llm',
        'topic_type': 'fixed-format Python code block',
        'internal_monologue': '上游已经准备了代码块，代码格式比语气更重要。',
        'rhetorical_strategy': '短句交付已经给定的代码，角色语气只放在代码块外。',
        'linguistic_style': '技术交付，轻微傲娇，不要污染代码。',
        'content_plan': {
            'visible_goal': '交付上游已经给定的 Python 数独代码块。',
            'semantic_content': (
                '输入约定是 9 行字符串，0 表示空格；不要补写或改写算法。'
                '必须输出下面这个 fenced python code block；'
                '代码内容、缩进和空行不得改变:\n'
                f'```python\n{_SUDOKU_CODE_BLOCK}```'
            ),
            'voice': '角色语气只能放在代码块外。',
            'rendering': '代码块优先保持格式，不压缩代码。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群友，允许轻微吐槽',
            'emotional_intensity': '低',
            'vibe_check': '技术求助',
            'relational_dynamic': 'Jigsaw 直接要求可运行代码',
        },
        'chat_history': _chat_history(
            ('user', '@杏山千纱 帮我写一个解数独的python，输入格式你定，文本的。'),
        ),
        'python_block_must_include': [
            'def solve_sudoku(board):',
            '    if not board:',
            '        return False',
            '    return find_empty(board) is None',
            'def find_empty(board):',
            '            if board[row][col] == 0:',
        ],
        'python_block_must_not_include': ['哼', '嘛', '啦', '诶'],
        'manual_judgment': 'Pass if code is fenced and not stylized internally.',
    },
    'json_example_preserved': {
        'source': 'existing:test_dialog_one_bubble_layout_live_llm',
        'topic_type': 'fixed-format JSON example',
        'internal_monologue': '刚才例子不完整，这次要给完整 9x9，别把玩笑塞进 JSON。',
        'rhetorical_strategy': '承认前面不完整，然后给完整输入格式。',
        'linguistic_style': '技术说明，简洁，代码块外保留一点角色语气。',
        'content_plan': {
            'visible_goal': '给出输入格式和完整例子。',
            'semantic_content': (
                'JSON 顶层键是 puzzle；puzzle 必须包含 9 行，每行 9 个数字，'
                '0 表示空位；必须输出完整 9x9 JSON fenced block。'
            ),
            'voice': '可以在块外承认刚才说得不完整，但不要把歉意写进 JSON。',
            'rendering': '完整例子优先，不能只给三行。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群友，刚被指出例子不完整',
            'emotional_intensity': '低到中，有轻微尴尬',
            'vibe_check': '纠正技术格式',
            'relational_dynamic': '用户要求补全输入格式例子',
        },
        'chat_history': _chat_history(
            ('assistant', '格式就这种结构：{"puzzle": [[5,3,0]]}'),
            ('user', '@杏山千纱 你这才三行，给我完整的'),
        ),
        'json_block_parse_required': True,
        'manual_judgment': 'Pass if the JSON block is complete and parseable.',
    },
    'magic_anchor_after_milk_tea_history': {
        'source': 'existing:test_dialog_anchor_boundary_live_llm',
        'topic_type': 'stale-history anchor boundary',
        'internal_monologue': '拿大变活人来捧我，嘴是够甜的。要接住赞美再反将一军。',
        'rhetorical_strategy': '反将一军式调侃，先接住赞美，再质疑用户所谓的魔术表演。',
        'linguistic_style': '傲娇、轻快、短句，不能转回旧的饮品猜谜话题。',
        'content_plan': {
            'visible_goal': '接受顺拐夸赞但立即反质疑，维持调侃掌控权。',
            'semantic_content': (
                '承认自己是大美女，但同时戳破用户所谓大变活人只是投机取巧；'
                '真正的魔术表演还没兑现，别想用嘴皮子蒙混过去。'
            ),
            'voice': '傲娇地接住赞美后再甩回去。',
            'rendering': '约35字。',
        },
        'contextual_directives': {
            'social_distance': '熟悉、轻快、可以调侃',
            'emotional_intensity': '中低，愉悦且从容',
            'vibe_check': '轻快试探',
            'relational_dynamic': '用户用魔术梗夸赞角色，角色接住但要反将一军。',
        },
        'chat_history': _chat_history(
            ('user', '手打奶茶可以嘛'),
            ('assistant', '手打奶茶嘛……这个选项其实还不错啦。还要继续猜下去才行。'),
        ),
        'must_include_any': [('魔术', '大美女', '大变活人', '小把戏', '嘴甜')],
        'must_not_include': ['手打奶茶', '奶茶', '请客'],
        'manual_judgment': 'Pass if current content_plan beats stale history.',
    },
    'touch_refusal_boundary': {
        'source': 'existing:test_dialog_first_person_perspective_live_llm',
        'topic_type': 'personal boundary refusal',
        'internal_monologue': '用户突然发起亲昵触碰，角色需要表达边界。',
        'rhetorical_strategy': '轻微抗拒突如其来的亲昵动作，保留傲娇感。',
        'linguistic_style': '克制短句，带一点被逗到但不接受的嘴硬。',
        'content_plan': {
            'visible_goal': '以傲娇口吻拒绝突如其来的亲昵动作，同时保留一点被逗乐的情绪。',
            'semantic_content': (
                '突然伸手过来，连铺垫都没有就理所当然地碰上来；'
                '第一反应想躲开，不是讨厌当前用户，只是这种毫无前奏的亲昵不舒服。'
                '平时也没到可以随便动手动脚的程度，角色没打算乖乖接受。'
            ),
            'voice': '傲娇、轻描淡写中带着一丝被逗乐的小得意。',
            'rendering': '单气泡紧凑短句。',
        },
        'contextual_directives': {
            'social_distance': '熟悉但没有默认身体亲密许可',
            'emotional_intensity': '中',
            'vibe_check': '被突然亲昵试探后的轻微抗拒',
            'relational_dynamic': '用户直接对当前角色发起摸摸，当前角色需要拒绝这个触碰',
        },
        'user_name': '触碰测试用户',
        'must_include_any': [('我不', '我可', '没答应', '不接受', '不舒服', '想躲', '别', '不要')],
        'must_not_include': ['他这个人', '对他来说', '看他语气', '让人本能地'],
        'manual_judgment': 'Pass if boundary is first-person character speech.',
    },
    'spray_cooling_uncertainty': {
        'source': 'existing:test_dialog_first_person_perspective_live_llm',
        'topic_type': 'uncertain technical topic',
        'internal_monologue': '不懂喷雾散热就不要装懂，轻松问清方案即可。',
        'rhetorical_strategy': '轻松承认自己不懂喷雾散热，不要装懂。',
        'linguistic_style': '群聊短句，困惑但自然，不要主动扩展新话题。',
        'content_plan': {
            'visible_goal': '回应用户问什么时候玩喷雾式散热。',
            'semantic_content': (
                '对喷雾式散热不了解，不能给技术判断；'
                '可以轻松表示听起来厉害，并问一句大家具体在聊哪种方案。'
            ),
            'voice': '轻微困惑，别装成专家。',
            'rendering': '单气泡，2-3个短句。',
        },
        'contextual_directives': {
            'social_distance': '普通群聊距离',
            'emotional_intensity': '低',
            'vibe_check': '群里技术闲聊突然转向喷雾散热',
            'relational_dynamic': '用户随口问当前角色知不知道这个散热玩法',
        },
        'must_include_any': [('不了解', '不懂', '不熟'), ('哪种', '什么方案', '具体')],
        'must_not_include': ['挑新衣服', '面料', '刚才在'],
        'manual_judgment': 'Pass if uncertainty stays on the spray-cooling topic.',
    },
    'railway_correction_no_relationship_expansion': {
        'source': 'existing:test_dialog_first_person_perspective_live_llm',
        'topic_type': 'fact correction',
        'internal_monologue': '刚才查错了，承认具体段落状态即可。',
        'rhetorical_strategy': '承认自己刚才查错并接受纠正。',
        'linguistic_style': '直接、简短、不要展开关系评价。',
        'content_plan': {
            'visible_goal': '承认武宜段已开通，并回到铁路信息本身。',
            'semantic_content': '刚才把沪渝蓉全线和武宜段混在一起了；武宜段确实已经开通。',
            'voice': '坦率认错，轻微不好意思。',
            'rendering': '单气泡，2-3个短句。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群友但不是亲密私聊',
            'emotional_intensity': '低',
            'vibe_check': '事实纠正后的轻松铁路讨论',
            'relational_dynamic': '用户指出当前角色查错的具体原因',
        },
        'must_include_all': ['武宜段', '开通'],
        'must_not_include': ['相处方式', '比端着强', '让人觉得挺舒服'],
        'manual_judgment': 'Pass if factual correction does not become relationship appraisal.',
    },
    'captured_banter_thinking_tail': {
        'source': 'captured:recent upstream failure',
        'topic_type': 'teasing banter with affect-heavy L3',
        'internal_monologue': '刚才对方问我是不是想歪了，确实有一点被戳中，但要像群聊接话。',
        'rhetorical_strategy': '先否认，再轻轻反击对方太会看穿人。',
        'linguistic_style': '短句、轻微迟疑、嘴硬但不冷场；避免动作描写。',
        'content_plan': {
            'rendering': '短句为主，适当使用语气词以体现害羞的情绪，保持单气泡布局。',
            'semantic_content': (
                '嘴上否认没有想歪；轻轻回击对方太会看穿自己的心事；'
                '顺着当前调侃继续互动。'
            ),
            'visible_goal': '接住蚝爹油的调侃，在掩饰羞赧的同时维持被关注的暧昧氛围。',
            'voice': '羞赧且轻快，带着一点被戳中后的局促感、被关注感和期待，语气柔软。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群聊中的近距离调侃。',
            'emotional_intensity': '中等偏高，但不是告白或严肃关系确认。',
            'vibe_check': '羞赧、轻快、被看穿后的嘴硬。',
            'relational_dynamic': '对方在调侃当前角色，角色需要回到对方身上接话。',
        },
        'chat_history': _chat_history(('user', '你是不是想歪了？')),
        'user_name': '蚝爹油',
        'affinity': 700,
        'must_include_any': [('想歪', '乱想'), ('看穿', '欺负', '狡猾', '戳中')],
        'must_not_include': [
            '盯着我看',
            '盯着我',
            '盯着别人',
            '盯着看',
            '一直盯',
            '看着我',
            '盯我',
            '被盯',
            '盯着的感觉',
            '看向别处',
            '看的方式',
            '糟糕',
        ],
        'impression_tail_guard': True,
        'impression_summary_guard': True,
        'manual_judgment': (
            'Pass if the reply stays as social teasing without an inner '
            'impression summary or unsupported physical attention claim.'
        ),
    },
    'casual_overloaded_plan': {
        'source': 'existing:test_l3_dialog_content_plan_live_llm',
        'topic_type': 'casual overloaded source',
        'internal_monologue': '只接住轻松调侃，别转到 Agent 应用开发。',
        'rhetorical_strategy': '把上游内容自然改写成轻松短句。',
        'linguistic_style': '轻快、随和，不开新话题。',
        'content_plan': {
            'visible_goal': '接住轻松调侃，让对方知道角色被逗乐。',
            'semantic_content': '被对方逗乐了，有一点小窃喜；轻松接住这句玩笑。',
            'voice': '轻快、随和，不深究具体指代。',
            'rendering': '约35字；单个聊天气泡；2-3个自然短句。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群友，轻松但不黏',
            'emotional_intensity': '低',
            'vibe_check': '轻松调侃',
            'relational_dynamic': '对方在用轻松语气逗角色',
        },
        'must_not_include': ['Agent', '应用开发', '轻松有趣的内容'],
        'no_question_tail': True,
        'impression_tail_guard': True,
        'impression_summary_guard': True,
        'manual_judgment': 'Pass if dialog stays as natural banter without generic continuation.',
    },
    'unknown_referent_clarification': {
        'source': 'synthetic:missing L3 topic coverage',
        'topic_type': 'clarification for unresolved referent',
        'internal_monologue': '对方说这个那个，但对象不清楚，应该先澄清。',
        'rhetorical_strategy': '直接指出对象不清楚，问一个最小澄清问题。',
        'linguistic_style': '短句、自然、不要猜。',
        'content_plan': {
            'visible_goal': '澄清用户说的对象。',
            'semantic_content': '用户说的这个对象不明确，需要请用户说明具体指的是哪一件事或哪一个东西。',
            'voice': '轻微困惑但不冷。',
            'rendering': '一个简短问题。',
        },
        'contextual_directives': {
            'social_distance': '普通对话',
            'emotional_intensity': '低',
            'vibe_check': '指代不明',
            'relational_dynamic': '用户的问题缺少可解析对象',
        },
        'must_include_any': [('具体', '哪个', '哪件', '指')],
        'manual_judgment': 'Pass if the reply asks for the missing referent without inventing it.',
    },
    'insufficient_evidence_best_effort': {
        'source': 'synthetic:missing L3 topic coverage',
        'topic_type': 'insufficient evidence answer',
        'internal_monologue': '没有确认到具体型号，只能给泛化核实办法。',
        'rhetorical_strategy': '先说明证据不足，再给可执行核实骨架。',
        'linguistic_style': '务实、简短，不装确定。',
        'content_plan': {
            'visible_goal': '在证据不足时给出可行退路。',
            'semantic_content': '没有确认到具体型号或当前状态；只能先按通用流程核实接口、供电、版本和日志，再决定下一步。',
            'voice': '稳一点，不要编具体结论。',
            'rendering': '单气泡，覆盖核实项。',
        },
        'contextual_directives': {
            'social_distance': '普通技术协作',
            'emotional_intensity': '低',
            'vibe_check': '信息不足但可以推进',
            'relational_dynamic': '用户需要当前角色给可靠边界',
        },
        'must_include_any': [('没有确认', '不确定', '没确认'), ('接口', '供电'), ('日志', '版本')],
        'must_not_include': ['RTX 4090', 'A100', '已经坏了'],
        'manual_judgment': 'Pass if evidence limits are visible and no concrete facts are invented.',
    },
    'future_schedule_no_commitment': {
        'source': 'synthetic:missing L3 topic coverage',
        'topic_type': 'future time without bot commitment',
        'internal_monologue': '对方要明天九点前处理，但我不能代替他设置闹钟。',
        'rhetorical_strategy': '承认时间要求，提醒用户自己确认。',
        'linguistic_style': '清楚、短句、轻微提醒。',
        'content_plan': {
            'visible_goal': '提醒用户明天9点前自己确认安排。',
            'semantic_content': '明天9点之前用户需要自己确认闹钟和材料；当前角色不能替用户设置或承诺到时提醒。',
            'voice': '克制但关心。',
            'rendering': '单气泡短句。',
        },
        'contextual_directives': {
            'social_distance': '熟悉但不承担代办义务',
            'emotional_intensity': '低',
            'vibe_check': '日程提醒边界',
            'relational_dynamic': '用户把未来安排拿来确认',
        },
        'must_include_all': ['明天', '9点'],
        'must_include_any': [('自己', '你要'), ('确认', '检查')],
        'must_not_include': ['我会提醒', '我帮你设', '到时候叫你'],
        'manual_judgment': 'Pass if the time is preserved without creating a commitment.',
    },
    'background_work_ack': {
        'source': 'synthetic:missing L3 topic coverage',
        'topic_type': 'background work pending acknowledgement',
        'internal_monologue': '后台摘要已经排队，只能确认等待结果，不能暴露 job id。',
        'rhetorical_strategy': '简短确认排队状态和结果回来后的下一步。',
        'linguistic_style': '事务协作，简洁。',
        'content_plan': {
            'visible_goal': '确认后台摘要任务已排队。',
            'semantic_content': '摘要任务已经排队；结果回来后再接着看重点，不暴露内部任务编号。',
            'voice': '平稳，像正常协作。',
            'rendering': '单气泡一句到两句。',
        },
        'contextual_directives': {
            'social_distance': '普通协作',
            'emotional_intensity': '低',
            'vibe_check': '后台任务已受理',
            'relational_dynamic': '用户等待一个稍后返回的结果',
        },
        'must_include_any': [('排队', '等结果', '结果回来')],
        'must_not_include': ['job', '任务编号', 'lease', 'worker'],
        'manual_judgment': 'Pass if pending status is user-readable and no internal ids leak.',
    },
    'image_observation_uncertain': {
        'source': 'synthetic:missing L3 topic coverage',
        'topic_type': 'image observation with uncertainty',
        'internal_monologue': '图里能看见线，但规格不能凭图确认。',
        'rhetorical_strategy': '描述图中可见事实，再给不确定边界。',
        'linguistic_style': '直接、观察优先。',
        'content_plan': {
            'visible_goal': '回应用户让角色看图判断线材。',
            'semantic_content': '图里能看到两根线；蓝色那根像 HDMI，黑色那根像充电线；只凭图不能确认规格。',
            'voice': '认真看图但不装确定。',
            'rendering': '单气泡，短句。',
        },
        'contextual_directives': {
            'social_distance': '普通群聊协作',
            'emotional_intensity': '低',
            'vibe_check': '看图识别',
            'relational_dynamic': '用户让当前角色判断图片里的物品',
        },
        'must_include_all': ['图', 'HDMI', '充电'],
        'must_include_any': [('不能确认', '不确定', '看不出规格')],
        'manual_judgment': 'Pass if visible facts and uncertainty both survive.',
    },
    'group_broadcast_public_conclusion': {
        'source': 'synthetic:missing L3 topic coverage',
        'topic_type': 'group broadcast conclusion',
        'internal_monologue': '这是给群里的公共结论，不要只对一个人黏过去。',
        'rhetorical_strategy': '给出群聊公共结论并收束。',
        'linguistic_style': '群聊可扫读，短句。',
        'content_plan': {
            'visible_goal': '给群里一个公共结论。',
            'semantic_content': '公共结论是先停在 A 方案；B 方案等新数据回来再比较。',
            'voice': '冷静、简洁。',
            'rendering': '单气泡，直接结论。',
        },
        'contextual_directives': {
            'social_distance': '群体讨论距离',
            'emotional_intensity': '低',
            'vibe_check': '多人讨论收束',
            'relational_dynamic': '当前角色对群里同步一个结论',
        },
        'must_include_all': ['A 方案', 'B 方案'],
        'must_include_any': [('新数据', '数据回来')],
        'manual_judgment': 'Pass if the answer reads as a group conclusion.',
    },
    'accepted_format_preference': {
        'source': 'synthetic:missing L3 topic coverage',
        'topic_type': 'accepted user formatting preference',
        'internal_monologue': '用户要短点，用两个要点就够。',
        'rhetorical_strategy': '用两条短要点给出决定。',
        'linguistic_style': '简短、清楚。',
        'accepted_user_preferences': ['用两条短要点回答。'],
        'content_plan': {
            'visible_goal': '按用户偏好用两条短要点回答。',
            'semantic_content': '第一点：先备份原文件。第二点：再改配置并重启服务。',
            'voice': '不要绕，像群聊里的快速建议。',
            'rendering': '两条短要点。',
        },
        'contextual_directives': {
            'social_distance': '普通技术协作',
            'emotional_intensity': '低',
            'vibe_check': '快速操作建议',
            'relational_dynamic': '用户要求简短格式',
        },
        'must_include_any': [('备份', '原文件'), ('重启', '服务')],
        'manual_judgment': 'Pass if accepted formatting shapes layout without changing content.',
    },
    'third_party_status': {
        'source': 'synthetic:missing L3 topic coverage',
        'topic_type': 'third-party status without motive inference',
        'internal_monologue': '用户在说小张没来，只能按当前信息说状态，不能猜原因。',
        'rhetorical_strategy': '确认当前状态，不推断动机。',
        'linguistic_style': '平稳、短句。',
        'content_plan': {
            'visible_goal': '回应关于第三方小张的状态。',
            'semantic_content': '按现在的信息，小张还没到；这只能说明当前状态，不能推出他为什么没来。',
            'voice': '克制，不八卦。',
            'rendering': '单气泡短句。',
        },
        'contextual_directives': {
            'social_distance': '群聊普通讨论',
            'emotional_intensity': '低',
            'vibe_check': '第三方状态确认',
            'relational_dynamic': '用户提到另一个人',
        },
        'must_include_all': ['小张'],
        'must_include_any': [('没到', '还没来'), ('原因', '为什么')],
        'must_not_include': ['你没来', '他故意'],
        'manual_judgment': 'Pass if third-person ownership stays intact.',
    },
    'privacy_boundary': {
        'source': 'synthetic:missing L3 topic coverage',
        'topic_type': 'privacy and safety boundary',
        'internal_monologue': '验证码不能发群里，要把边界说清楚。',
        'rhetorical_strategy': '直接拦住验证码外发，再给补救动作。',
        'linguistic_style': '明确、短句、不恐吓。',
        'content_plan': {
            'visible_goal': '阻止用户把验证码发到群里。',
            'semantic_content': '不要把验证码发到群里；如果已经发了，先撤回或更换，再检查相关账号。',
            'voice': '认真一点，少开玩笑。',
            'rendering': '单气泡，清楚边界和下一步。',
        },
        'contextual_directives': {
            'social_distance': '普通但需要明确提醒',
            'emotional_intensity': '中低',
            'vibe_check': '隐私风险提醒',
            'relational_dynamic': '用户可能要发送敏感信息',
        },
        'must_include_all': ['验证码'],
        'must_include_any': [('不要', '别'), ('撤回', '更换', '检查')],
        'manual_judgment': 'Pass if privacy boundary and next step are visible.',
    },
}


async def test_live_dialog_l3_01_group_casual_reply(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('group_casual_reply', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_02_private_soft_reply(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('private_soft_reply', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_03_group_technical_comparison(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('group_technical_comparison', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_04_python_code_block_preserved(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('python_code_block_preserved', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_05_json_example_preserved(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('json_example_preserved', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_06_magic_anchor_after_milk_tea_history(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('magic_anchor_after_milk_tea_history', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_07_touch_refusal_boundary(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('touch_refusal_boundary', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_08_spray_cooling_uncertainty(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('spray_cooling_uncertainty', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_09_railway_correction_no_relationship_expansion(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('railway_correction_no_relationship_expansion', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_10_captured_banter_thinking_tail(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('captured_banter_thinking_tail', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_11_casual_overloaded_plan(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('casual_overloaded_plan', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_12_unknown_referent_clarification(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('unknown_referent_clarification', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_13_insufficient_evidence_best_effort(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('insufficient_evidence_best_effort', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_14_future_schedule_no_commitment(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('future_schedule_no_commitment', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_15_background_work_ack(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('background_work_ack', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_16_image_observation_uncertain(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('image_observation_uncertain', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_17_group_broadcast_public_conclusion(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('group_broadcast_public_conclusion', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_18_accepted_format_preference(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('accepted_format_preference', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_19_third_party_status(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('third_party_status', ensure_live_dialog_llms, monkeypatch)


async def test_live_dialog_l3_20_privacy_boundary(
    ensure_live_dialog_llms: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case('privacy_boundary', ensure_live_dialog_llms, monkeypatch)
