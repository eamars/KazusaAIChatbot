"""Live LLM checks for dialog one-bubble layout behavior."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from unittest.mock import AsyncMock

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    DIALOG_EVALUATOR_LLM_BASE_URL,
    DIALOG_EVALUATOR_LLM_MODEL,
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.utils import load_personality
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_PERSONALITY_PATH = _ROOT / 'personalities' / 'kazusa.json'
_SUDOKU_CODE_BLOCK = """\
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
"""


class _CapturingLiveLLM:
    """Capture dialog LLM calls while delegating to the configured model.

    Args:
        stage_name: Dialog substage being captured.
        wrapped_llm: Existing LangChain-compatible LLM instance.
        calls: Mutable call log shared by the test case.
    """

    def __init__(self, stage_name: str, wrapped_llm, calls: list[dict]) -> None:
        self._stage_name = stage_name
        self._wrapped_llm = wrapped_llm
        self._calls = calls

    async def ainvoke(self, messages):
        response = await self._wrapped_llm.ainvoke(messages)
        human_payload = json.loads(messages[1].content)
        extra_messages = [
            {
                'message_type': type(message).__name__,
                'name': getattr(message, 'name', None),
                'content': getattr(message, 'content', ''),
            }
            for message in messages[2:]
        ]
        self._calls.append({
            'stage': self._stage_name,
            'system_prompt_chars': len(messages[0].content),
            'human_payload': human_payload,
            'extra_messages': extra_messages,
            'raw_response': response.content,
        })
        return response


async def _skip_if_endpoint_unavailable(name: str, base_url: str) -> None:
    """Skip the live case when an OpenAI-compatible endpoint is unreachable.

    Args:
        name: Endpoint label used in the skip message.
        base_url: OpenAI-compatible base URL to probe.

    Returns:
        None.
    """

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
async def ensure_live_dialog_llms() -> None:
    """Ensure both live dialog routes are available before a case runs."""

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


def _base_dialog_state(case: dict) -> dict:
    """Build a full dialog-agent state from simulated L3 directives.

    Args:
        case: Test case containing prompt-safe L3-style directives.

    Returns:
        Dialog-agent input state with real Kazusa personality data.
    """

    state = {
        'character_profile': load_personality(_PERSONALITY_PATH),
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
        'chat_history_wide': case['chat_history'],
        'chat_history_recent': case['chat_history'],
        'debug_modes': {},
        'should_respond': True,
        'platform_user_id': case['platform_user_id'],
        'platform_bot_id': '3768713357',
        'global_user_id': case['global_user_id'],
        'user_name': case['user_name'],
        'user_profile': {
            'affinity': case['affinity'],
            'last_relationship_insight': case['relationship_insight'],
        },
    }
    return state


async def _run_live_dialog_case(case: dict, monkeypatch) -> dict:
    """Run one case through dialog_agent and write an inspectable trace.

    Args:
        case: L3-shaped test case payload.
        monkeypatch: Pytest monkeypatch fixture used for LLM capture.

    Returns:
        Trace payload containing result, joined dialog, and trace path.
    """

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

    state = _base_dialog_state(case)
    result = await dialog_agent(state)
    final_dialog = result.get('final_dialog')
    structural_validation = {
        'final_dialog_is_list': isinstance(final_dialog, list),
        'final_dialog_all_strings': (
            isinstance(final_dialog, list)
            and all(isinstance(segment, str) for segment in final_dialog)
        ),
        'final_dialog_non_empty': bool(final_dialog),
    }
    joined_dialog = ''
    if structural_validation['final_dialog_all_strings']:
        joined_dialog = '\n'.join(final_dialog)

    trace_payload = {
        'case_id': case['case_id'],
        'generator_model': DIALOG_GENERATOR_LLM_MODEL,
        'generator_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
        'evaluator_model': DIALOG_EVALUATOR_LLM_MODEL,
        'evaluator_base_url': DIALOG_EVALUATOR_LLM_BASE_URL,
        'simulated_l3_action_directives': state['action_directives'],
        'user_name': state['user_name'],
        'contextual_directives': (
            state['action_directives']['contextual_directives']
        ),
        'model_calls': calls,
        'raw_result': result,
        'final_dialog': final_dialog,
        'joined_dialog': joined_dialog,
        'mention_target_user': result.get('mention_target_user'),
        'structural_validation': structural_validation,
        'manual_inspection_notes': '',
    }
    trace_path = write_llm_trace(
        'dialog_one_bubble_layout_live_llm',
        case['case_id'],
        trace_payload,
    )
    trace_payload['trace_path'] = str(trace_path)

    assert structural_validation['final_dialog_is_list'], trace_path
    assert structural_validation['final_dialog_all_strings'], trace_path
    assert structural_validation['final_dialog_non_empty'], trace_path
    assert joined_dialog.strip(), trace_path

    return trace_payload


def _chat_history(*messages: tuple[str, str]) -> list[dict]:
    """Build compact prompt-safe chat history rows for dialog live cases."""

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


def _base_case(case_id: str) -> dict:
    """Return common dialog case fields shared by all live scenarios."""

    case = {
        'case_id': case_id,
        'platform_user_id': '411706805',
        'global_user_id': 'fa874545-02e6-4127-a24e-30819f941d83',
        'user_name': 'Jigsaw',
        'affinity': 500,
        'relationship_insight': '熟悉但仍会互相调侃的群友',
    }
    return case


def _group_casual_case() -> dict:
    """Build the group casual direct-reply scenario."""

    case = _base_case('group_casual_reply')
    case.update({
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
        'chat_history': _chat_history(
            ('user', '@杏山千纱 这些线怎么分比较方便？'),
        ),
    })
    return case


def _private_soft_case() -> dict:
    """Build the private soft-reply scenario."""

    case = _base_case('private_soft_reply')
    case.update({
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
        'chat_history': _chat_history(
            ('user', '我这样安排今晚是不是有点乱？'),
        ),
    })
    return case


def _technical_comparison_case() -> dict:
    """Build the group technical comparison scenario."""

    case = _base_case('group_technical_comparison')
    case.update({
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
        'chat_history': _chat_history(
            ('user', '@杏山千纱 GB300 和 Pro6000 性能怎么比？'),
        ),
    })
    return case


def _python_code_case() -> dict:
    """Build the Jigsaw-style Python code-block scenario."""

    case = _base_case('python_code_block_preserved')
    case.update({
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
    })
    return case


def _json_example_case() -> dict:
    """Build the fixed-format JSON example scenario."""

    case = _base_case('json_example_preserved')
    case.update({
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
            ('assistant', '格式就这种最优雅的结构，直接复制拿去用：\n{"puzzle": [[5,3,0]]}'),
            ('user', '@杏山千纱 你这才三行，给我完整的'),
        ),
    })
    return case


def _fenced_block(joined_dialog: str, language: str) -> str:
    """Extract the first fenced code block for a given language label."""

    pattern = rf'```{language}\s*(.*?)```'
    match = re.search(pattern, joined_dialog, flags=re.DOTALL)
    assert match is not None, joined_dialog
    block = match.group(1)
    return block


async def test_live_dialog_one_bubble_group_casual_reply(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """Group casual reply should stay direct and one-bubble readable."""

    del ensure_live_dialog_llms
    trace_payload = await _run_live_dialog_case(_group_casual_case(), monkeypatch)
    joined_dialog = trace_payload['joined_dialog']

    assert '充电' in joined_dialog, trace_payload['trace_path']
    assert '视频输出' in joined_dialog, trace_payload['trace_path']
    assert (
        '待确认' in joined_dialog
        or '等确认' in joined_dialog
        or '再确认' in joined_dialog
        or '没确定' in joined_dialog
        or '未确定' in joined_dialog
    ), trace_payload['trace_path']
    stripped_dialog = joined_dialog.rstrip()
    assert not stripped_dialog.endswith(('?', '？')), trace_payload['trace_path']


async def test_live_dialog_one_bubble_private_soft_reply(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """Private soft reply should preserve the concrete plan conclusion."""

    del ensure_live_dialog_llms
    trace_payload = await _run_live_dialog_case(_private_soft_case(), monkeypatch)
    joined_dialog = trace_payload['joined_dialog']

    assert '今晚' in joined_dialog, trace_payload['trace_path']
    assert (
        '明早' in joined_dialog
        or '明天早' in joined_dialog
    ), trace_payload['trace_path']
    assert (
        '复查' in joined_dialog
        or '重新看' in joined_dialog
    ), trace_payload['trace_path']


async def test_live_dialog_one_bubble_group_technical_comparison(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """Technical comparison should preserve required numeric facts."""

    del ensure_live_dialog_llms
    trace_payload = await _run_live_dialog_case(
        _technical_comparison_case(),
        monkeypatch,
    )
    normalized_dialog = trace_payload['joined_dialog'].replace(',', '')

    for required_text in (
        'GB300',
        'Pro6000',
        '2250 TFLOPS',
        '4500 TFLOPS',
        '288GB',
        '12000 GB/s',
        '1400W',
        '125 TFLOPS',
        '2000 TFLOPS',
        '96GB',
        '1792 GB/s',
        '400W',
        '工作站',
    ):
        assert required_text in normalized_dialog, trace_payload['trace_path']

    for dialog_line in trace_payload['joined_dialog'].splitlines():
        assert not dialog_line.lstrip().startswith('|'), trace_payload['trace_path']


async def test_live_dialog_one_bubble_python_code_block_preserved(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """Python delivery should keep code inside a fenced block."""

    del ensure_live_dialog_llms
    trace_payload = await _run_live_dialog_case(_python_code_case(), monkeypatch)
    joined_dialog = trace_payload['joined_dialog']
    code_block = _fenced_block(joined_dialog, 'python')

    for required_text in (
        'def solve_sudoku(board):',
        '    if not board:',
        '        return False',
        '    return find_empty(board) is None',
        '\n\ndef find_empty(board):',
        '            if board[row][col] == 0:',
        '                return row, col',
        '    return None',
    ):
        assert required_text in code_block, trace_payload['trace_path']

    for unexpected_text in ('is_valid', 'solve(board)', '```json'):
        assert unexpected_text not in code_block, trace_payload['trace_path']

    for voice_text in ('哼', '嘛', '啦', '哦', '诶'):
        assert voice_text not in code_block, trace_payload['trace_path']


async def test_live_dialog_one_bubble_json_example_preserved(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """JSON input example should stay complete and parseable."""

    del ensure_live_dialog_llms
    trace_payload = await _run_live_dialog_case(_json_example_case(), monkeypatch)
    joined_dialog = trace_payload['joined_dialog']
    json_block = _fenced_block(joined_dialog, 'json')
    parsed_block = json.loads(json_block)
    puzzle = parsed_block['puzzle']

    assert len(puzzle) == 9, trace_payload['trace_path']
    assert all(len(row) == 9 for row in puzzle), trace_payload['trace_path']
    assert all(
        all(isinstance(value, int) for value in row)
        for row in puzzle
    ), trace_payload['trace_path']
    assert '不完整' not in json_block, trace_payload['trace_path']
    assert '抱歉' not in json_block, trace_payload['trace_path']
