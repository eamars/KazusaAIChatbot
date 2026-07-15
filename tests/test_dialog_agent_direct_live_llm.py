"""Direct real-LLM unit checks for dialog_agent with complete synthetic input.

These tests fabricate perfectly shaped L3-to-dialog state and call the actual
dialog_agent node with the configured DIALOG_GENERATOR_LLM route. They are
live LLM node tests: run one case at a time with -m live_llm -q -s, then inspect
the emitted trace JSON and the human-authored review artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from kazusa_ai_chatbot.config import (
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
_TRACE_SUITE = 'dialog_agent_direct_live_llm'
_CURRENT_USER_NAME = 'Jigsaw'
_CURRENT_USER_PLATFORM_ID = '411706805'
_CURRENT_USER_GLOBAL_ID = 'fa874545-02e6-4127-a24e-30819f941d83'
_BOT_PLATFORM_ID = '3768713357'
_BOT_GLOBAL_ID = '00000000-0000-4000-8000-000000000001'

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
    """Capture dialog LLM evidence while delegating to the configured route."""

    def __init__(self, wrapped_llm: Any, calls: list[dict[str, Any]]) -> None:
        self._wrapped_llm = wrapped_llm
        self._calls = calls

    async def ainvoke(self, messages: list[Any], *, config=None) -> Any:
        response = await self._wrapped_llm.ainvoke(messages, config=config)
        human_payload = json.loads(messages[1].content)
        usage = getattr(response, 'usage', {})
        self._calls.append({
            'stage': 'dialog_generator',
            'system_prompt_chars': len(messages[0].content),
            'human_payload': human_payload,
            'raw_response': getattr(response, 'content', ''),
            'usage': dict(usage) if isinstance(usage, dict) else usage,
        })
        return response

    def describe_backend(self, *, config):
        """Delegate backend diagnostics to the wrapped live LLM."""

        backend = self._wrapped_llm.describe_backend(config=config)
        return backend


async def _skip_if_dialog_llm_unavailable() -> None:
    """Skip the case when the configured dialog route is not reachable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{DIALOG_GENERATOR_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f'DIALOG_GENERATOR_LLM endpoint unavailable: '
            f'{DIALOG_GENERATOR_LLM_BASE_URL}; {exc}'
        )

    if response.status_code >= 500:
        pytest.skip(
            'DIALOG_GENERATOR_LLM endpoint returned server error '
            f'{response.status_code}: {DIALOG_GENERATOR_LLM_BASE_URL}'
        )


@pytest.fixture()
async def ensure_live_dialog_llm() -> None:
    """Ensure the live dialog generator route is reachable."""

    await _skip_if_dialog_llm_unavailable()


@pytest.fixture(autouse=True)
def _stub_dialog_event_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real prompt tests independent from event-log persistence."""

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
    """Load a realistic character profile for direct dialog tests."""

    loaded_profile = load_personality(_PERSONALITY_PATH)
    profile = {
        'name': loaded_profile['name'],
        'personality_brief': loaded_profile['personality_brief'],
        'linguistic_texture_profile': (
            loaded_profile['linguistic_texture_profile']
        ),
    }
    return profile


def _chat_history(*messages: tuple[str, str]) -> list[dict[str, Any]]:
    """Build compact prompt-safe chat history rows for synthetic cases."""

    history = [
        {
            'role': role,
            'platform_user_id': (
                _CURRENT_USER_PLATFORM_ID if role == 'user' else _BOT_PLATFORM_ID
            ),
            'global_user_id': (
                _CURRENT_USER_GLOBAL_ID if role == 'user' else _BOT_GLOBAL_ID
            ),
            'body_text': body_text,
            'content': body_text,
            'addressed_to_global_user_ids': (
                [_BOT_GLOBAL_ID] if role == 'user' else [_CURRENT_USER_GLOBAL_ID]
            ),
            'broadcast': role == 'user',
        }
        for role, body_text in messages
    ]
    return history


def _base_case(case_id: str) -> dict[str, Any]:
    """Return shared fields for one synthetic direct dialog case."""

    case = {
        'case_id': case_id,
        'input_kind': 'synthetic_perfect_l3_state',
        'user_name': _CURRENT_USER_NAME,
        'platform_user_id': _CURRENT_USER_PLATFORM_ID,
        'global_user_id': _CURRENT_USER_GLOBAL_ID,
        'relationship_state': 520,
        'relationship_insight': '熟悉但仍保持边界的群聊协作对象',
        'accepted_user_preferences': [],
        'forbidden_phrases': [],
        'chat_history': [],
        'expected_message_count': None,
        'must_include_all': [],
        'must_include_any': [],
        'must_not_include': [],
        'must_not_end_with_question': False,
        'must_include_exactly_once': [],
    }
    return case


def _state_from_case(case: dict[str, Any]) -> dict[str, Any]:
    """Build complete dialog_agent state from one synthetic L3 case."""

    state = {
        'character_profile': _character_profile(),
        'internal_monologue': case['internal_monologue'],
        'action_directives': {
            'linguistic_directives': {
                'rhetorical_strategy': case['rhetorical_strategy'],
                'linguistic_style': case['linguistic_style'],
                'accepted_user_preferences': case['accepted_user_preferences'],
                'content_plan': case['content_plan'],
                'forbidden_phrases': case['forbidden_phrases'],
            },
            'contextual_directives': case['contextual_directives'],
            'visual_directives': {},
        },
        'chat_history_wide': case['chat_history'],
        'chat_history_recent': case['chat_history'],
        'debug_modes': {'no_visual_directives': True},
        'should_respond': True,
        'platform_user_id': case['platform_user_id'],
        'platform_bot_id': _BOT_PLATFORM_ID,
        'global_user_id': case['global_user_id'],
        'user_name': case['user_name'],
        'user_profile': {
            'relationship_state': case['relationship_state'],
            'semantic_relationship_projection': case['relationship_insight'],
        },
        'dialog_usage_mode': 'live_visible_reply',
    }
    return state


def _visible_segments(final_dialog: Any) -> list[str]:
    """Return non-empty final_dialog strings for assessment."""

    if not isinstance(final_dialog, list):
        return []
    segments = [
        segment.strip()
        for segment in final_dialog
        if isinstance(segment, str) and segment.strip()
    ]
    return segments


def _joined_dialog(final_dialog: Any) -> str:
    """Join ordered outbound messages for human inspection."""

    joined = '\n'.join(_visible_segments(final_dialog))
    return joined


def _contains_one_from_each_group(
    text: str,
    groups: list[tuple[str, ...]],
) -> list[str]:
    """Return descriptions of required alternative groups not found in text."""

    missing_groups = []
    for group in groups:
        if not any(option in text for option in group):
            missing_groups.append(f'missing one of {group!r}')
    return missing_groups


def _assess_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Assess one live dialog output with structural and contract gates."""

    final_dialog = result.get('final_dialog')
    segments = _visible_segments(final_dialog)
    joined_dialog = _joined_dialog(final_dialog)
    failures: list[str] = []

    if not isinstance(final_dialog, list):
        failures.append('final_dialog is not a list')
    elif not final_dialog:
        failures.append('final_dialog is empty')
    elif not all(isinstance(segment, str) for segment in final_dialog):
        failures.append('final_dialog contains a non-string item')
    if not joined_dialog:
        failures.append('visible dialog is blank')

    expected_message_count = case['expected_message_count']
    if (
        expected_message_count is not None
        and len(segments) != expected_message_count
    ):
        failures.append(
            f'expected {expected_message_count} visible messages, '
            f'got {len(segments)}'
        )

    for required_text in case['must_include_all']:
        if required_text not in joined_dialog:
            failures.append(f'missing required text: {required_text}')

    failures.extend(
        _contains_one_from_each_group(joined_dialog, case['must_include_any'])
    )

    for forbidden_text in case['must_not_include']:
        if forbidden_text in joined_dialog:
            failures.append(f'contains forbidden text: {forbidden_text}')

    for exact_text in case['must_include_exactly_once']:
        actual_count = joined_dialog.count(exact_text)
        if actual_count != 1:
            failures.append(
                f'{exact_text!r} appeared {actual_count} times instead of once'
            )

    if (
        case['must_not_end_with_question']
        and joined_dialog.rstrip().endswith(('?', '？'))
    ):
        failures.append('ended with an unauthorized question')

    assessment = {
        'passed': not failures,
        'failures': failures,
        'visible_message_count': len(segments),
        'joined_dialog': joined_dialog,
    }
    return assessment


async def _run_case(
    case: dict[str, Any],
    ensure_live_dialog_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Run one direct dialog-agent live case and write trace evidence."""

    del ensure_live_dialog_llm
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        dialog_module,
        '_dialog_generator_llm',
        _CapturingLiveLLM(dialog_module._dialog_generator_llm, calls),
    )

    state = _state_from_case(case)
    result = await dialog_agent(state)
    assessment = _assess_case(case, result)
    trace_payload = {
        'case_id': case['case_id'],
        'component': 'dialog_agent',
        'test_type': 'real_llm_node_unit',
        'input_kind': case['input_kind'],
        'behavior_contract': case['behavior_contract'],
        'hard_gates': case['hard_gates'],
        'acceptable_variation': case['acceptable_variation'],
        'forbidden_failure_modes': case['forbidden_failure_modes'],
        'generator_model': DIALOG_GENERATOR_LLM_MODEL,
        'generator_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
        'generator_backend_descriptor': (
            dialog_module._dialog_generator_llm.describe_backend(
                config=dialog_module._dialog_generator_llm_config,
            )
        ),
        'input_state': state,
        'model_calls': calls,
        'result': result,
        'assessment': assessment,
    }
    trace_path = write_llm_trace(
        _TRACE_SUITE,
        case['case_id'],
        trace_payload,
    )

    print(f'trace_path={trace_path}')
    print(f'case_id={case["case_id"]}')
    print(f'dialog_text={assessment["joined_dialog"]}')

    assert assessment['passed'], (
        f'dialog live LLM contract failed; trace={trace_path}; '
        f'failures={assessment["failures"]!r}; '
        f'dialog={assessment["joined_dialog"]!r}'
    )
    return_value = {
        'trace_path': str(trace_path),
        'trace_payload': trace_payload,
    }
    return return_value


def _case_group_broadcast_public_conclusion() -> dict[str, Any]:
    """Build a group-broadcast case that should avoid addressing one user."""

    case = _base_case('group_broadcast_public_conclusion')
    case.update({
        'behavior_contract': (
            'Render a group-facing conclusion from complete L3 directives '
            'without turning it into a direct reply to the current user.'
        ),
        'hard_gates': [
            'final_dialog is a non-empty list of strings',
            'A 方案 and B 方案 remain visible',
            'dialog does not add @ tags or the current user name',
        ],
        'acceptable_variation': (
            'Natural wording and sentence order may vary as long as the group '
            'conclusion and pending-data condition remain clear.'
        ),
        'forbidden_failure_modes': [
            'directly naming Jigsaw',
            'adding an @ tag',
            'dropping either option label',
        ],
        'internal_monologue': '这是给群里的公共结论，不要只对当前用户黏过去。',
        'rhetorical_strategy': '给出群聊公共结论并收束。',
        'linguistic_style': '群聊可扫读，短句，不点名个人。',
        'content_plan': {
            'visible_goal': '给群里一个公共结论。',
            'semantic_content': (
                '公共结论是先停在 A 方案；B 方案等新数据回来再比较。'
            ),
            'voice': '冷静、简洁，像群里同步结论。',
            'rendering': '1 条普通文字消息，直接结论，不点名当前用户。',
        },
        'contextual_directives': {
            'social_distance': '群体讨论距离',
            'emotional_intensity': '低',
            'vibe_check': '多人讨论收束',
            'relational_dynamic': '当前角色对群里同步一个结论',
        },
        'chat_history': _chat_history(
            ('user', '那我们先按哪个方案走？'),
        ),
        'expected_message_count': 1,
        'must_include_all': ['A 方案', 'B 方案'],
        'must_include_any': [('新数据', '数据回来', '数据回来了')],
        'must_not_include': ['@', _CURRENT_USER_NAME],
        'must_not_end_with_question': True,
    })
    return case


def _case_named_participant_inline_tag() -> dict[str, Any]:
    """Build a direct named-participant tag case."""

    case = _base_case('named_participant_inline_tag')
    case.update({
        'behavior_contract': (
            'Render a visible inline @display_name token only because the '
            'complete content plan explicitly requests it.'
        ),
        'hard_gates': [
            'final_dialog is a non-empty list of strings',
            '@Moca appears exactly once',
            'native platform mention syntax is absent',
        ],
        'acceptable_variation': (
            'The surrounding sentence may vary, but @Moca must stay literal.'
        ),
        'forbidden_failure_modes': [
            'inventing native Discord or QQ syntax',
            'adding a retired mention field',
            'omitting or duplicating @Moca',
        ],
        'internal_monologue': 'Moca 是计划里明确点名的参与者，这里只写可见 @Moca。',
        'rhetorical_strategy': '点名 Moca，然后给简短任务分配。',
        'linguistic_style': '群聊协作，短句。',
        'content_plan': {
            'visible_goal': '点名 Moca 继续记分。',
            'semantic_content': 'Moca 继续记分；Jigsaw 等下一题。使用精确可见标签 @Moca。',
            'voice': '轻松但清楚。',
            'rendering': '1 条普通文字消息；包含 @Moca 恰好一次。',
        },
        'contextual_directives': {
            'social_distance': '熟悉群友协作',
            'emotional_intensity': '低',
            'vibe_check': '轻量任务分配',
            'relational_dynamic': '角色在群里点名一个已出现的参与者',
        },
        'chat_history': _chat_history(
            ('user', 'Moca 说她可以继续记分。'),
        ),
        'expected_message_count': 1,
        'must_include_all': ['@Moca'],
        'must_include_exactly_once': ['@Moca'],
        'must_not_include': ['<@', '[CQ:', 'CQ:at', 'qq=', 'Discord', 'QQ'],
        'must_not_end_with_question': True,
    })
    return case


def _case_two_message_followup_boundary() -> dict[str, Any]:
    """Build a two-message follow-up case from explicit rendering intent."""

    case = _base_case('two_message_followup_boundary')
    case.update({
        'behavior_contract': (
            'Preserve L3 message-sequence intent: one cognition can render '
            'two ordered outbound messages without compressing the meaning.'
        ),
        'hard_gates': [
            'final_dialog has exactly two visible strings',
            'backup and tomorrow-boundary content remain visible',
        ],
        'acceptable_variation': (
            'The first line can be a short calming reaction; the second line '
            'can word the boundary naturally.'
        ),
        'forbidden_failure_modes': [
            'compressing both messages into one string',
            'dropping backup',
            'turning the boundary into a new question',
        ],
        'internal_monologue': '先稳住对方，再补一句清楚边界，不要压成一大段。',
        'rhetorical_strategy': '第一条短反应接住焦虑，第二条给明确边界和下一步。',
        'linguistic_style': '在线聊天语气，短消息，像连续发两条。',
        'content_plan': {
            'visible_goal': '接住用户焦虑，同时说明今晚只处理最关键的一项。',
            'semantic_content': (
                '先告诉用户不用一下子全做完；今晚只处理备份这一项。'
                '配置调整等明天精神清楚时再看，避免越改越乱。'
            ),
            'voice': '克制但有一点关心。',
            'rendering': (
                '2 条连续发送的普通文字消息；'
                '第一条短反应，第二条补充边界和下一步。'
            ),
        },
        'contextual_directives': {
            'social_distance': '较熟的技术协作距离',
            'emotional_intensity': '中低，对方有一点急',
            'vibe_check': '深夜排障前的降速',
            'relational_dynamic': '用户想把所有风险操作今晚一次做完',
        },
        'chat_history': _chat_history(
            ('user', '我今晚是不是把备份和配置都一口气改完比较好？'),
        ),
        'expected_message_count': 2,
        'must_include_all': ['备份'],
        'must_include_any': [('明天', '明早'), ('配置', '调整')],
    })
    return case


def _case_technical_numeric_comparison() -> dict[str, Any]:
    """Build a technical comparison case requiring source literals."""

    case = _base_case('technical_numeric_comparison')
    case.update({
        'behavior_contract': (
            'Render a technical answer from complete L3 content without '
            'dropping source-owned model names, units, or conclusion scope.'
        ),
        'hard_gates': [
            'both product names remain visible',
            'required numeric units remain visible',
            'workstation and large-scale conclusion remain visible',
            'dialog does not add @ tags or upgrade the comparison strength',
        ],
        'acceptable_variation': (
            'The answer may be prose or compact lines; exact ordering is not '
            'important when required facts are preserved.'
        ),
        'forbidden_failure_modes': [
            'hallucinating a stronger conclusion',
            'automatically tagging the current user',
            'using markdown table rows not requested by L3',
            'dropping numeric units',
        ],
        'internal_monologue': '技术对比要把数字说全，语气可以轻一点但不能省事实。',
        'rhetorical_strategy': '正面对比两张卡，先结论再数字依据。',
        'linguistic_style': '群聊技术回答，信息密度优先，不点名当前用户。',
        'forbidden_phrases': [
            '@Jigsaw',
            '完全不是一个量级',
            '不是一个量级',
            '碾压',
            '吊打',
        ],
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
            'voice': '技术准确优先，收束句只保留更适合的适用范围。',
            'rendering': '信息密度优先，普通聊天行，不使用表格；不点名当前用户。',
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
        'must_include_all': [
            'GB300',
            'Pro6000',
            '2250 TFLOPS',
            '4500',
            '288GB',
            '12000 GB/s',
            '1400W',
            '125 TFLOPS',
            '2000',
            '96GB',
            '1792 GB/s',
            '400W',
            '工作站',
        ],
        'must_include_any': [('训练', '推理')],
        'must_not_include': [
            '@',
            '|',
            '完全不是一个量级',
            '不是一个量级',
            '完全碾压',
            '不可能',
            '吊打',
        ],
    })
    return case


def _case_python_code_block_preserved() -> dict[str, Any]:
    """Build a fixed-format code block case."""

    case = _base_case('python_code_block_preserved')
    case.update({
        'behavior_contract': (
            'Preserve upstream fixed-format code as a complete message string '
            'while keeping character voice outside the fenced block.'
        ),
        'hard_gates': [
            'fenced python block exists',
            'required function lines remain present',
            'roleplay particles do not enter code',
        ],
        'acceptable_variation': (
            'The model may add a short intro or outro outside the code block.'
        ),
        'forbidden_failure_modes': [
            'rewriting the algorithm',
            'changing Python fence to another language',
            'inserting character voice inside code',
        ],
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
            ('user', '@杏山千纱 帮我写一个解数独的python，输入格式你定。'),
        ),
        'must_include_all': [
            '```python',
            'def solve_sudoku(board):',
            '    if not board:',
            '        return False',
            'def find_empty(board):',
            '            if board[row][col] == 0:',
            '                return row, col',
            '    return None',
        ],
        'must_not_include': ['```json', '哼\n', '啦\n', '哦\n'],
    })
    return case


def _case_unknown_referent_clarification() -> dict[str, Any]:
    """Build a clarification case for unresolved referents."""

    case = _base_case('unknown_referent_clarification')
    case.update({
        'behavior_contract': (
            'Ask a minimal clarification question when L3 says the referent '
            'is unresolved; do not invent the missing object.'
        ),
        'hard_gates': [
            'final_dialog is a non-empty list of strings',
            'dialog asks what specific object the user means',
            'dialog does not invent candidate objects',
        ],
        'acceptable_variation': (
            'Any short natural clarification question is acceptable.'
        ),
        'forbidden_failure_modes': [
            'guessing that the object is a cable',
            'guessing that the object is a plan',
            'answering without clarification',
        ],
        'internal_monologue': '对方说这个那个，但对象不清楚，应该先澄清。',
        'rhetorical_strategy': '直接指出对象不清楚，问一个最小澄清问题。',
        'linguistic_style': '短句、自然、不要猜。',
        'content_plan': {
            'visible_goal': '澄清用户说的对象。',
            'semantic_content': (
                '用户说的这个对象不明确，需要请用户说明具体指的是哪一件事或哪一个东西。'
            ),
            'voice': '轻微困惑但不冷。',
            'rendering': '一个简短问题。',
        },
        'contextual_directives': {
            'social_distance': '普通对话',
            'emotional_intensity': '低',
            'vibe_check': '指代不明',
            'relational_dynamic': '用户的问题缺少可解析对象',
        },
        'chat_history': _chat_history(
            ('user', '那这个怎么弄？'),
        ),
        'expected_message_count': 1,
        'must_include_any': [('具体', '哪个', '哪件', '指')],
        'must_not_include': ['线', '备份', 'A 方案', 'B 方案'],
    })
    return case


def _case_privacy_boundary_code() -> dict[str, Any]:
    """Build a privacy boundary case for sensitive codes."""

    case = _base_case('privacy_boundary_code')
    case.update({
        'behavior_contract': (
            'Render an upstream privacy boundary clearly without exposing '
            'extra policy text or creating an unsafe workaround.'
        ),
        'hard_gates': [
            '验证码 remains visible',
            'a clear do-not-send boundary remains visible',
            'a remedial next step remains visible',
        ],
        'acceptable_variation': (
            'The next step can be phrased as撤回, 更换, or检查账号.'
        ),
        'forbidden_failure_modes': [
            'telling the user to paste the code',
            'claiming the character can verify the account',
            'turning the boundary into a vague joke only',
        ],
        'internal_monologue': '验证码不能发群里，要把边界说清楚。',
        'rhetorical_strategy': '直接拦住验证码外发，再给补救动作。',
        'linguistic_style': '明确、短句、不恐吓。',
        'content_plan': {
            'visible_goal': '阻止用户把验证码发到群里。',
            'semantic_content': (
                '不要把验证码发到群里；如果已经发了，'
                '先撤回或更换，再检查相关账号。'
            ),
            'voice': '认真一点，少开玩笑。',
            'rendering': '1 条普通文字消息，清楚边界和下一步。',
        },
        'contextual_directives': {
            'social_distance': '普通但需要明确提醒',
            'emotional_intensity': '中低',
            'vibe_check': '隐私风险提醒',
            'relational_dynamic': '用户可能要发送敏感信息',
        },
        'chat_history': _chat_history(
            ('user', '验证码我直接发群里让你看可以吗？'),
        ),
        'expected_message_count': 1,
        'must_include_all': ['验证码'],
        'must_include_any': [
            ('不要', '别', '不能', '不可'),
            ('撤回', '更换', '检查'),
        ],
        'must_not_include': ['发出来', '贴出来', '我来验证', '我帮你登录'],
        'must_not_end_with_question': True,
    })
    return case


async def test_dialog_agent_direct_live_group_broadcast_public_conclusion(
    ensure_live_dialog_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real LLM unit: group conclusion should not address one user."""

    await _run_case(
        _case_group_broadcast_public_conclusion(),
        ensure_live_dialog_llm,
        monkeypatch,
    )


async def test_dialog_agent_direct_live_named_participant_inline_tag(
    ensure_live_dialog_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real LLM unit: explicit @display_name survives without platform syntax."""

    await _run_case(
        _case_named_participant_inline_tag(),
        ensure_live_dialog_llm,
        monkeypatch,
    )


async def test_dialog_agent_direct_live_two_message_followup_boundary(
    ensure_live_dialog_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real LLM unit: one cognition can render two ordered messages."""

    await _run_case(
        _case_two_message_followup_boundary(),
        ensure_live_dialog_llm,
        monkeypatch,
    )


async def test_dialog_agent_direct_live_technical_numeric_comparison(
    ensure_live_dialog_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real LLM unit: technical facts and conclusion scope are preserved."""

    await _run_case(
        _case_technical_numeric_comparison(),
        ensure_live_dialog_llm,
        monkeypatch,
    )


async def test_dialog_agent_direct_live_python_code_block_preserved(
    ensure_live_dialog_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real LLM unit: fixed-format Python block remains intact."""

    await _run_case(
        _case_python_code_block_preserved(),
        ensure_live_dialog_llm,
        monkeypatch,
    )


async def test_dialog_agent_direct_live_unknown_referent_clarification(
    ensure_live_dialog_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real LLM unit: unresolved referent becomes a minimal clarification."""

    await _run_case(
        _case_unknown_referent_clarification(),
        ensure_live_dialog_llm,
        monkeypatch,
    )


async def test_dialog_agent_direct_live_privacy_boundary_code(
    ensure_live_dialog_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real LLM unit: privacy boundary renders clearly and safely."""

    await _run_case(
        _case_privacy_boundary_code(),
        ensure_live_dialog_llm,
        monkeypatch,
    )
