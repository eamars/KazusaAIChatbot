"""Real-model reproduction for an authoritative settled-relevance failure."""

from __future__ import annotations

import json
import sys
from time import perf_counter
from typing import Any
from unittest.mock import patch

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    build_character_production_state,
)
import kazusa_ai_chatbot.relevance.persona_relevance_agent as settled_module
from kazusa_ai_chatbot.relevance.persona_relevance_agent import (
    SettledRelevanceContractError,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace
from tests.test_relevance_turn_settlement_live_llm import (
    ensure_relevance_live_llms,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

_TRACE_SUITE = 'settled_relevance_captured_failure_live_llm'
_CHARACTER_GLOBAL_USER_ID = '00000000-0000-4000-8000-000000000001'
_CHARACTER_PLATFORM_USER_ID = '3768713357'
_CURRENT_GLOBAL_USER_ID = '4759394b-a4d2-4634-9d12-b6423a92a248'
_CURRENT_PLATFORM_USER_ID = '673225019'
_OTHER_GLOBAL_USER_ID = '07cec0ae-f56d-47be-b87e-608f0620de68'
_OTHER_PLATFORM_USER_ID = '2910137276'
_CHARACTER_NAME = '一之濑明日奈'
_CAPTURED_MESSAGE = '@一之濑明日奈 我说了有奖励么？'
_PRODUCTION_PROMPT_CHARS = 9311
_PRODUCTION_OUTPUT_CHARS = [428, 485]
_PRODUCTION_TRACE_ID = 'llmtrace_83574980c2b84229a7c1b64c365f80a6'
_PRODUCTION_OUTPUT_HASHES = [
    '521f6097fae6be56eec3df8e707f9234afea430c03a57e7690dae44670352b6a',
    '8b8e7ccf9acc762cef2942c64461d9c6a7018f48130403d04d65cba2ad67fd55',
]


class _RecordingLLM:
    """Delegate to the configured relevance model and retain every call."""

    def __init__(self, delegate: Any) -> None:
        """Bind the production LLM delegate.

        Args:
            delegate: Configured relevance-model interface used by production.
        """

        self._delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke the model and record its exact prompt and normalized output.

        Args:
            messages: System and human messages supplied by the relevance node.
            args: Additional positional arguments forwarded to the delegate.
            kwargs: Additional keyword arguments forwarded to the delegate.

        Returns:
            The delegate's normalized LLM response.
        """

        started_at = perf_counter()
        response = await self._delegate.ainvoke(
            messages,
            *args,
            **kwargs,
        )
        duration_ms = int((perf_counter() - started_at) * 1000)
        self.calls.append({
            'messages': [
                {
                    'type': message.__class__.__name__,
                    'content': str(message.content),
                }
                for message in messages
            ],
            'raw_response_text': str(response.content),
            'duration_ms': duration_ms,
        })
        return response


def _history_row(
    *,
    role: str,
    platform_user_id: str,
    global_user_id: str,
    body_text: str,
    addressed_to: list[str],
    turn_relation: str,
    reply_to_platform_user_id: str = '',
) -> dict[str, Any]:
    """Build one production-shaped history row for settlement projection.

    Args:
        role: Conversation author role.
        platform_user_id: Platform identity used for relation projection.
        global_user_id: Stable identity used for relation projection.
        body_text: Prompt-facing authored text.
        addressed_to: Typed addressee identities.
        turn_relation: Position relative to the active assembled turn.
        reply_to_platform_user_id: Optional typed reply target.

    Returns:
        A history row accepted by the settled-relevance projector.
    """

    reply_context: dict[str, str] = {}
    if reply_to_platform_user_id:
        reply_context = {
            'reply_to_platform_user_id': reply_to_platform_user_id,
        }
    row = {
        'role': role,
        'platform_user_id': platform_user_id,
        'global_user_id': global_user_id,
        'body_text': body_text,
        'addressed_to_global_user_ids': addressed_to,
        'broadcast': False,
        'reply_context': reply_context,
        'turn_temporal_relation': turn_relation,
    }
    return row


def _captured_history(after_message: str) -> list[dict[str, Any]]:
    """Build the ten external rows visible to the production failure.

    Args:
        after_message: Unrelated participant text that arrived after the
            active turn and enabled the ``already_resolved`` disposition.

    Returns:
        Chronological prompt-facing history with stable typed identities.
    """

    character_to_current = [_CURRENT_GLOBAL_USER_ID]
    current_to_character = [_CHARACTER_GLOBAL_USER_ID]
    before = 'before_active_turn'
    history = [
        _history_row(
            role='assistant',
            platform_user_id=_CHARACTER_PLATFORM_USER_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
            body_text='这样啊……心虚成这样，肯定是在脑补什么糟糕的事情吧？',
            addressed_to=character_to_current,
            turn_relation=before,
        ),
        _history_row(
            role='assistant',
            platform_user_id=_CHARACTER_PLATFORM_USER_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
            body_text='快点交代清楚！到底想到了哪里？一个字都不能漏掉哦。',
            addressed_to=character_to_current,
            turn_relation=before,
        ),
        _history_row(
            role='user',
            platform_user_id=_CURRENT_PLATFORM_USER_ID,
            global_user_id=_CURRENT_GLOBAL_USER_ID,
            body_text='@一之濑明日奈 在群聊说这些好羞耻哦，大家还在看着呢',
            addressed_to=current_to_character,
            turn_relation=before,
            reply_to_platform_user_id=_CHARACTER_PLATFORM_USER_ID,
        ),
        _history_row(
            role='user',
            platform_user_id=_OTHER_PLATFORM_USER_ID,
            global_user_id=_OTHER_GLOBAL_USER_ID,
            body_text='',
            addressed_to=[],
            turn_relation=before,
        ),
        _history_row(
            role='user',
            platform_user_id=_OTHER_PLATFORM_USER_ID,
            global_user_id=_OTHER_GLOBAL_USER_ID,
            body_text='',
            addressed_to=[],
            turn_relation=before,
        ),
        _history_row(
            role='assistant',
            platform_user_id=_CHARACTER_PLATFORM_USER_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
            body_text='哎呀，居然在这种时候跟我谈羞耻心？这也太可爱了吧。',
            addressed_to=character_to_current,
            turn_relation=before,
        ),
        _history_row(
            role='assistant',
            platform_user_id=_CHARACTER_PLATFORM_USER_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
            body_text=(
                '不过你看，大家都看着呢。既然你现在这么心虚，'
                '那干脆就趁这个机会坦白交代清楚吧。'
            ),
            addressed_to=character_to_current,
            turn_relation=before,
        ),
        _history_row(
            role='assistant',
            platform_user_id=_CHARACTER_PLATFORM_USER_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
            body_text='这样反而比较体面哦。',
            addressed_to=character_to_current,
            turn_relation=before,
        ),
        _history_row(
            role='assistant',
            platform_user_id=_CHARACTER_PLATFORM_USER_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
            body_text='所以，刚才在脑补什么呢？快点告诉我呀。',
            addressed_to=character_to_current,
            turn_relation=before,
        ),
        _history_row(
            role='user',
            platform_user_id=_OTHER_PLATFORM_USER_ID,
            global_user_id=_OTHER_GLOBAL_USER_ID,
            body_text=after_message,
            addressed_to=[],
            turn_relation='after_active_turn',
        ),
    ]
    return history


def _captured_state(after_message: str) -> dict[str, Any]:
    """Build the captured semantic state at the settled-relevance boundary.

    Args:
        after_message: Unrelated participant message visible after the turn.

    Returns:
        A production-shaped settled-relevance state.
    """

    cognition_state = build_character_production_state(
        updated_at='2026-07-30T04:19:28.745113Z',
    )
    state = {
        'conversation_scope': 'group',
        'active_character_name': _CHARACTER_NAME,
        'assembled_fragments': [{
            'body_text': _CAPTURED_MESSAGE,
            'semantic_target_labels': ['character'],
            'reply_target_label': 'character',
            'media_labels': [],
        }],
        'media_descriptions': [],
        'additional_media_present': False,
        'fresh_history': _captured_history(after_message),
        'scene_context': 'QQ group 638473184',
        'relationship_context': 'direct participant',
        'character_mood': '',
        'group_attention': 'high_noise',
        'bot_continuity': '所以，刚才在脑补什么呢？快点告诉我呀。',
        'character_cognition_state': cognition_state,
        'current_author_global_user_id': _CURRENT_GLOBAL_USER_ID,
        'current_author_platform_user_id': _CURRENT_PLATFORM_USER_ID,
        'character_global_user_id': _CHARACTER_GLOBAL_USER_ID,
        'platform_bot_id': _CHARACTER_PLATFORM_USER_ID,
        'observation_status': 'observation_complete',
    }
    return state


def _inspect_model_call(
    call: dict[str, Any],
    *,
    model_payload: dict[str, Any],
    available_dispositions: list[str],
) -> dict[str, Any]:
    """Apply production parsing and validation to one recorded model response.

    Args:
        call: Recorded prompt, response text, and duration.
        model_payload: Initial settled evidence used by both attempts.
        available_dispositions: Evidence-derived authoritative action space.

    Returns:
        Structured parsing and contract-validation evidence.
    """

    raw_response_text = call['raw_response_text']
    parsed_output: object = None
    parse_error = ''
    try:
        parsed_output = parse_llm_json_output(
            raw_response_text,
            deterministic_only=True,
        )
    except ValueError as exc:
        parse_error = str(exc)

    validation_reason = ''
    decision: dict[str, Any] | None = None
    assessment: dict[str, Any] | None = None
    if not parse_error:
        try:
            (
                decision,
                assessment,
                _,
            ) = settled_module._parse_authoritative_settled_response(
                raw_response_text,
                available_dispositions=available_dispositions,
                model_payload=model_payload,
            )
        except SettledRelevanceContractError as exc:
            validation_reason = exc.validation_reason

    inspected = {
        **call,
        'output_chars': len(raw_response_text),
        'parsed_output': parsed_output,
        'parse_error': parse_error,
        'validation_reason': validation_reason,
        'validated_decision': decision,
        'validated_assessment': assessment,
    }
    return inspected


async def _run_captured_probe(
    *,
    case_id: str,
    after_message: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Run one captured incident probe and persist evidence before asserting.

    Args:
        case_id: Stable artifact identifier.
        after_message: External after-turn message used in the live case.

    Returns:
        Public decision when successful and final typed error metadata when
        the bounded authoritative repair fails.
    """

    state = _captured_state(after_message)
    initial_messages = settled_module.build_settled_relevance_messages(
        state,
        observation_status='observation_complete',
    )
    model_payload = json.loads(str(initial_messages[1].content))
    available_dispositions = (
        settled_module._available_authoritative_dispositions(
            model_payload,
            'observation_complete',
        )
    )
    assert available_dispositions == ['proceed', 'already_resolved']

    recording_llm = _RecordingLLM(settled_module._relevance_agent_llm)
    result: dict[str, Any] | None = None
    final_error: dict[str, Any] | None = None
    with patch.object(
        settled_module,
        '_relevance_agent_llm',
        recording_llm,
    ):
        try:
            result = await settled_module.relevance_agent(state)
        except SettledRelevanceContractError as exc:
            final_error = {
                'error_class': exc.__class__.__name__,
                'message': str(exc),
                'validation_reason': exc.validation_reason,
                'attempt_count': exc.attempt_count,
                'stage': exc.stage,
                'error_code': exc.error_code,
            }

    inspected_calls = [
        _inspect_model_call(
            call,
            model_payload=model_payload,
            available_dispositions=available_dispositions,
        )
        for call in recording_llm.calls
    ]
    rendered_initial = ''.join(
        str(message.content)
        for message in initial_messages
    )
    write_llm_trace(
        _TRACE_SUITE,
        case_id,
        {
            'input_kind': 'captured_failure',
            'production_trace_reference': {
                'trace_id': _PRODUCTION_TRACE_ID,
                'prompt_chars': _PRODUCTION_PROMPT_CHARS,
                'output_chars': _PRODUCTION_OUTPUT_CHARS,
                'output_sha256': _PRODUCTION_OUTPUT_HASHES,
            },
            'replay_state': state,
            'initial_rendered_input_chars': len(rendered_initial),
            'initial_messages': [
                {
                    'type': message.__class__.__name__,
                    'content': str(message.content),
                }
                for message in initial_messages
            ],
            'available_dispositions': available_dispositions,
            'model_calls': inspected_calls,
            'public_result': result,
            'final_error': final_error,
            'route': 'RELEVANCE_AGENT_LLM',
            'model': settled_module.RELEVANCE_AGENT_LLM_MODEL,
        },
    )
    return_value = (result, final_error)
    return return_value


async def test_live_captured_qq_authoritative_repair_is_contract_valid(
    ensure_relevance_live_llms,
) -> None:
    """Reproduce the production input shape and require a valid disposition."""

    del ensure_relevance_live_llms
    after_message = (
        '真拿你没办法\n\n据说会转账的只有人类☝️\n'
        '因为金钱流通 文明才得以发展😤\n'
        '是不是这个原因呢❓😚\n'
        '在城市的各个角落每天都有人支付🤗\n'
        '真是令人发笑呢🤣\n'
        '只要有支付 之后就只需要看着🚬\n'
        '提问✋😁\n'
        '50元到账之后会怎么样呢😉\n'
        '三3️⃣\n二2️⃣\n一1️⃣\n时间到⏰⚡\n'
        '正确答案是⭕📄\n'
        '薯条会变得金黄酥脆🍟🍟🍟\n'
        '还有炸鸡 汉堡\n'
        '那个 叫什么来着 河南菜系的🤔\n'
        '对了😮\nKFC什么的🛸💥'
    )
    result, final_error = await _run_captured_probe(
        case_id='qq_638473184_message_148972430',
        after_message=after_message,
    )

    assert final_error is None, (
        'captured settled relevance still exhausts its authoritative repair: '
        f'{final_error}'
    )
    assert result is not None
    assert result['response_action'] == 'proceed'


async def test_live_short_unrelated_after_message_remains_contract_valid(
    ensure_relevance_live_llms,
) -> None:
    """Keep the same action space with a short unrelated after-turn message."""

    del ensure_relevance_live_llms
    result, final_error = await _run_captured_probe(
        case_id='short_unrelated_after_turn_control',
        after_message='今天晚饭吃什么？',
    )

    assert final_error is None, (
        'short after-turn control exhausted authoritative repair: '
        f'{final_error}'
    )
    assert result is not None
    assert result['response_action'] == 'proceed'
