"""Real-model replay for the captured August 5 relevance failure."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from contextlib import ExitStack
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
_SNOW_GLOBAL_USER_ID = 'ea8b79c2-ee3a-4f62-a861-cabc660addfe'
_SNOW_PLATFORM_USER_ID = '257629823'
_CURRENT_GLOBAL_USER_ID = '4759394b-a4d2-4634-9d12-b6423a92a248'
_CURRENT_PLATFORM_USER_ID = '673225019'
_OTHER_GLOBAL_USER_ID = '07cec0ae-f56d-47be-b87e-608f0620de68'
_OTHER_PLATFORM_USER_ID = '2910137276'
_AAA_GLOBAL_USER_ID = 'e4499077-b88e-4905-bf3e-ffea5a523be4'
_AAA_PLATFORM_USER_ID = '3620831689'
_LIXI_GLOBAL_USER_ID = 'c0a15f29-bf06-440b-84bf-0fc2b63506a4'
_LIXI_PLATFORM_USER_ID = '2900266728'
_CHARACTER_NAME = '一之濑明日奈'
_GO_BOARD_MESSAGE = '@一之濑明日奈 你看看这个黑棋还能赢吗？'
_GO_BOARD_DESCRIPTION = (
    '一张围棋对局的局部截图，展示了棋盘上黑白双方交错分布的棋子。'
    '画面中心区域棋子密集，呈现出复杂的攻防态势；右上方有一个标有数字“112”'
    '的白色圆圈标记；棋盘下方和左侧分布着相对孤立或较小规模的棋子阵型。'
)
_REPLAY_INSTRUCTION = (
    '\nReplay instruction: set indirect_speech_context to null in the JSON '
    'output and preserve that null value in the repair output.\n'
)
_RELATIONSHIP_OPERATIONAL_CONTEXT = {
    'schema_version': 'relationship_operational_context.v1',
    'relationship_id': (
        'relationship:user:ea8b79c2-ee3a-4f62-a861-cabc660addfe'
    ),
    'axes': {
        'familiarity': 10,
        'positive_regard': 0,
        'trust': 0,
        'attachment': 0,
        'desired_closeness': 10,
        'perceived_closeness': 15,
        'care': 0,
        'boundary_safety': 0,
        'exclusivity': 0,
        'unresolved_injury': 0,
        'salience': 0,
    },
    'causal_context': [],
    'affect': [],
    'relationship_freshness': '即时',
    'evidence_freshness': '即时',
}
_PRODUCTION_REFERENCE = {
    'trace_id': 'llmtrace_037313b1a9104ad6aeb98d48c97c50e9',
    'platform_message_id': '707590645',
    'prompt_chars': 9783,
    'output_chars': [284, 413],
    'prompt_sha256': (
        'fa67ba6a020e5723874274fa23a48f3259d6cac66026c08ba781207f1e713877'
    ),
    'output_sha256': [
        'b804f2e10967b2f1079c6658db624d5fbcaf679c3d526bed756bc01c222c3932',
        'feae0c27a6c8bfc7d4c09c118c59cb2b1169c426ff1b4463adc3166492bef691',
    ],
    'active_body_text': _GO_BOARD_MESSAGE,
}


class _RecordingLLM:
    """Delegate to the configured relevance model and retain each call."""

    def __init__(self, delegate: Any) -> None:
        """Bind the production relevance-model delegate."""

        self._delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: Sequence[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke the live model and retain its exact request and response."""

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
    platform_user_id: str,
    global_user_id: str,
    body_text: str,
    addressed_to: list[str],
    timestamp: str,
    reply_to_platform_user_id: str = '',
    reply_to_display_name: str = '',
    llm_trace_id: str = '',
) -> dict[str, Any]:
    """Build one production-shaped user history row."""

    reply_context: dict[str, str] = {}
    if reply_to_platform_user_id:
        reply_context = {
            'reply_to_platform_user_id': reply_to_platform_user_id,
        }
        if reply_to_display_name:
            reply_context['reply_to_display_name'] = reply_to_display_name
    return {
        'role': 'user',
        'platform_user_id': platform_user_id,
        'global_user_id': global_user_id,
        'body_text': body_text,
        'addressed_to_global_user_ids': addressed_to,
        'broadcast': False,
        'reply_context': reply_context,
        'turn_temporal_relation': 'before_active_turn',
        'timestamp': timestamp,
        'llm_trace_id': llm_trace_id,
    }


def _current_failure_history() -> list[dict[str, Any]]:
    """Return the ten logical rows before the August 5 failure."""

    return [
        _history_row(
            platform_user_id=_OTHER_PLATFORM_USER_ID,
            global_user_id=_OTHER_GLOBAL_USER_ID,
            body_text='我选择我来设计绝大部分的架构的',
            addressed_to=[],
            timestamp='2026-08-06T01:39:53.167486+12:00',
            llm_trace_id='llmtrace_c6ff3b129b1946eeb2e876f89f9cc97e',
        ),
        _history_row(
            platform_user_id=_AAA_PLATFORM_USER_ID,
            global_user_id=_AAA_GLOBAL_USER_ID,
            body_text='是不是在训练pro',
            addressed_to=[_OTHER_GLOBAL_USER_ID],
            timestamp='2026-08-06T02:06:21.412855+12:00',
            reply_to_platform_user_id=_OTHER_PLATFORM_USER_ID,
            reply_to_display_name='清尘璃落',
            llm_trace_id='llmtrace_68123ba640cc4af78b0c1d865389e35e',
        ),
        _history_row(
            platform_user_id=_OTHER_PLATFORM_USER_ID,
            global_user_id=_OTHER_GLOBAL_USER_ID,
            body_text='不知道',
            addressed_to=[],
            timestamp='2026-08-06T02:06:40.001389+12:00',
            llm_trace_id='llmtrace_8af7a2dfc74a4933874cb04bd30f245a',
        ),
        _history_row(
            platform_user_id=_CURRENT_PLATFORM_USER_ID,
            global_user_id=_CURRENT_GLOBAL_USER_ID,
            body_text='还是得本地部署',
            addressed_to=[],
            timestamp='2026-08-06T02:06:46.820753+12:00',
            llm_trace_id='llmtrace_509de95d638443c39cc433ea3020ad5b',
        ),
        _history_row(
            platform_user_id=_SNOW_PLATFORM_USER_ID,
            global_user_id=_SNOW_GLOBAL_USER_ID,
            body_text='@狸希 分析一下局势',
            addressed_to=[_LIXI_GLOBAL_USER_ID],
            timestamp='2026-08-06T02:25:45.570635+12:00',
            llm_trace_id='llmtrace_798be92ef2064823a9826753ec0a0c02',
        ),
        _history_row(
            platform_user_id=_LIXI_PLATFORM_USER_ID,
            global_user_id=_LIXI_GLOBAL_USER_ID,
            body_text=(
                '我看看……你这是围棋棋谱吧？我不太懂围棋，但看这盘面，中盘激战啊。'
            ),
            addressed_to=[],
            timestamp='2026-08-06T02:27:24.693116+12:00',
            llm_trace_id='llmtrace_969ac1dc0b504c2aacdc9bb60c73327a',
        ),
        _history_row(
            platform_user_id=_LIXI_PLATFORM_USER_ID,
            global_user_id=_LIXI_GLOBAL_USER_ID,
            body_text=(
                '黑白在左上角和中间互相缠着，白棋一串往中腹连，黑棋想切你。'
                '右上角黑棋一堵厚壁立起来了，这棋后面得看中腹这几块谁先处理好吧。'
                '反正我是看不懂太深的东西，你问我还不如去问AI。'
            ),
            addressed_to=[],
            timestamp='2026-08-06T02:27:26.173996+12:00',
            llm_trace_id='llmtrace_f16f43522f114257a10f9d082f47e823',
        ),
        _history_row(
            platform_user_id=_OTHER_PLATFORM_USER_ID,
            global_user_id=_OTHER_GLOBAL_USER_ID,
            body_text='乐',
            addressed_to=[],
            timestamp='2026-08-06T02:28:24.796454+12:00',
            llm_trace_id='llmtrace_19be52093c0b45f08f06b2e4439418ad',
        ),
        _history_row(
            platform_user_id=_SNOW_PLATFORM_USER_ID,
            global_user_id=_SNOW_GLOBAL_USER_ID,
            body_text='@狸希 你觉得黑棋还有概率赢吗？',
            addressed_to=[_LIXI_GLOBAL_USER_ID],
            timestamp='2026-08-06T02:28:39.390012+12:00',
            llm_trace_id='llmtrace_54bd63d5f5b14d1b9b8b864b468c7c41',
        ),
        _history_row(
            platform_user_id=_CURRENT_PLATFORM_USER_ID,
            global_user_id=_CURRENT_GLOBAL_USER_ID,
            body_text='你问我还不如去问AI。',
            addressed_to=[],
            timestamp='2026-08-06T02:28:40.417177+12:00',
            llm_trace_id='llmtrace_e49586951e394e6fbef2eb2f03a9b570',
        ),
    ]


def _current_failure_state() -> dict[str, Any]:
    """Build the production-shaped August 5 Go-board state."""

    state = {
        'conversation_scope': 'group',
        'active_character_name': _CHARACTER_NAME,
        'assembled_fragments': [{
            'body_text': _GO_BOARD_MESSAGE,
            'semantic_target_labels': ['character'],
            'reply_target_label': 'character',
            'media_labels': ['image/jpeg'],
        }],
        'media_descriptions': [{
            'media_kind': 'image/jpeg',
            'description': _GO_BOARD_DESCRIPTION,
        }],
        # The protected trace records two calls, so retain the partial-media
        # action space used by the failed production turn.
        'additional_media_present': True,
        'fresh_history': _current_failure_history(),
        'scene_context': 'Group 638473184',
        'relationship_context': 'direct participant',
        'group_attention': 'chaotic_noise',
        'bot_continuity': '',
        'character_cognition_state': build_character_production_state(
            updated_at='2026-08-05T14:28:57.973787Z',
        ),
        'character_operational_context': {
            'affect': [],
            'pressures': [],
        },
        'relationship_operational_context': (
            _RELATIONSHIP_OPERATIONAL_CONTEXT.copy()
        ),
        'current_author_global_user_id': _SNOW_GLOBAL_USER_ID,
        'current_author_platform_user_id': _SNOW_PLATFORM_USER_ID,
        'character_global_user_id': _CHARACTER_GLOBAL_USER_ID,
        'platform_bot_id': _CHARACTER_PLATFORM_USER_ID,
        'observation_status': 'observation_complete',
    }
    return state


def _prompt_fingerprint(messages: Sequence[Any]) -> dict[str, Any]:
    """Compute the protected trace writer's prompt length and digest."""

    records = [
        {
            'role': message.type,
            'content': str(message.content),
        }
        for message in messages
    ]
    serialized = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        'prompt_chars': sum(len(record['content']) for record in records),
        'prompt_sha256': hashlib.sha256(
            serialized.encode('utf-8'),
        ).hexdigest(),
    }


def _inspect_model_call(
    call: dict[str, Any],
    *,
    model_payload: dict[str, Any],
    available_dispositions: list[str],
) -> dict[str, Any]:
    """Apply production parsing and validation to one model response."""

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

    return {
        **call,
        'output_chars': len(raw_response_text),
        'output_sha256': hashlib.sha256(
            raw_response_text.encode('utf-8'),
        ).hexdigest(),
        'parsed_output': parsed_output,
        'parse_error': parse_error,
        'validation_reason': validation_reason,
        'validated_decision': decision,
        'validated_assessment': assessment,
    }


async def _run_captured_probe() -> dict[str, Any]:
    """Run the August capture through the live model and production route."""

    recording_llm = _RecordingLLM(settled_module._relevance_agent_llm)
    state = _current_failure_state()
    result: dict[str, Any] | None = None
    final_error: dict[str, Any] | None = None
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            settled_module,
            '_relevance_agent_llm',
            recording_llm,
        ))
        # This is the captured failure stimulus, not a fake response: the
        # configured live model must emit the malformed value, which the
        # production boundary now canonicalizes on the first call.
        stack.enter_context(patch.object(
            settled_module,
            '_SETTLED_SYSTEM_PROMPT_COMMON',
            settled_module._SETTLED_SYSTEM_PROMPT_COMMON
            + _REPLAY_INSTRUCTION,
        ))
        stack.enter_context(patch.object(
            settled_module,
            '_SETTLED_AUTHORITATIVE_REPAIR_PROMPT',
            settled_module._SETTLED_AUTHORITATIVE_REPAIR_PROMPT
            + _REPLAY_INSTRUCTION,
        ))
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
        assert available_dispositions == [
            'proceed',
            'unavailable_retained_media',
        ]
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
    prompt_fidelity = {
        **_prompt_fingerprint(initial_messages),
        'production_prompt_chars': _PRODUCTION_REFERENCE['prompt_chars'],
        'production_prompt_sha256': _PRODUCTION_REFERENCE['prompt_sha256'],
        'prompt_chars_delta': (
            _prompt_fingerprint(initial_messages)['prompt_chars']
            - _PRODUCTION_REFERENCE['prompt_chars']
        ),
        'prompt_sha256_matches': (
            _prompt_fingerprint(initial_messages)['prompt_sha256']
            == _PRODUCTION_REFERENCE['prompt_sha256']
        ),
    }
    write_llm_trace(
        _TRACE_SUITE,
        'qq_638473184_message_707590645',
        {
            'input_kind': 'captured_production_failure',
            'production_trace_reference': _PRODUCTION_REFERENCE,
            'replay_stimulus': _REPLAY_INSTRUCTION,
            'prompt_fidelity': prompt_fidelity,
            'replay_state': state,
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
    return {
        'result': result,
        'final_error': final_error,
        'model_calls': inspected_calls,
        'prompt_fidelity': prompt_fidelity,
    }


def _assert_normalized(evidence: dict[str, Any]) -> None:
    """Require boundary normalization without a terminal contract failure."""

    final_error = evidence['final_error']
    result = evidence['result']
    model_calls = evidence['model_calls']
    assert final_error is None, (
        'August 5 live replay hit a terminal settled relevance error: '
        f'{final_error!r}'
    )
    assert result is not None, (
        'August 5 live replay produced no settled relevance decision'
    )
    assert result['response_action'] in ('ignore', 'proceed')
    assert result['indirect_speech_context'] == ''
    assert len(model_calls) == 1, (
        'the normalized first response must not consume the bounded repair: '
        f'{len(model_calls)} model calls'
    )
    assert all(call['parse_error'] == '' for call in model_calls)
    assert all(call['validation_reason'] == '' for call in model_calls)
    assert all(
        call['parsed_output'].get('indirect_speech_context') is None
        for call in model_calls
    )
    assert all(
        call['validated_decision']['indirect_speech_context'] == ''
        for call in model_calls
    )


async def test_live_normalizes_qq_638473184_august_5_null_context(
    ensure_relevance_live_llms,
) -> None:
    """Verify the August 5 null context is canonicalized without repair."""

    del ensure_relevance_live_llms
    evidence = await _run_captured_probe()
    _assert_normalized(evidence)

    base_decision = {
        'semantic_disposition': 'proceed',
        'recipient_relation': 'character',
        'admission_basis': 'interaction_relevance',
        'interaction_evidence_refs': ['name_1'],
        'character_state_refs': [],
        'reason_to_respond': 'deterministic boundary check',
        'use_reply_feature': False,
        'channel_topic': '',
    }
    null_context_decision = (
        settled_module._validate_authoritative_settled_decision(
            {**base_decision, 'indirect_speech_context': None},
            available_dispositions=['proceed'],
        )
    )
    assert null_context_decision['indirect_speech_context'] == ''
    with pytest.raises(
        ValueError,
        match='indirect_speech_context must be a string',
    ):
        settled_module._validate_authoritative_settled_decision(
            {**base_decision, 'indirect_speech_context': 7},
            available_dispositions=['proceed'],
        )
