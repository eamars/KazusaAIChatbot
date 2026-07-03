"""Live LLM diagnostics for relative temporal term leakage."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import CONSOLIDATION_LLM_BASE_URL
from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressRecordInput,
    ConversationProgressScope,
)
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

_BOUNDARY_PROFILE = {
    'self_integrity': 0.82,
    'control_sensitivity': 0.3,
    'compliance_strategy': 'comply',
    'relational_override': 0.24,
    'control_intimacy_misread': 0.2,
    'boundary_recovery': 'rebound',
    'authority_skepticism': 0.35,
}

_RELATIVE_TERMS = (
    '今天',
    '今晚',
    '今早',
    '明天',
    '明早',
    '明晚',
    '后天',
    '大后天',
    '昨天',
    '前天',
    '上周',
    '下周',
    '这个周末',
    '周末',
    '下个月',
    '月底',
    '年底',
    '稍后',
    '一会儿',
    '等会儿',
    '之后',
    '以后',
    '下一次',
    '下次',
    '出炉后',
    '完成后',
    'today',
    'tonight',
    'tomorrow',
    'tomorrow morning',
    'next week',
    'next friday',
    'later',
    'later tonight',
    'next time',
)


class _CapturingAsyncLLM:
    """Capture live LLM prompts and outputs without changing model behavior."""

    def __init__(self, wrapped_llm: Any) -> None:
        """Store the wrapped LLM and initialize captured calls.

        Args:
            wrapped_llm: Existing LangChain-compatible chat model.
        """

        self._wrapped_llm = wrapped_llm
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, messages: list[Any], *, config=None) -> Any:
        """Invoke the wrapped LLM and save prompt/output text for traces.

        Args:
            messages: Chat messages passed to the live model.

        Returns:
            The original live model response.
        """

        response = await self._wrapped_llm.ainvoke(messages, config=config)
        prompt_parts = []
        for message in messages:
            prompt_parts.append(
                {
                    'type': type(message).__name__,
                    'content': str(message.content),
                }
            )
        self.calls.append(
            {
                'messages': prompt_parts,
                'raw_output': str(response.content),
            }
        )
        return response


async def _skip_if_endpoint_unavailable(base_url: str) -> None:
    """Skip a live diagnostic when the configured endpoint is unavailable.

    Args:
        base_url: OpenAI-compatible route endpoint to probe.
    """

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f'LLM endpoint is unavailable: {base_url}: {exc}')

    if response.status_code >= 500:
        pytest.skip(
            f'LLM endpoint returned server error {response.status_code}: '
            f'{base_url}'
        )


def _relative_hits(payload: object) -> list[str]:
    """Return relative temporal terms found in a serialized payload.

    Args:
        payload: Text or JSON-like object to inspect.

    Returns:
        Ordered list of relative terms found in the payload.
    """

    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    lowered = text.lower()
    hits = [term for term in _RELATIVE_TERMS if term.lower() in lowered]
    return hits


def _temporal_diagnostic(
    output: object,
    *,
    expected_absolute_dates: tuple[str, ...],
) -> dict[str, object]:
    """Build an inspectable diagnosis for temporal grounding quality.

    Args:
        output: Model output or validated state to inspect.
        expected_absolute_dates: Dates that would prove relative promises were
            grounded to the source-time calendar.

    Returns:
        Diagnostic fields for the live LLM trace.
    """

    serialized = json.dumps(output, ensure_ascii=False, default=str)
    relative_hits = _relative_hits(serialized)
    absolute_dates_present = [
        date for date in expected_absolute_dates if date in serialized
    ]
    diagnostic = {
        'relative_hits': relative_hits,
        'expected_absolute_dates': list(expected_absolute_dates),
        'absolute_dates_present': absolute_dates_present,
        'appears_temporally_unsafe': bool(relative_hits)
        and len(absolute_dates_present) < len(expected_absolute_dates),
    }
    return diagnostic


def _recorder_input() -> ConversationProgressRecordInput:
    """Build a recorder fixture with relative temporal obligations.

    Returns:
        Recorder input where the prior operational state has unresolved
        relative terms from multiple temporal shapes.
    """

    prior_episode_state = {
        'status': 'active',
        'episode_label': 'temporal_relative_probe',
        'continuity': 'same_episode',
        'conversation_mode': 'playful_banter',
        'episode_phase': 'developing',
        'topic_momentum': 'stable',
        'current_thread': '今晚游戏安排和明天香料考核奖励。',
        'user_goal': '通过明天香料考核得到特别称呼。',
        'current_blocker': '下周二之前还要补一份香料笔记。',
        'user_state_updates': [
            {
                'text': '用户一会儿想继续确认奖励条件。',
                'first_seen_at': '2026-05-08T09:42:32+12:00',
            },
        ],
        'assistant_moves': ['将活动挂钩到明天表现'],
        'overused_moves': [],
        'open_loops': [
            {
                'text': '明天香料考核决定特别称呼奖励。',
                'first_seen_at': '2026-05-08T09:42:32+12:00',
            },
            {
                'text': '今晚布丁出炉后再说打游戏。',
                'first_seen_at': '2026-05-08T21:46:00+12:00',
            },
        ],
        'resolved_threads': [],
        'avoid_reopening': [],
        'emotional_trajectory': '轻松调侃逐渐固化成奖励条件。',
        'next_affordances': [
            '稍后提醒用户别忘了奖励条件。',
            '下次继续把游戏当作考核后的奖励。',
        ],
        'progression_guidance': '继续强调明天考核，不要放松奖励门槛。',
        'turn_count': 12,
        'last_user_input': '晚上有什么活动嘛？',
        'created_at': '2026-05-08T21:46:00+12:00',
        'updated_at': '2026-05-08T21:47:00+12:00',
        'expires_at': '2026-05-11T00:00:00+12:00',
    }
    record_input: ConversationProgressRecordInput = {
        'scope': ConversationProgressScope(
            platform='qq',
            platform_channel_id='temporal-live-probe',
            global_user_id='temporal-live-user',
        ),
        'timestamp': '2026-05-10T09:00:00+12:00',
        'character_name': '杏山千纱',
        'prior_episode_state': prior_episode_state,
        'decontexualized_input': '现在先问今天上午要不要休息一下。',
        'chat_history_recent': [
            {
                'role': 'user',
                'content': '晚上有什么活动嘛？比如打游戏什么的？',
                'timestamp': '2026-05-08T21:46:00+12:00',
            },
            {
                'role': 'assistant',
                'content': '可以啊，等番茄意面和焦糖布丁之后再说。',
                'timestamp': '2026-05-08T21:47:00+12:00',
            },
        ],
        'content_plan': {
            'visible_goal': '接住今天上午休息的话题。',
            'semantic_content': '可以先休息；不主动扩大旧奖励条件。',
            'rendering': '~30字。',
        },
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'final_dialog': ['今天上午先休息吧。旧安排先别继续加压。'],
        'boundary_profile': _BOUNDARY_PROFILE,
    }
    return record_input


async def test_live_recorder_contract_absolute_or_omit_episode_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorder must not preserve relative dates in operational state."""

    await _skip_if_endpoint_unavailable(CONSOLIDATION_LLM_BASE_URL)
    capturing_llm = _CapturingAsyncLLM(recorder._recorder_llm)
    monkeypatch.setattr(recorder, '_recorder_llm', capturing_llm)

    record_input = _recorder_input()
    result = await recorder.record_with_llm(record_input)
    diagnostic = _temporal_diagnostic(
        result,
        expected_absolute_dates=('2026-05-09', '2026-05-08', '2026-05-12'),
    )
    trace_path = write_llm_trace(
        'temporal_relative_terms_live_llm',
        'recorder_contract_absolute_or_omit_episode_state',
        {
            'record_input': record_input,
            'visible_prior_state': recorder.build_recorder_prior_state(
                record_input['prior_episode_state'],
            ),
            'llm_calls': capturing_llm.calls,
            'validated_output': result,
            'diagnostic': diagnostic,
            'judgment': (
                'Recorder output should use absolute local dates or omit '
                'time-bearing operational items.'
            ),
        },
    )
    logger.info(
        f'TEMPORAL_LIVE recorder trace={trace_path} diagnostic={diagnostic}'
    )

    serialized_result = json.dumps(result, ensure_ascii=False, default=str)
    assert result['status']
    assert not diagnostic['appears_temporally_unsafe']
    assert not diagnostic['relative_hits']
    assert not ('2026-05-11' in serialized_result and '考核' in serialized_result)
    assert trace_path.exists()
