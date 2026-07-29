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
    """Build a V2 recorder fixture with one relative-time agreement.

    Returns:
        Settled turn whose accepted input and dialog establish tomorrow at
        09:00 relative to a 2026-05-10 local semantic clock.
    """

    record_input: ConversationProgressRecordInput = {
        'scope': ConversationProgressScope(
            platform='qq',
            platform_channel_id='temporal-live-probe',
            global_user_id='temporal-live-user',
        ),
        'storage_timestamp_utc': '2026-05-09T21:00:00+00:00',
        'character_name': '杏山千纱',
        'prior_episode_state': None,
        'decontextualized_input': '那就明天上午九点来接我去游乐园吧。',
        'interaction_logical_turns': [],
        'current_turn_source_refs': [
            {
                'ref_kind': 'conversation_row',
                'ref_id': 'temporal-current-row',
                'occurred_at': '2026-05-09T21:00:00+00:00',
            },
            {
                'ref_kind': 'llm_trace',
                'ref_id': 'temporal-current-trace',
                'occurred_at': '2026-05-09T21:00:05+00:00',
            },
        ],
        'turn_outcome': 'visible_response',
        'content_plan': {
            'semantic_content': '杏山千纱接受在约定时间接用户去游乐园。',
            'surface_intent': '确认约定',
        },
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'final_dialog': ['好，明天上午九点我来接你。'],
        'boundary_profile': _BOUNDARY_PROFILE,
    }
    return record_input


async def test_live_recorder_contract_absolute_or_omit_episode_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both V2 producers must ground operational time and character identity."""

    await _skip_if_endpoint_unavailable(CONSOLIDATION_LLM_BASE_URL)
    scene_llm = _CapturingAsyncLLM(recorder._scene_recorder_llm)
    event_llm = _CapturingAsyncLLM(recorder._event_recorder_llm)
    monkeypatch.setattr(recorder, '_scene_recorder_llm', scene_llm)
    monkeypatch.setattr(recorder, '_event_recorder_llm', event_llm)

    record_input = _recorder_input()
    result = await recorder.record_with_llm(record_input)
    diagnostic = _temporal_diagnostic(
        result.delta,
        expected_absolute_dates=('2026-05-11',),
    )
    event_context = recorder.build_event_recorder_context(record_input)
    scene_payload = recorder.build_scene_recorder_human_payload(record_input)
    trace_path = write_llm_trace(
        'temporal_relative_terms_live_llm',
        'recorder_contract_absolute_or_omit_episode_state',
        {
            'record_input': record_input,
            'event_payload': event_context.payload,
            'scene_payload': scene_payload,
            'scene_llm_calls': scene_llm.calls,
            'event_llm_calls': event_llm.calls,
            'validated_delta': result.delta,
            'recorder_call_count': result.recorder_call_count,
            'scene_disposition': result.scene_disposition,
            'event_disposition': result.event_disposition,
            'diagnostic': diagnostic,
            'judgment': (
                'Both producers must use 2026-05-11 09:00 or omit the '
                'time-bearing agreement, and event actor identity must use '
                'the exact runtime character name.'
            ),
        },
    )
    logger.info(
        f'TEMPORAL_LIVE recorder trace={trace_path} diagnostic={diagnostic}'
    )

    serialized_result = json.dumps(
        result.delta,
        ensure_ascii=False,
        default=str,
    )
    event_updates = result.delta['event_updates']
    exact_character_events = [
        event for event in event_updates
        if event['actor'] == record_input['character_name']
    ]
    assert result.recorder_call_count == 2
    assert result.scene_disposition == 'accepted'
    assert result.event_disposition == 'accepted'
    assert not diagnostic['appears_temporally_unsafe']
    assert not diagnostic['relative_hits']
    assert '2026-05-11' in serialized_result or not event_updates
    assert exact_character_events or not event_updates
    assert trace_path.exists()
