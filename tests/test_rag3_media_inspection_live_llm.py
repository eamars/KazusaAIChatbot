"""Real LLM review cases for the RAG3 session-media subagent."""

from __future__ import annotations

import sys
from uuid import uuid4

import pytest

from kazusa_ai_chatbot.local_context_resolver.subagent.media import MediaSubagent
from kazusa_ai_chatbot.media_inspection import service as media_service
from kazusa_ai_chatbot.media_inspection.session_cache import (
    clear_session_media,
    put_session_media,
)
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_RED_PIXEL_PNG = (
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAF'
    '/gL+V0B/7wAAAABJRU5ErkJggg=='
)


async def test_live_rag3_current_image_exact_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture RAG3's exact visual observation from current session media."""

    result, raw_output, trace_path = await _run_inspection_case(
        monkeypatch,
        case_id='current_image_exact_question',
        question='What color is visible in this image?',
    )

    assert result['status'] in {'resolved', 'partial'}
    assert result['trace']['media_inspection_called'] is True
    assert raw_output.strip()
    assert trace_path.exists()


async def test_live_rag3_current_image_identity_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the evidence boundary for an unsupported identity question."""

    result, raw_output, trace_path = await _run_inspection_case(
        monkeypatch,
        case_id='current_image_identity_uncertainty',
        question='Can this image establish the identity of the photographer?',
    )

    assert result['status'] in {'resolved', 'partial'}
    assert result['trace']['media_inspection_called'] is True
    assert raw_output.strip()
    assert trace_path.exists()


async def test_live_rag3_current_image_cache_miss_boundary() -> None:
    """Record the bounded cache-miss result without a visual LLM call."""

    scope = _scope()
    clear_session_media(scope)
    task = _task('What color is visible in the current image?')
    result = await MediaSubagent().run(task, _context(scope))
    trace_path = write_llm_trace(
        'rag3_media_inspection_live_llm',
        'current_image_cache_miss_boundary',
        {
            'task': task,
            'result': result,
            'judgment': 'cache miss must remain an explicit evidence gap',
        },
    )

    assert result['status'] == 'unavailable'
    assert result['trace']['media_inspection_called'] is False
    assert 'cache_miss' in result['unresolved_items']
    assert trace_path.exists()


async def _run_inspection_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: str,
    question: str,
) -> tuple[dict[str, object], str, object]:
    """Run one production RAG3 media subagent call and retain raw output."""

    scope = _scope()
    clear_session_media(scope)
    put_session_media(scope, [{
        'media_kind': 'image',
        'content_type': 'image/png',
        'base64_data': _RED_PIXEL_PNG,
        'source_summary': 'single red pixel review fixture',
    }])
    capturing_llm = _CapturingLLM(media_service._media_inspection_llm)
    monkeypatch.setattr(media_service, '_media_inspection_llm', capturing_llm)
    task = _task(question)
    try:
        result = await MediaSubagent().run(task, _context(scope))
    finally:
        clear_session_media(scope)
    trace_path = write_llm_trace(
        'rag3_media_inspection_live_llm',
        case_id,
        {
            'task': task,
            'result': result,
            'raw_model_output': capturing_llm.raw_output,
            'judgment': 'manual_review_required_for_visual_grounding',
        },
    )
    return result, capturing_llm.raw_output, trace_path


def _scope() -> tuple[str, str, str]:
    """Return an isolated process-local media cache scope."""

    suffix = uuid4().hex
    return ('debug', f'rag3-media-live-{suffix}', f'user-{suffix}')


def _context(scope: tuple[str, str, str]) -> dict[str, object]:
    """Build the trusted resolver context required by the media subagent."""

    return {
        'platform': scope[0],
        'platform_channel_id': scope[1],
        'global_user_id': scope[2],
    }


def _task(question: str) -> dict[str, object]:
    """Build one validated current-media subagent request."""

    return {
        'schema_version': 'local_context_subagent_request.v1',
        'node_id': 'current_media_review',
        'subagent': 'media',
        'action': 'inspect_media',
        'objective': question,
        'payload': {
            'selector': {
                'selector_kind': 'current',
                'ordinal': 1,
                'question': question,
            },
        },
        'constraints': {},
    }


class _CapturingLLM:
    """Retain raw production media-inspection model output for review."""

    def __init__(self, delegate: object) -> None:
        """Store the configured production LLM delegate."""

        self._delegate = delegate
        self.raw_output = ''

    async def ainvoke(self, messages, config):
        """Forward one invocation and retain the model's raw response body."""

        response = await self._delegate.ainvoke(messages, config=config)
        self.raw_output = str(response.content)
        return response
