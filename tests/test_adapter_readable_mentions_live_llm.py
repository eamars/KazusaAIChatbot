"""Live LLM contract check for adapter-readable mention text."""

from __future__ import annotations

import logging

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    MSG_DECONTEXTUALIZER_LLM_BASE_URL,
    RAG_PLANNER_LLM_BASE_URL,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc

logger = logging.getLogger(__name__)
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.live_llm,
    pytest.mark.skip(reason="retired RAG2 initializer owner; replacement is P3-P3"),
]

_TRACE_SUITE = 'adapter_readable_mentions_live_llm'
_CHARACTER_NAME = '杏山千纱'
_MENTIONED_DISPLAY_NAME = '蚝爹油'
_ADAPTER_SHAPED_TEXT = '@杏山千纱 你怎么评价群友 @蚝爹油'


class _CapturingLiveLLM:
    """Capture live LLM messages while delegating to the configured model."""

    def __init__(self, inner_llm: object) -> None:
        self.inner_llm = inner_llm
        self.messages = []
        self.raw_content = ''

    async def ainvoke(self, messages, *, config=None):
        """Invoke the wrapped live model and keep its prompt and output."""

        self.messages = messages
        response = await self.inner_llm.ainvoke(messages, config=config)
        self.raw_content = str(response.content)
        return response


async def _noop_async(*args, **kwargs) -> None:
    """Avoid persistent Cache2 writes in the focused live route check."""

    del args, kwargs


async def _check_endpoint(base_url: str, label: str) -> None:
    """Skip when one configured live LLM endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
    except httpx.HTTPError as exc:
        pytest.skip(f"{label} endpoint is unavailable: {base_url}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"{label} endpoint returned server error {response.status_code}: "
            f"{base_url}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure both live LLM endpoints needed by this contract are reachable."""

    await _check_endpoint(MSG_DECONTEXTUALIZER_LLM_BASE_URL, 'decontextualizer')
    await _check_endpoint(RAG_PLANNER_LLM_BASE_URL, 'rag planner')


def _decontext_state() -> dict:
    """Build a prompt-safe decontextualizer state from adapter-shaped text."""

    state = {
        'character_profile': {'name': _CHARACTER_NAME},
        'user_input': _ADAPTER_SHAPED_TEXT,
        'user_name': '接待小哥',
        'platform_user_id': '1285663364',
        'platform_bot_id': '3768713357',
        'message_envelope': {
            'body_text': _ADAPTER_SHAPED_TEXT,
            'raw_wire_text': _ADAPTER_SHAPED_TEXT,
            'mentions': [
                {
                    'platform_user_id': '3768713357',
                    'global_user_id': 'c5c8924e-432a-43ce-ab58-27e5f8fee4ec',
                    'display_name': _CHARACTER_NAME,
                    'entity_kind': 'bot',
                    'raw_text': '@杏山千纱',
                },
                {
                    'platform_user_id': '673225019',
                    'global_user_id': '',
                    'display_name': _MENTIONED_DISPLAY_NAME,
                    'entity_kind': 'user',
                    'raw_text': '@蚝爹油',
                },
            ],
            'attachments': [],
            'addressed_to_global_user_ids': [
                'c5c8924e-432a-43ce-ab58-27e5f8fee4ec',
            ],
            'broadcast': False,
        },
        'prompt_message_context': {
            'body_text': _ADAPTER_SHAPED_TEXT,
            'mentions': [
                {
                    'display_name': _CHARACTER_NAME,
                    'entity_kind': 'bot',
                    'global_user_id': 'c5c8924e-432a-43ce-ab58-27e5f8fee4ec',
                },
                {
                    'display_name': _MENTIONED_DISPLAY_NAME,
                    'entity_kind': 'user',
                    'global_user_id': '',
                },
            ],
            'attachments': [],
            'addressed_to_global_user_ids': [
                'c5c8924e-432a-43ce-ab58-27e5f8fee4ec',
            ],
            'broadcast': False,
        },
        'chat_history_recent': [
            {
                'role': 'user',
                'display_name': '我的锅',
                'body_text': '哈哈哈哈，针对NEKO',
            },
        ],
        'channel_topic': 'The group members are jokingly discussing someone being targeted.',
        'indirect_speech_context': '',
        'reply_context': {},
    }
    return state


def _initializer_state(query: str) -> dict:
    """Build a RAG initializer state from the decontextualized query."""

    turn_clock = build_turn_clock_from_storage_utc(
        '2026-05-11T13:08:13+00:00'
    )
    state = {
        'original_query': query,
        'character_name': _CHARACTER_NAME,
        'context': {
            'platform': 'qq',
            'platform_channel_id': '905393941',
            'platform_user_id': '1285663364',
            'global_user_id': 'debug-readable-mention-user',
            'user_name': '接待小哥',
            'current_timestamp_utc': turn_clock['storage_timestamp_utc'],
            'local_time_context': turn_clock['local_time_context'],
            'prompt_message_context': {
                'body_text': _ADAPTER_SHAPED_TEXT,
                'mentions': [
                    {
                        'display_name': _CHARACTER_NAME,
                        'entity_kind': 'bot',
                        'global_user_id': 'c5c8924e-432a-43ce-ab58-27e5f8fee4ec',
                    },
                    {
                        'display_name': _MENTIONED_DISPLAY_NAME,
                        'entity_kind': 'user',
                        'global_user_id': '',
                    },
                ],
                'attachments': [],
                'addressed_to_global_user_ids': [
                    'c5c8924e-432a-43ce-ab58-27e5f8fee4ec',
                ],
                'broadcast': False,
            },
            'conversation_progress': {
                'status': 'new_episode',
                'continuity': 'sharp_transition',
                'current_thread': (
                    'The user asks the active character to evaluate a '
                    'mentioned group member.'
                ),
            },
            'chat_history_recent': [
                {
                    'role': 'user',
                    'display_name': '我的锅',
                    'body_text': '哈哈哈哈，针对NEKO',
                },
            ],
            'chat_history_wide': [],
        },
    }
    return state


async def test_live_adapter_readable_mentions_drive_person_context(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Retired RAG2 initializer coverage remains a documented live skeleton."""

    del ensure_live_llm, monkeypatch
    pytest.skip("retired RAG2 initializer owner; replacement is P3-P3")
