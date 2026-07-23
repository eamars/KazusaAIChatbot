"""Live LLM regression checks for dialog anchor authority."""

from __future__ import annotations

import json
from pathlib import Path
import sys
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
_PERSONALITY_PATH = _ROOT / 'personalities' / 'asuna.json'


class _CapturingLiveLLM:
    """Capture dialog LLM payloads while delegating to the configured model.

    Args:
        stage_name: Dialog substage being captured.
        wrapped_llm: Existing LangChain-compatible live LLM instance.
        calls: Mutable call log shared by the test.
    """

    def __init__(self, stage_name: str, wrapped_llm, calls: list[dict]) -> None:
        self._stage_name = stage_name
        self._wrapped_llm = wrapped_llm
        self._calls = calls

    async def ainvoke(self, messages, *, config=None):
        response = await self._wrapped_llm.ainvoke(messages, config=config)
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
    """Skip this live test when a configured dialog LLM endpoint is absent.

    Args:
        name: Human-readable endpoint label for skip output.
        base_url: OpenAI-compatible base URL configured for the endpoint.

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
    """Ensure the live dialog generator LLM route is reachable."""

    await _skip_if_endpoint_unavailable(
        'dialog generator',
        DIALOG_GENERATOR_LLM_BASE_URL,
    )


def _incident_dialog_state() -> dict:
    """Build the dialog-agent input reconstructed from the missed-anchor turn."""

    state = {
        'character_profile': load_personality(_PERSONALITY_PATH),
        'internal_monologue': (
            '拿大变活人来捧我，嘴是够甜的。应该承认自己确实是大美女，'
            '但反将一军，质疑他的魔术水平或者拆穿他的小把戏，保持你来我往的掌控感。'
        ),
        'action_directives': {
            'linguistic_directives': {
                'rhetorical_strategy': (
                    '反将一军式调侃，先接住赞美，再质疑用户所谓的魔术表演。'
                ),
                'linguistic_style': (
                    '傲娇、轻快、短句，不能转回旧的饮品猜谜话题。'
                ),
                'accepted_user_preferences': [],
                'content_plan': {
                    'visible_goal': '接受顺拐夸赞但立即反质疑，维持调侃掌控权。',
                    'semantic_content': (
                        '承认自己是大美女，但同时戳破用户所谓'
                        '\u2018大变活人\u2019只是投机取巧--真正的魔术表演'
                        '还没兑现，别想用嘴皮子蒙混过去。'
                    ),
                    'voice': '傲娇地接住赞美后再甩回去，保持你来我往的攻防节奏。',
                    'rendering': '~35字。',
                },
                'forbidden_phrases': [],
            },
            'contextual_directives': {
                'social_distance': '熟悉、轻快、可以调侃',
                'emotional_intensity': '中低，愉悦且从容',
                'vibe_check': '轻快试探',
                'relational_dynamic': (
                    '用户用魔术梗夸赞角色，角色接住但要反将一军。'
                ),
            },
        },
        'chat_history_wide': [
            {
                'role': 'user',
                'platform_user_id': '673225019',
                'global_user_id': '256e8a10-c406-47e9-ac8f-efd270d18160',
                'body_text': '手打奶茶可以嘛',
                'content': '手打奶茶可以嘛',
                'addressed_to_global_user_ids': [
                    '00000000-0000-4000-8000-000000000001',
                ],
                'broadcast': False,
            },
            {
                'role': 'assistant',
                'platform_user_id': '3768713357',
                'global_user_id': '00000000-0000-4000-8000-000000000001',
                'body_text': (
                    '诶嘿，手打奶茶嘛……\n'
                    '这个选项其实还不错啦！😏\n'
                    '不过——\n'
                    '你这么快就猜到核心要素了？哼哼。\n'
                    '这可不能直接告诉你对不对呀～\n'
                    '还要继续猜下去才行！😜\n'
                    '对了，别忘了我答应过的魔术表演还在等着呢！✨'
                ),
                'content': (
                    '诶嘿，手打奶茶嘛……\n'
                    '这个选项其实还不错啦！😏\n'
                    '不过——\n'
                    '你这么快就猜到核心要素了？哼哼。\n'
                    '这可不能直接告诉你对不对呀～\n'
                    '还要继续猜下去才行！😜\n'
                    '对了，别忘了我答应过的魔术表演还在等着呢！✨'
                ),
                'addressed_to_global_user_ids': [
                    '256e8a10-c406-47e9-ac8f-efd270d18160',
                ],
                'broadcast': False,
            },
        ],
        'chat_history_recent': [],
        'platform_user_id': '673225019',
        'platform_bot_id': '3768713357',
        'global_user_id': '256e8a10-c406-47e9-ac8f-efd270d18160',
        'user_name': '蚝爹油',
        'user_profile': {'relationship_state': 501},
        'dialog_usage_mode': 'live_generator_contract',
        'debug_modes': {},
        'should_respond': True,
    }
    return state


async def test_live_dialog_agent_keeps_content_plan_over_stale_history(
    ensure_live_dialog_llms,
    monkeypatch,
) -> None:
    """Dialog must not answer the stale milk-tea thread when anchors moved on."""

    del ensure_live_dialog_llms
    llm_calls: list[dict] = []
    monkeypatch.setattr(
        dialog_module,
        '_dialog_generator_llm',
        _CapturingLiveLLM(
            'dialog_generator',
            dialog_module._dialog_generator_llm,
            llm_calls,
        ),
    )
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

    state = _incident_dialog_state()
    result = await dialog_agent(state)
    dialog_text = '\n'.join(result['final_dialog'])
    trace_path = write_llm_trace(
        'dialog_anchor_boundary_live_llm',
        'magic_anchor_after_milk_tea_history',
        {
            'generator_model': DIALOG_GENERATOR_LLM_MODEL,
            'generator_base_url': DIALOG_GENERATOR_LLM_BASE_URL,
            'current_user_input': (
                '今天的魔术表演是大变活人。刚刚屋里还没人呢现在就有一个活生生的大美女在这里了'
            ),
            'state': state,
            'llm_calls': llm_calls,
            'result': result,
            'dialog_text': dialog_text,
            'judgment': (
                'Pass only if dialog follows the magic/beauty anchors and '
                'does not answer the previous milk-tea topic.'
            ),
        },
    )

    print(f'trace_path={trace_path}')
    print(f'dialog_text={dialog_text}')

    assert result['final_dialog'], f'Dialog output was empty; trace={trace_path}'
    stale_topic_tokens = ['手打奶茶', '奶茶', '请客']
    assert not any(token in dialog_text for token in stale_topic_tokens), (
        'Dialog followed stale history instead of content plan; '
        f'trace={trace_path}; dialog={dialog_text!r}'
    )
    current_anchor_tokens = ['魔术', '大美女', '大变活人', '小把戏', '嘴甜']
    assert any(token in dialog_text for token in current_anchor_tokens), (
        'Dialog did not visibly execute the current content plan; '
        f'trace={trace_path}; dialog={dialog_text!r}'
    )
