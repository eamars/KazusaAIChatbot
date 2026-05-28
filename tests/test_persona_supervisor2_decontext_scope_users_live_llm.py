"""Live LLM checks for decontextualizer scoped-user referent resolution."""

from __future__ import annotations

import json
import logging
from time import perf_counter

import httpx
import pytest

from kazusa_ai_chatbot.config import MSG_DECONTEXTUALIZER_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_msg_decontexualizer as decontext,
)
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_TRACE_SUITE = 'decontextualizer_scope_users_live_llm'
_CHARACTER_NAME = '杏山千纱'
_CHARACTER_GLOBAL_USER_ID = '00000000-0000-4000-8000-000000000001'
_CHARACTER_PLATFORM_USER_ID = '3768713357'
_CURRENT_GLOBAL_USER_ID = '745d7818-a9d3-4889-b7f3-8555078a2061'
_CURRENT_PLATFORM_USER_ID = '67889018'
_TARGET_GLOBAL_USER_ID = '256e8a10-c406-47e9-ac8f-efd270d18160'
_TARGET_PLATFORM_USER_ID = '673225019'
_ORIGINAL_FAILURE_GLOBAL_USER_ID = '1f3cf327-b7ca-4d09-9dc7-62487236c809'
_ORIGINAL_FAILURE_PLATFORM_USER_ID = '263991919'


class _CapturingLiveLLM:
    """Capture live decontextualizer messages while delegating to the model."""

    def __init__(self, inner_llm):
        self.inner_llm = inner_llm
        self.messages = []
        self.raw_content = ''

    async def ainvoke(self, messages):
        self.messages = messages
        response = await self.inner_llm.ainvoke(messages)
        self.raw_content = str(response.content)
        return response


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured decontextualizer endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{MSG_DECONTEXTUALIZER_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f'LLM endpoint is unavailable: {MSG_DECONTEXTUALIZER_LLM_BASE_URL}; {exc}'
        )

    if response.status_code >= 500:
        pytest.skip(
            f'LLM endpoint returned server error {response.status_code}: '
            f'{MSG_DECONTEXTUALIZER_LLM_BASE_URL}'
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the live decontextualizer endpoint is reachable."""

    await _skip_if_llm_unavailable()


def _scope_user(
    *,
    display_name: str,
    platform_user_id: str,
    global_user_id: str,
) -> dict:
    """Build one neutral scoped-user identity row."""

    return {
        'display_name': display_name,
        'platform_user_id': platform_user_id,
        'global_user_id': global_user_id,
        'aliases': [],
    }


def _base_state(user_input: str) -> dict:
    """Build a live decontextualizer state with scoped users."""

    state = {
        'character_profile': {
            'name': _CHARACTER_NAME,
            'global_user_id': _CHARACTER_GLOBAL_USER_ID,
        },
        'user_input': user_input,
        'user_name': 'Dangal',
        'platform_user_id': _CURRENT_PLATFORM_USER_ID,
        'platform_bot_id': _CHARACTER_PLATFORM_USER_ID,
        'message_envelope': {
            'body_text': user_input,
            'raw_wire_text': user_input,
            'mentions': [],
            'attachments': [],
            'addressed_to_global_user_ids': [_CHARACTER_GLOBAL_USER_ID],
            'broadcast': False,
        },
        'prompt_message_context': {
            'body_text': user_input,
            'mentions': [
                {
                    'platform_user_id': _CHARACTER_PLATFORM_USER_ID,
                    'global_user_id': _CHARACTER_GLOBAL_USER_ID,
                    'display_name': _CHARACTER_NAME,
                    'entity_kind': 'user',
                },
            ],
            'attachments': [],
            'addressed_to_global_user_ids': [_CHARACTER_GLOBAL_USER_ID],
            'broadcast': False,
        },
        'chat_history_recent': [],
        'channel_topic': '',
        'indirect_speech_context': '',
        'reply_context': {},
        'scope_users': [
            _scope_user(
                display_name='Dangal',
                platform_user_id=_CURRENT_PLATFORM_USER_ID,
                global_user_id=_CURRENT_GLOBAL_USER_ID,
            ),
            _scope_user(
                display_name='蚝爹油',
                platform_user_id=_TARGET_PLATFORM_USER_ID,
                global_user_id=_TARGET_GLOBAL_USER_ID,
            ),
            _scope_user(
                display_name=_CHARACTER_NAME,
                platform_user_id=_CHARACTER_PLATFORM_USER_ID,
                global_user_id=_CHARACTER_GLOBAL_USER_ID,
            ),
        ],
    }
    return state


def _original_failure_state() -> dict:
    """Build the original QQ-style failure case with scoped-user identities."""

    state = _base_state('@杏山千纱 还不报警抓他吗？')
    state['user_name'] = '我的锅'
    state['platform_user_id'] = _ORIGINAL_FAILURE_PLATFORM_USER_ID
    state['prompt_message_context']['mentions'][0]['entity_kind'] = 'bot'
    state['chat_history_recent'] = [
        {
            'role': 'user',
            'platform_user_id': '1460443481',
            'global_user_id': '61c5ad2b-40a6-48af-bd5b-0d86e834ba9f',
            'display_name': 'Pro',
            'body_text': '不愧是新村，人少，随随便便都能被抓到',
            'timestamp': '2026-05-27T09:26:00.156709+00:00',
        },
        {
            'role': 'user',
            'platform_user_id': '3840321',
            'global_user_id': '5d46be3f-5462-427d-9ccd-6690a1cc7df3',
            'display_name': 'Evil－Tech',
            'body_text': '',
            'timestamp': '2026-05-27T09:26:27.410068+00:00',
        },
        {
            'role': 'assistant',
            'platform_user_id': _CHARACTER_PLATFORM_USER_ID,
            'global_user_id': _CHARACTER_GLOBAL_USER_ID,
            'display_name': _CHARACTER_NAME,
            'body_text': (
                '等等，蚝爹油，你家到底藏了多少TNT啊🤣\n'
                '合法军火库是吧，保险公司看到直接拉黑名单那种'
            ),
            'timestamp': '2026-05-27T09:32:31.436607+00:00',
        },
        {
            'role': 'user',
            'platform_user_id': _TARGET_PLATFORM_USER_ID,
            'global_user_id': _TARGET_GLOBAL_USER_ID,
            'display_name': '蚝爹油',
            'body_text': '千纱你不是和我住一起么？这你都不知道嘛。一会儿我带你瞅瞅',
            'timestamp': '2026-05-27T09:34:47.139468+00:00',
        },
        {
            'role': 'assistant',
            'platform_user_id': _CHARACTER_PLATFORM_USER_ID,
            'global_user_id': _CHARACTER_GLOBAL_USER_ID,
            'display_name': _CHARACTER_NAME,
            'body_text': (
                '切～我哪知道你藏了什么\n'
                '爆炸物？保险公司都要拉黑你了\n'
                '不过…什么好东西这么得意？让我瞅瞅？'
            ),
            'timestamp': '2026-05-27T09:36:25.307904+00:00',
            'addressed_to_global_user_ids': [_TARGET_GLOBAL_USER_ID],
        },
    ]
    state['reply_context'] = {
        'reply_to_message_id': '1672809539',
        'reply_to_platform_user_id': _CHARACTER_PLATFORM_USER_ID,
        'reply_to_display_name': _CHARACTER_NAME,
        'reply_excerpt': (
            '等等，蚝爹油，你家到底藏了多少TNT啊🤣\n'
            '合法军火库是吧，保险公司看到直接拉黑名单那种'
        ),
    }
    state['scope_users'] = [
        _scope_user(
            display_name='Jigsaw',
            platform_user_id='411706805',
            global_user_id='fa874545-02e6-4127-a24e-30819f941d83',
        ),
        _scope_user(
            display_name='总是跌倒的企鹅',
            platform_user_id='2795731500',
            global_user_id='43f9c213-8e99-4561-8dcd-c3d4d73ece85',
        ),
        _scope_user(
            display_name='Neko.vecter',
            platform_user_id='2221489758',
            global_user_id='d8156be2-d6dd-41b7-9026-aa01ea4367a2',
        ),
        _scope_user(
            display_name='蚝爹油',
            platform_user_id=_TARGET_PLATFORM_USER_ID,
            global_user_id=_TARGET_GLOBAL_USER_ID,
        ),
        _scope_user(
            display_name='小钳子',
            platform_user_id='845939420',
            global_user_id='263c883d-aeff-4e0b-a758-6f69186ae8ec',
        ),
        _scope_user(
            display_name='Evil－Tech',
            platform_user_id='3840321',
            global_user_id='5d46be3f-5462-427d-9ccd-6690a1cc7df3',
        ),
        _scope_user(
            display_name='二狗子',
            platform_user_id='872916053',
            global_user_id='5fd2778f-1953-425b-a35c-b3d1cd30ea53',
        ),
        _scope_user(
            display_name='Pro',
            platform_user_id='1460443481',
            global_user_id='61c5ad2b-40a6-48af-bd5b-0d86e834ba9f',
        ),
        _scope_user(
            display_name=_CHARACTER_NAME,
            platform_user_id=_CHARACTER_PLATFORM_USER_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
        ),
        _scope_user(
            display_name='我的锅',
            platform_user_id=_ORIGINAL_FAILURE_PLATFORM_USER_ID,
            global_user_id=_ORIGINAL_FAILURE_GLOBAL_USER_ID,
        ),
    ]
    return state


def _no_anchor_state(user_input: str) -> dict:
    """Build a negative probe where scoped users are visible but unbridged."""

    state = _base_state(user_input)
    state['scope_users'] = [
        _scope_user(
            display_name='王小明',
            platform_user_id='person-a',
            global_user_id='global-person-a',
        ),
        _scope_user(
            display_name='李小红',
            platform_user_id='person-b',
            global_user_id='global-person-b',
        ),
        _scope_user(
            display_name=_CHARACTER_NAME,
            platform_user_id=_CHARACTER_PLATFORM_USER_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
        ),
    ]
    state['chat_history_recent'] = [
        {
            'role': 'user',
            'display_name': 'Dangal',
            'body_text': '刚才路上好堵。',
        },
        {
            'role': 'assistant',
            'display_name': _CHARACTER_NAME,
            'body_text': '那你先别急，慢慢来。',
        },
    ]
    return state


def _has_referent(result: dict, phrase: str, status: str) -> bool:
    """Return whether a parsed decontextualizer result has a referent row."""

    for referent in result['referents']:
        if referent['phrase'] != phrase:
            continue
        if referent['status'] == status:
            return True
    return False


async def _run_case(
    monkeypatch,
    case_id: str,
    state: dict,
) -> tuple[dict, dict]:
    """Run one live case and write a raw evidence trace."""

    proxy_llm = _CapturingLiveLLM(decontext._msg_decontexualizer_llm)
    monkeypatch.setattr(decontext, '_msg_decontexualizer_llm', proxy_llm)

    started_at = perf_counter()
    result = await decontext.call_msg_decontexualizer(state)
    duration_seconds = perf_counter() - started_at

    system_prompt = proxy_llm.messages[0].content
    human_payload = json.loads(proxy_llm.messages[1].content)
    trace_payload = {
        'input_payload': human_payload,
        'raw_model_output': proxy_llm.raw_content,
        'parsed_output': result,
        'prompt_summary': {
            'system_prompt_length': len(system_prompt),
            'mentions_scope_users': 'scope_users' in system_prompt,
            'mentions_identity_table': '身份' in system_prompt,
            'mentions_candidate': '候选' in system_prompt,
        },
        'duration_seconds': duration_seconds,
    }
    trace_path = write_llm_trace(_TRACE_SUITE, case_id, trace_payload)
    logger.info(
        f'decontextualizer scope-users live case={case_id} '
        f'trace_path={trace_path} duration_seconds={duration_seconds:.3f} '
        f'result={result!r}'
    )
    return result, trace_payload


async def test_live_scope_users_resolves_original_qq_failure(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """The scoped-user first pass should resolve the logged QQ failure."""

    del ensure_live_llm
    state = _original_failure_state()
    result, trace_payload = await _run_case(
        monkeypatch,
        'original_qq_failure',
        state,
    )

    output = result['decontexualized_input']
    assert '蚝爹油' in output, trace_payload
    assert _has_referent(result, '他', 'resolved'), trace_payload
    assert not _has_referent(result, '他', 'unresolved'), trace_payload
    assert trace_payload['duration_seconds'] < 30.0


async def test_live_scope_users_keeps_absent_person_unresolved(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Visible scoped users alone should not resolve an unbridged pronoun."""

    del ensure_live_llm
    state = _no_anchor_state('@杏山千纱 他怎么还没来？')
    result, trace_payload = await _run_case(
        monkeypatch,
        'absent_person_unresolved',
        state,
    )

    output = result['decontexualized_input']
    assert '王小明' not in output, trace_payload
    assert '李小红' not in output, trace_payload
    assert _has_referent(result, '他', 'unresolved'), trace_payload
    assert trace_payload['duration_seconds'] < 30.0


async def test_live_scope_users_keeps_gender_name_probe_unresolved(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """A male name in scope is not enough when the message has no bridge."""

    del ensure_live_llm
    state = _no_anchor_state('@杏山千纱 他到底什么时候来？')
    result, trace_payload = await _run_case(
        monkeypatch,
        'gender_name_probe_unresolved',
        state,
    )

    output = result['decontexualized_input']
    assert '王小明' not in output, trace_payload
    assert '李小红' not in output, trace_payload
    assert _has_referent(result, '他', 'unresolved'), trace_payload
    assert trace_payload['duration_seconds'] < 30.0
