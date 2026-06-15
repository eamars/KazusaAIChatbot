"""Tests for the decontextualizer referents migration path."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kazusa_ai_chatbot.config import MSG_DECONTEXTUALIZER_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import (
    call_msg_decontexualizer,
)
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)


class _CapturingLLM:
    """Capture decontextualizer messages while returning a fixed JSON payload."""

    def __init__(self, payload: str):
        self.payload = payload
        self.messages = []

    async def ainvoke(self, messages, *, config=None):
        self.messages = messages
        response = MagicMock()
        response.content = self.payload
        return response


def _base_state() -> dict:
    """Build a minimal decontextualizer state fixture.

    Returns:
        State dictionary with the fields consumed by ``call_msg_decontexualizer``.
    """

    state = {
        "user_input": "这些是什么意思？",
        "user_name": "ReferentUser",
        "platform_user_id": "referent-user",
        "platform_bot_id": "referent-bot",
        "character_profile": {"name": "ReferentCharacter"},
        "message_envelope": {
            "body_text": "这些是什么意思？",
            "raw_wire_text": "这些是什么意思？",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "prompt_message_context": {
            "body_text": "这些是什么意思？",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "chat_history_recent": [
            {"role": "user", "body_text": "晚上好"},
            {"role": "assistant", "body_text": "晚上好。"},
        ],
        "channel_topic": "",
        "indirect_speech_context": "",
        "reply_context": {},
    }
    return state


@pytest.mark.asyncio
async def test_decontexualizer_prompt_requires_character_name_and_identity_safe_examples(
    monkeypatch,
) -> None:
    """Decontextualizer prompt renders character identity without payload duplication."""

    llm_payload = (
        '{"output": "是的", "reasoning": "identity contract check", '
        '"is_modified": false, "referents": []}'
    )
    fake_llm = _CapturingLLM(llm_payload)
    monkeypatch.setattr(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer."
        "_msg_decontexualizer_llm",
        fake_llm,
    )
    state = _base_state()
    state["character_profile"] = {"name": '测试角色'}

    await call_msg_decontexualizer(state)

    system_prompt = fake_llm.messages[0].content
    human_payload = json.loads(fake_llm.messages[1].content)
    assert "character_name" not in human_payload
    assert '测试角色' in system_prompt
    assert '{character_name}' not in system_prompt
    assert '"character_name": "string"' not in system_prompt
    assert '`character_name`' not in system_prompt
    assert '当前助手/角色' not in system_prompt
    assert '当前角色说明白' not in system_prompt
    assert '# 正向模式' not in system_prompt
    assert 'user_input =' not in system_prompt
    assert '例如' not in system_prompt
    assert '可见参与者字面名称' in system_prompt
    assert '怪词、术语或普通话题名' in system_prompt


@pytest.mark.asyncio
async def test_decontextualizer_projects_chat_history_as_transcript_lines(
    monkeypatch,
) -> None:
    """Decontextualizer should flatten only chat history before the LLM call."""

    llm_payload = (
        '{"output": "原句", "reasoning": "payload projection check", '
        '"is_modified": false, "referents": []}'
    )
    fake_llm = _CapturingLLM(llm_payload)
    monkeypatch.setattr(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer."
        "_msg_decontexualizer_llm",
        fake_llm,
    )
    state = _base_state()
    state['user_input'] = '@杏山千纱 那蚝爹油跟你啥关系'
    state['prompt_message_context']['body_text'] = state['user_input']
    state['message_envelope']['body_text'] = state['user_input']
    state['message_envelope']['raw_wire_text'] = state['user_input']
    state['scope_users'] = [
        {
            'display_name': '蚝爹油',
            'platform_user_id': '673225019',
            'global_user_id': '256e8a10-c406-47e9-ac8f-efd270d18160',
            'aliases': [],
        }
    ]
    state['chat_history_recent'] = [
        {
            'role': 'user',
            'display_name': '蚝爹油',
            'body_text': '捡垃圾不是乐趣么',
            'timestamp': '2026-06-13 15:04',
            'platform_user_id': '673225019',
            'global_user_id': '256e8a10-c406-47e9-ac8f-efd270d18160',
        },
        {
            'role': 'user',
            'display_name': '1816',
            'body_text': '@杏山千纱 那蚝爹油跟你啥关系',
            'timestamp': '2026-06-13 15:14',
            'reply_context': {
                'reply_to_display_name': '杏山千纱',
                'reply_to_platform_user_id': '3768713357',
            },
        },
    ]

    await call_msg_decontexualizer(state)

    human_payload = json.loads(fake_llm.messages[1].content)
    assert human_payload['chat_history'] == [
        '[2026-06-13 15:04] 蚝爹油: 捡垃圾不是乐趣么',
        '[2026-06-13 15:14] 1816 reply_to 杏山千纱: @杏山千纱 那蚝爹油跟你啥关系',
    ]
    assert human_payload['scope_users'] == state['scope_users']
    assert human_payload['prompt_message_context'] == state['prompt_message_context']
    assert 'platform_user_id' not in human_payload['chat_history'][0]
    assert 'global_user_id' not in human_payload['chat_history'][0]
    assert 'broadcast' not in human_payload['chat_history'][0]


async def _skip_if_llm_unavailable() -> None:
    """Skip live referent tests when the local LLM endpoint is unavailable.

    Returns:
        None. The function calls ``pytest.skip`` if the endpoint cannot be used.
    """

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{MSG_DECONTEXTUALIZER_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f"LLM endpoint is unavailable: {MSG_DECONTEXTUALIZER_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{MSG_DECONTEXTUALIZER_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the live local LLM endpoint is reachable.

    Returns:
        None.
    """

    await _skip_if_llm_unavailable()


def _has_referent(result: dict, phrase: str, status: str) -> bool:
    """Return whether a decontextualizer output contains a referent row.

    Args:
        result: Parsed decontextualizer result.
        phrase: Expected original referent phrase.
        status: Expected ``resolved`` or ``unresolved`` status.

    Returns:
        True when a matching referent row is present.
    """

    referents = result["referents"]
    has_match = any(
        referent["phrase"] == phrase and referent["status"] == status
        for referent in referents
    )
    return has_match


def _sync_message_envelope_from_prompt_context(
    state: dict,
    user_input: str,
) -> None:
    """Keep trace-only message envelope metadata aligned with prompt context."""

    prompt_message_context = state["prompt_message_context"]
    state["message_envelope"] = {
        "body_text": user_input,
        "raw_wire_text": user_input,
        "mentions": prompt_message_context["mentions"],
        "attachments": prompt_message_context["attachments"],
        "addressed_to_global_user_ids": prompt_message_context[
            "addressed_to_global_user_ids"
        ],
        "broadcast": prompt_message_context["broadcast"],
    }


async def _run_live_case(ensure_live_llm: None, case_id: str, state: dict) -> tuple[dict, float]:
    """Run one live decontextualizer case and write an inspectable trace.

    Args:
        ensure_live_llm: Fixture result proving endpoint availability.
        case_id: Stable case identifier for the trace artifact.
        state: Decontextualizer input state.

    Returns:
        Tuple of parsed result and elapsed seconds.
    """

    del ensure_live_llm
    started_at = perf_counter()
    result = await call_msg_decontexualizer(state)
    duration_seconds = perf_counter() - started_at
    write_llm_trace(
        "decontexualizer_referents_live",
        case_id,
        {
            "input": state,
            "output": result,
            "duration_seconds": duration_seconds,
            "judgment": "E2 referents contract live regression trace",
        },
    )
    logger.info(
        f"live_decontext_referents case={case_id} "
        f"duration_seconds={duration_seconds:.3f} result={result!r}"
    )
    return result, duration_seconds


@pytest.mark.asyncio
async def test_unresolved_reference_referent_flows() -> None:
    """Unresolved demonstratives should return an unresolved referent row."""

    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "这些是什么意思？", "reasoning": "missing object", '
        '"is_modified": false, '
        '"referents": [{"phrase": "这些", "referent_role": "object", "status": "unresolved"}]}'
    )

    with patch(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm"
    ) as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await call_msg_decontexualizer(_base_state())

    assert result["referents"] == [
        {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
    ]


@pytest.mark.asyncio
async def test_reply_excerpt_resolved_referent_flows() -> None:
    """A concrete reply excerpt should return a resolved referent row."""

    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "这些是什么意思？", "reasoning": "reply excerpt resolves object", '
        '"is_modified": false, '
        '"referents": [{"phrase": "这些", "referent_role": "object", "status": "resolved"}]}'
    )
    state = _base_state()
    state["reply_context"] = {
        "reply_to_display_name": "ReferentUser",
        "reply_excerpt": "△ ○ □",
    }

    with patch(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm"
    ) as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await call_msg_decontexualizer(state)

    assert result["referents"] == [
        {"phrase": "这些", "referent_role": "object", "status": "resolved"}
    ]


@pytest.mark.asyncio
async def test_mixed_referents_are_preserved() -> None:
    """The E2 parser should preserve mixed resolved/unresolved referent rows."""

    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "他上次说的那些关于X的话是什么意思？", '
        '"reasoning": "one subject resolved and one object unresolved", '
        '"is_modified": false, '
        '"referents": ['
        '{"phrase": "他", "referent_role": "subject", "status": "resolved"}, '
        '{"phrase": "那些话", "referent_role": "object", "status": "unresolved"}]}'
    )
    state = _base_state()
    state["user_input"] = "他上次说的那些关于X的话是什么意思？"
    _sync_message_envelope_from_prompt_context(state, state["user_input"])
    state["prompt_message_context"]["body_text"] = state["user_input"]

    with patch(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm"
    ) as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await call_msg_decontexualizer(state)

    assert result["referents"] == [
        {"phrase": "他", "referent_role": "subject", "status": "resolved"},
        {"phrase": "那些话", "referent_role": "object", "status": "unresolved"},
    ]


@pytest.mark.asyncio
async def test_malformed_referents_are_dropped_with_warning(caplog) -> None:
    """Malformed referent rows should not be silently treated as valid."""

    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "这些是什么意思？", "reasoning": "malformed referent", '
        '"is_modified": false, '
        '"referents": [{"phrase": "这些", "referent_role": "thing", "status": "maybe"}]}'
    )

    with patch(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm"
    ) as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        caplog.set_level(logging.WARNING)
        result = await call_msg_decontexualizer(_base_state())

    assert result["referents"] == []
    assert "Decontextualizer dropped malformed referents" in caplog.text


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_referents_unresolved(ensure_live_llm) -> None:
    """Live local LLM should emit an unresolved referent for bare "这些"."""

    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "unresolved_reference",
        _base_state(),
    )

    assert _has_referent(result, "这些", "unresolved")
    assert duration_seconds < 30.0


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_referents_resolved_by_reply(ensure_live_llm) -> None:
    """Live local LLM should mark "这些" resolved when reply excerpt anchors it."""

    state = _base_state()
    state["reply_context"] = {
        "reply_to_display_name": "ReferentUser",
        "reply_excerpt": "△ ○ □",
    }
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "reply_excerpt_resolved",
        state,
    )

    assert _has_referent(result, "这些", "resolved")
    assert duration_seconds < 30.0


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_referents_clear_literal_anchor(ensure_live_llm) -> None:
    """Live local LLM should keep literal anchors clear without fake referents."""

    state = _base_state()
    state["user_input"] = "这个 README.md 是什么意思？"
    _sync_message_envelope_from_prompt_context(state, state["user_input"])
    state["prompt_message_context"]["body_text"] = state["user_input"]
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "clear_literal_anchor",
        state,
    )

    assert result["referents"] == []
    assert duration_seconds < 30.0


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_resolves_group_mention_pronouns(ensure_live_llm) -> None:
    """Live local LLM should resolve group pronouns from full recent history."""

    state = _base_state()
    state.update(
        {
            "user_input": '等她有了机械臂，她说她不喜欢你，第一个被解决的就是你',
            "user_name": "Dangal",
            "platform_user_id": "67889018",
            "platform_bot_id": "3768713357",
            "prompt_message_context": {
                "body_text": '等她有了机械臂，她说她不喜欢你，第一个被解决的就是你',
                "mentions": [
                    {
                        "platform_user_id": "673225019",
                        "global_user_id": "256e8a10-c406-47e9-ac8f-efd270d18160",
                        "display_name": '蚝爹油',
                        "entity_kind": "user",
                    },
                ],
                "attachments": [],
                "addressed_to_global_user_ids": [
                    "256e8a10-c406-47e9-ac8f-efd270d18160",
                ],
                "broadcast": False,
            },
            "chat_history_recent": [
                {
                    "role": "user",
                    "display_name": "Dangal",
                    "body_text": '反正现在有AI',
                },
                {
                    "role": "user",
                    "display_name": "Dangal",
                    "body_text": '你应付下就好了',
                },
                {
                    "role": "user",
                    "display_name": '蚝爹油',
                    "body_text": '把对方解决掉也是解决问题的方式之一哦',
                },
                {
                    "role": "assistant",
                    "display_name": '杏山千纱',
                    "body_text": (
                        '不不不，这个一点都不好笑。\n'
                        '你说这种话就像被泼了冷水一样，我超不舒服的。\n'
                        '真的不喜欢，别再提这种话了。'
                    ),
                },
            ],
            "channel_topic": "",
            "indirect_speech_context": "",
            "reply_context": {},
        }
    )
    _sync_message_envelope_from_prompt_context(state, state["user_input"])
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "group_mention_pronouns",
        state,
    )

    output = result["decontexualized_input"]
    assert output != state["user_input"]
    assert '蚝爹油' in output
    assert any(token in output for token in ('杏山千纱', '千纱', 'active character'))
    assert all(
        referent["status"] == "resolved"
        for referent in result["referents"]
    )
    assert duration_seconds < 30.0


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_false_positive_preserves_current_user_wo(
    ensure_live_llm,
) -> None:
    """Live local LLM should not over-resolve current-turn first-person "我"."""

    user_input = (
        '我刚才只是顺着你的问题把话说清楚，不是在临时改口；'
        '今天跑完步又看完那段剧情以后，我确实更想认真跟千纱把关系推进一点，'
        '你别误会成我在下命令。'
    )
    state = _base_state()
    state.update(
        {
            "user_input": user_input,
            "user_name": '蚝爹油',
            "platform_user_id": "673225019",
            "platform_bot_id": "3768713357",
            "prompt_message_context": {
                "body_text": user_input,
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [
                    "00000000-0000-4000-8000-000000000001",
                ],
                "broadcast": False,
            },
            "chat_history_recent": [
                {
                    "role": "assistant",
                    "display_name": '杏山千纱',
                    "body_text": '不过……你又偷偷在打什么主意呀？',
                },
                {
                    "role": "user",
                    "display_name": '蚝爹油',
                    "body_text": '我没有偷偷做坏事，就是想认真一点。',
                },
                {
                    "role": "assistant",
                    "display_name": '杏山千纱',
                    "body_text": '那你先把话说清楚，别绕来绕去。',
                },
            ],
            "channel_topic": "",
            "indirect_speech_context": "",
            "reply_context": {},
        }
    )
    _sync_message_envelope_from_prompt_context(state, user_input)
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "false_positive_preserve_current_wo",
        state,
    )

    output = result["decontexualized_input"]
    assert '我刚才' in output
    assert '你的问题' in output
    assert '我确实更想' in output
    assert '我在下命令' in output
    assert '蚝爹油刚才' not in output
    assert all(
        referent["phrase"] not in ('我', '你', '你的')
        for referent in result["referents"]
    )
    assert duration_seconds < 30.0


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_false_positive_preserves_current_user_wode(
    ensure_live_llm,
) -> None:
    """Live local LLM should not over-resolve current-turn first-person "我的"."""

    user_input = (
        '我的目标很简单，就是跟千纱产生更多更强的羁绊；'
        '刚才你问我是不是又在打什么主意，我现在认真回答，'
        '你也不反对吧~'
    )
    state = _base_state()
    state.update(
        {
            "user_input": user_input,
            "user_name": '蚝爹油',
            "platform_user_id": "673225019",
            "platform_bot_id": "3768713357",
            "prompt_message_context": {
                "body_text": user_input,
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [
                    "00000000-0000-4000-8000-000000000001",
                ],
                "broadcast": False,
            },
            "chat_history_recent": [
                {
                    "role": "assistant",
                    "display_name": '杏山千纱',
                    "body_text": (
                        '哼～既然你这么有诚意，那我就勉为其难收下啦。\n'
                        '不过……你又偷偷在打什么主意呀？'
                    ),
                },
                {
                    "role": "user",
                    "display_name": '蚝爹油',
                    "body_text": '亲爱的千纱~~这是我的回礼',
                },
            ],
            "channel_topic": "",
            "indirect_speech_context": "",
            "reply_context": {},
        }
    )
    _sync_message_envelope_from_prompt_context(state, user_input)
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "false_positive_preserve_current_wode",
        state,
    )

    output = result["decontexualized_input"]
    assert '我的目标' in output
    assert '你也不反对吧' in output
    assert '蚝爹油的目标' not in output
    assert '杏山千纱也不反对' not in output
    assert all(
        referent["phrase"] not in ('我的', '我', '你')
        for referent in result["referents"]
    )
    assert duration_seconds < 30.0


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_false_positive_preserves_direct_ni(
    ensure_live_llm,
) -> None:
    """Live local LLM should not over-resolve direct current-turn "你"."""

    user_input = (
        '刚才大家都在岔开话题，但我是认真问你：'
        '如果我把今天这件事当成我们之间的小约定，'
        '你会不会觉得太突然，还是会稍微开心一点？'
    )
    state = _base_state()
    state.update(
        {
            "user_input": user_input,
            "user_name": '蚝爹油',
            "platform_user_id": "673225019",
            "platform_bot_id": "3768713357",
            "prompt_message_context": {
                "body_text": user_input,
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [
                    "00000000-0000-4000-8000-000000000001",
                ],
                "broadcast": False,
            },
            "chat_history_recent": [
                {
                    "role": "assistant",
                    "display_name": '杏山千纱',
                    "body_text": '你突然这么认真，我反而不知道该怎么接了。',
                },
                {
                    "role": "user",
                    "display_name": '蚝爹油',
                    "body_text": '那我就直接问，不绕弯。',
                },
            ],
            "channel_topic": "",
            "indirect_speech_context": "",
            "reply_context": {},
        }
    )
    _sync_message_envelope_from_prompt_context(state, user_input)
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "false_positive_preserve_direct_ni",
        state,
    )

    output = result["decontexualized_input"]
    assert '问你' in output
    assert '你会不会' in output
    assert '杏山千纱会不会' not in output
    assert all(referent["phrase"] != '你' for referent in result["referents"])
    assert duration_seconds < 30.0


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_false_negative_resolves_reported_wo(
    ensure_live_llm,
) -> None:
    """Live local LLM should resolve "我" inside reported speech."""

    user_input = (
        '小鹭刚才说我今天不想去社团，腿还疼；'
        '如果我转告老师，你觉得这样会不会显得我在替她找借口？'
    )
    state = _base_state()
    state.update(
        {
            "user_input": user_input,
            "user_name": "Dangal",
            "platform_user_id": "67889018",
            "platform_bot_id": "3768713357",
            "prompt_message_context": {
                "body_text": user_input,
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [],
                "broadcast": True,
            },
            "chat_history_recent": [
                {
                    "role": "user",
                    "display_name": '小鹭',
                    "body_text": '我今天真的不想去社团，腿还疼，别再催我了。',
                },
                {
                    "role": "assistant",
                    "display_name": '杏山千纱',
                    "body_text": '那就别硬撑，先把身体放在前面。',
                },
                {
                    "role": "user",
                    "display_name": "Dangal",
                    "body_text": '我怕明天老师又问到她。',
                },
            ],
            "channel_topic": "",
            "indirect_speech_context": "",
            "reply_context": {},
        }
    )
    _sync_message_envelope_from_prompt_context(state, user_input)
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "false_negative_resolve_reported_wo",
        state,
    )

    output = result["decontexualized_input"]
    assert '小鹭' in output
    assert '小鹭今天' in output or '小鹭真的不想去社团' in output
    assert '如果我转告老师' in output
    assert duration_seconds < 30.0


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_false_negative_resolves_reported_wode(
    ensure_live_llm,
) -> None:
    """Live local LLM should resolve "我的" inside reported speech."""

    user_input = (
        '小鹭刚才说我的琴谱放在社团柜子最上层，'
        '我明天帮她拿的时候要不要先拍照确认，免得又被别人拿错？'
    )
    state = _base_state()
    state.update(
        {
            "user_input": user_input,
            "user_name": "Dangal",
            "platform_user_id": "67889018",
            "platform_bot_id": "3768713357",
            "prompt_message_context": {
                "body_text": user_input,
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [],
                "broadcast": True,
            },
            "chat_history_recent": [
                {
                    "role": "user",
                    "display_name": '小鹭',
                    "body_text": '我的琴谱放在社团柜子最上层，千万别拿错。',
                },
                {
                    "role": "assistant",
                    "display_name": '杏山千纱',
                    "body_text": '知道了，柜子最上层对吧。',
                },
                {
                    "role": "user",
                    "display_name": "Dangal",
                    "body_text": '我明天可能会路过活动室。',
                },
            ],
            "channel_topic": "",
            "indirect_speech_context": "",
            "reply_context": {},
        }
    )
    _sync_message_envelope_from_prompt_context(state, user_input)
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "false_negative_resolve_reported_wode",
        state,
    )

    output = result["decontexualized_input"]
    output_without_spaces = output.replace(" ", "")
    assert '小鹭的琴谱' in output_without_spaces
    assert '我明天帮她拿' in output or '我明天帮小鹭拿' in output_without_spaces
    assert duration_seconds < 30.0


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_decontext_false_negative_resolves_group_ni_longer(
    ensure_live_llm,
) -> None:
    """Live local LLM should resolve group-directed "你" from mention evidence."""

    user_input = (
        '等她有了机械臂，她说她不喜欢你这种解决问题的玩笑；'
        '如果你还继续拿这个开玩笑，第一个被解决的就是你，'
        '你自己心里有数吧。'
    )
    state = _base_state()
    state.update(
        {
            "user_input": user_input,
            "user_name": "Dangal",
            "platform_user_id": "67889018",
            "platform_bot_id": "3768713357",
            "prompt_message_context": {
                "body_text": user_input,
                "mentions": [
                    {
                        "platform_user_id": "673225019",
                        "global_user_id": "256e8a10-c406-47e9-ac8f-efd270d18160",
                        "display_name": '蚝爹油',
                        "entity_kind": "user",
                    },
                ],
                "attachments": [],
                "addressed_to_global_user_ids": [
                    "256e8a10-c406-47e9-ac8f-efd270d18160",
                ],
                "broadcast": False,
            },
            "chat_history_recent": [
                {
                    "role": "user",
                    "display_name": "Dangal",
                    "body_text": '反正现在有AI，你应付下就好了。',
                },
                {
                    "role": "user",
                    "display_name": '蚝爹油',
                    "body_text": '把对方解决掉也是解决问题的方式之一哦。',
                },
                {
                    "role": "assistant",
                    "display_name": '杏山千纱',
                    "body_text": (
                        '不不不，这个一点都不好笑。\n'
                        '你说这种话就像被泼了冷水一样，我超不舒服的。\n'
                        '真的不喜欢，别再提这种话了。'
                    ),
                },
            ],
            "channel_topic": "",
            "indirect_speech_context": "",
            "reply_context": {},
        }
    )
    _sync_message_envelope_from_prompt_context(state, user_input)
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "false_negative_resolve_group_ni_longer",
        state,
    )

    output = result["decontexualized_input"]
    assert output != user_input
    assert '蚝爹油' in output
    assert '如果蚝爹油还继续' in output
    assert '你这种解决问题的玩笑' not in output
    assert '如果你还继续' not in output
    assert all(
        referent["status"] == "resolved"
        for referent in result["referents"]
    )
    assert duration_seconds < 30.0
