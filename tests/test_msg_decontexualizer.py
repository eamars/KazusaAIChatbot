"""Tests for persona_supervisor2_msg_decontexualizer.py — message decontextualization."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_msg_decontexualizer as decontextualizer_module,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import (
    call_msg_decontexualizer,
    multimedia_descriptor_agent,
    select_media_for_turn,
)

_FAILURE_INPUT = '等她有了机械臂，她说她不喜欢你，第一个被解决的就是你'
_RESOLVED_FAILURE_INPUT = (
    '等杏山千纱有了机械臂，杏山千纱说杏山千纱不喜欢蚝爹油，'
    '第一个被解决的就是蚝爹油'
)
_DANGAL_GLOBAL_USER_ID = '745d7818-a9d3-4889-b7f3-8555078a2061'
_TARGET_GLOBAL_USER_ID = '256e8a10-c406-47e9-ac8f-efd270d18160'
_CHARACTER_GLOBAL_USER_ID = '00000000-0000-4000-8000-000000000001'
_DANGAL_PLATFORM_USER_ID = '67889018'
_TARGET_PLATFORM_USER_ID = '673225019'
_BOT_PLATFORM_USER_ID = '3768713357'


def _base_state():
    """Minimal GlobalPersonaState for testing call_msg_decontexualizer."""
    return {
        "user_input": "他在干啥？",
        "user_name": "TestUser",
        "platform_user_id": "user_123",
        "platform_bot_id": "bot_456",
        "character_profile": {
            "name": "Character",
        },
        "message_envelope": {
            "body_text": "他在干啥？",
            "raw_wire_text": "他在干啥？",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "prompt_message_context": {
            "body_text": "他在干啥？",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "chat_history_recent": [
            {"name": "<speaker>", "user_id": "u1", "body_text": "The person mentioned earlier is cooking", "role": "user", "timestamp": "t1"},
        ],
        "channel_type": "group",
        "channel_name": "",
        "channel_topic": "general chat",
        "indirect_speech_context": "",
        "reply_context": {},
    }


def _multimedia_state() -> dict:
    """Build a minimal graph state for multimedia descriptor tests."""

    local_time_context = {
        "current_local_datetime": "2024-01-01 00:00",
        "current_local_weekday": "Monday",
    }
    state = {
        "storage_timestamp_utc": "2024-01-01T00:00:00Z",
        "local_time_context": local_time_context,
        "platform": "discord",
        "platform_message_id": "msg_123",
        "platform_user_id": "user_123",
        "global_user_id": "uuid-123",
        "user_name": "TestUser",
        "user_input": "Hello bot!",
        "user_multimedia_input": [],
        "user_profile": {"relationship_state": 500, "semantic_relationship_projection": ""},
        "platform_bot_id": "bot_456",
        "message_envelope": {
            "body_text": "Hello bot!",
            "raw_wire_text": "Hello bot!",
            "addressed_to_global_user_ids": [],
            "mentions": [],
            "attachments": [],
            "broadcast": True,
        },
        "prompt_message_context": {
            "body_text": "Hello bot!",
            "addressed_to_global_user_ids": [],
            "broadcast": True,
            "mentions": [],
            "attachments": [],
        },
        "character_profile": {
            "name": "Character",
            "global_user_id": "character-global-id",
            "mood": "neutral",
            "vibe_check": "calm",
        },
        "platform_channel_id": "chan_1",
        "channel_type": "group",
        "channel_name": "general",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "debug_modes": {},
    }
    state["cognitive_episode"] = build_text_chat_cognitive_episode(
        episode_id="episode-msg_123",
        percept_id="percept-msg_123-dialog",
        storage_timestamp_utc=state["storage_timestamp_utc"],
        local_time_context=local_time_context,
        user_input=state["user_input"],
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
        channel_type=state["channel_type"],
        platform_message_id=state["platform_message_id"],
        platform_user_id=state["platform_user_id"],
        global_user_id=state["global_user_id"],
        user_name=state["user_name"],
        debug_modes=state["debug_modes"],
        target_addressed_user_ids=[],
        target_broadcast=True,
    )
    return state


def test_select_media_for_turn_keeps_opening_and_newest_unique_images() -> None:
    """Media selection caps descriptions while exposing overflow."""

    rows = [
        {
            "content_type": "image/png",
            "base64_data": f"image-{index}",
            "description": "",
        }
        for index in range(5)
    ]
    rows.insert(2, dict(rows[1]))

    selected, additional_media_present = select_media_for_turn(rows)

    assert [row["base64_data"] for row in selected] == [
        "image-0",
        "image-2",
        "image-3",
        "image-4",
    ]
    assert additional_media_present is True


def _llm_response(content: str) -> MagicMock:
    """Return a small mock object shaped like a LangChain response."""

    response = MagicMock()
    response.content = content
    return response


def _qq_failure_history() -> list[dict]:
    """Return the pre-active QQ group rows needed to resolve the failure input."""

    history = [
        {
            'name': 'Dangal',
            'display_name': 'Dangal',
            'platform_message_id': 'qq-2026-05-08T01:48:45',
            'platform_user_id': _DANGAL_PLATFORM_USER_ID,
            'global_user_id': _DANGAL_GLOBAL_USER_ID,
            'role': 'user',
            'body_text': '反正现在有AI',
            'addressed_to_global_user_ids': [_CHARACTER_GLOBAL_USER_ID],
            'mentions': [],
            'broadcast': False,
            'reply_context': {},
            'timestamp': '2026-05-08T01:48:45.000000+00:00',
        },
        {
            'name': 'Dangal',
            'display_name': 'Dangal',
            'platform_message_id': 'qq-2026-05-08T01:48:52',
            'platform_user_id': _DANGAL_PLATFORM_USER_ID,
            'global_user_id': _DANGAL_GLOBAL_USER_ID,
            'role': 'user',
            'body_text': '你应付下就好了',
            'addressed_to_global_user_ids': [_CHARACTER_GLOBAL_USER_ID],
            'mentions': [],
            'broadcast': False,
            'reply_context': {},
            'timestamp': '2026-05-08T01:48:52.000000+00:00',
        },
        {
            'name': '蚝爹油',
            'display_name': '蚝爹油',
            'platform_message_id': 'qq-2026-05-08T01:48:58',
            'platform_user_id': _TARGET_PLATFORM_USER_ID,
            'global_user_id': _TARGET_GLOBAL_USER_ID,
            'role': 'user',
            'body_text': '把对方解决掉也是解决问题的方式之一哦',
            'addressed_to_global_user_ids': [_CHARACTER_GLOBAL_USER_ID],
            'mentions': [],
            'broadcast': False,
            'reply_context': {},
            'timestamp': '2026-05-08T01:48:58.000000+00:00',
        },
        {
            'name': '杏山千纱',
            'display_name': '杏山千纱',
            'platform_message_id': 'qq-2026-05-08T01:49:02',
            'platform_user_id': _BOT_PLATFORM_USER_ID,
            'global_user_id': _CHARACTER_GLOBAL_USER_ID,
            'role': 'assistant',
            'body_text': (
                '不不不，这个一点都不好笑。\n'
                '你说这种话就像被泼了冷水一样，我超不舒服的。\n'
                '真的不喜欢，别再提这种话了。'
            ),
            'addressed_to_global_user_ids': [_TARGET_GLOBAL_USER_ID],
            'mentions': [],
            'broadcast': False,
            'reply_context': {},
            'timestamp': '2026-05-08T01:49:02.000000+00:00',
        },
    ]
    return history


def _qq_failure_state(chat_history_recent: list[dict]) -> dict:
    """Build decontextualizer state for the QQ group referent failure."""

    state = _base_state()
    state.update(
        {
            'user_input': _FAILURE_INPUT,
            'user_name': 'Dangal',
            'platform_user_id': _DANGAL_PLATFORM_USER_ID,
            'platform_bot_id': _BOT_PLATFORM_USER_ID,
            'message_envelope': {
                'body_text': _FAILURE_INPUT,
                'raw_wire_text': (
                    f'[CQ:at,qq={_TARGET_PLATFORM_USER_ID}] '
                    f'{_FAILURE_INPUT}[CQ:image,file=referent.png]'
                ),
                'mentions': [
                    {
                        'platform_user_id': _TARGET_PLATFORM_USER_ID,
                        'global_user_id': _TARGET_GLOBAL_USER_ID,
                        'display_name': '蚝爹油',
                        'entity_kind': 'user',
                    },
                ],
                'attachments': [
                    {
                        'media_kind': 'image',
                        'description': '',
                        'summary_status': 'unavailable',
                    },
                ],
                'addressed_to_global_user_ids': [_TARGET_GLOBAL_USER_ID],
                'broadcast': False,
            },
            'prompt_message_context': {
                'body_text': _FAILURE_INPUT,
                'mentions': [
                    {
                        'platform_user_id': _TARGET_PLATFORM_USER_ID,
                        'global_user_id': _TARGET_GLOBAL_USER_ID,
                        'display_name': '蚝爹油',
                        'entity_kind': 'user',
                    },
                ],
                'attachments': [
                    {
                        'media_kind': 'image',
                        'description': '',
                        'summary_status': 'unavailable',
                    },
                ],
                'addressed_to_global_user_ids': [_TARGET_GLOBAL_USER_ID],
                'broadcast': False,
            },
            'chat_history_recent': chat_history_recent,
            'channel_topic': '',
            'indirect_speech_context': '',
            'reply_context': {},
        }
    )
    return state


class _HistoryAwareDecontextualizerLLM:
    """Fake LLM that resolves only when the needed QQ exchange is visible."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def ainvoke(self, messages: list, *, config=None) -> MagicMock:
        input_payload = json.loads(messages[1].content)
        self.payloads.append(input_payload)
        history_lines = input_payload['chat_history']

        if any('真的不喜欢' in line for line in history_lines):
            content = {
                'output': _RESOLVED_FAILURE_INPUT,
                'reasoning': 'visible group exchange identifies speaker and target',
                'is_modified': True,
                'referents': [
                    {
                        'phrase': '她',
                        'referent_role': 'subject',
                        'status': 'resolved',
                    },
                    {
                        'phrase': '你',
                        'referent_role': 'object',
                        'status': 'resolved',
                    },
                ],
            }
        else:
            content = {
                'output': _FAILURE_INPUT,
                'reasoning': 'filtered history lacks the group exchange',
                'is_modified': False,
                'referents': [],
            }

        response = MagicMock()
        response.content = json.dumps(content, ensure_ascii=False)
        return response


def test_vision_descriptor_prompt_declares_structured_prompt_sections() -> None:
    """Vision prompt should expose input, generation, and output contracts."""

    prompt = decontextualizer_module._VISION_DESCRIPTOR_PROMPT

    assert '# 输入格式' in prompt
    assert '# 生成步骤' in prompt
    assert '# 输出格式' in prompt


@pytest.mark.asyncio
async def test_multimedia_descriptor_updates_prompt_context_and_current_row(
    monkeypatch,
) -> None:
    """Image summaries should feed prompt context and current-row persistence."""

    state = _multimedia_state()
    state["message_envelope"]["body_text"] = ""
    state["message_envelope"]["attachments"] = [{
        "media_type": "image/jpeg",
        "base64_data": "image-bytes",
        "storage_shape": "inline",
    }]
    state["user_multimedia_input"] = [{
        "content_type": "image/jpeg",
        "base64_data": "image-bytes",
        "description": "",
    }]
    update_descriptions = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer.update_conversation_attachment_descriptions",
        update_descriptions,
    )

    response = _llm_response('{"description": "a desk with handwritten notes"}')
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._vision_descriptor_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await multimedia_descriptor_agent(state)

    assert result["user_multimedia_input"][0]["description"] == (
        "a desk with handwritten notes"
    )
    assert result["prompt_message_context"]["attachments"][0]["description"] == (
        "a desk with handwritten notes"
    )
    update_descriptions.assert_awaited_once_with(
        platform="discord",
        platform_channel_id="chan_1",
        platform_message_id="msg_123",
        descriptions=["a desk with handwritten notes"],
    )


@pytest.mark.asyncio
async def test_multimedia_descriptor_continues_when_vision_llm_fails(
    monkeypatch,
    caplog,
) -> None:
    """Vision descriptor failures should leave unavailable prompt summaries."""

    state = _multimedia_state()
    state["message_envelope"]["body_text"] = ""
    state["message_envelope"]["attachments"] = [{
        "media_type": "image/jpeg",
        "base64_data": "image-bytes",
        "storage_shape": "inline",
    }]
    state["user_multimedia_input"] = [{
        "content_type": "image/jpeg",
        "base64_data": "image-bytes",
        "description": "",
    }]
    update_descriptions = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer.update_conversation_attachment_descriptions",
        update_descriptions,
    )

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._vision_descriptor_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("vision down"))
        caplog.set_level(logging.WARNING)
        result = await multimedia_descriptor_agent(state)

    assert result["user_multimedia_input"][0]["description"] == ""
    assert result["prompt_message_context"]["attachments"][0] == {
        "media_kind": "image",
        "description": "",
        "summary_status": "unavailable",
    }
    update_descriptions.assert_not_awaited()
    assert "Image descriptor fallback after LLM exception" in caplog.text
    assert "vision down" in caplog.text

@pytest.mark.asyncio
async def test_decontexualizer_filtered_history_recreates_group_referent_loss():
    """Filtered history reproduces the logged empty-referents failure mode."""

    state = _qq_failure_state(_qq_failure_history()[:2])
    fake_llm = _HistoryAwareDecontextualizerLLM()

    with patch(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm",
        new=fake_llm,
    ):
        result = await call_msg_decontexualizer(state)

    history_lines = fake_llm.payloads[0]['chat_history']
    assert any('反正现在有AI' in line for line in history_lines)
    assert any('你应付下就好了' in line for line in history_lines)
    assert fake_llm.payloads[0]['prompt_message_context']['mentions'] == [
        {
            'platform_user_id': _TARGET_PLATFORM_USER_ID,
            'global_user_id': _TARGET_GLOBAL_USER_ID,
            'display_name': '蚝爹油',
            'entity_kind': 'user',
        },
    ]
    assert all(
        '真的不喜欢' not in line
        for line in fake_llm.payloads[0]['chat_history']
    )
    assert result == {
        'decontexualized_input': _FAILURE_INPUT,
        'referents': [],
    }


@pytest.mark.asyncio
async def test_decontexualizer_full_history_surfaces_group_referent_evidence():
    """Full recent history makes the third-party exchange visible to the stage."""

    state = _qq_failure_state(_qq_failure_history())
    fake_llm = _HistoryAwareDecontextualizerLLM()

    with patch(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm",
        new=fake_llm,
    ):
        result = await call_msg_decontexualizer(state)

    history_lines = fake_llm.payloads[0]['chat_history']
    assert any('把对方解决掉也是解决问题的方式之一哦' in line for line in history_lines)
    assert any('真的不喜欢' in line for line in history_lines)
    assert result == {
        'decontexualized_input': _RESOLVED_FAILURE_INPUT,
        'referents': [
            {
                'phrase': '她',
                'referent_role': 'subject',
                'status': 'resolved',
            },
            {
                'phrase': '你',
                'referent_role': 'object',
                'status': 'resolved',
            },
        ],
    }


@pytest.mark.asyncio
async def test_decontexualizer_returns_modified_input():
    """When LLM says is_modified=true, output should be the decontextualized text."""
    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "之前提到的那个人在干啥？", "reasoning": "resolved pronoun", '
        '"is_modified": true, '
        '"referents": [{"phrase": "他", "referent_role": "subject", "status": "resolved"}]}'
    )

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "之前提到的那个人在干啥？"
    assert result["referents"] == [
        {"phrase": "他", "referent_role": "subject", "status": "resolved"}
    ]


@pytest.mark.asyncio
async def test_decontexualizer_returns_original_when_not_modified():
    """When LLM says is_modified=false, output should be the original user_input."""
    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "他在干啥？", "reasoning": "already clear", '
        '"is_modified": false, '
        '"referents": []}'
    )

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "他在干啥？"
    assert result["referents"] == []


@pytest.mark.asyncio
async def test_decontexualizer_fallback_on_llm_error():
    """If LLM call raises, output falls back to original user_input."""
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "他在干啥？"
    assert result["referents"] == []


@pytest.mark.asyncio
async def test_decontexualizer_fallback_on_malformed_json():
    """If LLM returns garbage, output falls back to original user_input."""
    llm_response = MagicMock()
    llm_response.content = "not json at all"

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(_base_state())

    # parse_llm_json_output returns {} for garbage → is_modified defaults to False
    assert result["decontexualized_input"] == "他在干啥？"
    assert result["referents"] == []


@pytest.mark.asyncio
async def test_decontexualizer_forwards_reply_context_to_llm():
    """Reply metadata should be forwarded so reply-only follow-ups can be resolved."""
    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "是的，我是想让 active character 具体评价我。", '
        '"reasoning": "used reply excerpt", "is_modified": true, '
        '"referents": []}'
    )

    state = _base_state()
    state.update(
        {
            "user_input": "是的",
            "chat_history_recent": [
                {"role": "assistant", "body_text": "你是想让我怎么定义你呀？是想要一个具体的评价，还是仅仅在随口试探……唔。"},
                {"role": "user", "body_text": "要 active character 的具体评价"},
                {"role": "assistant", "body_text": "评价这种事……你是说，要我说明白对你的看法吗？唔……突然问这些，感觉胸口闷闷的。"},
            ],
            "message_envelope": {
                "body_text": "是的",
                "raw_wire_text": "是的",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": ["character-global"],
                "broadcast": False,
            },
            "prompt_message_context": {
                "body_text": "是的",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": ["character-global"],
                "broadcast": False,
            },
            "reply_context": {
                "reply_to_display_name": "<active character>",
                "reply_excerpt": "评价这种事……你是说，要我说明白对你的看法吗？唔……突然问这些，感觉胸口闷闷的。",
            },
        }
    )

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(state)

    payload = mock_llm.ainvoke.await_args.args[0][1].content
    system_prompt = mock_llm.ainvoke.await_args.args[0][0].content
    assert '"prompt_message_context": {' in payload
    assert '"message_envelope": {' not in payload
    assert "raw_wire_text" not in payload
    assert '"addressed_to_global_user_ids": ["character-global"]' in payload
    assert '"reply_excerpt": "评价这种事' in payload
    assert "# 输入格式" not in system_prompt
    assert "# 输入读取说明" not in system_prompt
    assert "# 本轮输入字段说明" not in system_prompt
    assert "# 来源与角色锚点" in system_prompt
    assert "prompt_message_context.body_text" in system_prompt
    assert "addressed_to_global_user_ids" in system_prompt
    assert "`broadcast`" in system_prompt
    assert "`mentions`" in system_prompt
    assert "prompt_message_context.attachments" in system_prompt
    assert "`summary_status`" in system_prompt
    assert "reply_context.reply_excerpt" in system_prompt
    assert result["decontexualized_input"] == "是的，我是想让 active character 具体评价我。"


def test_decontexualizer_prompt_explains_reply_ellipsis_decision_owner() -> None:
    """Prompt should anchor omitted decision questions to reply-source ownership."""

    system_prompt = decontextualizer_module._render_msg_decontexualizer_prompt(
        "Character",
    )

    assert "# 输入格式" not in system_prompt
    assert "# 输入读取说明" not in system_prompt
    assert "# 本轮输入字段说明" not in system_prompt
    assert '省略决策问题' in system_prompt
    assert '同时补出决策主体和动作对象' in system_prompt
    assert 'reply_context.reply_excerpt' in system_prompt
    assert '“帮你”标识当前用户是决策主体' in system_prompt
    assert '当前用户自己为被判断对象' in system_prompt
    assert '判断当前用户是否' in system_prompt
    assert '帮你看看' in system_prompt
    assert '要不要 / 该不该 / 值不值得' in system_prompt
    assert '第三方向当前用户发出邀请、通知、请求或建议' in system_prompt
    assert '附件描述、回复摘录和相邻历史可提供动作对象' in system_prompt


@pytest.mark.asyncio
async def test_decontexualizer_projects_group_name_into_channel_topic_text():
    """The LLM payload should receive scene text, not a new group-name field."""

    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "他在干啥？", "reasoning": "read scene", '
        '"is_modified": false, "referents": []}'
    )
    state = _base_state()
    state['channel_type'] = 'group'
    state['channel_name'] = '动画讨论群'
    state['channel_topic'] = '新番角色和剧情走向'

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        await call_msg_decontexualizer(state)

    payload = json.loads(mock_llm.ainvoke.await_args.args[0][1].content)
    assert payload['channel_topic'] == (
        '“动画讨论群”群聊中正在讨论：新番角色和剧情走向'
    )
    assert 'channel_name' not in payload
    assert state['channel_topic'] == '新番角色和剧情走向'


@pytest.mark.asyncio
async def test_decontexualizer_forwards_scope_users_as_neutral_identity_table():
    """Scoped users should reach the prompt as identity rows, not retry hints."""
    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "@杏山千纱 还不报警抓蚝爹油吗？", '
        '"reasoning": "used visible mention and identity table", '
        '"is_modified": true, '
        '"referents": [{"phrase": "他", "referent_role": "object", "status": "resolved"}]}'
    )
    scope_users = [
        {
            'display_name': '蚝爹油',
            'platform_user_id': _TARGET_PLATFORM_USER_ID,
            'global_user_id': _TARGET_GLOBAL_USER_ID,
            'aliases': [],
        },
        {
            'display_name': '杏山千纱',
            'platform_user_id': _BOT_PLATFORM_USER_ID,
            'global_user_id': _CHARACTER_GLOBAL_USER_ID,
            'aliases': [],
        },
    ]
    state = _base_state()
    state['scope_users'] = scope_users

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(state)

    messages = mock_llm.ainvoke.await_args.args[0]
    system_prompt = messages[0].content
    payload = json.loads(messages[1].content)
    assert payload['scope_users'] == scope_users
    assert all('retry' not in key for key in payload)
    assert 'scope_users' in system_prompt
    assert '身份' in system_prompt
    assert '其他来源已经桥接到某个可见身份' in system_prompt
    assert '稳定显示名' in system_prompt
    assert result == {
        'decontexualized_input': '@杏山千纱 还不报警抓蚝爹油吗？',
        'referents': [
            {'phrase': '他', 'referent_role': 'object', 'status': 'resolved'},
        ],
    }


@pytest.mark.asyncio
async def test_decontexualizer_parses_unresolved_reference_signal():
    """New ambiguity fields should flow through when the LLM marks unresolved."""
    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "这些是什么意思？", "reasoning": "missing referent", '
        '"is_modified": false, '
        '"referents": [{"phrase": "这些", "referent_role": "object", "status": "unresolved"}]}'
    )
    state = _base_state()
    state["user_input"] = "这些是什么意思？"
    state["message_envelope"]["body_text"] = state["user_input"]
    state["message_envelope"]["raw_wire_text"] = state["user_input"]
    state["prompt_message_context"]["body_text"] = state["user_input"]

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(state)

    assert result == {
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
        ],
    }


@pytest.mark.asyncio
async def test_decontexualizer_missing_new_fields_warns_and_defaults(caplog):
    """Missing ambiguity fields should warn and preserve old behavior."""
    llm_response = MagicMock()
    llm_response.content = '{"output": "他在干啥？", "reasoning": "legacy output", "is_modified": false}'

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        caplog.set_level(logging.WARNING)

        result = await call_msg_decontexualizer(_base_state())

    assert result["referents"] == []
    assert "Decontextualizer missing referent fields" in caplog.text
    assert "他在干啥？" in caplog.text


@pytest.mark.asyncio
async def test_decontexualizer_llm_exception_warns_with_input_preview(caplog):
    """LLM exception fallback should log at WARN with a user input preview."""
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        caplog.set_level(logging.WARNING)

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "他在干啥？"
    assert result["referents"] == []
    assert "Decontextualizer fallback after LLM exception" in caplog.text
    assert "LLM down" in caplog.text
    assert "他在干啥？" in caplog.text
