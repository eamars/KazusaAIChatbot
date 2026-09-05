"""Tests for persona_supervisor2_msg_decontextualizer.py — message decontextualization."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    build_goal_continuation_ref,
    build_tool_result_episode,
    build_user_message_episode,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_msg_decontextualizer as decontextualizer_module,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontextualizer import (
    call_msg_decontextualizer,
    multimedia_descriptor_agent,
    select_media_for_turn,
)
from kazusa_ai_chatbot.llm_tracing import failure_capsule

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


def _logical_turns(history_rows: list[dict]) -> list[dict[str, object]]:
    """Project row-shaped test evidence into complete logical turns."""

    turns: list[dict[str, object]] = []
    for index, row in enumerate(history_rows, start=1):
        role = str(row.get('role', 'user'))
        turns.append({
            'turn_id': f'test-turn-{index}',
            'role': role,
            'occurred_at': str(row.get('timestamp', f't{index}')),
            'display_name': str(
                row.get('display_name', row.get('name', role))
            ),
            'fragments': [str(row.get('body_text', ''))],
            'conversation_row_ids': [f'test-row-{index}'],
            'llm_trace_id': '',
            'platform_user_id': str(row.get('platform_user_id', '')),
            'global_user_id': str(row.get('global_user_id', '')),
            'addressed_to_global_user_ids': list(
                row.get('addressed_to_global_user_ids', [])
            ),
            'broadcast': bool(row.get('broadcast', False)),
            'reply_context': dict(row.get('reply_context', {})),
        })
    return turns


def _base_state():
    """Minimal GlobalPersonaState for testing call_msg_decontextualizer."""
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
        "ambient_logical_turns": _logical_turns([
            {"name": "<speaker>", "user_id": "u1", "body_text": "The person mentioned earlier is cooking", "role": "user", "timestamp": "t1"},
        ]),
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
        "ambient_logical_turns": [],
        "reply_context": {},
        "debug_modes": {},
    }
    state["cognitive_episode"] = build_user_message_episode(
        episode_id="episode-msg_123",
        origin={
            "platform": state["platform"],
            "platform_message_id": state["platform_message_id"],
        },
        target_scope={
            "platform": state["platform"],
            "platform_channel_id": state["platform_channel_id"],
            "channel_type": state["channel_type"],
            "current_platform_user_id": state["platform_user_id"],
            "current_global_user_id": state["global_user_id"],
            "current_display_name": state["user_name"],
            "target_addressed_user_ids": [],
            "target_broadcast": True,
        },
        dialog_percept={
            "schema_version": "percept.v1",
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": "percept-msg_123-dialog",
            "content": {
                "semantic_text": state["user_input"],
                "text": state["user_input"],
            },
            "observed_at": state["storage_timestamp_utc"],
        },
        media_percepts=[],
        evidence_refs=[],
        local_time_context=local_time_context,
        created_at=state["storage_timestamp_utc"],
        debug_controls=state["debug_modes"],
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


def _vision_descriptor_payload(description: str) -> dict[str, object]:
    """Return one exact-shape vision descriptor result."""

    return {
        "description": description,
        "visible_text": [],
        "salient_visual_facts": ["a desk and handwritten notes"],
        "spatial_or_scene_facts": ["the notes are on the desk"],
        "uncertainty": [],
    }


def _state_with_inline_image() -> dict:
    """Return a multimedia state containing one selected inline image."""

    base64_data = "aW1hZ2UtYnl0ZXM="
    state = _multimedia_state()
    state["message_envelope"]["body_text"] = ""
    state["message_envelope"]["attachments"] = [{
        "media_type": "image/jpeg",
        "base64_data": base64_data,
        "storage_shape": "inline",
    }]
    state["user_multimedia_input"] = [{
        "content_type": "image/jpeg",
        "base64_data": base64_data,
        "description": "",
    }]
    return state


def _decontextualizer_payload(
    *,
    output: str,
    reasoning: str,
    is_modified: bool,
    referents: list[dict[str, str]],
    role_explicit_content: str = "当前用户向当前角色发送当前输入。",
    response_operation: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a complete Chinese-context decontextualizer candidate."""

    if response_operation is None:
        response_operation = {
            "operation": "当前角色回应当前用户的当前输入",
            "response_owner_role": "当前角色",
            "response_content_provider_role": "无",
            "selection_required": False,
            "embedded_actor_role": "当前角色",
            "embedded_target_role": "当前用户",
        }
    return {
        "output": output,
        "role_explicit_content": role_explicit_content,
        "response_operation": response_operation,
        "reasoning": reasoning,
        "is_modified": is_modified,
        "referents": referents,
    }


def _decontextualizer_response(**kwargs: object) -> MagicMock:
    """Build one mock response with the complete decontextualizer contract."""

    response_text = json.dumps(
        _decontextualizer_payload(**kwargs),
        ensure_ascii=False,
    )
    response = _llm_response(response_text)
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
            'ambient_logical_turns': _logical_turns(chat_history_recent),
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
            content = _decontextualizer_payload(
                output=_RESOLVED_FAILURE_INPUT,
                reasoning='可见群聊交换明确了行动者和对象。',
                is_modified=True,
                referents=[
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
            )
        else:
            content = _decontextualizer_payload(
                output=_FAILURE_INPUT,
                reasoning='筛选后的历史缺少群聊交换。',
                is_modified=False,
                referents=[],
            )

        response = MagicMock()
        response.content = json.dumps(content, ensure_ascii=False)
        return response


def test_vision_descriptor_prompt_declares_structured_prompt_sections() -> None:
    """Vision prompt should expose input, generation, and output contracts."""

    prompt = decontextualizer_module._VISION_DESCRIPTOR_PROMPT

    assert '# 输入格式' in prompt
    assert '# 生成步骤' in prompt
    assert '# 输出格式' in prompt


































def test_decontextualizer_prompt_explains_reply_ellipsis_decision_owner() -> None:
    """Prompt should anchor omitted decision questions to reply-source ownership."""

    system_prompt = decontextualizer_module._render_msg_decontextualizer_prompt(
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
    assert '自由文本统一使用中文称谓' in system_prompt
    assert 'response_operation' in system_prompt
    assert '当前角色 | 当前用户 | 其他参与者 | 无' in system_prompt
    assert '四个角色字段只使用中文角色枚举' in system_prompt
    assert 'role_explicit_content` 与 `response_operation` 必须描述同一组角色方向' in system_prompt
    assert '当前用户继续辱骂当前角色' in system_prompt
    assert '当前用户直接对当前角色进行评价、命令或否定时' in system_prompt
    assert '外层回应动作和回应内嵌套动作分开判断' in system_prompt
    assert '当前用户会把选定的奖励给当前角色' in system_prompt
    assert '行动者和对象是两个独立字段' in system_prompt


def test_decontextualizer_prompt_names_response_content_provider_role() -> None:
    """The prompt exposes only the canonical reply-content provider key."""

    system_prompt = decontextualizer_module._render_msg_decontextualizer_prompt(
        "Character",
    )

    assert "response_content_provider_role" in system_prompt
    assert "selection_owner_role" not in system_prompt










