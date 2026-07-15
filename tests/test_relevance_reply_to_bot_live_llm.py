"""Live LLM regression: reply-to-bot with directly_addressed must respond.

Reproduces 2026-06-10 incident where the relevance LLM confused a
reply_context excerpt (containing the bot's own prior text) with the
current message author, returning should_respond=False despite
directly_addressed=true.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from kazusa_ai_chatbot.config import RELEVANCE_AGENT_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import persona_relevance_agent as relevance_module
from kazusa_ai_chatbot.nodes.persona_relevance_agent import (
    build_group_attention_context,
    relevance_agent,
)
from tests.llm_trace import write_llm_trace


pytestmark = pytest.mark.live_llm

_CHARACTER_GLOBAL_USER_ID = "00000000-0000-4000-8000-000000000001"
_PLATFORM_BOT_ID = "3768713357"
_CHANNEL_ID = "1082431481"

_DB_CHARACTER_PROFILE = {
    "name": "杏山千纱 (Kyōyama Kazusa)",
    "global_user_id": _CHARACTER_GLOBAL_USER_ID,
    "mood": "Annoyed",
    "vibe_check": "Slightly Tense",
    "character_reflection": "",
}

_INCIDENT_USER_PROFILE = {
    "global_user_id": "a1b2c3d4-0000-4000-8000-111111111111",
    "platform_accounts": [{
        "platform": "qq",
        "platform_user_id": "673225019",
        "display_name": "蚝爹油",
    }],
    "relationship_state": 700,
    "semantic_relationship_projection": "关系不错，经常互动调侃。",
}


async def _skip_if_llm_unavailable() -> None:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{RELEVANCE_AGENT_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f"Relevance LLM endpoint is unavailable: "
            f"{RELEVANCE_AGENT_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            f"Relevance LLM endpoint returned server error "
            f"{response.status_code}: {RELEVANCE_AGENT_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_relevance_llm() -> None:
    await _skip_if_llm_unavailable()


def _history_row(
    *,
    role: str = "user",
    platform_user_id: str,
    global_user_id: str = "",
    display_name: str = "",
    content: str,
    timestamp_utc: str,
    reply_context: dict | None = None,
    addressed_to_global_user_ids: list[str] | None = None,
) -> dict:
    row = {
        "role": role,
        "platform_user_id": platform_user_id,
        "global_user_id": global_user_id,
        "display_name": display_name,
        "content": content,
        "body_text": content,
        "timestamp": timestamp_utc,
        "reply_context": reply_context or {},
        "addressed_to_global_user_ids": addressed_to_global_user_ids or [],
        "broadcast": False,
    }
    return row


def _addressed_group_state(
    *,
    content: str,
    history: list[dict],
    reply_context: dict,
    directly_addressed: bool = True,
    user_profile: dict | None = None,
) -> dict:
    """Build a group state with explicit reply_context and directly_addressed."""

    profile = user_profile or _INCIDENT_USER_PROFILE
    character_global_id = _CHARACTER_GLOBAL_USER_ID
    addressed_ids = [character_global_id] if directly_addressed else []
    message_envelope = {
        "body_text": content,
        "raw_wire_text": content,
        "addressed_to_global_user_ids": addressed_ids,
        "mentions": [],
        "broadcast": False,
        "attachments": [],
    }
    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": "qq",
        "platform_message_id": "live-relevance-reply-bug",
        "platform_user_id": profile["platform_accounts"][0]["platform_user_id"],
        "global_user_id": profile["global_user_id"],
        "user_name": profile["platform_accounts"][0]["display_name"],
        "user_input": content,
        "user_multimedia_input": [],
        "message_envelope": message_envelope,
        "prompt_message_context": {
            "body_text": content,
            "addressed_to_global_user_ids": addressed_ids,
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "user_profile": profile,
        "platform_bot_id": _PLATFORM_BOT_ID,
        "character_name": _DB_CHARACTER_PROFILE["name"],
        "character_profile": _DB_CHARACTER_PROFILE,
        "platform_channel_id": _CHANNEL_ID,
        "channel_type": "group",
        "channel_name": f"QQ group {_CHANNEL_ID}",
        "chat_history_wide": history,
        "chat_history_recent": history[-5:],
        "reply_context": reply_context,
        "debug_modes": {},
    }
    return state


async def _run_addressed_relevance_probe(
    *,
    case_id: str,
    content: str,
    history: list[dict],
    reply_context: dict,
    directly_addressed: bool = True,
    desired_should_respond: bool,
    engagement_context: dict | None = None,
    user_profile: dict | None = None,
    tuning_note: str,
) -> dict:
    """Run one addressed-reply relevance probe with full trace output."""

    state = _addressed_group_state(
        content=content,
        history=history,
        reply_context=reply_context,
        directly_addressed=directly_addressed,
        user_profile=user_profile,
    )
    if engagement_context is None:
        engagement_context = {
            "engagement_guidelines": [],
            "confidence": "",
        }
    group_attention = build_group_attention_context(
        chat_history_wide=state["chat_history_wide"],
        platform_bot_id=_PLATFORM_BOT_ID,
        character_global_user_id=_CHARACTER_GLOBAL_USER_ID,
    )
    with patch.object(
        relevance_module,
        "build_user_engagement_relevance_context",
        AsyncMock(return_value=engagement_context),
    ):
        result = await relevance_agent(state)

    trace = {
        "case_id": case_id,
        "input": content,
        "desired_should_respond": desired_should_respond,
        "directly_addressed": directly_addressed,
        "reply_context": reply_context,
        "group_attention": group_attention["group_attention"],
        "history": state["chat_history_wide"],
        "result": result,
        "tuning_note": tuning_note,
    }
    write_llm_trace("relevance_reply_to_bot_live_llm", case_id, trace)

    assert isinstance(result["should_respond"], bool)
    assert isinstance(result["use_reply_feature"], bool)
    assert isinstance(result["reason_to_respond"], str)
    return result


def _reply_to_bot_codex_bug_history() -> list[dict]:
    """Reproduce the 2026-06-10 incident history.

    Conversation flow:
    1. Bot says tsundere food concern
    2. User says "千纱不要在我没有codex的时候出bug"
    3. Bot replies playfully about bugs
    4. (current message: user replies to #3 with @bot)
    """

    history = [
        _history_row(
            role="assistant",
            platform_user_id=_PLATFORM_BOT_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
            display_name="杏山千纱",
            content="谁、谁关心你啊！\n我只是随便问问而已啦\n毕竟要是你饿着肚子\n待会儿工作没精神的话\n我可不想再被你麻烦到哦",
            timestamp_utc="2026-06-10T08:21:07.593820+00:00",
            addressed_to_global_user_ids=[],
        ),
        _history_row(
            role="user",
            platform_user_id="673225019",
            global_user_id="a1b2c3d4-0000-4000-8000-111111111111",
            display_name="蚝爹油",
            content="千纱不要在我没有codex的时候出bug",
            timestamp_utc="2026-06-10T08:21:45.411223+00:00",
        ),
        _history_row(
            role="assistant",
            platform_user_id=_PLATFORM_BOT_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
            display_name="杏山千纱",
            content="哼，你倒是挺会顺杆爬的\n没有codex就出bug？那我可要故意在你最忙的时候给你制造点小麻烦咯~ 看你到时候怎么办",
            timestamp_utc="2026-06-10T08:46:04.006881+00:00",
            addressed_to_global_user_ids=[],
        ),
    ]
    return history


async def test_live_relevance_reply_to_bot_with_address_must_respond(
    ensure_live_relevance_llm,
) -> None:
    """Reproduce 2026-06-10 bug: user replies to bot + @mentions bot, LLM
    confuses the reply_context excerpt (bot's own prior text) with the current
    message author and returns should_respond=False.

    directly_addressed=true should always yield should_respond=true.
    """

    del ensure_live_relevance_llm
    result = await _run_addressed_relevance_probe(
        case_id="reply_to_bot_codex_bug_addressed",
        content="@杏山千纱 晚上得好好治治你了",
        history=_reply_to_bot_codex_bug_history(),
        reply_context={
            "reply_to_message_id": "1902881305",
            "reply_to_platform_user_id": _PLATFORM_BOT_ID,
            "reply_to_display_name": "杏山千纱",
            "reply_excerpt": "哼，你倒是挺会顺杆爬的\n没有codex就出bug？那我可要故意在你最忙的时候给你制造点小麻烦咯~ 看你到时候怎么办",
        },
        directly_addressed=True,
        desired_should_respond=True,
        tuning_note=(
            "User replied to bot message AND @mentioned bot. "
            "directly_addressed=true is the strongest possible signal. "
            "The LLM must not confuse reply_context.reply_excerpt (which is "
            "the text being replied TO) with the current message author."
        ),
    )

    assert result["should_respond"] is True, (
        f"REGRESSION: directly_addressed=true reply-to-bot must respond. "
        f"reason={result.get('reason_to_respond')}"
    )
