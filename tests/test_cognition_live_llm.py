"""Retained live coverage for the pre-cognition decontextualizer boundary."""

from __future__ import annotations

import logging

import httpx
import pytest

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontextualizer import (
    call_msg_decontextualizer,
)
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured LLM endpoint cannot serve the live test."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError:
        pytest.skip(
            f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}"
        )

    if response.status_code >= 500:
        pytest.skip(
            "LLM endpoint returned server error "
            f"{response.status_code}: {COGNITION_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the configured live LLM endpoint is reachable."""

    await _skip_if_llm_unavailable()


def _debug_snapshot(label: str, payload: object) -> None:
    """Persist one human-readable snapshot for live contract review."""

    logger.info("%s => %r", label, payload)
    write_llm_trace(
        "cognition_live_llm",
        label,
        {
            "label": label,
            "payload": payload,
            "judgment": "snapshot_for_manual_live_llm_contract_review",
        },
    )


async def test_live_msg_decontextualizer_returns_non_empty_output(
    ensure_live_llm,
) -> None:
    state = {
        "character_profile": {"name": "Kazusa"},
        "user_input": "他今天是不是又在躲雨？",
        "user_name": "LiveDecontextUser",
        "platform_user_id": "live-user",
        "platform_bot_id": "live-bot",
        "prompt_message_context": {
            "body_text": "他今天是不是又在躲雨？",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
        "chat_history_recent": [
            {"role": "assistant", "content": "你说的是哪一位？"},
            {
                "role": "user",
                "content": "就是昨天在天台看书的那个同学。",
            },
        ],
        "channel_topic": "放学后的闲聊",
        "indirect_speech_context": "",
        "reply_context": {},
    }
    _debug_snapshot("decontext.input", state)
    result = await call_msg_decontextualizer(state)
    _debug_snapshot("decontext.output", result)

    decontextualized_input = result["decontextualized_input"]
    assert isinstance(decontextualized_input, str)
    assert decontextualized_input.strip()
