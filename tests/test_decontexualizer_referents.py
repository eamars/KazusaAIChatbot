"""Tests for the decontextualizer referents migration path."""

from __future__ import annotations

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
    state["message_envelope"]["body_text"] = state["user_input"]
    state["message_envelope"]["raw_wire_text"] = state["user_input"]
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
    state["message_envelope"]["body_text"] = state["user_input"]
    state["message_envelope"]["raw_wire_text"] = state["user_input"]
    state["prompt_message_context"]["body_text"] = state["user_input"]
    result, duration_seconds = await _run_live_case(
        ensure_live_llm,
        "clear_literal_anchor",
        state,
    )

    assert result["referents"] == []
    assert duration_seconds < 30.0
