"""Tests for persona_supervisor2_msg_decontexualizer.py — message decontextualization."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import call_msg_decontexualizer


def _base_state():
    """Minimal GlobalPersonaState for testing call_msg_decontexualizer."""
    return {
        "user_input": "他在干啥？",
        "user_name": "TestUser",
        "platform_user_id": "user_123",
        "platform_bot_id": "bot_456",
        "chat_history_recent": [
            {"name": "Alice", "user_id": "u1", "content": "Bob is cooking", "role": "user", "timestamp": "t1"},
        ],
        "channel_topic": "general chat",
        "indirect_speech_context": "",
        "reply_context": {},
    }


@pytest.mark.asyncio
async def test_decontexualizer_returns_modified_input():
    """When LLM says is_modified=true, output should be the decontextualized text."""
    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "Bob在干啥？", "reasoning": "resolved pronoun", '
        '"is_modified": true, "reference_resolution_status": "resolved", '
        '"needs_clarification": false, "clarification_reason": ""}'
    )

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "Bob在干啥？"
    assert result["reference_resolution_status"] == "resolved"
    assert result["needs_clarification"] is False
    assert result["clarification_reason"] == ""


@pytest.mark.asyncio
async def test_decontexualizer_returns_original_when_not_modified():
    """When LLM says is_modified=false, output should be the original user_input."""
    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "他在干啥？", "reasoning": "already clear", '
        '"is_modified": false, "reference_resolution_status": "unchanged_clear", '
        '"needs_clarification": false, "clarification_reason": ""}'
    )

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "他在干啥？"
    assert result["reference_resolution_status"] == "unchanged_clear"
    assert result["needs_clarification"] is False


@pytest.mark.asyncio
async def test_decontexualizer_fallback_on_llm_error():
    """If LLM call raises, output falls back to original user_input."""
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "他在干啥？"
    assert result["reference_resolution_status"] == "unchanged_clear"
    assert result["needs_clarification"] is False


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
    assert result["reference_resolution_status"] == "unchanged_clear"
    assert result["needs_clarification"] is False


@pytest.mark.asyncio
async def test_decontexualizer_forwards_reply_context_to_llm():
    """Reply metadata should be forwarded so reply-only follow-ups can be resolved."""
    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "是的，我是想让千纱具体评价我。", '
        '"reasoning": "used reply excerpt", "is_modified": true, '
        '"reference_resolution_status": "resolved", '
        '"needs_clarification": false, "clarification_reason": ""}'
    )

    state = _base_state()
    state.update(
        {
            "user_input": "是的",
            "chat_history_recent": [
                {"role": "assistant", "content": "你是想让我怎么定义你呀？是想要一个具体的评价，还是仅仅在随口试探……唔。"},
                {"role": "user", "content": "要千纱的具体评价"},
                {"role": "assistant", "content": "评价这种事……你是说，要我说明白对你的看法吗？唔……突然问这些，感觉胸口闷闷的。"},
            ],
            "reply_context": {
                "reply_to_current_bot": True,
                "reply_to_display_name": "杏山千纱 (Kyōyama Kazusa)",
                "reply_excerpt": "评价这种事……你是说，要我说明白对你的看法吗？唔……突然问这些，感觉胸口闷闷的。",
            },
        }
    )

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(state)

    payload = mock_llm.ainvoke.await_args.args[0][1].content
    assert '"reply_to_current_bot": true' in payload
    assert '"reply_excerpt": "评价这种事' in payload
    assert result["decontexualized_input"] == "是的，我是想让千纱具体评价我。"


@pytest.mark.asyncio
async def test_decontexualizer_parses_unresolved_reference_signal():
    """New ambiguity fields should flow through when the LLM marks unresolved."""
    llm_response = MagicMock()
    llm_response.content = (
        '{"output": "这些是什么意思？", "reasoning": "missing referent", '
        '"is_modified": false, '
        '"reference_resolution_status": "unresolved_reference", '
        '"needs_clarification": true, '
        '"clarification_reason": "缺少这些指代的具体对象"}'
    )
    state = _base_state()
    state["user_input"] = "这些是什么意思？"

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await call_msg_decontexualizer(state)

    assert result == {
        "decontexualized_input": "这些是什么意思？",
        "reference_resolution_status": "unresolved_reference",
        "needs_clarification": True,
        "clarification_reason": "缺少这些指代的具体对象",
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

    assert result["reference_resolution_status"] == "unchanged_clear"
    assert result["needs_clarification"] is False
    assert "Decontextualizer missing ambiguity fields" in caplog.text
    assert "他在干啥？" in caplog.text


@pytest.mark.asyncio
async def test_decontexualizer_llm_exception_warns_with_input_preview(caplog):
    """LLM exception fallback should log at WARN with a user input preview."""
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer._msg_decontexualizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        caplog.set_level(logging.WARNING)

        result = await call_msg_decontexualizer(_base_state())

    assert result["decontexualized_input"] == "他在干啥？"
    assert "Decontextualizer fallback after LLM exception" in caplog.text
    assert "LLM down" in caplog.text
    assert "他在干啥？" in caplog.text
