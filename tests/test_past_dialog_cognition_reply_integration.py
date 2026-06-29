"""Direct reply integration tests for past-dialog cognition residual."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import pytest
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot import service


def _state() -> dict[str, Any]:
    return {
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "reply_context": {
            "reply_to_message_id": "message-past",
            "reply_to_platform_user_id": "platform-character",
            "reply_to_display_name": "Kazusa",
            "reply_excerpt": "I meant that the idea needed time.",
        },
        "prompt_message_context": {
            "body_text": "What did you mean there?",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
    }


def _assistant_row(**overrides: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "_id": "mongo-row-1",
        "conversation_row_id": "row-1",
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "role": "assistant",
        "platform_message_id": "message-past",
        "global_user_id": "character-1",
        "display_name": "Kazusa",
        "body_text": "I meant that the idea needed time.",
        "llm_trace_id": "trace-1",
        "timestamp": "2026-06-01T00:00:00Z",
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_complete_reply_metadata_still_loads_private_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    captured_queries: list[dict[str, str]] = []

    async def get_row(
        *,
        platform: str,
        platform_channel_id: str,
        platform_message_id: str,
    ) -> dict[str, Any]:
        captured_queries.append({
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "platform_message_id": platform_message_id,
        })
        return _assistant_row()

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        assert list(trace_ids) == ["trace-1"]
        return [{
            "trace_id": "trace-1",
            "stage_name": "l2a_conscious_framing",
            "sequence": 1,
            "parsed_output": {
                "internal_monologue": "private reply residual",
                "logical_stance": "earlier tentative stance",
                "character_intent": "clarify later if asked",
            },
            "created_at": "2026-06-01T00:00:01Z",
        }]

    state = _state()
    prompt_context_before = deepcopy(state["prompt_message_context"])
    monkeypatch.setattr(service, "get_conversation_by_platform_message_id", get_row)
    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    context = await service._load_reply_past_dialog_context(
        state,
        character_global_user_id="character-1",
    )

    assert captured_queries == [{
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "platform_message_id": "message-past",
    }]
    assert "private reply residual" in context
    assert state["prompt_message_context"] == prompt_context_before


@pytest.mark.asyncio
async def test_non_kazusa_reply_row_produces_no_residual_before_trace_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    async def get_row(
        *,
        platform: str,
        platform_channel_id: str,
        platform_message_id: str,
    ) -> dict[str, Any]:
        return _assistant_row(global_user_id="someone-else")

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        raise AssertionError(f"unexpected trace lookup: {list(trace_ids)}")

    monkeypatch.setattr(service, "get_conversation_by_platform_message_id", get_row)
    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    context = await service._load_reply_past_dialog_context(
        _state(),
        character_global_user_id="character-1",
    )

    assert context == ""


@pytest.mark.asyncio
async def test_reply_row_database_error_is_forgotten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def get_row(
        *,
        platform: str,
        platform_channel_id: str,
        platform_message_id: str,
    ) -> dict[str, Any]:
        del platform, platform_channel_id, platform_message_id
        raise PyMongoError("conversation store unavailable")

    monkeypatch.setattr(service, "get_conversation_by_platform_message_id", get_row)

    context = await service._load_reply_past_dialog_context(
        _state(),
        character_global_user_id="character-1",
    )

    assert context == ""


@pytest.mark.asyncio
async def test_missing_trace_id_and_empty_parsed_output_are_forgotten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    rows = [
        _assistant_row(llm_trace_id=""),
        _assistant_row(llm_trace_id="trace-empty"),
    ]
    trace_lookups: list[list[str]] = []

    async def get_row(
        *,
        platform: str,
        platform_channel_id: str,
        platform_message_id: str,
    ) -> dict[str, Any]:
        del platform, platform_channel_id, platform_message_id
        return rows.pop(0)

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        trace_lookups.append(list(trace_ids))
        return [{
            "trace_id": "trace-empty",
            "stage_name": "l2a_conscious_framing",
            "sequence": 1,
            "parsed_output": {},
            "created_at": "2026-06-01T00:00:01Z",
        }]

    monkeypatch.setattr(service, "get_conversation_by_platform_message_id", get_row)
    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    missing_trace_context = await service._load_reply_past_dialog_context(
        _state(),
        character_global_user_id="character-1",
    )
    empty_trace_context = await service._load_reply_past_dialog_context(
        _state(),
        character_global_user_id="character-1",
    )

    assert missing_trace_context == ""
    assert empty_trace_context == ""
    assert trace_lookups == [["trace-empty"]]
