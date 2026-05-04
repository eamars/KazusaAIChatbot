"""Deterministic tests for read-only reflection-cycle runtime."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.db import conversation_reflection as db_reflection_module
from kazusa_ai_chatbot.reflection_cycle import selector as selector_module
from kazusa_ai_chatbot.reflection_cycle import runtime as runtime_module
from kazusa_ai_chatbot.reflection_cycle.models import (
    ReflectionInputSet,
    ReflectionScopeInput,
)


@pytest.mark.asyncio
async def test_collect_reflection_inputs_uses_db_interface_only(monkeypatch) -> None:
    """Selector should call DB interfaces instead of executing Mongo commands."""

    now = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    channel_row = _channel_row(
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="private",
        character_message_count=1,
    )
    list_channels = AsyncMock(return_value={
        "rows": [channel_row],
        "diagnostics": {
            "channel_query_elapsed_ms": 3,
            "channel_row_count": 1,
            "pipeline_summary": {"collection": "conversation_history"},
        },
    })
    list_messages = AsyncMock(return_value=_messages(channel_type="private"))
    explain_query = AsyncMock(return_value={"available": False, "reason": "mocked"})

    monkeypatch.setattr(
        selector_module,
        "list_recent_character_message_channels",
        list_channels,
    )
    monkeypatch.setattr(
        selector_module,
        "list_reflection_scope_messages",
        list_messages,
    )
    monkeypatch.setattr(
        selector_module,
        "explain_monitored_channel_query",
        explain_query,
    )

    input_set = await selector_module.collect_reflection_inputs(
        lookback_hours=24,
        now=now,
    )

    assert input_set.fallback_used is False
    assert len(input_set.selected_scopes) == 1
    assert input_set.selected_scopes[0].assistant_message_count == 1
    assert input_set.selected_scopes[0].user_message_count == 1
    assert input_set.query_diagnostics["requested_channel_query_elapsed_ms"] == 3
    list_channels.assert_awaited_once()
    list_messages.assert_awaited_once()
    explain_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_collect_reflection_inputs_records_fallback(monkeypatch) -> None:
    """Fallback use must be explicit in the collected input artifact model."""

    now = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    fallback_channel = _channel_row(
        platform="qq",
        platform_channel_id="chan-2",
        channel_type="group",
        character_message_count=2,
    )
    list_channels = AsyncMock(side_effect=[
        {
            "rows": [],
            "diagnostics": {
                "channel_query_elapsed_ms": 2,
                "channel_row_count": 0,
                "pipeline_summary": {"collection": "conversation_history"},
            },
        },
        {
            "rows": [fallback_channel],
            "diagnostics": {
                "channel_query_elapsed_ms": 4,
                "channel_row_count": 1,
                "pipeline_summary": {"collection": "conversation_history"},
            },
        },
    ])
    monkeypatch.setattr(
        selector_module,
        "list_recent_character_message_channels",
        list_channels,
    )
    monkeypatch.setattr(
        selector_module,
        "list_reflection_scope_messages",
        AsyncMock(return_value=_messages(channel_type="group")),
    )
    monkeypatch.setattr(
        selector_module,
        "explain_monitored_channel_query",
        AsyncMock(return_value={"available": False, "reason": "mocked"}),
    )

    input_set = await selector_module.collect_reflection_inputs(
        lookback_hours=24,
        now=now,
    )

    assert input_set.fallback_used is True
    assert "No monitored channel" in input_set.fallback_reason
    assert input_set.selected_scopes[0].channel_type == "group"
    assert "fallback_channel_query_elapsed_ms" in input_set.query_diagnostics


def test_selector_source_has_no_direct_mongo_execution() -> None:
    """Reflection selector must not cross into direct DB execution."""

    source = inspect.getsource(selector_module)

    assert "get_db" not in source
    assert ".aggregate(" not in source
    assert ".find(" not in source
    assert ".command(" not in source


def test_monitored_channel_selection_uses_latest_character_message() -> None:
    """Monitored channel selection should not require user activity counters."""

    rows = [
        _channel_row(
            platform="qq",
            platform_channel_id="older",
            channel_type="private",
            character_message_count=1,
            last_character_message_timestamp="2026-05-03T22:00:00+00:00",
        ),
        _channel_row(
            platform="qq",
            platform_channel_id="newer",
            channel_type="private",
            character_message_count=1,
            last_character_message_timestamp="2026-05-03T23:00:00+00:00",
        ),
        _channel_row(
            platform="qq",
            platform_channel_id="missing-last-time",
            channel_type="private",
            character_message_count=1,
            last_character_message_timestamp="",
        ),
    ]

    selected_rows = selector_module.select_monitored_channel_rows(rows)

    assert len(selected_rows) == 2
    assert selected_rows[0]["_id"]["platform_channel_id"] == "newer"
    assert selected_rows[1]["_id"]["platform_channel_id"] == "older"


@pytest.mark.asyncio
async def test_db_interface_lists_monitored_channel_rows_readonly(monkeypatch) -> None:
    """The DB interface should use read operations and no persistence calls."""

    db = MagicMock()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    db.conversation_history.aggregate.return_value = cursor
    monkeypatch.setattr(db_reflection_module, "get_db", AsyncMock(return_value=db))

    result = await db_reflection_module.list_recent_character_message_channels(
        start_timestamp="2026-05-03T00:00:00+00:00",
        end_timestamp="2026-05-04T00:00:00+00:00",
        limit=25,
    )

    assert result["rows"] == []
    db.conversation_history.aggregate.assert_called_once()
    pipeline = db.conversation_history.aggregate.call_args.args[0]
    assert pipeline[0]["$match"]["role"] == "assistant"
    assert "user_message_count" not in json.dumps(pipeline)
    assert "assistant_message_count" not in json.dumps(pipeline)
    db.conversation_history.insert_one.assert_not_called()
    db.conversation_history.update_one.assert_not_called()
    db.conversation_history.delete_many.assert_not_called()


@pytest.mark.asyncio
async def test_db_interface_uses_message_field_allowlist(monkeypatch) -> None:
    """Reflection message reads should request only prompt-needed fields."""

    db = MagicMock()
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.to_list = AsyncMock(return_value=[{
        "role": "user",
        "body_text": "Message with an image.",
        "attachments": [{"description": "A small sketch."}],
        "timestamp": "2026-05-03T00:00:00+00:00",
    }])
    db.conversation_history.find.return_value = cursor
    monkeypatch.setattr(db_reflection_module, "get_db", AsyncMock(return_value=db))

    messages = await db_reflection_module.list_reflection_scope_messages(
        platform="qq",
        platform_channel_id="chan-1",
        start_timestamp="2026-05-03T00:00:00+00:00",
        end_timestamp="2026-05-04T00:00:00+00:00",
        limit=120,
    )

    find_args = db.conversation_history.find.call_args.args
    projection = find_args[1]
    assert messages[0]["attachments"][0]["description"] == "A small sketch."
    assert projection == {
        "_id": 0,
        "platform": 1,
        "platform_channel_id": 1,
        "channel_type": 1,
        "role": 1,
        "platform_user_id": 1,
        "global_user_id": 1,
        "display_name": 1,
        "body_text": 1,
        "timestamp": 1,
        "attachments.description": 1,
    }
    assert "attachments.base64_data" not in projection
    cursor.sort.assert_called_once_with("timestamp", -1)
    cursor.limit.assert_called_once_with(120)


@pytest.mark.asyncio
async def test_runtime_writes_local_artifact_only(monkeypatch, tmp_path: Path) -> None:
    """Prompt-only runtime should write a local artifact and skip LLM calls."""

    input_set = ReflectionInputSet(
        lookback_hours=24,
        requested_start="2026-05-03T00:00:00+00:00",
        requested_end="2026-05-04T00:00:00+00:00",
        effective_start="2026-05-03T00:00:00+00:00",
        effective_end="2026-05-04T00:00:00+00:00",
        fallback_used=False,
        fallback_reason="",
        selected_scopes=[_scope_input("scope_private", "private")],
        query_diagnostics={"explain_summary": {"available": False}},
    )
    monkeypatch.setattr(
        runtime_module,
        "collect_reflection_inputs",
        AsyncMock(return_value=input_set),
    )

    result = await runtime_module.run_readonly_reflection_evaluation(
        lookback_hours=24,
        now=datetime(2026, 5, 4, tzinfo=timezone.utc),
        output_dir=str(tmp_path),
        use_real_llm=False,
    )

    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert result.artifact_path.parent == tmp_path
    assert payload["readonly"] is True
    assert payload["hourly_reflections"][0]["llm_skipped"] is True
    assert payload["daily_syntheses"][0]["llm_skipped"] is True
    assert len(result.channel_results) == 1
    assert len(result.daily_results) == 1


@pytest.mark.asyncio
async def test_runtime_splits_channel_into_active_hour_buckets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Runtime should reflect every message-bearing hour before daily synthesis."""

    messages = [
        _message("user", "2026-05-03T10:00:00+00:00"),
        _message("assistant", "2026-05-03T10:05:00+00:00"),
        _message("user", "2026-05-03T11:00:00+00:00"),
        _message("assistant", "2026-05-03T12:00:00+00:00"),
        _message("user", "2026-05-03T13:00:00+00:00"),
        _message("assistant", "2026-05-03T13:05:00+00:00"),
    ]
    input_set = ReflectionInputSet(
        lookback_hours=24,
        requested_start="2026-05-03T00:00:00+00:00",
        requested_end="2026-05-04T00:00:00+00:00",
        effective_start="2026-05-03T00:00:00+00:00",
        effective_end="2026-05-04T00:00:00+00:00",
        fallback_used=False,
        fallback_reason="",
        selected_scopes=[
            ReflectionScopeInput(
                scope_ref="scope_private",
                platform="qq",
                platform_channel_id="chan-1",
                channel_type="private",
                assistant_message_count=3,
                user_message_count=3,
                total_message_count=6,
                first_timestamp="2026-05-03T10:00:00+00:00",
                last_timestamp="2026-05-03T13:05:00+00:00",
                messages=messages,
            )
        ],
        query_diagnostics={},
    )
    monkeypatch.setattr(
        runtime_module,
        "collect_reflection_inputs",
        AsyncMock(return_value=input_set),
    )

    result = await runtime_module.run_readonly_reflection_evaluation(
        lookback_hours=24,
        now=datetime(2026, 5, 4, tzinfo=timezone.utc),
        output_dir=str(tmp_path),
        use_real_llm=False,
    )

    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    hour_starts = [
        item["hour_start"]
        for item in payload["hourly_reflections"]
    ]
    active_hour_slots = (
        result.daily_results[0].prompt.human_payload["active_hour_slots"]
    )
    assert len(result.hourly_results) == 4
    assert hour_starts == [
        "2026-05-03T10:00:00+00:00",
        "2026-05-03T11:00:00+00:00",
        "2026-05-03T12:00:00+00:00",
        "2026-05-03T13:00:00+00:00",
    ]
    assert len(active_hour_slots) == 4
    assert active_hour_slots[0]["hour"] == "2026-05-03T10:00:00+00:00"


def _channel_row(
    *,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    character_message_count: int,
    last_character_message_timestamp: str = "2026-05-03T23:30:00+00:00",
) -> dict:
    """Build one aggregate-like monitored channel row."""

    row = {
        "_id": {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "channel_type": channel_type,
        },
        "character_message_count": character_message_count,
        "first_character_message_timestamp": "2026-05-03T23:00:00+00:00",
        "last_character_message_timestamp": last_character_message_timestamp,
    }
    return row


def _messages(*, channel_type: str) -> list[dict]:
    """Build typed conversation rows for selector tests."""

    del channel_type
    messages = [
        {
            "role": "user",
            "platform_user_id": "user-platform-1",
            "global_user_id": "user-global-1",
            "display_name": "User One",
            "body_text": "Can you help me outline the project plan?",
            "attachments": [],
            "timestamp": "2026-05-03T23:00:00+00:00",
        },
        {
            "role": "assistant",
            "platform_user_id": "bot-platform",
            "global_user_id": "character-global",
            "display_name": "Character",
            "body_text": "I can split it into milestones.",
            "attachments": [],
            "timestamp": "2026-05-03T23:01:00+00:00",
        },
    ]
    return messages


def _message(role: str, timestamp: str) -> dict:
    """Build one timestamped reflection test message."""

    message = {
        "role": role,
        "platform_user_id": f"{role}-platform",
        "global_user_id": f"{role}-global",
        "display_name": role,
        "body_text": f"{role} message at {timestamp}",
        "attachments": [],
        "timestamp": timestamp,
    }
    return message


def _scope_input(scope_ref: str, channel_type: str) -> ReflectionScopeInput:
    """Build a scope input for runtime artifact tests."""

    scope = ReflectionScopeInput(
        scope_ref=scope_ref,
        platform="qq",
        platform_channel_id="chan-1",
        channel_type=channel_type,
        assistant_message_count=1,
        user_message_count=1,
        total_message_count=2,
        first_timestamp="2026-05-03T23:00:00+00:00",
        last_timestamp="2026-05-03T23:01:00+00:00",
        messages=_messages(channel_type=channel_type),
    )
    return scope
