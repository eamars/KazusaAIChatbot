"""Deterministic tests for reflection-run repository contracts."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.db import reflection_cycle as db_reflection_runs
from kazusa_ai_chatbot.reflection_cycle import repository as repository_module
from kazusa_ai_chatbot.reflection_cycle.models import (
    READONLY_REFLECTION_PROMPT_VERSION,
    REFLECTION_RUN_KIND_HOURLY,
    REFLECTION_STATUS_DRY_RUN,
    ReflectionLLMResult,
    ReflectionScopeInput,
)
from kazusa_ai_chatbot.reflection_cycle.prompts import build_skipped_hourly_result


def test_hourly_run_document_uses_parent_scope_and_source_refs() -> None:
    """Hourly persistence should keep raw scope metadata and source refs."""

    scope = _hourly_scope()
    llm_result = build_skipped_hourly_result(scope)

    document = repository_module.build_hourly_run_document(
        scope=scope,
        result=llm_result,
        status=REFLECTION_STATUS_DRY_RUN,
        attempt_count=0,
    )

    assert document["_id"] == document["run_id"]
    assert document["run_kind"] == REFLECTION_RUN_KIND_HOURLY
    assert document["status"] == REFLECTION_STATUS_DRY_RUN
    assert document["prompt_version"] == READONLY_REFLECTION_PROMPT_VERSION
    assert document["scope"] == {
        "scope_ref": "scope_channel",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
    }
    assert document["hour_start"] == "2026-05-04T10:00:00+00:00"
    assert document["character_local_date"] == "2026-05-04"
    assert document["source_message_refs"] == [
        {
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
            "role": "user",
            "timestamp": "2026-05-04T10:05:00+00:00",
        },
        {
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
            "role": "assistant",
            "timestamp": "2026-05-04T10:07:00+00:00",
        },
    ]
    assert document["output"]["active_character_utterances"] == [
        "The channel rule should stay factual, not personal."
    ]


def test_hourly_scope_parent_parser_uses_timestamp_suffix_only() -> None:
    """Hourly parent parsing should strip only the timestamp suffix."""

    parsed_scope = repository_module.channel_scope_ref_for_hourly(
        "scope_20_slug_20260504T10Z",
    )
    unchanged_scope = repository_module.channel_scope_ref_for_hourly(
        "scope_20_slug",
    )

    assert parsed_scope == "scope_20_slug"
    assert unchanged_scope == "scope_20_slug"


@pytest.mark.asyncio
async def test_repository_persists_through_db_interface(monkeypatch) -> None:
    """Repository writes must call the reflection DB interface only."""

    upsert = AsyncMock()
    monkeypatch.setattr(
        repository_module.reflection_store,
        "upsert_reflection_run",
        upsert,
    )
    document = repository_module.build_hourly_run_document(
        scope=_hourly_scope(),
        result=_hourly_result(_hourly_scope()),
        status=REFLECTION_STATUS_DRY_RUN,
        attempt_count=0,
    )

    await repository_module.upsert_run(document)

    upsert.assert_awaited_once_with(document)


def test_reflection_cycle_source_has_no_direct_storage_operations() -> None:
    """Reflection package code should not execute raw storage operations."""

    forbidden = (
        "database client helper",
        ".insert_one(",
        ".update_one(",
        ".update_many(",
        ".delete_one(",
        ".delete_many(",
        ".replace_one(",
        ".count_documents(",
        ".$vectorSearch",
    )
    modules = [
        repository_module,
    ]
    offenders: list[str] = []
    for module in modules:
        source = inspect.getsource(module)
        for token in forbidden:
            if token in source:
                offenders.append(f"{module.__name__}:{token}")

    assert offenders == []


@pytest.mark.asyncio
async def test_db_interface_upsert_sets_id_to_run_id(monkeypatch) -> None:
    """The DB interface should persist run id as both _id and run_id."""

    collection = _FakeRunCollection()
    db = {"character_reflection_runs": collection}
    monkeypatch.setattr(
        db_reflection_runs,
        "get_db",
        AsyncMock(return_value=db),
    )
    document = {
        "run_id": "run-1",
        "run_kind": "hourly_slot",
        "status": "dry_run",
        "prompt_version": "p1",
        "attempt_count": 0,
        "scope": {
            "scope_ref": "scope-1",
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
        },
    }

    await db_reflection_runs.upsert_reflection_run(document)

    assert collection.payload["_id"] == "run-1"
    assert collection.payload["run_id"] == "run-1"
    assert collection.filter_doc == {"run_id": "run-1"}
    assert collection.upsert is True


def _hourly_scope() -> ReflectionScopeInput:
    """Build an hourly scope fixture with source messages."""

    return_value = ReflectionScopeInput(
        scope_ref="scope_channel_20260504T10Z",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        assistant_message_count=1,
        user_message_count=1,
        total_message_count=2,
        first_timestamp="2026-05-04T10:05:00+00:00",
        last_timestamp="2026-05-04T10:07:00+00:00",
        messages=[
            {
                "role": "user",
                "body_text": "User-specific detail should not become lore.",
                "timestamp": "2026-05-04T10:05:00+00:00",
            },
            {
                "role": "assistant",
                "body_text": "The channel rule should stay factual, not personal.",
                "timestamp": "2026-05-04T10:07:00+00:00",
            },
        ],
    )
    return return_value


def _hourly_result(scope: ReflectionScopeInput) -> ReflectionLLMResult:
    """Build a parsed hourly result fixture."""

    result = build_skipped_hourly_result(scope)
    result.parsed_output = {
        "topic_summary": "The character separated channel rules from user facts.",
        "conversation_quality_feedback": ["Stay factual."],
        "privacy_notes": ["Avoid user details."],
        "confidence": "high",
    }
    return result


class _FakeRunCollection:
    """Small collection fake for DB-interface upsert tests."""

    def __init__(self) -> None:
        self.filter_doc = {}
        self.payload = {}
        self.upsert = False

    async def replace_one(self, filter_doc: dict, payload: dict, *, upsert: bool):
        """Capture one replacement call."""

        self.filter_doc = filter_doc
        self.payload = payload
        self.upsert = upsert
