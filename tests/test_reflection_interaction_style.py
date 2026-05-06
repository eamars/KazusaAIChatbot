from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.db import conversation_reflection as conversation_reflection_store
from kazusa_ai_chatbot.reflection_cycle import interaction_style


class _ConversationCursor:
    """Small async cursor fake for private-scope resolver tests."""

    def __init__(self, rows: list[dict]) -> None:
        """Create the cursor with rows returned by ``to_list``."""

        self._rows = rows

    async def to_list(self, length: int | None = None) -> list[dict]:
        """Return stored rows."""

        return_value = list(self._rows)
        return return_value


class _ConversationHistoryCollection:
    """Small fake collection exposing ``find`` for resolver tests."""

    def __init__(self, rows: list[dict]) -> None:
        """Create the collection with projected rows."""

        self.rows = rows
        self.last_query: dict | None = None
        self.last_projection: dict | None = None

    def find(self, query: dict, projection: dict) -> _ConversationCursor:
        """Record the query and return a fake cursor."""

        self.last_query = dict(query)
        self.last_projection = dict(projection)
        cursor = _ConversationCursor(self.rows)
        return cursor


class _ConversationDb:
    """Small DB fake for resolver tests."""

    def __init__(self, rows: list[dict]) -> None:
        """Create the DB fake with conversation rows."""

        self.conversation_history = _ConversationHistoryCollection(rows)


class _FakeStyleExtractorLlm:
    """Capture style-extractor payload and return a fixed overlay."""

    def __init__(self) -> None:
        """Create the fake LLM."""

        self.payload: dict | None = None

    async def ainvoke(self, messages: list) -> SimpleNamespace:
        """Capture the human JSON payload and return a valid overlay."""

        self.payload = json.loads(messages[1].content)
        content = json.dumps(
            {
                "overlay": {
                    "speech_guidelines": ["保持轻快但清楚的表达。"],
                    "social_guidelines": [],
                    "pacing_guidelines": [],
                    "engagement_guidelines": [],
                    "confidence": "medium",
                }
            },
            ensure_ascii=False,
        )
        return_value = SimpleNamespace(content=content)
        return return_value


def _daily_doc(
    *,
    run_id: str = "daily-run-1",
    channel_type: str = "private",
    confidence: str = "medium",
) -> dict:
    """Build a minimal daily reflection document."""

    return_value = {
        "run_id": run_id,
        "run_kind": "daily_channel",
        "status": "succeeded",
        "scope": {
            "scope_ref": "scope-1",
            "platform": "qq",
            "platform_channel_id": f"{channel_type}-channel",
            "channel_type": channel_type,
        },
        "window_start": "2026-05-05T00:00:00+00:00",
        "window_end": "2026-05-05T23:59:59+00:00",
        "character_local_date": "2026-05-05",
        "source_reflection_run_ids": ["hourly-run-1"],
        "output": {
            "day_summary": "private event summary must not reach extractor",
            "active_hour_summaries": [{"summary": "event detail"}],
            "conversation_quality_patterns": ["先承接情绪，再使用轻微调侃。"],
            "synthesis_limitations": ["小时槽有限。"],
            "confidence": confidence,
        },
    }
    return return_value


@pytest.mark.asyncio
async def test_resolve_single_private_scope_user_id_returns_only_unique_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver returns a sole non-character user id and hides message content."""

    db = _ConversationDb([
        {"global_user_id": "character-id"},
        {"global_user_id": "user-1"},
        {"global_user_id": "user-1"},
    ])
    monkeypatch.setattr(
        conversation_reflection_store,
        "get_db",
        AsyncMock(return_value=db),
    )

    global_user_id = await conversation_reflection_store.resolve_single_private_scope_user_id(
        platform="qq",
        platform_channel_id="private-1",
        start_timestamp="2026-05-05T00:00:00+00:00",
        end_timestamp="2026-05-05T23:59:59+00:00",
        character_global_user_id="character-id",
    )

    assert global_user_id == "user-1"
    assert db.conversation_history.last_query == {
        "platform": "qq",
        "platform_channel_id": "private-1",
        "channel_type": "private",
        "timestamp": {
            "$gte": "2026-05-05T00:00:00+00:00",
            "$lte": "2026-05-05T23:59:59+00:00",
        },
        "role": "user",
    }
    assert db.conversation_history.last_projection == {"_id": 0, "global_user_id": 1}


@pytest.mark.asyncio
async def test_resolve_single_private_scope_user_id_rejects_none_or_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver returns empty when the private scope is not exactly one user."""

    for rows in ([], [{"global_user_id": "user-1"}, {"global_user_id": "user-2"}]):
        db = _ConversationDb(rows)
        monkeypatch.setattr(
            conversation_reflection_store,
            "get_db",
            AsyncMock(return_value=db),
        )

        global_user_id = await conversation_reflection_store.resolve_single_private_scope_user_id(
            platform="qq",
            platform_channel_id="private-1",
            start_timestamp="2026-05-05T00:00:00+00:00",
            end_timestamp="2026-05-05T23:59:59+00:00",
            character_global_user_id="character-id",
        )

        assert global_user_id == ""


@pytest.mark.asyncio
async def test_extract_user_style_overlay_uses_only_safe_daily_signal_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extractor prompt excludes source ids, summaries, platform ids, and hours."""

    fake_llm = _FakeStyleExtractorLlm()
    monkeypatch.setattr(
        interaction_style,
        "_interaction_style_extractor_llm",
        fake_llm,
    )

    overlay = await interaction_style.extract_user_style_overlay_from_daily_reflection(
        daily_doc=_daily_doc(),
        current_overlay=interaction_style.empty_interaction_style_overlay(),
    )

    assert overlay["speech_guidelines"] == ["保持轻快但清楚的表达。"]
    assert fake_llm.payload == {
        "channel_type": "private",
        "daily_confidence": "medium",
        "conversation_quality_patterns": ["先承接情绪，再使用轻微调侃。"],
        "synthesis_limitations": ["小时槽有限。"],
        "current_overlay": interaction_style.empty_interaction_style_overlay(),
    }


@pytest.mark.asyncio
async def test_run_daily_interaction_style_update_writes_private_and_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Daily style update writes user and group-channel style images."""

    private_doc = _daily_doc(run_id="private-daily", channel_type="private")
    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    overlay = {
        "speech_guidelines": ["Use compact warmth."],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }
    monkeypatch.setattr(
        interaction_style.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[private_doc, group_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "resolve_single_private_scope_user_id",
        AsyncMock(return_value="user-1"),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_user_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_group_channel_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "extract_user_style_overlay_from_daily_reflection",
        AsyncMock(return_value=overlay),
    )
    monkeypatch.setattr(
        interaction_style,
        "extract_group_channel_style_overlay_from_daily_reflection",
        AsyncMock(return_value=overlay),
    )
    upsert_user = AsyncMock()
    upsert_group = AsyncMock()
    monkeypatch.setattr(interaction_style, "upsert_user_style_image", upsert_user)
    monkeypatch.setattr(
        interaction_style,
        "upsert_group_channel_style_image",
        upsert_group,
    )

    result = await interaction_style.run_daily_interaction_style_update(
        character_local_date="2026-05-05",
        dry_run=False,
        is_primary_interaction_busy=lambda: False,
    )

    assert result.processed_count == 2
    assert result.succeeded_count == 2
    assert result.skipped_count == 0
    upsert_user.assert_awaited_once()
    upsert_group.assert_awaited_once()
    assert upsert_user.await_args.kwargs["source_reflection_run_ids"] == [
        "private-daily",
        "hourly-run-1",
    ]


@pytest.mark.asyncio
async def test_run_daily_interaction_style_update_skips_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-confidence daily reflections do not invoke extraction."""

    low_doc = _daily_doc(confidence="low")
    extractor = AsyncMock()
    monkeypatch.setattr(
        interaction_style.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[low_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "extract_user_style_overlay_from_daily_reflection",
        extractor,
    )

    result = await interaction_style.run_daily_interaction_style_update(
        character_local_date="2026-05-05",
        dry_run=False,
        is_primary_interaction_busy=lambda: False,
    )

    assert result.processed_count == 0
    assert result.succeeded_count == 0
    assert result.skipped_count == 1
    extractor.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_daily_interaction_style_update_skips_empty_output_without_upsert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fully empty extractor output does not overwrite a prior overlay."""

    private_doc = _daily_doc()
    monkeypatch.setattr(
        interaction_style.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[private_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "resolve_single_private_scope_user_id",
        AsyncMock(return_value="user-1"),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_user_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "extract_user_style_overlay_from_daily_reflection",
        AsyncMock(return_value=interaction_style.empty_interaction_style_overlay()),
    )
    upsert_user = AsyncMock()
    monkeypatch.setattr(interaction_style, "upsert_user_style_image", upsert_user)

    result = await interaction_style.run_daily_interaction_style_update(
        character_local_date="2026-05-05",
        dry_run=False,
        is_primary_interaction_busy=lambda: False,
    )

    assert result.processed_count == 1
    assert result.succeeded_count == 0
    assert result.skipped_count == 1
    upsert_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_daily_interaction_style_update_skips_rejected_private_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanitizer rejection from LLM output skips without failing the pass."""

    private_doc = _daily_doc()
    monkeypatch.setattr(
        interaction_style.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[private_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "resolve_single_private_scope_user_id",
        AsyncMock(return_value="user-1"),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_user_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "extract_user_style_overlay_from_daily_reflection",
        AsyncMock(
            side_effect=ValueError(
                "social_guidelines contains source detail markers"
            )
        ),
    )
    upsert_user = AsyncMock()
    monkeypatch.setattr(interaction_style, "upsert_user_style_image", upsert_user)

    result = await interaction_style.run_daily_interaction_style_update(
        character_local_date="2026-05-05",
        dry_run=False,
        is_primary_interaction_busy=lambda: False,
    )

    assert result.processed_count == 1
    assert result.succeeded_count == 0
    assert result.failed_count == 0
    assert result.skipped_count == 1
    upsert_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_daily_interaction_style_update_skips_rejected_group_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group style sanitizer rejection skips without failing the pass."""

    group_doc = _daily_doc(channel_type="group")
    monkeypatch.setattr(
        interaction_style.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[group_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_group_channel_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "extract_group_channel_style_overlay_from_daily_reflection",
        AsyncMock(
            side_effect=ValueError(
                "social_guidelines contains source detail markers"
            )
        ),
    )
    upsert_group = AsyncMock()
    monkeypatch.setattr(
        interaction_style,
        "upsert_group_channel_style_image",
        upsert_group,
    )

    result = await interaction_style.run_daily_interaction_style_update(
        character_local_date="2026-05-05",
        dry_run=False,
        is_primary_interaction_busy=lambda: False,
    )

    assert result.processed_count == 1
    assert result.succeeded_count == 0
    assert result.failed_count == 0
    assert result.skipped_count == 1
    upsert_group.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_daily_interaction_style_update_skips_already_applied_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful style update is not reprocessed on later worker ticks."""

    private_doc = _daily_doc(run_id="private-daily")
    existing_style_doc = {
        "overlay": {
            "speech_guidelines": ["Use compact warmth."],
            "social_guidelines": [],
            "pacing_guidelines": [],
            "engagement_guidelines": [],
            "confidence": "medium",
        },
        "source_reflection_run_ids": ["private-daily", "hourly-run-1"],
    }
    monkeypatch.setattr(
        interaction_style.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[private_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "resolve_single_private_scope_user_id",
        AsyncMock(return_value="user-1"),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_user_style_image",
        AsyncMock(return_value=existing_style_doc),
    )
    extractor = AsyncMock()
    monkeypatch.setattr(
        interaction_style,
        "extract_user_style_overlay_from_daily_reflection",
        extractor,
    )
    upsert_user = AsyncMock()
    monkeypatch.setattr(interaction_style, "upsert_user_style_image", upsert_user)

    result = await interaction_style.run_daily_interaction_style_update(
        character_local_date="2026-05-05",
        dry_run=False,
        is_primary_interaction_busy=lambda: False,
    )

    assert result.processed_count == 1
    assert result.succeeded_count == 0
    assert result.skipped_count == 1
    extractor.assert_not_awaited()
    upsert_user.assert_not_awaited()
