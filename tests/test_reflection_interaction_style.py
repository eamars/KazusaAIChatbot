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

    async def ainvoke(self, messages: list, *, config=None) -> SimpleNamespace:
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


def _interaction_style_sources_module():
    """Import the source-builder module under test."""

    from kazusa_ai_chatbot.reflection_cycle import interaction_style_sources

    return_value = interaction_style_sources
    return return_value


def _group_message(
    *,
    role: str,
    global_user_id: str,
    platform_user_id: str,
    body_text: str,
    addressed_to_global_user_ids: list[str] | None = None,
    broadcast: bool = False,
    reply_to_platform_user_id: str = "",
    reply_to_current_bot: bool | None = None,
    display_name: str = "Display Name",
) -> dict:
    """Build one group conversation row for source attribution tests."""

    reply_context: dict[str, object] = {}
    if reply_to_platform_user_id:
        reply_context["reply_to_platform_user_id"] = reply_to_platform_user_id
    if reply_to_current_bot is not None:
        reply_context["reply_to_current_bot"] = reply_to_current_bot

    message = {
        "platform": "qq",
        "platform_channel_id": "group-channel",
        "channel_type": "group",
        "role": role,
        "platform_user_id": platform_user_id,
        "global_user_id": global_user_id,
        "display_name": display_name,
        "body_text": body_text,
        "addressed_to_global_user_ids": addressed_to_global_user_ids or [],
        "broadcast": broadcast,
        "reply_context": reply_context,
        "timestamp": "2026-05-05T12:00:00+00:00",
    }
    return message


def _eligible_group_messages(
    *,
    target_global_user_id: str = "target-global",
    target_platform_user_id: str = "target-platform",
    character_global_user_id: str = "character-global",
    character_platform_user_id: str = "bot-platform",
    target_rows: int = 4,
    character_rows: int = 4,
    text_prefix: str = "target",
) -> list[dict]:
    """Build structurally attributed group rows above the source threshold."""

    messages: list[dict] = []
    for index in range(target_rows):
        messages.append(
            _group_message(
                role="user",
                global_user_id=target_global_user_id,
                platform_user_id=target_platform_user_id,
                body_text=f"{text_prefix} user text {index}",
                addressed_to_global_user_ids=[character_global_user_id],
                display_name=f"{text_prefix} user",
            )
        )
    for index in range(character_rows):
        messages.append(
            _group_message(
                role="assistant",
                global_user_id=character_global_user_id,
                platform_user_id=character_platform_user_id,
                body_text=f"{text_prefix} character text {index}",
                addressed_to_global_user_ids=[target_global_user_id],
                display_name="Character",
            )
        )
    return messages


def _hidden_reply_group_messages(
    *,
    target_global_user_id: str = "target-global",
    target_platform_user_id: str = "target-platform",
    character_global_user_id: str = "character-global",
    character_platform_user_id: str = "bot-platform",
) -> list[dict]:
    """Build group rows attributed only through reply-to platform ids."""

    messages: list[dict] = []
    for index in range(4):
        messages.append(
            _group_message(
                role="user",
                global_user_id=target_global_user_id,
                platform_user_id=target_platform_user_id,
                body_text=f"hidden user text {index}",
                reply_to_platform_user_id=character_platform_user_id,
            )
        )
    for index in range(4):
        messages.append(
            _group_message(
                role="assistant",
                global_user_id=character_global_user_id,
                platform_user_id=character_platform_user_id,
                body_text=f"hidden character text {index}",
                reply_to_platform_user_id=target_platform_user_id,
            )
        )
    return messages


def test_interaction_style_extractor_prompt_keeps_confidence_boundary() -> None:
    """Extractor prompt keeps daily confidence strict and overlay confidence soft."""

    prompt = interaction_style._INTERACTION_STYLE_EXTRACTOR_PROMPT
    retired_current_overlay_confidence = (
        '"confidence": "' + "low|medium|high|" + '"'
    )
    retired_output_overlay_confidence = (
        '"confidence": "' + "medium|high|" + '"'
    )

    assert '"daily_confidence": "medium|high"' in prompt
    assert retired_current_overlay_confidence not in prompt
    assert retired_output_overlay_confidence not in prompt
    assert '"confidence": "current confidence descriptor"' in prompt
    assert '"confidence": "confidence descriptor for this overlay"' in prompt


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
        "evidence_rows": [],
        "current_overlay": interaction_style.empty_interaction_style_overlay(),
    }


def test_group_user_style_sources_use_structural_targets_only() -> None:
    """Group sources use structural addressees and produce one target packet."""

    style_sources = _interaction_style_sources_module()
    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    messages = _eligible_group_messages()

    sources = style_sources.build_group_participant_user_style_sources(
        daily_doc=group_doc,
        messages=messages,
        character_global_user_id="character-global",
    )

    assert len(sources) == 1
    source = sources[0]
    assert source["source_kind"] == "group_participant_daily"
    assert source["global_user_id"] == "target-global"
    assert source["channel_type"] == "group"
    assert source["daily_confidence"] == "medium"
    assert source["source_reflection_run_ids"] == ["group-daily", "hourly-run-1"]
    roles = {row["role"] for row in source["evidence_rows"]}
    assert roles == {"target_user", "character"}


def test_group_user_style_sources_exclude_adjacent_user_and_broadcast_noise() -> None:
    """Adjacent users and unaddressed broadcasts stay out of target evidence."""

    style_sources = _interaction_style_sources_module()
    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    messages = _eligible_group_messages()
    messages.extend([
        _group_message(
            role="user",
            global_user_id="adjacent-global",
            platform_user_id="adjacent-platform",
            body_text="adjacent secret text",
            addressed_to_global_user_ids=[],
            display_name="Adjacent Display",
        ),
        _group_message(
            role="assistant",
            global_user_id="character-global",
            platform_user_id="bot-platform",
            body_text="broadcast noise text",
            addressed_to_global_user_ids=[],
            broadcast=True,
            display_name="Character",
        ),
    ])

    sources = style_sources.build_group_participant_user_style_sources(
        daily_doc=group_doc,
        messages=messages,
        character_global_user_id="character-global",
    )
    payload = style_sources.user_style_source_to_extractor_payload(
        source=sources[0],
        current_overlay=interaction_style.empty_interaction_style_overlay(),
    )
    payload_text = json.dumps(payload, ensure_ascii=False)

    assert len(sources) == 1
    assert "adjacent secret text" not in payload_text
    assert "broadcast noise text" not in payload_text
    assert "Adjacent Display" not in payload_text


def test_group_user_style_sources_exclude_other_assistant_rows() -> None:
    """Assistant evidence must belong to the active character identity."""

    style_sources = _interaction_style_sources_module()
    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    messages = _eligible_group_messages(character_rows=0)
    for index in range(4):
        messages.append(
            _group_message(
                role="assistant",
                global_user_id="other-assistant-global",
                platform_user_id="other-assistant-platform",
                body_text=f"other assistant text {index}",
                addressed_to_global_user_ids=["target-global"],
                display_name="Other Assistant",
            )
        )

    sources = style_sources.build_group_participant_user_style_sources(
        daily_doc=group_doc,
        messages=messages,
        character_global_user_id="character-global",
    )

    assert sources == []


def test_group_user_style_sources_accept_hidden_reply_target() -> None:
    """Hidden reply-to platform ids can provide the structural target proof."""

    style_sources = _interaction_style_sources_module()
    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    messages = _hidden_reply_group_messages()

    sources = style_sources.build_group_participant_user_style_sources(
        daily_doc=group_doc,
        messages=messages,
        character_global_user_id="character-global",
    )

    assert len(sources) == 1
    assert sources[0]["global_user_id"] == "target-global"
    evidence_text = json.dumps(sources[0]["evidence_rows"], ensure_ascii=False)
    assert "hidden user text" in evidence_text
    assert "hidden character text" in evidence_text


def test_group_user_style_source_thresholds_and_top_five_cap() -> None:
    """Group sources require strong evidence and cap to the top five users."""

    style_sources = _interaction_style_sources_module()
    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    source_specs = [
        ("user-1", "platform-1", 6, 6),
        ("user-2", "platform-2", 5, 5),
        ("user-3", "platform-3", 4, 5),
        ("user-4", "platform-4", 5, 3),
        ("user-5", "platform-5", 4, 4),
        ("user-6", "platform-6", 3, 5),
        ("under-threshold", "platform-7", 3, 2),
    ]
    messages: list[dict] = []
    for global_user_id, platform_user_id, target_rows, character_rows in source_specs:
        messages.extend(
            _eligible_group_messages(
                target_global_user_id=global_user_id,
                target_platform_user_id=platform_user_id,
                target_rows=target_rows,
                character_rows=character_rows,
                text_prefix=global_user_id,
            )
        )

    sources = style_sources.build_group_participant_user_style_sources(
        daily_doc=group_doc,
        messages=messages,
        character_global_user_id="character-global",
    )
    source_user_ids = [
        source["global_user_id"]
        for source in sources
    ]

    assert source_user_ids == ["user-1", "user-2", "user-3", "user-6", "user-5"]
    assert "user-4" not in source_user_ids
    assert "under-threshold" not in source_user_ids


def test_group_user_style_source_payload_hides_operational_metadata() -> None:
    """Extractor payload keeps only semantic fields and role/text evidence."""

    style_sources = _interaction_style_sources_module()
    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    messages = _eligible_group_messages()
    sources = style_sources.build_group_participant_user_style_sources(
        daily_doc=group_doc,
        messages=messages,
        character_global_user_id="character-global",
    )

    payload = style_sources.user_style_source_to_extractor_payload(
        source=sources[0],
        current_overlay=interaction_style.empty_interaction_style_overlay(),
    )
    payload_text = json.dumps(payload, ensure_ascii=False)

    assert set(payload) == {
        "channel_type",
        "daily_confidence",
        "conversation_quality_patterns",
        "synthesis_limitations",
        "evidence_rows",
        "current_overlay",
    }
    assert all(set(row) == {"role", "text"} for row in payload["evidence_rows"])
    assert "target-global" not in payload_text
    assert "target-platform" not in payload_text
    assert "source_reflection_run_ids" not in payload_text
    assert "addressed_to_global_user_ids" not in payload_text
    assert "reply_context" not in payload_text
    assert "Display Name" not in payload_text


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
    monkeypatch.setattr(
        interaction_style,
        "list_reflection_scope_messages",
        AsyncMock(return_value=[]),
        raising=False,
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
async def test_run_daily_interaction_style_update_writes_group_participant_user_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Daily style update writes group participant user-style overlays."""

    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    overlay = {
        "speech_guidelines": ["Use compact warmth."],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }
    list_messages = AsyncMock(return_value=_eligible_group_messages())
    monkeypatch.setattr(
        interaction_style.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[group_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "list_reflection_scope_messages",
        list_messages,
        raising=False,
    )
    monkeypatch.setattr(
        interaction_style,
        "get_group_channel_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_user_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "_run_interaction_style_extractor",
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

    assert result.processed_count == 1
    assert result.succeeded_count == 1
    assert result.skipped_count == 0
    upsert_group.assert_awaited_once()
    upsert_user.assert_awaited_once()
    assert upsert_user.await_args.kwargs["global_user_id"] == "target-global"
    assert upsert_user.await_args.kwargs["source_reflection_run_ids"] == [
        "group-daily",
        "hourly-run-1",
    ]


@pytest.mark.asyncio
async def test_group_participant_style_skips_already_applied_daily_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group participant style skips users already updated from the daily run."""

    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    existing_user_style = {
        "overlay": {
            "speech_guidelines": ["Use compact warmth."],
            "social_guidelines": [],
            "pacing_guidelines": [],
            "engagement_guidelines": [],
            "confidence": "medium",
        },
        "source_reflection_run_ids": ["group-daily", "hourly-run-1"],
    }
    overlay = {
        "speech_guidelines": ["Use compact warmth."],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }
    list_messages = AsyncMock(return_value=_eligible_group_messages())
    get_user_style = AsyncMock(return_value=existing_user_style)
    monkeypatch.setattr(
        interaction_style.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[group_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "list_reflection_scope_messages",
        list_messages,
        raising=False,
    )
    monkeypatch.setattr(
        interaction_style,
        "get_group_channel_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_user_style_image",
        get_user_style,
    )
    monkeypatch.setattr(
        interaction_style,
        "_run_interaction_style_extractor",
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

    assert result.processed_count == 1
    assert result.succeeded_count == 1
    assert result.skipped_count == 0
    upsert_group.assert_awaited_once()
    list_messages.assert_awaited_once()
    get_user_style.assert_awaited_once()
    upsert_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_channel_failure_does_not_suppress_participant_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected group-channel failure must not suppress participant style."""

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
        AsyncMock(return_value=[group_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_group_channel_style_image",
        AsyncMock(side_effect=RuntimeError("group branch unavailable")),
    )
    monkeypatch.setattr(
        interaction_style,
        "list_reflection_scope_messages",
        AsyncMock(return_value=_eligible_group_messages()),
        raising=False,
    )
    monkeypatch.setattr(
        interaction_style,
        "get_user_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "_run_interaction_style_extractor",
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

    assert result.processed_count == 1
    assert result.succeeded_count == 1
    assert result.failed_count == 0
    assert result.skipped_count == 0
    upsert_user.assert_awaited_once()
    upsert_group.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_participant_failure_preserves_group_channel_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected participant failure must not erase group-channel success."""

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
        AsyncMock(return_value=[group_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_group_channel_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        interaction_style,
        "list_reflection_scope_messages",
        AsyncMock(side_effect=RuntimeError("participant branch unavailable")),
        raising=False,
    )
    monkeypatch.setattr(
        interaction_style,
        "_run_interaction_style_extractor",
        AsyncMock(return_value=overlay),
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
    assert result.succeeded_count == 1
    assert result.failed_count == 0
    assert result.skipped_count == 0
    upsert_group.assert_awaited_once()


@pytest.mark.asyncio
async def test_later_group_participant_failure_preserves_prior_user_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later participant failure must not erase an earlier user write."""

    group_doc = _daily_doc(run_id="group-daily", channel_type="group")
    overlay = {
        "speech_guidelines": ["Use compact warmth."],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }
    messages = _eligible_group_messages(
        target_global_user_id="first-user",
        target_platform_user_id="first-platform",
        target_rows=5,
        character_rows=5,
        text_prefix="first",
    )
    messages.extend(
        _eligible_group_messages(
            target_global_user_id="second-user",
            target_platform_user_id="second-platform",
            target_rows=4,
            character_rows=4,
            text_prefix="second",
        )
    )
    existing_group_style = {
        "overlay": overlay,
        "source_reflection_run_ids": ["group-daily", "hourly-run-1"],
    }
    get_user_style = AsyncMock(
        side_effect=[None, RuntimeError("second user branch unavailable")]
    )
    monkeypatch.setattr(
        interaction_style.repository,
        "daily_channel_runs",
        AsyncMock(return_value=[group_doc]),
    )
    monkeypatch.setattr(
        interaction_style,
        "get_group_channel_style_image",
        AsyncMock(return_value=existing_group_style),
    )
    monkeypatch.setattr(
        interaction_style,
        "list_reflection_scope_messages",
        AsyncMock(return_value=messages),
        raising=False,
    )
    monkeypatch.setattr(
        interaction_style,
        "get_user_style_image",
        get_user_style,
    )
    monkeypatch.setattr(
        interaction_style,
        "_run_interaction_style_extractor",
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

    assert result.processed_count == 1
    assert result.succeeded_count == 1
    assert result.failed_count == 0
    assert result.skipped_count == 0
    assert upsert_user.await_count == 1
    assert upsert_user.await_args.kwargs["global_user_id"] == "first-user"
    upsert_group.assert_not_awaited()


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
    monkeypatch.setattr(
        interaction_style,
        "list_reflection_scope_messages",
        AsyncMock(return_value=[]),
        raising=False,
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
