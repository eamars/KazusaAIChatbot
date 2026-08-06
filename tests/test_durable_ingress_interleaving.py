"""Deterministic tests for durable ingress receipts and interleaving queries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bson import ObjectId

from kazusa_ai_chatbot.db import conversation as conversation_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock


OWNER_RECEIVED_AT = "2026-04-29T00:19:00+00:00"
RESPONSE_CUTOFF = "2026-04-29T00:20:00+00:00"


def _user_row(
    *,
    platform: str = "qq",
    platform_channel_id: str = "chan-1",
    platform_message_id: str = "msg-1",
    received_at: str = OWNER_RECEIVED_AT,
    row_id: Any = "owner-id",
) -> dict[str, Any]:
    """Build one user-role conversation row for query fixtures."""

    return {
        "_id": row_id,
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "role": "user",
        "platform_message_id": platform_message_id,
        "received_at": received_at,
    }


def _matches(query: dict[str, Any], row: dict[str, Any]) -> bool:
    """Apply the exact filter operators used by the durable ingress query."""

    for key, condition in query.items():
        if key == "$or":
            if not any(_matches(branch, row) for branch in condition):
                return False
            continue
        value = row.get(key)
        if isinstance(condition, dict) and any(
            operator in condition
            for operator in ("$gt", "$gte", "$lte", "$lt")
        ):
            for operator, bound in condition.items():
                if operator == "$gt" and not (
                    value is not None and value > bound
                ):
                    return False
                if operator == "$lte" and not (
                    value is not None and value <= bound
                ):
                    return False
            continue
        if value != condition:
            return False
    return True


class _FakeConversationHistory:
    """Miniature conversation_history matcher for the query contract."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)
        self.queries: list[dict[str, Any]] = []

    async def find_one(
        self,
        query: dict[str, Any],
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
    ) -> dict[str, Any] | None:
        """Return the earliest inserted row matching the query."""

        self.queries.append(query)
        for row in self.rows:
            if _matches(query, row):
                return {"_id": row["_id"]}
        return None


async def _query_result(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]],
    owner_row_id: Any = "owner-id",
) -> bool:
    """Run the durable interleaving query against fixture rows."""

    collection = _FakeConversationHistory(rows)
    db = MagicMock()
    db.conversation_history = collection
    monkeypatch.setattr(
        conversation_module,
        "get_db",
        AsyncMock(return_value=db),
    )
    result = await conversation_module.has_inbound_after(
        platform="qq",
        platform_channel_id="chan-1",
        owner_row_id=str(owner_row_id),
        owner_received_at=OWNER_RECEIVED_AT,
        response_cutoff_received_at=RESPONSE_CUTOFF,
    )
    return result


@pytest.mark.asyncio
async def test_has_inbound_after_counts_same_author_later_row(
    monkeypatch,
) -> None:
    """A later same-author user row in the same channel counts."""

    result = await _query_result(
        monkeypatch,
        [
            _user_row(
                platform_message_id="owner",
                received_at=OWNER_RECEIVED_AT,
                row_id="owner-id",
            ),
            _user_row(
                platform_message_id="same-author-later",
                received_at="2026-04-29T00:19:30+00:00",
                row_id="later-id",
            ),
        ],
    )
    assert result is True


@pytest.mark.asyncio
async def test_has_inbound_after_counts_different_author_later_row(
    monkeypatch,
) -> None:
    """A later different-author user row still counts."""

    result = await _query_result(
        monkeypatch,
        [
            _user_row(
                platform_message_id="owner",
                received_at=OWNER_RECEIVED_AT,
                row_id="owner-id",
            ),
            _user_row(
                platform_message_id="other-author",
                received_at="2026-04-29T00:19:31+00:00",
                row_id="other-id",
            ),
        ],
    )
    assert result is True


@pytest.mark.asyncio
async def test_has_inbound_after_ignores_unrelated_channel(
    monkeypatch,
) -> None:
    """Rows in another platform or channel never promote the response."""

    result = await _query_result(
        monkeypatch,
        [
            _user_row(
                platform_message_id="owner",
                received_at=OWNER_RECEIVED_AT,
                row_id="owner-id",
            ),
            _user_row(
                platform="qq",
                platform_channel_id="other-channel",
                platform_message_id="other-channel-later",
                received_at="2026-04-29T00:19:30+00:00",
                row_id="other-channel-id",
            ),
            _user_row(
                platform="discord",
                platform_channel_id="chan-1",
                platform_message_id="other-platform-later",
                received_at="2026-04-29T00:19:30+00:00",
                row_id="other-platform-id",
            ),
        ],
    )
    assert result is False


@pytest.mark.asyncio
async def test_has_inbound_after_includes_cutoff_and_excludes_after_cutoff(
    monkeypatch,
) -> None:
    """Rows at the cutoff count; rows after the cutoff never do."""

    result = await _query_result(
        monkeypatch,
        [
            _user_row(
                platform_message_id="owner",
                received_at=OWNER_RECEIVED_AT,
                row_id="owner-id",
            ),
            _user_row(
                platform_message_id="at-cutoff",
                received_at=RESPONSE_CUTOFF,
                row_id="cutoff-id",
            ),
            _user_row(
                platform_message_id="after-cutoff",
                received_at="2026-04-29T00:20:01+00:00",
                row_id="after-cutoff-id",
            ),
        ],
    )
    assert result is True


@pytest.mark.asyncio
async def test_has_inbound_after_uses_insertion_order_for_equal_instants(
    monkeypatch,
) -> None:
    """Equal server instants resolve by insertion order, not strict equality."""

    collection = _FakeConversationHistory([
        _user_row(
            platform_message_id="owner",
            received_at=OWNER_RECEIVED_AT,
            row_id=ObjectId("665000000000000000000001"),
        ),
        _user_row(
            platform_message_id="same-instant-later",
            received_at=OWNER_RECEIVED_AT,
            row_id=ObjectId("665000000000000000000002"),
        ),
    ])
    db = MagicMock()
    db.conversation_history = collection
    monkeypatch.setattr(
        conversation_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    result = await conversation_module.has_inbound_after(
        platform="qq",
        platform_channel_id="chan-1",
        owner_row_id="665000000000000000000001",
        owner_received_at=OWNER_RECEIVED_AT,
        response_cutoff_received_at=RESPONSE_CUTOFF,
    )

    assert result is True
    main_query = collection.queries[-1]
    assert "$or" in main_query


@pytest.mark.asyncio
async def test_has_inbound_after_anchors_equal_instant_to_owner_row(
    monkeypatch,
) -> None:
    """An earlier equal-instant row cannot make the owner self-intervene."""

    result = await _query_result(
        monkeypatch,
        [
            _user_row(
                platform_message_id="earlier",
                received_at=OWNER_RECEIVED_AT,
                row_id="earlier-id",
            ),
            _user_row(
                platform_message_id="owner",
                received_at=OWNER_RECEIVED_AT,
                row_id="owner-id",
            ),
        ],
    )

    assert result is False


@pytest.mark.asyncio
async def test_has_inbound_after_fails_closed_when_owner_row_is_missing(
    monkeypatch,
) -> None:
    """No owner row means no durable interleaving evidence."""

    result = await _query_result(
        monkeypatch,
        [
            _user_row(
                platform_message_id="later",
                received_at="2026-04-29T00:19:30+00:00",
                row_id="later-id",
            ),
        ],
        owner_row_id="missing-owner-id",
    )

    assert result is False


@pytest.mark.asyncio
async def test_has_inbound_after_owner_alone_is_false(monkeypatch) -> None:
    """The owner's own receipt alone never counts as an intervening row."""

    result = await _query_result(
        monkeypatch,
        [
            _user_row(
                platform_message_id="owner",
                received_at=OWNER_RECEIVED_AT,
                row_id="owner-id",
            ),
        ],
    )
    assert result is False


@pytest.mark.asyncio
async def test_has_inbound_after_ignores_legacy_rows_without_received_at(
    monkeypatch,
) -> None:
    """Legacy rows without received_at are never counted as evidence."""

    legacy_row = _user_row(
        platform_message_id="legacy-later",
        received_at="",
        row_id="legacy-id",
    )
    legacy_row.pop("received_at")
    result = await _query_result(
        monkeypatch,
        [
            _user_row(
                platform_message_id="owner",
                received_at=OWNER_RECEIVED_AT,
                row_id="owner-id",
            ),
            legacy_row,
        ],
    )
    assert result is False


@pytest.mark.asyncio
async def test_save_conversation_receipt_commits_before_embedding_enrichment(
    monkeypatch,
) -> None:
    """The receipt row is committed before slow embedding enrichment runs."""

    db = MagicMock()
    insert_result = MagicMock()
    insert_result.inserted_id = ObjectId("665000000000000000000010")
    db.conversation_history.update_one = AsyncMock()
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)
    call_order: list[str] = []

    async def _recording_insert(_doc):
        call_order.append("insert")
        return insert_result

    async def _recording_embedding(_text):
        call_order.append("embedding")
        return [0.2, 0.3]

    db.conversation_history.insert_one = _recording_insert
    monkeypatch.setattr(
        conversation_module,
        "get_db",
        AsyncMock(return_value=db),
    )
    monkeypatch.setattr(
        conversation_module,
        "get_document_text_embedding",
        _recording_embedding,
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.cache2_runtime.get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )

    row_id = await conversation_module.save_conversation_receipt({
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "role": "user",
        "platform_message_id": "msg-1",
        "platform_user_id": "platform-user",
        "global_user_id": "user-1",
        "display_name": "User",
        "body_text": "hello",
        "raw_wire_text": "hello",
        "content_type": "text",
        "addressed_to_global_user_ids": ["character-global"],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "timestamp": "2026-04-29T00:18:00+00:00",
        "received_at": OWNER_RECEIVED_AT,
    })

    assert row_id == "665000000000000000000010"
    assert call_order == ["insert", "embedding"]
    db.conversation_history.update_one.assert_awaited_once()
    update_args = db.conversation_history.update_one.await_args
    assert update_args.args[1] == {"$set": {"embedding": [0.2, 0.3]}}
    event = runtime.invalidate.await_args.args[0]
    assert event.reason == "save_conversation_receipt"
    assert event.storage_timestamp_utc == "2026-04-29T00:18:00+00:00"


@pytest.mark.asyncio
async def test_save_conversation_receipt_requires_received_at() -> None:
    """A receipt without received_at fails closed before any insert."""

    with pytest.raises(ValueError, match="received_at"):
        await conversation_module.save_conversation_receipt({
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "role": "user",
            "platform_message_id": "msg-1",
            "platform_user_id": "platform-user",
            "global_user_id": "user-1",
            "display_name": "User",
            "body_text": "hello",
            "raw_wire_text": "hello",
            "content_type": "text",
            "addressed_to_global_user_ids": ["character-global"],
            "mentions": [],
            "broadcast": False,
            "attachments": [],
            "timestamp": "2026-04-29T00:18:00+00:00",
        })


@pytest.mark.asyncio
async def test_save_conversation_receipt_keeps_row_when_embedding_fails(
    monkeypatch,
) -> None:
    """Embedding failure does not erase the committed ingress receipt."""

    db = MagicMock()
    insert_result = MagicMock()
    insert_result.inserted_id = ObjectId("665000000000000000000011")
    db.conversation_history.insert_one = AsyncMock(return_value=insert_result)
    db.conversation_history.update_one = AsyncMock()
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=0)
    monkeypatch.setattr(
        conversation_module,
        "get_db",
        AsyncMock(return_value=db),
    )
    monkeypatch.setattr(
        conversation_module,
        "get_document_text_embedding",
        AsyncMock(side_effect=RuntimeError("embedding unavailable")),
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.cache2_runtime.get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )

    row_id = await conversation_module.save_conversation_receipt({
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "role": "user",
        "platform_message_id": "msg-embedding-failure",
        "platform_user_id": "platform-user",
        "global_user_id": "user-1",
        "display_name": "User",
        "body_text": "hello",
        "raw_wire_text": "hello",
        "content_type": "text",
        "addressed_to_global_user_ids": ["character-global"],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "timestamp": "2026-04-29T00:18:00+00:00",
        "received_at": OWNER_RECEIVED_AT,
    })

    assert row_id == "665000000000000000000011"
    db.conversation_history.insert_one.assert_awaited_once()
    db.conversation_history.update_one.assert_not_awaited()
    runtime.invalidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_conversation_row_llm_trace_id_updates_committed_row(
    monkeypatch,
) -> None:
    """Intake attaches the turn trace id to the precommitted receipt row."""

    db = MagicMock()
    db.conversation_history.update_one = AsyncMock()
    monkeypatch.setattr(
        conversation_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    updated = await conversation_module.update_conversation_row_llm_trace_id(
        row_id="665000000000000000000010",
        llm_trace_id="trace-1",
    )

    assert updated is True
    query = db.conversation_history.update_one.await_args.args[0]
    assert query["_id"] == ObjectId("665000000000000000000010")
    update = db.conversation_history.update_one.await_args.args[1]
    assert update == {"$set": {"llm_trace_id": "trace-1"}}


@pytest.mark.asyncio
async def test_get_user_message_by_platform_message_id_requires_user_role(
    monkeypatch,
) -> None:
    """Background source lookup is scoped to user-role rows."""

    db = MagicMock()
    db.conversation_history.find_one = AsyncMock(return_value={"role": "user"})
    monkeypatch.setattr(
        conversation_module,
        "get_db",
        AsyncMock(return_value=db),
    )

    row = await conversation_module.get_user_message_by_platform_message_id(
        platform="qq",
        platform_channel_id="chan-1",
        platform_message_id="message-1",
    )

    assert row == {"role": "user"}
    query = db.conversation_history.find_one.await_args.args[0]
    assert query["platform"] == "qq"
    assert query["platform_channel_id"] == "chan-1"
    assert query["role"] == "user"
    assert query["platform_message_id"] == "message-1"


def test_self_cognition_episode_has_no_original_source_reply_target() -> None:
    """Self-cognition episodes never gain original-source reply metadata."""

    from kazusa_ai_chatbot.cognition_episode import (
        build_self_cognition_episode,
    )

    turn_clock = build_turn_clock("2026-05-10 21:00:00")
    episode = build_self_cognition_episode(
        case={
            "case_id": "case-1",
            "source_case_kind": "internal_monologue",
            "target_scope": {
                "platform": "qq",
                "platform_channel_id": "chan-1",
                "channel_type": "private",
                "current_global_user_id": "user-1",
            },
            "privacy_scope": "private",
        },
        percepts=[{
            "schema_version": "percept.v1",
            "percept_kind": "internal_context",
            "source_kind": "internal_thought",
            "source_id": "case-1",
            "content": {"summary": "still unresolved"},
            "observed_at": turn_clock["storage_timestamp_utc"],
        }],
        evidence_refs=[],
        local_time_context=turn_clock["local_time_context"],
        created_at=turn_clock["storage_timestamp_utc"],
    )

    assert episode["trigger_source"] == "self_cognition"
    assert "source_message_id" not in episode["origin_metadata"]


@pytest.mark.asyncio
async def test_self_cognition_delivery_passes_candidate_reply_unchanged(
    monkeypatch,
) -> None:
    """Self-cognition delivery never resolves a durable original source."""

    from kazusa_ai_chatbot.self_cognition.delivery import (
        deliver_selected_speak,
    )

    handle_send_message = AsyncMock(return_value={
        "conversation_message_id": "conversation-001",
        "delivery_tracking_id": "delivery-001",
        "adapter_message_id": "adapter-001",
    })
    monkeypatch.setattr(
        "kazusa_ai_chatbot.self_cognition.delivery.handle_send_message",
        handle_send_message,
    )

    result = await deliver_selected_speak(
        text="Checking in now.",
        delivery_target={
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "private",
            "source_platform_channel_id": "chan-1",
            "source_global_user_id": "user-1",
            "target_global_user_id": "user-1",
            "source_message_id": "candidate-source-id",
            "guild_id": None,
            "bot_permission_role": "user",
            "source_channel_type": "private",
            "source_platform_bot_id": "bot-1",
        },
        character_profile={"name": "Current Character"},
        adapter_registry=object(),
        now=datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc),
        reply_to_msg_id="candidate-reply-id",
    )

    assert result["status"] == "sent"
    args = handle_send_message.await_args.args[0]
    assert args["reply_to_msg_id"] == "candidate-reply-id"
