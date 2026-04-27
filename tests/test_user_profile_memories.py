"""Unit tests for persistent user-profile memory helpers."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.db import MemoryType
from kazusa_ai_chatbot.dispatcher import (
    AdapterRegistry,
    PendingTaskIndex,
    RawToolCall,
    TaskDispatcher,
    ToolCallEvaluator,
    ToolRegistry,
    build_send_message_tool,
)
from kazusa_ai_chatbot.db import bootstrap as bootstrap_module
from kazusa_ai_chatbot.db import users as users_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_persistence as persistence_module
from scripts import migrate_user_profile_memories as migrate_module


class _Cursor:
    """Small async cursor double for chained Motor calls."""

    def __init__(self, docs: list[dict]):
        self.docs = docs

    def sort(self, *_args):
        return self

    def limit(self, *_args):
        return self

    def batch_size(self, *_args):
        return self

    async def to_list(self, length=None):
        if length is None:
            return list(self.docs)
        return list(self.docs[:length])

    def __aiter__(self):
        self._iter_index = 0
        return self

    async def __anext__(self):
        if self._iter_index >= len(self.docs):
            raise StopAsyncIteration
        doc = self.docs[self._iter_index]
        self._iter_index += 1
        return doc


def _mock_db():
    """Create a mock DB with a ``user_profile_memories`` collection."""
    db = MagicMock()
    db.user_profile_memories = MagicMock()
    return db


@pytest.mark.asyncio
async def test_insert_profile_memories_populates_embedding_and_expiry(monkeypatch):
    db = _mock_db()
    db.user_profile_memories.find_one = AsyncMock(return_value=None)
    db.user_profile_memories.insert_one = AsyncMock()
    db.user_profile_memories.update_many = AsyncMock()
    monkeypatch.setattr(users_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(users_module, "get_text_embedding", AsyncMock(return_value=[0.1, 0.2]))

    persisted = await users_module.insert_profile_memories(
        "u1",
        [{
            "memory_type": MemoryType.OBJECTIVE_FACT,
            "content": "User lives in Auckland",
            "created_at": "2026-04-25T00:00:00+00:00",
            "dedup_key": "user_location",
        }],
    )

    assert len(persisted) == 1
    doc = db.user_profile_memories.insert_one.call_args.args[0]
    assert doc["embedding"] == [0.1, 0.2]
    assert doc["expires_at"] > doc["created_at"]
    assert doc["global_user_id"] == "u1"


@pytest.mark.asyncio
async def test_insert_profile_memories_skips_duplicate_fact(monkeypatch):
    existing = {
        "memory_id": "m1",
        "memory_type": MemoryType.OBJECTIVE_FACT,
        "content": "User likes tea",
        "dedup_key": "tea",
    }
    db = _mock_db()
    db.user_profile_memories.find_one = AsyncMock(return_value=existing)
    db.user_profile_memories.insert_one = AsyncMock()
    monkeypatch.setattr(users_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(users_module, "get_text_embedding", AsyncMock(return_value=[0.1]))

    persisted = await users_module.insert_profile_memories(
        "u1",
        [{
            "memory_type": MemoryType.OBJECTIVE_FACT,
            "content": "User likes tea",
            "dedup_key": "tea",
        }],
    )

    assert persisted == [existing]
    db.user_profile_memories.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_insert_profile_memories_updates_existing_commitment(monkeypatch):
    db = _mock_db()
    db.user_profile_memories.find_one = AsyncMock(return_value={
        "memory_id": "existing-memory",
        "created_at": "2026-04-20T00:00:00+00:00",
    })
    db.user_profile_memories.update_one = AsyncMock()
    db.user_profile_memories.insert_one = AsyncMock()
    monkeypatch.setattr(users_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(users_module, "get_text_embedding", AsyncMock(return_value=[0.1]))

    persisted = await users_module.insert_profile_memories(
        "u1",
        [{
            "memory_type": MemoryType.COMMITMENT,
            "content": "Kazusa will reply in English",
            "action": "Kazusa will reply in English",
            "commitment_id": "c1",
            "dedup_key": "reply_language",
            "due_time": None,
        }],
    )

    assert persisted[0]["memory_id"] == "existing-memory"
    db.user_profile_memories.update_one.assert_called_once()
    db.user_profile_memories.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_insert_profile_memories_supersedes_milestone_by_llm_scope(monkeypatch):
    db = _mock_db()
    db.user_profile_memories.find_one = AsyncMock(return_value=None)
    db.user_profile_memories.update_many = AsyncMock()
    db.user_profile_memories.insert_one = AsyncMock()
    monkeypatch.setattr(users_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(users_module, "get_text_embedding", AsyncMock(return_value=[0.1]))

    await users_module.insert_profile_memories(
        "u1",
        [{
            "memory_type": MemoryType.MILESTONE,
            "content": "User prefers English replies",
            "scope": "language_preference",
        }],
    )

    update_filter = db.user_profile_memories.update_many.call_args.args[0]
    assert update_filter["scope"] == "language_preference"


@pytest.mark.asyncio
async def test_query_profile_memories_recent_loads_unlimited_active_commitments(monkeypatch):
    db = _mock_db()
    commitments = [
        {
            "memory_id": f"c{i}",
            "memory_type": MemoryType.COMMITMENT,
            "content": f"commitment {i}",
            "status": "active",
            "expires_at": "2999-01-01T00:00:00+00:00",
        }
        for i in range(12)
    ]
    db.user_profile_memories.find.side_effect = [
        _Cursor([]),
        _Cursor([]),
        _Cursor([]),
        _Cursor(commitments),
    ]
    monkeypatch.setattr(users_module, "get_db", AsyncMock(return_value=db))

    results = await users_module.query_profile_memories_recent("u1")

    assert len(results) == 12
    assert all(item["memory_type"] == MemoryType.COMMITMENT for item in results)


@pytest.mark.asyncio
async def test_query_profile_memories_vector_uses_threshold_and_unsets_embedding(monkeypatch):
    db = _mock_db()
    db.user_profile_memories.aggregate = MagicMock(return_value=_Cursor([]))
    monkeypatch.setattr(users_module, "get_db", AsyncMock(return_value=db))

    await users_module.query_profile_memories_vector(
        "u1",
        [0.1, 0.2],
        thresholds={MemoryType.OBJECTIVE_FACT: 0.8},
    )

    pipeline = db.user_profile_memories.aggregate.call_args.args[0]
    assert pipeline[0]["$vectorSearch"]["filter"]["global_user_id"] == "u1"
    assert pipeline[3]["$match"]["score"]["$gte"] == 0.8
    assert pipeline[-1] == {"$unset": "embedding"}


@pytest.mark.asyncio
async def test_query_user_profile_memory_blocks_preserves_named_prompt_shapes(monkeypatch):
    recent = [
        {
            "memory_id": "d1",
            "memory_type": MemoryType.DIARY_ENTRY,
            "content": "User sounded excited",
            "created_at": "2026-04-25T00:00:00+00:00",
        },
        {
            "memory_id": "f1",
            "memory_type": MemoryType.OBJECTIVE_FACT,
            "content": "User lives in Auckland",
            "category": "location",
            "created_at": "2026-04-25T00:00:00+00:00",
        },
        {
            "memory_id": "m1",
            "memory_type": MemoryType.MILESTONE,
            "content": "User allowed English replies",
            "event_category": "permission",
            "scope": "language_preference",
            "created_at": "2026-04-25T00:00:00+00:00",
        },
        {
            "memory_id": "c1",
            "memory_type": MemoryType.COMMITMENT,
            "content": "Kazusa will reply in English",
            "action": "Kazusa will reply in English",
            "status": "active",
        },
    ]
    monkeypatch.setattr(users_module, "query_profile_memories_recent", AsyncMock(return_value=recent))
    monkeypatch.setattr(users_module, "query_profile_memories_vector", AsyncMock(return_value=[]))

    blocks = await users_module.query_user_profile_memory_blocks("u1", include_semantic=True, topic_embedding=[0.1])

    assert blocks["character_diary"][0]["entry"] == "User sounded excited"
    assert blocks["objective_facts"][0]["fact"] == "User lives in Auckland"
    assert blocks["milestones"][0]["event"] == "User allowed English replies"
    assert blocks["active_commitments"][0]["action"] == "Kazusa will reply in English"


def test_build_memory_docs_stores_milestone_facts_only_once():
    memories = persistence_module._build_memory_docs(
        diary_entries=[],
        objective_facts=[{
            "fact": "User changed the preferred form of address",
            "category": "relationship",
            "timestamp": "2026-04-25T00:00:00+00:00",
            "source": "conversation_extracted",
            "confidence": 0.9,
        }],
        active_commitments=[],
        new_facts=[{
            "description": "User changed the preferred form of address",
            "is_milestone": True,
            "milestone_category": "relationship_state",
            "scope": "relationship_addressing",
            "dedup_key": "addressing_change",
        }],
        timestamp="2026-04-25T00:00:00+00:00",
    )

    assert len(memories) == 1
    assert memories[0]["memory_type"] == MemoryType.MILESTONE
    assert memories[0]["scope"] == "relationship_addressing"
    assert memories[0]["dedup_key"] == "addressing_change"


@pytest.mark.asyncio
async def test_expire_overdue_profile_memories_marks_active_commitments(monkeypatch):
    db = _mock_db()
    result = MagicMock(modified_count=3)
    db.user_profile_memories.update_many = AsyncMock(return_value=result)
    monkeypatch.setattr(users_module, "get_db", AsyncMock(return_value=db))

    count = await users_module.expire_overdue_profile_memories()

    assert count == 3
    update_filter = db.user_profile_memories.update_many.call_args.args[0]
    assert update_filter["memory_type"] == MemoryType.COMMITMENT
    assert update_filter["status"] == "active"


def test_hydrate_user_profile_with_memory_blocks_uses_memory_milestones():
    hydrated = users_module.hydrate_user_profile_with_memory_blocks(
        {
            "affinity": 500,
            "user_image": {"milestones": [{"event": "stale"}], "recent_window": [{"summary": "kept"}]},
        },
        {
            "character_diary": [{"entry": "Diary"}],
            "objective_facts": [{"fact": "Fact"}],
            "active_commitments": [{"action": "Promise"}],
            "milestones": [{"event": "fresh milestone"}],
            "memories": [
                {
                    "memory_id": "m1",
                    "memory_type": MemoryType.DIARY_ENTRY,
                    "content": "Diary",
                    "created_at": "2026-04-25T00:00:00+00:00",
                },
                {
                    "memory_id": "m2",
                    "memory_type": MemoryType.OBJECTIVE_FACT,
                    "content": "Fact",
                    "created_at": "2026-04-24T00:00:00+00:00",
                },
            ],
        },
    )

    assert hydrated["character_diary"][0]["entry"] == "Diary"
    assert hydrated["objective_facts"][0]["fact"] == "Fact"
    assert hydrated["active_commitments"][0]["action"] == "Promise"
    assert hydrated["user_image"]["milestones"] == [{"event": "fresh milestone"}]
    assert hydrated["user_image"]["recent_window"] == [
        {"timestamp": "2026-04-25T00:00:00+00:00", "summary": "Diary"},
        {"timestamp": "2026-04-24T00:00:00+00:00", "summary": "Fact"},
        {"timestamp": "", "summary": "kept"},
    ]
    assert hydrated["user_image"]["historical_summary"] == "Fact"


def test_hydrate_user_profile_with_memory_blocks_falls_back_to_relationship_history():
    hydrated = users_module.hydrate_user_profile_with_memory_blocks(
        {
            "affinity": 500,
            "last_relationship_insight": "这是关系洞察。",
            "user_image": {"milestones": [], "recent_window": []},
        },
        {
            "character_diary": [],
            "objective_facts": [],
            "active_commitments": [],
            "milestones": [],
            "memories": [],
        },
    )

    assert hydrated["user_image"]["recent_window"] == []
    assert hydrated["user_image"]["historical_summary"] == "这是关系洞察。"


@pytest.mark.asyncio
async def test_db_writer_invalidates_profile_memory_cache_namespaces(monkeypatch):
    fake_runtime = MagicMock()
    fake_runtime.invalidate = AsyncMock(side_effect=[3, 1])
    monkeypatch.setattr(persistence_module, "get_rag_cache2_runtime", MagicMock(return_value=fake_runtime))
    monkeypatch.setattr(persistence_module, "insert_profile_memories", AsyncMock(return_value=[]))
    monkeypatch.setattr(persistence_module, "update_affinity", AsyncMock())
    monkeypatch.setattr(persistence_module, "update_last_relationship_insight", AsyncMock())
    monkeypatch.setattr(persistence_module, "upsert_character_state", AsyncMock())
    monkeypatch.setattr(persistence_module, "upsert_user_image", AsyncMock())
    monkeypatch.setattr(persistence_module, "upsert_character_self_image", AsyncMock())
    monkeypatch.setattr(persistence_module, "_update_user_image", AsyncMock(return_value=None))
    monkeypatch.setattr(persistence_module, "_update_character_image", AsyncMock(return_value=None))
    monkeypatch.setattr(persistence_module, "_task_dispatcher", None)
    monkeypatch.setattr(persistence_module, "_task_registry", None)

    result = await persistence_module.db_writer({
        "timestamp": "2026-04-25T00:00:00+00:00",
        "global_user_id": "u1",
        "user_name": "User",
        "character_profile": {"name": "Kazusa"},
        "metadata": {},
        "mood": "neutral",
        "global_vibe": "",
        "reflection_summary": "",
        "diary_entry": ["User sounded excited"],
        "interaction_subtext": "test",
        "last_relationship_insight": "",
        "new_facts": [],
        "future_promises": [],
        "user_profile": {"affinity": 500},
        "affinity_delta": 0,
        "decontexualized_input": "hello",
        "platform": "discord",
        "platform_channel_id": "chan-1",
    })

    invalidated = result["metadata"]["cache_invalidated"]
    assert invalidated == ["user_profile", "character_state"]
    assert result["metadata"]["cache_evicted_count"] == 4


@pytest.mark.asyncio
async def test_db_writer_records_dispatch_rejection_when_no_runtime_adapter_is_registered(
    monkeypatch,
    caplog,
):
    """Accepted outbound tasks should stay unscheduled when the brain has no runtime adapter."""

    fake_runtime = MagicMock()
    fake_runtime.invalidate = AsyncMock(return_value=0)
    monkeypatch.setattr(persistence_module, "get_rag_cache2_runtime", MagicMock(return_value=fake_runtime))
    monkeypatch.setattr(persistence_module, "insert_profile_memories", AsyncMock(return_value=[]))
    monkeypatch.setattr(persistence_module, "update_affinity", AsyncMock())
    monkeypatch.setattr(persistence_module, "update_last_relationship_insight", AsyncMock())
    monkeypatch.setattr(persistence_module, "upsert_character_state", AsyncMock())
    monkeypatch.setattr(persistence_module, "upsert_user_image", AsyncMock())
    monkeypatch.setattr(persistence_module, "upsert_character_self_image", AsyncMock())
    monkeypatch.setattr(persistence_module, "_update_user_image", AsyncMock(return_value=None))
    monkeypatch.setattr(persistence_module, "_update_character_image", AsyncMock(return_value=None))

    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    dispatcher = TaskDispatcher(
        ToolCallEvaluator(tool_registry, AdapterRegistry()),
        PendingTaskIndex(),
    )
    monkeypatch.setattr(persistence_module, "_task_dispatcher", dispatcher)
    monkeypatch.setattr(persistence_module, "_task_registry", tool_registry)
    monkeypatch.setattr(
        persistence_module,
        "_generate_raw_tool_calls",
        AsyncMock(return_value=[
            RawToolCall(
                tool="send_message",
                args={
                    "target_channel": "54369546",
                    "text": "今天天气真好呀",
                    "execute_at": "2026-04-25T05:39:33+00:00",
                },
            )
        ]),
    )

    with caplog.at_level(logging.WARNING, logger=persistence_module.__name__):
        result = await persistence_module.db_writer({
            "timestamp": "2026-04-25T17:38:33+12:00",
            "global_user_id": "u1",
            "user_name": "蚝爹油",
            "platform": "qq",
            "platform_channel_id": "673225019",
            "platform_message_id": "1615877136",
            "character_profile": {"name": "杏山千纱"},
            "metadata": {},
            "mood": "neutral",
            "global_vibe": "",
            "reflection_summary": "",
            "diary_entry": ["接受了在别的群发消息的请求。"],
            "interaction_subtext": "test",
            "last_relationship_insight": "",
            "new_facts": [],
            "future_promises": [{
                "target": "54369546群",
                "action": "杏山千纱将对54369546群发送“今天天气真好呀”",
                "due_time": "2026-04-25T05:39:33+00:00",
                "commitment_type": "future_promise",
                "dedup_key": "send_group_weather_message",
            }],
            "user_profile": {"affinity": 500},
            "affinity_delta": 0,
            "decontexualized_input": "千纱酱一分钟之后能去54369546群里发个消息，内容是今天天气真好呀",
        })

    assert result["metadata"]["scheduled_event_ids"] == []
    assert result["metadata"]["task_dispatch_rejected"] == ["send_message: no adapters registered"]
    assert "Task dispatch unavailable" in caplog.text
    assert "no adapters registered" in caplog.text


@pytest.mark.asyncio
async def test_bootstrap_adds_active_commitment_hot_path_index(monkeypatch):
    db = MagicMock()
    db.list_collection_names = AsyncMock(return_value=[
        "conversation_history",
        "user_profiles",
        "character_state",
        "memory",
        "user_profile_memories",
        "scheduled_events",
    ])
    db.drop_collection = AsyncMock()
    db.character_state.find_one = AsyncMock(return_value={"_id": "global"})
    db.character_state.insert_one = AsyncMock()
    for collection_name in (
        "conversation_history",
        "user_profiles",
        "scheduled_events",
        "memory",
        "user_profile_memories",
    ):
        getattr(db, collection_name).create_index = AsyncMock()
    monkeypatch.setattr(bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(bootstrap_module, "enable_vector_index", AsyncMock())

    await bootstrap_module.db_bootstrap()

    db.drop_collection.assert_not_awaited()
    index_calls = db.user_profile_memories.create_index.await_args_list
    assert any(
        call.args[0] == [
            ("global_user_id", 1),
            ("memory_type", 1),
            ("status", 1),
            ("deleted", 1),
            ("created_at", -1),
            ("expires_at", 1),
        ]
        and call.kwargs["name"] == "user_profile_memory_active_commitments"
        for call in index_calls
    )


@pytest.mark.asyncio
async def test_bootstrap_drops_legacy_rag_collections(monkeypatch):
    db = MagicMock()
    db.list_collection_names = AsyncMock(return_value=[
        "conversation_history",
        "user_profiles",
        "character_state",
        "memory",
        "user_profile_memories",
        "scheduled_events",
        "rag_cache_index",
        "rag_metadata_index",
    ])
    db.drop_collection = AsyncMock()
    db.character_state.find_one = AsyncMock(return_value={"_id": "global"})
    db.character_state.insert_one = AsyncMock()
    for collection_name in (
        "conversation_history",
        "user_profiles",
        "scheduled_events",
        "memory",
        "user_profile_memories",
    ):
        getattr(db, collection_name).create_index = AsyncMock()
    monkeypatch.setattr(bootstrap_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(bootstrap_module, "enable_vector_index", AsyncMock())

    await bootstrap_module.db_bootstrap()

    assert db.drop_collection.await_args_list[0].args == ("rag_cache_index",)
    assert db.drop_collection.await_args_list[1].args == ("rag_metadata_index",)


def test_migration_builder_promotes_legacy_milestones_and_future_defaults_commitments():
    memories = migrate_module._build_memories_from_profile(
        {
            "user_image": {
                "milestones": [{
                    "event": "Legacy milestone",
                    "timestamp": "2025-11-01T00:00:00+00:00",
                    "category": "relationship_state",
                    "scope": "relationship_addressing",
                }]
            },
            "active_commitments": [{
                "action": "Kazusa will remember the request",
                "created_at": "2025-11-01T00:00:00+00:00",
            }],
        },
        "2026-04-25T00:00:00+00:00",
    )

    milestone = next(memory for memory in memories if memory["memory_type"] == MemoryType.MILESTONE)
    commitment = next(memory for memory in memories if memory["memory_type"] == MemoryType.COMMITMENT)
    assert milestone["content"] == "Legacy milestone"
    assert milestone["scope"] == "relationship_addressing"
    assert commitment["due_time"] == "2026-05-05T00:00:00+00:00"


@pytest.mark.asyncio
async def test_migration_script_dry_run_counts_fixture_memories(monkeypatch, capsys):
    db = MagicMock()
    db.user_profiles.find.return_value = _Cursor([{
        "global_user_id": "u1",
        "character_diary": [{"entry": "User sounded excited"}],
        "objective_facts": [{"fact": "User lives in Auckland"}],
        "user_image": {"milestones": [{"event": "User allowed English replies"}]},
        "active_commitments": [{"action": "Kazusa will reply in English"}],
    }])
    monkeypatch.setattr(migrate_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(migrate_module, "insert_profile_memories", AsyncMock())
    monkeypatch.setattr(
        "sys.argv",
        ["migrate_user_profile_memories.py", "--dry-run", "--batch-size", "10"],
    )

    await migrate_module.async_main()

    out = capsys.readouterr().out
    assert "scanned_profiles=1" in out
    assert "would_write:4" in out
    migrate_module.insert_profile_memories.assert_not_called()
