"""Tests for the user-state snapshot and restore script."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from scripts import user_state_snapshot as snapshot_module


class _Cursor:
    """Small async cursor double for script tests."""

    def __init__(self, rows: list[dict]) -> None:
        """Store rows returned by ``to_list``.

        Args:
            rows: MongoDB-like rows exposed by the cursor.

        Returns:
            None.
        """
        self._rows = rows
        self.sort_spec = None

    def sort(self, spec, direction=None) -> "_Cursor":
        """Accept sort chaining used by snapshot queries.

        Args:
            spec: Sort field or list of sort pairs.
            direction: Optional sort direction for single-field sorts.

        Returns:
            This cursor double for fluent chaining.
        """
        self.sort_spec = (spec, direction)
        return self

    async def to_list(self, length=None) -> list[dict]:
        """Return the configured cursor rows.

        Args:
            length: Requested maximum row count.

        Returns:
            Configured MongoDB-like rows.
        """
        return self._rows


class _Collection:
    """Async collection double that records destructive operations."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        """Create the fake collection.

        Args:
            rows: Rows returned from ``find``.

        Returns:
            None.
        """
        self.rows = rows or []
        self.find_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self.inserted: list[list[dict]] = []
        self.update_many_calls: list[tuple[dict, dict]] = []
        self.update_one_calls: list[tuple[dict, dict, bool]] = []
        self.find_one_result: dict | None = None

    async def find_one(self, filter_doc: dict) -> dict | None:
        """Capture a single-row lookup and return the configured result.

        Args:
            filter_doc: MongoDB find filter.

        Returns:
            Configured profile document, if any.
        """
        self.find_calls.append(filter_doc)
        return self.find_one_result

    def find(self, filter_doc: dict, projection: dict | None = None) -> _Cursor:
        """Capture the find filter and return rows.

        Args:
            filter_doc: MongoDB find filter.
            projection: Optional MongoDB projection.

        Returns:
            Cursor double with configured rows.
        """
        self.find_calls.append(filter_doc)
        return _Cursor(self.rows)

    async def delete_many(self, filter_doc: dict) -> None:
        """Capture delete filters.

        Args:
            filter_doc: MongoDB delete filter.

        Returns:
            None.
        """
        self.delete_calls.append(filter_doc)

    async def insert_many(self, rows: list[dict]) -> None:
        """Capture inserted rows.

        Args:
            rows: Documents inserted by restore.

        Returns:
            None.
        """
        self.inserted.append(rows)

    async def update_many(self, filter_doc: dict, update_doc: dict) -> None:
        """Capture multi-document updates.

        Args:
            filter_doc: MongoDB update filter.
            update_doc: MongoDB update document.

        Returns:
            None.
        """
        self.update_many_calls.append((filter_doc, update_doc))

    async def update_one(
        self,
        filter_doc: dict,
        update_doc: dict,
        *,
        upsert: bool,
    ) -> None:
        """Capture one-document updates.

        Args:
            filter_doc: MongoDB update filter.
            update_doc: MongoDB update document.
            upsert: Whether restore requested an upsert.

        Returns:
            None.
        """
        self.update_one_calls.append((filter_doc, update_doc, upsert))


class _Db:
    """Database double exposing user-state collections."""

    def __init__(self) -> None:
        """Create all fake collections used by restore.

        Returns:
            None.
        """
        self.collections = {
            name: _Collection()
            for name in snapshot_module.USER_STATE_COLLECTIONS
        }
        self.user_profiles = self.collections["user_profiles"]

    def __getitem__(self, name: str) -> _Collection:
        """Return a fake collection by name.

        Args:
            name: Collection name.

        Returns:
            Fake collection.
        """
        return self.collections[name]


def test_conversation_history_filter_covers_user_owned_and_related_rows() -> None:
    """Conversation backup should include authored, addressed, and platform rows."""

    filter_doc = snapshot_module.conversation_history_filter(
        "global-user-1",
        [{"platform": "qq", "platform_user_id": "316"}],
    )

    assert {"global_user_id": "global-user-1"} in filter_doc["$or"]
    assert {"addressed_to_global_user_ids": "global-user-1"} in filter_doc["$or"]
    assert {"mentions.global_user_id": "global-user-1"} in filter_doc["$or"]
    assert {
        "platform": "qq",
        "reply_context.reply_to_platform_user_id": "316",
    } in filter_doc["$or"]
    assert {
        "platform": "qq",
        "mentions.platform_user_id": "316",
    } in filter_doc["$or"]


@pytest.mark.asyncio
async def test_resolve_user_scope_accepts_global_id_without_profile(monkeypatch) -> None:
    """A bare global id should still scope global-id keyed collections."""

    db = _Db()
    monkeypatch.setattr(snapshot_module, "get_user_profile", AsyncMock(return_value={}))
    monkeypatch.setattr(snapshot_module, "get_db", AsyncMock(return_value=db))

    scope = await snapshot_module.resolve_user_scope("global-user-1", None)

    assert scope == {
        "global_user_id": "global-user-1",
        "profile": {"global_user_id": "global-user-1"},
        "platform_accounts": [],
    }


@pytest.mark.asyncio
async def test_restore_user_state_replaces_scoped_documents_and_alias_refs(
    monkeypatch,
    tmp_path,
) -> None:
    """Restore should delete current scoped rows and insert snapshot rows."""

    db = _Db()
    monkeypatch.setattr(snapshot_module, "get_db", AsyncMock(return_value=db))
    snapshot_path = tmp_path / "user_state.json"
    documents = {
        name: []
        for name in snapshot_module.USER_STATE_COLLECTIONS
    }
    documents["user_profiles"] = [
        {
            "_id": "profile-object-id",
            "global_user_id": "global-user-1",
            "platform_accounts": [
                {"platform": "qq", "platform_user_id": "316"},
            ],
            "suspected_aliases": [],
        }
    ]
    documents["user_memory_units"] = [
        {
            "_id": "unit-object-id",
            "unit_id": "unit-1",
            "global_user_id": "global-user-1",
            "fact": "User likes concise tests.",
            "subjective_appraisal": "The preference is stable.",
            "relationship_signal": "Keep restore behavior inspectable.",
        }
    ]
    documents["memory"] = [
        {
            "_id": "memory-object-id",
            "source_global_user_id": "global-user-1",
            "memory_name": "testing preference",
            "content": "The user wants derived embeddings regenerated.",
            "memory_type": "fact",
            "source_kind": "conversation_extracted",
        }
    ]
    documents["conversation_history"] = [
        {
            "_id": "message-object-id",
            "global_user_id": "global-user-1",
            "body_text": "hello",
            "attachments": [],
        }
    ]
    monkeypatch.setattr(
        snapshot_module,
        "get_text_embedding",
        AsyncMock(return_value=[0.5, 0.25]),
    )
    payload = snapshot_module.build_snapshot_payload(
        identity={
            "requested_identifier": "316",
            "requested_platform": "qq",
            "global_user_id": "global-user-1",
            "platform_accounts": [
                {"platform": "qq", "platform_user_id": "316"},
            ],
        },
        documents=documents,
        alias_profile_refs=[
            {
                "global_user_id": "global-user-2",
                "suspected_aliases": ["global-user-1"],
            }
        ],
    )
    snapshot_module.write_snapshot_file(snapshot_path, payload)

    summary = await snapshot_module.restore_user_state(file_path=snapshot_path)

    assert summary["global_user_id"] == "global-user-1"
    assert db["user_profiles"].delete_calls == [{"global_user_id": "global-user-1"}]
    assert db["user_profiles"].inserted == [documents["user_profiles"]]
    assert db["user_memory_units"].delete_calls == [{"global_user_id": "global-user-1"}]
    assert db["user_memory_units"].inserted[0][0]["embedding"] == [0.5, 0.25]
    assert db["memory"].inserted[0][0]["embedding"] == [0.5, 0.25]
    assert db["conversation_history"].inserted[0][0]["embedding"] == [0.5, 0.25]
    assert db.user_profiles.update_many_calls == [
        (
            {
                "global_user_id": {"$ne": "global-user-1"},
                "suspected_aliases": "global-user-1",
            },
            {"$pull": {"suspected_aliases": "global-user-1"}},
        )
    ]
    assert db.user_profiles.update_one_calls == [
        (
            {"global_user_id": "global-user-2"},
            {"$set": {"suspected_aliases": ["global-user-1"]}},
            False,
        )
    ]


@pytest.mark.asyncio
async def test_restore_user_state_rejects_unexpected_user(monkeypatch, tmp_path) -> None:
    """Restore guard should prevent applying a snapshot to the wrong user."""

    db = _Db()
    monkeypatch.setattr(snapshot_module, "get_db", AsyncMock(return_value=db))
    documents = {
        name: []
        for name in snapshot_module.USER_STATE_COLLECTIONS
    }
    payload = snapshot_module.build_snapshot_payload(
        identity={
            "requested_identifier": "global-user-1",
            "requested_platform": None,
            "global_user_id": "global-user-1",
            "platform_accounts": [],
        },
        documents=documents,
        alias_profile_refs=[],
    )
    snapshot_path = tmp_path / "user_state.json"
    snapshot_module.write_snapshot_file(snapshot_path, payload)

    with pytest.raises(ValueError, match="not global-user-2"):
        await snapshot_module.restore_user_state(
            file_path=snapshot_path,
            expected_global_user_id="global-user-2",
        )


def test_build_snapshot_payload_strips_embeddings_recursively() -> None:
    """Snapshot files should not persist stored vector fields."""

    documents = {
        name: []
        for name in snapshot_module.USER_STATE_COLLECTIONS
    }
    documents["user_memory_units"] = [
        {
            "unit_id": "unit-1",
            "embedding": [1.0, 2.0],
            "source_refs": [{"embedding": [3.0]}],
        }
    ]
    documents["conversation_history"] = [
        {
            "platform_message_id": "msg-1",
            "embedding": [4.0],
        }
    ]

    payload = snapshot_module.build_snapshot_payload(
        identity={
            "requested_identifier": "global-user-1",
            "requested_platform": None,
            "global_user_id": "global-user-1",
            "platform_accounts": [],
        },
        documents=documents,
        alias_profile_refs=[],
    )

    memory_unit = payload["documents"]["user_memory_units"][0]
    conversation_row = payload["documents"]["conversation_history"][0]
    assert "embedding" not in memory_unit
    assert "embedding" not in memory_unit["source_refs"][0]
    assert "embedding" not in conversation_row
