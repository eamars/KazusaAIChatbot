"""Tests for the character-state snapshot and restore script."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from scripts import character_state_snapshot as snapshot_module


class _CharacterStateCollection:
    """Async collection double for the singleton character_state document."""

    def __init__(self, document: dict | None) -> None:
        """Store the document exposed by ``find_one``.

        Args:
            document: MongoDB-like document returned from the fake collection.

        Returns:
            None.
        """
        self.document = document
        self.find_filter: dict | None = None
        self.replace_calls: list[tuple[dict, dict, bool]] = []

    async def find_one(self, filter_doc: dict) -> dict | None:
        """Return the configured document and capture the filter.

        Args:
            filter_doc: MongoDB find filter supplied by the script.

        Returns:
            Configured MongoDB-like document, if any.
        """
        self.find_filter = filter_doc
        return self.document

    async def replace_one(
        self,
        filter_doc: dict,
        replacement: dict,
        *,
        upsert: bool,
    ) -> None:
        """Capture the replacement requested by restore.

        Args:
            filter_doc: MongoDB replacement filter supplied by the script.
            replacement: Full replacement document.
            upsert: Whether MongoDB should create the singleton if missing.

        Returns:
            None.
        """
        self.replace_calls.append((filter_doc, replacement, upsert))


class _Db:
    """Database double exposing only the character_state collection."""

    def __init__(self, document: dict | None) -> None:
        """Create the fake database around one document.

        Args:
            document: MongoDB-like character_state document.

        Returns:
            None.
        """
        self.character_state = _CharacterStateCollection(document)


def test_build_snapshot_payload_requires_global_document() -> None:
    """Snapshot payloads should only be built for the singleton document."""

    with pytest.raises(ValueError, match='requires _id "global"'):
        snapshot_module.build_snapshot_payload({"_id": "other"})


@pytest.mark.asyncio
async def test_snapshot_character_state_writes_default_shape(
    monkeypatch,
    tmp_path,
) -> None:
    """Snapshot mode should persist the current singleton document."""

    db = _Db(
        {
            "_id": "global",
            "name": "Kazusa",
            "mood": "focused",
            "global_vibe": "quiet",
            "updated_at": "2026-05-03T00:00:00+00:00",
        }
    )
    monkeypatch.setattr(snapshot_module, "get_db", AsyncMock(return_value=db))
    output_path = tmp_path / "state.json"

    payload = await snapshot_module.snapshot_character_state(output_path)
    saved_payload = snapshot_module.read_snapshot_file(output_path)

    assert db.character_state.find_filter == {"_id": "global"}
    assert payload["snapshot_type"] == "character_state"
    assert saved_payload["query"] == {
        "collection": "character_state",
        "_id": "global",
    }
    assert saved_payload["character_state"]["mood"] == "focused"


@pytest.mark.asyncio
async def test_restore_character_state_replaces_singleton_document(
    monkeypatch,
    tmp_path,
) -> None:
    """Restore mode should replace only character_state/_id: global."""

    db = _Db(None)
    monkeypatch.setattr(snapshot_module, "get_db", AsyncMock(return_value=db))
    snapshot_path = tmp_path / "state.json"
    snapshot_module.write_snapshot_file(
        snapshot_path,
        {
            "query": {
                "collection": "character_state",
                "_id": "global",
            },
            "character_state": {
                "name": "Kazusa",
                "mood": "restored",
                "updated_at": "2026-05-03T01:00:00+00:00",
            },
        },
    )

    restored_state = await snapshot_module.restore_character_state(snapshot_path)

    assert restored_state["_id"] == "global"
    assert db.character_state.replace_calls == [
        (
            {"_id": "global"},
            {
                "_id": "global",
                "name": "Kazusa",
                "mood": "restored",
                "updated_at": "2026-05-03T01:00:00+00:00",
            },
            True,
        )
    ]


def test_restore_rejects_wrong_document_id() -> None:
    """Restore validation should refuse snapshots for non-global rows."""

    with pytest.raises(ValueError, match='requires _id "global"'):
        snapshot_module.extract_character_state(
            {
                "character_state": {
                    "_id": "other",
                    "mood": "bad",
                }
            }
        )
