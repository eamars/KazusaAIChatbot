from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.db import interaction_style_images as style_store
from kazusa_ai_chatbot.db.schemas import InteractionStyleStatus


class _StyleImageCollection:
    """Small async collection fake for style-image DB tests."""

    def __init__(self) -> None:
        """Create an empty fake collection."""

        self.docs: dict[str, dict] = {}
        self.indexes: list[tuple[list[tuple[str, int]], str]] = []

    async def create_index(
        self,
        keys: list[tuple[str, int]],
        *,
        name: str,
        unique: bool = False,
    ) -> None:
        """Record an index creation call."""

        self.indexes.append((keys, name))

    async def find_one(self, query: dict, projection: dict | None = None) -> dict | None:
        """Return the first document matching equality fields in ``query``."""

        for document in self.docs.values():
            matches = all(document.get(key) == value for key, value in query.items())
            if not matches:
                continue
            return_value = dict(document)
            if projection is not None and projection.get("_id") == 0:
                return_value.pop("_id", None)
            return return_value
        return_value = None
        return return_value

    async def replace_one(
        self,
        query: dict,
        document: dict,
        *,
        upsert: bool,
    ) -> SimpleNamespace:
        """Store a replacement document by ``style_image_id``."""

        style_image_id = str(query["style_image_id"])
        self.docs[style_image_id] = dict(document)
        return_value = SimpleNamespace(upserted_id=style_image_id if upsert else None)
        return return_value


class _StyleImageDb:
    """Small async DB fake for style-image DB tests."""

    def __init__(self) -> None:
        """Create a fake DB with one style-image collection."""

        self.collection = _StyleImageCollection()
        self.created_collections: list[str] = []

    async def list_collection_names(self) -> list[str]:
        """Return currently created collection names."""

        return_value = list(self.created_collections)
        return return_value

    async def create_collection(self, name: str) -> None:
        """Record collection creation."""

        self.created_collections.append(name)

    def __getitem__(self, name: str) -> _StyleImageCollection:
        """Return the style-image collection."""

        assert name == style_store.INTERACTION_STYLE_IMAGE_COLLECTION
        return self.collection


def _patch_get_db(monkeypatch: pytest.MonkeyPatch, db: _StyleImageDb) -> None:
    """Patch the style-image module database accessor."""

    monkeypatch.setattr(style_store, "get_db", AsyncMock(return_value=db))


def test_validate_interaction_style_overlay_normalizes_guidelines() -> None:
    """Overlay validation trims, deduplicates, caps, and drops non-strings."""

    raw_overlay = {
        "speech_guidelines": [
            "  Keep replies brisk and warm.  ",
            "Keep replies brisk and warm.",
            42,
            "Use a light challenge before giving help.",
        ],
        "social_guidelines": ["Acknowledge effort before teasing."],
        "pacing_guidelines": [],
        "engagement_guidelines": ["Offer a clear next step."],
        "confidence": "HIGH",
        "ignored": "field",
    }

    overlay = style_store.validate_interaction_style_overlay(raw_overlay)

    assert overlay == {
        "speech_guidelines": [
            "Keep replies brisk and warm.",
            "Use a light challenge before giving help.",
        ],
        "social_guidelines": ["Acknowledge effort before teasing."],
        "pacing_guidelines": [],
        "engagement_guidelines": ["Offer a clear next step."],
        "confidence": "high",
    }


def test_validate_interaction_style_overlay_rejects_source_details() -> None:
    """Event-like markers cannot be persisted as style guidance."""

    raw_overlay = {
        "speech_guidelines": ["After 2026-05-05, mention the old event."],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }

    with pytest.raises(ValueError, match="source detail"):
        style_store.validate_interaction_style_overlay(raw_overlay)


def test_validate_interaction_style_overlay_allows_single_apostrophe() -> None:
    """A normal English apostrophe is not a quote-heavy source example."""

    raw_overlay = {
        "speech_guidelines": ["Don't over-explain routine confirmations."],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }

    overlay = style_store.validate_interaction_style_overlay(raw_overlay)

    assert overlay["speech_guidelines"] == [
        "Don't over-explain routine confirmations."
    ]


def test_validate_interaction_style_overlay_rejects_quote_examples() -> None:
    """Quote-heavy examples are treated as event-like leakage risk."""

    raw_overlay = {
        "speech_guidelines": ["Never say 'I missed you' directly."],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }

    with pytest.raises(ValueError, match="source detail"):
        style_store.validate_interaction_style_overlay(raw_overlay)


def test_validate_interaction_style_overlay_rejects_confident_empty_overlay() -> None:
    """An empty overlay cannot claim non-empty confidence."""

    raw_overlay = style_store.empty_interaction_style_overlay()
    raw_overlay["confidence"] = "medium"

    with pytest.raises(ValueError, match="confidence"):
        style_store.validate_interaction_style_overlay(raw_overlay)


@pytest.mark.asyncio
async def test_ensure_interaction_style_image_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index bootstrap creates the collection and all lookup indexes."""

    db = _StyleImageDb()
    _patch_get_db(monkeypatch, db)

    await style_store.ensure_interaction_style_image_indexes()

    assert style_store.INTERACTION_STYLE_IMAGE_COLLECTION in db.created_collections
    index_names = {name for _, name in db.collection.indexes}
    assert index_names == {
        "interaction_style_image_id_unique",
        "interaction_style_user_scope",
        "interaction_style_group_channel_scope",
    }


@pytest.mark.asyncio
async def test_upsert_user_style_image_replaces_current_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User style images are one current document with revision history."""

    db = _StyleImageDb()
    _patch_get_db(monkeypatch, db)
    overlay = {
        "speech_guidelines": ["Keep replies brisk."],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }

    first_doc = await style_store.upsert_user_style_image(
        global_user_id="user-1",
        overlay=overlay,
        source_reflection_run_ids=["run-1"],
        timestamp="2026-05-06T00:00:00+00:00",
    )
    second_doc = await style_store.upsert_user_style_image(
        global_user_id="user-1",
        overlay=overlay,
        source_reflection_run_ids=["run-2"],
        timestamp="2026-05-07T00:00:00+00:00",
    )

    assert first_doc["revision"] == 1
    assert second_doc["revision"] == 2
    assert second_doc["created_at"] == "2026-05-06T00:00:00+00:00"
    assert second_doc["updated_at"] == "2026-05-07T00:00:00+00:00"
    assert second_doc["source_reflection_run_ids"] == ["run-2"]
    assert second_doc["status"] == InteractionStyleStatus.ACTIVE


@pytest.mark.asyncio
async def test_build_interaction_style_context_private_omits_group_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private chat style context includes no group-channel style field."""

    db = _StyleImageDb()
    _patch_get_db(monkeypatch, db)
    await style_store.upsert_user_style_image(
        global_user_id="user-1",
        overlay={
            "speech_guidelines": ["Use light teasing."],
            "social_guidelines": [],
            "pacing_guidelines": [],
            "engagement_guidelines": [],
            "confidence": "medium",
        },
        source_reflection_run_ids=["run-1"],
        timestamp="2026-05-06T00:00:00+00:00",
    )

    context = await style_store.build_interaction_style_context(
        global_user_id="user-1",
        channel_type="private",
        platform="qq",
        platform_channel_id="private-1",
    )

    assert context["application_order"] == ["user_style"]
    assert context["user_style"]["speech_guidelines"] == ["Use light teasing."]
    assert "group_channel_style" not in context


@pytest.mark.asyncio
async def test_build_interaction_style_context_group_applies_user_then_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group chat style context includes user and group overlays in order."""

    db = _StyleImageDb()
    _patch_get_db(monkeypatch, db)
    await style_store.upsert_user_style_image(
        global_user_id="user-1",
        overlay={
            "speech_guidelines": ["Use light teasing."],
            "social_guidelines": [],
            "pacing_guidelines": [],
            "engagement_guidelines": [],
            "confidence": "medium",
        },
        source_reflection_run_ids=["run-1"],
        timestamp="2026-05-06T00:00:00+00:00",
    )
    await style_store.upsert_group_channel_style_image(
        platform="qq",
        platform_channel_id="group-1",
        overlay={
            "speech_guidelines": [],
            "social_guidelines": ["Keep group replies compact."],
            "pacing_guidelines": [],
            "engagement_guidelines": [],
            "confidence": "high",
        },
        source_reflection_run_ids=["run-2"],
        timestamp="2026-05-06T00:00:00+00:00",
    )

    context = await style_store.build_interaction_style_context(
        global_user_id="user-1",
        channel_type="group",
        platform="qq",
        platform_channel_id="group-1",
    )

    assert context["application_order"] == ["user_style", "group_channel_style"]
    assert context["user_style"]["speech_guidelines"] == ["Use light teasing."]
    assert context["group_channel_style"]["social_guidelines"] == [
        "Keep group replies compact."
    ]
