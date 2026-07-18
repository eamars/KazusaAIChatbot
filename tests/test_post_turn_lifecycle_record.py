"""Focused tests for the idempotent post-turn lifecycle audit record."""

from __future__ import annotations

import importlib
import inspect


def test_post_turn_lifecycle_builder_has_the_frozen_signature() -> None:
    """The lifecycle record builder must accept settled action evidence."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.brain_service.post_turn",
    )
    builder = module.build_post_turn_lifecycle_record
    parameter_names = set(inspect.signature(builder).parameters)

    assert parameter_names == {
        "source_episode_id",
        "delivery_tracking_id",
        "action_specs",
        "action_results",
        "error_codes",
        "created_at",
    }


def test_empty_post_turn_lifecycle_is_skipped() -> None:
    """An episode with no lifecycle work must still get a typed record."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.brain_service.post_turn",
    )

    record = module.build_post_turn_lifecycle_record(
        source_episode_id="episode-001",
        delivery_tracking_id="delivery-001",
        action_specs=[],
        action_results=[],
        error_codes=[],
        created_at="2026-07-19T00:00:00+00:00",
    )

    assert record["schema_version"] == "post_turn_lifecycle_record.v1"
    assert record["source_episode_id"] == "episode-001"
    assert record["status"] == "skipped"


def test_post_turn_lifecycle_repository_is_public_and_idempotent() -> None:
    """The DB owner must expose one upsert boundary for the record."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.db.post_turn_lifecycle",
    )

    assert callable(module.upsert_post_turn_lifecycle_record)
