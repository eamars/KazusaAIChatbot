"""Explicit MongoDB test helpers for the Stage 2 persistence gates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

import pytest
from pymongo.errors import ConnectionFailure

from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    validate_cognition_state,
)

TEST_DB_NAME = "_test_kazusa_live_llm"
SEED_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "cognition_core_v2_mongo_seed.json"
)


def assert_test_db_name(database_name: str) -> None:
    """Require the exact isolated database name for Stage 2 DB tests."""

    if database_name != TEST_DB_NAME:
        raise AssertionError(
            f"Stage 2 requires {TEST_DB_NAME!r}; received {database_name!r}"
        )


def assert_no_xdist() -> None:
    """Reject parallel workers because singleton restore is process-scoped."""

    if os.getenv("PYTEST_XDIST_WORKER"):
        raise AssertionError("Stage 2 MongoDB tests must not run under xdist")


def unique_owner_id(nodeid: str) -> str:
    """Build a readable unique owner id from one pytest node id."""

    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", nodeid).strip("-")
    return f"s2-{sanitized}-{os.urandom(8).hex()}"


def _load_seed_documents() -> list[dict[str, Any]]:
    """Load fixed seed rows from the checked-in Stage 2 fixture."""

    with SEED_PATH.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    documents = list(fixture["seed_documents"])
    if not isinstance(documents, list):
        raise AssertionError("seed_documents must be a list")
    for emotion_id in EMOTION_DEFINITIONS:
        for phase in fixture["emotion_seed_phases"]:
            seed_id = f"seed-s2-emotion-{emotion_id}-{phase}"
            documents.append(
                {
                    "collection": "conversation_history",
                    "selector": {"_id": seed_id},
                    "document": {
                        "_id": seed_id,
                        "seed_id": seed_id,
                        "emotion_id": emotion_id,
                        "phase": phase,
                        "owner_scope": "seed-s2-established",
                        "body_text": (
                            f"synthetic {emotion_id} {phase} evidence"
                        ),
                        "source_metadata": {
                            "fixture": "cognition_core_v2_stage_2",
                            "synthetic": True,
                        },
                    },
                }
            )
    return documents


async def seed_shared_documents(database: Any) -> None:
    """Insert fixed seed rows idempotently and validate existing content."""

    assert_test_db_name(database.name)
    for seed in _load_seed_documents():
        if seed.get("fixture_only"):
            continue
        collection = database[seed["collection"]]
        selector = seed["selector"]
        document = seed["document"]
        await collection.update_one(
            selector,
            {"$setOnInsert": document},
            upsert=True,
        )
        stored = await collection.find_one(selector)
        if not _contains_seed_fields(stored, document):
            raise AssertionError(
                f"Seed content mismatch for {seed['collection']}:{selector}"
            )
        cognition_state = document.get("cognition_state")
        if cognition_state is not None:
            validated_expected = validate_cognition_state(cognition_state)
            stored_state = validate_cognition_state(stored["cognition_state"])
            if stored_state != validated_expected:
                raise AssertionError(
                    f"Seed cognition state mismatch for {seed['collection']}:{selector}"
                )


def _contains_seed_fields(
    stored: Any,
    expected: Any,
) -> bool:
    """Validate fixed seed fields while preserving unrelated legacy fields."""

    if isinstance(expected, dict):
        if not isinstance(stored, dict):
            return False
        return all(
            key in stored and _contains_seed_fields(stored[key], value)
            for key, value in expected.items()
        )
    return stored == expected


@pytest.fixture
async def live_db() -> Any:
    """Provide the guarded isolated database or skip when Mongo is offline."""

    assert_no_xdist()
    from kazusa_ai_chatbot.db._client import get_db

    try:
        database = await get_db()
    except ConnectionFailure as exc:
        pytest.skip(f"MongoDB is unavailable for the live DB gate: {exc}")
    assert_test_db_name(database.name)
    return database
