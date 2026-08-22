"""Explicit MongoDB test helpers for the Stage 2 persistence gates."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import pytest
from pymongo.errors import ConnectionFailure

TEST_DB_NAME = "_test_kazusa_live_llm"


def _document_hash(document: dict[str, Any]) -> str:
    """Hash one JSON document with stable key ordering."""

    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_content_hash(
    stored: dict[str, Any],
    expected: dict[str, Any],
    generated_fields: list[str],
) -> str:
    """Hash persisted seed content after removing generated fields."""

    del expected
    generated = set(generated_fields) | {"_id"}
    content = {
        key: value
        for key, value in stored.items()
        if key not in generated
    }
    return _document_hash(content)


async def seed_shared_documents(database: Any) -> None:
    """Seed one neutral, idempotent document for isolation smoke tests."""

    document = {
        "_id": "shared-test-seed",
        "schema": "cognition_shared_test_seed.v1",
        "content_hash": _document_hash(
            {"purpose": "bounded live database isolation"},
        ),
    }
    await database.test_seed_documents.replace_one(
        {"_id": document["_id"]},
        document,
        upsert=True,
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
