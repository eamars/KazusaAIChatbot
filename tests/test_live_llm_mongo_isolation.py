"""Checkpoint B MongoDB isolation and persistence-boundary tests."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

import kazusa_ai_chatbot.db._client as client_module
from tests.live_llm_mongo import (
    TEST_DB_NAME,
    _document_hash,
    _seed_content_hash,
    assert_no_xdist,
    assert_test_db_name,
    live_db,
    seed_shared_documents,
    unique_owner_id,
)


_CLIENT_CONSTRUCTOR_NAME = "AsyncIO" + "MotorClient"

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.db._client import close_db, get_db
from kazusa_ai_chatbot.db.character import (
    get_character_cognition_state,
)
from kazusa_ai_chatbot.db.users import (
    get_user_cognition_state,
    replace_user_cognition_state,
)


class _FakeAdmin:
    async def command(self, name: str) -> None:
        """Accept the client ping used by the DB facade."""

        assert name == "ping"


class _FakeDatabase:
    name = "production"


class _FakeClient:
    created_uris: list[str] = []

    def __init__(self, uri: str) -> None:
        self.created_uris.append(uri)
        self.admin = _FakeAdmin()

    def __getitem__(self, name: str) -> _FakeDatabase:
        """Return a fake database handle for guard testing."""

        return _FakeDatabase()

    def close(self) -> None:
        """Match the client close operation for guard testing."""


def test_guard_rejects_non_test_database_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a non-test DB name before constructing a Mongo client."""

    _FakeClient.created_uris.clear()
    monkeypatch.setattr(client_module, "_client", None)
    monkeypatch.setattr(client_module, "_db", None)
    monkeypatch.setattr(client_module, "_db_loop", None)
    monkeypatch.setattr(client_module, "MONGODB_DB_NAME", "production")
    monkeypatch.setattr(client_module, _CLIENT_CONSTRUCTOR_NAME, _FakeClient)

    async def exercise_guard() -> None:
        with pytest.raises(client_module.DatabaseTestGuardError):
            await client_module.get_db()

    asyncio.run(exercise_guard())
    assert _FakeClient.created_uris == []


def test_guard_allows_reserved_stage3_database_with_explicit_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow the exact Stage 3 database only with its explicit marker."""

    monkeypatch.setenv("KAZUSA_TEST_DB_GUARD", "1")
    monkeypatch.setenv("STAGE3_DATABASE_GUARD", "1")
    monkeypatch.setattr(
        client_module,
        "MONGODB_DB_NAME",
        client_module.STAGE3_TEST_DATABASE_NAME,
    )

    client_module._assert_guarded_database_name()


def test_mongodb_log_descriptor_excludes_credentials_and_query_options() -> None:
    """Connection diagnostics must not expose raw URI secrets."""

    descriptor = client_module._sanitized_mongodb_endpoint_description(
        "mongodb://stage-user:stage-secret@mongo.example:27017/"
        "?tls=true&replicaSet=stage",
        client_module.STAGE3_TEST_DATABASE_NAME,
    )

    assert descriptor == (
        "mongodb://mongo.example:27017/_test_kazusa_core_v2"
    )
    assert "stage-user" not in descriptor
    assert "stage-secret" not in descriptor
    assert "tls=true" not in descriptor
    assert "replicaSet" not in descriptor


def test_checkpoint_b_helpers_enforce_the_exact_database_and_no_xdist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep DB test helpers exact and single-process."""

    assert_test_db_name(TEST_DB_NAME)
    assert_no_xdist()
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    with pytest.raises(AssertionError):
        assert_no_xdist()


def test_seed_hash_rejects_an_unexpected_document_sibling() -> None:
    """Treat every non-generated stored field as fixture-owned content."""

    expected = {"global_user_id": "seed-s2"}
    stored = {
        "_id": "database-generated-id",
        "global_user_id": "seed-s2",
        "profile_name": "preserved profile sibling",
        "unexpected": "drift",
    }

    assert _seed_content_hash(
        stored,
        expected,
        ["profile_name"],
    ) != _document_hash(expected)
    stored.pop("unexpected")
    assert _seed_content_hash(
        stored,
        expected,
        ["profile_name"],
    ) == _document_hash(expected)


@pytest.mark.asyncio
@pytest.mark.live_db
async def test_seed_is_idempotent_and_owner_rows_are_isolated(
    live_db,
    request: pytest.FixtureRequest,
) -> None:
    """Seed fixed fixtures once and isolate mutable rows by owner."""

    await seed_shared_documents(live_db)
    await seed_shared_documents(live_db)
    owner_ids = [
        unique_owner_id(request.node.nodeid),
        unique_owner_id(request.node.nodeid),
    ]
    try:
        for owner_id in owner_ids:
            await live_db.user_profiles.insert_one(
                {
                    "global_user_id": owner_id,
                    "cognition_state": build_acquaintance_user_state(
                        global_user_id=owner_id,
                        updated_at="2026-07-14T00:00:00Z",
                    ),
                }
            )
        first_state = build_acquaintance_user_state(
            global_user_id=owner_ids[0],
            updated_at="2026-07-14T00:00:00Z",
        )
        first_state["relationship"]["familiarity"] = 55
        await live_db.user_profiles.update_one(
            {"global_user_id": owner_ids[0]},
            {"$set": {"cognition_state": first_state}},
        )
        second_document = await live_db.user_profiles.find_one(
            {"global_user_id": owner_ids[1]},
        )
        assert second_document["cognition_state"]["relationship"][
            "familiarity"
        ] == 10
    finally:
        await live_db.user_profiles.delete_many(
            {"global_user_id": {"$in": owner_ids}},
        )


@pytest.mark.asyncio
@pytest.mark.live_db
async def test_character_singleton_is_restored_after_a_mutation(
    live_db,
) -> None:
    """Restore the exact singleton snapshot after a calibration mutation."""

    snapshot = await live_db.character_state.find_one({"_id": "global"})
    assert snapshot is not None
    try:
        await live_db.character_state.update_one(
            {"_id": "global"},
            {"$set": {"cognition_state.meaning_state.salience": 42}},
        )
        changed = await live_db.character_state.find_one({"_id": "global"})
        assert changed["cognition_state"]["meaning_state"]["salience"] == 42
    finally:
        await live_db.character_state.replace_one(
            {"_id": "global"},
            snapshot,
            upsert=True,
        )
    restored = await live_db.character_state.find_one({"_id": "global"})
    assert restored == snapshot


@pytest.mark.asyncio
@pytest.mark.live_db
async def test_facades_reload_validated_state_after_client_restart(live_db) -> None:
    """Reload one user state through the facade after closing the client."""

    source_id = "seed-s2-acquaintance"
    user_id = unique_owner_id("facade-reload")
    snapshot = await live_db.user_profiles.find_one(
        {"global_user_id": source_id},
    )
    assert snapshot is not None
    cloned = deepcopy(snapshot)
    cloned.pop("_id", None)
    cloned["global_user_id"] = user_id
    cloned["cognition_state"]["owner_user_id"] = user_id
    cloned["cognition_state"]["relationship"]["other_user_id"] = user_id
    cloned["cognition_state"]["relationship"]["relationship_id"] = (
        f"relationship:user:{user_id}"
    )
    await live_db.user_profiles.insert_one(cloned)
    try:
        state = await get_user_cognition_state(user_id)
        state["relationship"]["familiarity"] = 44
        await replace_user_cognition_state(user_id, state)
        await close_db()
        reloaded = await get_user_cognition_state(user_id)
        assert reloaded["relationship"]["familiarity"] == 44
        character_state = await get_character_cognition_state()
        assert character_state["state_scope"] == "character"
    finally:
        await close_db()
        database = await get_db()
        await database.user_profiles.delete_one(
            {"global_user_id": user_id},
        )
