"""Executable tests for scoped DSH accepted-task persistence."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any

import pytest


class _FakeCollection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.documents = rows
        self.indexes: dict[str, dict[str, Any]] = {}
        self.queries: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    async def create_index(self, keys: object, **options: object) -> str:
        name = str(options["name"])
        self.indexes[name] = {"keys": keys, **options}
        return name

    async def index_information(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.indexes)

    async def find_one_and_update(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
        **kwargs: object,
    ) -> dict[str, Any] | None:
        del kwargs
        self.queries.append(deepcopy(query))
        self.updates.append(deepcopy(update))
        for row in self.documents:
            if _matches(row, query):
                for key, value in update.get("$set", {}).items():
                    row[key] = deepcopy(value)
                for key in update.get("$unset", {}):
                    row.pop(key, None)
                return deepcopy(row)
        return None

    async def find_one(
        self,
        query: dict[str, Any],
        projection: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        del projection
        self.queries.append(deepcopy(query))
        for row in self.documents:
            if _matches(row, query):
                return deepcopy(row)
        return None


class _FakeDb:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.accepted_tasks = _FakeCollection(rows)

    def __getitem__(self, name: str) -> _FakeCollection:
        if name != "accepted_tasks":
            raise KeyError(name)
        return self.accepted_tasks


def _matches(row: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, expected in query.items():
        actual = row.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$exists" in expected and (key in row) != expected["$exists"]:
                return False
            continue
        if actual != expected:
            return False
    return True


def _task() -> dict[str, Any]:
    return {
        "schema_version": "accepted_task.v2",
        "accepted_task_id": "task-1",
        "task_kind": "task_resolution",
        "state": "delivered",
        "task_identity_key": "identity-1",
        "source_platform": "debug",
        "source_channel_id": "channel-1",
        "source_channel_type": "private",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "dsh_task_session_id": "session-1",
        "dsh_operation_generation": 0,
        "dsh_followup_open": True,
        "dsh_followup_claim_action_attempt_id": None,
        "revision": 3,
        "updated_at": "2026-08-30T22:00:00Z",
        "result_summary": "The task is complete.",
    }


def _module() -> Any:
    try:
        from kazusa_ai_chatbot.db import accepted_tasks
    except ModuleNotFoundError as exc:
        pytest.fail(f"accepted-task DB owner is unavailable: {exc}")
    return accepted_tasks


@pytest.mark.asyncio
async def test_dsh_task_updates_have_no_interaction_wait_state() -> None:
    """Accepted-task persistence exposes no user-interaction wait writer."""

    module = _module()
    source = inspect.getsource(module)
    assert "waiting_for_" + "interaction" not in source
    assert not hasattr(
        module,
        "mark_accepted_task_waiting_for_" + "interaction",
    )


@pytest.mark.asyncio
async def test_one_open_dsh_followup_is_scoped_and_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real repository creates the sole scoped DSH follow-up index."""

    module = _module()
    database = _FakeDb([])
    monkeypatch.setattr(module, "get_db", lambda: _async_value(database))
    await module.ensure_accepted_task_indexes()
    assert set(database.accepted_tasks.indexes) >= {
        "accepted_task_open_dsh_followup_unique",
        "accepted_task_scope_dsh_followup_lookup",
    }


@pytest.mark.asyncio
async def test_followup_claim_recovery_and_terminal_updates_are_revision_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claims are scoped CAS operations and identical attempts replay safely."""

    module = _module()
    database = _FakeDb([_task()])
    monkeypatch.setattr(module, "get_db", lambda: _async_value(database))
    claim = getattr(module, "claim_dsh_followup", None)
    if not callable(claim):
        pytest.fail("accepted-task DB owner lacks claim_dsh_followup")
    first = await claim(
        accepted_task_id="task-1",
        action_attempt_id="attempt-1",
        expected_revision=3,
        operation="continue",
        instruction="Continue with the supplied semantic instruction.",
    )
    replay = await claim(
        accepted_task_id="task-1",
        action_attempt_id="attempt-1",
        expected_revision=4,
        operation="continue",
        instruction="Continue with the supplied semantic instruction.",
    )
    assert first["accepted_task_id"] == replay["accepted_task_id"] == "task-1"
    assert first["dsh_followup_claim_action_attempt_id"] == "attempt-1"
    assert first["dsh_followup_open"] is False
    query = database.accepted_tasks.queries[0]
    assert query["accepted_task_id"] == "task-1"
    assert query["dsh_followup_open"] is True
    assert query["revision"] == 3
    with pytest.raises(ValueError):
        await claim(
            accepted_task_id="task-1",
            action_attempt_id="different-attempt",
            expected_revision=4,
            operation="continue",
            instruction="Continue with the supplied semantic instruction.",
        )


async def _async_value(value: object) -> object:
    return value
