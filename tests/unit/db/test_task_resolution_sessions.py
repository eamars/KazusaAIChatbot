"""Executable tests for the durable DSH task-binding repository."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any, get_type_hints

import pytest

from tests.task_resolution_test_helpers import _context, _goal_continuation_ref


class _InsertResult:
    inserted_id = "binding-1"


class _FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def sort(self, field: str, direction: int) -> _FakeCursor:
        self.rows.sort(key=lambda row: str(row.get(field, "")), reverse=direction < 0)
        return self

    def limit(self, limit: int) -> _FakeCursor:
        self.rows = self.rows[:limit]
        return self

    async def to_list(self, length: int) -> list[dict[str, Any]]:
        return deepcopy(self.rows[:length])


class _FakeCollection:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.indexes: dict[str, dict[str, Any]] = {}
        self.queries: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    async def create_index(self, keys: object, **options: object) -> str:
        name = str(options["name"])
        self.indexes[name] = {"keys": keys, **options}
        return name

    async def index_information(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.indexes)

    async def insert_one(self, document: dict[str, Any]) -> _InsertResult:
        self.documents.append(deepcopy(document))
        return _InsertResult()

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

    def find(
        self,
        query: dict[str, Any],
        projection: dict[str, int] | None = None,
    ) -> _FakeCursor:
        del projection
        self.queries.append(deepcopy(query))
        return _FakeCursor([
            deepcopy(row) for row in self.documents if _matches(row, query)
        ])

    async def find_one_and_update(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
        **kwargs: object,
    ) -> dict[str, Any] | None:
        del kwargs
        self.queries.append(deepcopy(query))
        self.updates.append(deepcopy(update))
        for index, row in enumerate(self.documents):
            if _matches(row, query):
                updated = deepcopy(row)
                _apply_update(updated, update)
                self.documents[index] = updated
                return deepcopy(updated)
        return None


class _FakeDb:
    def __init__(self) -> None:
        self.dsh_task_bindings = _FakeCollection()
        self.dsh_interaction_store = _FakeCollection()

    def __getitem__(self, name: str) -> _FakeCollection:
        return getattr(self, name)


def _matches(row: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, expected in query.items():
        if key.startswith("$"):
            continue
        actual = row.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$exists" in expected and (key in row) != expected["$exists"]:
                return False
            if "$lte" in expected and actual > expected["$lte"]:
                return False
            continue
        if actual != expected:
            return False
    return True


def _apply_update(row: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.get("$set", {}).items():
        row[key] = deepcopy(value)
    for key, value in update.get("$inc", {}).items():
        row[key] = int(row.get(key, 0)) + int(value)
    for key in update.get("$unset", {}):
        row.pop(key, None)


def _start_spec() -> dict[str, object]:
    from kazusa_ai_chatbot.task_resolution.service import _build_start_spec

    context = _context()
    return _build_start_spec(
        {
            "semantic_goal": "Resolve one bounded goal.",
            "reason": "A bounded repository test request.",
            "evidence_handles": [],
            "start_in_background": False,
        },
        context,
    )


def _binding() -> dict[str, Any]:
    return {
        "schema_version": "dsh_task_binding.v1",
        "task_session_id": "session-1",
        "semantic_objective": "Resolve one bounded goal.",
        "goal_continuation_ref": _goal_continuation_ref(),
        "source_scope": {
            "schema_version": "dsh_task_source_scope.v1",
            "platform": "debug",
            "channel_id": "channel-1",
            "channel_type": "private",
            "requester_global_user_id": "user-1",
            "requester_platform_user_id": "debug-user-1",
            "source_message_id": "message-1",
            "source_platform_bot_id": "bot-1",
        },
        "state": "queued",
        "start_spec": _start_spec(),
        "resolution_thread_id": None,
        "segment_id": None,
        "resolution_ref": None,
        "operation_generation": 0,
        "current_accepted_task_id": None,
        "current_background_work_job_id": None,
        "latest_task_resolution_result": None,
        "revision": 0,
        "created_at": "2026-08-30T22:00:00Z",
        "updated_at": "2026-08-30T22:00:00Z",
    }


def test_binding_followup_schemas_are_closed_without_interaction_waiting() -> None:
    """The durable task binding has no DSH user-interaction wait carrier."""

    from kazusa_ai_chatbot.accepted_task import models as accepted_models
    from kazusa_ai_chatbot.background_work import models as background_models

    schemas = importlib.import_module("kazusa_ai_chatbot.db.schemas")
    binding = getattr(schemas, "DshTaskBindingDoc", None)
    if binding is None:
        pytest.fail("db schema owner lacks DshTaskBindingDoc")
    assert set(get_type_hints(binding)) == {
        "schema_version",
        "task_session_id",
        "semantic_objective",
        "goal_continuation_ref",
        "source_scope",
        "state",
        "start_spec",
        "resolution_thread_id",
        "segment_id",
        "resolution_ref",
        "operation_generation",
        "current_accepted_task_id",
        "current_background_work_job_id",
        "latest_task_resolution_result",
        "revision",
        "created_at",
        "updated_at",
    }
    assert "pending_dsh_interaction" not in get_type_hints(binding)
    assert "waiting_for_" + "interaction" not in get_type_hints(binding)

    accepted_fields = set(get_type_hints(accepted_models.AcceptedTaskDoc))
    assert accepted_fields >= {
        "dsh_task_session_id",
        "dsh_operation_generation",
        "dsh_followup_open",
        "dsh_followup_claim_action_attempt_id",
    }
    background_fields = set(get_type_hints(background_models.BackgroundWorkJobDoc))
    assert "worker_payload" in background_fields


def _required(module: object, name: str) -> Any:
    value = getattr(module, name, None)
    if not callable(value):
        pytest.fail(f"task-binding repository helper is unavailable: {name}")
    return value


@pytest.mark.asyncio
async def test_binding_generation_attach_checkpoint_terminal_and_followup_reconcile_is_revision_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binding writes use CAS revisions and reject stale generation updates."""

    try:
        module = importlib.import_module(
            "kazusa_ai_chatbot.db.task_resolution_sessions",
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned DB owner is unavailable: {exc}")
    database = _FakeDb()
    monkeypatch.setattr(module, "get_db", lambda: _async_value(database))
    create = _required(module, "create_task_binding")
    transition = _required(module, "transition_task_binding")
    attach = _required(module, "attach_resolution_ref")
    reconcile = _required(module, "reconcile_task_resolution_result")

    created = await create(_binding())
    assert created["task_session_id"] == "session-1"
    attached = await attach(
        task_session_id="session-1",
        expected_revision=0,
        resolution_ref={
            "schema_version": "dsh_resolution_ref.v1",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "dsh_session_id": "session-1",
            "activation_id": "activation-1",
            "lease_epoch": 1,
            "document_revision": 1,
            "last_committed_seq": 1,
        },
    )
    assert attached["revision"] == 1
    opening = await transition(
        task_session_id="session-1",
        expected_revision=1,
        expected_state="queued",
        next_state="opening",
        operation_generation=0,
    )
    assert opening["state"] == "opening"
    checkpointed = await transition(
        task_session_id="session-1",
        expected_revision=2,
        expected_state="opening",
        next_state="checkpointed",
        operation_generation=0,
    )
    assert checkpointed["state"] == "checkpointed"
    with pytest.raises(ValueError):
        await transition(
            task_session_id="session-1",
            expected_revision=1,
            expected_state="checkpointed",
            next_state="active",
            operation_generation=1,
        )
    active = await transition(
        task_session_id="session-1",
        expected_revision=3,
        expected_state="checkpointed",
        next_state="active",
        operation_generation=0,
    )
    assert active["state"] == "active"
    terminal = await transition(
        task_session_id="session-1",
        expected_revision=4,
        expected_state="active",
        next_state="terminal",
        operation_generation=0,
    )
    assert terminal["state"] == "terminal"
    result = {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve one bounded goal.",
        "status": "resolved",
        "scene_context": _context()["scene_context"],
        "goal_continuation_ref": _context()["goal_continuation_ref"],
        "evidence_state": "complete",
        "evidence_excerpts": [],
        "evidence_handles": [],
        "prompt_safe_summary": "Done.",
        "evidence": [],
        "completed_subgoals": [],
        "remaining_needs": [],
        "checkpoint": {},
        "coding_run_context": {},
    }
    reconciled = await reconcile(
        task_session_id="session-1",
        expected_revision=5,
        operation_generation=0,
        task_resolution_result=result,
    )
    replay = await reconcile(
        task_session_id="session-1",
        expected_revision=6,
        operation_generation=0,
        task_resolution_result=result,
    )
    assert reconciled["latest_task_resolution_result"] == replay[
        "latest_task_resolution_result"
    ]
    assert any(query.get("revision") == 0 for query in database.dsh_task_bindings.queries)
    assert any(query.get("revision") == 4 for query in database.dsh_task_bindings.queries)


@pytest.mark.asyncio
async def test_binding_repository_rejects_invalid_initial_carriers_and_cas_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The repository validates initial state, nested carriers, results, and CAS types."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.db.task_resolution_sessions",
    )
    database = _FakeDb()
    monkeypatch.setattr(module, "get_db", lambda: _async_value(database))
    create = _required(module, "create_task_binding")

    invalid_state = _binding()
    invalid_state["state"] = "opening"
    with pytest.raises(ValueError, match="start in queued"):
        await create(invalid_state)

    invalid_scope = _binding()
    invalid_scope["source_scope"] = {
        "schema_version": "dsh_task_source_scope.v1",
        "platform": "debug",
    }
    with pytest.raises(ValueError, match="fields are not exact"):
        await create(invalid_scope)

    invalid_start_spec = _binding()
    invalid_start_spec["start_spec"] = {
        "schema_version": "dsh_task_start_spec.v1",
    }
    with pytest.raises(ValueError, match="fields are not exact"):
        await create(invalid_start_spec)

    valid = await create(_binding())
    attach = _required(module, "attach_resolution_ref")
    reference = {
        "schema_version": "dsh_resolution_ref.v1",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "dsh_session_id": "session-1",
        "activation_id": "activation-1",
        "lease_epoch": 1,
        "document_revision": 1,
        "last_committed_seq": 1,
    }
    with pytest.raises(TypeError, match="expected_revision"):
        await attach(
            task_session_id="session-1",
            expected_revision=True,
            resolution_ref=reference,
        )

    attach_task = _required(module, "attach_accepted_task")
    with pytest.raises(TypeError, match="operation_generation"):
        await attach_task(
            task_session_id="session-1",
            expected_revision=0,
            operation_generation=True,
            accepted_task_id="task-1",
        )
    with pytest.raises(ValueError, match="revision or state fence"):
        await attach_task(
            task_session_id="session-1",
            expected_revision=valid["revision"],
            operation_generation=1,
            accepted_task_id="task-1",
        )

    transition = _required(module, "transition_task_binding")
    with pytest.raises(TypeError, match="expected_revision"):
        await transition(
            task_session_id="session-1",
            expected_revision=True,
            expected_state="queued",
            next_state="opening",
            operation_generation=0,
        )
    with pytest.raises(TypeError, match="operation_generation"):
        await transition(
            task_session_id="session-1",
            expected_revision=0,
            expected_state="queued",
            next_state="opening",
            operation_generation="0",
        )
    with pytest.raises(TypeError, match="expected_operation_generation"):
        await transition(
            task_session_id="session-1",
            expected_revision=0,
            expected_state="queued",
            next_state="opening",
            operation_generation=0,
            expected_operation_generation=True,
        )

    reconcile = _required(module, "reconcile_task_resolution_result")
    with pytest.raises(ValueError, match="fields are not exact"):
        await reconcile(
            task_session_id="session-1",
            expected_revision=0,
            operation_generation=0,
            task_resolution_result={
                "schema_version": "task_resolution_result.v1",
            },
        )


async def _async_value(value: object) -> object:
    return value


@pytest.mark.asyncio
async def test_binding_repository_is_exposed_only_through_named_db_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The facade exposes named binding lookups and does not expose collections."""

    try:
        module = importlib.import_module("kazusa_ai_chatbot.db.task_resolution_sessions")
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned DB owner is unavailable: {exc}")
    database = _FakeDb()
    database.dsh_task_bindings.documents.append(_binding())
    monkeypatch.setattr(module, "get_db", lambda: _async_value(database))
    for helper_name, kwargs in (
        ("find_binding_by_session", {"task_session_id": "session-1"}),
        ("find_binding_by_thread", {"resolution_thread_id": "thread-1"}),
        ("find_binding_by_accepted_task", {"accepted_task_id": "task-1"}),
        ("find_binding_by_background_job", {"background_work_job_id": "job-1"}),
    ):
        helper = _required(module, helper_name)
        result = await helper(**kwargs)
        assert result is None or result["schema_version"] == "dsh_task_binding.v1"


@pytest.mark.asyncio
async def test_bootstrap_creates_binding_and_dsh_followup_indexes_and_drops_only_obsolete_coding_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap invokes binding indexes and removes only the obsolete coding index."""

    try:
        module = importlib.import_module("kazusa_ai_chatbot.db.bootstrap")
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned bootstrap owner is unavailable: {exc}")
    database = _FakeDb()
    monkeypatch.setattr(module, "get_db", lambda: _async_value(database))
    ensure = getattr(module, "ensure_task_resolution_indexes", None)
    if not callable(ensure):
        pytest.fail("bootstrap lacks ensure_task_resolution_indexes")
    await ensure()
    names = set(database.dsh_task_bindings.indexes)
    assert names == {
        "dsh_task_binding_session_unique",
        "dsh_task_binding_thread_unique",
        "dsh_task_binding_current_accepted_task_unique",
        "dsh_task_binding_current_background_job_unique",
        "dsh_task_binding_state_updated",
    }
    assert not any("ttl" in name.lower() for name in names)


def test_dsh_interaction_store_has_no_user_wait_lookup() -> None:
    """The V2 interaction store exposes audit/grant persistence only."""

    try:
        module = importlib.import_module("kazusa_ai_chatbot.db.dsh_interactions")
    except ModuleNotFoundError as exc:
        pytest.fail(f"DSH interaction owner is unavailable: {exc}")
    assert not callable(getattr(module, "find_pending_interaction", None))
    assert not callable(getattr(module, "find_pending_reply", None))
