"""Executable tests for generation-bound DSH background jobs."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any, get_args

import pytest

from tests.task_resolution_test_helpers import _goal_continuation_ref


class _FakeCollection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.queries: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.indexes: dict[str, dict[str, Any]] = {}

    async def create_index(self, keys: object, **options: object) -> str:
        name = str(options["name"])
        self.indexes[name] = {"keys": keys, **options}
        return name

    async def insert_one(self, document: dict[str, Any]) -> object:
        self.rows.append(deepcopy(document))
        return object()

    async def find_one_and_update(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
        **kwargs: object,
    ) -> dict[str, Any] | None:
        del kwargs
        self.queries.append(deepcopy(query))
        self.updates.append(deepcopy(update))
        for row in self.rows:
            if _matches(row, query):
                for key, value in update.get("$set", {}).items():
                    row[key] = deepcopy(value)
                for key, value in update.get("$inc", {}).items():
                    row[key] = int(row.get(key, 0)) + int(value)
                return deepcopy(row)
        return None

    async def find_one(
        self,
        query: dict[str, Any],
        projection: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        del projection
        self.queries.append(deepcopy(query))
        for row in self.rows:
            if _matches(row, query):
                return deepcopy(row)
        return None


class _FakeDb:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.background_work_jobs = _FakeCollection(rows)

    def __getitem__(self, name: str) -> _FakeCollection:
        if name != "background_work_jobs":
            raise KeyError(name)
        return self.background_work_jobs


def _matches(row: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, expected in query.items():
        actual = row.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$exists" in expected and (key in row) != expected["$exists"]:
                return False
            if "$lte" in expected and actual > expected["$lte"]:
                return False
            continue
        if actual != expected:
            return False
    return True


def _payload() -> dict[str, object]:
    return {
        "schema_version": "task_orchestrator_worker_payload.v2",
        "operation": "open_dsh_resolution",
        "task_session_id": "session-1",
        "operation_generation": 0,
        "control": None,
    }


def _execution_context() -> dict[str, object]:
    """Build the complete trusted V2 context required by queue admission."""

    return {
        "schema_version": "task_resolution_execution_context.v2",
        "character_name": "Test Character",
        "platform": "debug",
        "channel_id": "channel-1",
        "channel_type": "private",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "source_message_id": "message-1",
        "source_platform_bot_id": "bot-1",
        "source_trigger_source": "user_message",
        "source_llm_trace_id": "trace-1",
        "brain_conversation_ref": "episode-task-001",
        "scene_context": {
            "channel_scope": "private",
            "character_role": "Test Character",
            "current_user_role": "Test User",
            "semantic_scene": "A deterministic background-work scene.",
            "public_group_scene": "",
            "conversation_continuity": "The same goal remains active.",
            "semantic_temporal_context": "The current test turn.",
        },
        "goal_continuation_ref": _goal_continuation_ref(),
        "local_time_context": {},
        "prompt_message_context": {"text": "Resolve one goal."},
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "persona_summary": "A deterministic test persona.",
        "conversation_summary": "A deterministic test conversation.",
        "current_timestamp_utc": "2026-08-30T22:00:00+00:00",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": [],
        "session_media_refs": [],
        "max_output_chars": 3000,
    }


def _job(state: str = "queued") -> dict[str, Any]:
    continuation_ref = _goal_continuation_ref()
    return {
        "schema_version": "background_work_job.v2",
        "job_id": "job-1",
        "idempotency_key": "background_work:task-1",
        "source_action_attempt_id": "attempt-1",
        "source_llm_trace_id": "trace-1",
        "correlation_write_status": "written",
        "correlation_conflict_source_llm_trace_id": "",
        "accepted_task_id": "task-1",
        "task_identity_key": "identity-1",
        "semantic_objective": "Resolve one goal.",
        "goal_continuation_ref": continuation_ref,
        "status": state,
        "delivery_state": "not_ready",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "source_platform": "debug",
        "source_channel_id": "channel-1",
        "source_channel_type": "private",
        "source_message_id": "message-1",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "created_at": "2026-08-30T22:00:00Z",
        "updated_at": "2026-08-30T22:00:00Z",
        "lease_owner": None,
        "lease_expires_at": None,
        "attempt_count": 0,
        "max_attempts": 4,
        "requested_worker": "task_orchestrator",
        "worker_payload": _payload(),
        "task_execution_context": {
            **_execution_context(),
            "goal_continuation_ref": continuation_ref,
        },
        "task_resolution_result": None,
        "artifact_text": "",
        "failure_summary": "",
        "result_summary": "",
        "completed_at": "",
        "delivery_attempt_count": 0,
        "delivery_failure_summary": "",
        "delivery_tracking_id": "",
        "delivered_conversation_message_id": "",
        "delivered_at": "",
    }


def _task_result() -> dict[str, object]:
    """Build one complete terminal result for the real DB validator."""

    context = _execution_context()
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve one goal.",
        "status": "resolved",
        "scene_context": context["scene_context"],
        "goal_continuation_ref": context["goal_continuation_ref"],
        "evidence_state": "complete",
        "evidence_excerpts": ["bounded evidence"],
        "evidence_handles": ["evidence-1"],
        "prompt_safe_summary": "The task is complete.",
        "evidence": [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "evidence-1",
            "task_node_id": "dsh",
            "specialist": "dsh",
            "summary": "bounded evidence",
            "provenance_refs": ["receipt-1"],
            "limitations": [],
        }],
        "completed_subgoals": ["bounded evidence"],
        "remaining_needs": [],
        "checkpoint": {},
        "coding_run_context": {},
    }


def _module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned background-work owner is unavailable: {exc}")


@pytest.mark.asyncio
async def test_task_worker_payload_v2_accepts_only_generation_bound_open_or_continue() -> None:
    """The real payload validator accepts only the closed V2 shape."""

    models = _module("kazusa_ai_chatbot.background_work.models")
    annotations = getattr(models.TaskOrchestratorWorkerPayloadV2, "__annotations__", {})
    assert set(annotations) == {
        "schema_version",
        "operation",
        "task_session_id",
        "operation_generation",
        "control",
    }
    jobs = _module("kazusa_ai_chatbot.background_work.jobs")
    validator = getattr(jobs, "validate_task_orchestrator_worker_payload", None)
    if not callable(validator):
        pytest.fail("background-work payload validator is unavailable")
    normalized = validator(_payload())
    assert normalized == _payload()
    for invalid in (
        {**_payload(), "schema_version": "task_orchestrator_worker_payload.v1"},
        {**_payload(), "operation": "resume_task_resolution"},
        {**_payload(), "authority_token": "never-queued"},
    ):
        with pytest.raises(ValueError):
            validator(invalid)


@pytest.mark.asyncio
async def test_queue_validates_binding_generation_goal_scope_and_payload_v2_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue admission validates lineage and carries no authority token."""

    jobs = _module("kazusa_ai_chatbot.background_work.jobs")
    queue = getattr(jobs, "enqueue_background_work_request", None)
    if not callable(queue):
        pytest.fail("background-work queue helper is unavailable")
    continuation_ref = _goal_continuation_ref()
    request = {
        "job_id": "job-1",
        "source_action_attempt_id": "attempt-1",
        "source_llm_trace_id": "trace-1",
        "idempotency_key": "background_work:task-1",
        "accepted_task_id": "task-1",
        "task_identity_key": "identity-1",
        "semantic_objective": "Resolve one goal.",
        "goal_continuation_ref": continuation_ref,
        "requested_worker": "task_orchestrator",
        "worker_payload": _payload(),
        "task_execution_context": {
            **_execution_context(),
            "goal_continuation_ref": continuation_ref,
        },
        "source_platform": "debug",
        "source_channel_id": "channel-1",
        "source_channel_type": "private",
        "source_message_id": "message-1",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "storage_timestamp_utc": "2026-08-30T22:00:00Z",
    }
    stored: list[dict[str, object]] = []

    async def insert(job: dict[str, object]) -> dict[str, object]:
        stored.append(job)
        return job

    monkeypatch.setattr(jobs, "insert_background_work_job", insert)
    result = await queue(request)
    assert result["job_id"] == "job-1"
    assert result["accepted_task_id"] == "task-1"
    assert stored[0]["worker_payload"] == _payload()
    assert "authority_token" not in stored[0]


def test_dsh_job_states_have_no_user_interaction_wait() -> None:
    """The background-job status contract has no user-interaction pause."""

    models = _module("kazusa_ai_chatbot.background_work.models")
    assert set(get_args(models.BackgroundWorkJobStatus)) == {
        "queued",
        "in_progress",
        "completed",
        "failed",
        "canceled",
        "delivery_in_progress",
        "delivery_failed",
        "delivered",
    }


def test_background_work_public_exports_are_dsh_task_or_future_speak_only() -> None:
    """The package export list contains the canonical V2 task surface."""

    module = _module("kazusa_ai_chatbot.background_work")
    public_names = set(getattr(module, "__all__", ()))
    assert "TaskOrchestratorWorkerPayloadV2" in public_names
    assert "FutureSpeakWorkerPayloadV1" in public_names
    assert not public_names & {
        "TaskOrchestratorWorkerPayloadV1",
        "CodingRunContextV1",
    }


@pytest.mark.asyncio
async def test_job_claim_excludes_v1_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claim filters exclude pre-cutover worker payloads."""

    module = _module("kazusa_ai_chatbot.db.background_work_jobs")
    database = _FakeDb([{**_job("queued"), "worker_payload": {
        **_payload(),
        "schema_version": "task_orchestrator_worker_payload.v1",
    }}])
    monkeypatch.setattr(module, "get_db", lambda: _async_value(database))
    claim = getattr(module, "claim_background_work_job", None)
    if not callable(claim):
        pytest.fail("background-work claim helper is unavailable")
    result = await claim(
        lease_owner="worker-1",
        lease_seconds=30,
        now_utc="2026-08-30T22:00:00Z",
        max_attempts=4,
    )
    assert result is None
    query = database.background_work_jobs.queries[0]
    assert query.get("status", {}).get("$in")


@pytest.mark.asyncio
async def test_terminal_sink_is_idempotent_under_delivery_state_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal completion settles one job and ignores a replay."""

    module = _module("kazusa_ai_chatbot.db.background_work_jobs")
    database = _FakeDb([{
        **_job("in_progress"),
        "lease_owner": "worker-1",
    }])
    monkeypatch.setattr(module, "get_db", lambda: _async_value(database))
    complete = getattr(module, "complete_background_work_job", None)
    if not callable(complete):
        pytest.fail("background-work completion helper is unavailable")
    first = await complete(
        job_id="job-1",
        lease_owner="worker-1",
        task_resolution_result=_task_result(),
        artifact_text="",
        result_summary="done",
        completed_at="2026-08-30T22:01:00Z",
    )
    second = await complete(
        job_id="job-1",
        lease_owner="worker-1",
        task_resolution_result=_task_result(),
        artifact_text="",
        result_summary="done",
        completed_at="2026-08-30T22:01:00Z",
    )
    assert first["job_id"] == second["job_id"] == "job-1"
    assert first["status"] == second["status"] == "completed"
    assert database.background_work_jobs.queries[0]["lease_owner"] == "worker-1"


async def _async_value(value: object) -> object:
    return value
