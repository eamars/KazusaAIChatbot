"""Tests for accepted-task lifecycle and duplicate rejection contracts."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from typing import Any

import pytest
from pymongo.errors import DuplicateKeyError
from tests.test_task_resolution_orchestrator import _goal_continuation_ref


def _create_request(**overrides: object) -> dict[str, object]:
    """Return one semantic accepted-task creation request."""

    request: dict[str, object] = {
        "task_kind": "future_speak",
        "semantic_objective": "2026-05-16 10:00 drink water reminder",
        "accepted_task_summary": "Remind the user to drink water.",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "source_trigger_source": "user_message",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "source_message_id": "message-001",
        "source_platform_bot_id": "debug-bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "debug-user-001",
        "requester_display_name": "Test User",
        "storage_timestamp_utc": "2026-05-15T21:00:00+00:00",
    }
    request.update(overrides)
    return request


def test_identity_key_excludes_source_message_id() -> None:
    """Repeated turns should share identity even when message ids differ."""

    from kazusa_ai_chatbot.accepted_task.lifecycle import (
        build_task_identity_key,
    )

    first_key = build_task_identity_key(_create_request())
    second_key = build_task_identity_key(
        _create_request(source_message_id="message-002"),
    )
    other_user_key = build_task_identity_key(
        _create_request(
            source_message_id="message-002",
            requester_global_user_id="global-user-002",
        ),
    )
    other_objective_key = build_task_identity_key(
        _create_request(
            semantic_objective="2026-05-16 11:00 drink water reminder",
        ),
    )

    assert first_key == second_key
    assert first_key != other_user_key
    assert first_key != other_objective_key


def test_repository_module_imports_in_a_fresh_process() -> None:
    """The repository module must not depend on prior lifecycle import order."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import kazusa_ai_chatbot.db.accepted_tasks",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_accepted_task_package_imports_in_a_fresh_process() -> None:
    """The public lifecycle package must not depend on prior import order."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import kazusa_ai_chatbot.accepted_task",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_create_or_return_active_claims_enqueueing_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new accepted task should claim the active identity before queueing."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))

    result = await lifecycle.create_or_return_active_accepted_task(
        _create_request(),
    )

    assert result["status"] == "created"
    task = result["task"]
    assert task["state"] == "enqueueing"
    assert task["schema_version"] == "accepted_task.v2"
    assert task["task_kind"] == "future_speak"
    assert task["completion_status"] == "none"
    assert task["accepted_task_id"].startswith("task-")
    assert task["active_identity_key"] == task["task_identity_key"]
    assert task["first_source_message_id"] == "message-001"
    assert task["related_source_message_ids"] == ["message-001"]
    assert len(fake_db.accepted_tasks.documents) == 1


def test_task_document_requires_goal_continuation_ref() -> None:
    """Task-resolution documents fail closed without continuation lineage."""

    from kazusa_ai_chatbot.accepted_task import lifecycle

    request = _create_request(
        task_kind="task_resolution",
        goal_continuation_ref=None,
    )

    with pytest.raises(ValueError, match="goal_continuation_ref"):
        lifecycle._build_enqueueing_task_doc(
            request,
            task_identity_key=lifecycle.build_task_identity_key(request),
        )


@pytest.mark.asyncio
async def test_create_or_return_active_persists_goal_continuation_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted task creation retains the task's canonical continuation ref."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))
    request = _create_request(
        task_kind="task_resolution",
        goal_continuation_ref=_goal_continuation_ref(),
    )

    result = await lifecycle.create_or_return_active_accepted_task(request)

    assert result["task"]["goal_continuation_ref"] == (
        _goal_continuation_ref()
    )


@pytest.mark.asyncio
async def test_create_or_return_active_rejects_duplicate_active_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated active task should return the existing task without insert."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))

    first = await lifecycle.create_or_return_active_accepted_task(
        _create_request(),
    )
    second = await lifecycle.create_or_return_active_accepted_task(
        _create_request(
            source_message_id="message-002",
            storage_timestamp_utc="2026-05-15T21:01:00+00:00",
        ),
    )

    assert first["status"] == "created"
    assert second["status"] == "already_active"
    assert second["task"]["accepted_task_id"] == first["task"]["accepted_task_id"]
    assert second["task"]["related_source_message_ids"] == [
        "message-001",
        "message-002",
    ]
    assert len(fake_db.accepted_tasks.documents) == 1


@pytest.mark.asyncio
async def test_repository_duplicate_ref_mismatch_preserves_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repository duplicate handling gates mismatches before mutation."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
    from kazusa_ai_chatbot.db import accepted_tasks as repository
    from kazusa_ai_chatbot.db.errors import DatabaseOperationError

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))
    request = _create_request(
        task_kind="task_resolution",
        goal_continuation_ref=_goal_continuation_ref(),
    )
    task_identity_key = lifecycle.build_task_identity_key(request)
    first_task = lifecycle._build_enqueueing_task_doc(
        request,
        task_identity_key=task_identity_key,
    )

    created = await repository.insert_or_get_active_accepted_task(
        first_task,
        source_message_id="message-001",
        observed_at="2026-05-15T21:00:00+00:00",
    )
    assert created["status"] == "created"

    matching_duplicate = deepcopy(first_task)
    matching_duplicate["accepted_task_id"] = "task-matching-duplicate"
    matching = await repository.insert_or_get_active_accepted_task(
        matching_duplicate,
        source_message_id="message-002",
        observed_at="2026-05-15T21:01:00+00:00",
    )
    assert matching["status"] == "already_active"
    assert matching["task"]["related_source_message_ids"] == [
        "message-001",
        "message-002",
    ]
    before_mismatch = deepcopy(fake_db.accepted_tasks.documents[0])

    mismatching_duplicate = deepcopy(first_task)
    mismatching_duplicate["accepted_task_id"] = "task-mismatching-duplicate"
    mismatching_duplicate["goal_continuation_ref"] = build_goal_continuation_ref(
        source_episode_id="episode-task-002",
        source_message_id="message-2",
        branch_id="b1",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal-001",
        },
    )
    with pytest.raises(
        DatabaseOperationError,
        match="different goal_continuation_ref",
    ):
        await repository.insert_or_get_active_accepted_task(
            mismatching_duplicate,
            source_message_id="message-003",
            observed_at="2026-05-15T21:02:00+00:00",
        )

    current = fake_db.accepted_tasks.documents[0]
    assert current["updated_at"] == before_mismatch["updated_at"]
    assert current["related_source_message_ids"] == (
        before_mismatch["related_source_message_ids"]
    )
    assert current == before_mismatch


@pytest.mark.asyncio
async def test_future_speak_duplicate_authority_mismatch_fails_before_related_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An active future-speak duplicate with a different authority is rejected."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.cognition_shared.contracts import (
        build_scheduled_future_speech_authority,
    )
    from kazusa_ai_chatbot.db import accepted_tasks as repository
    from kazusa_ai_chatbot.db.errors import DatabaseOperationError

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))

    def build_authority(*, trigger_local: str) -> dict[str, Any]:
        authority = build_scheduled_future_speech_authority(
            source_episode_id="episode-2026-05-15-001",
            source_message_id="message-001",
            source_action_attempt_id="action_attempt:future-speak-001",
            source_llm_trace_id="llmtrace_source-1",
            accepted_at_utc="2026-05-15T21:00:00+00:00",
            timezone="Pacific/Auckland",
            trigger_local=trigger_local,
            platform="debug",
            channel_type="private",
            audience_kind="private",
            semantic_objective="Remind the user to drink water.",
            authorized_content_summary="在约定时间提醒用户喝水。",
            authorized_detail_refs=[
                {
                    "evidence_handle": "e1",
                    "semantic_summary": "当前对话明确约定在约定时间提醒喝水。",
                    "provenance_role": "current_event",
                }
            ],
        )
        return dict(authority)

    request = _create_request(
        scheduled_future_speech_authority=build_authority(
            trigger_local="2026-05-16 10:00",
        ),
    )
    task_identity_key = lifecycle.build_task_identity_key(request)
    first_task = lifecycle._build_enqueueing_task_doc(
        request,
        task_identity_key=task_identity_key,
    )

    created = await repository.insert_or_get_active_accepted_task(
        first_task,
        source_message_id="message-001",
        observed_at="2026-05-15T21:00:00+00:00",
    )
    assert created["status"] == "created"

    before_mismatch = deepcopy(fake_db.accepted_tasks.documents[0])

    mismatching_duplicate = deepcopy(first_task)
    mismatching_duplicate["accepted_task_id"] = "task-mismatching-duplicate"
    mismatching_duplicate["scheduled_future_speech_authority"] = (
        build_authority(trigger_local="2026-05-16 11:00")
    )
    with pytest.raises(
        DatabaseOperationError,
        match="different goal_continuation_ref or scheduled authority",
    ):
        await repository.insert_or_get_active_accepted_task(
            mismatching_duplicate,
            source_message_id="message-999",
            observed_at="2026-05-15T21:05:00+00:00",
        )

    current = fake_db.accepted_tasks.documents[0]
    assert current == before_mismatch
    assert current["related_source_message_ids"] == ["message-001"]
    assert current["updated_at"] == "2026-05-15T21:00:00+00:00"


@pytest.mark.asyncio
async def test_repository_rejects_v1_task_document() -> None:
    """The big-bang persistence boundary accepts only v2 task rows."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository
    from kazusa_ai_chatbot.db.errors import DatabaseOperationError

    request = _create_request()
    task = lifecycle._build_enqueueing_task_doc(
        request,
        task_identity_key=lifecycle.build_task_identity_key(request),
    )
    task["schema_version"] = "accepted_task.v1"

    with pytest.raises(DatabaseOperationError, match="schema_version"):
        await repository.insert_or_get_active_accepted_task(
            task,
            source_message_id="message-001",
            observed_at="2026-05-15T21:00:00+00:00",
        )


@pytest.mark.asyncio
async def test_v2_index_setup_replaces_conflicting_v1_named_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The big-bang startup drops the old action-kind index definition."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    assert lifecycle is not None

    fake_db = _FakeDb()
    index_name = "accepted_task_open_coding_run_context_lookup"
    fake_db.accepted_tasks.indexes[index_name] = {
        "key": [
            ("source_platform", 1),
            ("source_channel_id", 1),
            ("requester_global_user_id", 1),
            ("action_kind", 1),
            ("updated_at", -1),
        ],
        "partialFilterExpression": {
            "coding_run_context.followup_open": True,
        },
    }
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))

    await repository.ensure_accepted_task_indexes()

    assert fake_db.accepted_tasks.dropped_indexes == [index_name]
    replacement = fake_db.accepted_tasks.indexes[index_name]
    assert ("task_kind", 1) in replacement["key"]
    assert ("action_kind", 1) not in replacement["key"]


@pytest.mark.asyncio
async def test_mark_pending_records_internal_executor_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pending transition reserves the stable id before job insertion."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))

    created = await lifecycle.create_or_return_active_accepted_task(
        _create_request(),
    )
    pending = await lifecycle.mark_accepted_task_pending(
        accepted_task_id=created["task"]["accepted_task_id"],
        executor_ref="job-001",
        updated_at="2026-05-15T21:00:01+00:00",
    )

    assert pending is not None
    assert pending["state"] == "pending"
    assert pending["executor_kind"] == "background_work"
    assert pending["executor_ref"] == "job-001"

    repeated = await lifecycle.mark_accepted_task_pending(
        accepted_task_id=created["task"]["accepted_task_id"],
        executor_ref="job-001",
        updated_at="2026-05-15T21:00:02+00:00",
    )
    assert repeated is not None
    assert repeated["state"] == "pending"

    conflicting = await lifecycle.mark_accepted_task_pending(
        accepted_task_id=created["task"]["accepted_task_id"],
        executor_ref="job-other",
        updated_at="2026-05-15T21:00:03+00:00",
    )
    assert conflicting is None


@pytest.mark.asyncio
async def test_enqueue_failure_releases_active_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed internal enqueue must not block the same task forever."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))

    first = await lifecycle.create_or_return_active_accepted_task(
        _create_request(),
    )
    failed = await lifecycle.mark_accepted_task_enqueue_failed(
        accepted_task_id=first["task"]["accepted_task_id"],
        failure_summary="worker job insert failed",
        updated_at="2026-05-15T21:00:01+00:00",
    )
    second = await lifecycle.create_or_return_active_accepted_task(
        _create_request(source_message_id="message-002"),
    )

    assert failed is not None
    assert failed["state"] == "enqueue_failed"
    assert "active_identity_key" not in failed
    assert second["status"] == "created"
    assert second["task"]["accepted_task_id"] != first["task"]["accepted_task_id"]
    assert len(fake_db.accepted_tasks.documents) == 2


@pytest.mark.asyncio
async def test_recover_stale_enqueueing_releases_old_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale enqueueing locks should be moved to enqueue_failed."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))
    first = await lifecycle.create_or_return_active_accepted_task(
        _create_request(storage_timestamp_utc="2026-05-15T21:00:00+00:00"),
    )

    recovered_count = await lifecycle.recover_stale_enqueueing_tasks(
        stale_before_utc="2026-05-15T21:05:00+00:00",
        recovered_at="2026-05-15T21:06:00+00:00",
    )
    second = await lifecycle.create_or_return_active_accepted_task(
        _create_request(
            source_message_id="message-002",
            storage_timestamp_utc="2026-05-15T21:06:01+00:00",
        ),
    )

    old_task = fake_db.accepted_tasks.documents[0]
    assert recovered_count == 1
    assert old_task["accepted_task_id"] == first["task"]["accepted_task_id"]
    assert old_task["state"] == "enqueue_failed"
    assert "active_identity_key" not in old_task
    assert second["status"] == "created"


@pytest.mark.asyncio
async def test_status_check_returns_latest_active_task_for_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress checks should read accepted-task state without creating work."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))
    created = await lifecycle.create_or_return_active_accepted_task(
        _create_request(),
    )
    await lifecycle.mark_accepted_task_pending(
        accepted_task_id=created["task"]["accepted_task_id"],
        executor_ref="job-001",
        updated_at="2026-05-15T21:00:01+00:00",
    )

    status = await lifecycle.check_accepted_task_status({
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "debug-user-001",
    })

    assert status["status"] == "active"
    assert status["task"]["state"] == "pending"
    assert status["task"]["accepted_task_summary"] == (
        "Remind the user to drink water."
    )
    assert len(fake_db.accepted_tasks.documents) == 1


@pytest.mark.asyncio
async def test_status_check_with_incomplete_scope_does_not_match_global_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress lookups should fail closed when trusted scope is incomplete."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))
    created = await lifecycle.create_or_return_active_accepted_task(
        _create_request(),
    )
    await lifecycle.mark_accepted_task_pending(
        accepted_task_id=created["task"]["accepted_task_id"],
        executor_ref="job-001",
        updated_at="2026-05-15T21:00:01+00:00",
    )

    status = await lifecycle.check_accepted_task_status({
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "requester_global_user_id": "global-user-001",
    })

    assert status == {"status": "none"}


async def test_recover_stale_delivery_in_progress_restores_retryable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted delivery claims should not stay stuck in progress."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))

    created = await lifecycle.create_or_return_active_accepted_task(
        _create_request(source_message_id="message-001"),
    )
    await lifecycle.mark_accepted_task_pending(
        accepted_task_id=created["task"]["accepted_task_id"],
        executor_ref="background_work_job:job-001",
        updated_at="2026-05-16T09:00:01+00:00",
    )
    await lifecycle.mark_accepted_task_running(
        accepted_task_id=created["task"]["accepted_task_id"],
        started_at="2026-05-16T09:00:01.500000+00:00",
    )
    await lifecycle.mark_tool_result_ready(
        accepted_task_id=created["task"]["accepted_task_id"],
        artifact_text="Drink water.",
        result_summary="Reminder text is ready.",
        completed_at="2026-05-16T09:00:02+00:00",
    )
    await lifecycle.mark_accepted_task_delivery_in_progress(
        accepted_task_id=created["task"]["accepted_task_id"],
        delivery_tracking_id="delivery-001",
        updated_at="2026-05-16T09:00:03+00:00",
    )

    recovered_count = await lifecycle.recover_stale_delivery_in_progress_tasks(
        stale_before_utc="2026-05-16T09:01:00+00:00",
        recovered_at="2026-05-16T09:02:00+00:00",
    )

    task = fake_db.accepted_tasks.documents[0]
    assert recovered_count == 1
    assert task["state"] == "delivery_retryable"
    assert task["delivery_failure_summary"] == (
        "Accepted task delivery did not complete."
    )
    assert task["updated_at"] == "2026-05-16T09:02:00+00:00"
    assert task["active_identity_key"] == task["task_identity_key"]


@pytest.mark.asyncio
async def test_terminal_transitions_require_running_and_delivery_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal task writes cannot skip pending, running, or delivery states."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))
    created = await lifecycle.create_or_return_active_accepted_task(
        _create_request(),
    )
    task_id = created["task"]["accepted_task_id"]
    await lifecycle.mark_accepted_task_pending(
        accepted_task_id=task_id,
        executor_ref="job-001",
        updated_at="2026-05-15T21:00:01+00:00",
    )

    premature_result = await lifecycle.mark_tool_result_ready(
        accepted_task_id=task_id,
        artifact_text="Result",
        result_summary="Result is ready.",
        completed_at="2026-05-15T21:00:02+00:00",
    )
    assert premature_result is None

    await lifecycle.mark_accepted_task_running(
        accepted_task_id=task_id,
        started_at="2026-05-15T21:00:03+00:00",
    )
    premature_claim = await lifecycle.mark_accepted_task_delivery_in_progress(
        accepted_task_id=task_id,
        delivery_tracking_id="delivery-premature",
        updated_at="2026-05-15T21:00:03.500000+00:00",
    )
    assert premature_claim is None
    ready = await lifecycle.mark_tool_result_ready(
        accepted_task_id=task_id,
        artifact_text="Result",
        result_summary="Result is ready.",
        completed_at="2026-05-15T21:00:04+00:00",
    )
    assert ready is not None
    premature_delivery = await lifecycle.mark_accepted_task_delivered(
        accepted_task_id=task_id,
        delivered_conversation_message_id="message-001",
        delivered_at="2026-05-15T21:00:05+00:00",
    )
    assert premature_delivery is None

    await lifecycle.mark_accepted_task_delivery_in_progress(
        accepted_task_id=task_id,
        delivery_tracking_id="delivery-001",
        updated_at="2026-05-15T21:00:06+00:00",
    )
    delivered = await lifecycle.mark_accepted_task_delivered(
        accepted_task_id=task_id,
        delivered_conversation_message_id="message-001",
        delivered_at="2026-05-15T21:00:07+00:00",
    )
    assert delivered is not None


@pytest.mark.asyncio
async def test_future_speak_completion_requires_running_future_speak_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler completion cannot terminalize another accepted-task kind."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks as repository

    fake_db = _FakeDb()
    monkeypatch.setattr(repository, "get_db", _fake_get_db(fake_db))
    future_request = _create_request()
    future_request["task_kind"] = "future_speak"
    future_request["semantic_objective"] = "Remind the user tomorrow."
    future_request["accepted_task_summary"] = "Remind the user tomorrow."
    future = await lifecycle.create_or_return_active_accepted_task(future_request)
    future_id = future["task"]["accepted_task_id"]
    await lifecycle.mark_accepted_task_pending(
        accepted_task_id=future_id,
        executor_ref="job-future",
        updated_at="2026-05-15T21:00:01+00:00",
    )
    await lifecycle.mark_accepted_task_running(
        accepted_task_id=future_id,
        started_at="2026-05-15T21:00:02+00:00",
    )

    completed = await lifecycle.mark_future_speak_accepted_task_delivered(
        accepted_task_id=future_id,
        delivered_at="2026-05-15T21:00:03+00:00",
    )
    assert completed is not None
    assert completed["state"] == "delivered"

    task = await lifecycle.create_or_return_active_accepted_task(
        _create_request(
            task_kind="task_resolution",
            semantic_objective="Resolve one bounded task.",
            accepted_task_summary="Resolve one bounded task.",
            goal_continuation_ref=_goal_continuation_ref(),
        )
    )
    task_id = task["task"]["accepted_task_id"]
    await lifecycle.mark_accepted_task_pending(
        accepted_task_id=task_id,
        executor_ref="job-task",
        updated_at="2026-05-15T21:00:04+00:00",
    )
    await lifecycle.mark_accepted_task_running(
        accepted_task_id=task_id,
        started_at="2026-05-15T21:00:05+00:00",
    )
    rejected = await lifecycle.mark_future_speak_accepted_task_delivered(
        accepted_task_id=task_id,
        delivered_at="2026-05-15T21:00:06+00:00",
    )
    assert rejected is None


class _FakeInsertResult:
    inserted_id = "fake-inserted-id"


class _FakeUpdateResult:
    def __init__(self, modified_count: int) -> None:
        self.modified_count = modified_count


class _FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def sort(self, field_name: str, direction: int) -> "_FakeCursor":
        reverse = direction < 0
        self._rows = sorted(
            self._rows,
            key=lambda row: str(row.get(field_name, "")),
            reverse=reverse,
        )
        return self

    def limit(self, limit: int) -> "_FakeCursor":
        self._rows = self._rows[:limit]
        return self

    async def to_list(self, length: int) -> list[dict[str, Any]]:
        return [deepcopy(row) for row in self._rows[:length]]


class _FakeAcceptedTaskCollection:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.indexes: dict[str, dict[str, Any]] = {}
        self.dropped_indexes: list[str] = []

    async def create_index(self, *args: object, **kwargs: object) -> str:
        keys = args[0]
        if isinstance(keys, str):
            normalized_keys = [(keys, 1)]
        else:
            normalized_keys = list(keys)
        index_name = str(kwargs["name"])
        self.indexes[index_name] = {
            "key": normalized_keys,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"name", "unique"}
            },
        }
        return index_name

    async def index_information(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.indexes)

    async def drop_index(self, index_name: str) -> None:
        self.dropped_indexes.append(index_name)
        self.indexes.pop(index_name, None)

    async def insert_one(self, document: dict[str, Any]) -> _FakeInsertResult:
        active_identity_key = document.get("active_identity_key")
        if active_identity_key:
            for existing in self.documents:
                if existing.get("active_identity_key") == active_identity_key:
                    raise DuplicateKeyError("duplicate active task")
        self.documents.append(deepcopy(document))
        return _FakeInsertResult()

    async def find_one(
        self,
        query: dict[str, Any],
        projection: dict[str, int] | None = None,
        sort: list[tuple[str, int]] | None = None,
    ) -> dict[str, Any] | None:
        del projection
        rows = [row for row in self.documents if _matches(row, query)]
        if sort:
            for field_name, direction in reversed(sort):
                rows = sorted(
                    rows,
                    key=lambda row: str(row.get(field_name, "")),
                    reverse=direction < 0,
                )
        if not rows:
            return None
        return deepcopy(rows[0])

    def find(
        self,
        query: dict[str, Any],
        projection: dict[str, int] | None = None,
    ) -> _FakeCursor:
        del projection
        rows = [deepcopy(row) for row in self.documents if _matches(row, query)]
        return _FakeCursor(rows)

    async def find_one_and_update(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
        **kwargs: object,
    ) -> dict[str, Any] | None:
        del kwargs
        for index, row in enumerate(self.documents):
            if _matches(row, query):
                updated = deepcopy(row)
                _apply_update(updated, update)
                self.documents[index] = updated
                return deepcopy(updated)
        return None

    async def update_many(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
    ) -> _FakeUpdateResult:
        modified_count = 0
        for index, row in enumerate(self.documents):
            if _matches(row, query):
                updated = deepcopy(row)
                _apply_update(updated, update)
                self.documents[index] = updated
                modified_count += 1
        return _FakeUpdateResult(modified_count)


class _FakeDb:
    def __init__(self) -> None:
        self.accepted_tasks = _FakeAcceptedTaskCollection()

    def __getitem__(self, name: str) -> _FakeAcceptedTaskCollection:
        if name != "accepted_tasks":
            raise KeyError(name)
        return self.accepted_tasks


def _fake_get_db(fake_db: _FakeDb):
    async def get_db() -> _FakeDb:
        return fake_db

    return get_db


def _matches(row: dict[str, Any], query: dict[str, Any]) -> bool:
    for field_name, expected in query.items():
        actual = row.get(field_name)
        if isinstance(expected, dict) and (
            "$in" in expected or "$lte" in expected
        ):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$lte" in expected and not (str(actual) <= str(expected["$lte"])):
                return False
            continue
        if actual != expected:
            return False
    return True


def _apply_update(row: dict[str, Any], update: dict[str, Any]) -> None:
    set_values = update.get("$set")
    if isinstance(set_values, dict):
        row.update(deepcopy(set_values))
    unset_values = update.get("$unset")
    if isinstance(unset_values, dict):
        for field_name in unset_values:
            row.pop(field_name, None)
    add_to_set_values = update.get("$addToSet")
    if isinstance(add_to_set_values, dict):
        for field_name, value in add_to_set_values.items():
            current = row.setdefault(field_name, [])
            if value not in current:
                current.append(value)
