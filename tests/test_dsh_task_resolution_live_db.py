"""Real-Mongo lifecycle and fencing coverage for DSH task resolution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

import pytest
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from tests.task_resolution_test_helpers import _context, _goal_continuation_ref


async def _live_database() -> Any:
    """Return the guarded live test database after an explicit connectivity check."""

    from kazusa_ai_chatbot.db._client import get_db

    try:
        database = await get_db()
        await database.command("ping")
    except ConnectionFailure as exc:
        pytest.fail(f"Mongo live integration unavailable: {exc}")
    return database


def _binding(suffix: str, *, session_id: str | None = None) -> dict[str, Any]:
    """Build one unique valid DSH task binding row."""

    session = session_id or f"live-session-{suffix}"
    objective = f"Resolve live objective {suffix}."
    return {
        "schema_version": "dsh_task_binding.v1",
        "task_session_id": session,
        "semantic_objective": objective,
        "goal_continuation_ref": _goal_continuation_ref(),
        "source_scope": {
            "schema_version": "dsh_task_source_scope.v1",
            "platform": "debug",
            "channel_id": f"live-channel-{suffix}",
            "channel_type": "private",
            "requester_global_user_id": f"live-user-{suffix}",
            "requester_platform_user_id": f"live-platform-user-{suffix}",
            "source_message_id": f"live-message-{suffix}",
            "source_platform_bot_id": "live-bot",
        },
        "state": "queued",
        "start_spec": _start_spec(objective),
        "resolution_thread_id": None,
        "segment_id": None,
        "resolution_ref": None,
        "operation_generation": 0,
        "current_accepted_task_id": None,
        "current_background_work_job_id": None,
        "latest_task_resolution_result": None,
        "revision": 0,
        "created_at": "2026-08-30T00:00:00Z",
        "updated_at": "2026-08-30T00:00:00Z",
    }


def _start_spec(objective: str) -> dict[str, object]:
    """Build a complete canonical start carrier for a live binding row."""

    from kazusa_ai_chatbot.task_resolution.service import _build_start_spec

    context = _context()
    context["prompt_message_context"] = {
        **context["prompt_message_context"],
        "semantic_goal": objective,
    }
    return _build_start_spec(
        {
            "semantic_goal": objective,
            "reason": "A bounded live repository test request.",
            "evidence_handles": [],
            "start_in_background": False,
        },
        context,
    )


def _resolution_ref(suffix: str, session_id: str) -> dict[str, Any]:
    """Build one complete generation-zero DSH resolution reference."""

    return {
        "schema_version": "dsh_resolution_ref.v1",
        "resolution_thread_id": f"live-thread-{suffix}",
        "segment_id": f"live-segment-{suffix}",
        "dsh_session_id": session_id,
        "activation_id": f"live-activation-{suffix}",
        "lease_epoch": 1,
        "document_revision": 1,
        "last_committed_seq": 1,
    }


def _task_request(suffix: str) -> dict[str, object]:
    """Build one unique task-resolution accepted-task request."""

    return {
        "task_kind": "task_resolution",
        "semantic_objective": f"Resolve live objective {suffix}.",
        "accepted_task_summary": "Resolve one live bounded objective.",
        "goal_continuation_ref": _goal_continuation_ref(),
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "source_trigger_source": "user_message",
        "source_platform": "debug",
        "source_channel_id": f"live-channel-{suffix}",
        "source_channel_type": "private",
        "source_message_id": f"live-message-{suffix}",
        "source_platform_bot_id": "live-bot",
        "source_character_name": "Test Character",
        "requester_global_user_id": f"live-user-{suffix}",
        "requester_platform_user_id": f"live-platform-user-{suffix}",
        "requester_display_name": "Live Test User",
        "storage_timestamp_utc": "2026-08-30T00:00:00Z",
    }


def _task_result() -> dict[str, object]:
    """Build a complete valid terminal task-resolution result."""

    context = _context()
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve one bounded goal.",
        "status": "resolved",
        "scene_context": context["scene_context"],
        "goal_continuation_ref": context["goal_continuation_ref"],
        "evidence_state": "complete",
        "evidence_excerpts": ["bounded live evidence"],
        "evidence_handles": ["live-evidence-1"],
        "prompt_safe_summary": "The live task is complete.",
        "evidence": [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "live-evidence-1",
            "task_node_id": "dsh",
            "specialist": "dsh",
            "summary": "bounded live evidence",
            "provenance_refs": ["live-receipt-1"],
            "limitations": [],
        }],
        "completed_subgoals": ["bounded live evidence"],
        "remaining_needs": [],
        "checkpoint": {},
        "coding_run_context": {},
    }


def _interaction(suffix: str, thread_id: str, segment_id: str) -> dict[str, Any]:
    """Build one one-shot interaction grant for the real interaction store."""

    return {
        "schema_version": "dsh_interaction_pending.v1",
        "interaction_id": f"live-interaction-{suffix}",
        "issuer": "dsh-live-test",
        "nonce": f"live-nonce-{suffix}",
        "request_digest": f"sha256:live-request-{suffix}",
        "decision": {"decision": "allow_once"},
        "status": "open",
        "grant_status": "available",
        "resolution_thread_id": thread_id,
        "segment_id": segment_id,
        "activation_id": f"live-activation-{suffix}",
        "lease_epoch": 1,
        "tool_name": "kazusa_remember_information",
        "arguments_digest": f"sha256:live-args-{suffix}",
        "workspace_fingerprint": "sha256:live-workspace",
        "scope_fingerprint": "sha256:live-scope",
        "policy_epoch": "dsh-standard-policy-v2",
        "expires_at": "2099-01-01T00:00:00Z",
    }


def _job(suffix: str, session_id: str) -> dict[str, Any]:
    """Build one complete queued task-orchestrator job."""

    continuation_ref = _goal_continuation_ref()
    return {
        "schema_version": "background_work_job.v2",
        "job_id": f"live-job-{suffix}",
        "idempotency_key": f"live-idempotency-{suffix}",
        "source_action_attempt_id": f"live-attempt-{suffix}",
        "source_llm_trace_id": f"live-trace-{suffix}",
        "correlation_write_status": "written",
        "correlation_conflict_source_llm_trace_id": "",
        "accepted_task_id": f"live-accepted-placeholder-{suffix}",
        "task_identity_key": f"live-identity-{suffix}",
        "semantic_objective": f"Resolve live objective {suffix}.",
        "goal_continuation_ref": continuation_ref,
        "status": "queued",
        "delivery_state": "queued",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "source_platform": "debug",
        "source_channel_id": f"live-channel-{suffix}",
        "source_channel_type": "private",
        "source_message_id": f"live-message-{suffix}",
        "source_platform_bot_id": "live-bot",
        "source_character_name": "Test Character",
        "requester_global_user_id": f"live-user-{suffix}",
        "requester_platform_user_id": f"live-platform-user-{suffix}",
        "requester_display_name": "Live Test User",
        "created_at": "2026-08-30T00:00:00Z",
        "updated_at": "2026-08-30T00:00:00Z",
        "lease_owner": None,
        "lease_expires_at": None,
        "attempt_count": 0,
        "max_attempts": 4,
        "requested_worker": "task_orchestrator",
        "worker_payload": {
            "schema_version": "task_orchestrator_worker_payload.v2",
            "operation": "open_dsh_resolution",
            "task_session_id": session_id,
            "operation_generation": 0,
            "control": None,
        },
        "task_execution_context": _context(),
        "task_resolution_result": {},
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


async def _cleanup_live_rows(database: Any, session_id: str, suffix: str) -> None:
    """Remove only rows created by one unique live test invocation."""

    await database["dsh_task_bindings"].delete_many({
        "task_session_id": session_id,
    })
    await database["accepted_tasks"].delete_many({
        "dsh_task_session_id": session_id,
    })
    await database["background_work_jobs"].delete_many({
        "job_id": f"live-job-{suffix}",
    })
    await database["dsh_interaction_store"].delete_many({
        "interaction_id": f"live-interaction-{suffix}",
    })


async def _ensure_live_indexes() -> None:
    """Create all durable indexes exercised by the live lifecycle tests."""

    from kazusa_ai_chatbot.db.accepted_tasks import ensure_accepted_task_indexes
    from kazusa_ai_chatbot.db.background_work_jobs import (
        ensure_background_work_job_indexes,
    )
    from kazusa_ai_chatbot.db.dsh_interactions import ensure_indexes
    from kazusa_ai_chatbot.db.task_resolution_sessions import (
        ensure_task_binding_indexes,
    )

    await ensure_task_binding_indexes()
    await ensure_accepted_task_indexes()
    await ensure_background_work_job_indexes()
    await ensure_indexes()


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_live_binding_promotion_followup_interaction_terminal_and_delivery_are_exactly_once() -> None:
    """Persist one full DSH lifecycle and verify every replay is idempotent."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import (
        accepted_tasks,
        background_work_jobs,
        dsh_interactions,
    )
    from kazusa_ai_chatbot.db import task_resolution_sessions as bindings

    database = await _live_database()
    suffix = uuid4().hex
    session_id = f"live-session-{suffix}"
    thread_id = f"live-thread-{suffix}"
    segment_id = f"live-segment-{suffix}"
    try:
        await _ensure_live_indexes()
        created = await bindings.create_task_binding(_binding(suffix))
        attached_ref = await bindings.attach_resolution_ref(
            task_session_id=session_id,
            expected_revision=0,
            resolution_ref=_resolution_ref(suffix, session_id),
        )
        opening = await bindings.transition_task_binding(
            task_session_id=session_id,
            expected_revision=1,
            expected_state="queued",
            next_state="opening",
            operation_generation=0,
        )
        accepted = await lifecycle.create_or_return_active_accepted_task(
            _task_request(suffix),
            dsh_task_session_id=session_id,
            dsh_operation_generation=0,
        )
        accepted_id = accepted["task"]["accepted_task_id"]
        pending = await lifecycle.mark_accepted_task_pending(
            accepted_task_id=accepted_id,
            executor_ref=f"live-job-{suffix}",
            updated_at="2026-08-30T00:00:01Z",
        )
        assert pending is not None
        job = await background_work_jobs.insert_background_work_job(
            _job(suffix, session_id),
        )
        job_id = job["job_id"]
        accepted_attached = await bindings.attach_accepted_task(
            task_session_id=session_id,
            expected_revision=opening["revision"],
            operation_generation=0,
            accepted_task_id=accepted_id,
        )
        await database["background_work_jobs"].update_one(
            {"job_id": job_id},
            {"$set": {"accepted_task_id": accepted_id}},
        )
        job_attached = await bindings.attach_background_job(
            task_session_id=session_id,
            expected_revision=accepted_attached["revision"],
            operation_generation=0,
            background_work_job_id=job_id,
        )
        checkpointed = await bindings.transition_task_binding(
            task_session_id=session_id,
            expected_revision=job_attached["revision"],
            expected_state="opening",
            next_state="checkpointed",
            operation_generation=0,
        )
        active = await bindings.transition_task_binding(
            task_session_id=session_id,
            expected_revision=checkpointed["revision"],
            expected_state="checkpointed",
            next_state="active",
            operation_generation=0,
        )
        running = await lifecycle.mark_accepted_task_running(
            accepted_task_id=accepted_id,
            started_at="2026-08-30T00:00:02Z",
        )
        assert running is not None

        interaction = _interaction(suffix, thread_id, segment_id)
        first_interaction = await dsh_interactions.create_interaction(interaction)
        replay_interaction = await dsh_interactions.create_interaction(
            deepcopy(interaction),
        )
        assert {
            key: value
            for key, value in replay_interaction.items()
            if key != "_id"
        } == {
            key: value
            for key, value in first_interaction.items()
            if key != "_id"
        }
        consumed = await dsh_interactions.consume_one_shot_grant(
            resolution_thread_id=thread_id,
            segment_id=segment_id,
            activation_id=interaction["activation_id"],
            lease_epoch=1,
            tool_name=interaction["tool_name"],
            arguments_digest=interaction["arguments_digest"],
            workspace_fingerprint=interaction["workspace_fingerprint"],
            scope_fingerprint=interaction["scope_fingerprint"],
            policy_epoch=interaction["policy_epoch"],
            now="2026-08-30T00:00:04Z",
        )
        assert consumed is not None
        assert await dsh_interactions.consume_one_shot_grant(
            resolution_thread_id=thread_id,
            segment_id=segment_id,
            activation_id=interaction["activation_id"],
            lease_epoch=1,
            tool_name=interaction["tool_name"],
            arguments_digest=interaction["arguments_digest"],
            workspace_fingerprint=interaction["workspace_fingerprint"],
            scope_fingerprint=interaction["scope_fingerprint"],
            policy_epoch=interaction["policy_epoch"],
            now="2026-08-30T00:00:04Z",
        ) is None
        delivered_interaction = await dsh_interactions.update_interaction(
            interaction["interaction_id"],
            {
                "status": "delivered",
                "delivered_platform_message_id": f"live-platform-{suffix}",
            },
        )
        assert delivered_interaction["status"] == "delivered"

        terminal = await bindings.transition_task_binding(
            task_session_id=session_id,
            expected_revision=active["revision"],
            expected_state="active",
            next_state="terminal",
            operation_generation=0,
        )
        result = _task_result()
        reconciled = await bindings.reconcile_task_resolution_result(
            task_session_id=session_id,
            expected_revision=terminal["revision"],
            operation_generation=0,
            task_resolution_result=result,
        )
        replay_reconciled = await bindings.reconcile_task_resolution_result(
            task_session_id=session_id,
            expected_revision=terminal["revision"],
            operation_generation=0,
            task_resolution_result=result,
        )
        assert reconciled["latest_task_resolution_result"] == result
        assert replay_reconciled["latest_task_resolution_result"] == result

        ready = await lifecycle.mark_tool_result_ready(
            accepted_task_id=accepted_id,
            artifact_text="Live artifact.",
            result_summary="Live result.",
            completed_at="2026-08-30T00:00:05Z",
            result_kind="resolved",
            completion_status="resolved",
            remaining_needs=[],
        )
        assert ready is not None
        delivery_claim = await lifecycle.mark_accepted_task_delivery_in_progress(
            accepted_task_id=accepted_id,
            delivery_tracking_id=f"live-delivery-{suffix}",
            updated_at="2026-08-30T00:00:06Z",
        )
        assert delivery_claim is not None
        delivered = await lifecycle.mark_accepted_task_delivered(
            accepted_task_id=accepted_id,
            delivered_conversation_message_id=f"live-message-delivery-{suffix}",
            delivered_at="2026-08-30T00:00:07Z",
        )
        assert delivered is not None
        assert await lifecycle.mark_accepted_task_delivery_in_progress(
            accepted_task_id=accepted_id,
            delivery_tracking_id=f"live-replay-{suffix}",
            updated_at="2026-08-30T00:00:08Z",
        ) is None
        delivered_replay = await lifecycle.mark_accepted_task_delivered(
            accepted_task_id=accepted_id,
            delivered_conversation_message_id=f"live-message-delivery-{suffix}",
            delivered_at="2026-08-30T00:00:07Z",
        )
        assert delivered_replay is not None
        assert delivered_replay["dsh_followup_open"] is True

        followup = await accepted_tasks.create_followup(
            accepted_task_id=accepted_id,
            task_session_id=session_id,
            operation="continue",
            instruction="Continue the same bounded objective.",
            action_attempt_id=f"live-followup-attempt-{suffix}",
            operation_generation=1,
            binding=reconciled,
        )
        followup_replay = await accepted_tasks.create_followup(
            accepted_task_id=accepted_id,
            task_session_id=session_id,
            operation="continue",
            instruction="Continue the same bounded objective.",
            action_attempt_id=f"live-followup-attempt-{suffix}",
            operation_generation=1,
            binding=reconciled,
        )
        assert followup["accepted_task_id"] == followup_replay["accepted_task_id"]
        assert followup["dsh_task_session_id"] == session_id
        assert followup["dsh_operation_generation"] == 1
        assert followup["state"] == "pending"

        claimed_job = await background_work_jobs.claim_background_work_job(
            lease_owner=f"live-worker-{suffix}",
            lease_seconds=60,
            now_utc="2026-08-30T00:00:09Z",
            max_attempts=4,
        )
        assert claimed_job is not None
        completed_job = await background_work_jobs.complete_background_work_job(
            job_id=job_id,
            lease_owner=f"live-worker-{suffix}",
            task_resolution_result=result,
            artifact_text="Live artifact.",
            result_summary="Live result.",
            completed_at="2026-08-30T00:00:10Z",
        )
        assert completed_job is not None
        deliverable = await background_work_jobs.find_deliverable_background_work_jobs(
            limit=10,
        )
        assert any(row["job_id"] == job_id for row in deliverable)
        job_delivery_claim = await background_work_jobs.mark_background_work_delivery_in_progress(
            job_id=job_id,
            delivery_tracking_id=f"live-job-delivery-{suffix}",
            started_at="2026-08-30T00:00:11Z",
        )
        assert job_delivery_claim is not None
        job_delivered = await background_work_jobs.mark_background_work_delivered(
            job_id=job_id,
            delivered_conversation_message_id=f"live-job-message-{suffix}",
            delivered_at="2026-08-30T00:00:12Z",
        )
        assert job_delivered is not None
        assert await background_work_jobs.mark_background_work_delivery_in_progress(
            job_id=job_id,
            delivery_tracking_id=f"live-job-replay-{suffix}",
            started_at="2026-08-30T00:00:13Z",
        ) is None
        assert created["task_session_id"] == session_id
        assert attached_ref["resolution_thread_id"] == thread_id
    finally:
        await _cleanup_live_rows(database, session_id, suffix)


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_live_one_open_followup_index_and_generation_cas_reject_duplicates() -> None:
    """Use real unique indexes and Mongo CAS filters to reject duplicates."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.db import accepted_tasks
    from kazusa_ai_chatbot.db import task_resolution_sessions as bindings

    database = await _live_database()
    suffix = uuid4().hex
    session_id = f"live-session-{suffix}"
    try:
        await _ensure_live_indexes()
        binding = await bindings.create_task_binding(_binding(suffix))
        binding_collection = database[bindings.DSH_TASK_BINDINGS_COLLECTION]
        accepted_collection = database[accepted_tasks.ACCEPTED_TASKS_COLLECTION]
        index_info = await accepted_collection.index_information()
        followup_index = index_info[
            accepted_tasks.DSH_FOLLOWUP_UNIQUE_INDEX_NAME
        ]
        assert followup_index["unique"] is True
        assert followup_index["partialFilterExpression"] == {
            "schema_version": "accepted_task.v2",
            "task_kind": "task_resolution",
            "dsh_followup_open": True,
        }
        binding_indexes = await binding_collection.index_information()
        assert binding_indexes[
            "dsh_task_binding_session_unique"
        ]["unique"] is True

        accepted = await lifecycle.create_or_return_active_accepted_task(
            _task_request(suffix),
            dsh_task_session_id=session_id,
            dsh_operation_generation=0,
        )
        accepted_id = accepted["task"]["accepted_task_id"]
        assert await lifecycle.mark_accepted_task_pending(
            accepted_task_id=accepted_id,
            executor_ref=f"live-job-{suffix}",
            updated_at="2026-08-30T00:00:01Z",
        ) is not None
        assert await lifecycle.mark_accepted_task_running(
            accepted_task_id=accepted_id,
            started_at="2026-08-30T00:00:02Z",
        ) is not None
        assert await lifecycle.mark_tool_result_ready(
            accepted_task_id=accepted_id,
            artifact_text="Live artifact.",
            result_summary="Live result.",
            completed_at="2026-08-30T00:00:03Z",
            result_kind="resolved",
            completion_status="resolved",
            remaining_needs=[],
        ) is not None
        assert await lifecycle.mark_accepted_task_delivery_in_progress(
            accepted_task_id=accepted_id,
            delivery_tracking_id=f"live-delivery-{suffix}",
            updated_at="2026-08-30T00:00:04Z",
        ) is not None
        assert await lifecycle.mark_accepted_task_delivered(
            accepted_task_id=accepted_id,
            delivered_conversation_message_id=f"live-message-{suffix}",
            delivered_at="2026-08-30T00:00:05Z",
        ) is not None
        delivered_row = await accepted_collection.find_one(
            {"accepted_task_id": accepted_id},
            {"_id": 0},
        )
        assert delivered_row is not None
        duplicate_open = deepcopy(dict(delivered_row))
        duplicate_open["accepted_task_id"] = f"live-duplicate-task-{suffix}"
        with pytest.raises(DuplicateKeyError):
            await accepted_collection.insert_one(duplicate_open)

        first_claim = await bindings.attach_accepted_task(
            task_session_id=session_id,
            expected_revision=binding["revision"],
            operation_generation=0,
            accepted_task_id=accepted_id,
        )
        with pytest.raises(ValueError, match="revision or state fence"):
            await bindings.attach_accepted_task(
                task_session_id=session_id,
                expected_revision=binding["revision"],
                operation_generation=1,
                accepted_task_id=f"live-other-task-{suffix}",
            )
        with pytest.raises(ValueError, match="revision or state fence"):
            await bindings.transition_task_binding(
                task_session_id=session_id,
                expected_revision=binding["revision"],
                expected_state="queued",
                next_state="opening",
                operation_generation=0,
            )
        assert first_claim["revision"] == binding["revision"] + 1
    finally:
        await _cleanup_live_rows(database, session_id, suffix)


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_live_worker_lease_loss_cannot_overwrite_newer_dsh_binding_revision() -> None:
    """A renewed worker lease and a newer binding revision fence stale writes."""

    from kazusa_ai_chatbot.db import background_work_jobs
    from kazusa_ai_chatbot.db import task_resolution_sessions as bindings

    database = await _live_database()
    suffix = uuid4().hex
    session_id = f"live-session-{suffix}"
    try:
        await _ensure_live_indexes()
        binding = await bindings.create_task_binding(_binding(suffix))
        opening = await bindings.transition_task_binding(
            task_session_id=session_id,
            expected_revision=binding["revision"],
            expected_state="queued",
            next_state="opening",
            operation_generation=0,
        )
        with pytest.raises(ValueError, match="revision or state fence"):
            await bindings.transition_task_binding(
                task_session_id=session_id,
                expected_revision=binding["revision"],
                expected_state="queued",
                next_state="opening",
                operation_generation=0,
            )

        job = await background_work_jobs.insert_background_work_job(
            _job(suffix, session_id),
        )
        job_id = job["job_id"]
        first_claim = await background_work_jobs.claim_background_work_job(
            lease_owner=f"live-worker-a-{suffix}",
            lease_seconds=30,
            now_utc="2026-08-30T00:00:00Z",
            max_attempts=4,
        )
        assert first_claim is not None
        second_claim = await background_work_jobs.claim_background_work_job(
            lease_owner=f"live-worker-b-{suffix}",
            lease_seconds=30,
            now_utc="2026-08-30T00:01:00Z",
            max_attempts=4,
        )
        assert second_claim is not None
        assert second_claim["job_id"] == job_id
        stale_completion = await background_work_jobs.complete_background_work_job(
            job_id=job_id,
            lease_owner=f"live-worker-a-{suffix}",
            task_resolution_result=_task_result(),
            artifact_text="stale artifact",
            result_summary="stale result",
            completed_at="2026-08-30T00:01:01Z",
        )
        assert stale_completion is None
        current_completion = await background_work_jobs.complete_background_work_job(
            job_id=job_id,
            lease_owner=f"live-worker-b-{suffix}",
            task_resolution_result=_task_result(),
            artifact_text="current artifact",
            result_summary="current result",
            completed_at="2026-08-30T00:01:02Z",
        )
        assert current_completion is not None
        assert current_completion["result_summary"] == "current result"
        assert opening["revision"] == binding["revision"] + 1
    finally:
        await _cleanup_live_rows(database, session_id, suffix)
