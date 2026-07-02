"""Tests for background-work result-ready delivery boundaries."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_chain_core.prompt_selection import (
    build_cognition_prompt_source_payload,
    select_cognition_prompt_variant,
)


def _completed_job() -> dict:
    """Build one completed generic background-work job document."""

    job = {
        "job_id": "job-001",
        "task_brief": "Generate a Fibonacci function snippet.",
        "worker": "text_artifact",
        "status": "completed",
        "artifact_text": "def fib(n): return n",
        "failure_summary": "",
        "result_summary": "Generated a compact Fibonacci snippet.",
        "worker_metadata": {"task_type": "coding_snippet"},
        "source_platform": "debug",
        "source_channel_id": "debug-private-1",
        "source_channel_type": "private",
        "source_message_id": "message-1",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-1",
        "requester_platform_user_id": "platform-user-1",
        "requester_display_name": "Test User",
        "created_at": "2026-06-06T00:00:00+00:00",
        "completed_at": "2026-06-06T00:01:00+00:00",
    }
    return job


def _accepted_task_completed_job() -> dict:
    """Build one completed job for a new accepted-task-backed request."""

    job = _completed_job()
    job["accepted_task_id"] = "task-001"
    job["task_identity_key"] = "accepted_task:v1:abc"
    return job


def _patch_delivery_recovery(
    monkeypatch: pytest.MonkeyPatch,
    delivery_module,
    *,
    background_count: int = 0,
    accepted_task_count: int = 0,
) -> tuple[AsyncMock, AsyncMock]:
    recover_background = AsyncMock(return_value=background_count)
    recover_accepted_task = AsyncMock(return_value=accepted_task_count)
    monkeypatch.setattr(
        delivery_module,
        "recover_stale_background_work_delivery_in_progress",
        recover_background,
    )
    monkeypatch.setattr(
        delivery_module,
        "recover_stale_delivery_in_progress_tasks",
        recover_accepted_task,
    )
    return recover_background, recover_accepted_task


def test_result_source_builder_creates_prompt_safe_episode() -> None:
    """Completed background work should become source-bound cognition."""

    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )
    builder = getattr(result_source, "build_result_ready_episode_from_job")

    episode = builder(_completed_job())
    serialized = json.dumps(episode, ensure_ascii=False).lower()

    assert episode["trigger_source"] == "background_work_result_ready"
    assert episode["input_sources"] == ["background_work_result"]
    assert episode["output_mode"] == "visible_reply"
    assert "fibonacci" in serialized
    assert episode["percepts"][0]["metadata"]["source_platform_bot_id"] == "bot-1"
    for forbidden in (
        "lease",
        "retry",
        "job_ref",
        "adapter_id",
        "mongodb",
    ):
        assert forbidden not in serialized


def test_result_source_payload_uses_generic_background_work_metadata() -> None:
    """Prompt payload should keep task/result context without legacy labels."""

    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )
    episode = result_source.build_result_ready_episode_from_job(_completed_job())
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l2d_action_selection",
    )

    payload = build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    )

    result_payload = payload["background_work_result"]
    metadata = result_payload["metadata"]
    assert metadata == {
        "task_brief": "Generate a Fibonacci function snippet.",
        "failure_summary": "",
        "result_summary": "Generated a compact Fibonacci snippet.",
        "source_character_name": "Test Character",
    }
    serialized_payload = json.dumps(payload, ensure_ascii=False)
    assert "work_kind" not in serialized_payload
    assert "objective_summary" not in serialized_payload
    assert "source_platform_bot_id" not in serialized_payload
    assert "worker_metadata" not in serialized_payload


def test_accepted_task_result_source_builder_creates_prompt_safe_episode() -> None:
    """Accepted-task jobs should produce accepted-task result-ready cognition."""

    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )

    episode = result_source.build_result_ready_episode_from_job(
        _accepted_task_completed_job()
    )
    serialized = json.dumps(episode, ensure_ascii=False).lower()

    assert episode["trigger_source"] == "accepted_task_result_ready"
    assert episode["input_sources"] == ["accepted_task_result"]
    assert episode["output_mode"] == "visible_reply"
    metadata = episode["percepts"][0]["metadata"]
    assert metadata["accepted_task_id"] == "task-001"
    assert metadata["accepted_task_summary"] == (
        "Generate a Fibonacci function snippet."
    )
    for forbidden in (
        "background_work_result_ready",
        "background_work_result",
        "worker_metadata",
        "worker",
        "job_ref",
        "queue_state",
    ):
        assert forbidden not in serialized


def test_accepted_task_result_payload_uses_semantic_metadata() -> None:
    """Prompt payload should expose accepted-task result fields only."""

    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )
    episode = result_source.build_result_ready_episode_from_job(
        _accepted_task_completed_job()
    )
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l2d_action_selection",
    )

    payload = build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    )

    result_payload = payload["accepted_task_result"]
    metadata = result_payload["metadata"]
    assert metadata == {
        "accepted_task_summary": "Generate a Fibonacci function snippet.",
        "failure_summary": "",
        "result_summary": "Generated a compact Fibonacci snippet.",
        "source_character_name": "Test Character",
    }
    serialized_payload = json.dumps(payload, ensure_ascii=False).lower()
    for forbidden in (
        "background_work_result",
        "worker_metadata",
        "worker",
        "job_ref",
        "queue_state",
    ):
        assert forbidden not in serialized_payload


@pytest.mark.asyncio
async def test_service_result_ready_delivery_uses_dispatcher_boundary(
    monkeypatch,
) -> None:
    """Result delivery should run cognition then delegate sending to service."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )
    episode = result_source.build_result_ready_episode_from_job(_completed_job())
    handle_send_message = AsyncMock(return_value={
        "conversation_message_id": "conversation-001",
        "delivery_tracking_id": "delivery-001",
        "adapter_message_id": "adapter-001",
    })
    persona_supervisor2 = AsyncMock(return_value={
        "final_dialog": ["@Test User Here is the requested result."],
        "consolidation_state": {
            "final_dialog": ["@Test User Here is the requested result."],
        },
    })
    post_turn = AsyncMock()

    monkeypatch.setattr(service_module, "_adapter_registry", object())
    monkeypatch.setattr(
        service_module,
        "_refresh_runtime_character_state",
        AsyncMock(),
    )
    monkeypatch.setattr(
        service_module,
        "get_user_profile",
        AsyncMock(return_value={"global_user_id": "global-user-1"}),
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-global-1"),
    )
    monkeypatch.setattr(
        service_module,
        "compose_character_profile",
        lambda *_args, **_kwargs: {
            "name": "Test Character",
            "global_user_id": "character-global-1",
        },
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "build_promoted_reflection_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service_module,
        "load_conversation_episode_state",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(service_module, "persona_supervisor2", persona_supervisor2)
    monkeypatch.setattr(service_module, "handle_send_message", handle_send_message)
    monkeypatch.setattr(
        service_module,
        "_run_background_work_result_post_turn",
        post_turn,
    )

    result = await service_module._deliver_background_work_result_episode(
        episode,
    )

    assert result == {
        "status": "delivered",
        "conversation_message_id": "conversation-001",
        "delivery_tracking_id": "delivery-001",
        "adapter_message_id": "adapter-001",
    }
    persona_state = persona_supervisor2.await_args.args[0]
    assert persona_state["cognitive_episode"] == episode
    assert persona_state["reason_to_respond"] == "background_work_result_ready"
    assert persona_state["user_input"].startswith(
        "Background work result is completed."
    )
    send_args = handle_send_message.await_args.args[0]
    assert send_args["text"] == "@Test User Here is the requested result."
    assert send_args["target_channel"] == "debug-private-1"
    assert send_args["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "display_name": "Test User",
            "platform_user_id": "platform-user-1",
        }
    ]
    dispatch_context = handle_send_message.await_args.args[1]
    assert dispatch_context.bot_permission_role == "background_work_result"
    assert dispatch_context.source_platform_bot_id == "bot-1"
    post_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_delivery_tick_syncs_accepted_task_delivery_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted-task delivery should move through in-progress to delivered."""

    delivery_module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.delivery"
    )
    job = _accepted_task_completed_job()
    job["delivery_state"] = "ready"
    find_jobs = AsyncMock(return_value=[job])
    mark_job_in_progress = AsyncMock(return_value={
        **job,
        "delivery_state": "in_progress",
        "delivery_tracking_id": "delivery-001",
    })
    mark_job_delivered = AsyncMock(return_value={**job, "status": "delivered"})
    mark_task_in_progress = AsyncMock()
    mark_task_delivered = AsyncMock()
    mark_task_failed = AsyncMock()
    _patch_delivery_recovery(monkeypatch, delivery_module)
    deliver_episode = AsyncMock(return_value={
        "status": "delivered",
        "conversation_message_id": "conversation-001",
    })

    monkeypatch.setattr(
        delivery_module,
        "find_deliverable_background_work_jobs",
        find_jobs,
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivery_in_progress",
        mark_job_in_progress,
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivered",
        mark_job_delivered,
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_accepted_task_delivery_in_progress",
        mark_task_in_progress,
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_accepted_task_delivered",
        mark_task_delivered,
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_accepted_task_delivery_failed",
        mark_task_failed,
    )

    result = await delivery_module.run_background_work_delivery_tick(
        deliver_result_episode_func=deliver_episode,
        limit=1,
    )

    assert result == {
        "processed_count": 1,
        "delivered_count": 1,
        "failed_count": 0,
        "recovered_count": 0,
    }
    mark_task_in_progress.assert_awaited_once()
    assert mark_task_in_progress.await_args.kwargs["accepted_task_id"] == (
        "task-001"
    )
    mark_task_delivered.assert_awaited_once()
    assert mark_task_delivered.await_args.kwargs["accepted_task_id"] == (
        "task-001"
    )
    assert mark_task_delivered.await_args.kwargs[
        "delivered_conversation_message_id"
    ] == "conversation-001"
    mark_task_failed.assert_not_awaited()
    delivered_episode = deliver_episode.await_args.args[0]
    assert delivered_episode["trigger_source"] == "accepted_task_result_ready"


@pytest.mark.asyncio
async def test_delivery_tick_recovers_stale_delivery_claims_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stuck internal and accepted-task delivery claims should be retried."""

    delivery_module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.delivery"
    )
    find_jobs = AsyncMock(return_value=[])
    recover_background, recover_accepted_task = _patch_delivery_recovery(
        monkeypatch,
        delivery_module,
        background_count=2,
        accepted_task_count=1,
    )
    monkeypatch.setattr(
        delivery_module,
        "find_deliverable_background_work_jobs",
        find_jobs,
    )
    monkeypatch.setattr(
        delivery_module,
        "storage_utc_now",
        lambda: datetime(2026, 5, 16, 9, 10, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        delivery_module,
        "BACKGROUND_WORK_WORKER_LEASE_SECONDS",
        120,
    )

    result = await delivery_module.run_background_work_delivery_tick(
        deliver_result_episode_func=AsyncMock(),
        limit=1,
    )

    assert result == {
        "processed_count": 0,
        "delivered_count": 0,
        "failed_count": 0,
        "recovered_count": 3,
    }
    recover_background.assert_awaited_once_with(
        stale_before_utc="2026-05-16T09:08:00+00:00",
        recovered_at="2026-05-16T09:10:00+00:00",
    )
    recover_accepted_task.assert_awaited_once_with(
        stale_before_utc="2026-05-16T09:08:00+00:00",
        recovered_at="2026-05-16T09:10:00+00:00",
    )
    find_jobs.assert_awaited_once_with(limit=1)


def test_delivery_failure_summary_field_exists_in_job_schema() -> None:
    """Job doc should carry delivery_failure_summary separate from failure_summary."""

    models = importlib.import_module("kazusa_ai_chatbot.background_work.models")
    annotations = models.BackgroundWorkJobDoc.__annotations__
    assert "delivery_failure_summary" in annotations
    assert "failure_summary" in annotations


def test_delivery_failure_summary_initialized_empty() -> None:
    """New jobs should start with empty delivery_failure_summary."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    build = getattr(jobs, "_build_job_document")
    request = {
        "action_attempt_id": "action_attempt:dfs-001",
        "idempotency_key": "background_work:dfs-001",
        "task_brief": "Test delivery failure field.",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test",
        "source_channel_type": "private",
        "source_message_id": "message-001",
        "source_platform_bot_id": "bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "debug-user-001",
        "requester_display_name": "Test User",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "storage_timestamp_utc": "2026-06-06T00:00:00+00:00",
    }
    job = build(
        request,
        job_id="job-dfs-001",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )
    assert job["delivery_failure_summary"] == ""
    assert job["failure_summary"] == ""


def test_delivery_retry_cap_query_uses_max_delivery_attempts() -> None:
    """find_deliverable should filter by delivery_attempt_count cap."""

    db_module = importlib.import_module(
        "kazusa_ai_chatbot.db.background_work_jobs"
    )
    import inspect

    sig = inspect.signature(db_module.find_deliverable_background_work_jobs)
    assert "max_delivery_attempts" in sig.parameters
    param = sig.parameters["max_delivery_attempts"]
    assert param.default > 0
