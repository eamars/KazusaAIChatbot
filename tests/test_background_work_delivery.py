"""Tests for background-work result-ready delivery boundaries."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.action_spec.results import build_text_surface_output
from tests.cognition_test_helpers import (
    canonical_episode_identity_snapshot,
    canonical_service_character_profile,
)
from tests.task_resolution_test_helpers import (
    _goal_continuation_ref,
    accepted_task_completed_job,
    recorded_task_checkpoint,
    resume_queue_request,
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
        "goal_continuation_ref": _goal_continuation_ref(),
    }
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


def test_result_source_builder_requires_accepted_task_identity() -> None:
    """Completed result delivery must be tied to an accepted task."""

    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )

    with pytest.raises(ValueError):
        result_source.build_result_ready_episode_from_job(_completed_job())


def test_tool_result_source_builder_creates_prompt_safe_episode() -> None:
    """Accepted-task jobs should produce canonical tool-result cognition."""

    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )

    episode = result_source.build_result_ready_episode_from_job(
        accepted_task_completed_job()
    )
    serialized = json.dumps(episode, ensure_ascii=False).lower()

    assert episode["trigger_source"] == "tool_result"
    assert episode["percepts"][0]["source_kind"] == "tool_result"
    assert episode["origin_metadata"]["task_id"] == "task-001"
    assert episode["origin_metadata"]["task_kind"] == "accepted_task"
    assert episode["origin_metadata"]["platform_message_id"] == (
        "tool-result:task-001"
    )
    assert episode["origin_metadata"]["source_message_id"] == "message-1"
    assert episode["origin_metadata"]["source_llm_trace_id"] == (
        "llmtrace-parent-1"
    )
    assert episode["origin_metadata"]["source_background_work_job_id"] == (
        "job-001"
    )
    assert episode["origin_metadata"]["active_turn_platform_message_ids"] == [
        "tool-result:task-001"
    ]
    assert episode["percepts"][0]["content"]["semantic_summary"] == (
        "A public source resolved the requested fact."
    )
    assert "tool_result" in serialized
    retired_result_source = "accepted_" + "task_result_ready"
    assert retired_result_source not in serialized
    for forbidden in ("worker_metadata", "worker", "job_ref", "queue_state"):
        assert forbidden not in serialized


def test_tool_result_payload_uses_semantic_metadata() -> None:
    """Result source should expose a typed tool-result cognition outcome."""

    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )
    episode = result_source.build_result_ready_episode_from_job(
        accepted_task_completed_job()
    )
    metadata = episode["percepts"][0]["content"]
    source = metadata["cognition_source"]
    assert source["source_kind"] == "tool_result"
    assert source["source_id"] == "task-001"
    assert source["occurred_at"] == "2026-06-06T00:01:00+00:00"
    assert source["semantic_summary"] == (
        "A public source resolved the requested fact."
    )
    assert source["semantic_objective"] == (
        "Resolve one bounded public question."
    )
    assert source["task_status"] == "resolved"
    assert source["evidence_state"] == "complete"
    assert source["evidence_excerpts"] == [
        "A public source resolved the requested fact."
    ]
    assert source["evidence_handles"] == ["https://example.com/source"]
    assert source["remaining_needs"] == []
    assert source["goal_continuation_ref"] == (
        accepted_task_completed_job()["task_resolution_result"][
            "goal_continuation_ref"
        ]
    )
    serialized_payload = json.dumps(episode, ensure_ascii=False).lower()
    assert "tool_result" in serialized_payload
    retired_result_source = "accepted_" + "task_result_ready"
    assert retired_result_source not in serialized_payload
    for forbidden in ("worker_metadata", "worker", "job_ref", "queue_state"):
        assert forbidden not in serialized_payload


def test_result_source_preserves_typed_task_status_and_ref() -> None:
    """A stored task result reaches the tool-result episode without flattening."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )
    request = resume_queue_request()
    job = jobs._build_job_document(
        request,
        job_id="job-typed-result-001",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )
    _checkpoint, task_result = recorded_task_checkpoint(status="resolved")
    job.update({
        "status": "completed",
        "completed_at": "2026-06-06T00:01:00+00:00",
        "result_summary": task_result["prompt_safe_summary"],
        "task_resolution_result": task_result,
    })

    episode = result_source.build_result_ready_episode_from_job(job)
    source = episode["percepts"][0]["content"]["cognition_source"]

    assert source["semantic_objective"] == task_result["semantic_objective"]
    assert source["task_status"] == "resolved"
    assert source["evidence_state"] == "complete"
    assert source["goal_continuation_ref"] == task_result[
        "goal_continuation_ref"
    ]
    assert episode["origin_metadata"]["goal_continuation_ref"] == (
        task_result["goal_continuation_ref"]
    )


def test_tool_result_source_builder_ignores_untyped_job_summary() -> None:
    """Untrusted job summaries cannot replace the stored typed result."""

    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )
    job = accepted_task_completed_job()
    enriched_summary = (
        "The task needs additional user-provided information.\n"
        "Specific blocker: Please narrow the question before more source reading.\n"
        "Remaining limitation: Source-reading report limit would be exceeded."
    )
    job["result_summary"] = enriched_summary

    episode = result_source.build_result_ready_episode_from_job(job)

    assert episode["trigger_source"] == "tool_result"
    assert episode["percepts"][0]["source_kind"] == "tool_result"
    assert episode["percepts"][0]["content"]["semantic_summary"] == (
        "A public source resolved the requested fact."
    )


def test_successful_delivery_summary_uses_validated_semantic_result() -> None:
    """Resolved accepted-task delivery retains the result-owned semantic summary."""

    worker = importlib.import_module("kazusa_ai_chatbot.background_work.worker")
    job = accepted_task_completed_job()
    task_result = job["task_resolution_result"]
    assert isinstance(task_result, dict)
    marker = "PLAN3_E2E_BETA_SELECTED"
    task_result["prompt_safe_summary"] = f"The selected marker is {marker}."
    task_result["evidence"][0]["summary"] = "receipt-only-reference"
    task_result["evidence_handles"] = ["receipt-only-reference"]

    summary = worker._task_result_delivery_summary(task_result)

    assert summary.startswith(f"The selected marker is {marker}.")
    assert "receipt-only-reference" not in summary
    assert "https://example.com/source" in summary


def test_partial_delivery_summary_retains_semantic_result_and_limitations() -> None:
    """Partial delivery retains result meaning and declared remaining needs."""

    worker = importlib.import_module("kazusa_ai_chatbot.background_work.worker")
    job = accepted_task_completed_job()
    task_result = job["task_resolution_result"]
    assert isinstance(task_result, dict)
    task_result["status"] = "partial"
    task_result["evidence_state"] = "partial"
    task_result["prompt_safe_summary"] = "The selected result is partial."
    task_result["evidence"][0]["summary"] = "receipt-only-reference"
    task_result["evidence_handles"] = ["receipt-only-reference"]
    task_result["remaining_needs"] = ["One bounded source remains unavailable."]

    summary = worker._task_result_delivery_summary(task_result)

    assert summary.startswith("The selected result is partial.")
    assert "receipt-only-reference" not in summary
    assert "Remaining limitations: One bounded source remains unavailable." in summary


def test_non_success_delivery_summary_retains_existing_blocker_contract() -> None:
    """Non-success delivery continues to use its summary plus remaining needs."""

    worker = importlib.import_module("kazusa_ai_chatbot.background_work.worker")
    _checkpoint, task_result = recorded_task_checkpoint(status="needs_user_input")

    summary = worker._task_result_delivery_summary(task_result)

    assert summary == (
        "Continuation is pending.\n"
        "Remaining limitation: Continue the DSH task."
    )


@pytest.mark.asyncio
async def test_service_result_ready_delivery_uses_dispatcher_boundary(
    monkeypatch,
) -> None:
    """Result delivery should run cognition then delegate sending to service."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_work.result_source"
    )
    episode = result_source.build_result_ready_episode_from_job(
        accepted_task_completed_job()
    )
    handle_send_message = AsyncMock(return_value={
        "conversation_message_id": "conversation-001",
        "delivery_tracking_id": "delivery-001",
        "adapter_message_id": "adapter-001",
    })
    persona_supervisor2 = AsyncMock(return_value={
        "final_dialog": ["@Test User Here is the requested result."],
        "surface_outputs": [build_text_surface_output(
            fragments=["@Test User Here is the requested result."],
            created_at=episode["created_at"],
        )],
        "consolidation_state": {
            "final_dialog": ["@Test User Here is the requested result."],
        },
    })
    post_turn = AsyncMock()
    ensure_trace = AsyncMock(return_value={"accepted": True})
    finalize_trace = AsyncMock()

    monkeypatch.setattr(service_module, "_adapter_registry", object())
    monkeypatch.setattr(
        service_module,
        "_active_character_name_snapshot",
        "Current Character",
    )
    monkeypatch.setattr(
        service_module,
        "_refresh_runtime_character_state",
        AsyncMock(),
    )
    character_profile = canonical_service_character_profile(
        marker="background-result",
        global_user_id="character-global-1",
    )
    identity_snapshot = canonical_episode_identity_snapshot(
        marker="background-result",
        global_user_id="character-global-1",
    )
    character_profile["name"] = "Current Character"
    identity_snapshot["character_profile"]["name"] = "Current Character"
    monkeypatch.setattr(
        service_module,
        "_load_latest_character_profile_snapshot",
        AsyncMock(return_value=character_profile),
    )
    monkeypatch.setattr(
        service_module,
        "load_latest_identity_for_episode",
        AsyncMock(return_value=identity_snapshot),
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
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "get_user_message_by_platform_message_id",
        AsyncMock(return_value={
            "_id": "source-row-1",
            "platform": "debug",
            "platform_channel_id": "debug-private-1",
            "role": "user",
            "platform_message_id": "message-1",
            "received_at": "2026-06-06T00:00:00+00:00",
        }),
    )
    monkeypatch.setattr(
        service_module,
        "has_inbound_after",
        AsyncMock(return_value=False),
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
        service_module.llm_tracing,
        "ensure_llm_trace_run",
        ensure_trace,
    )
    monkeypatch.setattr(
        service_module.llm_tracing,
        "finalize_llm_trace_run",
        finalize_trace,
    )
    monkeypatch.setattr(
        service_module,
        "_run_accepted_task_result_post_turn",
        post_turn,
    )
    monkeypatch.setattr(
        service_module,
        "upsert_post_turn_lifecycle_record",
        AsyncMock(),
    )

    result = await service_module._deliver_accepted_task_result_episode(
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
    assert persona_state["llm_trace_id"] == (
        ensure_trace.await_args.kwargs["trace_id"]
    )
    assert persona_state["reason_to_respond"] == "tool_result"
    assert persona_state["platform_message_id"] == "tool-result:task-001"
    assert persona_state["active_turn_platform_message_ids"] == [
        "tool-result:task-001"
    ]
    assert persona_state["user_input"].startswith(
        "Tool result is completed."
    )
    send_args = handle_send_message.await_args.args[0]
    assert send_args["text"] == "@Test User Here is the requested result."
    assert send_args["target_channel"] == "debug-private-1"
    assert send_args["reply_to_msg_id"] == "message-1"
    assert send_args["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "display_name": "Test User",
            "platform_user_id": "platform-user-1",
        }
    ]
    dispatch_context = handle_send_message.await_args.args[1]
    assert dispatch_context.bot_permission_role == "accepted_task_result"
    assert dispatch_context.source_message_id == "tool-result:task-001"
    assert dispatch_context.source_platform_bot_id == "bot-1"
    assert dispatch_context.source_character_name == "Current Character"
    assert ensure_trace.await_args.kwargs["parent_llm_trace_id"] == (
        "llmtrace-parent-1"
    )
    assert ensure_trace.await_args.kwargs["source_background_work_job_id"] == (
        "job-001"
    )
    finalize_trace.assert_awaited_once()
    assert finalize_trace.await_args.kwargs["status"] == "succeeded"
    assert finalize_trace.await_args.kwargs["delivery_tracking_id"] == (
        "delivery-001"
    )
    ensure_identity = service_module._ensure_character_global_identity
    assert ensure_identity.await_args.kwargs["character_name"] == "Current Character"
    post_turn.assert_awaited_once()
    post_turn_state = post_turn.await_args.args[0]
    assert post_turn_state["conversation_progress_turn_outcome"] == (
        "visible_response"
    )


@pytest.mark.asyncio
async def test_background_reply_target_uses_original_source_on_durable_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result older than 120 seconds replies to the original source row."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    monkeypatch.setattr(
        service_module,
        "get_user_message_by_platform_message_id",
        AsyncMock(return_value={
            "_id": "source-row-1",
            "platform": "debug",
            "platform_channel_id": "debug-private-1",
            "role": "user",
            "platform_message_id": "message-1",
            "received_at": "2026-06-06T00:00:00+00:00",
        }),
    )
    has_inbound_after = AsyncMock(return_value=False)
    monkeypatch.setattr(
        service_module,
        "has_inbound_after",
        has_inbound_after,
    )
    monkeypatch.setattr(
        service_module,
        "storage_utc_now_iso",
        lambda: "2026-06-06T00:02:05+00:00",
    )

    reply_target = await service_module._resolve_background_reply_target(
        platform="debug",
        platform_channel_id="debug-private-1",
        source_message_id="message-1",
    )

    assert reply_target == "message-1"
    has_inbound_after.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_reply_target_intervening_qualifies_under_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An intervening user row qualifies a fresh original-source reply."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    monkeypatch.setattr(
        service_module,
        "get_user_message_by_platform_message_id",
        AsyncMock(return_value={
            "_id": "source-row-1",
            "platform": "debug",
            "platform_channel_id": "debug-private-1",
            "role": "user",
            "platform_message_id": "message-1",
            "received_at": "2026-06-06T00:01:00+00:00",
        }),
    )
    monkeypatch.setattr(
        service_module,
        "has_inbound_after",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        service_module,
        "storage_utc_now_iso",
        lambda: "2026-06-06T00:02:00+00:00",
    )

    reply_target = await service_module._resolve_background_reply_target(
        platform="debug",
        platform_channel_id="debug-private-1",
        source_message_id="message-1",
    )

    assert reply_target == "message-1"


@pytest.mark.asyncio
async def test_background_reply_target_strict_age_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly 120 seconds without interleaving keeps a normal send."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    monkeypatch.setattr(
        service_module,
        "get_user_message_by_platform_message_id",
        AsyncMock(return_value={
            "_id": "source-row-1",
            "platform": "debug",
            "platform_channel_id": "debug-private-1",
            "role": "user",
            "platform_message_id": "message-1",
            "received_at": "2026-06-06T00:00:00+00:00",
        }),
    )
    monkeypatch.setattr(
        service_module,
        "has_inbound_after",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        service_module,
        "storage_utc_now_iso",
        lambda: "2026-06-06T00:02:00+00:00",
    )

    reply_target = await service_module._resolve_background_reply_target(
        platform="debug",
        platform_channel_id="debug-private-1",
        source_message_id="message-1",
    )

    assert reply_target is None


@pytest.mark.asyncio
async def test_background_reply_target_fails_closed_on_missing_source_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing original source row produces a normal send."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    monkeypatch.setattr(
        service_module,
        "get_user_message_by_platform_message_id",
        AsyncMock(return_value=None),
    )
    has_inbound_after = AsyncMock(return_value=True)
    monkeypatch.setattr(
        service_module,
        "has_inbound_after",
        has_inbound_after,
    )

    reply_target = await service_module._resolve_background_reply_target(
        platform="debug",
        platform_channel_id="debug-private-1",
        source_message_id="message-1",
    )

    assert reply_target is None
    has_inbound_after.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_reply_target_fails_closed_without_source_received_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy source row without received_at keeps a normal send."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    monkeypatch.setattr(
        service_module,
        "get_user_message_by_platform_message_id",
        AsyncMock(return_value={
            "_id": "source-row-1",
            "platform": "debug",
            "platform_channel_id": "debug-private-1",
            "role": "user",
            "platform_message_id": "message-1",
        }),
    )
    has_inbound_after = AsyncMock(return_value=True)
    monkeypatch.setattr(
        service_module,
        "has_inbound_after",
        has_inbound_after,
    )

    reply_target = await service_module._resolve_background_reply_target(
        platform="debug",
        platform_channel_id="debug-private-1",
        source_message_id="message-1",
    )

    assert reply_target is None
    has_inbound_after.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_reply_target_rejects_synthetic_or_blank_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic tool-result identities are never reply targets."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    source_lookup = AsyncMock()
    monkeypatch.setattr(
        service_module,
        "get_user_message_by_platform_message_id",
        source_lookup,
    )

    synthetic_target = await service_module._resolve_background_reply_target(
        platform="debug",
        platform_channel_id="debug-private-1",
        source_message_id="tool-result:task-001",
    )
    blank_target = await service_module._resolve_background_reply_target(
        platform="debug",
        platform_channel_id="debug-private-1",
        source_message_id="",
    )

    assert synthetic_target is None
    assert blank_target is None
    source_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_accepted_task_result_post_turn_skips_consolidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Result-ready accepted tasks must not enter consolidation origin routing."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    lifecycle = AsyncMock(return_value={
        "final_dialog": ["@Test User Here is the requested result."],
    })
    progress = AsyncMock()
    consolidation = AsyncMock()
    residue = AsyncMock()
    monkeypatch.setattr(
        service_module,
        "_run_post_turn_memory_lifecycle_background",
        lifecycle,
    )
    monkeypatch.setattr(
        service_module,
        "_run_conversation_progress_record_background",
        progress,
    )
    monkeypatch.setattr(
        service_module,
        "_run_consolidation_background",
        consolidation,
    )
    monkeypatch.setattr(
        service_module,
        "_run_internal_monologue_residue_record_background",
        residue,
    )

    await service_module._run_accepted_task_result_post_turn(
        {
            "final_dialog": ["@Test User Here is the requested result."],
            "episode_trace": {},
            "conversation_progress_turn_outcome": "visible_response",
        },
        visible_response_sent=True,
    )

    lifecycle.assert_awaited_once()
    progress.assert_awaited_once()
    consolidation.assert_not_awaited()
    residue.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivery_tick_syncs_accepted_task_delivery_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted-task delivery should move through in-progress to delivered."""

    delivery_module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.delivery"
    )
    job = accepted_task_completed_job()
    job["delivery_state"] = "ready"
    find_jobs = AsyncMock(return_value=[job])
    mark_job_in_progress = AsyncMock(return_value={
        **job,
        "delivery_state": "in_progress",
        "delivery_tracking_id": "delivery-001",
    })
    mark_job_delivered = AsyncMock(return_value={**job, "status": "delivered"})
    mark_task_in_progress = AsyncMock(return_value={
        "state": "delivery_in_progress",
    })
    mark_task_delivered = AsyncMock(return_value={"state": "delivered"})
    mark_task_failed = AsyncMock(return_value={"state": "delivery_retryable"})
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
    assert delivered_episode["trigger_source"] == "tool_result"


@pytest.mark.asyncio
async def test_delivery_tick_retries_job_when_accepted_task_claim_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job cannot enter cognition delivery without its accepted-task claim."""

    delivery_module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.delivery"
    )
    job = accepted_task_completed_job()
    job["delivery_state"] = "ready"
    marked_job = {**job, "status": "delivery_in_progress"}
    deliver_episode = AsyncMock()
    mark_job_failed = AsyncMock(return_value={
        **job,
        "status": "delivery_failed",
    })
    _patch_delivery_recovery(monkeypatch, delivery_module)
    monkeypatch.setattr(
        delivery_module,
        "find_deliverable_background_work_jobs",
        AsyncMock(return_value=[job]),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivery_in_progress",
        AsyncMock(return_value=marked_job),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_accepted_task_delivery_in_progress",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivery_failed",
        mark_job_failed,
    )

    result = await delivery_module.run_background_work_delivery_tick(
        deliver_result_episode_func=deliver_episode,
        limit=1,
    )

    assert result == {
        "processed_count": 1,
        "delivered_count": 0,
        "failed_count": 1,
        "recovered_count": 0,
    }
    deliver_episode.assert_not_awaited()
    mark_job_failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_delivery_tick_retries_job_when_accepted_finalization_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing accepted-task terminal write cannot report job delivery."""

    delivery_module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.delivery"
    )
    job = accepted_task_completed_job()
    job["delivery_state"] = "ready"
    marked_job = {**job, "status": "delivery_in_progress"}
    mark_job_delivered = AsyncMock()
    mark_job_failed = AsyncMock(return_value={
        **job,
        "status": "delivery_failed",
    })
    _patch_delivery_recovery(monkeypatch, delivery_module)
    monkeypatch.setattr(
        delivery_module,
        "find_deliverable_background_work_jobs",
        AsyncMock(return_value=[job]),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivery_in_progress",
        AsyncMock(return_value=marked_job),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_accepted_task_delivery_in_progress",
        AsyncMock(return_value={"state": "delivery_in_progress"}),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_accepted_task_delivered",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivered",
        mark_job_delivered,
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivery_failed",
        mark_job_failed,
    )

    result = await delivery_module.run_background_work_delivery_tick(
        deliver_result_episode_func=AsyncMock(return_value={
            "status": "delivered",
            "conversation_message_id": "conversation-001",
        }),
        limit=1,
    )

    assert result["delivered_count"] == 0
    assert result["failed_count"] == 1
    mark_job_delivered.assert_not_awaited()
    mark_job_failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_delivery_tick_retries_job_when_job_finalization_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing job terminal write remains observable as retryable failure."""

    delivery_module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.delivery"
    )
    job = accepted_task_completed_job()
    job["delivery_state"] = "ready"
    marked_job = {**job, "status": "delivery_in_progress"}
    mark_job_failed = AsyncMock(return_value={
        **job,
        "status": "delivery_failed",
    })
    _patch_delivery_recovery(monkeypatch, delivery_module)
    monkeypatch.setattr(
        delivery_module,
        "find_deliverable_background_work_jobs",
        AsyncMock(return_value=[job]),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivery_in_progress",
        AsyncMock(return_value=marked_job),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_accepted_task_delivery_in_progress",
        AsyncMock(return_value={"state": "delivery_in_progress"}),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_accepted_task_delivered",
        AsyncMock(return_value={"state": "delivered"}),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivered",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        delivery_module,
        "mark_background_work_delivery_failed",
        mark_job_failed,
    )

    result = await delivery_module.run_background_work_delivery_tick(
        deliver_result_episode_func=AsyncMock(return_value={
            "status": "delivered",
            "conversation_message_id": "conversation-001",
        }),
        limit=1,
    )

    assert result["delivered_count"] == 0
    assert result["failed_count"] == 1
    mark_job_failed.assert_awaited_once()


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
    build = jobs._build_job_document
    request = resume_queue_request()
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
