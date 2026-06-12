"""Tests for background-work result-ready delivery boundaries."""

from __future__ import annotations

import importlib
import json

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
        "final_dialog": ["Here is the requested result."],
        "mention_target_user": True,
        "consolidation_state": {
            "final_dialog": ["Here is the requested result."],
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
    assert send_args["text"] == "Here is the requested result."
    assert send_args["target_channel"] == "debug-private-1"
    assert send_args["delivery_mentions"][0]["global_user_id"] == "global-user-1"
    dispatch_context = handle_send_message.await_args.args[1]
    assert dispatch_context.bot_permission_role == "background_work_result"
    assert dispatch_context.source_platform_bot_id == "bot-1"
    post_turn.assert_awaited_once()


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
