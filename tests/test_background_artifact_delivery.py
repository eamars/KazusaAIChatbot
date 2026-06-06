"""Tests for background artifact result-ready delivery boundaries."""

from __future__ import annotations

import importlib

from unittest.mock import AsyncMock

import pytest


def _completed_job() -> dict:
    return {
        "job_id": "job-001",
        "work_kind": "coding_snippet",
        "objective": "Generate a Fibonacci function snippet.",
        "artifact_text": "def fib(n): return n",
        "failure_summary": "",
        "source_platform": "debug",
        "source_channel_id": "debug-private-1",
        "source_channel_type": "private",
        "source_message_id": "message-1",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-1",
        "requester_platform_user_id": "platform-user-1",
        "requester_display_name": "Test User",
        "completed_at": "2026-05-16T00:00:00+00:00",
    }


def test_result_source_builder_creates_prompt_safe_episode() -> None:
    """Completed jobs should become source-bound cognition episodes."""

    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_artifact.result_source"
    )
    builder = getattr(result_source, "build_result_ready_episode_from_job")

    episode = builder(_completed_job())

    assert episode["trigger_source"] == "background_artifact_result_ready"
    assert episode["input_sources"] == ["background_artifact_result"]
    assert episode["output_mode"] == "visible_reply"


@pytest.mark.asyncio
async def test_service_result_ready_delivery_uses_dispatcher_boundary(
    monkeypatch,
) -> None:
    """Result delivery should run cognition then delegate sending to dispatcher."""

    service_module = importlib.import_module("kazusa_ai_chatbot.service")
    result_source = importlib.import_module(
        "kazusa_ai_chatbot.background_artifact.result_source"
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
        "_run_background_artifact_result_post_turn",
        post_turn,
    )

    result = await service_module._deliver_background_artifact_result_episode(
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
    assert persona_state["reason_to_respond"] == "background_artifact_result_ready"
    send_args = handle_send_message.await_args.args[0]
    assert send_args["text"] == "Here is the requested result."
    assert send_args["target_channel"] == "debug-private-1"
    assert send_args["delivery_mentions"][0]["global_user_id"] == "global-user-1"
    post_turn.assert_awaited_once()
