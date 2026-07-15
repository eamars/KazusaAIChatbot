"""Focused tests for first-class image cognition input."""

from __future__ import annotations

import pytest
pytest.skip("Stage 1 assertions replaced by the V2 contract suite", allow_module_level=True)

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot import chat_input_queue as queue_module
from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.consolidation.origin import (
    build_user_message_consolidation_origin,
)
from kazusa_ai_chatbot.nodes.persona_relevance_agent import relevance_agent
from kazusa_ai_chatbot.time_boundary import build_turn_clock


_TURN_CLOCK = build_turn_clock("2026-05-10 09:30:00")


class _CapturingServiceGraph:
    """Capture graph input while returning a no-response result."""

    def __init__(self) -> None:
        """Initialize an empty captured state."""
        self.state: dict[str, object] = {}

    async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
        """Record the graph state passed by the service.

        Args:
            state: Initial graph state for the current turn.

        Returns:
            Minimal graph result that skips assistant persistence.
        """
        self.state = state
        result: dict[str, object] = {
            "should_respond": False,
            "use_reply_feature": False,
            "final_dialog": [],
            "future_promises": [],
            "consolidation_state": {},
        }
        return result


def _time_context() -> dict[str, str]:
    """Build the minimal character-local time context for episode fixtures."""
    time_context = _TURN_CLOCK["local_time_context"]
    return time_context


def _episode_kwargs() -> dict[str, object]:
    """Build stable text-chat episode kwargs for image tests."""
    kwargs: dict[str, object] = {
        "episode_id": "episode-image-1",
        "percept_id": "percept-dialog",
        "storage_timestamp_utc": _TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": _TURN_CLOCK["local_time_context"],
        "user_input": "Does this support my plan?",
        "platform": "debug",
        "platform_channel_id": "debug-private-1",
        "channel_type": "private",
        "platform_message_id": "platform-message-9",
        "platform_user_id": "platform-user-9",
        "global_user_id": "global-user-9",
        "user_name": "Image User",
        "active_turn_platform_message_ids": ["platform-message-9"],
        "active_turn_conversation_row_ids": ["conversation-row-9"],
        "debug_modes": {"think_only": False},
        "target_addressed_user_ids": ["character-1"],
        "target_broadcast": False,
    }
    return kwargs


def _image_observation() -> dict[str, object]:
    """Build a structured image observation fixture."""
    observation: dict[str, object] = {
        "observation_origin": "current_attachment",
        "source_message_id": "platform-message-9",
        "media_kind": "image",
        "summary_status": "available",
        "summary": "A desk setup with handwritten study notes.",
        "visible_text": ["Chapter 4 review"],
        "salient_visual_facts": ["open notebook", "laptop on the desk"],
        "spatial_or_scene_facts": ["notebook is left of the laptop"],
        "uncertainty": ["small text near the screen is unclear"],
    }
    return observation


def _minimal_relevance_state() -> dict[str, object]:
    """Build relevance-agent state with one described image attachment."""
    state: dict[str, object] = {
        "storage_timestamp_utc": _TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": _TURN_CLOCK["local_time_context"],
        "platform": "debug",
        "platform_message_id": "platform-message-9",
        "platform_user_id": "platform-user-9",
        "global_user_id": "global-user-9",
        "user_name": "Image User",
        "user_input": "Does this support my plan?",
        "user_multimedia_input": [{
            "content_type": "image/png",
            "base64_data": "",
            "description": "image shows a desk setup",
            "image_observation": _image_observation(),
        }],
        "user_profile": {"relationship_state": 500, "semantic_relationship_projection": ""},
        "platform_bot_id": "bot-1",
        "message_envelope": {
            "body_text": "Does this support my plan?",
            "raw_wire_text": "Does this support my plan?",
            "addressed_to_global_user_ids": ["character-1"],
            "mentions": [],
            "attachments": [],
            "broadcast": False,
        },
        "prompt_message_context": {
            "body_text": "Does this support my plan?",
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "character_name": "Character",
        "character_profile": {
            "name": "Character",
            "global_user_id": "character-1",
            "mood": "neutral",
            "vibe_check": "calm",
        },
        "platform_channel_id": "debug-private-1",
        "channel_type": "private",
        "channel_name": "Direct",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "debug_modes": {},
    }
    state["cognitive_episode"] = build_text_chat_cognitive_episode(
        **_episode_kwargs(),
        media_description_rows=[{
            "content_type": "image/png",
            "description": "image shows a desk setup",
            "image_observation": _image_observation(),
        }],
    )
    return state


def _service_request() -> service_module.ChatRequest:
    """Build a reply request whose target row has only image description."""
    request = service_module.ChatRequest(
        platform="debug",
        platform_channel_id="debug-private-1",
        channel_type="private",
        platform_message_id="platform-message-10",
        platform_user_id="platform-user-9",
        platform_bot_id="bot-1",
        display_name="Image User",
        channel_name="Direct",
        content_type="text",
        message_envelope={
            "body_text": "This is the one I meant.",
            "raw_wire_text": "This is the one I meant.",
            "mentions": [],
            "reply": {
                "platform_message_id": "platform-message-9",
                "derivation": "platform_native",
            },
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
    )
    return request


def _queued_item(request: service_module.ChatRequest) -> queue_module.QueuedChatItem:
    """Wrap a service request in the queue item shape."""
    future: asyncio.Future[service_module.ChatResponse] = (
        asyncio.get_running_loop().create_future()
    )
    item = queue_module.QueuedChatItem(
        sequence=10,
        request=request,
        storage_timestamp_utc=_TURN_CLOCK["storage_timestamp_utc"],
        local_timestamp=_TURN_CLOCK["local_timestamp"],
        local_time_context=_TURN_CLOCK["local_time_context"],
        future=future,
    )
    return item


def _patch_service_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    graph: _CapturingServiceGraph,
) -> None:
    """Patch service dependencies around initial-state construction."""
    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        {"name": "Active Character", "personality_brief": "brief"},
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {"mood": "calm", "vibe_check": "steady"},
    )
    monkeypatch.setattr(
        service_module,
        "get_character_runtime_state",
        AsyncMock(return_value={"mood": "calm", "vibe_check": "steady"}),
    )
    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-1"),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_global_user_id",
        AsyncMock(return_value="global-user-9"),
    )
    monkeypatch.setattr(
        service_module,
        "get_user_profile",
        AsyncMock(return_value={"relationship_state": 500}),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        AsyncMock(return_value="conversation-row-10"),
    )
    monkeypatch.setattr(
        service_module,
        "build_promoted_reflection_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(service_module, "_graph", graph)


def _llm_response(content: str) -> MagicMock:
    """Build a minimal LangChain-like response object."""
    response = MagicMock()
    response.content = content
    return response


@pytest.mark.asyncio
async def test_relevance_keeps_image_descriptor_out_of_user_input() -> None:
    """Relevance should not make descriptor text look user-authored."""
    response = _llm_response(
        '{"should_respond": true, "reason_to_respond": "direct", '
        '"use_reply_feature": false, "channel_topic": "", '
        '"indirect_speech_context": ""}'
    )

    with patch("kazusa_ai_chatbot.nodes.persona_relevance_agent._relevance_agent_llm") as llm:
        llm.ainvoke = AsyncMock(return_value=response)
        result = await relevance_agent(_minimal_relevance_state())

    _, human_message = llm.ainvoke.await_args.args[0]
    human_payload = json.loads(human_message.content)

    assert result["user_input"] == "Does this support my plan?"
    assert human_payload["user_message"]["content"] == (
        "Does this support my plan?"
    )
    assert "Image attachment:" not in human_message.content
    assert "image shows a desk setup" not in human_message.content


def test_source_payload_projects_structured_image_observations() -> None:
    """Cognition prompt payload should expose typed visual facts."""
    episode = build_text_chat_cognitive_episode(
        **_episode_kwargs(),
        media_description_rows=[{
            "content_type": "image/png",
            "description": "A desk setup with handwritten study notes.",
            "image_observation": _image_observation(),
        }],
    )
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l1_subconscious",
    )

    source_payload = build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    )

    assert source_payload == {
        "media_observations": {
            "image_observations": [
                {
                    key: value
                    for key, value in _image_observation().items()
                    if key != "source_message_id"
                }
            ],
            "audio_observations": [],
        },
    }


@pytest.mark.asyncio
async def test_quoted_image_description_enters_prompt_and_cognition_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replying to a stored image should reuse its durable description."""
    graph = _CapturingServiceGraph()
    _patch_service_dependencies(monkeypatch, graph)
    previous_row = {
        "platform_user_id": "platform-user-8",
        "display_name": "Previous User",
        "body_text": "",
        "attachments": [{
            "media_type": "image/png",
            "description": "stored image shows a dessert counter",
            "storage_shape": "inline",
        }],
    }
    monkeypatch.setattr(
        service_module,
        "get_conversation_by_platform_message_id",
        AsyncMock(return_value=previous_row),
    )

    await service_module._process_queued_chat_item(_queued_item(_service_request()))

    prompt_reply = graph.state["prompt_message_context"]["reply"]
    episode = graph.state["cognitive_episode"]

    assert prompt_reply["attachments"] == [{
        "media_kind": "image",
        "description": "stored image shows a dessert counter",
        "summary_status": "available",
    }]
    assert episode["input_sources"] == ["dialog_text", "image_observation"]
    assert episode["percepts"][1]["metadata"]["observation_origin"] == (
        "quoted_reply_attachment"
    )
    assert episode["percepts"][1]["content"] == (
        "stored image shows a dessert counter"
    )


def test_multimodal_consolidation_origin_is_metadata_only() -> None:
    """Image-observation turns should consolidate without percept content."""
    episode = build_text_chat_cognitive_episode(
        **_episode_kwargs(),
        media_description_rows=[{
            "content_type": "image/png",
            "description": "private image content should not persist here",
            "image_observation": _image_observation(),
        }],
    )

    metadata = build_user_message_consolidation_origin(episode=episode)

    assert metadata["input_sources"] == ["dialog_text", "image_observation"]
    assert "percepts" not in metadata
    assert "private image content should not persist here" not in str(metadata)
