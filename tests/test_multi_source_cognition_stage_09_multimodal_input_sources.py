"""Tests for multimodal user-message cognitive episode admission."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import chat_input_queue as queue_module
from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l1 as l1_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as l2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2c2 as l2c2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_msg_decontexualizer as decontextualizer_module
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS,
    MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS,
    build_text_chat_cognitive_episode,
    build_text_chat_media_description_rows,
    replace_text_chat_media_percepts,
)
from kazusa_ai_chatbot.rag.cognitive_episode_adapter import (
    RAGEpisodeAdapterError,
    build_text_chat_rag_request,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_prompt_selection import (
    CognitionPromptSelectionError,
    CognitionPromptStage,
    build_cognition_prompt_source_payload,
    select_cognition_prompt_variant,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock


_APPROVED_STAGES: tuple[CognitionPromptStage, ...] = (
    "l1_subconscious",
    "l2a_conscious_framing",
    "l2b_boundary_appraisal",
    "l2c1_judgment_synthesis",
    "l2c2_social_context_appraisal",
    "l3_style_agent",
    "l3_content_anchor_agent",
    "l3_preference_adapter",
    "l3_visual_agent",
)
_MULTIMODAL_PROMPT_VARIANTS = (
    "text_chat_user_message_image_observation",
    "text_chat_user_message_audio_observation",
    "text_chat_user_message_image_audio_observation",
)
_TURN_CLOCK = build_turn_clock("2026-05-10 09:30:00")
_PROMPT_FINGERPRINTS = (
    (
        "_COGNITION_SUBCONSCIOUS_PROMPT",
        l1_module._COGNITION_SUBCONSCIOUS_PROMPT,
        4535,
        "436df67750af7f6ace060e07b03c019f63617df6b617bd0f6f05bbe6de60221d",
    ),
    (
        "_COGNITION_CONSCIOUSNESS_PROMPT",
        l2_module._COGNITION_CONSCIOUSNESS_PROMPT,
        13407,
        "0d23375541528afaf9e1db57b2201c8055a3942cc8cc9c789cb4532546cefade",
    ),
    (
        "_BOUNDARY_CORE_PROMPT",
        l2_module._BOUNDARY_CORE_PROMPT,
        10304,
        "2e40706a0efa6f53330f7093021e5b7be5951b2ac39d2c6647e51b4aa6e0525c",
    ),
    (
        "_JUDGEMENT_CORE_PROMPT",
        l2_module._JUDGEMENT_CORE_PROMPT,
        6932,
        "d478625f8b47a9b9c2ef33f0ae52f956e7ebe227b7e74dbaea0d81b8b7ad2ae6",
    ),
    (
        "_CONTEXTUAL_AGENT_PROMPT",
        l2c2_module._CONTEXTUAL_AGENT_PROMPT,
        5402,
        "3d99f7054f25facf32f17d187aeec52d6cb415868480eda836f674e3293e7cb2",
    ),
    (
        "_STYLE_AGENT_PROMPT",
        l3_module._STYLE_AGENT_PROMPT,
        6894,
        "664f0c8fe115dc0a0683595c815c072c7423dc2c6f047b9561938c7831238c0a",
    ),
    (
        "_CONTENT_ANCHOR_AGENT_PROMPT",
        l3_module._CONTENT_ANCHOR_AGENT_PROMPT,
        17122,
        "8bed4dc81bab7831d4a900c779eed347a2c1b44ef92968db6321ce10f968ad9c",
    ),
    (
        "_PREFERENCE_ADAPTER_PROMPT",
        l3_module._PREFERENCE_ADAPTER_PROMPT,
        7521,
        "2f09e1ee799141a78cd24b783b6f9a493663d3aa677d7ed449a719d8ae392def",
    ),
    (
        "_VISUAL_AGENT_PROMPT",
        l3_module._VISUAL_AGENT_PROMPT,
        7826,
        "552498b619657ce9aa11099aa7a4abec3236956691b0908994158798af75743a",
    ),
)


class _CapturingServiceGraph:
    """Capture service graph input and return a no-response result."""

    def __init__(self) -> None:
        """Initialize the graph with an empty captured state."""
        self.state: dict[str, object] = {}

    async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
        """Record one graph invocation.

        Args:
            state: Initial service graph state.

        Returns:
            Minimal graph result that skips response persistence work.
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


class _DescriptorResponse:
    """Small response wrapper for patched descriptor LLM calls."""

    def __init__(self, content: str) -> None:
        """Store the response content.

        Args:
            content: JSON content returned from the fake descriptor LLM.
        """
        self.content = content


def _time_context() -> dict[str, str]:
    """Build the fixed time context shared by focused episode tests.

    Returns:
        Minimal character-local time context accepted by the episode validator.
    """
    time_context = _TURN_CLOCK["local_time_context"]
    return time_context


def _builder_kwargs() -> dict[str, object]:
    """Build stable text-chat episode kwargs for media-admission tests.

    Returns:
        Keyword arguments accepted by `build_text_chat_cognitive_episode`.
    """
    kwargs: dict[str, object] = {
        "episode_id": "episode-stage-09",
        "percept_id": "percept-dialog",
        "storage_timestamp_utc": _TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": _TURN_CLOCK["local_time_context"],
        "user_input": "Can you look at this and tell me what matters?",
        "platform": "debug",
        "platform_channel_id": "debug-private-1",
        "channel_type": "private",
        "platform_message_id": "platform-message-9",
        "platform_user_id": "platform-user-9",
        "global_user_id": "global-user-9",
        "user_name": "Stage Nine",
        "active_turn_platform_message_ids": ["platform-message-9"],
        "active_turn_conversation_row_ids": ["conversation-row-9"],
        "debug_modes": {"think_only": False},
        "target_addressed_user_ids": ["character-1"],
        "target_broadcast": False,
    }
    return kwargs


def _service_request(
    *,
    attachments: list[dict[str, object]],
) -> service_module.ChatRequest:
    """Build a service chat request containing the supplied attachments.

    Args:
        attachments: Typed attachment references for the message envelope.

    Returns:
        Chat request accepted by the service queue processor.
    """
    request = service_module.ChatRequest(
        platform="debug",
        platform_channel_id="debug-private-1",
        channel_type="private",
        platform_message_id="platform-message-9",
        platform_user_id="platform-user-9",
        platform_bot_id="bot-1",
        display_name="Stage Nine",
        channel_name="Direct",
        content_type="mixed",
        message_envelope={
            "body_text": "Can you look at this?",
            "raw_wire_text": "Can you look at this?",
            "mentions": [],
            "attachments": attachments,
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
    )
    return request


def _queued_item(request: service_module.ChatRequest) -> queue_module.QueuedChatItem:
    """Build a queued item around a service request.

    Args:
        request: Chat request to process.

    Returns:
        Queue item with a future bound to the active event loop.
    """
    future: asyncio.Future[service_module.ChatResponse] = (
        asyncio.get_running_loop().create_future()
    )
    item = queue_module.QueuedChatItem(
        sequence=9,
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
    """Patch service dependencies outside the episode-construction boundary.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        graph: Fake graph used to capture the initial service state.

    Returns:
        None.
    """
    monkeypatch.setattr(
        service_module,
        "_static_character_profile",
        {"name": "Active Character", "personality_brief": "brief"},
    )
    monkeypatch.setattr(
        service_module,
        "_runtime_character_state",
        {"mood": "calm", "global_vibe": "steady"},
    )
    monkeypatch.setattr(
        service_module,
        "get_character_runtime_state",
        AsyncMock(return_value={"mood": "calm", "global_vibe": "steady"}),
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
        AsyncMock(return_value={"affinity": 500}),
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        AsyncMock(return_value="conversation-row-9"),
    )
    monkeypatch.setattr(
        service_module,
        "build_promoted_reflection_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(service_module, "_graph", graph)


def _descriptor_state() -> dict[str, object]:
    """Build a descriptor-node state with a text-only cognitive episode.

    Returns:
        Minimal state accepted by `multimedia_descriptor_agent`.
    """
    episode = build_text_chat_cognitive_episode(**_builder_kwargs())
    state: dict[str, object] = {
        "storage_timestamp_utc": _TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": _TURN_CLOCK["local_time_context"],
        "platform": "debug",
        "platform_channel_id": "debug-private-1",
        "platform_message_id": "platform-message-9",
        "platform_user_id": "platform-user-9",
        "global_user_id": "global-user-9",
        "user_name": "Stage Nine",
        "message_envelope": {
            "body_text": "Can you look at this?",
            "raw_wire_text": "Can you look at this?",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
        "user_multimedia_input": [],
        "prompt_message_context": {
            "body_text": "Can you look at this?",
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "cognitive_episode": episode,
    }
    return state


def _stage_08_text_only_snapshot() -> dict[str, object]:
    """Return the exact text-only episode shape preserved from Stage 08.

    Returns:
        Expected text-chat `CognitiveEpisode` dictionary.
    """
    snapshot: dict[str, object] = {
        "episode_id": "episode-stage-09",
        "trigger_source": "user_message",
        "input_sources": ["dialog_text"],
        "output_mode": "visible_reply",
        "percepts": [
            {
                "percept_id": "percept-dialog",
                "input_source": "dialog_text",
                "content": "Can you look at this and tell me what matters?",
                "visibility": "model_visible",
                "metadata": {},
            },
        ],
        "target_scope": {
            "platform": "debug",
            "platform_channel_id": "debug-private-1",
            "channel_type": "private",
            "current_platform_user_id": "platform-user-9",
            "current_global_user_id": "global-user-9",
            "current_display_name": "Stage Nine",
            "target_addressed_user_ids": ["character-1"],
            "target_broadcast": False,
        },
        "origin_metadata": {
            "platform": "debug",
            "platform_message_id": "platform-message-9",
            "active_turn_platform_message_ids": ["platform-message-9"],
            "active_turn_conversation_row_ids": ["conversation-row-9"],
            "debug_modes": {"think_only": False},
        },
        "storage_timestamp_utc": _TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": _TURN_CLOCK["local_time_context"],
    }
    return snapshot


def _rag_request_kwargs(episode: dict[str, object]) -> dict[str, object]:
    """Build common RAG adapter kwargs for one episode.

    Args:
        episode: Cognitive episode supplied to the adapter.

    Returns:
        Keyword arguments accepted by `build_text_chat_rag_request`.
    """
    kwargs: dict[str, object] = {
        "episode": episode,
        "decontexualized_input": "Can you look at this and tell me what matters?",
        "character_profile": {
            "global_user_id": "character-1",
            "name": "Active Character",
        },
        "user_profile": {"affinity": 500},
        "prompt_message_context": {
            "body_text": "Can you look at this and tell me what matters?",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
    }
    return kwargs


def test_text_only_builder_output_matches_stage_08_snapshot() -> None:
    baseline_episode = build_text_chat_cognitive_episode(**_builder_kwargs())
    explicit_empty_episode = build_text_chat_cognitive_episode(
        **_builder_kwargs(),
        media_description_rows=[],
    )

    assert baseline_episode == _stage_08_text_only_snapshot()
    assert explicit_empty_episode == _stage_08_text_only_snapshot()


def test_builder_adds_bounded_image_and_audio_percepts_without_raw_media() -> None:
    multimedia_input = [
        {
            "content_type": "image/png",
            "base64_data": "raw-image-bytes",
            "url": "https://example.invalid/image.png",
            "description": " image shows a whiteboard plan ",
        },
        {
            "content_type": "audio/ogg",
            "base64_data": "raw-audio-bytes",
            "description": " user says the deadline is today ",
        },
    ]
    media_rows = build_text_chat_media_description_rows(multimedia_input)

    episode = build_text_chat_cognitive_episode(
        **_builder_kwargs(),
        media_description_rows=media_rows,
    )

    expected_image_observation = {
        "observation_origin": "current_attachment",
        "source_message_id": "",
        "media_kind": "image",
        "summary_status": "available",
        "summary": "image shows a whiteboard plan",
        "visible_text": [],
        "salient_visual_facts": [],
        "spatial_or_scene_facts": [],
        "uncertainty": [],
    }
    assert media_rows == [
        {
            "content_type": "image/png",
            "description": "image shows a whiteboard plan",
            "image_observation": expected_image_observation,
        },
        {
            "content_type": "audio/ogg",
            "description": "user says the deadline is today",
        },
    ]
    assert episode["input_sources"] == [
        "dialog_text",
        "image_observation",
        "audio_observation",
    ]
    assert episode["percepts"][1:] == [
        {
            "percept_id": "percept-dialog:media:1",
            "input_source": "image_observation",
            "content": "image shows a whiteboard plan",
            "visibility": "model_visible",
            "metadata": {
                "observation_origin": "current_attachment",
                "source_message_id": "",
                "media_kind": "image",
                "summary_status": "available",
                "media_index": 1,
                "image_observation": expected_image_observation,
            },
        },
        {
            "percept_id": "percept-dialog:media:2",
            "input_source": "audio_observation",
            "content": "user says the deadline is today",
            "visibility": "model_visible",
            "metadata": {"content_type": "audio/ogg", "media_index": 2},
        },
    ]

    rendered_episode = json.dumps(episode, ensure_ascii=False)
    assert "raw-image-bytes" not in rendered_episode
    assert "raw-audio-bytes" not in rendered_episode
    assert "https://example.invalid/image.png" not in rendered_episode


def test_builder_drops_empty_unsupported_and_over_cap_media_rows() -> None:
    overlong_description = (
        "x" * (MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS + 50)
    )
    multimedia_input = [
        {"content_type": "video/mp4", "description": "video is unsupported"},
        {"content_type": "image/png", "description": "   "},
        {"content_type": 9, "description": "bad content type"},
        {"content_type": "image/png", "description": overlong_description},
        {"content_type": "image/jpeg", "description": "second image"},
        {"content_type": "audio/ogg", "description": "first audio"},
        {"content_type": "audio/mpeg", "description": "second audio"},
        {"content_type": "image/gif", "description": "fifth accepted drops"},
    ]
    media_rows = build_text_chat_media_description_rows(multimedia_input)

    episode = build_text_chat_cognitive_episode(
        **_builder_kwargs(),
        media_description_rows=media_rows,
    )
    media_percepts = episode["percepts"][1:]

    assert len(media_percepts) == MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS
    assert media_percepts[0]["content"] == f'{"x" * 797}...'
    assert len(media_percepts[0]["content"]) == (
        MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS
    )
    assert media_percepts[-1]["content"] == "second audio"
    assert "fifth accepted drops" not in json.dumps(
        episode,
        ensure_ascii=False,
    )


def test_replace_text_chat_media_percepts_refreshes_existing_episode_without_mutation() -> None:
    initial_rows = [{"content_type": "image/png", "description": "old image"}]
    initial_episode = build_text_chat_cognitive_episode(
        **_builder_kwargs(),
        media_description_rows=initial_rows,
    )
    original_episode = deepcopy(initial_episode)

    refreshed_episode = replace_text_chat_media_percepts(
        episode=initial_episode,
        media_description_rows=[
            {"content_type": "audio/ogg", "description": "new audio"},
        ],
    )

    assert initial_episode == original_episode
    assert refreshed_episode is not initial_episode
    assert refreshed_episode["episode_id"] == initial_episode["episode_id"]
    assert refreshed_episode["target_scope"] == initial_episode["target_scope"]
    assert refreshed_episode["origin_metadata"] == (
        initial_episode["origin_metadata"]
    )
    assert refreshed_episode["percepts"][0] == original_episode["percepts"][0]
    assert refreshed_episode["input_sources"] == [
        "dialog_text",
        "audio_observation",
    ]
    assert refreshed_episode["percepts"][1:] == [
        {
            "percept_id": "percept-dialog:media:1",
            "input_source": "audio_observation",
            "content": "new audio",
            "visibility": "model_visible",
            "metadata": {"content_type": "audio/ogg", "media_index": 1},
        },
    ]


def test_rag_accepts_multimodal_profiles_and_projects_dialog_text_only() -> None:
    media_profiles = [
        [{"content_type": "image/png", "description": "image shows a plan"}],
        [{"content_type": "audio/ogg", "description": "audio mentions a date"}],
        [
            {"content_type": "image/png", "description": "image shows a plan"},
            {"content_type": "audio/ogg", "description": "audio mentions a date"},
        ],
    ]

    for media_rows in media_profiles:
        episode = build_text_chat_cognitive_episode(
            **_builder_kwargs(),
            media_description_rows=media_rows,
        )
        request = build_text_chat_rag_request(**_rag_request_kwargs(episode))

        assert request["original_query"] == (
            "Can you look at this and tell me what matters?"
        )
        assert request["current_user_id"] == "global-user-9"
        assert request["character_user_id"] == "character-1"
        rendered_request = json.dumps(request, ensure_ascii=False)
        assert "image_observation" not in rendered_request
        assert "audio_observation" not in rendered_request
        assert "media_observations" not in rendered_request
        assert "image shows a plan" not in rendered_request
        assert "audio mentions a date" not in rendered_request


def test_rag_rejects_pure_media_or_unsupported_source_profiles() -> None:
    unsupported_order = build_text_chat_cognitive_episode(
        **_builder_kwargs(),
        media_description_rows=[
            {"content_type": "image/png", "description": "image shows a plan"},
        ],
    )
    unsupported_order["input_sources"] = [
        "image_observation",
        "dialog_text",
    ]

    with pytest.raises(RAGEpisodeAdapterError):
        build_text_chat_rag_request(**_rag_request_kwargs(unsupported_order))

    pure_media = deepcopy(unsupported_order)
    pure_media["input_sources"] = ["image_observation"]
    pure_media["percepts"] = [pure_media["percepts"][1]]

    with pytest.raises(CognitiveEpisodeValidationError):
        build_text_chat_rag_request(**_rag_request_kwargs(pure_media))


def test_selector_maps_exact_multimodal_profiles_to_prompt_keys() -> None:
    media_profiles = [
        (
            [{"content_type": "image/png", "description": "image shows a plan"}],
            ["dialog_text", "image_observation"],
            "text_chat_user_message_image_observation",
        ),
        (
            [{"content_type": "audio/ogg", "description": "audio mentions a date"}],
            ["dialog_text", "audio_observation"],
            "text_chat_user_message_audio_observation",
        ),
        (
            [
                {"content_type": "image/png", "description": "image shows a plan"},
                {"content_type": "audio/ogg", "description": "audio mentions a date"},
            ],
            ["dialog_text", "image_observation", "audio_observation"],
            "text_chat_user_message_image_audio_observation",
        ),
    ]

    for media_rows, input_sources, variant in media_profiles:
        episode = build_text_chat_cognitive_episode(
            **_builder_kwargs(),
            media_description_rows=media_rows,
        )
        for stage in _APPROVED_STAGES:
            selection = select_cognition_prompt_variant(
                episode=episode,
                stage=stage,
            )

            assert selection == {
                "stage": stage,
                "variant": variant,
                "prompt_key": f"{stage}.{variant}",
                "trigger_source": "user_message",
                "input_sources": input_sources,
                "output_mode": "visible_reply",
            }


def test_selector_rejects_unapproved_multimodal_ordering() -> None:
    episode = build_text_chat_cognitive_episode(
        **_builder_kwargs(),
        media_description_rows=[
            {"content_type": "image/png", "description": "image shows a plan"},
        ],
    )
    episode["input_sources"] = ["image_observation", "dialog_text"]

    with pytest.raises(CognitionPromptSelectionError, match="input_sources"):
        select_cognition_prompt_variant(
            episode=episode,
            stage="l1_subconscious",
        )


def test_source_payload_contains_structured_image_observations() -> None:
    episode = build_text_chat_cognitive_episode(
        **_builder_kwargs(),
        media_description_rows=[
            {
                "content_type": "image/png",
                "description": "image shows a whiteboard",
            },
            {
                "content_type": "audio/ogg",
                "description": "audio says the meeting moved",
            },
        ],
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
                    "observation_origin": "current_attachment",
                    "media_kind": "image",
                    "summary_status": "available",
                    "summary": "image shows a whiteboard",
                    "visible_text": [],
                    "salient_visual_facts": [],
                    "spatial_or_scene_facts": [],
                    "uncertainty": [],
                },
            ],
            "audio_observations": ["audio says the meeting moved"],
        },
    }
    rendered_payload = json.dumps(source_payload, ensure_ascii=False)
    assert "image/png" not in rendered_payload
    assert "audio/ogg" not in rendered_payload
    assert "percept-dialog:media" not in rendered_payload
    assert "base64_data" not in rendered_payload


@pytest.mark.asyncio
async def test_l2a_multimodal_user_turn_keeps_promoted_reflection_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image-observation user turns should keep normal user-message context."""

    promoted_reflection_context = {
        "self_guidance": [{"summary": "Stay concise around study support."}],
        "global_lore": [],
    }
    user_memory_context = empty_user_memory_context()
    episode = build_text_chat_cognitive_episode(
        **_builder_kwargs(),
        media_description_rows=[{
            "content_type": "image/png",
            "description": "image shows a whiteboard",
        }],
    )
    state = {
        "user_profile": {
            "affinity": 500,
            "last_relationship_insight": "steady baseline",
        },
        "rag_result": {
            "answer": "",
            "memory_evidence": [],
            "world_evidence": [],
            "user_image": {
                "user_memory_context": user_memory_context,
            },
        },
        "cognitive_episode": episode,
        "character_profile": {
            "name": "Character",
            "personality_brief": {"mbti": "INTJ"},
            "mood": "calm",
            "global_vibe": "steady",
        },
        "decontexualized_input": "Can you look at this image?",
        "conversation_progress": {"status": "active"},
        "indirect_speech_context": "",
        "emotional_appraisal": "steady",
        "interaction_subtext": "routine",
        "promoted_reflection_context": promoted_reflection_context,
    }
    response = _DescriptorResponse(json.dumps({
        "internal_monologue": "Use the image as evidence.",
        "logical_stance": "CONFIRM",
        "character_intent": "ANSWER",
    }))
    conscious_llm = AsyncMock()
    conscious_llm.ainvoke = AsyncMock(return_value=response)
    monkeypatch.setattr(l2_module, "_conscious_llm", conscious_llm)

    await l2_module.call_cognition_consciousness(state)

    rendered_messages = conscious_llm.ainvoke.await_args.args[0]
    prompt_payload = json.loads(rendered_messages[1].content)
    assert prompt_payload["media_observations"]["image_observations"]
    assert prompt_payload["promoted_reflection_context"] == promoted_reflection_context


def test_l1_l2_l3_prompt_maps_accept_multimodal_variants() -> None:
    for module in (l1_module, l2_module, l3_module):
        module_source = inspect.getsource(module)
        for variant in _MULTIMODAL_PROMPT_VARIANTS:
            assert variant in module_source


def test_existing_l1_l2_l3_prompt_bytes_are_unchanged() -> None:
    for prompt_name, prompt_text, expected_length, expected_digest in (
        _PROMPT_FINGERPRINTS
    ):
        encoded_prompt = prompt_text.encode("utf-8")
        digest = hashlib.sha256(encoded_prompt).hexdigest()

        assert len(encoded_prompt) == expected_length, prompt_name
        assert digest == expected_digest, prompt_name


@pytest.mark.asyncio
async def test_service_initial_episode_receives_preexisting_image_and_audio_descriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _CapturingServiceGraph()
    _patch_service_dependencies(monkeypatch, graph)
    captured_builder_kwargs: dict[str, object] = {}
    real_builder = service_module.build_text_chat_cognitive_episode

    def _capturing_builder(**kwargs: object) -> dict[str, object]:
        captured_builder_kwargs.update(kwargs)
        episode = real_builder(**kwargs)
        return episode

    monkeypatch.setattr(
        service_module,
        "build_text_chat_cognitive_episode",
        _capturing_builder,
    )
    request = _service_request(
        attachments=[
            {
                "media_type": "image/png",
                "base64_data": "raw-image-bytes",
                "description": "image shows a diagram",
                "url": "https://example.invalid/image.png",
            },
            {
                "media_type": "audio/ogg",
                "description": "audio says hello",
            },
        ],
    )
    item = _queued_item(request)

    await service_module._process_queued_chat_item(item)

    expected_image_observation = {
        "observation_origin": "current_attachment",
        "source_message_id": "",
        "media_kind": "image",
        "summary_status": "available",
        "summary": "image shows a diagram",
        "visible_text": [],
        "salient_visual_facts": [],
        "spatial_or_scene_facts": [],
        "uncertainty": [],
    }
    assert captured_builder_kwargs["media_description_rows"] == [
        {
            "content_type": "image/png",
            "description": "image shows a diagram",
            "image_observation": expected_image_observation,
        },
        {
            "content_type": "audio/ogg",
            "description": "audio says hello",
        },
    ]
    episode = graph.state["cognitive_episode"]
    assert episode["input_sources"] == [
        "dialog_text",
        "image_observation",
        "audio_observation",
    ]
    rendered_episode = json.dumps(episode, ensure_ascii=False)
    assert "raw-image-bytes" not in rendered_episode
    assert "https://example.invalid/image.png" not in rendered_episode


@pytest.mark.asyncio
async def test_descriptor_refreshes_cognitive_episode_after_image_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _descriptor_state()
    original_episode = deepcopy(state["cognitive_episode"])
    state["message_envelope"]["attachments"] = [
        {
            "media_type": "image/png",
            "base64_data": "raw-image-bytes",
            "storage_shape": "inline",
        },
    ]
    state["user_multimedia_input"] = [
        {
            "content_type": "image/png",
            "base64_data": "raw-image-bytes",
            "description": "",
        },
    ]
    descriptor_llm = AsyncMock(
        ainvoke=AsyncMock(
            return_value=_DescriptorResponse(
                '{"description": "image shows a whiteboard"}'
            ),
        ),
    )
    update_descriptions = AsyncMock(return_value=True)
    monkeypatch.setattr(
        decontextualizer_module,
        "_vision_descriptor_llm",
        descriptor_llm,
    )
    monkeypatch.setattr(
        decontextualizer_module,
        "update_conversation_attachment_descriptions",
        update_descriptions,
    )

    result = await decontextualizer_module.multimedia_descriptor_agent(state)

    assert state["cognitive_episode"] == original_episode
    assert result["cognitive_episode"]["input_sources"] == [
        "dialog_text",
        "image_observation",
    ]
    assert result["cognitive_episode"]["percepts"][1]["content"] == (
        "image shows a whiteboard"
    )
    rendered_episode = json.dumps(
        result["cognitive_episode"],
        ensure_ascii=False,
    )
    assert "raw-image-bytes" not in rendered_episode
    update_descriptions.assert_awaited_once()


@pytest.mark.asyncio
async def test_descriptor_audio_description_passes_through_without_audio_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _descriptor_state()
    state["message_envelope"]["attachments"] = [
        {
            "media_type": "audio/ogg",
            "description": "audio says hello",
            "storage_shape": "url_only",
        },
    ]
    state["user_multimedia_input"] = [
        {
            "content_type": "audio/ogg",
            "base64_data": "raw-audio-bytes",
            "description": "audio says hello",
        },
    ]
    descriptor_llm = AsyncMock()
    descriptor_llm.ainvoke = AsyncMock()
    update_descriptions = AsyncMock(return_value=True)
    monkeypatch.setattr(
        decontextualizer_module,
        "_vision_descriptor_llm",
        descriptor_llm,
    )
    monkeypatch.setattr(
        decontextualizer_module,
        "update_conversation_attachment_descriptions",
        update_descriptions,
    )

    result = await decontextualizer_module.multimedia_descriptor_agent(state)

    descriptor_llm.ainvoke.assert_not_awaited()
    assert result["user_multimedia_input"] == state["user_multimedia_input"]
    assert result["cognitive_episode"]["input_sources"] == [
        "dialog_text",
        "audio_observation",
    ]
    assert result["cognitive_episode"]["percepts"][1]["content"] == (
        "audio says hello"
    )
    rendered_episode = json.dumps(
        result["cognitive_episode"],
        ensure_ascii=False,
    )
    assert "raw-audio-bytes" not in rendered_episode


@pytest.mark.asyncio
async def test_descriptor_image_description_without_base64_skips_vision_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _descriptor_state()
    state["message_envelope"]["attachments"] = [
        {
            "media_type": "image/png",
            "description": "adapter supplied image summary",
            "storage_shape": "url_only",
        },
    ]
    state["user_multimedia_input"] = [
        {
            "content_type": "image/png",
            "base64_data": "",
            "description": "adapter supplied image summary",
        },
    ]
    descriptor_llm = AsyncMock()
    descriptor_llm.ainvoke = AsyncMock()
    update_descriptions = AsyncMock(return_value=True)
    monkeypatch.setattr(
        decontextualizer_module,
        "_vision_descriptor_llm",
        descriptor_llm,
    )
    monkeypatch.setattr(
        decontextualizer_module,
        "update_conversation_attachment_descriptions",
        update_descriptions,
    )

    result = await decontextualizer_module.multimedia_descriptor_agent(state)

    descriptor_llm.ainvoke.assert_not_awaited()
    assert result["user_multimedia_input"] == state["user_multimedia_input"]
    assert result["cognitive_episode"]["input_sources"] == [
        "dialog_text",
        "image_observation",
    ]
    assert result["cognitive_episode"]["percepts"][1]["content"] == (
        "adapter supplied image summary"
    )
    update_descriptions.assert_awaited_once()
