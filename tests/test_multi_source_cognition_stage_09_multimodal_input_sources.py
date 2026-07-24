"""V2 multimodal evidence-admission tests."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS,
    MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS,
    build_text_chat_media_description_rows,
    replace_text_chat_media_percepts,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_msg_decontextualizer as decontextualizer_module,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_core_v2_test_helpers import canonical_user_message_episode
from kazusa_ai_chatbot.rag.cognitive_episode_adapter import (
    RAGEpisodeAdapterError,
    build_text_chat_rag_request,
)


TURN_CLOCK = build_turn_clock("2026-05-10 09:30:00")
NOW = "2026-05-09T21:30:00Z"


def _builder_kwargs() -> dict[str, object]:
    return {
        "episode_id": "episode-multimodal-v2",
        "percept_id": "percept-dialog",
        "storage_timestamp_utc": TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": TURN_CLOCK["local_time_context"],
        "user_input": "Please inspect the attached material.",
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


def _rag_request_kwargs(episode: dict[str, object]) -> dict[str, object]:
    return {
        "episode": episode,
        "decontextualized_input": "Please inspect the attached material.",
        "character_profile": {
            "global_user_id": "character-1",
            "name": "Test Character",
        },
        "user_profile": {},
        "prompt_message_context": {
            "body_text": "Please inspect the attached material.",
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


def _descriptor_state() -> dict[str, object]:
    episode = canonical_user_message_episode(**_builder_kwargs())
    return {
        "storage_timestamp_utc": TURN_CLOCK["storage_timestamp_utc"],
        "local_time_context": TURN_CLOCK["local_time_context"],
        "platform": "debug",
        "platform_channel_id": "debug-private-1",
        "platform_message_id": "platform-message-9",
        "platform_user_id": "platform-user-9",
        "global_user_id": "global-user-9",
        "user_name": "Stage Nine",
        "message_envelope": {
            "body_text": "Please inspect the attached material.",
            "raw_wire_text": "Please inspect the attached material.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
        "user_multimedia_input": [],
        "prompt_message_context": {
            "body_text": "Please inspect the attached material.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
        "cognitive_episode": episode,
    }


def _media_percepts(episode: dict[str, object]) -> list[dict[str, object]]:
    """Return only bounded media percepts from a canonical episode."""

    percepts = episode["percepts"]
    assert isinstance(percepts, list)
    return [
        percept
        for percept in percepts
        if isinstance(percept, dict)
        and percept.get("source_kind") in {
            "image_observation",
            "audio_observation",
        }
    ]


def test_episode_admits_bounded_image_and_audio_descriptions() -> None:
    """Retain semantic media descriptions while excluding transport data."""

    media_rows = build_text_chat_media_description_rows([
        {
            "content_type": "image/png",
            "description": "the image shows a whiteboard plan",
            "base64_data": "raw-image-bytes",
        },
        {
            "content_type": "audio/ogg",
            "description": "the audio mentions tomorrow's deadline",
            "url": "https://example.invalid/audio.ogg",
        },
    ])
    episode = canonical_user_message_episode(
        **_builder_kwargs(),
        media_description_rows=media_rows,
    )

    assert [percept["source_kind"] for percept in episode["percepts"]] == [
        "dialog",
        "image_observation",
        "audio_observation",
        "system_event",
    ]
    rendered = str(episode)
    assert "raw-image-bytes" not in rendered
    assert "example.invalid" not in rendered


def test_text_only_builder_is_stable_with_explicit_empty_media() -> None:
    """Adding the media boundary preserves the text-only episode shape."""

    implicit = canonical_user_message_episode(**_builder_kwargs())
    explicit = canonical_user_message_episode(
        **_builder_kwargs(),
        media_description_rows=[],
    )

    assert implicit == explicit
    assert [percept["source_kind"] for percept in implicit["percepts"]] == [
        "dialog",
        "system_event",
    ]
    assert len(implicit["percepts"]) == 2


def test_media_builder_drops_invalid_rows_and_enforces_caps() -> None:
    """Only bounded image/audio semantic descriptions enter the episode."""

    overlong = "x" * (MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS + 50)
    rows = build_text_chat_media_description_rows([
        {"content_type": "video/mp4", "description": "unsupported"},
        {"content_type": "image/png", "description": "   "},
        {"content_type": 9, "description": "invalid type"},
        {"content_type": "image/png", "description": overlong},
        {"content_type": "image/jpeg", "description": "second image"},
        {"content_type": "audio/ogg", "description": "first audio"},
        {"content_type": "audio/mpeg", "description": "second audio"},
        {"content_type": "image/gif", "description": "over cap"},
    ])
    episode = canonical_user_message_episode(
        **_builder_kwargs(),
        media_description_rows=rows,
    )
    media_percepts = _media_percepts(episode)

    assert len(media_percepts) == MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS
    assert len(media_percepts[0]["content"]["description"]) == (
        MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS
    )
    assert media_percepts[0]["content"]["description"].endswith("...")
    assert media_percepts[-1]["content"]["description"] == "second audio"
    assert "over cap" not in json.dumps(episode)


def test_media_refresh_is_non_mutating_and_bounded() -> None:
    """Replace media percepts without mutating the accepted episode."""

    original = canonical_user_message_episode(
        **_builder_kwargs(),
        media_description_rows=[{
            "content_type": "image/png",
            "description": "old image",
        }],
    )
    snapshot = deepcopy(original)
    refreshed = replace_text_chat_media_percepts(
        episode=original,
        media_description_rows=[
            {
                "content_type": "audio/ogg",
                "description": f"audio observation {index}",
            }
            for index in range(MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS + 2)
        ],
    )

    assert original == snapshot
    assert len(_media_percepts(refreshed)) == (
        MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS
    )
    assert [percept["source_kind"] for percept in refreshed["percepts"]] == [
        "dialog",
        "audio_observation",
        "audio_observation",
        "audio_observation",
        "audio_observation",
        "system_event",
    ]


def test_connector_projects_media_as_separate_typed_evidence() -> None:
    """Keep media provenance separate from the current episode evidence."""

    episode = canonical_user_message_episode(
        **_builder_kwargs(),
        media_description_rows=[{
            "content_type": "image/png",
            "description": "the image shows a whiteboard plan",
        }],
    )
    character = build_character_production_state(updated_at=NOW)
    user_state = build_acquaintance_user_state(
        global_user_id="global-user-9",
        updated_at=NOW,
    )
    payload = build_cognition_input_from_global_state(
        {
            "storage_timestamp_utc": TURN_CLOCK["storage_timestamp_utc"],
            "global_user_id": "global-user-9",
            "channel_type": "private",
            "cognitive_episode": episode,
            "user_input": "Please inspect the attached material.",
            "decontextualized_input": "Please inspect the attached material.",
            "rag_result": {
                "user_memory_unit_candidates": [],
            },
            "user_multimedia_input": [{
                "content_type": "image/png",
                "description": "the image shows a whiteboard plan",
            }],
        },
        mutable_state=user_state,
        character_state=character,
    )
    validate_cognition_core_input(payload)

    assert [
        row["evidence_ref"]["source_kind"] for row in payload["evidence"]
    ] == ["episode", "media_observation"]
    media_row = payload["evidence"][1]
    assert media_row["evidence_handle"] == "e2"
    assert media_row["visible_to"] == list(
        EVIDENCE_SOURCE_QUESTION_IDS["media_observation"]
    )


@pytest.mark.parametrize(
    "media_rows",
    [
        [{"content_type": "image/png", "description": "image shows a plan"}],
        [{"content_type": "audio/ogg", "description": "audio names a date"}],
        [
            {"content_type": "image/png", "description": "image shows a plan"},
            {"content_type": "audio/ogg", "description": "audio names a date"},
        ],
    ],
)
def test_rag_accepts_multimodal_episode_but_projects_dialog_only(
    media_rows: list[dict[str, str]],
) -> None:
    """RAG keeps query ownership and excludes separate media evidence."""

    episode = canonical_user_message_episode(
        **_builder_kwargs(),
        media_description_rows=media_rows,
    )

    request = build_text_chat_rag_request(
        **_rag_request_kwargs(episode),  # type: ignore[arg-type]
    )

    assert request["original_query"] == (
        "Please inspect the attached material."
    )
    assert request["current_user_id"] == "global-user-9"
    rendered = json.dumps(request)
    assert "image shows a plan" not in rendered
    assert "audio names a date" not in rendered
    assert "image_observation" not in rendered
    assert "audio_observation" not in rendered


def test_rag_rejects_invalid_multimodal_source_profiles() -> None:
    """The adapter fails closed for reordered or media-only episodes."""

    episode = canonical_user_message_episode(
        **_builder_kwargs(),
        media_description_rows=[{
            "content_type": "image/png",
            "description": "image shows a plan",
        }],
    )
    reordered = deepcopy(episode)
    reordered["percepts"] = [
        reordered["percepts"][1],
        reordered["percepts"][0],
        *reordered["percepts"][2:],
    ]

    with pytest.raises(RAGEpisodeAdapterError):
        build_text_chat_rag_request(
            **_rag_request_kwargs(reordered),  # type: ignore[arg-type]
        )

    media_only = deepcopy(episode)
    media_only["percepts"] = [
        media_only["percepts"][1],
        media_only["percepts"][2],
    ]
    with pytest.raises(RAGEpisodeAdapterError):
        build_text_chat_rag_request(
            **_rag_request_kwargs(media_only),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "description", "expected_source"),
    [
        ("audio/ogg", "audio says hello", "audio_observation"),
        (
            "image/png",
            "adapter supplied image summary",
            "image_observation",
        ),
    ],
)
async def test_descriptor_uses_adapter_description_without_llm(
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
    description: str,
    expected_source: str,
) -> None:
    """Prompt-safe adapter descriptions refresh the episode without a model."""

    state = _descriptor_state()
    state["message_envelope"]["attachments"] = [{
        "media_type": content_type,
        "description": description,
        "storage_shape": "url_only",
    }]
    state["user_multimedia_input"] = [{
        "content_type": content_type,
        "base64_data": "" if content_type.startswith("image/") else "raw-audio",
        "description": description,
    }]
    vision_llm = SimpleNamespace(ainvoke=AsyncMock())
    update_descriptions = AsyncMock(return_value=True)
    monkeypatch.setattr(
        decontextualizer_module,
        "_vision_descriptor_llm",
        vision_llm,
    )
    monkeypatch.setattr(
        decontextualizer_module,
        "update_conversation_attachment_descriptions",
        update_descriptions,
    )

    result = await decontextualizer_module.multimedia_descriptor_agent(state)

    vision_llm.ainvoke.assert_not_awaited()
    assert [
        percept["source_kind"]
        for percept in result["cognitive_episode"]["percepts"]
    ] == ["dialog", expected_source, "system_event"]
    assert result["cognitive_episode"]["percepts"][1]["content"][
        "description"
    ] == description
    assert "raw-audio" not in json.dumps(result["cognitive_episode"])
    update_descriptions.assert_awaited_once()
