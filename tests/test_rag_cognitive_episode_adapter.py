"""Tests for projecting cognitive episodes into the RAG request boundary."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    build_internal_thought_episode,
    build_user_message_episode,
)
from kazusa_ai_chatbot.rag.cognitive_episode_adapter import (
    RAGEpisodeAdapterError,
    build_text_chat_rag_request,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


def _valid_episode() -> CognitiveEpisodeV1:
    """Build a valid canonical text-chat cognitive episode for adapter tests.

    Returns:
        Valid user-message cognitive episode.
    """
    storage_timestamp_utc = "2026-04-26T12:00:00+00:00"
    turn_clock = build_turn_clock_from_storage_utc(storage_timestamp_utc)
    episode = build_user_message_episode(
        episode_id="episode-1",
        origin={
            "platform": "qq",
            "platform_message_id": "msg-1",
            "active_turn_platform_message_ids": ["msg-1", "msg-2"],
            "active_turn_conversation_row_ids": ["row-1", "row-2"],
        },
        target_scope={
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "user-1",
            "current_display_name": "User",
            "target_addressed_user_ids": ["character-1"],
            "target_broadcast": False,
        },
        dialog_percept={
            "schema_version": "percept.v1",
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": "msg-1",
            "content": {"semantic_text": "Need current evidence."},
            "observed_at": storage_timestamp_utc,
        },
        media_percepts=[],
        evidence_refs=[],
        local_time_context=turn_clock["local_time_context"],
        created_at=storage_timestamp_utc,
        debug_controls={},
    )
    return episode


def _internal_thought_episode() -> CognitiveEpisodeV1:
    """Build a valid internal-thought cognitive episode for adapter tests.

    Returns:
        Valid internal-thought cognitive episode.
    """
    return build_internal_thought_episode(
        latch={
            "latch_id": "latch-1",
            "source_episode_id": "source-episode-1",
            "source_action_attempt_id": "action-attempt-1",
            "continuation_objective": "需要回看群聊前文再判断这个内部想法。",
            "target_scope": {
                "platform": "qq",
                "platform_channel_id": "chan-1",
                "channel_type": "group",
                "current_platform_user_id": "user-1",
                "current_global_user_id": "user-1",
                "current_display_name": "User",
                "target_addressed_user_ids": ["user-1"],
                "target_broadcast": False,
            },
            "privacy_scope": "private",
            "continuation_depth": 1,
            "status": "claimed",
            "claim_token": "claim-token-1",
            "attempt_count": 1,
            "max_attempts": 3,
            "expires_at": "2026-04-27T12:00:00+00:00",
            "claim_expires_at": "2026-04-26T13:00:00+00:00",
        },
        evidence_refs=[],
        local_time_context=build_turn_clock_from_storage_utc(
            "2026-04-26T12:00:00+00:00",
        )["local_time_context"],
        created_at="2026-04-26T12:00:00+00:00",
        claim_token="claim-token-1",
    )


def _request_kwargs() -> dict:
    """Build common adapter inputs around one valid episode.

    Returns:
        Keyword arguments accepted by `build_text_chat_rag_request`.
    """
    kwargs = {
        "episode": _valid_episode(),
        "decontexualized_input": "Need current evidence.",
        "character_profile": {
            "global_user_id": "character-1",
            "name": "Kazusa",
        },
        "user_profile": {"relationship_state": 500},
        "prompt_message_context": {
            "body_text": "Need current evidence.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [{"role": "user", "content": "previous turn"}],
        "chat_history_wide": [{"role": "assistant", "content": "older turn"}],
        "reply_context": {"platform_message_id": "reply-1"},
        "indirect_speech_context": "No indirect speech.",
        "conversation_progress": {
            "status": "active",
            "continuity": "same_episode",
            "current_thread": "Pickup plan is active.",
        },
        "conversation_episode_state": {
            "updated_at": "2026-04-26T23:00:00+00:00",
            "turn_count": 7,
        },
        "promoted_reflection_context": {"summary": "recent reflection"},
    }
    return kwargs


def test_valid_text_chat_episode_builds_exact_rag_request_shape() -> None:
    request = build_text_chat_rag_request(**_request_kwargs())

    expected_request = {
        "original_query": "Need current evidence.",
        "character_name": "Kazusa",
        "context": {
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
            "character_profile": {
                "global_user_id": "character-1",
                "name": "Kazusa",
            },
            "active_turn_platform_message_ids": ["msg-1", "msg-2"],
            "active_turn_conversation_row_ids": ["row-1", "row-2"],
            "global_user_id": "user-1",
            "user_name": "User",
            "user_profile": {"relationship_state": 500},
            "current_timestamp_utc": "2026-04-26T12:00:00+00:00",
            "local_time_context": {
                "current_local_datetime": "2026-04-27 00:00",
                "current_local_weekday": "Monday",
            },
            "prompt_message_context": {
                "body_text": "Need current evidence.",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": ["character-1"],
                "broadcast": False,
            },
            "channel_topic": "test",
            "chat_history_recent": [{"role": "user", "content": "previous turn"}],
            "chat_history_wide": [{"role": "assistant", "content": "older turn"}],
            "reply_context": {"platform_message_id": "reply-1"},
            "indirect_speech_context": "No indirect speech.",
            "conversation_progress": {
                "status": "active",
                "continuity": "same_episode",
                "current_thread": "Pickup plan is active.",
            },
            "conversation_episode_state": {
                "updated_at": "2026-04-26T23:00:00+00:00",
                "turn_count": 7,
            },
            "promoted_reflection_context": {"summary": "recent reflection"},
        },
        "current_user_id": "user-1",
        "character_user_id": "character-1",
    }
    assert request == expected_request
    assert "current_" "timestamp" not in request["context"]
    assert "time_context" not in request["context"]


def test_optional_context_fields_default_to_none() -> None:
    kwargs = _request_kwargs()
    del kwargs["conversation_progress"]
    del kwargs["conversation_episode_state"]
    del kwargs["promoted_reflection_context"]

    request = build_text_chat_rag_request(**kwargs)

    assert request["context"]["conversation_progress"] is None
    assert request["context"]["conversation_episode_state"] is None
    assert request["context"]["promoted_reflection_context"] is None


def test_forbidden_context_keys_are_not_exposed() -> None:
    request = build_text_chat_rag_request(**_request_kwargs())

    forbidden_keys = {
        "cognitive_episode",
        "message_envelope",
        "episode_focus",
        "trigger_source",
        "input_sources",
        "percepts",
    }
    assert forbidden_keys.isdisjoint(set(request["context"]))


def test_internal_thought_episode_builds_rag_request_shape() -> None:
    kwargs = _request_kwargs()
    kwargs["episode"] = _internal_thought_episode()
    kwargs["decontexualized_input"] = "回看群聊前文，判断这个内部想法是否有足够证据。"

    request = build_text_chat_rag_request(**kwargs)

    assert request["original_query"] == (
        "回看群聊前文，判断这个内部想法是否有足够证据。"
    )
    assert request["character_name"] == "Kazusa"
    assert request["current_user_id"] == "user-1"
    assert request["character_user_id"] == "character-1"
    assert request["context"]["platform"] == "qq"
    assert request["context"]["platform_channel_id"] == "chan-1"
    assert request["context"]["channel_type"] == "group"
    assert request["context"]["global_user_id"] == "user-1"
    assert request["context"]["user_name"] == "User"
    assert request["context"]["active_turn_platform_message_ids"] == []
    assert request["context"]["active_turn_conversation_row_ids"] == []
    assert request["context"]["prompt_message_context"] == (
        kwargs["prompt_message_context"]
    )
    assert "cognitive_episode" not in request["context"]
    assert "trigger_source" not in request["context"]
    assert "input_sources" not in request["context"]


def test_unregistered_trigger_source_is_rejected() -> None:
    kwargs = _request_kwargs()
    episode = deepcopy(kwargs["episode"])
    episode["trigger_source"] = "unregistered_source"
    kwargs["episode"] = episode

    with pytest.raises(ValueError):
        build_text_chat_rag_request(**kwargs)


def test_internal_thought_without_internal_monologue_source_is_rejected() -> None:
    kwargs = _request_kwargs()
    episode = _internal_thought_episode()
    episode["percepts"][0]["source_kind"] = "retrieved_memory"
    kwargs["episode"] = episode

    with pytest.raises(RAGEpisodeAdapterError):
        build_text_chat_rag_request(**kwargs)


def test_unapproved_user_message_source_set_is_rejected() -> None:
    kwargs = _request_kwargs()
    episode = deepcopy(kwargs["episode"])
    episode["percepts"].append(
        {
            "schema_version": "percept.v1",
            "percept_kind": "retrieved_memory",
            "source_kind": "retrieved_memory",
            "source_id": "memory-1",
            "content": {"semantic_text": "memory observed"},
            "observed_at": episode["created_at"],
        }
    )
    kwargs["episode"] = episode

    with pytest.raises(RAGEpisodeAdapterError):
        build_text_chat_rag_request(**kwargs)


@pytest.mark.parametrize("field_name", ["global_user_id", "name"])
def test_missing_character_profile_required_fields_raise(field_name: str) -> None:
    kwargs = _request_kwargs()
    del kwargs["character_profile"][field_name]

    with pytest.raises(RAGEpisodeAdapterError):
        build_text_chat_rag_request(**kwargs)


@pytest.mark.parametrize("field_name", ["global_user_id", "name"])
def test_empty_character_profile_required_fields_raise(field_name: str) -> None:
    kwargs = _request_kwargs()
    kwargs["character_profile"][field_name] = ""

    with pytest.raises(RAGEpisodeAdapterError):
        build_text_chat_rag_request(**kwargs)
