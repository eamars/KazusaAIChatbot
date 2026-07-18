"""Tests for the source-neutral cognitive episode contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot import cognition_episode as cognition_episode_module
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    CognitiveEpisodeValidationError,
    TargetScopeV1,
    TextChatCompatibilityProjection,
    build_user_message_episode,
    project_model_visible_percepts,
    project_text_chat_compatibility_fields,
    validate_cognitive_episode_v1,
)
from kazusa_ai_chatbot.consolidation.schema import ConsolidatorState
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    CognitionState,
    GlobalPersonaState,
)
from kazusa_ai_chatbot.state import IMProcessState


CREATED_AT = "2026-05-09T07:30:00+00:00"


def _local_time_context() -> dict[str, str]:
    """Return one valid configured-local time context."""

    return {
        "current_local_datetime": "2026-05-09 19:30",
        "current_local_weekday": "Saturday",
    }


def _target_scope() -> TargetScopeV1:
    """Return the current debug-channel target scope."""

    return {
        "platform": "debug",
        "platform_channel_id": "debug-private-1",
        "channel_type": "private",
        "current_platform_user_id": "platform-user-1",
        "current_global_user_id": "global-user-1",
        "current_display_name": "Test User",
        "target_addressed_user_ids": ["global-user-1"],
        "target_broadcast": False,
    }


def _dialog_percept() -> dict[str, object]:
    """Return one prompt-visible current-user percept."""

    return {
        "schema_version": "percept.v1",
        "percept_kind": "dialog",
        "source_kind": "dialog",
        "source_id": "platform-message-1",
        "content": {
            "semantic_text": "hello from current text chat",
            "text": "hello from current text chat",
        },
        "observed_at": CREATED_AT,
    }


def _valid_episode() -> CognitiveEpisodeV1:
    """Build one canonical user-message episode."""

    return build_user_message_episode(
        episode_id="episode-1",
        origin={
            "platform": "debug",
            "platform_message_id": "platform-message-1",
            "active_turn_platform_message_ids": ["platform-message-1"],
            "active_turn_conversation_row_ids": ["conversation-row-1"],
            "debug_modes": {"think_only": False},
        },
        target_scope=_target_scope(),
        dialog_percept=_dialog_percept(),
        media_percepts=[],
        evidence_refs=[],
        local_time_context=_local_time_context(),
        created_at=CREATED_AT,
        debug_controls={"think_only": False},
    )


def test_public_contract_shapes_expose_expected_fields() -> None:
    """The canonical envelope and state boundaries expose their native fields."""

    assert set(CognitiveEpisodeV1.__annotations__) == {
        "schema_version",
        "episode_id",
        "trigger_source",
        "origin_metadata",
        "target_scope",
        "percepts",
        "evidence_refs",
        "created_at",
        "privacy_scope",
        "continuation_depth",
    }
    assert set(TargetScopeV1.__annotations__) == {
        "platform",
        "platform_channel_id",
        "channel_type",
        "current_platform_user_id",
        "current_global_user_id",
        "current_display_name",
        "target_addressed_user_ids",
        "target_broadcast",
        "permission_ref",
    }
    assert set(TextChatCompatibilityProjection.__annotations__) == {
        "storage_timestamp_utc",
        "local_time_context",
        "user_input",
        "platform",
        "platform_channel_id",
        "channel_type",
        "platform_message_id",
        "active_turn_platform_message_ids",
        "active_turn_conversation_row_ids",
        "platform_user_id",
        "global_user_id",
        "user_name",
    }
    for state_schema in (
        IMProcessState,
        GlobalPersonaState,
        CognitionState,
        ConsolidatorState,
    ):
        assert "storage_timestamp_utc" in state_schema.__annotations__
        assert "local_time_context" in state_schema.__annotations__
        assert "timestamp" not in state_schema.__annotations__
        assert "time_context" not in state_schema.__annotations__


def test_user_message_builder_creates_valid_canonical_episode() -> None:
    """The user-message builder creates the exact five-source envelope."""

    episode = _valid_episode()

    assert validate_cognitive_episode_v1(episode) == episode
    assert episode["schema_version"] == "cognitive_episode.v1"
    assert episode["episode_id"] == "episode-1"
    assert episode["trigger_source"] == "user_message"
    assert episode["created_at"] == CREATED_AT
    assert episode["privacy_scope"] == "private"
    assert episode["continuation_depth"] == 0
    assert episode["target_scope"] == _target_scope()
    assert [percept["percept_kind"] for percept in episode["percepts"]] == [
        "dialog",
        "local_time_context",
    ]


def test_model_projection_contains_dialog_and_local_time_percepts() -> None:
    """Model projection preserves content while omitting deterministic ids."""

    episode = _valid_episode()
    rows = project_model_visible_percepts(episode)

    assert rows == [
        {
            "input_source": "dialog",
            "content": {
                "semantic_text": "hello from current text chat",
                "text": "hello from current text chat",
            },
            "speaker_role": "current_user",
            "addressee_role": "self",
            "first_person_role": "current_user",
            "implicit_imperative_subject_role": "self",
        },
        {
            "input_source": "local_time_context",
            "content": {"local_time_context": _local_time_context()},
        },
    ]
    assert "platform-message-1" not in str(rows)


def test_compatibility_projection_returns_current_text_chat_fields_only() -> None:
    """RAG receives the bounded adapter-neutral text-chat projection."""

    projection = project_text_chat_compatibility_fields(_valid_episode())

    assert set(projection) == set(TextChatCompatibilityProjection.__annotations__)
    assert projection == {
        "storage_timestamp_utc": CREATED_AT,
        "local_time_context": _local_time_context(),
        "user_input": "hello from current text chat",
        "platform": "debug",
        "platform_channel_id": "debug-private-1",
        "channel_type": "private",
        "platform_message_id": "platform-message-1",
        "active_turn_platform_message_ids": ["platform-message-1"],
        "active_turn_conversation_row_ids": ["conversation-row-1"],
        "platform_user_id": "platform-user-1",
        "global_user_id": "global-user-1",
        "user_name": "Test User",
    }


def test_dialog_semantic_projection_is_model_owned() -> None:
    """Role meaning is attached to the dialog percept as bounded metadata."""

    role_content = "The current user asks the character to choose the next move."
    response_operation = {
        "operation": "Choose one next action for the current user.",
        "response_owner_role": "self",
        "selection_owner_role": "self",
        "selection_required": True,
        "embedded_actor_role": "current_user",
        "embedded_target_role": "self",
    }

    projected = cognition_episode_module.attach_dialog_semantic_projection(
        _valid_episode(),
        role_content,
        response_operation,
    )

    dialog_content = projected["percepts"][0]["content"]
    assert dialog_content["role_explicit_content"] == role_content
    assert dialog_content["response_operation"] == response_operation


def test_validate_rejects_empty_percepts() -> None:
    """The canonical envelope requires at least one percept."""

    episode = deepcopy(_valid_episode())
    episode["percepts"] = []

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode_v1(episode)


def test_validate_rejects_legacy_episode_fields() -> None:
    """Legacy parallel time fields cannot enter the canonical envelope."""

    episode = deepcopy(_valid_episode())
    episode["timestamp"] = CREATED_AT

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode_v1(episode)


def test_validate_rejects_unregistered_source_and_excess_depth() -> None:
    """Source ownership and bounded continuation depth are deterministic."""

    unsupported = deepcopy(_valid_episode())
    unsupported["trigger_source"] = "unregistered_source"

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode_v1(unsupported)

    too_deep = deepcopy(_valid_episode())
    too_deep["continuation_depth"] = 2

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode_v1(too_deep)
