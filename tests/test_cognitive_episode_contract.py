"""Tests for the source-neutral cognitive episode contract."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

from kazusa_ai_chatbot import cognition_episode as cognition_episode_module
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    CognitiveEpisodeValidationError,
    CognitivePercept,
    OriginMetadata,
    TargetScope,
    TextChatCompatibilityProjection,
    build_text_chat_cognitive_episode,
    project_text_chat_compatibility_fields,
    validate_cognitive_episode,
)
from kazusa_ai_chatbot.consolidation.schema import (
    ConsolidatorState,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    CognitionState,
    GlobalPersonaState,
)
from kazusa_ai_chatbot.state import IMProcessState


def _local_time_context() -> dict[str, str]:
    local_time_context = {
        "current_local_datetime": "2026-05-09 19:30",
        "current_local_weekday": "Saturday",
    }
    return local_time_context


def _builder_kwargs() -> dict:
    kwargs = {
        "episode_id": "episode-1",
        "percept_id": "percept-1",
        "storage_timestamp_utc": "2026-05-09T07:30:00+00:00",
        "local_time_context": _local_time_context(),
        "user_input": "hello from current text chat",
        "platform": "debug",
        "platform_channel_id": "debug-private-1",
        "channel_type": "private",
        "platform_message_id": "platform-message-1",
        "platform_user_id": "platform-user-1",
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "active_turn_platform_message_ids": ["platform-message-1"],
        "active_turn_conversation_row_ids": ["conversation-row-1"],
        "debug_modes": {"think_only": False},
        "target_addressed_user_ids": ["global-user-1"],
        "target_broadcast": False,
    }
    return kwargs


def _valid_episode() -> CognitiveEpisode:
    episode = build_text_chat_cognitive_episode(**_builder_kwargs())
    return episode


def test_public_contract_shapes_expose_expected_fields() -> None:
    assert set(CognitivePercept.__annotations__) == {
        "percept_id",
        "input_source",
        "content",
        "visibility",
        "metadata",
    }
    assert set(TargetScope.__annotations__) == {
        "platform",
        "platform_channel_id",
        "channel_type",
        "current_platform_user_id",
        "current_global_user_id",
        "current_display_name",
        "target_addressed_user_ids",
        "target_broadcast",
    }
    assert set(OriginMetadata.__annotations__) == {
        "platform",
        "platform_message_id",
        "active_turn_platform_message_ids",
        "active_turn_conversation_row_ids",
        "debug_modes",
    }
    assert set(CognitiveEpisode.__annotations__) == {
        "episode_id",
        "trigger_source",
        "input_sources",
        "output_mode",
        "percepts",
        "target_scope",
        "origin_metadata",
        "storage_timestamp_utc",
        "local_time_context",
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


def test_text_chat_builder_creates_valid_user_message_episode() -> None:
    episode = _valid_episode()

    validate_cognitive_episode(episode)

    assert episode["episode_id"] == "episode-1"
    assert episode["trigger_source"] == "user_message"
    assert episode["input_sources"] == ["dialog_text"]
    assert episode["output_mode"] == "visible_reply"
    assert episode["storage_timestamp_utc"] == "2026-05-09T07:30:00+00:00"
    assert episode["local_time_context"] == _local_time_context()
    assert episode["target_scope"] == {
        "platform": "debug",
        "platform_channel_id": "debug-private-1",
        "channel_type": "private",
        "current_platform_user_id": "platform-user-1",
        "current_global_user_id": "global-user-1",
        "current_display_name": "Test User",
        "target_addressed_user_ids": ["global-user-1"],
        "target_broadcast": False,
    }
    assert episode["origin_metadata"] == {
        "platform": "debug",
        "platform_message_id": "platform-message-1",
        "active_turn_platform_message_ids": ["platform-message-1"],
        "active_turn_conversation_row_ids": ["conversation-row-1"],
        "debug_modes": {"think_only": False},
    }
    assert episode["percepts"] == [
        {
            "percept_id": "percept-1",
            "input_source": "dialog_text",
            "content": "hello from current text chat",
            "visibility": "model_visible",
            "metadata": {},
        }
    ]


def test_background_artifact_result_ready_builder_creates_valid_episode() -> None:
    """Completed artifact jobs should enter cognition through a typed source."""

    builder = getattr(
        cognition_episode_module,
        "build_background_artifact_result_ready_cognitive_episode",
    )

    episode = builder(
        episode_id="background_artifact_result_ready:job-001",
        percept_id="background_artifact_result_ready:job-001:result:0",
        storage_timestamp_utc="2026-05-09T07:30:00+00:00",
        local_time_context=_local_time_context(),
        job_id="job-001",
        work_kind="coding_snippet",
        objective_summary="Generate a Fibonacci function snippet.",
        artifact_text="def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        failure_summary="",
        platform="debug",
        platform_channel_id="debug-private-1",
        channel_type="private",
        platform_message_id="job-001",
        requester_platform_user_id="platform-user-1",
        requester_global_user_id="global-user-1",
        requester_display_name="Test User",
        source_character_name="Test Character",
    )

    validate_cognitive_episode(episode)

    assert episode["trigger_source"] == "background_artifact_result_ready"
    assert episode["input_sources"] == ["background_artifact_result"]
    assert episode["output_mode"] == "visible_reply"
    assert episode["target_scope"]["current_global_user_id"] == "global-user-1"
    assert episode["percepts"][0]["input_source"] == "background_artifact_result"
    assert episode["percepts"][0]["visibility"] == "model_visible"
    assert episode["percepts"][0]["metadata"] == {
        "job_id": "job-001",
        "work_kind": "coding_snippet",
        "objective_summary": "Generate a Fibonacci function snippet.",
        "failure_summary": "",
        "source_character_name": "Test Character",
    }


def test_background_work_result_ready_builder_creates_valid_episode() -> None:
    """Completed background work should enter cognition through a typed source."""

    builder = getattr(
        cognition_episode_module,
        "build_background_work_result_ready_cognitive_episode",
    )

    episode = builder(
        episode_id="background_work_result_ready:job-001",
        percept_id="background_work_result_ready:job-001:result:0",
        storage_timestamp_utc="2026-06-06T07:30:00+00:00",
        local_time_context=_local_time_context(),
        task_brief="Generate a Fibonacci function snippet.",
        artifact_text="def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        failure_summary="",
        result_summary="Generated a compact Fibonacci snippet.",
        worker="text_artifact",
        worker_metadata={"task_type": "coding_snippet"},
        platform="debug",
        platform_channel_id="debug-private-1",
        channel_type="private",
        platform_message_id="job-001",
        requester_platform_user_id="platform-user-1",
        requester_global_user_id="global-user-1",
        requester_display_name="Test User",
        source_platform_bot_id="bot-1",
        source_character_name="Test Character",
    )

    validate_cognitive_episode(episode)

    assert episode["trigger_source"] == "background_work_result_ready"
    assert episode["input_sources"] == ["background_work_result"]
    assert episode["output_mode"] == "visible_reply"
    assert episode["target_scope"]["current_global_user_id"] == "global-user-1"
    assert episode["percepts"][0]["input_source"] == "background_work_result"
    assert episode["percepts"][0]["visibility"] == "model_visible"
    assert episode["percepts"][0]["metadata"] == {
        "task_brief": "Generate a Fibonacci function snippet.",
        "failure_summary": "",
        "result_summary": "Generated a compact Fibonacci snippet.",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "worker": "text_artifact",
        "worker_metadata": {"task_type": "coding_snippet"},
    }


def test_text_chat_builder_defaults_optional_collections() -> None:
    kwargs = _builder_kwargs()
    del kwargs["active_turn_platform_message_ids"]
    del kwargs["active_turn_conversation_row_ids"]
    del kwargs["debug_modes"]
    del kwargs["target_addressed_user_ids"]

    episode = build_text_chat_cognitive_episode(**kwargs)

    validate_cognitive_episode(episode)
    assert episode["target_scope"]["target_addressed_user_ids"] == []
    assert episode["origin_metadata"]["active_turn_platform_message_ids"] == []
    assert episode["origin_metadata"]["active_turn_conversation_row_ids"] == []
    assert episode["origin_metadata"]["debug_modes"] == {}


def test_compatibility_projection_returns_current_text_chat_fields_only() -> None:
    episode = _valid_episode()

    projection = project_text_chat_compatibility_fields(episode)

    assert set(projection) == set(TextChatCompatibilityProjection.__annotations__)
    assert projection == {
        "storage_timestamp_utc": "2026-05-09T07:30:00+00:00",
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


def test_validate_rejects_empty_percepts() -> None:
    episode = deepcopy(_valid_episode())
    episode["percepts"] = []

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode(episode)


def test_validate_rejects_unsupported_output_mode() -> None:
    episode = deepcopy(_valid_episode())
    episode["output_mode"] = "send_later"

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode(episode)


def test_validate_rejects_missing_target_scope() -> None:
    episode = deepcopy(_valid_episode())
    del episode["target_scope"]

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode(episode)


def test_validate_rejects_mismatched_input_sources() -> None:
    episode = deepcopy(_valid_episode())
    episode["input_sources"] = ["retrieved_memory"]

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode(episode)


def test_validate_rejects_duplicate_percept_ids() -> None:
    episode = deepcopy(_valid_episode())
    second_percept = deepcopy(episode["percepts"][0])
    second_percept["content"] = "second percept"
    episode["percepts"].append(second_percept)

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode(episode)


def test_validate_rejects_missing_time_context_fields() -> None:
    episode = deepcopy(_valid_episode())
    del episode["local_time_context"]["current_local_weekday"]

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode(episode)


def test_validate_rejects_legacy_episode_time_fields() -> None:
    episode = deepcopy(_valid_episode())
    episode["timestamp"] = "2026-05-09T07:30:00+00:00"
    episode["time_context"] = _local_time_context()

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode(episode)


def _remove_episode_id(episode: dict) -> None:
    del episode["episode_id"]


def _remove_trigger_source(episode: dict) -> None:
    del episode["trigger_source"]


def _use_non_bool_debug_mode(episode: dict) -> None:
    episode["origin_metadata"]["debug_modes"]["think_only"] = "false"


def _use_unsupported_visibility(episode: dict) -> None:
    episode["percepts"][0]["visibility"] = "public"


def _remove_dialog_text_from_user_message(episode: dict) -> None:
    episode["input_sources"] = ["retrieved_memory"]
    episode["percepts"][0]["input_source"] = "retrieved_memory"


@pytest.mark.parametrize(
    "mutate_episode",
    [
        _remove_episode_id,
        _remove_trigger_source,
        _use_non_bool_debug_mode,
        _use_unsupported_visibility,
        _remove_dialog_text_from_user_message,
    ],
)
def test_validate_rejects_named_structural_rule_gaps(
    mutate_episode: Callable[[dict], None],
) -> None:
    episode = deepcopy(_valid_episode())
    mutate_episode(episode)

    with pytest.raises(CognitiveEpisodeValidationError):
        validate_cognitive_episode(episode)
