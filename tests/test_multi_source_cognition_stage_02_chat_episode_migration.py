"""V2 text-chat episode boundary tests."""

from copy import deepcopy

import pytest

from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.cognition_episode import validate_cognitive_episode_v1
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.time_boundary import (
    local_time_context_from_storage_utc,
)
from tests.cognition_core_v2_test_helpers import canonical_user_message_episode


STORAGE_TIMESTAMP_UTC = "2026-05-01T00:00:00+00:00"
V2_TIMESTAMP = "2026-05-01T00:00:00Z"


def _episode(
    *,
    debug_modes: dict[str, bool] | None = None,
) -> dict:
    """Build one canonical adapter-neutral text episode."""

    return canonical_user_message_episode(
        episode_id="user_message:debug:channel-1:message-1",
        percept_id="user_message:debug:channel-1:message-1:dialog_text:0",
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        user_input="hello",
        platform="debug",
        platform_channel_id="channel-1",
        channel_type="private",
        platform_message_id="message-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=["message-1"],
        active_turn_conversation_row_ids=["row-1"],
        debug_modes=dict(debug_modes or {}),
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )


def test_text_chat_episode_remains_adapter_neutral_and_typed() -> None:
    """The adapter passes one typed episode into the V2 persona boundary."""

    episode = canonical_user_message_episode(
        episode_id="episode-1",
        percept_id="percept-1",
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        user_input="hello",
        platform="debug",
        platform_channel_id="channel-1",
        channel_type="private",
        platform_message_id="message-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=["message-1"],
        active_turn_conversation_row_ids=[],
        debug_modes={},
    )

    validate_cognitive_episode_v1(episode)
    assert episode["trigger_source"] == "user_message"
    assert [percept["source_kind"] for percept in episode["percepts"]] == [
        "dialog",
        "system_event",
    ]
    assert episode["percepts"][0]["content"]["semantic_text"] == "hello"


@pytest.mark.parametrize(
    ("platform_channel_id", "platform_message_id", "row_id", "sequence", "reference"),
    [
        ("channel-1", "message-1", "row-1", 7, "channel-1:message-1"),
        ("", "", "row-1", 7, "direct:row-1"),
        ("", "", None, 7, "direct:queue-7"),
    ],
)
def test_service_episode_ids_retain_stable_fallback_order(
    platform_channel_id: str,
    platform_message_id: str,
    row_id: str | None,
    sequence: int,
    reference: str,
) -> None:
    """Message, persisted-row, and queue identities remain deterministic."""

    episode_id, percept_id = service_module._build_text_chat_episode_ids(
        platform="debug",
        platform_channel_id=platform_channel_id,
        platform_message_id=platform_message_id,
        conversation_row_id=row_id,
        queue_sequence=sequence,
    )

    assert episode_id == f"user_message:debug:{reference}"
    assert percept_id == f"{episode_id}:dialog_text:0"


@pytest.mark.parametrize(
    "debug_modes",
    [
        {},
        {"think_only": True},
        {"listen_only": True},
    ],
)
def test_text_episode_preserves_debug_controls_and_origin_flags(
    debug_modes: dict[str, bool],
) -> None:
    """Service-owned debug controls reach canonical episode metadata."""

    episode = _episode(debug_modes=debug_modes)

    origin_metadata = episode["origin_metadata"]
    assert origin_metadata["platform"] == "debug"
    assert origin_metadata["platform_message_id"] == "message-1"
    assert origin_metadata["active_turn_platform_message_ids"] == ["message-1"]
    assert origin_metadata["active_turn_conversation_row_ids"] == ["row-1"]
    assert origin_metadata["debug_modes"] == debug_modes
    assert origin_metadata["debug_controls"] == debug_modes
    assert episode["target_scope"] == {
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "channel_type": "private",
        "current_platform_user_id": "platform-user-1",
        "current_global_user_id": "user-1",
        "current_display_name": "User",
        "target_addressed_user_ids": ["character-1"],
        "target_broadcast": False,
    }


def test_v2_connector_preserves_episode_and_projects_current_percept() -> None:
    """The canonical episode crosses the persona connector without rewriting."""

    episode = _episode()
    original = deepcopy(episode)
    character_state = build_character_production_state(updated_at=V2_TIMESTAMP)
    state = {
        "cognitive_episode": episode,
        "global_user_id": "user-1",
        "user_input": "fallback text",
        "decontextualized_input": "fallback semantic text",
        "user_multimedia_input": [],
        "rag_result": {"memory_evidence": []},
    }

    payload = build_cognition_input_from_global_state(
        state,
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=V2_TIMESTAMP,
        ),
        character_state=character_state,
    )

    assert payload["episode"] == original
    assert episode == original
    assert payload["evidence"][0]["semantic_text"] == "hello"
    assert payload["evidence"][0]["evidence_ref"]["source_id"] == (
        "user_message:debug:channel-1:message-1:dialog_text:0"
    )
