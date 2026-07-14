"""Checkpoint F connector mapping tests for the canonical V2 caller."""

import json

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)


NOW = "2026-07-14T00:00:00Z"


def _global_state() -> dict[str, object]:
    """Build the adapter-owned fields needed by the V2 mapper."""

    return {
        "character_profile": {"global_user_id": "character-1"},
        "character_cognition_state": build_character_production_state(
            updated_at=NOW,
        ),
        "storage_timestamp_utc": NOW,
        "user_input": "hello",
        "decontexualized_input": "hello",
        "prompt_message_context": {},
        "cognitive_episode": {
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "output_mode": "live_response",
        },
        "user_multimedia_input": [],
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "channel_type": "dm",
        "channel_name": "",
        "platform_message_id": "message-1",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {},
        "platform_bot_id": "bot-1",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "rag_result": {"memory_evidence": []},
    }


def test_persona_connector_maps_one_native_user_scope() -> None:
    """The caller sends native V2 state and typed evidence to the core."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_input_from_global_state,
    )

    payload = build_cognition_input_from_global_state(
        _global_state(),
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        ),
    )

    assert payload["schema_version"] == "cognition_core_input.v2"
    assert payload["state_scope"] == "user"
    assert payload["mutable_state"]["state_scope"] == "user"
    assert payload["evidence"][0]["evidence_ref"]["source_kind"] == "episode"
    assert "platform_channel_id" not in json.dumps(payload)


def test_connector_keeps_media_as_typed_evidence_without_wire_payloads() -> None:
    """Media descriptions remain semantic evidence while raw bytes and URLs stay out."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_input_from_global_state,
    )

    state = _global_state()
    state["user_multimedia_input"] = [{
        "content_type": "image/png",
        "base64_data": "raw-bytes",
        "url": "https://example.invalid/image.png",
        "description": "whiteboard observation",
    }]
    state["user_input"] = ""
    state["decontexualized_input"] = ""
    payload = build_cognition_input_from_global_state(
        state,
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        ),
    )
    rendered = json.dumps(payload, ensure_ascii=False)

    assert "whiteboard observation" in rendered
    assert "raw-bytes" not in rendered
    assert "example.invalid" not in rendered
