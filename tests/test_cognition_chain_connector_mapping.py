"""Tests for Kazusa graph connectors around the cognition-chain core."""

import json


def _global_state() -> dict:
    return {
        "character_profile": {
            "global_user_id": "character-1",
            "name": "Kazusa",
            "description": "character",
            "gender": "unknown",
            "age": "unknown",
            "birthday": "unknown",
            "background": "none",
            "personality": "steady",
            "mood": "neutral",
            "global_vibe": "calm",
        },
        "storage_timestamp_utc": "2026-06-12T00:00:00Z",
        "local_time_context": {"local_time": "day"},
        "user_input": "hello",
        "prompt_message_context": {},
        "cognitive_episode": {
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "live_response",
        },
        "user_multimedia_input": [],
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "channel_type": "dm",
        "platform_message_id": "message-1",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {
            "global_user_id": "user-1",
            "user_name": "User",
            "affinity": 500,
            "affinity_level": "known",
            "last_relationship_insight": "none",
        },
        "platform_bot_id": "bot-1",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "conversation_progress": {},
        "promoted_reflection_context": {},
        "internal_monologue_residue_context": "",
        "debug_modes": {},
        "should_respond": True,
        "decontexualized_input": "hello",
        "referents": [],
        "rag_result": {"answer": "", "memory_evidence": []},
        "resolver_context": "",
        "internal_monologue": "considered",
        "action_directives": {},
        "interaction_subtext": "plain",
        "emotional_appraisal": "neutral",
        "character_intent": "respond",
        "logical_stance": "answer",
        "judgment_note": "safe",
        "social_distance": "near",
        "emotional_intensity": "low",
        "vibe_check": "calm",
        "relational_dynamic": "stable",
        "final_dialog": "",
        "mood": "neutral",
        "global_vibe": "calm",
        "reflection_summary": "",
        "subjective_appraisals": [],
        "affinity_delta": 0,
        "last_relationship_insight": "none",
        "new_facts": [],
        "future_promises": [],
    }


def _chain_output() -> dict:
    return {
        "schema_version": "cognition_chain_output.v1",
        "cognition_residue": {
            "emotional_appraisal": "neutral",
            "interaction_subtext": "plain",
            "internal_monologue": "considered",
            "logical_stance": "answer",
            "character_intent": "respond",
            "judgment_note": "safe",
            "social_distance": "near",
            "emotional_intensity": "low",
            "vibe_check": "calm",
            "relational_dynamic": "stable",
        },
        "semantic_action_requests": [{
            "capability": "speak",
            "decision": "visible_reply",
            "detail": "answer the user",
            "reason": "direct greeting",
        }],
        "resolver_capability_requests": [],
        "chain_trace": {
            "stage_order": ["l1", "l2a", "l2b", "l2c1", "l2c2", "l2d"],
            "selected_actions_summary": "speak",
            "resolver_summary": "",
            "warnings": [],
        },
    }


def test_persona_connector_maps_global_state_to_chain_input() -> None:
    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_chain_input_from_global_state,
    )

    chain_input = build_cognition_chain_input_from_global_state(_global_state())

    assert chain_input["schema_version"] == "cognition_chain_input.v1"
    assert chain_input["current_event"]["decontextualized_input"] == "hello"
    assert chain_input["scene"]["platform"] == "debug"
    assert "platform_channel_id" not in chain_input["scene"]
    assert "action_specs" not in chain_input


def test_persona_connector_projects_media_observations_to_chain_input() -> None:
    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_chain_input_from_global_state,
    )

    state = _global_state()
    state["user_multimedia_input"] = [
        {
            "content_type": "image/png",
            "base64_data": "raw-bytes",
            "url": "https://example.invalid/image.png",
            "description": " image shows a whiteboard ",
        },
        {
            "content_type": "audio/ogg",
            "base64_data": "raw-audio",
            "description": " user says the deadline moved ",
        },
    ]

    chain_input = build_cognition_chain_input_from_global_state(state)

    assert chain_input["current_event"]["media_observations"] == [
        {
            "modality": "image",
            "observation": "image shows a whiteboard",
            "source_summary": "current attachment",
        },
        {
            "modality": "audio",
            "observation": "user says the deadline moved",
            "source_summary": "current attachment",
        },
    ]
    percept_sources = [
        percept["input_source"]
        for percept in chain_input["episode"]["model_visible_percepts"]
    ]
    assert percept_sources == [
        "chat_message",
        "image_observation",
        "audio_observation",
    ]
    rendered = json.dumps(chain_input, ensure_ascii=False)
    assert "raw-bytes" not in rendered
    assert "raw-audio" not in rendered
    assert "example.invalid" not in rendered


def test_persona_connector_materializes_semantic_actions_after_core_output() -> None:
    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        apply_cognition_chain_output_to_global_state,
    )

    update = apply_cognition_chain_output_to_global_state(
        _chain_output(),
        _global_state(),
    )

    assert update["internal_monologue"] == "considered"
    assert update["action_specs"][0]["kind"] == "speak"
    assert update["action_specs"][0]["params"]["surface_requirements"][
        "detail"
    ] == "answer the user"


def test_l3_surface_connector_projects_selected_speak_without_action_spec_leak() -> None:
    from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
        build_text_surface_input_from_global_state,
    )

    state = _global_state()
    state["action_specs"] = [{
        "kind": "speak",
        "reason": "direct greeting",
        "params": {
            "surface_requirements": {
                "decision": "visible_reply",
                "intent": "respond",
                "detail": "answer the user",
                "tone": "warm",
            },
        },
    }]

    surface_input = build_text_surface_input_from_global_state(state)

    assert surface_input["schema_version"] == "cognition_text_surface_input.v1"
    assert surface_input["selected_text_surface_intent"]["detail"] == (
        "answer the user"
    )
    assert "action_specs" not in surface_input
