"""Shared deterministic dialog-state fixtures for node owner tests."""

from __future__ import annotations

from tests.cognition_test_helpers import canonical_episode


def build_dialog_state() -> dict[str, object]:
    """Build the minimal renderer state with canonical current-turn grounding."""

    episode = canonical_episode(
        content="Infer which option fits my stated preference.",
    )
    surface_input = {
        "schema_version": "text_surface_input.v2",
        "episode": episode,
        "intention": {
            "route": "speech",
            "intention": "answer by inference",
            "target_roles": [],
            "reason": "the current request asks for an inference",
            "goal_continuation_ref": None,
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "warm",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "brief conversational speech",
        "character_expression_context": {
            "tempo": "steady",
            "linguistic_texture": (
                "Light hesitation in concise spoken clauses."
            ),
        },
        "visual_character_context": (
            "A physical mannerism accompanies emotion."
        ),
    }
    surface_output = {
        "schema_version": "text_surface_output.v2",
        "content_plan": "Answer the current request by inference.",
        "content_requirements": [
            "Preserve the requested response operation and current time scope.",
        ],
        "visible_boundaries": [],
        "addressee_plan": [{
            "handle": "current_user",
            "display_name": "Current User",
            "semantic_role": "direct_recipient",
            "wording_policy": "second_person_allowed",
        }],
        "delivery_profile": {
            "lexical_register": "warm",
            "sentence_shape": "concise",
            "rhythm": "steady",
            "hesitation": "light",
            "punctuation": "restrained",
        },
        "selected_surface_intent": "answer by inference",
        "permitted_action_results": [],
        "lexical_avoidances": [],
    }
    return {
        "internal_monologue": "I can answer directly.",
        "text_surface_input_v2": surface_input,
        "text_surface_output_v2": surface_output,
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "platform-user",
        "platform_bot_id": "platform-bot",
        "global_user_id": "global-user",
        "user_name": "Current User",
        "user_profile": {},
        "character_profile": {},
        "cognitive_episode": episode,
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "dialog_usage_mode": "unit_test",
        "llm_trace_id": "visible-speech-test",
    }
