"""Structural output contracts for normalized cognition stage payloads."""

from __future__ import annotations

from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_prompt_selection import (
    CognitionPromptStage,
)


class CognitionOutputContractError(ValueError):
    """Raised when a normalized cognition stage output is structurally invalid."""


_REQUIRED_OUTPUT_FIELDS: dict[CognitionPromptStage, dict[str, type[object]]] = {
    "l1_subconscious": {
        "emotional_appraisal": str,
        "interaction_subtext": str,
    },
    "l2a_consciousness": {
        "internal_monologue": str,
        "character_intent": str,
        "logical_stance": str,
    },
    "l2b_boundary_core": {
        "boundary_core_assessment": dict,
    },
    "l2c_judgment_core": {
        "logical_stance": str,
        "character_intent": str,
        "judgment_note": str,
    },
    "l3_contextual_agent": {
        "social_distance": str,
        "emotional_intensity": str,
        "vibe_check": str,
        "relational_dynamic": str,
        "expression_willingness": str,
    },
    "l3_style_agent": {
        "rhetorical_strategy": str,
        "linguistic_style": str,
        "forbidden_phrases": list,
    },
    "l3_content_anchor_agent": {
        "content_anchors": list,
    },
    "l3_preference_adapter": {
        "accepted_user_preferences": list,
    },
    "l3_visual_agent": {
        "facial_expression": list,
        "body_language": list,
        "gaze_direction": list,
        "visual_vibe": list,
    },
}


def validate_cognition_output_contract(
    *,
    stage: CognitionPromptStage,
    payload: dict[str, object],
) -> None:
    """Validate normalized cognition output shape for one stage.

    Args:
        stage: L1, L2, or L3 cognition stage that produced the payload.
        payload: Normalized return dict built by the stage handler.

    Raises:
        CognitionOutputContractError: If the stage is unknown, a required key
            is missing, or a required value has the wrong structural type.
    """
    if stage not in _REQUIRED_OUTPUT_FIELDS:
        raise CognitionOutputContractError(f"stage is not supported: {stage}")

    required_fields = _REQUIRED_OUTPUT_FIELDS[stage]
    for field_name, expected_type in required_fields.items():
        if field_name not in payload:
            raise CognitionOutputContractError(
                f"{stage}.{field_name} is required"
            )
        field_value = payload[field_name]
        if not isinstance(field_value, expected_type):
            raise CognitionOutputContractError(
                f"{stage}.{field_name} must be a {expected_type.__name__}"
            )
