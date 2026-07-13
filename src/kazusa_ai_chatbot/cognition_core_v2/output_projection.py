"""Projection from authoritative V2 state into the unchanged V1 output shape."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EmotionActivation,
    WorkspaceResult,
)


def project_v1_output(
    activations: Mapping[str, EmotionActivation],
    workspace: WorkspaceResult,
    action_requests: list[dict[str, str]],
    warnings: Iterable[str],
) -> dict[str, object]:
    """Map causal projections and workspace output onto the V1 chain schema.

    Args:
        activations: Deterministically derived concurrent emotion projections.
        workspace: Collapsed selected and suppressed cognition bids.
        action_requests: Route-only action selection output.
        warnings: Validation-local branch or parser failure summaries.

    Returns:
        A V1-shaped output ready for public contract validation.
    """

    active = [
        activation
        for activation in activations.values()
        if activation.activation > 0.0
    ]
    active.sort(key=lambda activation: (-activation.activation, activation.emotion_id))
    dominant = active[0] if active else None
    affect_summary = "no causally active emotion"
    intensity = "none"
    if dominant is not None:
        affect_summary = _affect_summary(dominant, active[1:])
        intensity = _activation_descriptor(dominant.activation)
    output = {
        "schema_version": "cognition_chain_output.v1",
        "cognition_residue": {
            "emotional_appraisal": affect_summary,
            "interaction_subtext": workspace.internal_summary,
            "internal_monologue": workspace.internal_summary,
            "logical_stance": workspace.public_intention or "defer until grounded",
            "character_intent": workspace.public_intention or "no public action",
            "judgment_note": _judgment_note(dominant),
            "social_distance": "preserve current relationship boundary",
            "emotional_intensity": intensity,
            "vibe_check": "causally grounded" if dominant is not None else "unassessed",
            "relational_dynamic": "retain unresolved motives",
        },
        "semantic_action_requests": action_requests,
        "resolver_capability_requests": [],
        "chain_trace": {
            "stage_order": ["v2"],
            "selected_actions_summary": workspace.selected_bid_id or "",
            "resolver_summary": "",
            "warnings": list(warnings),
        },
    }
    return output


def _affect_summary(
    dominant: EmotionActivation,
    secondary: list[EmotionActivation],
) -> str:
    """Describe concurrent affect labels without exposing their raw strengths."""

    if not secondary:
        return dominant.emotion_id
    labels = ", ".join(activation.emotion_id for activation in secondary)
    summary = f"{dominant.emotion_id} with {labels}"
    return summary


def _activation_descriptor(activation: float) -> str:
    """Project a normalized activation into a model-safe qualitative descriptor."""

    if activation >= 0.75:
        return "strong"
    if activation >= 0.4:
        return "present"
    return "fading"


def _judgment_note(dominant: EmotionActivation | None) -> str:
    """Describe the causal basis without presenting numeric internal state."""

    if dominant is None:
        return "no causal root crossed activation guard"
    source_refs = ", ".join(dominant.causal_source_refs)
    note = f"{dominant.emotion_id} derives from {source_refs}"
    return note
