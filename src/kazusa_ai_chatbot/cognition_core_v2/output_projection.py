"""Semantic output projections owned by the V2 cognition boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ExpressionPolicyV2,
    SemanticAffectProjectionV2,
    SemanticRelationshipProjectionV2,
    StateUpdateV2,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_numeric_band,
)


def project_affect(
    activations: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any] | None = None,
) -> list[SemanticAffectProjectionV2]:
    """Project persistent activations without exposing scores or identifiers."""

    result: list[SemanticAffectProjectionV2] = []
    for activation in activations:
        result.append(
            {
                "emotion": activation["emotion_id"],
                "phase": _phase_label(activation),
                "intensity": project_numeric_band(activation["score"]),
                "trend": activation["trend"],
                "cause_summary": _cause_summary(activation, state),
            }
        )
    return result


def project_relationship(
    relationship: Mapping[str, Any] | None,
) -> SemanticRelationshipProjectionV2 | None:
    """Project relationship state into bounded semantic axis summaries."""

    if relationship is None:
        return None
    axes = {
        field_name: project_numeric_band(
            relationship[field_name],
            signed=field_name in {"positive_regard", "trust", "boundary_safety"},
        )
        for field_name in (
            "familiarity",
            "positive_regard",
            "trust",
            "attachment",
            "desired_closeness",
            "perceived_closeness",
            "care",
            "boundary_safety",
            "exclusivity",
            "unresolved_injury",
            "salience",
        )
        if field_name in relationship
    }
    return {
        "relationship_summary": "relationship context remains bounded and causal",
        "axis_summaries": axes,
    }


def build_state_update(
    previous_state: Mapping[str, Any],
    replacement_state: Mapping[str, Any],
    comparison_results: Sequence[Mapping[str, Any]] = (),
) -> StateUpdateV2:
    """Build the sole persistence envelope from a complete replacement state."""

    changed_paths = _changed_paths(previous_state, replacement_state)
    owner_key = (
        replacement_state["owner_user_id"]
        if replacement_state["state_scope"] == "user"
        else "global"
    )
    normalized_comparisons = [dict(row) for row in comparison_results]
    normalized_comparisons.sort(
        key=lambda row: (
            str(row.get("entity_kind", "")),
            str(row.get("entity_id", "")),
            str(row.get("outcome", "")),
        )
    )
    return {
        "state_scope": replacement_state["state_scope"],
        "owner_key": owner_key,
        "replacement_state": dict(replacement_state),
        "comparison_results": normalized_comparisons,
        "changed_paths": changed_paths,
    }


def default_expression_policy(
    route: str,
    affect: Sequence[Mapping[str, Any]],
) -> ExpressionPolicyV2:
    """Derive deterministic visible-expression limits from route and affect."""

    if route in {"silence", "deferral"}:
        visibility = "none" if route == "silence" else "private"
    else:
        visibility = "visible"
    intensity = "restrained"
    if any(row.get("intensity") in {"high", "very high"} for row in affect):
        intensity = "strong"
    elif affect:
        intensity = "moderate"
    return {
        "visibility": visibility,
        "emotional_tone": affect[0]["emotion"] if affect else "neutral",
        "intensity": intensity,
        "directness": "balanced",
    }


def _phase_label(activation: Mapping[str, Any]) -> str:
    """Translate cause lifecycle into a semantic phase."""

    if activation.get("cause_status") == "resolved":
        return "fading after resolution"
    if activation.get("phase") == "fading":
        return "fading"
    return "currently active"


def _cause_summary(
    activation: Mapping[str, Any],
    state: Mapping[str, Any] | None,
) -> str:
    """Project the actual primary causal reason without private identifiers."""

    if isinstance(state, Mapping):
        root = activation.get("primary_root")
        if isinstance(root, Mapping):
            kind = root.get("kind")
            entity_id = root.get("entity_id")
            field_name = {
                "goal": "goals",
                "threat": "threats",
                "event": "active_events",
                "knowledge_gap": "knowledge_gaps",
            }.get(kind)
            if field_name is not None:
                for entity in state.get(field_name, []):
                    if isinstance(entity, Mapping) and entity.get("entity_id") == entity_id:
                        description = entity.get("description")
                        if isinstance(description, str) and description.strip():
                            return description[:500]
            if kind == "relationship" and isinstance(state.get("relationship"), Mapping):
                return "the current relationship carries the activating social pressure"
            if kind == "meaning" and isinstance(state.get("meaning_state"), Mapping):
                return "purpose and agency remain persistently low"
    return "the retained primary cause remains grounded in the current episode"


def _changed_paths(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> list[str]:
    """Return stable nested paths for causal and activation changes."""

    paths: list[str] = []

    def visit(before: Any, after: Any, path: str) -> None:
        if path.rsplit(".", 1)[-1] in {
            "updated_at",
            "created_at",
            "last_reinforced_at",
        }:
            return
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            for key in sorted(set(before) | set(after)):
                visit(before.get(key), after.get(key), f"{path}.{key}" if path else str(key))
            return
        if isinstance(before, list) and isinstance(after, list):
            before_by_id = {
                item.get("entity_id", item.get("emotion_id")): item
                for item in before
                if isinstance(item, Mapping)
            }
            after_by_id = {
                item.get("entity_id", item.get("emotion_id")): item
                for item in after
                if isinstance(item, Mapping)
            }
            if before_by_id or after_by_id:
                for item_id in sorted(set(before_by_id) | set(after_by_id), key=str):
                    visit(
                        before_by_id.get(item_id),
                        after_by_id.get(item_id),
                        f"{path}[{item_id}]",
                    )
                return
        if before != after:
            paths.append(path)

    visit(previous, current, "")
    return sorted(set(path.lstrip(".")) for path in paths if path)
