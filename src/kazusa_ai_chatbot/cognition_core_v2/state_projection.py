"""Semantic projection from native V2 state into bounded model context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


RAW_STATE_KEYS = frozenset({
    "entity_id",
    "owner_user_id",
    "created_at",
    "updated_at",
    "started_at",
    "last_reinforced_at",
    "primary_root",
    "root_refs",
    "evidence_refs",
    "state_scope",
    "schema_version",
    "scope",
    "kind",
})


@dataclass(frozen=True)
class PromptProjectionV2:
    """Hold prompt-safe values and private handle bindings separately."""

    payload: dict[str, Any]
    handle_to_ref: dict[str, dict[str, str]]


def project_numeric_band(value: int, *, signed: bool = False) -> str:
    """Translate a bounded scalar into the frozen semantic band vocabulary."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("projection value must be an integer")
    if signed:
        if not -100 <= value <= 100:
            raise ValueError("signed projection value is out of range")
        if value <= -61:
            return "strongly negative"
        if value <= -21:
            return "negative"
        if value <= 20:
            return "neutral or mixed"
        if value <= 60:
            return "positive"
        return "strongly positive"
    if not 0 <= value <= 100:
        raise ValueError("unsigned projection value is out of range")
    if value == 0:
        return "none"
    if value <= 20:
        return "very low"
    if value <= 40:
        return "low"
    if value <= 60:
        return "moderate"
    if value <= 80:
        return "high"
    return "very high"


def project_duration(started_at: str, now: str) -> str:
    """Translate elapsed UTC time into the frozen semantic duration labels."""

    elapsed = _parse_utc(now) - _parse_utc(started_at)
    seconds = max(0.0, elapsed.total_seconds())
    if seconds < 10 * 60:
        return "immediate"
    if seconds < 2 * 3600:
        return "recent"
    if seconds < 24 * 3600:
        return "earlier"
    if seconds < 7 * 24 * 3600:
        return "within recent days"
    return "older"


def project_relationship_context(
    relationship: Mapping[str, Any],
) -> dict[str, Any]:
    """Project native relationship axes into qualitative prompt context."""

    axes: dict[str, str] = {}
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
    ):
        value = relationship.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool):
            axes[field_name] = project_numeric_band(
                value,
                signed=field_name in {
                    "positive_regard",
                    "trust",
                    "boundary_safety",
                },
            )
    return {
        "relationship_summary": "native V2 relationship context",
        "axes": axes,
    }


def project_trend(previous: int, current: int) -> str:
    """Return direction using the fixed four-point change rule."""

    difference = current - previous
    if difference >= 4:
        return "rising"
    if difference <= -4:
        return "falling"
    return "stable"


def project_state_for_prompt(
    state: Mapping[str, Any],
    *,
    character_constraints: Mapping[str, Any],
    relationship_context: Mapping[str, Any] | None = None,
    evidence: Sequence[Mapping[str, Any]] = (),
) -> PromptProjectionV2:
    """Project all prompt-visible state into semantic descriptors.

    Persistent ids and raw scalar values stay in ``handle_to_ref`` for
    deterministic mapping and are never included in ``payload``.
    """

    handle_to_ref: dict[str, dict[str, str]] = {}
    payload: dict[str, Any] = {
        "goals": [],
        "threats": [],
        "events": [],
        "knowledge_gaps": [],
        "affect": [],
        "causal_candidates": [],
        "evidence": [],
        "roles": {
            "self": "the active character",
            "current_user": "the current conversation participant",
        },
        "character_constraints": _project_constraints(character_constraints),
    }
    for field_name, prompt_name, prefix in (
        ("goals", "goals", "g"),
        ("threats", "threats", "t"),
        ("active_events", "events", "e"),
        ("knowledge_gaps", "knowledge_gaps", "k"),
    ):
        for index, entity in enumerate(state[field_name], start=1):
            handle = f"{prefix}{index}"
            handle_to_ref[handle] = {
                "scope": state["state_scope"],
                "kind": _kind_for_field(field_name),
                "entity_id": entity["entity_id"],
            }
            payload[prompt_name].append(
                _project_entity(handle, entity, state["updated_at"])
            )
    relationship = state.get("relationship")
    if relationship is None:
        relationship = relationship_context
    if isinstance(relationship, Mapping):
        handle_to_ref["r1"] = {
            "scope": "user",
            "kind": "relationship",
            "entity_id": relationship["relationship_id"],
        }
        payload["relationship"] = _project_relationship(relationship)
    for activation in state["affect_activations"]:
        payload["affect"].append(_project_activation(activation, state))
    for index, drive_id in enumerate(state.get("drives", {}), start=1):
        handle_to_ref[f"d{index}"] = {
            "scope": state["state_scope"],
            "kind": "drive",
            "entity_id": drive_id,
        }
    for index, standard in enumerate(state.get("standards", []), start=1):
        handle_to_ref[f"s{index}"] = {
            "scope": state["state_scope"],
            "kind": "standard",
            "entity_id": standard["standard_id"],
        }
    if isinstance(state.get("meaning_state"), Mapping):
        handle_to_ref["m1"] = {
            "scope": state["state_scope"],
            "kind": "meaning",
            "entity_id": "meaning:character",
        }
    owner_user_id = state.get("owner_user_id")
    handle_to_ref["self"] = {
        "scope": "character",
        "kind": "meaning",
        "entity_id": "meaning:character",
    }
    if isinstance(owner_user_id, str) and owner_user_id:
        handle_to_ref["current_user"] = {
            "scope": "user",
            "kind": "relationship",
            "entity_id": f"relationship:user:{owner_user_id}",
        }
    for index, row in enumerate(evidence, start=1):
        evidence_handle = row.get("evidence_handle")
        if not isinstance(evidence_handle, str):
            continue
        evidence_ref = row.get("evidence_ref")
        if isinstance(evidence_ref, Mapping):
            payload["evidence"].append({
                "handle": evidence_handle,
                "source_kind": evidence_ref.get("source_kind", "unknown"),
                "semantic_summary": row.get(
                    "semantic_text",
                    evidence_ref.get("semantic_summary", ""),
                ),
            })
        for kind, prefix, description in (
            ("event", "ce", "the current episode event"),
            ("threat", "ct", "the possible current threat"),
            ("knowledge_gap", "ck", "the possible current knowledge gap"),
        ):
            handle = f"{prefix}{index}"
            handle_to_ref[handle] = {
                "scope": state["state_scope"],
                "kind": kind,
                "entity_id": f"candidate:{kind}:{evidence_handle}",
            }
            payload["causal_candidates"].append({
                "handle": handle,
                "candidate_kind": kind,
                "evidence_handle": evidence_handle,
                "description": description,
                "lifecycle": "candidate pending grounded appraisal",
            })
    validate_prompt_projection(payload)
    return PromptProjectionV2(payload=payload, handle_to_ref=handle_to_ref)


def validate_prompt_projection(payload: Mapping[str, Any]) -> None:
    """Reject raw state fields or private sentinel values in model payloads."""

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key in RAW_STATE_KEYS:
                    raise ValueError(f"raw state key leaked into prompt: {key}")
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)


def _project_entity(
    handle: str,
    entity: Mapping[str, Any],
    now: str,
) -> dict[str, Any]:
    """Project one causal entity without ids, timestamps, or raw axes."""

    result: dict[str, Any] = {
        "handle": handle,
        "description": entity["description"],
        "lifecycle": _lifecycle_label(entity["status"]),
        "salience": project_numeric_band(entity["salience"]),
        "duration": project_duration(entity["created_at"], now),
        "causal_roles": _project_roles(entity.get("role_refs", [])),
    }
    for field_name, signed in (
        ("importance", False),
        ("progress", False),
        ("obstruction", False),
        ("urgency", False),
        ("residual_pressure", False),
        ("harm", False),
        ("responsibility", False),
        ("uncertainty", False),
        ("relevance", False),
        ("trust", False),
        ("attachment", False),
        ("positive_regard", True),
    ):
        if field_name in entity:
            result[field_name] = project_numeric_band(
                entity[field_name],
                signed=signed,
            )
    return result


def _project_roles(value: Any) -> list[str]:
    """Project structured causal roles into semantic relationship phrases."""

    if not isinstance(value, list):
        return []
    labels: list[str] = []
    for role in value:
        if not isinstance(role, Mapping):
            continue
        role_name = role.get("role")
        if isinstance(role_name, str) and role_name.strip():
            labels.append(f"{role_name.strip()} role is causally relevant")
    return labels


def _project_relationship(relationship: Mapping[str, Any]) -> dict[str, Any]:
    """Project relationship axes into semantic labels."""
    return {
        "handle": "r1",
        **project_relationship_context(relationship),
    }


def _project_constraints(constraints: Mapping[str, Any]) -> dict[str, Any]:
    """Project character constraints separately from mutable user state."""

    drives = {
        drive_id: {
            "importance": project_numeric_band(row["importance"]),
            "pressure": project_numeric_band(row["pressure"]),
        }
        for drive_id, row in constraints["drives"].items()
    }
    standards = [
        {
            "description": row["description"],
            "importance": project_numeric_band(row["importance"]),
        }
        for row in constraints["standards"]
    ]
    meaning = {
        field_name: project_numeric_band(constraints["meaning_state"][field_name])
        for field_name in (
            "purpose_coherence",
            "agency",
            "identity_continuity",
            "salience",
        )
    }
    return {"drives": drives, "standards": standards, "meaning_state": meaning}


def _project_activation(
    activation: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, str]:
    """Project activation lifecycle controls into natural language."""

    return {
        "emotion": activation["emotion_id"],
        "phase": (
            "the cause is still active"
            if activation["cause_status"] == "active"
            else "the feeling is fading after resolution"
        ),
        "intensity": project_numeric_band(activation["score"]),
        "trend": activation["trend"],
        "cause_summary": _activation_cause_summary(activation, state),
    }


def _activation_cause_summary(
    activation: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    """Describe the actual primary cause without exposing its identifier."""

    root = activation.get("primary_root")
    if not isinstance(root, Mapping):
        return "a grounded causal source remains in context"
    fields = {
        "goal": "goals",
        "threat": "threats",
        "event": "active_events",
        "knowledge_gap": "knowledge_gaps",
    }
    field_name = fields.get(root.get("kind"))
    if field_name is not None:
        for entity in state.get(field_name, []):
            if (
                isinstance(entity, Mapping)
                and entity.get("entity_id") == root.get("entity_id")
            ):
                description = entity.get("description")
                if isinstance(description, str) and description.strip():
                    return description[:500]
    if root.get("kind") == "relationship":
        return "the current relationship carries the activating social pressure"
    if root.get("kind") == "meaning":
        return "purpose and agency remain persistently low"
    return "a grounded causal source remains in context"


def _kind_for_field(field_name: str) -> str:
    """Return the canonical singular entity kind."""

    return {
        "goals": "goal",
        "threats": "threat",
        "active_events": "event",
        "knowledge_gaps": "knowledge_gap",
    }[field_name]


def _lifecycle_label(status: str) -> str:
    """Translate deterministic status into a model-facing descriptor."""

    return {
        "pursuing": "in progress",
        "blocked": "blocked and needs resolution",
        "satisfied": "completed",
        "failed": "failed and needs recovery",
        "abandoned": "released",
        "active": "active and unresolved",
        "resolved": "resolved",
        "replaced": "superseded",
        "open": "open and uncertain",
        "reduced": "partly reduced but uncertain",
    }.get(status, status)


def _parse_utc(value: str) -> datetime:
    """Parse a required UTC Z timestamp."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("projection timestamp must end in Z")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None:
        raise ValueError("projection timestamp must be timezone aware")
    return parsed.astimezone(timezone.utc)
