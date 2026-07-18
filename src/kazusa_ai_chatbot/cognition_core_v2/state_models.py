"""Exact persistent state models and validation for cognition core V2."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, TypedDict

SCHEMA_VERSION = "cognition_state.v2"
CHARACTER_SCOPE = "character"
USER_SCOPE = "user"
CHARACTER_OWNER = "global"

EMOTION_IDS = (
    "joy",
    "fear",
    "anger",
    "sadness",
    "disgust",
    "surprise",
    "love_attachment",
    "compassion_empathy",
    "gratitude",
    "jealousy",
    "envy",
    "pride",
    "shame",
    "guilt",
    "embarrassment",
    "curiosity",
    "awe",
    "nostalgia",
    "loneliness",
    "relief",
    "ennui_existential_angst",
)


class CognitionStateError(ValueError):
    """Raised when native cognition state violates a reducer invariant."""

STATE_LIST_CAPS = {
    "goals": 16,
    "threats": 16,
    "active_events": 32,
    "knowledge_gaps": 16,
    "affect_activations": 32,
}

DRIVE_IDS = (
    "autonomy",
    "connection",
    "safety",
    "competence",
    "care",
    "integrity",
    "exploration",
    "meaning",
)

STANDARD_DEFAULTS = (
    ("honesty", "be truthful", 80),
    ("avoid_harm", "avoid causing needless harm", 85),
    ("respect_boundaries", "respect personal boundaries", 85),
    ("follow_through", "honor accepted commitments", 80),
    ("self_respect", "protect dignity and autonomy", 75),
)

GOAL_KINDS = (
    "ordinary_response",
    "relationship_connection",
    "bond_protection",
    "trust_verification",
    "autonomy_boundary",
    "safety",
    "obstruction_resolution",
    "loss_recovery",
    "moral_repair",
    "social_care",
    "reciprocity",
    "epistemic_exploration",
    "meaning_reconstruction",
    "self_improvement",
)

ENTITY_KINDS = (
    "relationship",
    "goal",
    "threat",
    "event",
    "knowledge_gap",
    "drive",
    "standard",
    "meaning",
)

ROLE_VALUES = (
    "actor",
    "experiencer",
    "target",
    "object",
    "affected_goal",
    "affected_relationship",
)

ROLE_ENTITY_KINDS = (
    "character",
    "user",
    "group",
    "third_party",
    "goal",
    "relationship",
    "standard",
    "object",
)

EVIDENCE_SOURCE_KINDS = (
    "episode",
    "action_result",
    "resolver_observation",
    "tool_result",
    "scheduler_event",
    "promoted_memory",
    "promoted_reflection",
    "media_observation",
)

COMMON_ENTITY_KEYS = {
    "entity_id",
    "description",
    "salience",
    "role_refs",
    "evidence_refs",
    "created_at",
    "updated_at",
}

ENTITY_LIST_KINDS = {
    "goals": "goal",
    "threats": "threat",
    "active_events": "event",
    "knowledge_gaps": "knowledge_gap",
}

ENTITY_LIST_FIELDS = {
    "goal": "goals",
    "threat": "threats",
    "event": "active_events",
    "knowledge_gap": "knowledge_gaps",
}

GOAL_FIELDS = {
    "goal_kind",
    "status",
    "importance",
    "progress",
    "obstruction",
    "expected_success",
    "controllability",
    "recoverability",
    "urgency",
}

THREAT_FIELDS = {
    "status",
    "likelihood",
    "expected_harm",
    "uncertainty",
    "controllability",
    "coping_potential",
    "residual_pressure",
}

EVENT_FIELDS = {
    "status",
    "outcome_impact",
    "responsibility",
    "intentionality",
    "harm",
    "unfairness",
    "exposure",
    "repair_need",
    "reparability",
    "expectation_mismatch",
    "norm_violation",
    "contamination_risk",
    "identity_threat",
    "comparison_gap",
    "vastness",
    "memory_warmth",
    "temporal_loss",
}

GAP_FIELDS = {
    "status",
    "relevance",
    "uncertainty",
    "learnability",
    "novelty",
    "model_accommodation",
}

ACTIVATION_KEYS = {
    "activation_id",
    "emotion_id",
    "primary_root",
    "root_refs",
    "phase",
    "score",
    "peak_score",
    "trend",
    "cause_status",
    "started_at",
    "updated_at",
    "last_reinforced_at",
}


class RelationshipStateV2(TypedDict):
    relationship_id: str
    other_user_id: str
    familiarity: int
    positive_regard: int
    trust: int
    attachment: int
    desired_closeness: int
    perceived_closeness: int
    care: int
    boundary_safety: int
    exclusivity: int
    unresolved_injury: int
    salience: int
    updated_at: str
    evidence_refs: list[dict[str, Any]]


class UserCognitionStateV2(TypedDict):
    schema_version: str
    state_scope: str
    owner_user_id: str
    updated_at: str
    relationship: RelationshipStateV2
    goals: list[dict[str, Any]]
    threats: list[dict[str, Any]]
    active_events: list[dict[str, Any]]
    knowledge_gaps: list[dict[str, Any]]
    affect_activations: list[dict[str, Any]]


class CharacterCognitionStateV2(TypedDict):
    schema_version: str
    state_scope: str
    updated_at: str
    drives: dict[str, dict[str, int]]
    standards: list[dict[str, Any]]
    meaning_state: dict[str, Any]
    goals: list[dict[str, Any]]
    threats: list[dict[str, Any]]
    active_events: list[dict[str, Any]]
    knowledge_gaps: list[dict[str, Any]]
    affect_activations: list[dict[str, Any]]


def storage_utc_now_iso() -> str:
    """Return a Mongo-safe UTC timestamp with a terminal Z."""

    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return timestamp.replace("+00:00", "Z")


def build_acquaintance_user_state(
    *,
    global_user_id: str,
    updated_at: str,
) -> UserCognitionStateV2:
    """Build the canonical state for a newly encountered user."""

    _require_nonempty_text(global_user_id, "global_user_id")
    _validate_timestamp(updated_at, "updated_at")
    return {
        "schema_version": SCHEMA_VERSION,
        "state_scope": USER_SCOPE,
        "owner_user_id": global_user_id,
        "updated_at": updated_at,
        "relationship": {
            "relationship_id": f"relationship:user:{global_user_id}",
            "other_user_id": global_user_id,
            "familiarity": 10,
            "positive_regard": 0,
            "trust": 0,
            "attachment": 0,
            "desired_closeness": 10,
            "perceived_closeness": 10,
            "care": 0,
            "boundary_safety": 0,
            "exclusivity": 0,
            "unresolved_injury": 0,
            "salience": 0,
            "updated_at": updated_at,
            "evidence_refs": [],
        },
        "goals": [],
        "threats": [],
        "active_events": [],
        "knowledge_gaps": [],
        "affect_activations": [],
    }


def build_character_production_state(
    *,
    updated_at: str,
) -> CharacterCognitionStateV2:
    """Build the canonical singleton state for the character."""

    _validate_timestamp(updated_at, "updated_at")
    return {
        "schema_version": SCHEMA_VERSION,
        "state_scope": CHARACTER_SCOPE,
        "updated_at": updated_at,
        "drives": {
            drive_id: {
                "importance": importance,
                "pressure": pressure,
            }
            for drive_id, importance, pressure in (
                ("autonomy", 75, 20),
                ("connection", 70, 20),
                ("safety", 65, 15),
                ("competence", 70, 20),
                ("care", 80, 20),
                ("integrity", 80, 15),
                ("exploration", 65, 20),
                ("meaning", 70, 15),
            )
        },
        "standards": [
            {
                "standard_id": standard_id,
                "description": description,
                "importance": importance,
            }
            for standard_id, description, importance in STANDARD_DEFAULTS
        ],
        "meaning_state": {
            "purpose_coherence": 70,
            "agency": 70,
            "identity_continuity": 80,
            "salience": 0,
        },
        "goals": [],
        "threats": [],
        "active_events": [],
        "knowledge_gaps": [],
        "affect_activations": [],
    }


def validate_cognition_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one user- or character-scoped state document."""

    if not isinstance(state, Mapping):
        raise CognitionStateError("state must be a mapping")
    payload = deepcopy(dict(state))
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "state_scope",
            "updated_at",
            "goals",
            "threats",
            "active_events",
            "knowledge_gaps",
            "affect_activations",
            "drives",
            "standards",
            "meaning_state",
        }
        if payload.get("state_scope") == CHARACTER_SCOPE
        else {
            "schema_version",
            "state_scope",
            "owner_user_id",
            "updated_at",
            "relationship",
            "goals",
            "threats",
            "active_events",
            "knowledge_gaps",
            "affect_activations",
        },
        "cognition state",
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CognitionStateError("unsupported cognition state schema")
    scope = payload.get("state_scope")
    if scope not in {USER_SCOPE, CHARACTER_SCOPE}:
        raise CognitionStateError("state_scope must be user or character")
    _require_nonempty_text(payload.get("updated_at"), "updated_at")
    _validate_timestamp(payload["updated_at"], "updated_at")
    if scope == USER_SCOPE:
        _validate_user_state(payload)
    else:
        _validate_character_state(payload)
    return payload


def validate_relationship_state(
    relationship: Mapping[str, Any],
    *,
    owner_user_id: str,
) -> RelationshipStateV2:
    """Validate and copy one native relationship state for its exact owner."""

    if not isinstance(relationship, Mapping):
        raise CognitionStateError("relationship must be a mapping")
    payload = deepcopy(dict(relationship))
    _require_nonempty_text(owner_user_id, "owner_user_id")
    _validate_relationship(payload, owner_user_id)
    return payload  # type: ignore[return-value]


def resolve_state_scope(
    caller: str,
    target_user_id: str | None = None,
    *,
    origin_scope: tuple[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve the frozen caller-to-scope matrix."""

    if caller == "resolver_recurrence":
        if origin_scope is None:
            raise CognitionStateError(
                "resolver recurrence requires origin_scope"
            )
        _validate_scope_tuple(origin_scope)
        return origin_scope

    user_required_callers = {
        "persona_user_message",
        "tool_result",
        "background_result",
        "group_sender",
    }
    user_optional_callers = {"self_cognition", "scheduled_tick"}
    character_callers = {
        "reflection",
        "internal_thought",
    }
    if caller in user_required_callers:
        if not target_user_id:
            raise CognitionStateError(
                f"{caller} requires a target user owner"
            )
        _require_nonempty_text(target_user_id, "target_user_id")
        return USER_SCOPE, target_user_id
    if caller in user_optional_callers and target_user_id:
        _require_nonempty_text(target_user_id, "target_user_id")
        return USER_SCOPE, target_user_id
    if caller in user_optional_callers or caller in character_callers:
        return CHARACTER_SCOPE, CHARACTER_OWNER
    raise CognitionStateError(f"unsupported state-scope caller: {caller!r}")


def prune_terminal_entities(state: Mapping[str, Any]) -> dict[str, Any]:
    """Prune oldest unreferenced terminal entities until every cap holds."""

    payload = deepcopy(dict(state))
    protected_refs = _protected_entity_refs(payload.get("affect_activations"))
    for field_name, cap in STATE_LIST_CAPS.items():
        entities = payload.get(field_name)
        if not isinstance(entities, list):
            raise CognitionStateError(f"{field_name} must be a list")
        if len(entities) <= cap or field_name == "affect_activations":
            continue
        kind = ENTITY_LIST_KINDS[field_name]
        removable = [
            entity
            for entity in entities
            if isinstance(entity, Mapping)
            and _is_terminal_entity(entity)
            and (kind, entity.get("entity_id")) not in protected_refs
        ]
        removable.sort(
            key=lambda entity: (
                entity["updated_at"],
                entity["entity_id"],
            )
        )
        remove_count = len(entities) - cap
        if len(removable) < remove_count:
            raise CognitionStateError(
                f"{field_name} capacity is protected by active causes"
            )
        removal_ids = {
            entity["entity_id"] for entity in removable[:remove_count]
        }
        payload[field_name] = [
            entity
            for entity in entities
            if entity["entity_id"] not in removal_ids
        ]
    return validate_cognition_state(payload)


def _validate_user_state(state: dict[str, Any]) -> None:
    """Validate the user-scoped state shape and relationship owner."""

    owner_user_id = state["owner_user_id"]
    _require_nonempty_text(owner_user_id, "owner_user_id")
    relationship = state["relationship"]
    if not isinstance(relationship, Mapping):
        raise CognitionStateError("relationship must be a mapping")
    _validate_relationship(relationship, owner_user_id)
    _validate_entity_lists(state)


def _validate_character_state(state: dict[str, Any]) -> None:
    """Validate the character singleton state and fixed defaults."""

    drives = state["drives"]
    if not isinstance(drives, Mapping) or set(drives) != set(DRIVE_IDS):
        raise CognitionStateError("character drives must match the registry")
    for drive_id in DRIVE_IDS:
        drive = drives[drive_id]
        _require_exact_keys(drive, {"importance", "pressure"}, drive_id)
        _validate_number(drive["importance"], 0, 100, drive_id)
        _validate_number(drive["pressure"], 0, 100, drive_id)

    standards = state["standards"]
    if not isinstance(standards, list) or len(standards) != 5:
        raise CognitionStateError("character standards must contain five items")
    expected_ids = [default[0] for default in STANDARD_DEFAULTS]
    actual_ids = []
    for standard in standards:
        if not isinstance(standard, Mapping):
            raise CognitionStateError("standard must be a mapping")
        _require_exact_keys(
            standard,
            {"standard_id", "description", "importance"},
            "standard",
        )
        _require_nonempty_text(standard["standard_id"], "standard_id")
        _require_nonempty_text(standard["description"], "description")
        _validate_number(standard["importance"], 0, 100, "standard.importance")
        actual_ids.append(standard["standard_id"])
    if actual_ids != expected_ids:
        raise CognitionStateError("character standards must match the registry")

    meaning = state["meaning_state"]
    if not isinstance(meaning, Mapping):
        raise CognitionStateError("meaning_state must be a mapping")
    meaning_keys = {
        "purpose_coherence",
        "agency",
        "identity_continuity",
        "salience",
    }
    if "low_coherence_since" in meaning:
        meaning_keys.add("low_coherence_since")
    _require_exact_keys(meaning, meaning_keys, "meaning_state")
    for field_name in (
        "purpose_coherence",
        "agency",
        "identity_continuity",
        "salience",
    ):
        _validate_number(meaning[field_name], 0, 100, field_name)
    if "low_coherence_since" in meaning:
        _validate_timestamp(meaning["low_coherence_since"], "low_coherence_since")
    _validate_entity_lists(state)


def _validate_relationship(
    relationship: Mapping[str, Any],
    owner_user_id: str,
) -> None:
    """Validate one relationship state and its structured evidence."""

    required = {
        "relationship_id",
        "other_user_id",
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
        "updated_at",
        "evidence_refs",
    }
    _require_exact_keys(relationship, required, "relationship")
    _require_nonempty_text(relationship["relationship_id"], "relationship_id")
    if relationship["other_user_id"] != owner_user_id:
        raise CognitionStateError("relationship owner does not match state owner")
    _require_nonempty_text(relationship["other_user_id"], "other_user_id")
    _validate_number(relationship["familiarity"], 0, 100, "familiarity")
    for field_name in ("positive_regard", "trust", "boundary_safety"):
        _validate_number(relationship[field_name], -100, 100, field_name)
    for field_name in (
        "attachment",
        "desired_closeness",
        "perceived_closeness",
        "care",
        "exclusivity",
        "unresolved_injury",
        "salience",
    ):
        _validate_number(relationship[field_name], 0, 100, field_name)
    _validate_timestamp(relationship["updated_at"], "relationship.updated_at")
    _validate_evidence_refs(relationship["evidence_refs"], "relationship")


def _validate_entity_lists(state: Mapping[str, Any]) -> None:
    """Validate exact entity fields, caps, ownership, and activation roots."""

    for field_name, cap in STATE_LIST_CAPS.items():
        entities = state[field_name]
        if not isinstance(entities, list) or len(entities) > cap:
            raise CognitionStateError(f"{field_name} exceeds its state cap")
        if field_name == "affect_activations":
            _validate_activations(entities, state)
            continue
        seen_ids: set[str] = set()
        for entity in entities:
            if not isinstance(entity, Mapping):
                raise CognitionStateError(f"{field_name} entity must be a mapping")
            entity_id = entity.get("entity_id")
            if entity_id in seen_ids:
                raise CognitionStateError(f"duplicate {field_name} entity id")
            seen_ids.add(entity_id)
            _validate_entity(entity, field_name)


def _validate_entity(entity: Mapping[str, Any], field_name: str) -> None:
    """Validate one exact causal entity record."""

    expected_fields = COMMON_ENTITY_KEYS | {
        "status",
        *{
            "goal": GOAL_FIELDS,
            "threat": THREAT_FIELDS,
            "event": EVENT_FIELDS,
            "knowledge_gap": GAP_FIELDS,
        }[ENTITY_LIST_KINDS[field_name]],
    }
    _require_exact_keys(entity, expected_fields, field_name)
    _require_nonempty_text(entity["entity_id"], f"{field_name}.entity_id")
    _require_nonempty_text(entity["description"], f"{field_name}.description")
    if len(entity["description"]) > 500:
        raise CognitionStateError(f"{field_name}.description exceeds 500 characters")
    _validate_number(entity["salience"], 0, 100, f"{field_name}.salience")
    _validate_role_refs(entity["role_refs"], field_name)
    _validate_evidence_refs(entity["evidence_refs"], field_name)
    _validate_timestamp(entity["created_at"], f"{field_name}.created_at")
    _validate_timestamp(entity["updated_at"], f"{field_name}.updated_at")
    _validate_status(entity, field_name)
    kind = ENTITY_LIST_KINDS[field_name]
    if kind == "goal":
        if entity["goal_kind"] not in GOAL_KINDS:
            raise CognitionStateError("invalid goal_kind")
        for field_name in GOAL_FIELDS - {"goal_kind", "status"}:
            _validate_number(entity[field_name], 0, 100, field_name)
    elif kind == "threat":
        for field_name in THREAT_FIELDS - {"status"}:
            _validate_number(entity[field_name], 0, 100, field_name)
    elif kind == "event":
        _validate_number(entity["outcome_impact"], -100, 100, "outcome_impact")
        for field_name in EVENT_FIELDS - {"status", "outcome_impact"}:
            _validate_number(entity[field_name], 0, 100, field_name)
    else:
        for field_name in GAP_FIELDS - {"status"}:
            _validate_number(entity[field_name], 0, 100, field_name)


def _validate_status(entity: Mapping[str, Any], field_name: str) -> None:
    """Validate the status vocabulary for one entity collection."""

    statuses = {
        "goals": {"pursuing", "blocked", "satisfied", "failed", "abandoned"},
        "threats": {"active", "resolved", "replaced"},
        "active_events": {"active", "resolved", "replaced"},
        "knowledge_gaps": {"open", "reduced", "resolved"},
    }
    if entity["status"] not in statuses[field_name]:
        raise CognitionStateError(f"invalid {field_name} status")


def _validate_activations(
    activations: list[Any],
    state: Mapping[str, Any],
) -> None:
    """Validate the derived activation cache and root existence."""

    seen_emotions: set[str] = set()
    for activation in activations:
        if not isinstance(activation, Mapping):
            raise CognitionStateError("affect activation must be a mapping")
        _require_exact_keys(activation, ACTIVATION_KEYS, "affect activation")
        emotion_id = activation["emotion_id"]
        if emotion_id not in EMOTION_IDS:
            raise CognitionStateError("unknown affect activation emotion")
        if emotion_id in seen_emotions:
            raise CognitionStateError("duplicate affect activation emotion")
        seen_emotions.add(emotion_id)
        if activation["activation_id"] != f"emotion:{emotion_id}":
            raise CognitionStateError("affect activation id is not canonical")
        root_refs = activation["root_refs"]
        if not isinstance(root_refs, list) or not 1 <= len(root_refs) <= 8:
            raise CognitionStateError("affect activation root_refs is invalid")
        for root_ref in root_refs:
            _validate_root_ref(root_ref, state)
        _validate_root_ref(activation["primary_root"], state)
        if activation["primary_root"] not in root_refs:
            raise CognitionStateError("primary root must be retained")
        if activation["phase"] not in {"active", "fading"}:
            raise CognitionStateError("invalid affect activation phase")
        if activation["trend"] not in {"rising", "stable", "falling"}:
            raise CognitionStateError("invalid affect activation trend")
        if activation["cause_status"] not in {
            "active",
            "resolved",
            "replaced",
        }:
            raise CognitionStateError("invalid affect activation cause status")
        _validate_number(activation["score"], 0, 100, "activation.score")
        _validate_number(activation["peak_score"], 0, 100, "activation.peak_score")
        if activation["peak_score"] < activation["score"]:
            raise CognitionStateError(
                "activation peak_score cannot be below score"
            )
        if activation["score"] <= 10:
            raise CognitionStateError(
                "inactive affect activation must be removed"
            )
        if activation["phase"] == "active" and (
            activation["score"] < 25
            or activation["cause_status"] != "active"
        ):
            raise CognitionStateError(
                "active affect activation does not satisfy lifecycle guards"
            )
        for field_name in (
            "started_at",
            "updated_at",
            "last_reinforced_at",
        ):
            _validate_timestamp(activation[field_name], field_name)


def _validate_root_ref(
    root_ref: Any,
    state: Mapping[str, Any] | None = None,
) -> None:
    """Validate one canonical entity reference and optionally its existence."""

    if not isinstance(root_ref, Mapping):
        raise CognitionStateError("activation root must be a mapping")
    _require_exact_keys(root_ref, {"scope", "kind", "entity_id"}, "root ref")
    if root_ref["scope"] not in {USER_SCOPE, CHARACTER_SCOPE}:
        raise CognitionStateError("activation root scope is invalid")
    if root_ref["kind"] not in ENTITY_KINDS:
        raise CognitionStateError("activation root kind is invalid")
    _require_nonempty_text(root_ref["entity_id"], "activation root entity_id")
    if state is not None:
        if root_ref["scope"] != state["state_scope"]:
            raise CognitionStateError("activation root scope is not mutable")
        if not _root_exists(state, root_ref):
            raise CognitionStateError("activation root does not exist")


def _root_exists(state: Mapping[str, Any], root_ref: Mapping[str, Any]) -> bool:
    """Return whether a root points to a current state object."""

    kind = root_ref["kind"]
    entity_id = root_ref["entity_id"]
    if kind in ENTITY_LIST_FIELDS:
        return any(
            entity.get("entity_id") == entity_id
            for entity in state[ENTITY_LIST_FIELDS[kind]]
        )
    if kind == "relationship":
        relationship = state.get("relationship")
        return (
            isinstance(relationship, Mapping)
            and relationship["relationship_id"] == entity_id
        )
    if kind == "meaning":
        return (
            entity_id == "meaning:character"
            and state["state_scope"] == CHARACTER_SCOPE
        )
    if kind == "drive":
        return entity_id in state.get("drives", {})
    if kind == "standard":
        return any(
            standard["standard_id"] == entity_id
            for standard in state.get("standards", [])
        )
    return False


def _validate_role_refs(value: Any, label: str) -> None:
    """Validate structured role references."""

    if not isinstance(value, list) or len(value) > 8:
        raise CognitionStateError(f"{label}.role_refs is invalid")
    for role_ref in value:
        if not isinstance(role_ref, Mapping):
            raise CognitionStateError(f"{label}.role_refs must be structured")
        _require_exact_keys(
            role_ref,
            {"role", "entity_kind", "entity_id"},
            "role ref",
        )
        if role_ref["role"] not in ROLE_VALUES:
            raise CognitionStateError("role ref role is invalid")
        if role_ref["entity_kind"] not in ROLE_ENTITY_KINDS:
            raise CognitionStateError("role ref entity_kind is invalid")
        _require_nonempty_text(role_ref["entity_id"], "role ref entity_id")


def _validate_evidence_refs(value: Any, label: str) -> None:
    """Validate complete evidence records without reducing provenance."""

    if not isinstance(value, list) or len(value) > 8:
        raise CognitionStateError(f"{label}.evidence_refs is invalid")
    for evidence_ref in value:
        if not isinstance(evidence_ref, Mapping):
            raise CognitionStateError(
                f"{label}.evidence_refs must be structured"
            )
        _require_exact_keys(
            evidence_ref,
            {"source_kind", "source_id", "occurred_at", "semantic_summary"},
            "evidence ref",
        )
        if evidence_ref["source_kind"] not in EVIDENCE_SOURCE_KINDS:
            raise CognitionStateError("evidence source_kind is invalid")
        _require_nonempty_text(evidence_ref["source_id"], "evidence source_id")
        _validate_timestamp(evidence_ref["occurred_at"], "evidence occurred_at")
        _require_nonempty_text(
            evidence_ref["semantic_summary"],
            "evidence semantic_summary",
        )
        if len(evidence_ref["semantic_summary"]) > 500:
            raise CognitionStateError("evidence semantic_summary is too long")


def _protected_entity_refs(activations: Any) -> set[tuple[str, str]]:
    """Collect entities referenced by active or fading activations."""

    protected: set[tuple[str, str]] = set()
    if not isinstance(activations, list):
        return protected
    for activation in activations:
        if not isinstance(activation, Mapping):
            continue
        if activation.get("phase") not in {"active", "fading"}:
            continue
        for root_ref in activation.get("root_refs", []):
            if isinstance(root_ref, Mapping):
                protected.add((root_ref.get("kind"), root_ref.get("entity_id")))
    return protected


def _is_terminal_entity(entity: Mapping[str, Any]) -> bool:
    """Return whether one causal entity has a terminal status."""

    return entity.get("status") in {
        "satisfied",
        "failed",
        "abandoned",
        "resolved",
        "replaced",
    }


def _require_exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    label: str,
) -> None:
    """Reject missing or extra fields at every persistent shape boundary."""

    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionStateError(f"{label} fields are not exact")


def _require_nonempty_text(value: Any, label: str) -> None:
    """Require a trimmed non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise CognitionStateError(f"{label} must be a non-empty string")


def _validate_number(value: Any, minimum: int, maximum: int, label: str) -> None:
    """Require an integer scalar inside an inclusive range."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise CognitionStateError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise CognitionStateError(f"{label} is outside its allowed range")


def _validate_timestamp(value: Any, label: str) -> None:
    """Require a UTC ISO-8601 timestamp ending in Z."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise CognitionStateError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CognitionStateError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CognitionStateError(f"{label} must include UTC")


def _validate_scope_tuple(scope: tuple[str, str]) -> None:
    """Validate a recurrence scope tuple."""

    if not isinstance(scope, tuple) or len(scope) != 2:
        raise CognitionStateError("origin_scope must be a scope tuple")
    state_scope, owner = scope
    if state_scope == USER_SCOPE:
        _require_nonempty_text(owner, "origin_scope owner")
    elif state_scope == CHARACTER_SCOPE and owner != CHARACTER_OWNER:
        raise CognitionStateError("character scope owner must be global")
    elif state_scope not in {USER_SCOPE, CHARACTER_SCOPE}:
        raise CognitionStateError("origin_scope scope is invalid")
