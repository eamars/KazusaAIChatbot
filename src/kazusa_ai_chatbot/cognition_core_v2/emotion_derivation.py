"""Deterministic typed emotion derivation and persistent activation lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)


FADE_MULTIPLIER = 0.4
BEGIN_THRESHOLD = 40
SUSTAIN_THRESHOLD = 25
INACTIVE_THRESHOLD = 10
REINFORCEMENT_DELTA = 10
_SELF_IDS = {"character:global", "self", "character"}


def derive_emotion_activation_v2(
    emotion_id: str,
    *,
    candidates: Sequence[Mapping[str, Any]],
    previous: Mapping[str, Any] | None,
    updated_at: str,
) -> dict[str, Any] | None:
    """Derive one thresholded persistent activation from typed candidates."""

    if emotion_id not in EMOTION_DEFINITIONS:
        raise ValueError("emotion id is not in the registry")
    _validate_timestamp(updated_at)
    normalized = [_normalize_candidate(candidate) for candidate in candidates]
    normalized.sort(
        key=lambda candidate: (
            -candidate["score"],
            -candidate["salience"],
            _root_sort_key(candidate["root_ref"]),
        )
    )
    passing = [candidate for candidate in normalized if candidate["score"] > INACTIVE_THRESHOLD]
    if not passing:
        if previous is None or previous["score"] <= INACTIVE_THRESHOLD:
            return None
        return _copy_fading_activation(previous, updated_at)

    primary = passing[0]
    active_candidates = [
        candidate for candidate in passing
        if candidate["cause_status"] == "active"
    ]
    previous_score = previous["score"] if previous is not None else 0
    score = primary["score"]
    if previous is None and score < BEGIN_THRESHOLD:
        return None
    if previous is not None and not active_candidates and primary["cause_status"] == "resolved":
        return _copy_fading_activation(previous, updated_at, score=score)
    phase = "active" if score >= SUSTAIN_THRESHOLD and active_candidates else "fading"
    if previous is None:
        trend = "rising"
    elif score >= previous_score + 4:
        trend = "rising"
    elif score <= previous_score - 4:
        trend = "falling"
    else:
        trend = "stable"
    cause_status = "active" if active_candidates else primary["cause_status"]
    last_reinforced = previous["last_reinforced_at"] if previous else updated_at
    if previous is not None and score >= previous_score + REINFORCEMENT_DELTA:
        last_reinforced = updated_at
    return {
        "activation_id": f"emotion:{emotion_id}",
        "emotion_id": emotion_id,
        "primary_root": primary["root_ref"],
        "root_refs": [candidate["root_ref"] for candidate in passing[:8]],
        "phase": phase,
        "score": score,
        "peak_score": max(score, previous["peak_score"] if previous else 0),
        "trend": trend,
        "cause_status": cause_status,
        "started_at": previous["started_at"] if previous else updated_at,
        "updated_at": updated_at,
        "last_reinforced_at": last_reinforced,
    }


def derive_persistent_emotion_activations(
    state: Mapping[str, Any],
    *,
    updated_at: str,
    character_constraints: Mapping[str, Any] | None = None,
    relationship_context: Mapping[str, Any] | None = None,
    transition_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Derive all twenty-one activations from typed mutable causes."""

    previous_rows = {
        row["emotion_id"]: row
        for row in state.get("affect_activations", [])
        if isinstance(row, Mapping) and "emotion_id" in row
    }
    effective_constraints = _effective_character_constraints(
        state,
        character_constraints,
    )
    activations: list[dict[str, Any]] = []
    for emotion_id in EMOTION_DEFINITIONS:
        candidates = _candidates_for_emotion(
            state,
            emotion_id,
            character_constraints=effective_constraints,
            relationship_context=relationship_context,
            transition_context=transition_context,
        )
        activation = derive_emotion_activation_v2(
            emotion_id,
            candidates=candidates,
            previous=previous_rows.get(emotion_id),
            updated_at=updated_at,
        )
        if activation is not None and activation["score"] > INACTIVE_THRESHOLD:
            activations.append(activation)
    return activations


def _candidates_for_emotion(
    state: Mapping[str, Any],
    emotion_id: str,
    *,
    character_constraints: Mapping[str, Any] | None,
    relationship_context: Mapping[str, Any] | None,
    transition_context: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build exact candidate rows without reading missing internal axes."""

    candidates: list[dict[str, Any]] = []
    for event in _mapping_list(state.get("active_events")):
        score = _event_score(
            state,
            event,
            emotion_id,
            character_constraints,
            transition_context,
        )
        if score is not None:
            candidates.append(_candidate(state, "event", event, score))
    for goal in _mapping_list(state.get("goals")):
        score = _goal_score(state, goal, emotion_id, character_constraints)
        if score is not None:
            candidates.append(_candidate(state, "goal", goal, score))
    for threat in _mapping_list(state.get("threats")):
        score = _threat_score(state, threat, emotion_id, relationship_context)
        if score is not None:
            candidates.append(_candidate(state, "threat", threat, score))
    for gap in _mapping_list(state.get("knowledge_gaps")):
        score = _gap_score(state, gap, emotion_id)
        if score is not None:
            candidates.append(_candidate(state, "knowledge_gap", gap, score))

    relationship = _relationship(state, relationship_context)
    if relationship is not None:
        score = _relationship_score(
            state,
            relationship,
            emotion_id,
            character_constraints,
        )
        if score is not None and state.get("state_scope") == "user":
            candidates.append({
                "root_ref": _root(state, "relationship", relationship["relationship_id"]),
                "score": score,
                "cause_status": "active",
                "salience": relationship["salience"],
            })
    meaning = state.get("meaning_state")
    if emotion_id == "ennui_existential_angst" and isinstance(meaning, Mapping):
        score = _meaning_score(state, meaning)
        if score is not None:
            candidates.append({
                "root_ref": _root(state, "meaning", "meaning:character"),
                "score": score,
                "cause_status": "active",
                "salience": meaning["salience"],
            })
    return candidates


def _event_score(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    emotion_id: str,
    character_constraints: Mapping[str, Any] | None,
    transition_context: Mapping[str, Any] | None,
) -> int | None:
    """Apply the event-based formula and role/evidence guards."""

    salience = event["salience"]
    outcome = event["outcome_impact"]
    harm = event["harm"]
    responsibility = event["responsibility"]
    if emotion_id == "joy" and outcome > 0:
        return _minimum(outcome, salience)
    if emotion_id == "anger":
        blocked_goals = [
            goal for goal in _mapping_list(state.get("goals"))
            if goal.get("status") == "blocked"
        ]
        if not blocked_goals:
            return None
        goal = max(blocked_goals, key=lambda row: row["importance"])
        relationship = _relationship(state, None)
        return _minimum(
            max(
                goal["importance"],
                event["harm"],
                relationship["unresolved_injury"] if relationship else 0,
            ),
            max(goal["obstruction"], event["unfairness"], event["intentionality"]),
            salience,
        )
    if emotion_id == "sadness" and outcome < 0:
        failed_goals = [
            goal for goal in _mapping_list(state.get("goals"))
            if goal.get("status") == "failed"
        ]
        if not failed_goals:
            return None
        goal = max(failed_goals, key=lambda row: row["importance"])
        relationship = _relationship(state, None)
        return _minimum(
            max(-outcome, 100 - goal["recoverability"]),
            max(
                goal["importance"],
                relationship["attachment"] if relationship else 0,
                -outcome,
            ),
            salience,
        )
    if emotion_id == "disgust" and max(event["contamination_risk"], event["norm_violation"]) >= 40:
        if not _has_role(event, {"target", "object"}):
            return None
        return _minimum(
            max(event["contamination_risk"], event["norm_violation"]),
            salience,
        )
    if emotion_id == "surprise" and event["expectation_mismatch"] >= 40:
        return _minimum(event["expectation_mismatch"], salience)
    if emotion_id == "compassion_empathy":
        if not _has_other_experiencer(event, state) or event["harm"] < 40:
            return None
        care = _constraint_axis(character_constraints, "drives", "care", "importance")
        return _minimum(care, event["harm"], salience)
    if emotion_id == "gratitude":
        if not _has_other_actor(event, state) or outcome <= 0:
            return None
        return _minimum(event["responsibility"], outcome, salience)
    if emotion_id == "envy":
        if not _has_other_actor(event, state) or not _has_role(event, {"object"}):
            return None
        if event["comparison_gap"] < 40:
            return None
        competence = _constraint_axis(character_constraints, "drives", "competence", "pressure")
        return _minimum(event["comparison_gap"], competence, salience)
    if emotion_id == "pride":
        if _has_self_actor(event, state) and outcome > 0:
            return _minimum(responsibility, outcome, salience)
        return None
    if emotion_id == "shame":
        if not _has_self_actor(event, state) or responsibility < 40:
            return None
        if event["norm_violation"] < 40 or event["identity_threat"] < 40:
            return None
        return _minimum(responsibility, event["norm_violation"], event["identity_threat"], salience)
    if emotion_id == "guilt":
        if not _has_self_actor(event, state) or responsibility < 40:
            return None
        harm_or_norm = max(event["harm"], event["norm_violation"])
        if harm_or_norm < 40 or event["repair_need"] < 40:
            return None
        return _minimum(responsibility, harm_or_norm, event["repair_need"], salience)
    if emotion_id == "embarrassment":
        if not _has_self_actor(event, state) or responsibility < 40:
            return None
        if event["exposure"] < 40 or event["expectation_mismatch"] < 40:
            return None
        if event["harm"] >= 40 or event["identity_threat"] >= 50:
            return None
        return _minimum(responsibility, event["exposure"], event["expectation_mismatch"], salience)
    if emotion_id == "awe":
        if event["vastness"] < 40:
            return None
        gap = _matching_gap(state, event)
        if gap is None:
            return None
        return _minimum(
            event["vastness"],
            max(gap["novelty"], gap["model_accommodation"]),
            salience,
        )
    if emotion_id == "nostalgia":
        if not _has_source(event, "promoted_memory") or not _has_cue(event):
            return None
        continuity = _constraint_axis(character_constraints, "meaning_state", "identity_continuity")
        return _minimum(event["memory_warmth"], event["temporal_loss"], continuity, salience)
    if emotion_id == "relief":
        return _transition_score(transition_context, salience)
    return None


def _goal_score(
    state: Mapping[str, Any],
    goal: Mapping[str, Any],
    emotion_id: str,
    character_constraints: Mapping[str, Any] | None,
) -> int | None:
    """Apply goal-based reward, obstruction, loss, and pride formulas."""

    if emotion_id == "joy" and goal["status"] == "satisfied":
        return _minimum(goal["progress"], goal["importance"], goal["salience"])
    if emotion_id == "pride" and goal["status"] == "satisfied" and _goal_is_self_owned(goal, state):
        return _minimum(goal["progress"], goal["importance"], goal["salience"])
    return None


def _threat_score(
    state: Mapping[str, Any],
    threat: Mapping[str, Any],
    emotion_id: str,
    relationship_context: Mapping[str, Any] | None,
) -> int | None:
    """Apply active threat, jealousy, and fear formulas."""

    if emotion_id == "fear" and threat["status"] == "active":
        return _minimum(
            threat["likelihood"],
            threat["expected_harm"],
            max(threat["uncertainty"], 100 - threat["coping_potential"]),
            threat["salience"],
        )
    if emotion_id == "jealousy" and threat["status"] == "active":
        relationship = _relationship(state, relationship_context)
        if relationship is None or not _has_role(threat, {"target", "object"}):
            return None
        if relationship["attachment"] < 40:
            return None
        if not _has_third_party(threat):
            return None
        return _minimum(
            relationship["attachment"],
            max(
                relationship["exclusivity"],
                relationship["desired_closeness"],
            ),
            threat["likelihood"],
            threat["salience"],
        )
    return None


def _gap_score(
    state: Mapping[str, Any],
    gap: Mapping[str, Any],
    emotion_id: str,
) -> int | None:
    """Apply the open/reduced knowledge-gap curiosity formula."""

    if emotion_id != "curiosity" or gap["status"] not in {"open", "reduced"}:
        return None
    return _minimum(
        gap["relevance"],
        gap["uncertainty"],
        gap["learnability"],
        max(gap["novelty"], gap["model_accommodation"]),
        gap["salience"],
    )


def _relationship_score(
    state: Mapping[str, Any],
    relationship: Mapping[str, Any],
    emotion_id: str,
    character_constraints: Mapping[str, Any] | None,
) -> int | None:
    """Apply love and loneliness formulas to relationship state."""

    if emotion_id == "love_attachment":
        if max(relationship["attachment"], relationship["care"]) < 40:
            return None
        return _minimum(
            max(relationship["attachment"], relationship["care"]),
            max(
                max(relationship["positive_regard"], 0),
                max(relationship["trust"], 0),
            ),
            relationship["salience"],
        )
    if emotion_id == "loneliness":
        pressure = _constraint_axis(character_constraints, "drives", "connection", "pressure")
        gap = max(
            relationship["desired_closeness"] - relationship["perceived_closeness"],
            0,
        )
        if pressure < 40 or gap <= 0:
            return None
        return _minimum(
            pressure,
            gap,
            max(
                relationship["attachment"],
                relationship["care"],
                relationship["salience"],
            ),
        )
    return None


def _meaning_score(
    state: Mapping[str, Any],
    meaning: Mapping[str, Any],
) -> int | None:
    """Apply the twenty-four-hour existential-angst guard."""

    low_since = meaning.get("low_coherence_since")
    updated_at = state["updated_at"]
    if not isinstance(low_since, str):
        return None
    if _parse_timestamp(updated_at) - _parse_timestamp(low_since) < 24 * 3600:
        return None
    drives = state["drives"]
    return _minimum(
        100 - meaning["purpose_coherence"],
        100 - meaning["agency"],
        drives["meaning"]["pressure"],
        meaning["salience"],
    )


def _candidate(
    state: Mapping[str, Any],
    kind: str,
    entity: Mapping[str, Any],
    score: int,
) -> dict[str, Any]:
    """Build one typed root candidate."""

    status = entity["status"]
    cause_status = (
        "active"
        if status in {"pursuing", "blocked", "active", "open", "reduced"}
        else "replaced"
        if status == "replaced"
        else "resolved"
    )
    return {
        "root_ref": _root(state, kind, entity["entity_id"]),
        "score": score,
        "cause_status": cause_status,
        "salience": entity["salience"],
    }


def _normalize_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact candidate record."""

    if set(candidate) != {"root_ref", "score", "cause_status", "salience"}:
        raise ValueError("emotion candidate fields are not exact")
    root_ref = candidate["root_ref"]
    if not isinstance(root_ref, Mapping) or set(root_ref) != {"scope", "kind", "entity_id"}:
        raise ValueError("emotion candidate root is incomplete")
    if root_ref["scope"] not in {"user", "character"} or root_ref["kind"] not in {
        "relationship",
        "goal",
        "threat",
        "event",
        "knowledge_gap",
        "drive",
        "standard",
        "meaning",
    }:
        raise ValueError("emotion candidate root is invalid")
    if not isinstance(root_ref["entity_id"], str) or not root_ref["entity_id"]:
        raise ValueError("emotion candidate root id is invalid")
    score = candidate["score"]
    salience = candidate["salience"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise ValueError("emotion candidate score is invalid")
    if isinstance(salience, bool) or not isinstance(salience, int) or not 0 <= salience <= 100:
        raise ValueError("emotion candidate salience is invalid")
    if candidate["cause_status"] not in {"active", "resolved", "replaced"}:
        raise ValueError("emotion candidate cause status is invalid")
    return {
        "root_ref": dict(root_ref),
        "score": score,
        "cause_status": candidate["cause_status"],
        "salience": salience,
    }


def _copy_fading_activation(
    previous: Mapping[str, Any],
    updated_at: str,
    *,
    score: int | None = None,
) -> dict[str, Any] | None:
    """Preserve one resolved activation until explicit elapsed decay removes it."""

    resolved_score = previous["score"] if score is None else score
    if resolved_score <= INACTIVE_THRESHOLD:
        return None
    copied = dict(previous)
    copied.update({
        "phase": "fading",
        "score": resolved_score,
        "trend": "stable" if resolved_score == previous["score"] else "falling",
        "cause_status": "resolved",
        "updated_at": updated_at,
    })
    return copied


def _root(state: Mapping[str, Any], kind: str, entity_id: str) -> dict[str, str]:
    """Build a scope-qualified canonical root."""

    return {
        "scope": state["state_scope"],
        "kind": kind,
        "entity_id": entity_id,
    }


def _relationship(
    state: Mapping[str, Any],
    relationship_context: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Read relationship only from its dedicated scope/context boundary."""

    relationship = state.get("relationship")
    if isinstance(relationship, Mapping):
        return relationship
    if isinstance(relationship_context, Mapping):
        return relationship_context
    return None


def _effective_character_constraints(
    state: Mapping[str, Any],
    character_constraints: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Use a dedicated constraint projection without merging it into state."""

    if character_constraints is not None:
        return character_constraints
    if state["state_scope"] != "character":
        return None
    return {
        "drives": state["drives"],
        "meaning_state": state["meaning_state"],
    }


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    """Return mapping rows from one persistent state list."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _has_role(entity: Mapping[str, Any], roles: set[str]) -> bool:
    """Return whether a typed role reference has one of the requested roles."""

    return any(
        isinstance(role, Mapping) and role["role"] in roles
        for role in entity["role_refs"]
    )


def _has_self_actor(entity: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Return whether actor ownership belongs to the current character."""

    owner = state.get("owner_user_id")
    return any(
        isinstance(role, Mapping)
        and role["role"] == "actor"
        and role["entity_kind"] == "character"
        and role["entity_id"] in _SELF_IDS
        for role in entity["role_refs"]
    ) or any(
        isinstance(role, Mapping)
        and role["role"] == "actor"
        and role["entity_kind"] == "user"
        and role["entity_id"] == owner
        for role in entity["role_refs"]
    )


def _has_other_actor(entity: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Return whether a typed actor is not the current self actor."""

    owner = state.get("owner_user_id")
    return any(
        isinstance(role, Mapping)
        and role["role"] == "actor"
        and not (
            role["entity_kind"] == "character"
            and role["entity_id"] in _SELF_IDS
        )
        and not (
            role["entity_kind"] == "user"
            and role["entity_id"] == owner
        )
        for role in entity["role_refs"]
    )


def _has_other_experiencer(entity: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Return whether another entity is the typed emotional experiencer."""

    owner = state.get("owner_user_id")
    return any(
        isinstance(role, Mapping)
        and role["role"] == "experiencer"
        and not (
            role["entity_kind"] == "character"
            and role["entity_id"] in _SELF_IDS
        )
        and not (
            role["entity_kind"] == "user"
            and role["entity_id"] == owner
        )
        for role in entity["role_refs"]
    )


def _has_third_party(entity: Mapping[str, Any]) -> bool:
    """Return whether a threat carries a canonical third-party role."""

    return any(
        isinstance(role, Mapping) and role["entity_kind"] == "third_party"
        for role in entity["role_refs"]
    )


def _goal_is_self_owned(goal: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Return whether the goal names the current self as actor."""

    return _has_self_actor(goal, state)


def _matching_gap(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Find a gap attached to the event through an affected-goal role."""

    ids = {
        role["entity_id"]
        for role in event["role_refs"]
        if isinstance(role, Mapping) and role["role"] == "affected_goal"
    }
    for gap in _mapping_list(state.get("knowledge_gaps")):
        if gap["entity_id"] in ids:
            return gap
    return None


def _has_source(event: Mapping[str, Any], source_kind: str) -> bool:
    """Return whether an event has a typed evidence source."""

    return any(
        isinstance(ref, Mapping) and ref["source_kind"] == source_kind
        for ref in event["evidence_refs"]
    )


def _has_cue(event: Mapping[str, Any]) -> bool:
    """Require a current episode/media cue beside promoted memory."""

    return any(
        isinstance(ref, Mapping)
        and ref["source_kind"] in {"episode", "media_observation"}
        for ref in event["evidence_refs"]
    )


def _constraint_axis(
    constraints: Mapping[str, Any] | None,
    container: str,
    key: str,
    field: str | None = None,
) -> int:
    """Read a required cross-scope constraint axis."""

    if not isinstance(constraints, Mapping):
        return 0
    value = constraints[container]
    if field is not None:
        value = value[key][field]
    else:
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("cross-scope constraint axis must be an integer")
    return value


def _transition_score_evidence_marker(value: Any) -> bool:
    """Keep transition evidence checking typed and local."""

    return isinstance(value, Mapping) and value.get("source_kind") in {
        "action_result",
        "resolver_observation",
        "accepted_task_result",
        "scheduler_event",
    }


def _has_typed_evidence(context: Mapping[str, Any]) -> bool:
    """Require a typed evidence record on a transition context."""

    return _transition_score_evidence_marker(context.get("evidence_ref"))


def _transition_score(
    transition_context: Mapping[str, Any] | None,
    salience: int,
) -> int | None:
    """Derive relief from an explicit prior/current transition."""

    if not isinstance(transition_context, Mapping):
        return None
    prior = transition_context["prior"]
    current = transition_context["current"]
    if prior["status"] != "active" or current["status"] != "resolved":
        return None
    if not _has_typed_evidence(transition_context):
        return None
    reduction = prior["residual_pressure"] - current["residual_pressure"]
    return _minimum(prior["residual_pressure"], reduction, salience)


def _validate_timestamp(value: str) -> None:
    """Validate one UTC timestamp used by an activation row."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("activation timestamp must be UTC")
    _parse_timestamp(value)


def _parse_timestamp(value: str) -> float:
    """Return a UTC timestamp as epoch seconds."""

    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("activation timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("activation timestamp must include UTC")
    return parsed.astimezone(timezone.utc).timestamp()


def _root_sort_key(root_ref: Mapping[str, Any]) -> str:
    """Return the lexical root ordering key."""

    return ":".join(
        str(root_ref[field_name])
        for field_name in ("scope", "kind", "entity_id")
    )


def _minimum(*values: int) -> int:
    """Return a clamped minimum score."""

    return max(0, min(100, *values))
