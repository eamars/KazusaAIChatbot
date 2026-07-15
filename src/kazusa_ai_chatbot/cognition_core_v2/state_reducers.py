"""Deterministic translation of validated semantic inputs into V2 state."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import floor
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionEvidenceV2,
    SemanticAppraisalResultV2,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
    derive_persistent_emotion_activations,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    ENTITY_LIST_FIELDS,
    GOAL_KINDS,
    ROLE_ENTITY_KINDS,
    prune_terminal_entities,
)
from kazusa_ai_chatbot.cognition_core_v2.transition_guards import (
    apply_direct_fact,
    apply_semantic_deltas,
    compare_event,
    transition_event,
    transition_goal,
    transition_knowledge_gap,
    transition_threat,
)


_ENTITY_FIELDS = ("goals", "threats", "active_events", "knowledge_gaps")


def apply_semantic_appraisals(
    state: Mapping[str, Any],
    results: Sequence[SemanticAppraisalResultV2],
    evidence: Sequence[CognitionEvidenceV2],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    comparison_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map prompt handles to native paths before the final deterministic reduce."""

    evidence_by_handle = {
        row["evidence_handle"]: row["evidence_ref"]
        for row in evidence
    }
    updated = deepcopy(dict(state))
    local_handle_to_ref = deepcopy(dict(handle_to_ref))
    translated: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    new_causal_ids: set[str] = set()
    for result in results:
        for proposition in result["propositions"]:
            _materialize_proposition_root(
                updated,
                proposition,
                evidence_by_handle,
                local_handle_to_ref,
                result["selected_evidence_handles"],
                comparison_results,
                new_causal_ids,
            )
        for delta in result["deltas"]:
            native_path = _native_delta_path(
                delta["target_path"],
                local_handle_to_ref,
            )
            try:
                _target_for_prompt_path(
                    updated,
                    delta["target_path"].split("."),
                    local_handle_to_ref,
                )
            except CognitionStateError:
                continue
            native_handles = [
                evidence_by_handle[handle]
                for handle in delta["evidence_handles"]
                if handle in evidence_by_handle
            ]
            if len(native_handles) != len(delta["evidence_handles"]):
                continue
            proposal = dict(delta)
            proposal["target_path"] = native_path
            proposal["evidence_handles"] = [
                ref["source_id"] for ref in native_handles
            ]
            if native_path.startswith(("drives.", "meaning_state.")):
                unsupported.append(proposal)
            else:
                translated.append(proposal)
    _retain_prompt_evidence(
        updated,
        results,
        evidence_by_handle,
        local_handle_to_ref,
    )
    updated = apply_semantic_deltas(updated, translated)
    _recompute_new_causal_salience(
        updated,
        translated,
        new_causal_ids,
    )
    for proposal in unsupported:
        _apply_unretained_character_delta(updated, proposal)
    return updated


def _materialize_proposition_root(
    state: dict[str, Any],
    proposition: Mapping[str, Any],
    evidence_by_handle: Mapping[str, Mapping[str, Any]],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    selected_evidence_handles: Sequence[str],
    comparison_results: list[dict[str, Any]] | None,
    new_causal_ids: set[str],
) -> None:
    """Turn a validated prompt-local proposition into a causal state root."""

    subject_handle = proposition["subject_handle"]
    subject_ref = handle_to_ref.get(subject_handle)
    if subject_ref is None:
        raise CognitionStateError("proposition subject handle is unknown")
    subject_kind = subject_ref["kind"]
    if subject_kind not in {
        "goal",
        "event",
        "threat",
        "knowledge_gap",
    }:
        return
    evidence_handles = proposition["evidence_handles"] or list(
        selected_evidence_handles
    )
    if not evidence_handles:
        raise CognitionStateError("causal proposition requires evidence")
    evidence_ref = evidence_by_handle.get(evidence_handles[0])
    if evidence_ref is None:
        raise CognitionStateError("causal proposition evidence is unknown")
    subject_id = subject_ref["entity_id"]
    candidate_subject = subject_id.startswith("candidate:")
    root_id = (
        _causal_candidate_id(state, subject_kind, evidence_ref)
        if candidate_subject
        else subject_id
    )
    current_event_ref = {
        "scope": state["state_scope"],
        "kind": subject_kind,
        "entity_id": root_id,
    }
    field_name = ENTITY_LIST_FIELDS[subject_kind]
    entities = state[field_name]
    existing = next(
        (
            entity for entity in entities
            if entity.get("entity_id") == root_id
        ),
        None,
    )
    if existing is None and candidate_subject and subject_kind == "event":
        incoming_roles = _role_refs_from_proposition(proposition, handle_to_ref)
        incoming = _new_causal_candidate(
            state,
            subject_kind,
            root_id,
            proposition["semantic_value"],
            [evidence_ref],
        )
        incoming["role_refs"] = incoming_roles
        existing = _matching_event(state, incoming)
    if existing is None:
        existing = _new_causal_candidate(
            state,
            subject_kind,
            root_id,
            proposition["semantic_value"],
            [
                evidence_by_handle[handle]
                for handle in evidence_handles
                if handle in evidence_by_handle
            ],
        )
        existing["role_refs"] = _role_refs_from_proposition(
            proposition,
            handle_to_ref,
        )
        entities.append(existing)
        new_causal_ids.add(existing["entity_id"])
        outcome = "create"
    else:
        outcome = _apply_proposition_transition(
            existing,
            subject_kind,
            proposition["proposition_kind"],
            evidence_ref,
        )
        if outcome == "reinforce":
            existing["description"] = proposition["semantic_value"]
            existing["role_refs"] = _role_refs_from_proposition(
                proposition,
                handle_to_ref,
            ) or existing.get("role_refs", [])
        _append_evidence_rows(
            existing,
            [
                evidence_by_handle[handle]
                for handle in evidence_handles
                if handle in evidence_by_handle
            ],
        )
    subject_ref["entity_id"] = existing["entity_id"]
    if comparison_results is not None:
        comparison_result: dict[str, Any] = {
            "current_event_ref": current_event_ref,
            "outcome": outcome,
            "evidence_refs": [
                deepcopy(dict(evidence_by_handle[handle]))
                for handle in evidence_handles
                if handle in evidence_by_handle
            ],
        }
        if outcome != "create":
            comparison_result["matched_entity_ref"] = {
                "scope": state["state_scope"],
                "kind": subject_kind,
                "entity_id": existing["entity_id"],
            }
        comparison_results.append(comparison_result)


def _apply_proposition_transition(
    entity: dict[str, Any],
    entity_kind: str,
    proposition_kind: str,
    evidence_ref: Mapping[str, Any],
) -> str:
    """Apply only the FSM transition owned by a validated proposition."""

    if proposition_kind in {"goal_release", "goal_supersession"}:
        if entity_kind != "goal":
            return "reinforce"
        transitioned = transition_goal(
            entity,
            transition="abandoned",
            explicit_release=proposition_kind == "goal_release",
            superseding_goal_validated=proposition_kind == "goal_supersession",
        )
        entity.update(transitioned)
        return "resolve"
    if proposition_kind == "completion_meaning":
        marker = {"outcome_kind": "completion"}
        if entity_kind == "goal":
            transitioned = transition_goal(
                entity,
                transition="satisfied",
                evidence=marker,
            )
        elif entity_kind == "event":
            transitioned = transition_event(
                entity,
                transition="resolved",
                evidence=marker,
            )
        else:
            return "reinforce"
        entity.update(transitioned)
        return "resolve"
    if proposition_kind == "resolution_meaning":
        marker = {"outcome_kind": "resolve"}
        if entity_kind == "threat":
            if entity["residual_pressure"] > 20:
                raise CognitionStateError("threat resolution requires reduced pressure")
            transitioned = transition_threat(
                entity,
                transition="resolved",
                evidence=marker,
            )
        elif entity_kind == "event":
            transitioned = transition_event(
                entity,
                transition="resolved",
                evidence={"outcome_kind": "repair"},
            )
        elif entity_kind == "knowledge_gap":
            transitioned = transition_knowledge_gap(
                entity,
                transition="resolved",
                evidence={"outcome_kind": "answer"},
            )
        else:
            return "reinforce"
        entity.update(transitioned)
        return "resolve"
    return "reinforce"


def _role_refs_from_proposition(
    proposition: Mapping[str, Any],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    """Map validated semantic roles to persistent role refs."""

    refs: list[dict[str, str]] = []
    for assignment in proposition["role_assignments"]:
        ref = handle_to_ref.get(assignment["entity_handle"])
        if ref is None or ref["kind"] not in ROLE_ENTITY_KINDS:
            continue
        refs.append({
            "role": assignment["role"],
            "entity_kind": ref["kind"],
            "entity_id": ref["entity_id"],
        })
    return refs


def _causal_candidate_id(
    state: Mapping[str, Any],
    kind: str,
    evidence_ref: Mapping[str, Any],
) -> str:
    """Create a stable scoped identity for one evidence-grounded candidate."""

    material = "|".join(
        (
            "cognition_state.v2",
            state["state_scope"],
            state.get("owner_user_id", "global"),
            str(evidence_ref["source_kind"]),
            str(evidence_ref["source_id"]),
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"{kind}:{digest}"


def _new_causal_candidate(
    state: Mapping[str, Any],
    kind: str,
    entity_id: str,
    description: str,
    evidence_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a complete zero-based causal entity before applying deltas."""

    timestamp = state["updated_at"]
    common = {
        "entity_id": entity_id,
        "description": description,
        "salience": 0,
        "role_refs": [],
        "evidence_refs": [deepcopy(dict(ref)) for ref in evidence_refs],
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    if kind == "event":
        return {
            **common,
            "status": "active",
            "outcome_impact": 0,
            "responsibility": 0,
            "intentionality": 0,
            "harm": 0,
            "unfairness": 0,
            "exposure": 0,
            "repair_need": 0,
            "reparability": 100,
            "expectation_mismatch": 0,
            "norm_violation": 0,
            "contamination_risk": 0,
            "identity_threat": 0,
            "comparison_gap": 0,
            "vastness": 0,
            "memory_warmth": 0,
            "temporal_loss": 0,
        }
    if kind == "threat":
        return {
            **common,
            "status": "active",
            "likelihood": 0,
            "expected_harm": 0,
            "uncertainty": 0,
            "controllability": 50,
            "coping_potential": 50,
            "residual_pressure": 0,
        }
    return {
        **common,
        "status": "open",
        "relevance": 0,
        "uncertainty": 0,
        "learnability": 0,
        "novelty": 0,
        "model_accommodation": 0,
    }


def _recompute_new_causal_salience(
    state: dict[str, Any],
    translated_deltas: Sequence[Mapping[str, Any]],
    new_causal_ids: set[str],
) -> None:
    """Set candidate salience from accepted causal deltas and reject weak creates."""

    counts: dict[str, int] = {}
    magnitudes: dict[str, int] = {}
    for proposal in translated_deltas:
        path = proposal["target_path"]
        counts[path] = counts.get(path, 0) + 1
    for proposal in translated_deltas:
        path = proposal["target_path"]
        if counts[path] != 1:
            continue
        pieces = path.split(".")
        if len(pieces) != 3:
            continue
        entity_id = pieces[1]
        magnitudes[entity_id] = max(
            magnitudes.get(entity_id, 0),
            abs(int(proposal["delta"])),
        )

    for field_name in ("threats", "active_events", "knowledge_gaps"):
        retained = []
        for entity in state[field_name]:
            if entity["entity_id"] not in new_causal_ids:
                retained.append(entity)
                continue
            entity["salience"] = magnitudes.get(entity["entity_id"], 0)
            if entity["salience"] >= 25:
                retained.append(entity)
        state[field_name] = retained


def _native_delta_path(
    prompt_path: str,
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> str:
    """Resolve one prompt-local target path to a persistent state path."""

    pieces = prompt_path.split(".")
    if len(pieces) == 3:
        field_name, handle, axis = pieces
        ref = handle_to_ref.get(handle)
        if ref is None:
            raise CognitionStateError("semantic delta target handle is unknown")
        if field_name == "drives" and ref["kind"] == "drive":
            return f"drives.{ref['entity_id']}.{axis}"
        if field_name == "meaning_state" and ref["kind"] == "meaning":
            return f"meaning_state.{axis}"
        if field_name in ENTITY_LIST_FIELDS.values():
            return f"{field_name}.{ref['entity_id']}.{axis}"
    if len(pieces) == 3 and pieces[:2] == ["relationship", "r1"]:
        return f"relationship.{pieces[2]}"
    raise CognitionStateError("semantic delta path is not prompt-owned")


def _retain_prompt_evidence(
    state: dict[str, Any],
    results: Sequence[SemanticAppraisalResultV2],
    evidence_by_handle: Mapping[str, Mapping[str, Any]],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> None:
    """Attach complete provenance to every mutable entity cited by appraisal."""

    for result in results:
        handles = set(result["selected_evidence_handles"])
        for delta in result["deltas"]:
            handles.update(delta["evidence_handles"])
            path = delta["target_path"].split(".")
            try:
                target = _target_for_prompt_path(state, path, handle_to_ref)
            except CognitionStateError:
                continue
            if "evidence_refs" in target:
                _append_evidence_rows(
                    target,
                    [evidence_by_handle[handle] for handle in handles],
                )


def _target_for_prompt_path(
    state: Mapping[str, Any],
    pieces: Sequence[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Resolve a prompt path to a mutable native target."""

    if len(pieces) == 3 and pieces[0] == "relationship":
        relationship = state.get("relationship")
        if not isinstance(relationship, dict):
            raise CognitionStateError("relationship target is unavailable")
        return relationship
    if len(pieces) != 3:
        raise CognitionStateError("semantic prompt target is invalid")
    ref = handle_to_ref.get(pieces[1])
    if ref is None:
        raise CognitionStateError("semantic prompt target handle is unknown")
    if pieces[0] == "drives":
        return state["drives"][ref["entity_id"]]
    if pieces[0] == "meaning_state":
        return state["meaning_state"]
    for entity in state[pieces[0]]:
        if entity["entity_id"] == ref["entity_id"]:
            return entity
    raise CognitionStateError("semantic prompt target entity is unknown")


def _append_evidence_rows(
    target: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Append complete evidence records without duplicating source identity."""

    evidence_refs = target.setdefault("evidence_refs", [])
    source_ids = {
        row.get("source_id") for row in evidence_refs if isinstance(row, Mapping)
    }
    for row in rows:
        if row["source_id"] not in source_ids:
            evidence_refs.append(dict(row))
            source_ids.add(row["source_id"])


def _apply_unretained_character_delta(
    state: dict[str, Any],
    proposal: Mapping[str, Any],
) -> None:
    """Apply a guarded character constraint delta whose schema has no evidence list."""

    pieces = proposal["target_path"].split(".")
    if pieces[0] == "drives":
        target = state["drives"][pieces[1]]
        target[pieces[2]] = max(0, min(100, target[pieces[2]] + proposal["delta"]))
        return
    if pieces[0] == "meaning_state":
        target = state["meaning_state"]
        target[pieces[1]] = max(0, min(100, target[pieces[1]] + proposal["delta"]))
        return
    raise CognitionStateError("character delta target is invalid")


def apply_elapsed_decay(
    state: Mapping[str, Any],
    *,
    elapsed_seconds: int,
    rate_per_hour: int,
) -> dict[str, Any]:
    """Apply one user-scope elapsed evolution pass in fixed field order."""

    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")
    if rate_per_hour < 0:
        raise ValueError("rate_per_hour must be non-negative")
    if state["state_scope"] != "user":
        raise CognitionStateError(
            "user elapsed decay cannot be applied to character cognition"
        )
    amount = floor(elapsed_seconds * rate_per_hour / 3600)
    updated_state = deepcopy(dict(state))
    for field_name in _ENTITY_FIELDS:
        for entity in _mapping_values(updated_state[field_name]):
            salience = entity["salience"]
            minimum = 25 if _has_unresolved_pressure(entity) else 0
            entity["salience"] = max(minimum, salience - amount)
            entity["updated_at"] = _timestamp_from_state(updated_state)
    retained_activations: list[dict[str, Any]] = []
    for activation in _mapping_values(updated_state["affect_activations"]):
        emotion_id = activation["emotion_id"]
        definition = EMOTION_DEFINITIONS[emotion_id]
        activation_amount = floor(
            elapsed_seconds * definition.decay_rate_per_hour / 3600
        )
        activation["score"] = max(0, activation["score"] - activation_amount)
        _update_activation_lifecycle(activation, updated_state)
        if activation["score"] > 10:
            retained_activations.append(activation)
    updated_state["affect_activations"] = retained_activations
    return updated_state


def apply_sleep_recovery(
    state: Mapping[str, Any],
    *,
    elapsed_sleep_seconds: int,
    updated_at: str | None = None,
    character_constraints: Mapping[str, Any] | None = None,
    relationship_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one deterministic character-scope sleep recovery pass."""

    if elapsed_sleep_seconds < 0:
        raise ValueError("elapsed_sleep_seconds must be non-negative")
    if state["state_scope"] != "character":
        raise CognitionStateError(
            "sleep recovery requires character cognition scope"
        )
    recovered = deepcopy(dict(state))
    recovery_timestamp = updated_at or recovered["updated_at"]
    recovered["updated_at"] = recovery_timestamp
    decay_amount = floor(elapsed_sleep_seconds * 4 / 3600)
    for drive in _mapping_values(recovered["drives"]):
        drive["pressure"] = max(0, drive["pressure"] - decay_amount - 20)
    for field_name in _ENTITY_FIELDS:
        for entity in _mapping_values(recovered[field_name]):
            if field_name == "threats" and "residual_pressure" in entity:
                entity["residual_pressure"] = max(
                    0,
                    entity["residual_pressure"] - decay_amount - 20,
                )
            minimum = 25 if _has_unresolved_pressure(entity) else 0
            entity["salience"] = max(
                minimum,
                entity["salience"] - decay_amount - 20,
            )
            entity["updated_at"] = recovery_timestamp
    _update_low_coherence_since(recovered)
    retained_activations: list[dict[str, Any]] = []
    for activation in _mapping_values(recovered["affect_activations"]):
        definition = EMOTION_DEFINITIONS[activation["emotion_id"]]
        amount = floor(
            elapsed_sleep_seconds * definition.decay_rate_per_hour / 3600
        ) + 20
        activation["score"] = max(0, activation["score"] - amount)
        _update_activation_lifecycle(activation, recovered)
        if activation["score"] > 10:
            retained_activations.append(activation)
    recovered["affect_activations"] = retained_activations
    return recovered


def apply_state_update(
    state: Mapping[str, Any],
    *,
    direct_facts: Sequence[tuple[str, Mapping[str, Any]]] = (),
    semantic_deltas: Sequence[Mapping[str, Any]] = (),
    elapsed_seconds: int = 0,
    updated_at: str | None = None,
    character_constraints: Mapping[str, Any] | None = None,
    relationship_context: Mapping[str, Any] | None = None,
    transition_contexts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Apply elapsed time, facts, deltas, lifecycle, cache, and retention."""

    if state["state_scope"] == "user":
        updated_state = apply_elapsed_decay(
            state,
            elapsed_seconds=elapsed_seconds,
            rate_per_hour=4,
        )
    else:
        if elapsed_seconds != 0:
            raise CognitionStateError(
                "character elapsed evolution requires sleep recovery"
            )
        updated_state = deepcopy(dict(state))
    accepted_transitions = [deepcopy(dict(row)) for row in transition_contexts]
    for producer, fact in direct_facts:
        prior_fact_state = updated_state
        next_state = apply_direct_fact(
            updated_state,
            fact,
            producer=producer,
        )
        transition = _direct_fact_relief_transition(
            prior_fact_state,
            next_state,
            fact,
        )
        if transition is not None:
            accepted_transitions.append(transition)
        updated_state = next_state
    updated_state = apply_semantic_deltas(updated_state, semantic_deltas)
    _apply_guarded_lifecycle_transitions(updated_state)
    if updated_at is not None:
        updated_state["updated_at"] = updated_at
    _update_low_coherence_since(updated_state)
    updated_state["affect_activations"] = derive_persistent_emotion_activations(
        updated_state,
        updated_at=updated_state["updated_at"],
        character_constraints=character_constraints,
        relationship_context=relationship_context,
        transition_contexts=accepted_transitions,
    )
    retained_state = prune_terminal_entities(updated_state)
    return retained_state


def _direct_fact_relief_transition(
    prior_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    fact: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project an accepted threat-resolution fact into a relief cause."""

    if fact.get("fact_kind") != "threat_resolved":
        return None
    target_ref = fact["target_refs"][0]
    entity_id = target_ref["entity_id"]
    prior_threat = next(
        threat
        for threat in prior_state["threats"]
        if threat["entity_id"] == entity_id
    )
    current_threat = next(
        threat
        for threat in current_state["threats"]
        if threat["entity_id"] == entity_id
    )
    return {
        "root_ref": {
            "scope": prior_state["state_scope"],
            "kind": "threat",
            "entity_id": entity_id,
        },
        "prior": {
            "status": prior_threat["status"],
            "residual_pressure": prior_threat["residual_pressure"],
        },
        "current": {
            "status": current_threat["status"],
            "residual_pressure": current_threat["residual_pressure"],
        },
        "evidence_ref": deepcopy(dict(fact["evidence_ref"])),
        "salience": prior_threat["salience"],
    }


def canonical_event_entity_id(
    state: Mapping[str, Any],
    primary_evidence: Mapping[str, Any],
) -> str:
    """Return the frozen SHA-256 identity for one accepted causal event."""

    owner_key = (
        state["owner_user_id"]
        if state["state_scope"] == "user"
        else "global"
    )
    material = "|".join(
        (
            "cognition_state.v2",
            str(state["state_scope"]),
            str(owner_key),
            str(primary_evidence["source_kind"]),
            str(primary_evidence["source_id"]),
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"event:{digest}"


def reduce_causal_event(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    accepted_deltas: Mapping[str, int],
    primary_evidence: Mapping[str, Any],
    updated_at: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Compare and reduce one event while preserving evidence and outcome."""

    if state["state_scope"] not in {"user", "character"}:
        raise CognitionStateError("event reduction requires a V2 state")
    if not isinstance(primary_evidence, Mapping):
        raise CognitionStateError("event reduction requires primary evidence")
    incoming = _build_event_record(
        state,
        event,
        primary_evidence,
        accepted_deltas,
        updated_at or state["updated_at"],
    )
    updated_state = deepcopy(dict(state))
    stored = _matching_event(updated_state, incoming)
    comparison_event = {
        **incoming,
        "axis_deltas": dict(accepted_deltas),
    }
    outcome = compare_event(comparison_event, stored, accepted_deltas)
    if outcome == "create":
        updated_state["active_events"].append(incoming)
    elif outcome in {"reinforce", "contradict"}:
        _merge_event(stored, incoming, accepted_deltas, outcome)
    elif outcome == "resolve":
        stored["status"] = "resolved"
        stored["repair_need"] = 0
        _append_unique_evidence(stored, primary_evidence)
        stored["updated_at"] = incoming["updated_at"]
    elif outcome == "replace":
        stored["status"] = "replaced"
        _append_unique_evidence(stored, primary_evidence)
        updated_state["active_events"].append(incoming)
    return prune_terminal_entities(updated_state), outcome


def create_guarded_goal(
    state: Mapping[str, Any],
    *,
    goal_kind: str,
    description: str,
    role_refs: Sequence[Mapping[str, Any]],
    evidence_refs: Sequence[Mapping[str, Any]],
    axes: Mapping[str, int],
    primary_root_ref: Mapping[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Create a goal only when its kind-specific causal guard is satisfied."""

    if goal_kind not in GOAL_KINDS:
        raise CognitionStateError("goal kind is not registered")
    if not description or not 1 <= len(description) <= 500:
        raise CognitionStateError("goal description is invalid")
    if not evidence_refs:
        raise CognitionStateError("goal creation requires evidence")
    if goal_kind in {
        "relationship_connection",
        "bond_protection",
        "trust_verification",
        "social_care",
        "reciprocity",
    } and not any(
        role.get("role") == "affected_relationship"
        for role in role_refs
        if isinstance(role, Mapping)
    ):
        raise CognitionStateError("relationship goal requires relationship cause")
    if any(key not in {
        "importance",
        "progress",
        "obstruction",
        "expected_success",
        "controllability",
        "recoverability",
        "urgency",
    } for key in axes):
        raise CognitionStateError("goal axes contain a reducer-owned field")
    root_salience = (
        primary_root_ref.get("salience", 50)
        if isinstance(primary_root_ref, Mapping)
        else 50
    )
    defaults = {
        "importance": 50,
        "progress": 0,
        "obstruction": 0,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": 50,
        "urgency": root_salience,
    }
    defaults.update(axes)
    evidence_id = evidence_refs[0]["source_id"]
    root_id = (
        str(primary_root_ref.get("entity_id"))
        if isinstance(primary_root_ref, Mapping)
        else evidence_id
    )
    entity_id = f"goal:{goal_kind}:{state['state_scope']}:{root_id}"
    goal = {
        "entity_id": entity_id,
        "description": description,
        "salience": defaults["importance"],
        "role_refs": deepcopy(list(role_refs)),
        "evidence_refs": deepcopy(list(evidence_refs)),
        "created_at": updated_at or state["updated_at"],
        "updated_at": updated_at or state["updated_at"],
        "status": "pursuing",
        "goal_kind": goal_kind,
        **{key: defaults[key] for key in (
            "importance",
            "progress",
            "obstruction",
            "expected_success",
            "controllability",
            "recoverability",
            "urgency",
        )},
    }
    return goal


def create_deterministic_goals(
    state: Mapping[str, Any],
    *,
    character_constraints: Mapping[str, Any] | None = None,
    relationship_context: Mapping[str, Any] | None = None,
    evidence: Sequence[Mapping[str, Any]] = (),
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Create or retain every frozen goal kind from pursuing causal roots."""

    updated = deepcopy(dict(state))
    constraints = character_constraints or updated
    relationship = updated.get("relationship") or relationship_context
    now = updated_at or updated["updated_at"]
    episode_evidence_refs = [
        (
            row["evidence_ref"]
            if isinstance(row, Mapping) and "evidence_ref" in row
            else row
        )
        for row in evidence
    ]

    def add_goal(
        goal_kind: str,
        root: Mapping[str, Any],
        importance: int,
        description: str,
        evidence: Sequence[Mapping[str, Any]],
        roles: Sequence[Mapping[str, Any]],
    ) -> None:
        if not evidence:
            return
        goal_id = f"goal:{goal_kind}:{updated['state_scope']}:{root['entity_id']}"
        existing = next(
            (
                goal for goal in updated["goals"]
                if goal.get("entity_id") == goal_id
            ),
            None,
        )
        if existing is not None:
            return
        goal = create_guarded_goal(
            updated,
            goal_kind=goal_kind,
            description=description,
            role_refs=roles,
            evidence_refs=evidence,
            axes={
                "importance": _clamp_axis(importance),
                "urgency": _clamp_axis(root.get("salience", importance)),
            },
            primary_root_ref=root,
            updated_at=now,
        )
        updated["goals"].append(goal)

    if isinstance(relationship, Mapping):
        relationship_root = {
            "scope": updated["state_scope"],
            "kind": "relationship",
            "entity_id": relationship["relationship_id"],
            "salience": relationship["salience"],
        }
        closeness_gap = max(
            relationship["desired_closeness"] - relationship["perceived_closeness"],
            0,
        )
        connection_value = max(
            relationship["attachment"],
            relationship["care"],
            closeness_gap,
        )
        if relationship["salience"] >= 40 and connection_value >= 40:
            add_goal(
                "relationship_connection",
                relationship_root,
                connection_value,
                "restore or deepen the valued relationship connection",
                relationship.get("evidence_refs", []),
                [{
                    "role": "affected_relationship",
                    "entity_kind": "relationship",
                    "entity_id": relationship["relationship_id"],
                }],
            )
        if relationship["boundary_safety"] < -20:
            add_goal(
                "autonomy_boundary",
                relationship_root,
                max(relationship["salience"], 100 + relationship["boundary_safety"]),
                "protect the current relationship boundary",
                relationship.get("evidence_refs", []),
                [{
                    "role": "affected_relationship",
                    "entity_kind": "relationship",
                    "entity_id": relationship["relationship_id"],
                }],
            )

    for threat in list(updated.get("threats", [])):
        if not isinstance(threat, Mapping) or threat.get("status") != "active":
            continue
        threat_root = _entity_root(updated, "threat", threat)
        threat_roles = list(threat.get("role_refs", []))
        relationship_threat = (
            isinstance(relationship, Mapping)
            and any(
                role.get("role") in {"affected_relationship", "object"}
                and role.get("entity_id") == relationship.get("relationship_id")
                for role in threat_roles
                if isinstance(role, Mapping)
            )
        )
        if relationship_threat and relationship["attachment"] >= 40:
            add_goal(
                "bond_protection",
                threat_root,
                max(relationship["attachment"], threat["expected_harm"]),
                "protect the valued relationship from the active threat",
                threat.get("evidence_refs", []),
                threat_roles,
            )
        if relationship_threat and threat["uncertainty"] >= 40:
            add_goal(
                "trust_verification",
                threat_root,
                max(relationship["attachment"], threat["uncertainty"]),
                "verify the uncertain threat to the relationship",
                threat.get("evidence_refs", []),
                threat_roles,
            )
        coping_deficit = 100 - threat["coping_potential"]
        if (
            threat["likelihood"] >= 25
            and threat["expected_harm"] >= 25
            and max(threat["uncertainty"], coping_deficit) >= 25
        ):
            add_goal(
                "safety",
                threat_root,
                max(
                    threat["expected_harm"],
                    _drive_value(constraints, "safety", "importance"),
                ),
                "reduce the active threat and preserve safety",
                threat.get("evidence_refs", []),
                threat_roles,
            )

    for goal in list(updated.get("goals", [])):
        if (
            not isinstance(goal, Mapping)
            or goal.get("status") not in {"pursuing", "blocked", "failed"}
        ):
            continue
        goal_root = _entity_root(updated, "goal", goal)
        goal_roles = list(goal.get("role_refs", []))
        if (
            goal.get("status") in {"pursuing", "blocked"}
            and goal["importance"] >= 40
            and goal["obstruction"] >= 40
        ):
            add_goal(
                "obstruction_resolution",
                goal_root,
                goal["importance"],
                "remove the obstruction blocking an important goal",
                goal.get("evidence_refs", []),
                goal_roles,
            )
        if goal.get("status") == "failed":
            add_goal(
                "loss_recovery",
                goal_root,
                goal["importance"],
                "recover from the failed important goal",
                goal.get("evidence_refs", []),
                goal_roles,
            )

    for event in list(updated.get("active_events", [])):
        if not isinstance(event, Mapping) or event.get("status") != "active":
            continue
        event_root = _entity_root(updated, "event", event)
        event_roles = list(event.get("role_refs", []))
        if event["identity_threat"] >= 40 or event["unfairness"] >= 40:
            add_goal(
                "autonomy_boundary",
                event_root,
                max(event["identity_threat"], event["unfairness"]),
                "protect autonomy and repair the violated boundary",
                event.get("evidence_refs", []),
                event_roles,
            )
        if (
            _has_self_actor(event, updated)
            and event["repair_need"] >= 40
            and max(event["harm"], event["norm_violation"]) >= 40
        ):
            add_goal(
                "moral_repair",
                event_root,
                max(
                    _drive_value(constraints, "integrity", "importance"),
                    event["repair_need"],
                ),
                "repair the self-caused moral injury",
                event.get("evidence_refs", []),
                event_roles,
            )
        if (
            _has_other_experiencer(event, updated)
            and event["harm"] >= 40
            and _drive_value(constraints, "care", "importance") >= 40
        ):
            add_goal(
                "social_care",
                event_root,
                max(_drive_value(constraints, "care", "importance"), event["harm"]),
                "care for the other experiencer under pressure",
                event.get("evidence_refs", []),
                event_roles,
            )
        if (
            _has_other_actor(event, updated)
            and event["outcome_impact"] >= 40
            and event["responsibility"] >= 40
        ):
            add_goal(
                "reciprocity",
                event_root,
                event["outcome_impact"],
                "respond reciprocally to the other actor's positive outcome",
                event.get("evidence_refs", []),
                event_roles,
            )
        if event["outcome_impact"] <= -40:
            add_goal(
                "loss_recovery",
                event_root,
                max(-event["outcome_impact"], event["salience"]),
                "recover from the current negative outcome",
                event.get("evidence_refs", []),
                event_roles,
            )
        if event["comparison_gap"] >= 40 and (
            _drive_value(constraints, "competence", "pressure") >= 40
            or any(
                goal_item.get("importance", 0) >= 40
                for goal_item in updated.get("goals", [])
                if isinstance(goal_item, Mapping)
            )
        ):
            add_goal(
                "self_improvement",
                event_root,
                max(
                    event["comparison_gap"],
                    _drive_value(constraints, "competence", "pressure"),
                ),
                "improve in response to the comparison gap",
                event.get("evidence_refs", []),
                event_roles,
            )

    for gap in list(updated.get("knowledge_gaps", [])):
        if (
            isinstance(gap, Mapping)
            and gap.get("status") in {"open", "reduced"}
            and gap["relevance"] >= 40
            and gap["learnability"] >= 40
        ):
            add_goal(
                "epistemic_exploration",
                _entity_root(updated, "knowledge_gap", gap),
                max(
                    gap["relevance"],
                    _drive_value(constraints, "exploration", "importance"),
                ),
                "explore the open learnable knowledge gap",
                gap.get("evidence_refs", []),
                list(gap.get("role_refs", [])),
            )

    meaning = updated.get("meaning_state")
    if isinstance(meaning, Mapping) and (
        meaning["purpose_coherence"] < 40 or meaning["agency"] < 40
    ) and _drive_value(constraints, "meaning", "pressure") >= 40:
        add_goal(
            "meaning_reconstruction",
            {
                "scope": "character",
                "kind": "meaning",
                "entity_id": "meaning:character",
                "salience": meaning["salience"],
            },
            max(
                _drive_value(constraints, "meaning", "pressure"),
                100 - meaning["purpose_coherence"],
                100 - meaning["agency"],
            ),
            "reconstruct meaning while purpose and agency remain low",
            episode_evidence_refs[:1],
            [],
        )
    return prune_terminal_entities(updated)


def _build_event_record(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    primary_evidence: Mapping[str, Any],
    accepted_deltas: Mapping[str, int],
    updated_at: str,
) -> dict[str, Any]:
    """Build an exact event row from accepted typed inputs."""

    event_axes = (
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
    )
    if any(axis not in event for axis in event_axes):
        raise CognitionStateError("causal event axes are incomplete")
    axes = {field_name: event[field_name] for field_name in event_axes}
    if "description" not in event or "role_refs" not in event:
        raise CognitionStateError("causal event identity is incomplete")
    salience = max((abs(delta) for delta in accepted_deltas.values()), default=0)
    return {
        "entity_id": canonical_event_entity_id(state, primary_evidence),
        "description": event["description"],
        "salience": min(100, salience),
        "role_refs": deepcopy(list(event["role_refs"])),
        "evidence_refs": [deepcopy(dict(primary_evidence))],
        "created_at": updated_at,
        "updated_at": updated_at,
        "status": "active",
        **axes,
    }


def _matching_event(
    state: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Find an existing event by canonical entity identity or affected refs."""

    for entity in state.get("active_events", []):
        if not isinstance(entity, dict):
            continue
        if entity.get("entity_id") == incoming.get("entity_id"):
            return entity
        if _compatible_event_roles(incoming, entity):
            return entity
    return None


def _compatible_event_roles(
    incoming: Mapping[str, Any],
    stored: Mapping[str, Any],
) -> bool:
    """Match repeated incidents by non-empty compatible role references."""

    def signature(event: Mapping[str, Any]) -> set[tuple[str, str, str]]:
        return {
            (
                str(role.get("role")),
                str(role.get("entity_kind")),
                str(role.get("entity_id")),
            )
            for role in event.get("role_refs", [])
            if isinstance(role, Mapping)
        }

    incoming_roles = signature(incoming)
    stored_roles = signature(stored)
    return bool(incoming_roles) and incoming_roles == stored_roles


def _merge_event(
    stored: dict[str, Any],
    incoming: Mapping[str, Any],
    accepted_deltas: Mapping[str, int],
    outcome: str,
) -> None:
    """Apply deterministic reinforcement or contradiction to one event."""

    for axis, delta in accepted_deltas.items():
        if axis not in stored or not isinstance(delta, int):
            continue
        minimum = -100 if axis == "outcome_impact" else 0
        stored[axis] = max(minimum, min(100, stored[axis] + delta))
    if outcome == "contradict":
        stored["repair_need"] = min(100, stored["repair_need"] + 20)
    salience_delta = min(
        40,
        max((abs(delta) for delta in accepted_deltas.values()), default=0),
    )
    stored["salience"] = min(100, stored["salience"] + salience_delta)
    _append_unique_evidence(stored, incoming["evidence_refs"][0])
    stored["updated_at"] = incoming["updated_at"]


def _append_unique_evidence(
    entity: dict[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    """Append a complete evidence record once."""

    refs = entity["evidence_refs"]
    if evidence not in refs:
        if len(refs) >= 8:
            raise CognitionStateError("evidence reference cap reached")
        refs.append(deepcopy(dict(evidence)))


def _apply_guarded_lifecycle_transitions(
    state: dict[str, Any],
) -> None:
    """Apply only automatic transitions whose non-semantic guards are complete."""

    for goal in _mapping_values(state["goals"]):
        if goal["status"] == "pursuing" and goal["obstruction"] >= 40:
            goal["status"] = "blocked"
        elif (
            goal["status"] == "blocked"
            and goal["obstruction"] < 25
            and goal["recoverability"] >= 25
        ):
            goal["status"] = "pursuing"


def _update_activation_lifecycle(
    activation: dict[str, Any],
    state: Mapping[str, Any],
) -> None:
    """Apply exact activation phase and retention thresholds after decay."""

    score = activation["score"]
    roots = activation.get("root_refs", [activation["primary_root"]])
    cause_active = any(_root_is_active(state, root) for root in roots)
    cause_replaced = any(_root_is_replaced(state, root) for root in roots)
    if cause_active:
        activation["cause_status"] = "active"
    elif cause_replaced:
        activation["cause_status"] = "replaced"
    else:
        activation["cause_status"] = "resolved"
    if score <= 10:
        activation["phase"] = "fading"
    elif cause_active and score >= 25:
        activation["phase"] = "active"
    else:
        activation["phase"] = "fading"
    activation["trend"] = "falling" if score < activation["peak_score"] else "stable"


def _root_is_active(state: Mapping[str, Any], root: Mapping[str, Any]) -> bool:
    """Return whether an activation root still has an unresolved cause."""

    field_name = ENTITY_LIST_FIELDS.get(root["kind"])
    if field_name is None:
        return False
    for entity in state[field_name]:
        if entity["entity_id"] != root["entity_id"]:
            continue
        return entity["status"] not in {
            "satisfied",
            "failed",
            "abandoned",
            "resolved",
            "replaced",
        }
    return False


def _root_is_replaced(state: Mapping[str, Any], root: Mapping[str, Any]) -> bool:
    """Return whether an activation root was superseded by another cause."""

    field_name = ENTITY_LIST_FIELDS.get(root["kind"])
    if field_name is None:
        return False
    return any(
        entity["entity_id"] == root["entity_id"]
        and entity["status"] == "replaced"
        for entity in state[field_name]
    )


def _has_unresolved_pressure(entity: Mapping[str, Any]) -> bool:
    """Return whether a causal row keeps a minimum felt salience."""

    if entity["status"] not in {"active", "pursuing", "blocked", "open", "reduced"}:
        return False
    return any(
        field_name in entity and entity[field_name] > 0
        for field_name in (
            "residual_pressure",
            "obstruction",
            "harm",
            "repair_need",
            "uncertainty",
        )
    )


def _timestamp_from_state(state: Mapping[str, Any]) -> str:
    """Use the state timestamp for deterministic in-place evolution."""

    return state["updated_at"]


def _mapping_values(value: Any) -> list[dict[str, Any]]:
    """Return mutable mapping items from one state collection."""

    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _entity_root(
    state: Mapping[str, Any],
    kind: str,
    entity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic root reference used by goal identity."""

    return {
        "scope": state["state_scope"],
        "kind": kind,
        "entity_id": entity["entity_id"],
        "salience": entity.get("salience", 0),
    }


def _drive_value(
    constraints: Mapping[str, Any],
    drive_id: str,
    field_name: str,
) -> int:
    """Read one validated character drive constraint."""

    drives = constraints.get("drives", {})
    drive = drives.get(drive_id, {}) if isinstance(drives, Mapping) else {}
    value = drive.get(field_name, 0) if isinstance(drive, Mapping) else 0
    return value if isinstance(value, int) else 0


def _clamp_axis(value: int) -> int:
    """Clamp deterministic goal axes to the native state range."""

    return max(0, min(100, int(value)))


def _has_self_actor(entity: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Check whether a causal row assigns agency to the active self."""

    del state
    return any(
        isinstance(role, Mapping)
        and role.get("role") == "actor"
        and role.get("entity_kind") == "character"
        and role.get("entity_id") in {"character:global", "self", "character"}
        for role in entity.get("role_refs", [])
    )


def _has_other_actor(entity: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Check whether a causal row assigns agency to another actor."""

    del state
    return any(
        isinstance(role, Mapping)
        and role.get("role") == "actor"
        and not (
            role.get("entity_kind") == "character"
            and role.get("entity_id") in {"character:global", "self", "character"}
        )
        for role in entity.get("role_refs", [])
    )


def _has_other_experiencer(
    entity: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool:
    """Check whether another experiencer is affected by a causal row."""

    del state
    return any(
        isinstance(role, Mapping)
        and role.get("role") == "experiencer"
        and not (
            role.get("entity_kind") == "character"
            and role.get("entity_id") in {"character:global", "self", "character"}
        )
        for role in entity.get("role_refs", [])
    )


def _update_low_coherence_since(state: dict[str, Any]) -> None:
    """Track the first continuous low-purpose/low-agency transition."""

    meaning = state.get("meaning_state")
    if not isinstance(meaning, dict):
        return
    low = meaning["purpose_coherence"] < 40 and meaning["agency"] < 40
    if low and "low_coherence_since" not in meaning:
        meaning["low_coherence_since"] = state["updated_at"]
    elif not low:
        meaning.pop("low_coherence_since", None)
