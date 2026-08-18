"""Deterministic translation of validated semantic inputs into V2 state."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date
from math import floor
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionEvidenceV2,
    SemanticAppraisalResultV2,
    SemanticDeltaApplicationResultV2,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
    derive_persistent_emotion_activations,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    ENTITY_LIST_FIELDS,
    GOAL_KINDS,
    MAX_PROCESSED_SOURCE_IDS,
    ROLE_ENTITY_KINDS,
    CognitionStateError,
    prune_terminal_entities,
)
from kazusa_ai_chatbot.cognition_core_v2.transition_guards import (
    apply_direct_fact,
    apply_semantic_deltas,
    compare_event,
    retain_bounded_evidence,
    transition_event,
    transition_goal,
    transition_knowledge_gap,
    transition_threat,
)

_ENTITY_FIELDS = ("goals", "threats", "active_events", "knowledge_gaps")
MAX_EVIDENCE_REFS_PER_TARGET = 8
CHARACTER_ELAPSED_SALIENCE_RATE_PER_HOUR = 4
USER_SALIENCE_DECAY_RATE_PER_HOUR = 4
SECONDS_PER_HOUR = 3600
FAMILIARITY_DATE_INCREMENT = 1
FAMILIARITY_DAILY_BONUS_INCREMENT = 1
RELATIONSHIP_DAILY_INCREMENT_CAP = 2
RELATIONSHIP_MAINTENANCE_SOURCE_PREFIX = "episode:"
TRUSTED_RELATIONSHIP_FACT_PRODUCERS = frozenset({
    "action_result",
    "resolver_observation",
    "tool_result",
})
TRUSTED_RELATIONSHIP_FACT_KINDS = frozenset({
    "goal_progress_observed",
    "goal_completed",
    "goal_terminal_failure",
    "goal_obstruction_removed",
    "threat_resolved",
    "event_repaired",
    "knowledge_answered",
})


def apply_semantic_appraisals(
    state: Mapping[str, Any],
    results: Sequence[SemanticAppraisalResultV2],
    evidence: Sequence[CognitionEvidenceV2],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    comparison_results: list[dict[str, Any]] | None = None,
) -> SemanticDeltaApplicationResultV2:
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
    batch_terminalizations: set[tuple[str, str, str]] = set()
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
                batch_terminalizations,
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
    delta_result = apply_semantic_deltas(updated, translated)
    updated = delta_result["updated_state"]
    _recompute_new_causal_salience(
        updated,
        translated,
        new_causal_ids,
        batch_terminalizations,
    )
    for proposal in unsupported:
        _apply_unretained_character_delta(updated, proposal)
    _reassert_terminal_postconditions(
        updated,
        results,
        local_handle_to_ref,
    )
    return {
        "updated_state": updated,
        "accepted_delta_receipts": delta_result[
            "accepted_delta_receipts"
        ],
        "rejected_delta_receipts": delta_result[
            "rejected_delta_receipts"
        ],
    }


def _reassert_terminal_postconditions(
    state: dict[str, Any],
    results: Sequence[SemanticAppraisalResultV2],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> None:
    """Restore canonical axes selected by accepted terminal propositions.

    Args:
        state: Reduced mutable state after all accepted appraisal deltas.
        results: Appraisal batch whose terminal assertions own postconditions.
        handle_to_ref: Native bindings after candidate materialization.
    """

    contracts = {
        "goal_completed": ("goal", {"progress": 100}),
        "event_completed": ("event", {"repair_need": 0}),
        "event_repaired": (
            "event",
            {"repair_need": 0, "reparability": 100},
        ),
        "threat_resolved": ("threat", {"residual_pressure": 0}),
        "knowledge_answered": ("knowledge_gap", {"uncertainty": 0}),
    }
    for result in results:
        for proposition in result["propositions"]:
            contract = contracts.get(proposition["proposition_kind"])
            if contract is None:
                continue
            entity_kind, axes = contract
            subject_ref = handle_to_ref[proposition["subject_handle"]]
            if subject_ref["kind"] != entity_kind:
                raise CognitionStateError(
                    "terminal proposition postcondition kind is invalid"
                )
            field_name = ENTITY_LIST_FIELDS[entity_kind]
            entity = next(
                (
                    row
                    for row in state[field_name]
                    if row["entity_id"] == subject_ref["entity_id"]
                ),
                None,
            )
            if entity is None:
                raise CognitionStateError(
                    "terminal proposition postcondition target is unknown"
                )
            entity.update(axes)


def _materialize_proposition_root(
    state: dict[str, Any],
    proposition: Mapping[str, Any],
    evidence_by_handle: Mapping[str, Mapping[str, Any]],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    selected_evidence_handles: Sequence[str],
    comparison_results: list[dict[str, Any]] | None,
    new_causal_ids: set[str],
    batch_terminalizations: set[tuple[str, str, str]],
) -> None:
    """Turn a validated prompt-local proposition into a causal state root."""

    if proposition["proposition_kind"] == "outcome_pending":
        return
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
    subject_id = subject_ref["entity_id"]
    candidate_subject = subject_id.startswith("candidate:")
    if candidate_subject:
        candidate_parts = subject_id.split(":", maxsplit=2)
        candidate_evidence_handle = (
            candidate_parts[2]
            if len(candidate_parts) == 3
            and candidate_parts[1] in {"event", "threat", "knowledge_gap"}
            else ""
        )
        if candidate_evidence_handle not in evidence_handles:
            raise CognitionStateError(
                "causal candidate evidence does not match its source"
            )
        evidence_handles = [candidate_evidence_handle]
    evidence_ref = evidence_by_handle.get(evidence_handles[0])
    if evidence_ref is None:
        raise CognitionStateError("causal proposition evidence is unknown")
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
    terminal_status = _proposition_terminal_status(
        subject_kind,
        proposition["proposition_kind"],
    )
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
        if terminal_status:
            outcome = _apply_proposition_transition(
                existing,
                subject_kind,
                proposition["proposition_kind"],
                evidence_ref,
            )
            if outcome == "resolve":
                batch_terminalizations.add((
                    subject_kind,
                    existing["entity_id"],
                    terminal_status,
                ))
        else:
            new_causal_ids.add(existing["entity_id"])
            outcome = "create"
    else:
        superseding_goal_validated = _validate_goal_supersession(
            state,
            existing,
            proposition,
            handle_to_ref,
        )
        terminalization = (
            subject_kind,
            existing["entity_id"],
            terminal_status,
        )
        if terminal_status and terminalization in batch_terminalizations:
            outcome = "resolve"
        else:
            outcome = _apply_proposition_transition(
                existing,
                subject_kind,
                proposition["proposition_kind"],
                evidence_ref,
                superseding_goal_validated=superseding_goal_validated,
            )
            if outcome == "resolve" and terminal_status:
                batch_terminalizations.add(terminalization)
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


def _proposition_terminal_status(
    entity_kind: str,
    proposition_kind: str,
) -> str:
    """Return the terminal status owned by one semantic proposition."""

    if proposition_kind in {"goal_release", "goal_supersession"}:
        return "abandoned" if entity_kind == "goal" else ""
    statuses = {
        "goal_completed": ("goal", "satisfied"),
        "event_completed": ("event", "resolved"),
        "threat_resolved": ("threat", "resolved"),
        "event_repaired": ("event", "resolved"),
        "knowledge_answered": ("knowledge_gap", "resolved"),
    }
    terminal_kind = statuses.get(proposition_kind)
    if terminal_kind is None or terminal_kind[0] != entity_kind:
        return ""
    return terminal_kind[1]


def _apply_proposition_transition(
    entity: dict[str, Any],
    entity_kind: str,
    proposition_kind: str,
    evidence_ref: Mapping[str, Any],
    *,
    superseding_goal_validated: bool = False,
) -> str:
    """Apply only the FSM transition owned by a validated proposition."""

    if proposition_kind in {"goal_release", "goal_supersession"}:
        if entity_kind != "goal":
            return "reinforce"
        transitioned = transition_goal(
            entity,
            transition="abandoned",
            explicit_release=proposition_kind == "goal_release",
            superseding_goal_validated=superseding_goal_validated,
        )
        entity.update(transitioned)
        return "resolve"
    candidate = deepcopy(entity)
    if proposition_kind == "goal_completed" and entity_kind == "goal":
        candidate["progress"] = 100
        transitioned = transition_goal(
            candidate,
            transition="satisfied",
            evidence={"outcome_kind": "completion"},
        )
    elif proposition_kind == "event_completed" and entity_kind == "event":
        candidate["repair_need"] = 0
        transitioned = transition_event(
            candidate,
            transition="resolved",
            evidence={"outcome_kind": "completion"},
        )
    elif proposition_kind == "threat_resolved" and entity_kind == "threat":
        candidate["residual_pressure"] = 0
        transitioned = transition_threat(
            candidate,
            transition="resolved",
            evidence={"outcome_kind": "resolve"},
        )
    elif proposition_kind == "event_repaired" and entity_kind == "event":
        candidate["repair_need"] = 0
        candidate["reparability"] = 100
        transitioned = transition_event(
            candidate,
            transition="resolved",
            evidence={"outcome_kind": "repair"},
        )
    elif (
        proposition_kind == "knowledge_answered"
        and entity_kind == "knowledge_gap"
    ):
        candidate["uncertainty"] = 0
        transitioned = transition_knowledge_gap(
            candidate,
            transition="resolved",
            evidence={"outcome_kind": "answer"},
        )
    else:
        return "reinforce"
    entity.update(transitioned)
    return "resolve"


def _validate_goal_supersession(
    state: Mapping[str, Any],
    subject: Mapping[str, Any],
    proposition: Mapping[str, Any],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> bool:
    """Validate a distinct pursuing replacement before abandoning an old goal."""

    if proposition["proposition_kind"] != "goal_supersession":
        return False
    object_handle = proposition.get("object_handle")
    object_ref = handle_to_ref.get(object_handle)
    if object_ref is None or object_ref["kind"] != "goal":
        raise CognitionStateError("goal supersession target is invalid")
    if object_ref["entity_id"] == subject["entity_id"]:
        raise CognitionStateError("goal supersession target must be distinct")
    replacement = next(
        (
            goal
            for goal in state["goals"]
            if goal["entity_id"] == object_ref["entity_id"]
        ),
        None,
    )
    if replacement is None or replacement["status"] != "pursuing":
        raise CognitionStateError("goal supersession requires a pursuing goal")
    return True


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
        "evidence_refs": retain_bounded_evidence(
            [],
            evidence_refs,
            preserve_primary=True,
        ),
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
    batch_terminalizations: set[tuple[str, str, str]],
) -> None:
    """Set candidate salience and reject weak creates without a terminal claim.

    A candidate terminalized by an accepted proposition in this same batch is
    retained through salience recomputation so its terminal postcondition can
    resolve against the surviving native entity. Non-terminal candidates below
    the threshold remain prunable.
    """

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

    terminalized_ids = {
        entity_id
        for _, entity_id, _ in batch_terminalizations
    }
    for field_name in ("threats", "active_events", "knowledge_gaps"):
        retained = []
        for entity in state[field_name]:
            if entity["entity_id"] not in new_causal_ids:
                retained.append(entity)
                continue
            entity["salience"] = magnitudes.get(entity["entity_id"], 0)
            if (
                entity["salience"] >= 25
                or entity["entity_id"] in terminalized_ids
            ):
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

    rows_by_target: dict[
        str,
        tuple[dict[str, Any], list[Mapping[str, Any]]],
    ] = {}
    for result in results:
        for delta in result["deltas"]:
            path = delta["target_path"].split(".")
            try:
                target = _target_for_prompt_path(state, path, handle_to_ref)
            except CognitionStateError:
                continue
            if "evidence_refs" in target:
                target_key = ".".join(path[:2])
                _, rows = rows_by_target.setdefault(target_key, (target, []))
                rows.extend(
                    evidence_by_handle[handle]
                    for handle in delta["evidence_handles"]
                )
    for target, rows in rows_by_target.values():
        _retain_current_batch_evidence(target, rows)


def _retain_current_batch_evidence(
    target: dict[str, Any],
    current_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Pin cited batch evidence before retaining the newest historical rows.

    Args:
        target: Mutable relationship or causal entity receiving appraisal deltas.
        current_rows: Complete evidence rows cited by accepted deltas for target.
    """

    pinned_rows: list[dict[str, Any]] = []
    pinned_identities: set[tuple[str, str]] = set()
    for row in current_rows:
        identity = (row["source_kind"], row["source_id"])
        if identity in pinned_identities:
            continue
        pinned_rows.append(deepcopy(dict(row)))
        pinned_identities.add(identity)
    preserve_primary = (
        "relationship_id" not in target and bool(target["evidence_refs"])
    )
    primary_row = (
        deepcopy(dict(target["evidence_refs"][0]))
        if preserve_primary
        else None
    )
    primary_identity = (
        (primary_row["source_kind"], primary_row["source_id"])
        if primary_row is not None
        else None
    )
    required_count = len(pinned_rows)
    if primary_identity is not None and primary_identity not in pinned_identities:
        required_count += 1
    if required_count > MAX_EVIDENCE_REFS_PER_TARGET:
        raise CognitionStateError(
            "semantic appraisal current evidence exceeds retention capacity"
        )

    historical_rows: list[dict[str, Any]] = []
    historical_identities: set[tuple[str, str]] = set()
    for row in target["evidence_refs"]:
        identity = (row["source_kind"], row["source_id"])
        if (
            identity == primary_identity
            or identity in pinned_identities
            or identity in historical_identities
        ):
            continue
        historical_rows.append(deepcopy(dict(row)))
        historical_identities.add(identity)

    retained_rows = [
        row
        for row in pinned_rows
        if (row["source_kind"], row["source_id"]) != primary_identity
    ]
    if primary_row is not None:
        retained_rows.insert(0, primary_row)
    historical_capacity = MAX_EVIDENCE_REFS_PER_TARGET - len(retained_rows)
    retained_historical = (
        historical_rows[-historical_capacity:]
        if historical_capacity > 0
        else []
    )
    target["evidence_refs"] = [*retained_rows, *retained_historical]


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
    target["evidence_refs"] = retain_bounded_evidence(
        evidence_refs,
        rows,
        preserve_primary="relationship_id" not in target,
    )


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


def apply_character_elapsed_decay(
    state: Mapping[str, Any],
    *,
    elapsed_seconds: int,
) -> dict[str, Any]:
    """Return an effective character state after ordinary elapsed fading.

    The returned state remains an in-memory view. Its ``updated_at`` token and
    every persisted entity timestamp remain unchanged until a later semantic
    write commits the evolved base through the character compare-and-set
    owner.
    """

    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")
    if state["state_scope"] != "character":
        raise CognitionStateError(
            "character elapsed decay requires character cognition scope"
        )

    updated_state = deepcopy(dict(state))
    salience_amount = floor(
        elapsed_seconds * CHARACTER_ELAPSED_SALIENCE_RATE_PER_HOUR / 3600
    )
    for field_name in _ENTITY_FIELDS:
        for entity in _mapping_values(updated_state[field_name]):
            minimum = 25 if _has_unresolved_pressure(entity) else 0
            entity["salience"] = max(
                minimum,
                entity["salience"] - salience_amount,
            )

    retained_activations: list[dict[str, Any]] = []
    for activation in _mapping_values(updated_state["affect_activations"]):
        definition = EMOTION_DEFINITIONS[activation["emotion_id"]]
        activation_amount = floor(
            elapsed_seconds * definition.decay_rate_per_hour / 3600
        )
        activation["score"] = max(
            0,
            activation["score"] - activation_amount,
        )
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
            rate_per_hour=USER_SALIENCE_DECAY_RATE_PER_HOUR,
        )
    else:
        updated_state = apply_character_elapsed_decay(
            state,
            elapsed_seconds=elapsed_seconds,
        )
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
    delta_result = apply_semantic_deltas(updated_state, semantic_deltas)
    updated_state = delta_result["updated_state"]
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


def apply_relationship_maintenance(
    state: Mapping[str, Any],
    *,
    source_episode_id: str,
    interaction_date_utc: str,
    elapsed_seconds: int,
    accepted_relationship_deltas: Sequence[Mapping[str, Any]] = (),
    trusted_facts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Apply monotonic relationship familiarity and salience maintenance."""

    if state["state_scope"] != "user":
        raise CognitionStateError(
            "relationship maintenance requires user cognition scope"
        )
    if (
        not isinstance(source_episode_id, str)
        or not source_episode_id
        or source_episode_id.startswith(RELATIONSHIP_MAINTENANCE_SOURCE_PREFIX)
    ):
        raise CognitionStateError("relationship source identity is invalid")
    _validate_interaction_date(interaction_date_utc)
    if elapsed_seconds < 0:
        raise CognitionStateError("relationship elapsed time is invalid")

    updated_state = deepcopy(dict(state))
    relationship = updated_state["relationship"]
    maintenance = relationship["relationship_maintenance"]
    source_id = (
        f"{RELATIONSHIP_MAINTENANCE_SOURCE_PREFIX}{source_episode_id}"
    )
    last_date = maintenance["last_interaction_date_utc"]
    processed_source_ids = maintenance["processed_source_ids"]
    if last_date is not None and interaction_date_utc < last_date:
        return updated_state
    if source_id in processed_source_ids:
        return updated_state
    if last_date is None or interaction_date_utc > last_date:
        next_source_ids = [source_id]
        relationship["familiarity"] = min(
            100,
            relationship["familiarity"] + FAMILIARITY_DATE_INCREMENT,
        )
    else:
        if len(processed_source_ids) >= MAX_PROCESSED_SOURCE_IDS:
            raise CognitionStateError(
                "relationship maintenance source ledger exceeds its cap"
            )
        next_source_ids = [*processed_source_ids, source_id]

    relationship["salience"] = max(
        0,
        relationship["salience"]
        - floor(
            elapsed_seconds
            * USER_SALIENCE_DECAY_RATE_PER_HOUR
            / SECONDS_PER_HOUR
        ),
    )
    has_unique_relationship_delta, strongest_delta = (
        _strongest_relationship_delta(
            accepted_relationship_deltas,
        )
    )
    relationship["salience"] = min(
        100,
        relationship["salience"] + strongest_delta,
    )
    qualifies_for_bonus = has_unique_relationship_delta or any(
        _is_trusted_relationship_fact(fact)
        for fact in trusted_facts
    )
    daily_increment = 0
    if (
        qualifies_for_bonus
        and maintenance["last_bonus_date_utc"] != interaction_date_utc
    ):
        daily_increment = min(
            FAMILIARITY_DAILY_BONUS_INCREMENT,
            RELATIONSHIP_DAILY_INCREMENT_CAP - FAMILIARITY_DATE_INCREMENT,
        )
        maintenance["last_bonus_date_utc"] = interaction_date_utc
    relationship["familiarity"] = min(
        100,
        relationship["familiarity"] + daily_increment,
    )
    maintenance["last_interaction_date_utc"] = interaction_date_utc
    maintenance["last_source_id"] = source_id
    maintenance["processed_source_ids"] = next_source_ids
    relationship["updated_at"] = updated_state["updated_at"]
    return updated_state


def _validate_interaction_date(value: str) -> None:
    """Validate the canonical UTC interaction date carrier."""

    if not isinstance(value, str):
        raise CognitionStateError("relationship interaction date is invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise CognitionStateError(
            "relationship interaction date is invalid"
        ) from exc
    if parsed.isoformat() != value:
        raise CognitionStateError("relationship interaction date is invalid")


def _strongest_relationship_delta(
    receipts: Sequence[Mapping[str, Any]],
) -> tuple[bool, int]:
    """Return unique-receipt presence and strongest applied delta."""

    unique_targets: set[str] = set()
    strongest = 0
    for receipt in receipts:
        if receipt.get("duplicate_disposition") != "unique":
            continue
        target_path = receipt.get("target_path")
        axis = receipt.get("relationship_axis")
        applied_delta = receipt.get("applied_delta")
        if (
            not isinstance(target_path, str)
            or not target_path.startswith("relationship.")
            or not isinstance(axis, str)
            or not isinstance(applied_delta, int)
            or target_path in unique_targets
        ):
            continue
        unique_targets.add(target_path)
        strongest = max(strongest, abs(applied_delta))
    return bool(unique_targets), strongest


def _is_trusted_relationship_fact(fact: Mapping[str, Any]) -> bool:
    """Recognize a guarded user-specific fact eligible for a daily bonus."""

    producer = fact.get("producer")
    fact_kind = fact.get("fact_kind")
    nested_fact = fact.get("fact")
    if isinstance(nested_fact, Mapping):
        fact_kind = nested_fact.get("fact_kind", fact_kind)
    return (
        isinstance(producer, str)
        and isinstance(fact_kind, str)
        and producer in TRUSTED_RELATIONSHIP_FACT_PRODUCERS
        and fact_kind in TRUSTED_RELATIONSHIP_FACT_KINDS
    )


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
        "evidence_refs": retain_bounded_evidence(
            [],
            evidence_refs,
            preserve_primary=True,
        ),
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
    reconcile_salience_gated_goals: bool = False,
) -> dict[str, Any]:
    """Create goals and optionally reconcile final salience-gated goals."""

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
        connection_goal_id = (
            "goal:relationship_connection:"
            f"{updated['state_scope']}:{relationship['relationship_id']}"
        )
        connection_goal = next(
            (
                goal for goal in updated["goals"]
                if goal.get("entity_id") == connection_goal_id
            ),
            None,
        )
        connection_goal_is_eligible = (
            relationship["salience"] >= 40 and closeness_gap >= 40
        )
        if (
            reconcile_salience_gated_goals
            and not connection_goal_is_eligible
        ):
            updated["goals"] = [
                goal
                for goal in updated["goals"]
                if not (
                    isinstance(goal, Mapping)
                    and goal.get("entity_id") == connection_goal_id
                    and goal.get("status") in {"pursuing", "blocked"}
                )
            ]
            connection_goal = None
        if (
            closeness_gap == 0
            and connection_goal is not None
            and connection_goal["status"] in {"pursuing", "blocked"}
        ):
            connection_goal["progress"] = 100
            transitioned_goal = transition_goal(
                connection_goal,
                transition="satisfied",
                evidence={"outcome_kind": "completion"},
            )
            connection_goal.update(transitioned_goal)
            connection_goal["updated_at"] = now
        if connection_goal_is_eligible:
            add_goal(
                "relationship_connection",
                relationship_root,
                connection_value,
                "恢复或加深珍贵的关系连接",
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
                _drive_value(constraints, "autonomy", "importance"),
                "保护当前关系边界",
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
                "保护珍贵关系免受当前威胁",
                threat.get("evidence_refs", []),
                threat_roles,
            )
        if relationship_threat and threat["uncertainty"] >= 40:
            add_goal(
                "trust_verification",
                threat_root,
                max(relationship["attachment"], threat["uncertainty"]),
                "核实关系面临的不确定威胁",
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
                "降低当前威胁并维护安全",
                threat.get("evidence_refs", []),
                threat_roles,
            )
        if (
            _has_other_experiencer(threat, updated)
            and threat["residual_pressure"] >= 40
            and _drive_value(constraints, "care", "importance") >= 40
        ):
            add_goal(
                "social_care",
                threat_root,
                max(
                    _drive_value(constraints, "care", "importance"),
                    threat["residual_pressure"],
                ),
                "在压力下照顾其他体验者",
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
                "消除阻碍重要目标的障碍",
                goal.get("evidence_refs", []),
                goal_roles,
            )
        if goal.get("status") == "failed":
            add_goal(
                "loss_recovery",
                goal_root,
                goal["importance"],
                "从失败的重要目标中恢复",
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
                max(
                    _drive_value(constraints, "autonomy", "importance"),
                    event["identity_threat"],
                    event["unfairness"],
                ),
                "保护自主性并修复被侵犯的边界",
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
                "修复当前角色造成的道德伤害",
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
                "在压力下照顾其他体验者",
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
                "回应其他行动者的正向结果",
                event.get("evidence_refs", []),
                event_roles,
            )
        if event["outcome_impact"] <= -40:
            add_goal(
                "loss_recovery",
                event_root,
                -event["outcome_impact"],
                "从当前负面结果中恢复",
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
                "针对比较差距进行自我提升",
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
                "探索仍开放且可学习的知识缺口",
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
            "在目标感和能动性偏低时重建意义",
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
        if entity.get("status") != "active":
            continue
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

    entity["evidence_refs"] = retain_bounded_evidence(
        entity["evidence_refs"],
        [evidence],
        preserve_primary=True,
    )


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
