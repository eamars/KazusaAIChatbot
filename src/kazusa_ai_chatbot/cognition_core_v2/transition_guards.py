"""Deterministic V2 facts, deltas, event comparison, and FSM guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    EVIDENCE_SOURCE_KINDS,
    ENTITY_LIST_FIELDS,
    ENTITY_KINDS,
)


USER_FACT_PRODUCERS = frozenset(
    {"action_result", "resolver_observation", "accepted_task_result"}
)
SCHEDULER_PRODUCER = "scheduler_event"
PROMOTED_SOURCE_PRODUCER = "promoted_source_metadata"
DIRECT_FACT_KINDS = frozenset(
    {
        "goal_progress_observed",
        "goal_completed",
        "goal_terminal_failure",
        "goal_obstruction_removed",
        "threat_resolved",
        "event_repaired",
        "knowledge_answered",
        "deadline_reached",
        "source_occurred",
    }
)
TERMINAL_GOAL_STATUSES = frozenset(
    {"satisfied", "failed", "abandoned"}
)
TERMINAL_ENTITY_STATUSES = frozenset({"resolved", "replaced"})
_EVIDENCE_BY_PRODUCER = {
    "action_result": "action_result",
    "resolver_observation": "resolver_observation",
    "accepted_task_result": "accepted_task_result",
    "scheduler_event": "scheduler_event",
    "promoted_source_metadata": {
        "promoted_memory",
        "promoted_reflection",
        "media_observation",
    },
}


def apply_direct_fact(
    state: Mapping[str, Any],
    fact: Mapping[str, Any],
    *,
    producer: str,
) -> dict[str, Any]:
    """Apply one exact trusted fact while retaining its complete evidence."""

    if producer not in {
        *USER_FACT_PRODUCERS,
        SCHEDULER_PRODUCER,
        PROMOTED_SOURCE_PRODUCER,
    }:
        raise CognitionStateError("direct-fact producer is not trusted")
    if not isinstance(fact, Mapping):
        raise CognitionStateError("direct fact must be a mapping")
    fact_kind = fact.get("fact_kind")
    if fact_kind not in DIRECT_FACT_KINDS:
        raise CognitionStateError("direct fact kind is not allowlisted")
    _validate_fact_keys(fact, fact_kind)
    evidence_ref = _validate_evidence_for_producer(
        fact["evidence_ref"],
        producer,
    )
    if fact_kind == "source_occurred":
        if producer not in {SCHEDULER_PRODUCER, PROMOTED_SOURCE_PRODUCER}:
            raise CognitionStateError("producer cannot emit source occurrence")
    elif producer not in USER_FACT_PRODUCERS:
        if fact_kind != "deadline_reached" or producer != SCHEDULER_PRODUCER:
            raise CognitionStateError("producer cannot emit this direct fact")

    updated_state = deepcopy(dict(state))
    target = _validate_target_ref(
        updated_state,
        fact["target_refs"],
        allow_role=fact_kind == "source_occurred",
    )
    kind = target["kind"]
    entity = _find_entity(updated_state, kind, target["entity_id"])
    _append_evidence(entity, evidence_ref)

    if fact_kind == "source_occurred":
        return updated_state
    if entity["status"] in TERMINAL_GOAL_STATUSES | TERMINAL_ENTITY_STATUSES:
        raise CognitionStateError("terminal target cannot receive a direct fact")

    if fact_kind == "goal_progress_observed":
        _require_kind(kind, "goal", fact_kind)
        entity["progress"] = fact["observed_progress"]
        if entity["progress"] == 100:
            entity["status"] = "satisfied"
    elif fact_kind == "goal_completed":
        _require_kind(kind, "goal", fact_kind)
        entity["progress"] = 100
        entity["status"] = "satisfied"
    elif fact_kind == "goal_terminal_failure":
        _require_kind(kind, "goal", fact_kind)
        entity["recoverability"] = 0
        entity["status"] = "failed"
    elif fact_kind == "goal_obstruction_removed":
        _require_kind(kind, "goal", fact_kind)
        if entity["status"] != "blocked":
            raise CognitionStateError("obstruction removal requires blocked goal")
        entity["obstruction"] = 0
        if entity["recoverability"] >= 25:
            entity["status"] = "pursuing"
    elif fact_kind == "threat_resolved":
        _require_kind(kind, "threat", fact_kind)
        if entity["status"] != "active":
            raise CognitionStateError("threat resolution requires active threat")
        entity["residual_pressure"] = 0
        entity["status"] = "resolved"
    elif fact_kind == "event_repaired":
        _require_kind(kind, "event", fact_kind)
        if entity["status"] != "active":
            raise CognitionStateError("event repair requires active event")
        entity["repair_need"] = 0
        entity["reparability"] = 100
        entity["status"] = "resolved"
    elif fact_kind == "knowledge_answered":
        _require_kind(kind, "knowledge_gap", fact_kind)
        if entity["status"] not in {"open", "reduced"}:
            raise CognitionStateError("knowledge answer requires open gap")
        entity["uncertainty"] = 0
        entity["status"] = "resolved"
    elif fact_kind == "deadline_reached":
        _require_kind(kind, "goal", fact_kind)
        entity["urgency"] = 100
    return updated_state


def apply_semantic_deltas(
    state: Mapping[str, Any],
    deltas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply unique bounded deltas; duplicate targets invalidate only themselves."""

    normalized: list[tuple[str, int, list[str], str]] = []
    target_counts: dict[str, int] = {}
    for proposal in deltas:
        if not isinstance(proposal, Mapping):
            raise CognitionStateError("semantic delta must be a mapping")
        required = {"target_path", "delta", "evidence_handles", "reason"}
        if set(proposal) != required:
            raise CognitionStateError("semantic delta fields are not exact")
        path = proposal["target_path"]
        delta = proposal["delta"]
        handles = proposal["evidence_handles"]
        reason = proposal["reason"]
        if not isinstance(path, str) or not path:
            raise CognitionStateError("semantic delta target is invalid")
        _require_integer(delta, -40, 40, "delta")
        if (
            not isinstance(handles, list)
            or not 1 <= len(handles) <= 8
            or any(not isinstance(handle, str) or not handle for handle in handles)
        ):
            raise CognitionStateError("semantic delta evidence handles are invalid")
        if not isinstance(reason, str) or not 1 <= len(reason) <= 500:
            raise CognitionStateError("semantic delta reason is invalid")
        target_counts[path] = target_counts.get(path, 0) + 1
        normalized.append((path, delta, list(handles), reason))

    updated_state = deepcopy(dict(state))
    for path, delta, handles, reason in sorted(normalized):
        if target_counts[path] != 1:
            continue
        target = _apply_delta_path(updated_state, path, delta)
        _retain_delta_evidence(target, handles)
        target["updated_at"] = updated_state["updated_at"]
    return updated_state


def compare_event(
    current_event: Mapping[str, Any],
    stored_event: Mapping[str, Any] | None,
    accepted_deltas: Mapping[str, int],
) -> str:
    """Classify an event by canonical refs, axes, evidence, and outcome."""

    if stored_event is None:
        if accepted_deltas and _event_salience(current_event) >= 25:
            return "create"
        return "unrelated"
    if not _same_event_refs(current_event, stored_event):
        return "unrelated"
    if _has_typed_outcome(current_event, {"completion", "repair", "safety"}):
        return "resolve"
    if current_event.get("supersedes_ref"):
        return "replace"
    if not accepted_deltas:
        return "unrelated"
    if _same_direction(current_event, accepted_deltas):
        return "reinforce"
    return "contradict"


def transition_goal(
    goal: Mapping[str, Any],
    *,
    transition: str,
    evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    explicit_release: bool = False,
    superseding_goal_validated: bool = False,
) -> dict[str, Any]:
    """Apply one guarded goal FSM transition to a copied goal."""

    updated_goal = deepcopy(dict(goal))
    status = updated_goal["status"]
    if status in TERMINAL_GOAL_STATUSES:
        raise CognitionStateError("terminal goal cannot transition")
    if transition == "blocked":
        if updated_goal["obstruction"] < 40:
            raise CognitionStateError("goal obstruction is below block threshold")
    elif transition == "pursuing":
        if status != "blocked" or updated_goal["obstruction"] >= 25:
            raise CognitionStateError("goal recovery guard failed")
        if updated_goal["recoverability"] < 25:
            raise CognitionStateError("goal recovery is not viable")
    elif transition == "satisfied":
        if updated_goal["progress"] != 100 or not _has_typed_outcome(
            evidence,
            {"completion"},
        ):
            raise CognitionStateError("goal completion evidence is required")
    elif transition == "failed":
        if updated_goal["recoverability"] >= 25 or not _has_typed_outcome(
            evidence,
            {"adverse", "failure", "obstruction"},
        ):
            raise CognitionStateError("goal failure guard failed")
    elif transition == "abandoned":
        if not explicit_release and not superseding_goal_validated:
            raise CognitionStateError("goal abandonment guard failed")
    else:
        raise CognitionStateError("unknown goal transition")
    updated_goal["status"] = transition
    return updated_goal


def transition_threat(
    threat: Mapping[str, Any],
    *,
    transition: str,
    evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    successor_ref: str | None = None,
) -> dict[str, Any]:
    """Apply one guarded threat FSM transition."""

    updated_threat = deepcopy(dict(threat))
    if updated_threat["status"] != "active":
        raise CognitionStateError("terminal threat cannot transition")
    if transition == "resolved":
        if updated_threat["residual_pressure"] > 20:
            raise CognitionStateError("threat residual pressure remains high")
        if not _has_typed_outcome(evidence, {"resolve", "safety"}):
            raise CognitionStateError("threat resolution evidence is required")
    elif transition == "replaced":
        if not successor_ref:
            raise CognitionStateError("replaced threat requires successor")
    else:
        raise CognitionStateError("unknown threat transition")
    updated_threat["status"] = transition
    return updated_threat


def transition_event(
    event: Mapping[str, Any],
    *,
    transition: str,
    evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    successor_ref: str | None = None,
) -> dict[str, Any]:
    """Apply one guarded causal-event FSM transition."""

    updated_event = deepcopy(dict(event))
    if updated_event["status"] != "active":
        raise CognitionStateError("terminal event cannot transition")
    if transition == "resolved":
        if updated_event["repair_need"] > 0:
            raise CognitionStateError("event repair is incomplete")
        if not _has_typed_outcome(evidence, {"completion", "repair", "safety"}):
            raise CognitionStateError("event resolution evidence is required")
    elif transition == "replaced":
        if not successor_ref:
            raise CognitionStateError("replaced event requires successor")
    else:
        raise CognitionStateError("unknown event transition")
    updated_event["status"] = transition
    return updated_event


def transition_knowledge_gap(
    gap: Mapping[str, Any],
    *,
    transition: str,
    previous_uncertainty: int | None = None,
    accepted_uncertainty_decrease: int | None = None,
    evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply one guarded knowledge-gap transition."""

    updated_gap = deepcopy(dict(gap))
    status = updated_gap["status"]
    if status == "resolved":
        raise CognitionStateError("resolved knowledge gap cannot transition")
    decrease = accepted_uncertainty_decrease
    if decrease is None and previous_uncertainty is not None:
        decrease = previous_uncertainty - updated_gap["uncertainty"]
    if transition == "reduced":
        if decrease is None or decrease < 20:
            raise CognitionStateError(
                "knowledge reduction requires a 20-point decrease"
            )
        if updated_gap["uncertainty"] <= 0:
            raise CognitionStateError("resolved gap requires answer evidence")
    elif transition == "resolved":
        if updated_gap["uncertainty"] != 0:
            raise CognitionStateError("knowledge resolution guard failed")
        if not _has_typed_outcome(evidence, {"answer", "completion"}):
            raise CognitionStateError("knowledge answer evidence is required")
    else:
        raise CognitionStateError("unknown knowledge-gap transition")
    updated_gap["status"] = transition
    return updated_gap


def _validate_fact_keys(fact: Mapping[str, Any], fact_kind: str) -> None:
    """Reject fields outside the fixed direct-fact envelope."""

    required = {"fact_id", "fact_kind", "target_refs", "evidence_ref"}
    if fact_kind == "goal_progress_observed":
        required.add("observed_progress")
    if set(fact) != required:
        raise CognitionStateError("direct fact contains unexpected fields")
    if not isinstance(fact["fact_id"], str) or not fact["fact_id"].strip():
        raise CognitionStateError("direct fact id is invalid")
    if fact_kind == "goal_progress_observed":
        _require_integer(fact["observed_progress"], 0, 100, "observed_progress")


def _validate_evidence_for_producer(
    value: Any,
    producer: str,
) -> dict[str, Any]:
    """Validate and copy evidence while checking producer provenance."""

    if not isinstance(value, Mapping):
        raise CognitionStateError("direct fact evidence must be structured")
    required = {"source_kind", "source_id", "occurred_at", "semantic_summary"}
    if set(value) != required:
        raise CognitionStateError("direct fact evidence fields are not exact")
    source_kind = value["source_kind"]
    expected = _EVIDENCE_BY_PRODUCER[producer]
    if source_kind not in EVIDENCE_SOURCE_KINDS or (
        source_kind != expected
        if isinstance(expected, str)
        else source_kind not in expected
    ):
        raise CognitionStateError("evidence source does not match producer")
    if not isinstance(value["source_id"], str) or not value["source_id"].strip():
        raise CognitionStateError("evidence source_id is invalid")
    if (
        not isinstance(value["occurred_at"], str)
        or not value["occurred_at"].endswith("Z")
    ):
        raise CognitionStateError("evidence occurred_at is invalid")
    try:
        from datetime import datetime

        datetime.fromisoformat(value["occurred_at"][:-1] + "+00:00")
    except (TypeError, ValueError) as exc:
        raise CognitionStateError("evidence occurred_at is invalid") from exc
    summary = value["semantic_summary"]
    if not isinstance(summary, str) or not 1 <= len(summary) <= 500:
        raise CognitionStateError("evidence semantic_summary is invalid")
    return deepcopy(dict(value))


def _validate_target_ref(
    state: Mapping[str, Any],
    target_refs: Any,
    *,
    allow_role: bool = False,
) -> dict[str, str]:
    """Validate one canonical target and enforce mutable scope ownership."""

    if not isinstance(target_refs, list) or len(target_refs) != 1:
        raise CognitionStateError("direct fact requires one target reference")
    target = target_refs[0]
    if not isinstance(target, Mapping):
        raise CognitionStateError("direct fact target must be structured")
    if set(target) == {"scope", "kind", "entity_id"}:
        if target["scope"] != state["state_scope"]:
            raise CognitionStateError("direct fact target scope is not mutable")
        if target["kind"] not in ENTITY_KINDS:
            raise CognitionStateError("direct fact target kind is not canonical")
        if not isinstance(target["entity_id"], str) or not target["entity_id"].strip():
            raise CognitionStateError("direct fact target id is invalid")
        if target["kind"] not in ENTITY_LIST_FIELDS:
            raise CognitionStateError("direct fact target kind is not fact-addressable")
        _find_entity(state, target["kind"], target["entity_id"])
        return {
            "scope": target["scope"],
            "kind": target["kind"],
            "entity_id": target["entity_id"],
        }
    if not allow_role or set(target) != {"role", "entity_kind", "entity_id"}:
        raise CognitionStateError("direct fact target fields are not exact")
    if target["role"] not in {
        "actor",
        "experiencer",
        "target",
        "object",
        "affected_goal",
        "affected_relationship",
    }:
        raise CognitionStateError("direct fact target role is invalid")
    if target["entity_kind"] not in {
        "character",
        "user",
        "group",
        "third_party",
        "goal",
        "relationship",
        "standard",
        "object",
    }:
        raise CognitionStateError("direct fact target entity kind is invalid")
    if not isinstance(target["entity_id"], str) or not target["entity_id"].strip():
        raise CognitionStateError("direct fact target id is invalid")
    matches = []
    for kind, field_name in ENTITY_LIST_FIELDS.items():
        for entity in state[field_name]:
            if any(
                isinstance(role_ref, Mapping)
                and role_ref == target
                for role_ref in entity.get("role_refs", [])
            ):
                matches.append((kind, entity))
    if len(matches) != 1:
        raise CognitionStateError("direct fact role target is not uniquely addressable")
    kind, entity = matches[0]
    return {
        "scope": state["state_scope"],
        "kind": kind,
        "entity_id": entity["entity_id"],
    }


def _append_evidence(entity: dict[str, Any], evidence_ref: Mapping[str, Any]) -> None:
    """Append a complete evidence record without lossy tokenization."""

    refs = entity["evidence_refs"]
    if not isinstance(refs, list) or len(refs) >= 8:
        if isinstance(refs, list) and evidence_ref in refs:
            return
        raise CognitionStateError("evidence reference cap reached")
    if evidence_ref not in refs:
        refs.append(deepcopy(dict(evidence_ref)))


def _find_entity(
    state: Mapping[str, Any],
    kind: str,
    entity_id: str,
) -> dict[str, Any]:
    """Return one mutable entity from a canonical state list."""

    field_name = ENTITY_LIST_FIELDS.get(kind)
    if field_name is None:
        raise CognitionStateError("target kind is not an entity list")
    entities = state[field_name]
    if not isinstance(entities, list):
        raise CognitionStateError("target entity list is invalid")
    for entity in entities:
        if isinstance(entity, dict) and entity["entity_id"] == entity_id:
            return entity
    raise CognitionStateError("direct fact target does not exist")


def _apply_delta_path(
    state: dict[str, Any],
    path: str,
    delta: int,
) -> dict[str, Any]:
    """Apply one exact path and return the changed target record."""

    pieces = path.split(".")
    if len(pieces) == 2 and pieces[0] == "relationship":
        target = state.get("relationship")
        if not isinstance(target, dict) or pieces[1] not in {
            "positive_regard",
            "trust",
            "attachment",
            "desired_closeness",
            "perceived_closeness",
            "care",
            "boundary_safety",
            "exclusivity",
            "unresolved_injury",
        }:
            raise CognitionStateError("semantic relationship path is invalid")
        _set_bounded_value(target, pieces[1], delta, 10)
        return target
    if len(pieces) == 2 and pieces[0] == "meaning_state":
        target = state.get("meaning_state")
        if not isinstance(target, dict) or pieces[1] not in {
            "purpose_coherence",
            "agency",
            "identity_continuity",
        }:
            raise CognitionStateError("semantic meaning path is invalid")
        _set_bounded_value(target, pieces[1], delta, 10)
        return target
    if len(pieces) == 3 and pieces[0] == "drives":
        drives = state.get("drives")
        if (
            not isinstance(drives, dict)
            or pieces[1] not in drives
            or pieces[2] != "pressure"
        ):
            raise CognitionStateError("semantic drive path is invalid")
        target = drives[pieces[1]]
        if not isinstance(target, dict):
            raise CognitionStateError("semantic drive target is invalid")
        _set_bounded_value(target, "pressure", delta, 40)
        return target
    if len(pieces) != 3 or pieces[0] not in ENTITY_LIST_FIELDS.values():
        raise CognitionStateError("semantic delta path is not allowlisted")
    kind = {
        "goals": "goal",
        "threats": "threat",
        "active_events": "event",
        "knowledge_gaps": "knowledge_gap",
    }[pieces[0]]
    target = _find_entity(state, kind, pieces[1])
    allowed_axes = {
        "goals": {
            "obstruction",
            "expected_success",
            "controllability",
            "recoverability",
            "urgency",
        },
        "threats": {
            "likelihood",
            "expected_harm",
            "uncertainty",
            "controllability",
            "coping_potential",
            "residual_pressure",
        },
        "active_events": {
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
        },
        "knowledge_gaps": {
            "relevance",
            "uncertainty",
            "learnability",
            "novelty",
            "model_accommodation",
        },
    }[pieces[0]]
    if pieces[2] not in allowed_axes:
        raise CognitionStateError("semantic entity axis is not allowlisted")
    _set_bounded_value(target, pieces[2], delta, 40)
    return target


def _retain_delta_evidence(target: dict[str, Any], handles: Sequence[str]) -> None:
    """Require every delta handle to already exist on its target."""

    evidence_refs = target.get("evidence_refs")
    if not isinstance(evidence_refs, list):
        raise CognitionStateError("delta target evidence is invalid")
    source_ids = {
        ref.get("source_id")
        for ref in evidence_refs
        if isinstance(ref, Mapping)
    }
    if not set(handles).issubset(source_ids):
        raise CognitionStateError("semantic delta evidence handle is unknown")


def _set_bounded_value(
    target: dict[str, Any],
    axis: str,
    delta: int,
    per_event_limit: int,
) -> None:
    """Apply one signed or unsigned bounded change."""

    _require_integer(delta, -per_event_limit, per_event_limit, axis)
    current = target[axis]
    if isinstance(current, bool) or not isinstance(current, int):
        raise CognitionStateError(f"semantic target {axis} is not an integer")
    minimum = -100 if axis in {
        "positive_regard",
        "trust",
        "boundary_safety",
        "outcome_impact",
    } else 0
    target[axis] = max(minimum, min(100, current + delta))


def _event_salience(event: Mapping[str, Any]) -> int:
    """Read the required event salience axis."""

    value = event.get("salience")
    return value if isinstance(value, int) else 0


def _same_event_refs(
    current_event: Mapping[str, Any],
    stored_event: Mapping[str, Any],
) -> bool:
    """Return whether canonical role/affected refs identify one event."""

    current_roles = _event_role_signature(current_event)
    stored_roles = _event_role_signature(stored_event)
    return bool(current_roles) and current_roles == stored_roles


def _event_role_signature(event: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    """Normalize event role refs for repeated-incident matching."""

    roles = event.get("role_refs", [])
    if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
        return ()
    normalized = {
        (
            str(role.get("role")),
            str(role.get("entity_kind")),
            str(role.get("entity_id")),
        )
        for role in roles
        if isinstance(role, Mapping)
    }
    return tuple(sorted(normalized))


def _same_direction(
    current_event: Mapping[str, Any],
    accepted_deltas: Mapping[str, int],
) -> bool:
    """Return whether accepted axes reinforce current event directions."""

    source_deltas = current_event.get("axis_deltas")
    if not isinstance(source_deltas, Mapping):
        return False
    for axis, delta in accepted_deltas.items():
        source_delta = source_deltas.get(axis)
        if not isinstance(source_delta, int):
            return False
        if source_delta == 0 or delta == 0 or (source_delta > 0) != (delta > 0):
            return False
    return True


def _has_typed_outcome(
    evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    markers: set[str],
) -> bool:
    """Find explicit typed completion, repair, safety, or outcome evidence."""

    if isinstance(evidence, Mapping):
        rows = [evidence]
    elif isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        rows = [row for row in evidence if isinstance(row, Mapping)]
    else:
        rows = []
    for row in rows:
        fact_kind = row.get("fact_kind")
        outcome_kind = row.get("outcome_kind")
        semantic_summary = row.get("semantic_summary", "")
        values = {fact_kind, outcome_kind}
        if isinstance(semantic_summary, str):
            values.update(semantic_summary.lower().split())
        if markers.intersection(values):
            return True
        if fact_kind in {
            "goal_completed",
            "event_repaired",
            "knowledge_answered",
            "threat_resolved",
        }:
            return True
    return False


def _require_kind(kind: Any, expected: str, fact_kind: str) -> None:
    """Require a direct fact to target its declared canonical kind."""

    if kind != expected:
        raise CognitionStateError(
            f"{fact_kind} requires target kind {expected!r}"
        )


def _require_integer(
    value: Any,
    minimum: int,
    maximum: int,
    label: str,
) -> None:
    """Require an integer scalar inside an inclusive range."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise CognitionStateError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise CognitionStateError(f"{label} is outside its range")
