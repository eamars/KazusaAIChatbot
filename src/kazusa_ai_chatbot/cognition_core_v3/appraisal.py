"""Canonical appraisal validation and caller-owned state binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CANONICAL_A1_FAMILIES,
    CANONICAL_A2_FAMILIES,
    CANONICAL_FAMILY_AXES,
    CANONICAL_SHIFT_VALUES,
    CanonicalAppraisal,
)
from kazusa_ai_chatbot.cognition_shared.state_models import validate_cognition_state
from kazusa_ai_chatbot.cognition_shared.state_projection import RELATIONSHIP_AXIS_FIELDS
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    create_guarded_goal,
    materialize_causal_root,
)
from kazusa_ai_chatbot.cognition_shared.transition_guards import (
    apply_semantic_deltas,
    retain_bounded_evidence,
)


class AppraisalContractError(ValueError):
    """A mechanically invalid fixed-slot appraisal."""


_SHIFT_DELTAS = {
    "slight_increase": 8,
    "moderate_increase": 20,
    "strong_increase": 40,
    "slight_decrease": -8,
    "moderate_decrease": -20,
    "strong_decrease": -40,
    "stable": 0,
    "uncertain": 0,
}
_EVENT_AXES = frozenset({
    "outcome_impact", "responsibility", "intentionality", "harm", "unfairness",
    "exposure", "repair_need", "reparability", "expectation_mismatch",
    "norm_violation", "contamination_risk", "identity_threat", "comparison_gap",
    "vastness", "memory_warmth", "temporal_loss",
})


def _guarded_delta(value: int, target_path: str) -> int:
    """Keep a shift within the native reducer limit for its target domain."""

    if target_path.startswith(("relationship.", "meaning_state.")):
        limit = 10
    elif target_path.startswith("drives."):
        limit = 40
    else:
        limit = 40
    return max(-limit, min(limit, value))


def _text(value: object, name: str, maximum: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise AppraisalContractError(f"{name} must be bounded non-empty text")
    return value.strip()


def _record_causal_salience(
    state: dict[str, object],
    target_path: str,
    applied_delta: int,
) -> None:
    """Retain applied causal magnitude on native event-like roots."""

    pieces = target_path.split(".")
    if len(pieces) != 3 or pieces[0] not in {
        "active_events", "threats", "knowledge_gaps",
    }:
        return
    rows = state.get(pieces[0])
    if not isinstance(rows, list):
        raise AppraisalContractError("causal target collection is invalid")
    target = next(
        (row for row in rows if isinstance(row, dict) and row.get("entity_id") == pieces[1]),
        None,
    )
    if target is None:
        raise AppraisalContractError("causal target root is missing")
    target["salience"] = max(
        int(target.get("salience", 0)),
        min(100, abs(applied_delta)),
    )


def validate_canonical_appraisal(
    raw: object,
    *,
    families: tuple[str, ...],
) -> tuple[CanonicalAppraisal, ...]:
    if not isinstance(raw, Mapping) or tuple(raw) != families:
        raise AppraisalContractError("appraisal family slots are not exact")
    result: list[CanonicalAppraisal] = []
    for family in families:
        row = raw[family]
        if not isinstance(row, Mapping) or set(row) != {
            "applicable", "semantic_summary", "cause_summary", "axis_changes",
        }:
            raise AppraisalContractError(f"{family} slot is not exact")
        if not isinstance(row["applicable"], bool):
            raise AppraisalContractError(f"{family}.applicable must be boolean")
        axes = row["axis_changes"]
        if not isinstance(axes, list) or len(axes) > len(CANONICAL_FAMILY_AXES[family]):
            raise AppraisalContractError(f"{family}.axis_changes exceeds its domain")
        seen: set[str] = set()
        clean: list[dict[str, object]] = []
        for change in axes:
            if not isinstance(change, Mapping) or set(change) != {"axis", "shift", "reason"}:
                raise AppraisalContractError(f"{family}.axis_changes row is not exact")
            axis = change["axis"]
            if axis not in CANONICAL_FAMILY_AXES[family] or axis in seen:
                raise AppraisalContractError(f"{family} axis is unknown or duplicated")
            if change["shift"] not in CANONICAL_SHIFT_VALUES:
                raise AppraisalContractError(f"{family} shift is unsupported")
            seen.add(str(axis))
            clean.append({
                "axis": axis,
                "shift": change["shift"],
                "reason": _text(change["reason"], f"{family}.{axis}.reason", 500),
            })
        result.append(CanonicalAppraisal(
            family=family,
            applicable=row["applicable"],
            semantic_summary=_text(row["semantic_summary"], f"{family}.semantic_summary"),
            cause_summary=_text(row["cause_summary"], f"{family}.cause_summary"),
            axis_changes=tuple(clean),
        ))
    return tuple(result)


def validate_canonical_appraisal_stage(
    raw: object,
    *,
    stage_name: str,
) -> tuple[CanonicalAppraisal, ...]:
    families = CANONICAL_A1_FAMILIES if stage_name == "A1" else CANONICAL_A2_FAMILIES
    if stage_name not in {"A1", "A2"}:
        raise AppraisalContractError("appraisal stage must be A1 or A2")
    return validate_canonical_appraisal(raw, families=families)


def bind_axis_changes(
    payload: Mapping[str, object],
    appraisals: Sequence[CanonicalAppraisal],
    goal: Mapping[str, object] | None = None,
    willingness: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Bind every accepted axis to an authoritative native reducer target."""

    state = deepcopy(dict(payload["mutable_state"]))
    scope = str(state["state_scope"])
    episode = payload.get("episode")
    episode_mapping = episode if isinstance(episode, Mapping) else {}
    fallback_summary = (
        str(episode_mapping.get("current_request") or "").strip()
        or str(episode_mapping.get("semantic_operation") or "").strip()
        or "current observation"
    )
    fallback_id = str(
        episode_mapping.get("episode_id") or "current_observation"
    ).strip()
    fallback_evidence = {
        "source_kind": "episode",
        "source_id": fallback_id,
        "occurred_at": str(state["updated_at"]),
        "semantic_summary": fallback_summary[:500],
    }
    evidence = next(
        (dict(row["evidence_ref"]) for row in payload.get("evidence", [])
         if isinstance(row, Mapping) and isinstance(row.get("evidence_ref"), Mapping)),
        fallback_evidence,
    )
    evidence.setdefault("occurred_at", str(state["updated_at"]))
    evidence.setdefault("semantic_summary", fallback_summary[:500])
    evidence_id = str(evidence["source_id"])
    timestamp = str(state["updated_at"])
    receipts: list[dict[str, object]] = []
    intended: list[tuple[int, dict[str, object]]] = []
    character_intended: list[tuple[int, dict[str, object]]] = []
    event_axes = set(_EVENT_AXES)
    goal_axes = {"obstruction", "expected_success", "controllability", "recoverability", "urgency"}
    threat_axes = {"likelihood", "expected_harm", "uncertainty", "controllability", "coping_potential", "residual_pressure"}
    gap_axes = {"relevance", "uncertainty", "learnability", "novelty", "model_accommodation"}
    meaning_axes = {"purpose_coherence", "agency", "identity_continuity"}
    drive_axes = {
        "autonomy_pressure": "autonomy", "connection_pressure": "connection",
        "safety_pressure": "safety", "competence_pressure": "competence",
        "care_pressure": "care", "integrity_pressure": "integrity",
        "exploration_pressure": "exploration", "meaning_pressure": "meaning",
    }
    event_causes = {
        item.family: item.cause_summary for item in appraisals if item.cause_summary
    }
    first_cause = next(
        (item.cause_summary for item in appraisals if item.cause_summary),
        fallback_summary,
    )
    event_families = {"event_agency", "moral_identity"}
    epistemic_event_axes = {"comparison_gap", "vastness", "memory_warmth", "temporal_loss"}
    need_event = True
    threat_exclusive_axes = threat_axes - {"controllability"}
    need_threat = any(
        item.family == "goal_threat_outcome" and change["axis"] in threat_exclusive_axes
        for item in appraisals for change in item.axis_changes
    )
    need_gap = any(
        item.family == "epistemic_comparison_memory" and change["axis"] in gap_axes
        for item in appraisals for change in item.axis_changes
    )
    if scope == "user" and isinstance(state.get("relationship"), dict):
        relationship = state["relationship"]
        refs = relationship.setdefault("evidence_refs", [])
        relationship["evidence_refs"] = retain_bounded_evidence(
            refs,
            [evidence],
            preserve_primary=False,
        )
    roots: dict[str, str] = {}
    root_causes: dict[str, str] = {}
    for kind, needed, family in (
        ("event", need_event, "event_agency"),
        ("threat", need_threat, "goal_threat_outcome"),
        ("knowledge_gap", need_gap, "epistemic_comparison_memory"),
    ):
        if needed:
            state, root_id, _created = materialize_causal_root(
                state, kind=kind, primary_evidence=evidence,
                description=event_causes.get(family, first_cause),
                updated_at=timestamp,
            )
            roots[kind] = root_id
            root_causes[kind] = event_causes.get(family, evidence["semantic_summary"])
    event_root = roots.get("event")
    goal_root_ref = None
    if event_root:
        goal_root_ref = {"scope": scope, "kind": "event", "entity_id": event_root}
    goal_value = goal or {
        "intent": "understand the current request",
        "cause_summary": evidence["semantic_summary"],
    }
    goal_row = create_guarded_goal(
        state, goal_kind="ordinary_response", description=str(goal_value["intent"]),
        role_refs=[], evidence_refs=[deepcopy(evidence)], axes={},
        primary_root_ref=goal_root_ref, updated_at=timestamp,
    )
    goal_id = str(goal_row["entity_id"])
    if not any(row.get("entity_id") == goal_id for row in state["goals"]):
        state["goals"].append(goal_row)
    roots["goal"] = goal_id
    root_causes["goal"] = str(goal_value.get("cause_summary", evidence["semantic_summary"]))

    for item in appraisals:
        for change in item.axis_changes:
            axis = str(change["axis"])
            requested = _SHIFT_DELTAS[str(change["shift"])]
            receipt: dict[str, object] = {
                "family": item.family, "axis": axis, "shift": change["shift"],
                "reason": str(change["reason"]), "requested_delta": requested,
                "target_paths": [],
            }
            index = len(receipts)
            receipts.append(receipt)
            if requested == 0:
                receipt["disposition"] = "no_numeric_change"
                continue
            targets: list[str] = []
            if axis in RELATIONSHIP_AXIS_FIELDS:
                if scope == "user":
                    targets = [f"relationship.{axis}"]
                else:
                    receipt["disposition"] = "scope_inapplicable"
            elif item.family == "goal_threat_outcome" and axis in goal_axes:
                targets = [f"goals.{goal_id}.{axis}"]
                if axis == "controllability" and "threat" in roots:
                    targets.append(f"threats.{roots['threat']}.controllability")
            elif item.family == "goal_threat_outcome" and axis in threat_axes:
                targets = [f"threats.{roots['threat']}.{axis}"]
            elif item.family == "goal_threat_outcome" and axis == "outcome_impact":
                targets = [f"active_events.{roots['event']}.{axis}"]
            elif item.family == "epistemic_comparison_memory" and axis in gap_axes:
                targets = [f"knowledge_gaps.{roots['knowledge_gap']}.{axis}"]
            elif (item.family in event_families and axis in event_axes) or axis in epistemic_event_axes:
                targets = [f"active_events.{roots['event']}.{axis}"]
            elif axis in drive_axes:
                if scope == "character":
                    targets = [f"drives.{drive_axes[axis]}.pressure"]
                else:
                    receipt["disposition"] = "scope_inapplicable"
            elif axis in meaning_axes:
                if scope == "character":
                    targets = [f"meaning_state.{axis}"]
                else:
                    receipt["disposition"] = "scope_inapplicable"
            else:
                receipt["disposition"] = "scope_inapplicable"
            receipt["target_paths"] = targets
            if not targets:
                continue
            delta = _guarded_delta(requested, targets[0])
            for target_path in targets:
                proposal = {
                    "target_path": target_path, "delta": delta,
                    "evidence_handles": [evidence_id], "reason": str(change["reason"]),
                }
                (character_intended if target_path.startswith(("drives.", "meaning_state.")) else intended).append((index, proposal))
            if targets and not target_path.startswith(("drives.", "meaning_state.")):
                receipt["disposition"] = "awaiting_reducer"

    if intended:
        result = apply_semantic_deltas(state, [proposal for _index, proposal in intended])
        state = result["updated_state"]
        accepted = {row["target_path"]: row for row in result["accepted_delta_receipts"]}
        for index, proposal in intended:
            receipt = receipts[index]
            applied = accepted.get(proposal["target_path"])
            if applied is None:
                raise AppraisalContractError(f"missing reducer receipt for {proposal['target_path']}")
            receipt.setdefault("applied_targets", []).append(applied)
            _record_causal_salience(
                state,
                proposal["target_path"],
                int(applied["applied_delta"]),
            )
        for receipt in receipts:
            if receipt.get("disposition") == "awaiting_reducer":
                receipt["disposition"] = "applied"
    if character_intended:
        temporary = deepcopy(state)
        for _index, proposal in character_intended:
            if proposal["target_path"].startswith("drives."):
                temporary["drives"][proposal["target_path"].split(".")[1]]["evidence_refs"] = [deepcopy(evidence)]
            else:
                temporary["meaning_state"]["evidence_refs"] = [deepcopy(evidence)]
        result = apply_semantic_deltas(temporary, [proposal for _index, proposal in character_intended])
        state = result["updated_state"]
        for drive in state.get("drives", {}).values():
            drive.pop("evidence_refs", None)
            drive.pop("updated_at", None)
        state.get("meaning_state", {}).pop("evidence_refs", None)
        state.get("meaning_state", {}).pop("updated_at", None)
        accepted = {row["target_path"]: row for row in result["accepted_delta_receipts"]}
        for index, proposal in character_intended:
            applied = accepted.get(proposal["target_path"])
            if applied is None:
                raise AppraisalContractError(f"missing reducer receipt for {proposal['target_path']}")
            receipts[index].setdefault("applied_targets", []).append(applied)
            receipts[index]["disposition"] = "applied"
            _record_causal_salience(
                state,
                proposal["target_path"],
                int(applied["applied_delta"]),
            )
    validated = validate_cognition_state(state)
    for receipt in receipts:
        disposition = receipt.get("disposition")
        if disposition == "awaiting_reducer":
            raise AppraisalContractError("axis receipt remained unresolved")
        if disposition not in {"applied", "no_numeric_change", "scope_inapplicable"}:
            raise AppraisalContractError("axis receipt has unsupported disposition")
    transitions = []
    for kind, root_id in roots.items():
        if kind == "knowledge_gap":
            continue
        if kind not in {"event", "threat", "knowledge_gap"}:
            continue
        row = next((entity for entity in validated[{"event": "active_events", "threat": "threats", "knowledge_gap": "knowledge_gaps"}[kind]] if entity["entity_id"] == root_id), None)
        if row is not None:
            transitions.append({
                "root_ref": {"scope": scope, "kind": kind, "entity_id": root_id},
                "prior": {"status": "active", "residual_pressure": 0},
                "current": {"status": row["status"], "residual_pressure": 0},
                "evidence_ref": deepcopy(evidence), "salience": row["salience"],
            })
    provenance = []
    for item in appraisals:
        if not item.cause_summary:
            continue
        if item.family == "relationship_social" and scope == "user":
            primary = {"scope": scope, "kind": "relationship", "entity_id": validated["relationship"]["relationship_id"]}
            refs = [primary]
        elif item.family == "goal_threat_outcome":
            primary = {"scope": scope, "kind": "goal", "entity_id": roots["goal"]}
            refs = [primary]
            if "threat" in roots:
                refs.append({"scope": scope, "kind": "threat", "entity_id": roots["threat"]})
        elif item.family == "epistemic_comparison_memory":
            if "knowledge_gap" in roots:
                primary = {"scope": scope, "kind": "knowledge_gap", "entity_id": roots["knowledge_gap"]}
            else:
                primary = {"scope": scope, "kind": "goal", "entity_id": roots["goal"]}
            refs = [primary]
            if "event" in roots:
                refs.append({"scope": scope, "kind": "event", "entity_id": roots["event"]})
        elif item.family == "existential_drive" and scope == "character":
            primary = {"scope": scope, "kind": "meaning", "entity_id": "meaning:character"}
            refs = [primary]
            refs.extend({"scope": scope, "kind": "drive", "entity_id": drive} for drive in drive_axes.values() if drive in validated.get("drives", {}))
        else:
            if "event" in roots:
                primary = {"scope": scope, "kind": "event", "entity_id": roots["event"]}
            else:
                primary = {"scope": scope, "kind": "goal", "entity_id": roots["goal"]}
            refs = [primary]
        provenance.append({
            "family": item.family, "cause_summary": item.cause_summary,
            "cause_status": "active", "primary_root": primary, "root_refs": refs,
        })
    return validated, transitions, receipts, provenance


def bind_canonical_appraisal(
    appraisals: Sequence[CanonicalAppraisal],
    *,
    cause_status: str = "active",
) -> dict[str, object]:
    return {
        "appraisals": [
            {
                "family": item.family,
                "applicable": item.applicable,
                "semantic_summary": item.semantic_summary,
                "cause_summary": item.cause_summary,
                "axis_changes": [dict(row) for row in item.axis_changes],
            }
            for item in appraisals
        ],
        "cause_provenance": [
            {
                "family": item.family,
                "cause_summary": item.cause_summary,
                "cause_status": cause_status,
            }
            for item in appraisals if item.cause_summary
        ],
    }


__all__ = [
    "AppraisalContractError",
    "bind_axis_changes",
    "bind_canonical_appraisal",
    "validate_canonical_appraisal",
    "validate_canonical_appraisal_stage",
]
