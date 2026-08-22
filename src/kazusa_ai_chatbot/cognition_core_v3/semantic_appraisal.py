"""Canonical V3 semantic-appraisal validation and reduction owners."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from kazusa_ai_chatbot.cognition_shared.contracts import (
    ROLE_ENTITY_KINDS,
    SemanticAppraisalResultV2,
    SemanticQuestionV2,
)
from kazusa_ai_chatbot.cognition_core_v3.semantic_source_planner import (
    question_proposition_kinds,
)

MAX_APPRAISAL_OBJECT_HANDLES = 8

MAX_APPRAISAL_SEMANTIC_TEXT_CHARS = 200

MAX_APPRAISAL_DELTA_REASON_CHARS = 300

MAX_ERROR_ALLOWLIST_ITEMS = 40

DELTA_LIMIT_NARROW = 10

DELTA_LIMIT_WIDE = 40

_DELTA_LIMIT_BY_STATE_FIELD = {
    "relationship": DELTA_LIMIT_NARROW,
    "meaning_state": DELTA_LIMIT_NARROW,
}

_SEMANTIC_APPRAISAL_RESULT_FIELDS = {
    "question_id",
    "selected_evidence_handles",
    "selected_role_handles",
    "propositions",
    "deltas",
    "explanation",
}

_PROPOSITION_SUBJECT_KINDS = {
    "goal_release": "goal",
    "goal_supersession": "goal",
    "goal_completed": "goal",
    "event_completed": "event",
    "threat_resolved": "threat",
    "event_repaired": "event",
    "knowledge_answered": "knowledge_gap",
}

_PROPOSITION_SUBJECT_KIND_SETS = {
    "outcome_pending": frozenset({
        "goal",
        "event",
        "threat",
        "knowledge_gap",
    }),
}

_CANDIDATE_ORIGIN_MISSING = "candidate_origin_missing"

_PRODUCER_HANDLE_DOMAIN_INVALID = "producer_handle_domain_invalid"

_SEMANTIC_BOUNDARY_TERMINAL = "semantic_boundary_terminal"

class _SemanticBoundaryValidationError(ValueError):
    """Carry one typed semantic-appraisal boundary disposition."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str,
        field_path: str | None = None,
    ) -> None:
        """Attach the validator-owned failure kind and field path."""

        super().__init__(message)
        self.failure_kind = failure_kind
        self.field_path = field_path

def _boundary_validation_error(
    message: str,
    *,
    failure_kind: str,
    field_path: str | None = None,
) -> _SemanticBoundaryValidationError:
    """Build one typed terminal boundary error."""

    return _SemanticBoundaryValidationError(
        message,
        failure_kind=failure_kind,
        field_path=field_path,
    )

def _terminal_boundary_error(
    message: str,
    *,
    field_path: str | None = None,
) -> _SemanticBoundaryValidationError:
    """Build one known semantic boundary rejection."""

    return _boundary_validation_error(
        message,
        failure_kind=_SEMANTIC_BOUNDARY_TERMINAL,
        field_path=field_path,
    )

def _delta_limit_for_state_field(state_field: str) -> int:
    """Return the reducer's per-event delta bound for one state field."""

    return _DELTA_LIMIT_BY_STATE_FIELD.get(state_field, DELTA_LIMIT_WIDE)

def merge_semantic_appraisal_item(
    accepted_result: SemanticAppraisalResultV2 | None,
    item_result: SemanticAppraisalResultV2,
) -> SemanticAppraisalResultV2:
    """Merge one validated item into the bounded family result."""

    if accepted_result is None:
        merged_result = deepcopy(item_result)
        return merged_result
    prior_signatures = set(_emitted_proposition_signatures(accepted_result))
    item_signatures = _emitted_proposition_signatures(item_result)
    if any(signature in prior_signatures for signature in item_signatures):
        raise _terminal_boundary_error(
            "semantic appraisal proposition is duplicated",
            field_path="propositions",
        )
    return {
        "question_id": accepted_result["question_id"],
        "selected_evidence_handles": _ordered_handle_union(
            accepted_result["selected_evidence_handles"],
            item_result["selected_evidence_handles"],
        ),
        "selected_role_handles": _ordered_handle_union(
            accepted_result["selected_role_handles"],
            item_result["selected_role_handles"],
        ),
        "propositions": [
            *deepcopy(accepted_result["propositions"]),
            *deepcopy(item_result["propositions"]),
        ],
        "deltas": [
            *deepcopy(accepted_result["deltas"]),
            *deepcopy(item_result["deltas"]),
        ],
        "explanation": (
            f"{accepted_result['explanation']} {item_result['explanation']}"
        ),
    }

def _ordered_handle_union(
    first: Sequence[str],
    second: Sequence[str],
) -> list[str]:
    """Return one stable duplicate-free handle union."""

    return list(dict.fromkeys([*first, *second]))

def _emitted_proposition_signatures(
    result: SemanticAppraisalResultV2 | None,
) -> list[str]:
    """Project emitted proposition identities for bounded loop exclusion."""

    if result is None:
        return []
    return [
        "|".join((
            proposition["proposition_kind"],
            proposition["subject_handle"],
            proposition.get("object_handle", ""),
        ))
        for proposition in result["propositions"]
    ]

def validate_semantic_appraisal_result(
    parsed: object,
    question: SemanticQuestionV2,
    evidence_handles: set[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    *,
    maximum_propositions: int = 8,
    maximum_deltas: int = 8,
    maximum_explanation_chars: int = 1000,
) -> SemanticAppraisalResultV2:
    """Validate one appraisal without interpreting its semantic prose."""

    validated_result = _validate_semantic_appraisal_contract(
        parsed,
        question,
        evidence_handles,
        handle_to_ref,
        maximum_propositions=maximum_propositions,
        maximum_deltas=maximum_deltas,
        maximum_explanation_chars=maximum_explanation_chars,
        enforce_semantic_ownership=True,
    )
    return validated_result

def _validate_semantic_appraisal_contract(
    parsed: object,
    question: SemanticQuestionV2,
    evidence_handles: set[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    *,
    maximum_propositions: int,
    maximum_deltas: int,
    maximum_explanation_chars: int,
    enforce_semantic_ownership: bool,
) -> SemanticAppraisalResultV2:
    """Validate shared carriers and optionally strict semantic ownership."""

    _validate_question_handle_authority(question, handle_to_ref)
    if not isinstance(parsed, Mapping):
        raise ValueError("semantic appraisal must return an object")
    if set(parsed) != _SEMANTIC_APPRAISAL_RESULT_FIELDS:
        raise ValueError("semantic appraisal fields are not exact")
    if parsed["question_id"] != question["question_id"]:
        raise ValueError("semantic appraisal question id does not match")
    if (
        not isinstance(parsed["propositions"], list)
        or len(parsed["propositions"]) > maximum_propositions
    ):
        raise ValueError("semantic propositions are invalid")
    if (
        not isinstance(parsed["deltas"], list)
        or len(parsed["deltas"]) > maximum_deltas
    ):
        raise ValueError("semantic deltas are invalid")
    selected_evidence = _validate_handles(
        parsed["selected_evidence_handles"],
        evidence_handles,
        "selected evidence",
        minimum=0,
        maximum=len(evidence_handles),
        failure_kind=_PRODUCER_HANDLE_DOMAIN_INVALID,
        field_path="selected_evidence_handles",
    )
    selected_evidence_set = set(selected_evidence)
    propositions = [
        _validate_proposition(
            row,
            question,
            selected_evidence_set,
            handle_to_ref,
            enforce_semantic_ownership=enforce_semantic_ownership,
        )
        for row in parsed["propositions"]
    ]
    deltas = [
        _validate_delta(
            row,
            question,
            selected_evidence_set,
            handle_to_ref,
        )
        for row in parsed["deltas"]
    ]
    selected_roles = _validate_handles(
        parsed["selected_role_handles"],
        set(question["permitted_role_handles"])
        | set(question["permitted_role_assignment_handles"]),
        "selected roles",
        minimum=0,
        maximum=(
            len(question["permitted_role_handles"])
            + len(question["permitted_role_assignment_handles"])
        ),
        failure_kind=_PRODUCER_HANDLE_DOMAIN_INVALID,
        field_path="selected_role_handles",
    )
    referenced_role_handles = _referenced_role_handles(
        propositions,
        deltas,
    )
    if set(selected_roles) != referenced_role_handles:
        raise _terminal_boundary_error(
            "selected roles must match the handles referenced by valid "
            "subject, object, assignment, and delta fields",
            field_path="selected_role_handles",
        )
    paths = [delta["target_path"] for delta in deltas]
    if len(paths) != len(set(paths)):
        raise _terminal_boundary_error(
            "one appraisal cannot duplicate a target path",
            field_path="deltas[*].target_path",
        )
    explanation = parsed["explanation"]
    if (
        not isinstance(explanation, str)
        or not 1 <= len(explanation) <= maximum_explanation_chars
    ):
        raise ValueError("semantic appraisal explanation is invalid")
    return {
        "question_id": question["question_id"],
        "selected_evidence_handles": selected_evidence,
        "selected_role_handles": selected_roles,
        "propositions": propositions,
        "deltas": deltas,
        "explanation": explanation,
    }

def _validate_proposition(
    value: Any,
    question: SemanticQuestionV2,
    evidence_handles: set[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    *,
    enforce_semantic_ownership: bool,
) -> dict[str, Any]:
    """Validate one semantic proposition and its role assignments."""

    if not isinstance(value, Mapping):
        raise ValueError("semantic proposition must be an object")
    allowed = {
        "proposition_kind",
        "subject_handle",
        "evidence_handles",
        "role_assignments",
        "semantic_value",
    }
    if "object_handle" in value:
        allowed.add("object_handle")
    if set(value) != allowed:
        raise ValueError("semantic proposition fields are not exact")
    proposition_kind = value["proposition_kind"]
    subject = value["subject_handle"]
    if subject not in set(question["permitted_role_handles"]):
        raise _boundary_validation_error(
            "semantic proposition subject handle "
            f"{subject!r} is not permitted; allowed role handles: "
            f"{_allowlist_hint(question['permitted_role_handles'])}",
            failure_kind=_PRODUCER_HANDLE_DOMAIN_INVALID,
            field_path="proposition.subject_handle",
        )
    if "object_handle" in value and value["object_handle"] not in set(
        question["permitted_role_handles"]
    ):
        raise _boundary_validation_error(
            "semantic proposition object handle "
            f"{value['object_handle']!r} is not permitted; allowed role "
            f"handles: {_allowlist_hint(question['permitted_role_handles'])}",
            failure_kind=_PRODUCER_HANDLE_DOMAIN_INVALID,
            field_path="proposition.object_handle",
        )
    assignments = value["role_assignments"]
    if not isinstance(assignments, list) or len(assignments) > 8:
        raise ValueError("semantic proposition roles are invalid")
    normalized_assignments: list[dict[str, str]] = []
    for assignment in assignments:
        if not isinstance(assignment, Mapping) or set(assignment) != {
            "role",
            "entity_handle",
        }:
            raise ValueError("semantic role assignment is invalid")
        if assignment["role"] not in {
            "actor",
            "experiencer",
            "target",
            "object",
            "affected_goal",
            "affected_relationship",
        }:
            raise _terminal_boundary_error(
                "semantic role value is invalid",
                field_path="proposition.role_assignments[*].role",
            )
        if assignment["entity_handle"] not in set(
            question["permitted_role_assignment_handles"]
        ):
            permitted_handles = sorted(
                set(question["permitted_role_assignment_handles"])
            )
            raise _boundary_validation_error(
                "role_assignments[*].entity_handle must be one of "
                + json.dumps(permitted_handles),
                failure_kind=_PRODUCER_HANDLE_DOMAIN_INVALID,
                field_path="proposition.role_assignments[*].entity_handle",
            )
        normalized_assignments.append(dict(assignment))
    referenced_handles = [subject]
    if "object_handle" in value:
        referenced_handles.append(value["object_handle"])
    referenced_handles.extend(
        assignment["entity_handle"]
        for assignment in normalized_assignments
    )
    raw_evidence_handles = value["evidence_handles"]
    if isinstance(raw_evidence_handles, list):
        _validate_candidate_evidence_binding(
            referenced_handles,
            raw_evidence_handles,
            handle_to_ref,
            field_path="proposition.evidence_handles",
        )
    if enforce_semantic_ownership:
        permitted_kinds = question_proposition_kinds(
            question["question_kind"]
        )
        if proposition_kind not in permitted_kinds:
            raise _terminal_boundary_error(
                "semantic proposition kind "
                f"{proposition_kind!r} is not owned by question; "
                f"permitted kinds: {json.dumps(permitted_kinds)}",
                field_path="proposition.proposition_kind",
            )
        required_subject_kind = _PROPOSITION_SUBJECT_KINDS.get(
            proposition_kind
        )
        if (
            required_subject_kind is not None
            and handle_to_ref[subject]["kind"] != required_subject_kind
        ):
            raise _terminal_boundary_error(
                "semantic proposition kind requires subject kind "
                f"{required_subject_kind!r}; received "
                f"{handle_to_ref[subject]['kind']!r}",
                field_path="proposition.subject_handle",
            )
        permitted_subject_kinds = _PROPOSITION_SUBJECT_KIND_SETS.get(
            proposition_kind
        )
        if (
            permitted_subject_kinds is not None
            and handle_to_ref[subject]["kind"]
            not in permitted_subject_kinds
        ):
            raise _terminal_boundary_error(
                "semantic proposition subject kind "
                f"{handle_to_ref[subject]['kind']!r} is not permitted for "
                f"{proposition_kind!r}; permitted kinds: "
                f"{json.dumps(sorted(permitted_subject_kinds))}",
                field_path="proposition.subject_handle",
            )
        if proposition_kind == "goal_supersession":
            if "object_handle" not in value:
                raise _terminal_boundary_error(
                    "goal supersession requires an object handle",
                    field_path="proposition.object_handle",
                )
            if (
                not subject.startswith("g")
                or not value["object_handle"].startswith("g")
            ):
                raise _terminal_boundary_error(
                    "goal supersession requires two goal handles",
                    field_path="proposition.subject_handle",
                )
            if subject == value["object_handle"]:
                raise _terminal_boundary_error(
                    "goal supersession requires a distinct goal",
                    field_path="proposition.object_handle",
                )
    cited = _validate_handles(
        raw_evidence_handles,
        evidence_handles,
        "proposition evidence",
        failure_kind=_PRODUCER_HANDLE_DOMAIN_INVALID,
        field_path="proposition.evidence_handles",
    )
    result = {
        "proposition_kind": proposition_kind,
        "subject_handle": subject,
        "evidence_handles": cited,
        "role_assignments": normalized_assignments,
        "semantic_value": _require_text(
            value.get("semantic_value"),
            "semantic_value",
        ),
    }
    if "object_handle" in value:
        result["object_handle"] = value["object_handle"]
    return result

def _validate_delta(
    value: Any,
    question: SemanticQuestionV2,
    evidence_handles: set[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Validate one allowlisted semantic numeric delta."""

    if not isinstance(value, Mapping) or set(value) != {
        "target_path",
        "delta",
        "evidence_handles",
        "reason",
    }:
        raise ValueError("semantic delta fields are not exact")
    path = value["target_path"]
    if path not in set(question["permitted_delta_paths"]):
        raise _terminal_boundary_error(
            f"semantic delta path {path!r} is not owned by question; "
            f"permitted paths: "
            f"{_allowlist_hint(question['permitted_delta_paths'])}",
            field_path="delta.target_path",
        )
    delta = value["delta"]
    delta_limit = _delta_limit_for_state_field(path.split(".")[0])
    if (
        isinstance(delta, bool)
        or not isinstance(delta, int)
        or not -delta_limit <= delta <= delta_limit
    ):
        raise _terminal_boundary_error(
            "semantic delta must be a JSON integer from "
            f"{-delta_limit} through {delta_limit}; "
            f"received {type(delta).__name__}",
            field_path="delta.delta",
        )
    path_handle = path.split(".")[1]
    raw_evidence_handles = value["evidence_handles"]
    if isinstance(raw_evidence_handles, list):
        _validate_candidate_evidence_binding(
            [path_handle],
            raw_evidence_handles,
            handle_to_ref,
            field_path="delta.evidence_handles",
        )
    cited = _validate_handles(
        raw_evidence_handles,
        evidence_handles,
        "delta evidence",
        failure_kind=_PRODUCER_HANDLE_DOMAIN_INVALID,
        field_path="delta.evidence_handles",
    )
    return {
        "target_path": path,
        "delta": delta,
        "evidence_handles": cited,
        "reason": _require_text(
            value["reason"],
            "reason",
            maximum=MAX_APPRAISAL_DELTA_REASON_CHARS,
        ),
    }

def _allowlist_hint(values: Sequence[str]) -> str:
    """Render a bounded sorted allowlist for one contract error message."""

    sorted_values = sorted(values)
    shown_values = sorted_values[:MAX_ERROR_ALLOWLIST_ITEMS]
    hint = json.dumps(shown_values)
    if len(sorted_values) > MAX_ERROR_ALLOWLIST_ITEMS:
        hint = (
            f"{hint} "
            f"(+{len(sorted_values) - MAX_ERROR_ALLOWLIST_ITEMS} more)"
        )
    return hint

def _validate_handles(
    value: Any,
    allowed: set[str],
    label: str,
    *,
    minimum: int = 1,
    maximum: int = MAX_APPRAISAL_OBJECT_HANDLES,
    failure_kind: str | None = None,
    field_path: str | None = None,
) -> list[str]:
    """Validate a bounded duplicate-free handle list."""

    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(
            f"{label} handles must contain between {minimum} and {maximum} "
            f"items; allowed: {_allowlist_hint(allowed)}"
        )
    invalid_handles = [
        handle
        for handle in value
        if not isinstance(handle, str) or handle not in allowed
    ]
    if invalid_handles:
        rejected_text = json.dumps(
            sorted({str(handle) for handle in invalid_handles})
        )
        message = (
            f"{label} contains unknown handles {rejected_text}; "
            f"allowed: {_allowlist_hint(allowed)}"
        )
        if failure_kind is not None:
            raise _boundary_validation_error(
                message,
                failure_kind=failure_kind,
                field_path=field_path,
            )
        raise ValueError(message)
    if len(value) != len(set(value)):
        if failure_kind is not None:
            raise _terminal_boundary_error(
                f"{label} handles are duplicated",
                field_path=field_path,
            )
        raise ValueError(f"{label} handles are duplicated")
    return list(value)

def _referenced_role_handles(
    propositions: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Return every role handle referenced by valid proposition or delta rows."""

    referenced: set[str] = set()
    for proposition in propositions:
        referenced.add(proposition["subject_handle"])
        if "object_handle" in proposition:
            referenced.add(proposition["object_handle"])
        referenced.update(
            assignment["entity_handle"]
            for assignment in proposition["role_assignments"]
        )
    for delta in deltas:
        referenced.add(delta["target_path"].split(".")[1])
    return referenced

def _validate_candidate_evidence_binding(
    candidate_handles: Sequence[str],
    cited_evidence_handles: Sequence[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    *,
    field_path: str,
) -> None:
    """Require every prompt-local candidate to cite its source evidence."""

    cited = set(cited_evidence_handles)
    for handle in candidate_handles:
        evidence_handle = _candidate_evidence_handle(handle, handle_to_ref)
        if evidence_handle is not None and evidence_handle not in cited:
            raise _boundary_validation_error(
                "causal candidates must cite originating evidence: "
                f"{handle}->{evidence_handle}",
                failure_kind=_CANDIDATE_ORIGIN_MISSING,
                field_path=field_path,
            )

def _candidate_evidence_handle(
    candidate_handle: str,
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Map one candidate handle back to its exact evidence handle."""

    ref = handle_to_ref.get(candidate_handle)
    if ref is None:
        return None
    entity_id = ref.get("entity_id")
    if not isinstance(entity_id, str) or not entity_id.startswith("candidate:"):
        return None
    pieces = entity_id.split(":", maxsplit=2)
    if len(pieces) == 3 and pieces[1] in {
        "event",
        "threat",
        "knowledge_gap",
    }:
        return pieces[2]
    return None

def _validate_question_handle_authority(
    question: SemanticQuestionV2,
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> None:
    """Require every question handle to exist in the canonical projection."""

    canonical_handles = set(handle_to_ref)
    permitted_handles = set(question["permitted_role_handles"])
    if not permitted_handles <= canonical_handles:
        raise _terminal_boundary_error(
            "semantic question contains a non-canonical role handle",
            field_path="question.permitted_role_handles",
        )
    assignment_handles = set(question["permitted_role_assignment_handles"])
    if not assignment_handles <= canonical_handles:
        raise _terminal_boundary_error(
            "semantic question contains a non-canonical assignment handle",
            field_path="question.permitted_role_assignment_handles",
        )
    for handle in assignment_handles:
        if handle in {"self", "current_user"}:
            continue
        if handle_to_ref[handle]["kind"] not in ROLE_ENTITY_KINDS:
            raise _terminal_boundary_error(
                "semantic question contains a non-role assignment handle",
                field_path="question.permitted_role_assignment_handles",
            )
    for path in question["permitted_delta_paths"]:
        pieces = path.split(".")
        if len(pieces) >= 3 and pieces[1] not in canonical_handles:
            raise _terminal_boundary_error(
                "semantic question contains a non-canonical path handle",
                field_path="question.permitted_delta_paths",
            )

def _require_text(
    value: Any,
    label: str,
    *,
    maximum: int = MAX_APPRAISAL_SEMANTIC_TEXT_CHARS,
) -> str:
    """Require bounded non-empty semantic text."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{label} must be non-empty text up to {maximum} characters"
        )
    return value

__all__ = [
    "DELTA_LIMIT_NARROW",
    "DELTA_LIMIT_WIDE",
    "merge_semantic_appraisal_item",
    "validate_semantic_appraisal_result",
]
