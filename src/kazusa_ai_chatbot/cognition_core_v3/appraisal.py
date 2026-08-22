"""Validate fixed A1/A2 appraisal family objects through canonical owners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kazusa_ai_chatbot.cognition_core_v3.semantic_appraisal import (
    merge_semantic_appraisal_item,
    validate_semantic_appraisal_result,
)
from kazusa_ai_chatbot.cognition_core_v3.semantic_source_planner import (
    question_proposition_kind_semantics,
    question_proposition_kinds,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import APPRAISAL_FAMILY_ORDER


def family_proposition_kinds(question_kind: str) -> tuple[str, ...]:
    """Read one family's proposition vocabulary from the V2 owner."""

    return question_proposition_kinds(question_kind)


def family_proposition_kind_semantics(question_kind: str) -> dict[str, str]:
    """Read one family's proposition meanings from the V2 owner."""

    return question_proposition_kind_semantics(question_kind)


_APPRAISAL_PROPOSITION_FIELDS = frozenset({
    "proposition_kind",
    "subject_handle",
    "evidence_handles",
    "role_assignments",
    "semantic_value",
})
_APPRAISAL_DELTA_FIELDS = frozenset({
    "target_path",
    "delta",
    "evidence_handles",
    "reason",
})
_APPRAISAL_ROLE_ASSIGNMENT_FIELDS = frozenset({
    "role",
    "entity_handle",
})


def validate_appraisal_stage_output(
    raw: Mapping[str, object],
    *,
    planned_families: Sequence[str],
    role_assignment_handles_by_evidence: Mapping[
        str, Sequence[str]
    ] | None = None,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    """Validate the exact family-object boundary for one A1 or A2 call."""

    if not isinstance(raw, Mapping):
        raise TypeError("appraisal stage output must be an object")
    ordered_families = tuple(planned_families)
    if not ordered_families or any(
        family not in APPRAISAL_FAMILY_ORDER for family in ordered_families
    ):
        raise ValueError("appraisal stage family roster is invalid")
    if tuple(raw) != ordered_families:
        raise ValueError(
            "appraisal stage output fields must exactly match planned families"
        )

    validated: dict[str, dict[str, list[dict[str, object]]]] = {}
    for family in ordered_families:
        value = raw[family]
        if not isinstance(value, Mapping) or set(value) != {
            "propositions",
            "deltas",
        }:
            raise ValueError(
                f"appraisal family {family} must contain only propositions and deltas"
            )
        family_result: dict[str, list[dict[str, object]]] = {}
        for field_name in ("propositions", "deltas"):
            rows = value[field_name]
            if not isinstance(rows, list) or len(rows) > 8:
                raise ValueError(
                    f"appraisal family {family} {field_name} must contain at most eight"
                )
            cleaned_rows: list[dict[str, object]] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError(
                        f"appraisal family {family} {field_name} rows must be objects"
                    )
                if field_name == "propositions":
                    allowed_fields = _APPRAISAL_PROPOSITION_FIELDS | {
                        "object_handle"
                    }
                    if set(row) - allowed_fields or not _APPRAISAL_PROPOSITION_FIELDS <= set(row):
                        raise ValueError(
                            f"appraisal family {family} proposition fields are not exact"
                        )
                    if (
                        not isinstance(row["proposition_kind"], str)
                        or not isinstance(row["subject_handle"], str)
                        or not isinstance(row["semantic_value"], str)
                    ):
                        raise TypeError(
                            f"appraisal family {family} proposition scalar fields are invalid"
                        )
                    if (
                        "object_handle" in row
                        and row["object_handle"] is not None
                        and not isinstance(row["object_handle"], str)
                    ):
                        raise TypeError(
                            f"appraisal family {family} object handle is invalid"
                        )
                    evidence = row["evidence_handles"]
                    if not isinstance(evidence, list) or any(
                        not isinstance(handle, str) for handle in evidence
                    ):
                        raise TypeError(
                            f"appraisal family {family} proposition evidence is invalid"
                        )
                    assignments = row["role_assignments"]
                    if not isinstance(assignments, list) or len(assignments) > 8:
                        raise ValueError(
                            f"appraisal family {family} role assignments are invalid"
                        )
                    for assignment in assignments:
                        if (
                            not isinstance(assignment, Mapping)
                            or set(assignment) != _APPRAISAL_ROLE_ASSIGNMENT_FIELDS
                        ):
                            raise ValueError(
                                "appraisal role assignment fields are not exact"
                            )
                        if not isinstance(assignment["role"], str) or not isinstance(
                            assignment["entity_handle"], str
                        ):
                            raise TypeError(
                                "appraisal role assignment values are invalid"
                            )
                    if role_assignment_handles_by_evidence is not None:
                        authorized_handles: set[str] | None = None
                        for evidence_handle in evidence:
                            evidence_authority = (
                                role_assignment_handles_by_evidence.get(
                                    evidence_handle,
                                    (),
                                )
                            )
                            if (
                                isinstance(evidence_authority, (str, bytes))
                                or not isinstance(evidence_authority, Sequence)
                                or any(
                                    not isinstance(handle, str)
                                    for handle in evidence_authority
                                )
                            ):
                                raise TypeError(
                                    "appraisal evidence role authority is invalid"
                                )
                            evidence_authority = {
                                handle
                                for handle in evidence_authority
                            }
                            authorized_handles = (
                                evidence_authority
                                if authorized_handles is None
                                else authorized_handles & evidence_authority
                            )
                        if authorized_handles is None:
                            authorized_handles = set()
                        if any(
                            assignment["entity_handle"] not in authorized_handles
                            for assignment in assignments
                        ):
                            raise ValueError(
                                "appraisal role assignment is not authorized "
                                "by every cited evidence handle"
                            )
                else:
                    if set(row) != _APPRAISAL_DELTA_FIELDS:
                        raise ValueError(
                            f"appraisal family {family} delta fields are not exact"
                        )
                    if not isinstance(row["target_path"], str):
                        raise TypeError(
                            f"appraisal family {family} delta path is invalid"
                        )
                    if len(row["target_path"].split(".")) != 3:
                        raise ValueError(
                            f"appraisal family {family} delta path is invalid"
                        )
                    if isinstance(row["delta"], bool) or not isinstance(
                        row["delta"], int
                    ):
                        raise TypeError(
                            f"appraisal family {family} delta value is invalid"
                        )
                    if not isinstance(row["reason"], str):
                        raise TypeError(
                            f"appraisal family {family} delta reason is invalid"
                        )
                    evidence = row["evidence_handles"]
                    if not isinstance(evidence, list) or any(
                        not isinstance(handle, str) for handle in evidence
                    ):
                        raise TypeError(
                            f"appraisal family {family} delta evidence is invalid"
                        )
                cleaned_row = dict(row)
                if cleaned_row.get("object_handle") is None:
                    cleaned_row.pop("object_handle", None)
                cleaned_rows.append(cleaned_row)
            family_result[field_name] = cleaned_rows
        validated[family] = family_result
    return validated


def _selected_handles_for_product(
    product: Mapping[str, object],
    *,
    product_kind: str,
) -> tuple[list[str], list[str]]:
    """Derive one product's V2 metadata in authored field order."""

    selected_evidence = list(product["evidence_handles"])
    selected_roles: list[str] = []
    if product_kind == "proposition":
        selected_roles.append(str(product["subject_handle"]))
        if "object_handle" in product:
            selected_roles.append(str(product["object_handle"]))
        selected_roles.extend(
            str(assignment["entity_handle"])
            for assignment in product["role_assignments"]
        )
    else:
        selected_roles.append(str(product["target_path"]).split(".")[1])
    return (
        list(dict.fromkeys(selected_evidence)),
        list(dict.fromkeys(selected_roles)),
    )


def reduce_appraisal_stage_output(
    raw: Mapping[str, object],
    *,
    planned_families: Sequence[str],
    questions_by_family: Mapping[str, Mapping[str, object]],
    evidence_handles: Sequence[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    role_assignment_handles_by_evidence: Mapping[
        str, Sequence[str]
    ] | None = None,
) -> list[dict[str, object]]:
    """Bridge family objects into canonical V2 validated appraisal rows."""

    validated = validate_appraisal_stage_output(
        raw,
        planned_families=planned_families,
        role_assignment_handles_by_evidence=(
            role_assignment_handles_by_evidence
        ),
    )
    evidence_domain = set(evidence_handles)
    reduced_rows: list[dict[str, object]] = []

    for family_name in planned_families:
        question = questions_by_family.get(family_name)
        if not isinstance(question, Mapping):
            raise TypeError(
                f"appraisal family {family_name} has no planned question"
            )
        family_result = validated[family_name]
        merged_result: dict[str, object] | None = None
        for product_kind, products in (
            ("proposition", family_result["propositions"]),
            ("delta", family_result["deltas"]),
        ):
            for product in products:
                selected_evidence, selected_roles = (
                    _selected_handles_for_product(
                        product,
                        product_kind=product_kind,
                    )
                )
                item_result = {
                    "question_id": question["question_id"],
                    "selected_evidence_handles": selected_evidence,
                    "selected_role_handles": selected_roles,
                    "propositions": (
                        [product] if product_kind == "proposition" else []
                    ),
                    "deltas": [product] if product_kind == "delta" else [],
                    "explanation": (
                        product["semantic_value"]
                        if product_kind == "proposition"
                        else product["reason"]
                    ),
                }
                validated_item = validate_semantic_appraisal_result(
                    item_result,
                    question,
                    evidence_domain,
                    handle_to_ref,
                    maximum_propositions=1 if product_kind == "proposition" else 0,
                    maximum_deltas=1 if product_kind == "delta" else 0,
                    maximum_explanation_chars=1000,
                )
                merged_result = merge_semantic_appraisal_item(
                    merged_result,
                    validated_item,
                )

        if merged_result is None:
            merged_result = {
                "question_id": question["question_id"],
                "selected_evidence_handles": [],
                "selected_role_handles": [],
                "propositions": [],
                "deltas": [],
                "explanation": "无新增受支持语义判断。",
            }
        validated_row = validate_semantic_appraisal_result(
            merged_result,
            question,
            evidence_domain,
            handle_to_ref,
            maximum_propositions=8,
            maximum_deltas=8,
            maximum_explanation_chars=1000,
        )
        reduced_rows.append(validated_row)

    return reduced_rows
