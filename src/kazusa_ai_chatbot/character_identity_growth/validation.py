"""Structural validation for character identity growth contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime
from typing import cast

from kazusa_ai_chatbot.character_identity_growth import models


class IdentityContractViolation(ValueError):
    """Typed bounded contract failure from one model-facing decision.

    The exception carries a closed ``violations`` list whose entries contain
    only stable ``code``, ``field``, and ``expected`` values. The readable
    message aggregates the same facts and never includes raw model output,
    repository identifiers, or private transcript content.
    """

    def __init__(
        self,
        *,
        violations: list[dict[str, str]],
        message: str = "",
    ) -> None:
        self.violations = violations
        super().__init__(message or self._default_message())

    def _default_message(self) -> str:
        """Build one bounded human-readable summary of the violations."""

        if not self.violations:
            return "identity contract validation failed"
        lines = [
            f"field={entry['field']} code={entry['code']} "
            f"expected={entry['expected']}"
            for entry in self.violations
        ]
        return "; ".join(lines)


def _contract_error(
    *,
    code: str,
    field: str,
    expected: str,
) -> IdentityContractViolation:
    """Raise one typed bounded contract violation."""

    violation = {
        "code": code,
        "field": field,
        "expected": expected,
    }
    raise IdentityContractViolation(violations=[violation])


def _ordered_source_ids(
    source_ids: Sequence[str] | set[str] | frozenset[str],
) -> tuple[str, ...]:
    """Preserve prompt row order while keeping internal set callers stable."""

    if isinstance(source_ids, (set, frozenset)):
        return tuple(sorted(source_ids))
    return tuple(source_ids)


def _require_wire_exact_keys(
    payload: Mapping[str, object],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    """Report every structural key violation in one bounded failure."""

    actual = frozenset(payload)
    violations: list[dict[str, str]] = []
    for key in sorted(actual.difference(expected)):
        violations.append({
            "code": "unknown_key",
            "field": f"{context}.{key}",
            "expected": "only the closed model-facing keys",
        })
    for key in sorted(expected.difference(actual)):
        violations.append({
            "code": "missing_required_key",
            "field": f"{context}.{key}",
            "expected": "every closed model-facing key",
        })
    if violations:
        raise IdentityContractViolation(violations=violations)


def validate_effective_identity(
    payload: Mapping[str, object],
) -> models.CharacterEffectiveIdentityV1:
    """Validate and copy one complete semantic identity snapshot."""

    _require_exact_keys(
        payload,
        expected=models.TOP_LEVEL_IDENTITY_KEYS,
        context="effective identity",
    )
    personality = _require_mapping(
        payload["personality_brief"],
        context="personality_brief",
    )
    boundary = _require_mapping(
        payload["boundary_profile"],
        context="boundary_profile",
    )
    linguistic = _require_mapping(
        payload["linguistic_texture_profile"],
        context="linguistic_texture_profile",
    )
    self_image = _require_mapping(
        payload["self_image"],
        context="self_image",
    )
    _require_exact_keys(
        personality,
        expected=models.PERSONALITY_KEYS,
        context="personality_brief",
    )
    _require_exact_keys(
        boundary,
        expected=models.BOUNDARY_KEYS,
        context="boundary_profile",
    )
    _require_exact_keys(
        linguistic,
        expected=models.LINGUISTIC_TEXTURE_KEYS,
        context="linguistic_texture_profile",
    )
    _require_exact_keys(
        self_image,
        expected=models.SELF_IMAGE_KEYS,
        context="self_image",
    )

    for path in models.TEXT_IDENTITY_PATHS:
        value = _value_at_path(payload, path)
        _require_bounded_text(
            value,
            context=path,
            max_chars=models.TEXT_LIMIT_BY_PATH[path],
        )
    _require_age(payload["age"])
    for path in models.NUMERIC_IDENTITY_PATHS:
        _require_unit_number(_value_at_path(payload, path), context=path)
    for path, allowed_values in models.ENUM_VALUES_BY_PATH.items():
        _require_enum(
            _value_at_path(payload, path),
            context=path,
            allowed_values=allowed_values,
        )
    _require_text_list(
        self_image["current_growth_edges"],
        context="self_image.current_growth_edges",
        max_items=models.GROWTH_EDGE_COUNT_LIMIT,
        max_chars=models.GROWTH_EDGE_LIMIT,
    )

    copied = deepcopy(dict(payload))
    return_value = cast(models.CharacterEffectiveIdentityV1, copied)
    return return_value


def validate_identity_patch(
    payload: Mapping[str, object],
) -> models.IdentityPatchV1:
    """Validate one strict tagged identity replacement."""

    path = _require_bounded_text(
        payload.get("path"),
        context="path",
        max_chars=160,
    )
    if path not in models.ALLOWED_IDENTITY_PATHS:
        raise ValueError(f"unsupported identity patch path: {path}")
    value_kind = _require_bounded_text(
        payload.get("value_kind"),
        context=f"{path}.value_kind",
        max_chars=40,
    )

    if path in models.TEXT_IDENTITY_PATHS:
        expected_kind = "text"
        replacement_key = "replacement_text"
        replacement_value = _require_bounded_text(
            payload.get(replacement_key),
            context=f"{path}.{replacement_key}",
            max_chars=models.TEXT_LIMIT_BY_PATH[path],
        )
    elif path in models.INTEGER_IDENTITY_PATHS:
        expected_kind = "integer"
        replacement_key = "replacement_integer"
        replacement_value = _require_age(payload.get(replacement_key))
    elif path in models.NUMERIC_IDENTITY_PATHS:
        expected_kind = "semantic_band"
        replacement_key = "replacement_band"
        replacement_value = _require_enum(
            payload.get(replacement_key),
            context=f"{path}.{replacement_key}",
            allowed_values=frozenset(models.SEMANTIC_BAND_VALUES),
        )
    elif path in models.ENUM_IDENTITY_PATHS:
        expected_kind = "closed_enum"
        replacement_key = "replacement_enum"
        replacement_value = _require_enum(
            payload.get(replacement_key),
            context=f"{path}.{replacement_key}",
            allowed_values=models.ENUM_VALUES_BY_PATH[path],
        )
    else:
        expected_kind = "text_list"
        replacement_key = "replacement_items"
        replacement_value = _require_text_list(
            payload.get(replacement_key),
            context=f"{path}.{replacement_key}",
            max_items=models.GROWTH_EDGE_COUNT_LIMIT,
            max_chars=models.GROWTH_EDGE_LIMIT,
        )

    if value_kind != expected_kind:
        raise ValueError(
            f"{path} requires value_kind={expected_kind!r}, "
            f"received {value_kind!r}"
        )
    expected_keys = frozenset({"path", "value_kind", replacement_key})
    _require_exact_keys(
        payload,
        expected=expected_keys,
        context=f"identity patch {path}",
    )
    validated: dict[str, object] = {
        "path": path,
        "value_kind": value_kind,
        replacement_key: deepcopy(replacement_value),
    }
    return_value = cast(models.IdentityPatchV1, validated)
    return return_value


def validate_v2_wire_patch(
    payload: Mapping[str, object],
    *,
    context: str,
) -> models.IdentityPatchV1:
    """Validate one uniform model-facing patch and map it to a tagged patch."""

    actual_keys = frozenset(payload)
    unknown_keys = sorted(actual_keys.difference({"path", "replacement"}))
    if unknown_keys:
        _contract_error(
            code="unknown_key",
            field=f"{context}.{unknown_keys[0]}",
            expected="only path and replacement",
        )
    missing_keys = sorted({"path", "replacement"}.difference(actual_keys))
    if missing_keys:
        _contract_error(
            code="missing_required_key",
            field=f"{context}.{missing_keys[0]}",
            expected="path and replacement",
        )
    path = payload["path"]
    if not isinstance(path, str) or not path.strip():
        _contract_error(
            code="wrong_type",
            field=f"{context}.path",
            expected="a nonempty identity path string",
        )
    path = path.strip()
    if path not in models.ALLOWED_IDENTITY_PATHS:
        _contract_error(
            code="unsupported_value",
            field=f"{context}.path",
            expected="one allowed identity path",
        )
    replacement = payload["replacement"]

    if path in models.TEXT_IDENTITY_PATHS:
        value_kind = "text"
        replacement_key = "replacement_text"
        replacement_value = _require_bounded_text(
            replacement,
            context=f"{context}.replacement",
            max_chars=models.TEXT_LIMIT_BY_PATH[path],
        )
    elif path in models.INTEGER_IDENTITY_PATHS:
        value_kind = "integer"
        replacement_key = "replacement_integer"
        replacement_value = _require_age(replacement)
    elif path in models.NUMERIC_IDENTITY_PATHS:
        value_kind = "semantic_band"
        replacement_key = "replacement_band"
        replacement_value = _require_enum(
            replacement,
            context=f"{context}.replacement",
            allowed_values=frozenset(models.SEMANTIC_BAND_VALUES),
        )
    elif path in models.ENUM_IDENTITY_PATHS:
        value_kind = "closed_enum"
        replacement_key = "replacement_enum"
        replacement_value = _require_enum(
            replacement,
            context=f"{context}.replacement",
            allowed_values=models.ENUM_VALUES_BY_PATH[path],
        )
    else:
        value_kind = "text_list"
        replacement_key = "replacement_items"
        replacement_value = _require_text_list(
            replacement,
            context=f"{context}.replacement",
            max_items=models.GROWTH_EDGE_COUNT_LIMIT,
            max_chars=models.GROWTH_EDGE_LIMIT,
        )
    validated: dict[str, object] = {
        "path": path,
        "value_kind": value_kind,
        replacement_key: deepcopy(replacement_value),
    }
    return_value = cast(models.IdentityPatchV1, validated)
    return return_value


def _require_wire_text(
    value: object,
    *,
    field: str,
    expected: str,
) -> str:
    """Require bounded nonempty text with a typed contract failure."""

    if not isinstance(value, str) or not value.strip():
        _contract_error(
            code="wrong_type",
            field=field,
            expected=expected,
        )
    return value.strip()


def _require_wire_enum(
    value: object,
    *,
    field: str,
    allowed_values: frozenset[str],
) -> str:
    """Require one supported enum member with a typed contract failure."""

    if not isinstance(value, str) or value not in allowed_values:
        _contract_error(
            code="unsupported_value",
            field=field,
            expected=f"one of {sorted(allowed_values)}",
        )
    return value


def _require_wire_index_list(
    value: object,
    *,
    field: str,
    max_items: int,
) -> list[int]:
    """Require a bounded list of unique one-based integers."""

    if not isinstance(value, list):
        _contract_error(
            code="wrong_type",
            field=field,
            expected="a list of one-based integers",
        )
    if len(value) > max_items:
        _contract_error(
            code="invalid_provenance",
            field=field,
            expected=f"at most {max_items} unique indices",
        )
    validated: list[int] = []
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            _contract_error(
                code="invalid_index",
                field=item_field,
                expected="an integer from 1 through the prompt row count",
            )
        validated.append(item)
    if len(validated) != len(set(validated)):
        _contract_error(
            code="invalid_provenance",
            field=field,
            expected="unique one-based indices",
        )
    return validated


def _resolve_wire_indices(
    indices: list[int],
    *,
    field: str,
    source_ids: tuple[str, ...],
) -> list[str]:
    """Map validated one-based prompt indices to repository identifiers."""

    resolved: list[str] = []
    for position, index in enumerate(indices):
        if index > len(source_ids):
            _contract_error(
                code="invalid_index",
                field=f"{field}[{position}]",
                expected=f"an integer from 1 through {len(source_ids)}",
            )
        resolved.append(source_ids[index - 1])
    return resolved


def _internal_v1_proposal_to_wire(
    proposal: Mapping[str, object],
    *,
    evidence_ref_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
) -> models.IdentityProposalWireV2:
    """Map a validated internal proposal back to the wire shape."""

    evidence_index_by_id = {
        source_id: index
        for index, source_id in enumerate(evidence_ref_ids, start=1)
    }
    candidate_index_by_id = {
        source_id: index
        for index, source_id in enumerate(candidate_ids, start=1)
    }
    raw_evidence_ids = proposal.get("evidence_ref_ids")
    if not isinstance(raw_evidence_ids, list):
        _contract_error(
            code="wrong_type",
            field="proposal_decision.evidence_ref_ids",
            expected="a list of repository evidence identifiers",
        )
    evidence_indices: list[int] = []
    for source_id in raw_evidence_ids:
        if not isinstance(source_id, str):
            _contract_error(
                code="wrong_type",
                field="proposal_decision.evidence_ref_ids",
                expected="repository evidence identifiers",
            )
        index = evidence_index_by_id.get(source_id)
        if index is None:
            _contract_error(
                code="invalid_provenance",
                field="proposal_decision.evidence_ref_ids",
                expected="identifiers present in the prompt input",
            )
        evidence_indices.append(index)
    raw_candidate_id = proposal.get("candidate_id")
    candidate_index: int | None = None
    if isinstance(raw_candidate_id, str) and raw_candidate_id.strip():
        candidate_index = candidate_index_by_id.get(raw_candidate_id.strip())
        if candidate_index is None:
            _contract_error(
                code="invalid_provenance",
                field="proposal_decision.candidate_id",
                expected="a candidate identifier present in the prompt input",
            )
    elif raw_candidate_id is not None:
        _contract_error(
            code="wrong_type",
            field="proposal_decision.candidate_id",
            expected="a repository candidate identifier or null",
        )
    raw_contradictions = proposal.get("contradiction_candidate_ids")
    if not isinstance(raw_contradictions, list):
        _contract_error(
            code="wrong_type",
            field="proposal_decision.contradiction_candidate_ids",
            expected="a list of repository candidate identifiers",
        )
    contradiction_indices: list[int] = []
    for source_id in raw_contradictions:
        if not isinstance(source_id, str):
            _contract_error(
                code="wrong_type",
                field="proposal_decision.contradiction_candidate_ids",
                expected="repository candidate identifiers",
            )
        index = candidate_index_by_id.get(source_id)
        if index is None:
            _contract_error(
                code="invalid_provenance",
                field="proposal_decision.contradiction_candidate_ids",
                expected="identifiers present in the prompt input",
            )
        contradiction_indices.append(index)
    raw_patches = proposal.get("proposed_changes")
    if not isinstance(raw_patches, list):
        _contract_error(
            code="wrong_type",
            field="proposal_decision.proposed_changes",
            expected="a list of tagged identity patches",
        )
    proposed_changes: list[dict[str, object]] = []
    for index, raw_patch in enumerate(raw_patches):
        patch = _require_mapping(
            raw_patch,
            context=f"proposal_decision.proposed_changes[{index}]",
        )
        validated_patch = validate_identity_patch(
            patch,
        )
        path = validated_patch["path"]
        replacement_key = {
            "text": "replacement_text",
            "integer": "replacement_integer",
            "semantic_band": "replacement_band",
            "closed_enum": "replacement_enum",
            "text_list": "replacement_items",
        }[validated_patch["value_kind"]]
        proposed_changes.append({
            "path": path,
            "replacement": deepcopy(validated_patch[replacement_key]),
        })
    wire: dict[str, object] = {
        "action": proposal["action"],
        "candidate_index": candidate_index,
        "proposed_changes": proposed_changes,
        "character_authorship": proposal["character_authorship"],
        "identity_relevance": proposal["identity_relevance"],
        "global_applicability": proposal["global_applicability"],
        "confidence": proposal["confidence"],
        "private_detail_risk": proposal["private_detail_risk"],
        "character_owned_abstraction": proposal["character_owned_abstraction"],
        "evidence_indices": evidence_indices,
        "contradiction_candidate_indices": contradiction_indices,
    }
    return_value = cast(models.IdentityProposalWireV2, wire)
    return return_value


def _internal_v1_review_to_wire(
    review: Mapping[str, object],
    *,
    evidence_ref_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
) -> models.IdentityReviewWireV2:
    """Map a validated internal review to the model-facing wire shape."""
    candidate_index_by_id = {
        source_id: index
        for index, source_id in enumerate(candidate_ids, start=1)
    }
    raw_selected_id = review.get("selected_candidate_id")
    selected_candidate_index: int | None = None
    if isinstance(raw_selected_id, str) and raw_selected_id.strip():
        selected_candidate_index = candidate_index_by_id.get(
            raw_selected_id.strip()
        )
        if selected_candidate_index is None:
            _contract_error(
                code="invalid_provenance",
                field="identity review decision.selected_candidate_id",
                expected="a candidate identifier present in the prompt input",
            )
    raw_rejected_ids = review.get("rejected_candidate_ids")
    if not isinstance(raw_rejected_ids, list):
        _contract_error(
            code="wrong_type",
            field="identity review decision.rejected_candidate_ids",
            expected="a list of repository candidate identifiers",
        )
    rejected_candidate_indices: list[int] = []
    for source_id in raw_rejected_ids:
        if not isinstance(source_id, str):
            _contract_error(
                code="wrong_type",
                field="identity review decision.rejected_candidate_ids",
                expected="repository candidate identifiers",
            )
        index = candidate_index_by_id.get(source_id)
        if index is None:
            _contract_error(
                code="invalid_provenance",
                field="identity review decision.rejected_candidate_ids",
                expected="identifiers present in the prompt input",
            )
        rejected_candidate_indices.append(index)
    wire: dict[str, object] = {
        "verdict": review["verdict"],
        "selected_candidate_index": selected_candidate_index,
        "rejected_candidate_indices": rejected_candidate_indices,
        "character_authorship": review["character_authorship"],
        "identity_relevance": review["identity_relevance"],
        "coherence": review["coherence"],
        "global_applicability": review["global_applicability"],
        "review_confidence": review["review_confidence"],
        "private_detail_risk": review["private_detail_risk"],
        "character_owned_summary": review["character_owned_summary"],
        "privacy_safe_evidence_summaries": review[
            "privacy_safe_evidence_summaries"
        ],
    }
    return_value = cast(models.IdentityReviewWireV2, wire)
    return return_value


def validate_evidence_ref(
    payload: Mapping[str, object],
) -> models.IdentityEvidenceRefV1:
    """Validate one repository-owned root evidence reference."""

    expected_keys = frozenset({
        "schema_version",
        "evidence_ref_id",
        "root_episode_id",
        "correlation_id",
        "source_kind",
        "derived_reflection_run_ids",
        "character_local_date",
        "scope_kind",
        "captured_at",
    })
    _require_exact_keys(
        payload,
        expected=expected_keys,
        context="identity evidence ref",
    )
    schema_version = _require_bounded_text(
        payload["schema_version"],
        context="schema_version",
        max_chars=80,
    )
    if schema_version != models.IDENTITY_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            "identity evidence ref schema_version must be "
            f"{models.IDENTITY_EVIDENCE_SCHEMA_VERSION!r}"
        )
    source_kind = _require_enum(
        payload["source_kind"],
        context="source_kind",
        allowed_values=models.EVIDENCE_SOURCE_KINDS,
    )
    scope_kind = _require_enum(
        payload["scope_kind"],
        context="scope_kind",
        allowed_values=models.EVIDENCE_SCOPE_KINDS,
    )
    derived_ids = _require_text_list(
        payload["derived_reflection_run_ids"],
        context="derived_reflection_run_ids",
        max_items=32,
        max_chars=240,
    )
    if derived_ids != sorted(set(derived_ids)):
        raise ValueError(
            "derived_reflection_run_ids must be sorted and unique"
        )
    local_date = _require_iso_date(
        payload["character_local_date"],
        context="character_local_date",
    )
    captured_at = _require_iso_datetime(
        payload["captured_at"],
        context="captured_at",
    )
    validated: dict[str, object] = {
        "schema_version": schema_version,
        "evidence_ref_id": _require_bounded_text(
            payload["evidence_ref_id"],
            context="evidence_ref_id",
            max_chars=240,
        ),
        "root_episode_id": _require_bounded_text(
            payload["root_episode_id"],
            context="root_episode_id",
            max_chars=500,
        ),
        "correlation_id": _require_bounded_text(
            payload["correlation_id"],
            context="correlation_id",
            max_chars=500,
        ),
        "source_kind": source_kind,
        "derived_reflection_run_ids": derived_ids,
        "character_local_date": local_date,
        "scope_kind": scope_kind,
        "captured_at": captured_at,
    }
    return_value = cast(models.IdentityEvidenceRefV1, validated)
    return return_value


def validate_identity_evidence_card(
    payload: Mapping[str, object],
    *,
    evidence_ref: Mapping[str, object],
) -> models.IdentityEvidenceCardV1:
    """Validate one prompt-safe card against repository-owned provenance."""

    expected_keys = frozenset({
        "schema_version",
        "evidence_ref_id",
        "source_kind",
        "character_local_date",
        "scope_kind",
        "decontextualized_event",
        "character_cognition_summary",
        "visible_self_expression_summary",
    })
    _require_exact_keys(
        payload,
        expected=expected_keys,
        context="identity evidence card",
    )
    validated_ref = validate_evidence_ref(evidence_ref)
    schema_version = _require_bounded_text(
        payload["schema_version"],
        context="identity evidence card schema_version",
        max_chars=80,
    )
    if schema_version != models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION:
        raise ValueError(
            "identity evidence card schema_version must be "
            f"{models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION!r}"
        )
    evidence_ref_id = _require_bounded_text(
        payload["evidence_ref_id"],
        context="identity evidence card evidence_ref_id",
        max_chars=240,
    )
    if evidence_ref_id != validated_ref["evidence_ref_id"]:
        raise ValueError(
            "identity evidence card evidence_ref_id does not match "
            "repository provenance"
        )

    matched_fields = (
        "source_kind",
        "character_local_date",
        "scope_kind",
    )
    validated_values: dict[str, str] = {}
    for field_name in matched_fields:
        value = _require_bounded_text(
            payload[field_name],
            context=f"identity evidence card {field_name}",
            max_chars=80,
        )
        if value != validated_ref[field_name]:
            raise ValueError(
                f"identity evidence card {field_name} does not match "
                "repository provenance"
            )
        validated_values[field_name] = value

    validated: dict[str, object] = {
        "schema_version": schema_version,
        "evidence_ref_id": evidence_ref_id,
        **validated_values,
        "decontextualized_event": _require_bounded_text(
            payload["decontextualized_event"],
            context="identity evidence card decontextualized_event",
            max_chars=models.IDENTITY_EVIDENCE_CARD_TEXT_LIMIT,
        ),
        "character_cognition_summary": _require_optional_bounded_text(
            payload["character_cognition_summary"],
            context="identity evidence card character_cognition_summary",
            max_chars=models.IDENTITY_EVIDENCE_CARD_TEXT_LIMIT,
        ),
        "visible_self_expression_summary": (
            _require_optional_bounded_text(
                payload["visible_self_expression_summary"],
                context=(
                    "identity evidence card "
                    "visible_self_expression_summary"
                ),
                max_chars=models.IDENTITY_EVIDENCE_CARD_TEXT_LIMIT,
            )
        ),
    }
    return_value = cast(models.IdentityEvidenceCardV1, validated)
    return return_value


def validate_identity_proposal_wire(
    payload: Mapping[str, object],
    *,
    evidence_ref_ids: Sequence[str] | set[str],
    candidate_ids: Sequence[str] | set[str],
) -> models.IdentityProposalDecisionV1:
    """Validate one strict V2 proposal and map it to internal V1."""

    ordered_evidence_ids = _ordered_source_ids(evidence_ref_ids)
    ordered_candidate_ids = _ordered_source_ids(candidate_ids)
    wire_payload = cast(
        models.IdentityProposalWireV2,
        dict(payload),
    )
    _require_wire_exact_keys(
        wire_payload,
        expected=models.PROPOSAL_WIRE_KEYS,
        context="identity proposal decision",
    )

    action = _require_wire_enum(
        wire_payload["action"],
        field="identity proposal decision.action",
        allowed_values=models.PROPOSAL_ACTIONS,
    )
    raw_candidate_index = wire_payload["candidate_index"]
    if raw_candidate_index is not None and (
        isinstance(raw_candidate_index, bool)
        or not isinstance(raw_candidate_index, int)
        or raw_candidate_index < 1
    ):
        _contract_error(
            code="invalid_index",
            field="identity proposal decision.candidate_index",
            expected="null or an integer from 1 through the candidate count",
        )
    candidate_index = cast(int | None, raw_candidate_index)
    if (
        candidate_index is not None
        and candidate_index > len(ordered_candidate_ids)
    ):
        _contract_error(
            code="invalid_index",
            field="identity proposal decision.candidate_index",
            expected=f"an integer from 1 through {len(ordered_candidate_ids)}",
        )
    candidate_id = (
        ordered_candidate_ids[candidate_index - 1]
        if candidate_index is not None
        else None
    )

    raw_patches = wire_payload["proposed_changes"]
    if not isinstance(raw_patches, list):
        _contract_error(
            code="wrong_type",
            field="identity proposal decision.proposed_changes",
            expected="a list of uniform path/replacement patches",
        )
    if len(raw_patches) > models.IDENTITY_PATCH_LIMIT:
        _contract_error(
            code="invalid_provenance",
            field="identity proposal decision.proposed_changes",
            expected=f"at most {models.IDENTITY_PATCH_LIMIT} patches",
        )
    patches = [
        validate_v2_wire_patch(
            _require_mapping(
                raw_patch,
                context=(
                    "identity proposal decision "
                    f"proposed_changes[{index}]"
                ),
            ),
            context=(
                "identity proposal decision "
                f"proposed_changes[{index}]"
            ),
        )
        for index, raw_patch in enumerate(raw_patches)
    ]
    patch_paths = [patch["path"] for patch in patches]
    if len(patch_paths) != len(set(patch_paths)):
        _contract_error(
            code="cross_field_inconsistency",
            field="identity proposal decision.proposed_changes",
            expected="unique identity paths",
        )

    evidence_indices = _require_wire_index_list(
        wire_payload["evidence_indices"],
        field="identity proposal decision.evidence_indices",
        max_items=models.IDENTITY_EVIDENCE_CARD_LIMIT,
    )
    contradiction_indices = _require_wire_index_list(
        wire_payload["contradiction_candidate_indices"],
        field="identity proposal decision.contradiction_candidate_indices",
        max_items=models.IDENTITY_CANDIDATE_PROMPT_LIMIT,
    )
    cited_evidence = _resolve_wire_indices(
        evidence_indices,
        field="evidence_indices",
        source_ids=ordered_evidence_ids,
    )
    contradiction_ids = _resolve_wire_indices(
        contradiction_indices,
        field="contradiction_candidate_indices",
        source_ids=ordered_candidate_ids,
    )
    if (
        candidate_id is not None
        and candidate_id in contradiction_ids
    ):
        _contract_error(
            code="cross_field_inconsistency",
            field="identity proposal decision.contradiction_candidate_indices",
            expected="indices excluding the selected candidate",
        )

    if action == "no_change":
        if candidate_index is not None or patches or contradiction_indices:
            _contract_error(
                code="cross_field_inconsistency",
                field="identity proposal decision.action",
                expected="no_change with null candidate and empty changes",
            )
    else:
        if not patches:
            _contract_error(
                code="cross_field_inconsistency",
                field="identity proposal decision.proposed_changes",
                expected="at least one change for a change action",
            )
        if not evidence_indices:
            _contract_error(
                code="cross_field_inconsistency",
                field="identity proposal decision.evidence_indices",
                expected="at least one evidence index for a change action",
            )
        if action == "corroborate_candidate":
            if candidate_index is None:
                _contract_error(
                    code="cross_field_inconsistency",
                    field="identity proposal decision.candidate_index",
                    expected="exactly one valid candidate index",
                )
        elif action in {
            "explicit_self_redefinition",
            "inferred_growth",
        } and (
            candidate_index is not None
        ):
            _contract_error(
                code="cross_field_inconsistency",
                field="identity proposal decision.candidate_index",
                expected="null for a new proposal action",
            )

    character_authorship = _require_wire_enum(
        wire_payload["character_authorship"],
        field="identity proposal decision.character_authorship",
        allowed_values=models.CHARACTER_AUTHORSHIP_VALUES,
    )
    identity_relevance = _require_wire_enum(
        wire_payload["identity_relevance"],
        field="identity proposal decision.identity_relevance",
        allowed_values=models.IDENTITY_RELEVANCE_VALUES,
    )
    global_applicability = _require_wire_enum(
        wire_payload["global_applicability"],
        field="identity proposal decision.global_applicability",
        allowed_values=models.GLOBAL_APPLICABILITY_VALUES,
    )
    confidence = _require_wire_enum(
        wire_payload["confidence"],
        field="identity proposal decision.confidence",
        allowed_values=models.CONFIDENCE_VALUES,
    )
    private_detail_risk = _require_wire_enum(
        wire_payload["private_detail_risk"],
        field="identity proposal decision.private_detail_risk",
        allowed_values=models.PRIVATE_DETAIL_RISK_VALUES,
    )
    character_owned_abstraction = _require_wire_text(
        wire_payload["character_owned_abstraction"],
        field="identity proposal decision.character_owned_abstraction",
        expected="nonempty bounded detail-free text",
    )
    if len(character_owned_abstraction) > 1200:
        _contract_error(
            code="invalid_provenance",
            field="identity proposal decision.character_owned_abstraction",
            expected="at most 1200 characters",
        )
    if action == "explicit_self_redefinition":
        if character_authorship != "self_declared":
            _contract_error(
                code="cross_field_inconsistency",
                field="identity proposal decision.character_authorship",
                expected="self_declared for explicit_self_redefinition",
            )
    elif action != "no_change":
        if character_authorship != "inferred":
            _contract_error(
                code="cross_field_inconsistency",
                field="identity proposal decision.character_authorship",
                expected="inferred for inferred growth actions",
            )

    readiness_ready = (
        confidence == "high"
        and identity_relevance == "durable"
        and global_applicability == "global"
        and private_detail_risk == "low"
    )
    if action == "explicit_self_redefinition":
        if not readiness_ready:
            _contract_error(
                code="cross_field_inconsistency",
                field="identity proposal decision.confidence",
                expected=(
                    "high confidence, durable global relevance, "
                    "and low private-detail risk"
                ),
            )
        reason_code = "candidate_ready"
    elif action == "no_change":
        if private_detail_risk == "high":
            reason_code = "privacy_blocked"
        elif contradiction_indices:
            reason_code = "contradiction_blocked"
        else:
            reason_code = "proposal_no_change"
    elif readiness_ready:
        reason_code = "candidate_ready"
    else:
        reason_code = "candidate_emerging"

    leaked_handles = sorted({
        handle
        for handle in set(ordered_evidence_ids).union(ordered_candidate_ids)
        if handle in character_owned_abstraction
    })
    if leaked_handles:
        _contract_error(
            code="handle_leakage",
            field="identity proposal decision.character_owned_abstraction",
            expected="detail-free text without opaque input handles",
        )

    validated: dict[str, object] = {
        "schema_version": models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION,
        "action": action,
        "candidate_id": candidate_id,
        "proposed_changes": patches,
        "character_authorship": character_authorship,
        "identity_relevance": identity_relevance,
        "global_applicability": global_applicability,
        "confidence": confidence,
        "private_detail_risk": private_detail_risk,
        "character_owned_abstraction": character_owned_abstraction,
        "evidence_ref_ids": cited_evidence,
        "contradiction_candidate_ids": contradiction_ids,
        "reason_code": reason_code,
    }
    return_value = cast(models.IdentityProposalDecisionV1, validated)
    return return_value


def validate_identity_review_wire(
    payload: Mapping[str, object],
    *,
    proposal: Mapping[str, object],
    evidence_ref_ids: Sequence[str] | set[str],
    candidate_ids: Sequence[str] | set[str],
    candidate_change_kinds: Mapping[str, str] | None = None,
) -> models.IdentityReviewDecisionV1:
    """Validate one strict V2 review and copy accepted proposal patches."""

    ordered_evidence_ids = _ordered_source_ids(evidence_ref_ids)
    ordered_candidate_ids = _ordered_source_ids(candidate_ids)
    wire_payload = cast(
        models.IdentityReviewWireV2,
        dict(payload),
    )
    _require_wire_exact_keys(
        wire_payload,
        expected=models.REVIEW_WIRE_KEYS,
        context="identity review decision",
    )

    wire_proposal = cast(
        models.IdentityProposalWireV2,
        dict(proposal),
    )
    validated_proposal = validate_identity_proposal_wire(
        wire_proposal,
        evidence_ref_ids=ordered_evidence_ids,
        candidate_ids=ordered_candidate_ids,
    )
    _, proposal_contradiction_indices = _wire_indices_from_proposal(
        wire_proposal
    )

    verdict = _require_wire_enum(
        wire_payload["verdict"],
        field="identity review decision.verdict",
        allowed_values=models.REVIEW_VERDICTS,
    )
    raw_selected_index = wire_payload["selected_candidate_index"]
    if raw_selected_index is not None and (
        isinstance(raw_selected_index, bool)
        or not isinstance(raw_selected_index, int)
        or raw_selected_index < 1
    ):
        _contract_error(
            code="invalid_index",
            field="identity review decision.selected_candidate_index",
            expected="null or an integer from 1 through the candidate count",
        )
    selected_candidate_index = cast(int | None, raw_selected_index)
    if (
        selected_candidate_index is not None
        and selected_candidate_index > len(ordered_candidate_ids)
    ):
        _contract_error(
            code="invalid_index",
            field="identity review decision.selected_candidate_index",
            expected=f"an integer from 1 through {len(ordered_candidate_ids)}",
        )
    selected_candidate_id = (
        ordered_candidate_ids[selected_candidate_index - 1]
        if selected_candidate_index is not None
        else None
    )
    rejected_indices = _require_wire_index_list(
        wire_payload["rejected_candidate_indices"],
        field="identity review decision.rejected_candidate_indices",
        max_items=models.IDENTITY_CANDIDATE_PROMPT_LIMIT,
    )
    rejected_candidate_ids = _resolve_wire_indices(
        rejected_indices,
        field="rejected_candidate_indices",
        source_ids=ordered_candidate_ids,
    )
    if selected_candidate_id in rejected_candidate_ids:
        _contract_error(
            code="cross_field_inconsistency",
            field="identity review decision.rejected_candidate_indices",
            expected="indices excluding the selected candidate",
        )

    character_owned_summary = _require_wire_text(
        wire_payload["character_owned_summary"],
        field="identity review decision.character_owned_summary",
        expected="nonempty bounded detail-free text",
    )
    if len(character_owned_summary) > 1200:
        _contract_error(
            code="invalid_provenance",
            field="identity review decision.character_owned_summary",
            expected="at most 1200 characters",
        )
    raw_summaries = wire_payload["privacy_safe_evidence_summaries"]
    if not isinstance(raw_summaries, list):
        _contract_error(
            code="wrong_type",
            field="identity review decision.privacy_safe_evidence_summaries",
            expected="a list of detail-free evidence summaries",
        )
    if len(raw_summaries) > models.IDENTITY_EVIDENCE_CARD_LIMIT:
        _contract_error(
            code="invalid_provenance",
            field="identity review decision.privacy_safe_evidence_summaries",
            expected=(
                f"at most {models.IDENTITY_EVIDENCE_CARD_LIMIT} summaries"
            ),
        )
    summaries: list[str] = []
    for index, raw_summary in enumerate(raw_summaries):
        summary = _require_wire_text(
            raw_summary,
            field=(
                "identity review decision."
                f"privacy_safe_evidence_summaries[{index}]"
            ),
            expected="nonempty detail-free text",
        )
        if len(summary) > models.IDENTITY_EVIDENCE_CARD_TEXT_LIMIT:
            _contract_error(
                code="invalid_provenance",
                field=(
                    "identity review decision."
                    f"privacy_safe_evidence_summaries[{index}]"
                ),
                expected=(
                    f"at most {models.IDENTITY_EVIDENCE_CARD_TEXT_LIMIT} "
                    "characters"
                ),
            )
        summaries.append(summary)

    proposal_action = validated_proposal["action"]
    proposal_candidate_id = validated_proposal["candidate_id"]
    if verdict == "accept":
        if proposal_action == "no_change":
            _contract_error(
                code="cross_field_inconsistency",
                field="identity review decision.verdict",
                expected="no_change cannot be accepted",
            )
        if not summaries:
            _contract_error(
                code="cross_field_inconsistency",
                field=(
                    "identity review decision."
                    "privacy_safe_evidence_summaries"
                ),
                expected="at least one privacy-safe evidence summary",
            )
        if proposal_action == "corroborate_candidate":
            if selected_candidate_id != proposal_candidate_id:
                _contract_error(
                    code="cross_field_inconsistency",
                    field=(
                        "identity review decision."
                        "selected_candidate_index"
                    ),
                    expected="the corroborated proposal candidate",
                )
            change_kinds = candidate_change_kinds or {}
            accepted_change_kind = change_kinds.get(proposal_candidate_id)
            if accepted_change_kind not in models.ACCEPTED_CHANGE_KINDS:
                _contract_error(
                    code="invalid_provenance",
                    field="identity review decision.selected_candidate_index",
                    expected="a candidate with an accepted change kind",
                )
        elif proposal_action == "explicit_self_redefinition":
            if selected_candidate_id is not None:
                _contract_error(
                    code="cross_field_inconsistency",
                    field="identity review decision.selected_candidate_index",
                    expected="null for a new proposal",
                )
            accepted_change_kind = "explicit_self_redefinition"
        else:
            if selected_candidate_id is not None:
                _contract_error(
                    code="cross_field_inconsistency",
                    field="identity review decision.selected_candidate_index",
                    expected="null for a new proposal",
                )
            accepted_change_kind = "inferred_growth"
        accepted_changes = deepcopy(
            validated_proposal["proposed_changes"]
        )
        expected_authorship = (
            "self_declared"
            if accepted_change_kind == "explicit_self_redefinition"
            else "inferred"
        )
    else:
        if selected_candidate_index is not None:
            _contract_error(
                code="cross_field_inconsistency",
                field="identity review decision.selected_candidate_index",
                expected="null for reject or no_change",
            )
        accepted_change_kind = None
        accepted_changes = []
        if verdict == "no_change" and proposal_action != "no_change":
            _contract_error(
                code="cross_field_inconsistency",
                field="identity review decision.verdict",
                expected="no_change only for a proposal no_change",
            )
        if verdict == "reject" and proposal_action == "no_change":
            _contract_error(
                code="cross_field_inconsistency",
                field="identity review decision.verdict",
                expected="reject only for a proposed change",
            )
        if (
            proposal_action == "corroborate_candidate"
            and proposal_candidate_id is not None
            and proposal_candidate_id not in rejected_candidate_ids
        ):
            _contract_error(
                code="cross_field_inconsistency",
                field="identity review decision.rejected_candidate_indices",
                expected="the corroborated candidate rejected or accepted",
            )
        expected_authorship = None

    contradictions = set(
        validated_proposal["contradiction_candidate_ids"]
    )
    if not contradictions.issubset(set(rejected_candidate_ids)):
        _contract_error(
            code="cross_field_inconsistency",
            field="identity review decision.rejected_candidate_indices",
            expected="every proposal contradiction candidate rejected",
        )
    character_authorship = _require_wire_enum(
        wire_payload["character_authorship"],
        field="identity review decision.character_authorship",
        allowed_values=models.CHARACTER_AUTHORSHIP_VALUES,
    )
    identity_relevance = _require_wire_enum(
        wire_payload["identity_relevance"],
        field="identity review decision.identity_relevance",
        allowed_values=models.IDENTITY_RELEVANCE_VALUES,
    )
    coherence = _require_wire_enum(
        wire_payload["coherence"],
        field="identity review decision.coherence",
        allowed_values=models.REVIEW_COHERENCE_VALUES,
    )
    global_applicability = _require_wire_enum(
        wire_payload["global_applicability"],
        field="identity review decision.global_applicability",
        allowed_values=models.GLOBAL_APPLICABILITY_VALUES,
    )
    review_confidence = _require_wire_enum(
        wire_payload["review_confidence"],
        field="identity review decision.review_confidence",
        allowed_values=models.CONFIDENCE_VALUES,
    )
    private_detail_risk = _require_wire_enum(
        wire_payload["private_detail_risk"],
        field="identity review decision.private_detail_risk",
        allowed_values=models.PRIVATE_DETAIL_RISK_VALUES,
    )

    if verdict == "no_change":
        reason_code = "proposal_no_change"
    elif verdict == "reject":
        if private_detail_risk == "high":
            reason_code = "privacy_blocked"
        elif (
            coherence == "conflicting"
            or proposal_contradiction_indices
        ):
            reason_code = "contradiction_blocked"
        else:
            reason_code = "review_rejected"
    elif (
        review_confidence == "high"
        and identity_relevance == "durable"
        and coherence == "coherent"
        and global_applicability == "global"
        and private_detail_risk == "low"
    ):
        reason_code = "candidate_ready"
    else:
        reason_code = "candidate_emerging"

    if verdict == "accept" and expected_authorship is not None:
        if character_authorship != expected_authorship:
            _contract_error(
                code="cross_field_inconsistency",
                field="identity review decision.character_authorship",
                expected=f"{expected_authorship} for the accepted change",
            )
    if (
        verdict == "accept"
        and accepted_change_kind == "explicit_self_redefinition"
        and reason_code != "candidate_ready"
    ):
        _contract_error(
            code="cross_field_inconsistency",
            field="identity review decision.review_confidence",
            expected=(
                "high confidence, durable coherent global relevance, "
                "and low private-detail risk"
            ),
        )

    leaked_handles = sorted({
        handle
        for handle in set(ordered_evidence_ids).union(ordered_candidate_ids)
        if any(handle in text for text in (character_owned_summary, *summaries))
    })
    if leaked_handles:
        _contract_error(
            code="handle_leakage",
            field="identity review decision.character_owned_summary",
            expected="detail-free text without opaque input handles",
        )

    validated: dict[str, object] = {
        "schema_version": models.IDENTITY_REVIEW_DECISION_SCHEMA_VERSION,
        "verdict": verdict,
        "selected_candidate_id": selected_candidate_id,
        "rejected_candidate_ids": rejected_candidate_ids,
        "accepted_change_kind": accepted_change_kind,
        "accepted_changes": accepted_changes,
        "character_authorship": character_authorship,
        "identity_relevance": identity_relevance,
        "coherence": coherence,
        "global_applicability": global_applicability,
        "review_confidence": review_confidence,
        "private_detail_risk": private_detail_risk,
        "character_owned_summary": character_owned_summary,
        "privacy_safe_evidence_summaries": summaries,
        "reason_code": reason_code,
    }
    return_value = cast(models.IdentityReviewDecisionV1, validated)
    return return_value


def validate_identity_proposal_decision(
    payload: Mapping[str, object],
    *,
    evidence_ref_ids: Sequence[str] | set[str],
    candidate_ids: Sequence[str] | set[str],
) -> models.IdentityProposalDecisionV1:
    """Validate the internal V1 proposal contract used after the stage."""

    _require_exact_keys(
        payload,
        expected=frozenset(models.IdentityProposalDecisionV1.__annotations__),
        context="internal identity proposal decision",
    )
    if payload["schema_version"] != (
        models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION
    ):
        raise ValueError("internal identity proposal schema_version is invalid")
    ordered_evidence_ids = _ordered_source_ids(evidence_ref_ids)
    ordered_candidate_ids = _ordered_source_ids(candidate_ids)
    wire_proposal = _internal_v1_proposal_to_wire(
        payload,
        evidence_ref_ids=ordered_evidence_ids,
        candidate_ids=ordered_candidate_ids,
    )
    return validate_identity_proposal_wire(
        wire_proposal,
        evidence_ref_ids=ordered_evidence_ids,
        candidate_ids=ordered_candidate_ids,
    )


def validate_identity_review_decision(
    payload: Mapping[str, object],
    *,
    proposal: Mapping[str, object],
    evidence_ref_ids: Sequence[str] | set[str],
    candidate_ids: Sequence[str] | set[str],
    candidate_change_kinds: Mapping[str, str] | None = None,
) -> models.IdentityReviewDecisionV1:
    """Validate the internal V1 review contract used after the stage."""

    _require_exact_keys(
        payload,
        expected=frozenset(models.IdentityReviewDecisionV1.__annotations__),
        context="internal identity review decision",
    )
    if payload["schema_version"] != (
        models.IDENTITY_REVIEW_DECISION_SCHEMA_VERSION
    ):
        raise ValueError("internal identity review schema_version is invalid")
    ordered_evidence_ids = _ordered_source_ids(evidence_ref_ids)
    ordered_candidate_ids = _ordered_source_ids(candidate_ids)
    validated_proposal = validate_identity_proposal_decision(
        proposal,
        evidence_ref_ids=ordered_evidence_ids,
        candidate_ids=ordered_candidate_ids,
    )
    wire_proposal = _internal_v1_proposal_to_wire(
        proposal,
        evidence_ref_ids=ordered_evidence_ids,
        candidate_ids=ordered_candidate_ids,
    )
    wire_review = _internal_v1_review_to_wire(
        payload,
        evidence_ref_ids=ordered_evidence_ids,
        candidate_ids=ordered_candidate_ids,
    )
    resolved_change_kinds = dict(candidate_change_kinds or {})
    proposal_candidate_id = validated_proposal["candidate_id"]
    internal_change_kind = payload.get("accepted_change_kind")
    if (
        validated_proposal["action"] == "corroborate_candidate"
        and isinstance(proposal_candidate_id, str)
        and isinstance(internal_change_kind, str)
    ):
        resolved_change_kinds[proposal_candidate_id] = internal_change_kind
    return validate_identity_review_wire(
        wire_review,
        proposal=wire_proposal,
        evidence_ref_ids=ordered_evidence_ids,
        candidate_ids=ordered_candidate_ids,
        candidate_change_kinds=resolved_change_kinds,
    )


def _wire_indices_from_proposal(
    wire_proposal: Mapping[str, object],
) -> tuple[list[int], list[int]]:
    """Read the closed wire provenance indices from a normalized proposal."""

    raw_evidence = wire_proposal.get("evidence_indices")
    raw_contradictions = wire_proposal.get(
        "contradiction_candidate_indices"
    )
    if not isinstance(raw_evidence, list) or not isinstance(
        raw_contradictions,
        list,
    ):
        _contract_error(
            code="wrong_type",
            field="proposal_decision.evidence_indices",
            expected="closed wire provenance index lists",
        )
    evidence_indices = [int(index) for index in raw_evidence]
    contradiction_indices = [
        int(index)
        for index in raw_contradictions
    ]
    return evidence_indices, contradiction_indices


def _require_exact_keys(
    payload: Mapping[str, object],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    """Require one mapping to contain exactly its closed contract keys."""

    actual = frozenset(payload)
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
    if missing:
        raise ValueError(f"{context} missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {unknown}")


def _require_mapping(
    value: object,
    *,
    context: str,
) -> Mapping[str, object]:
    """Require a string-keyed mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    invalid_keys = [
        key
        for key in value
        if not isinstance(key, str)
    ]
    if invalid_keys:
        raise ValueError(f"{context} keys must be strings")
    return_value = cast(Mapping[str, object], value)
    return return_value


def _require_bounded_text(
    value: object,
    *,
    context: str,
    max_chars: int,
) -> str:
    """Require nonempty trimmed text within a declared bound."""

    if not isinstance(value, str):
        raise ValueError(f"{context} must be text")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"{context} must be nonempty")
    if len(trimmed) > max_chars:
        raise ValueError(
            f"{context} exceeds maximum length {max_chars}"
        )
    return trimmed


def _require_optional_bounded_text(
    value: object,
    *,
    context: str,
    max_chars: int,
) -> str:
    """Require trimmed text that may be empty within a declared bound."""

    if not isinstance(value, str):
        raise ValueError(f"{context} must be text")
    trimmed = value.strip()
    if len(trimmed) > max_chars:
        raise ValueError(
            f"{context} exceeds maximum length {max_chars}"
        )
    return trimmed


def _require_optional_identifier(
    value: object,
    *,
    context: str,
) -> str | None:
    """Require a bounded opaque identifier or null."""

    if value is None:
        return None
    identifier = _require_bounded_text(
        value,
        context=context,
        max_chars=240,
    )
    return identifier


def _require_age(value: object) -> int:
    """Require a non-boolean integer in the declared age range."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("age must be an integer")
    if not 0 <= value <= 10000:
        raise ValueError("age must be between 0 and 10000")
    return value


def _require_unit_number(value: object, *, context: str) -> float:
    """Require a non-boolean numeric value in the unit interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{context} must be between 0 and 1")
    return normalized


def _require_enum(
    value: object,
    *,
    context: str,
    allowed_values: frozenset[str],
) -> str:
    """Require one supported string enum member."""

    if not isinstance(value, str) or value not in allowed_values:
        raise ValueError(
            f"{context} must be one of {sorted(allowed_values)}"
        )
    return value


def _require_text_list(
    value: object,
    *,
    context: str,
    max_items: int,
    max_chars: int,
) -> list[str]:
    """Require a bounded list of unique nonempty strings."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    if len(value) > max_items:
        raise ValueError(f"{context} exceeds maximum item count {max_items}")
    validated = [
        _require_bounded_text(
            item,
            context=f"{context}[{index}]",
            max_chars=max_chars,
        )
        for index, item in enumerate(value)
    ]
    if len(validated) != len(set(validated)):
        raise ValueError(f"{context} must contain unique items")
    return validated


def _reject_handle_leakage(
    texts: list[str],
    *,
    handles: set[str],
) -> None:
    """Reject opaque input handles copied into persistent free text."""

    leaked_handles = sorted({
        handle
        for handle in handles
        if any(handle in text for text in texts)
    })
    if leaked_handles:
        raise ValueError(
            "identity generated free text contains opaque input handles"
        )


def _require_iso_date(value: object, *, context: str) -> str:
    """Require one ISO calendar date."""

    text = _require_bounded_text(value, context=context, max_chars=10)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO date") from exc
    return parsed.isoformat()


def _require_iso_datetime(value: object, *, context: str) -> str:
    """Require one ISO datetime with timezone information."""

    text = _require_bounded_text(value, context=context, max_chars=80)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{context} must include a timezone")
    return text


def _value_at_path(
    payload: Mapping[str, object],
    path: str,
) -> object:
    """Resolve one declared path from a complete identity mapping."""

    current: object = payload
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise ValueError(f"effective identity missing path: {path}")
        current = current[segment]
    return current
