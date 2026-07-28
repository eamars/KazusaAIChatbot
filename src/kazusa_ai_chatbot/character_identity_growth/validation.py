"""Structural validation for character identity growth contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime
from typing import cast

from kazusa_ai_chatbot.character_identity_growth import models


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


def validate_identity_proposal_decision(
    payload: Mapping[str, object],
    *,
    evidence_ref_ids: set[str],
    candidate_ids: set[str],
) -> models.IdentityProposalDecisionV1:
    """Validate one closed proposal-stage decision."""

    expected_keys = frozenset({
        "schema_version",
        "action",
        "candidate_id",
        "proposed_changes",
        "character_authorship",
        "identity_relevance",
        "global_applicability",
        "confidence",
        "private_detail_risk",
        "character_owned_abstraction",
        "evidence_ref_ids",
        "contradiction_candidate_ids",
        "reason_code",
    })
    _require_exact_keys(
        payload,
        expected=expected_keys,
        context="identity proposal decision",
    )
    schema_version = _require_bounded_text(
        payload["schema_version"],
        context="identity proposal decision schema_version",
        max_chars=80,
    )
    if schema_version != models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION:
        raise ValueError(
            "identity proposal decision schema_version must be "
            f"{models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION!r}"
        )
    action = _require_enum(
        payload["action"],
        context="identity proposal decision action",
        allowed_values=models.PROPOSAL_ACTIONS,
    )
    candidate_id = _require_optional_identifier(
        payload["candidate_id"],
        context="identity proposal decision candidate_id",
    )
    if candidate_id is not None and candidate_id not in candidate_ids:
        raise ValueError("identity proposal candidate_id is not in the input")

    raw_patches = payload["proposed_changes"]
    if not isinstance(raw_patches, list):
        raise ValueError("identity proposal proposed_changes must be a list")
    if len(raw_patches) > models.IDENTITY_PATCH_LIMIT:
        raise ValueError("identity proposal exceeds the patch limit")
    patches = [
        validate_identity_patch(
            _require_mapping(
                raw_patch,
                context=f"identity proposal proposed_changes[{index}]",
            )
        )
        for index, raw_patch in enumerate(raw_patches)
    ]
    patch_paths = [patch["path"] for patch in patches]
    if len(patch_paths) != len(set(patch_paths)):
        raise ValueError("identity proposal contains duplicate patch paths")

    cited_evidence = _require_text_list(
        payload["evidence_ref_ids"],
        context="identity proposal evidence_ref_ids",
        max_items=models.IDENTITY_EVIDENCE_CARD_LIMIT,
        max_chars=240,
    )
    unknown_evidence = sorted(set(cited_evidence).difference(evidence_ref_ids))
    if unknown_evidence:
        raise ValueError(
            "identity proposal cites unknown evidence refs: "
            f"{unknown_evidence}"
        )
    contradiction_ids = _require_text_list(
        payload["contradiction_candidate_ids"],
        context="identity proposal contradiction_candidate_ids",
        max_items=models.IDENTITY_CANDIDATE_PROMPT_LIMIT,
        max_chars=240,
    )
    unknown_candidates = sorted(
        set(contradiction_ids).difference(candidate_ids)
    )
    if unknown_candidates:
        raise ValueError(
            "identity proposal cites unknown contradiction candidates: "
            f"{unknown_candidates}"
        )
    if candidate_id is not None and candidate_id in contradiction_ids:
        raise ValueError(
            "identity proposal candidate cannot contradict itself"
        )

    if action == "no_change":
        if candidate_id is not None or patches or contradiction_ids:
            raise ValueError(
                "identity proposal no_change cannot carry candidate changes"
            )
    else:
        if not patches:
            raise ValueError(
                "identity proposal change action requires proposed_changes"
            )
        if not cited_evidence:
            raise ValueError(
                "identity proposal change action requires evidence refs"
            )
        if action == "corroborate_candidate" and candidate_id is None:
            raise ValueError(
                "corroborate_candidate requires an input candidate_id"
            )
        if action != "corroborate_candidate" and candidate_id is not None:
            raise ValueError(
                "new identity proposals cannot provide candidate_id"
            )

    reason_code = _require_enum(
        payload["reason_code"],
        context="identity proposal decision reason_code",
        allowed_values=models.IDENTITY_GROWTH_REASON_CODES,
    )
    character_authorship = _require_enum(
        payload["character_authorship"],
        context="identity proposal character_authorship",
        allowed_values=models.CHARACTER_AUTHORSHIP_VALUES,
    )
    if (
        action == "explicit_self_redefinition"
        and character_authorship != "self_declared"
    ):
        raise ValueError(
            "identity proposal explicit action requires self_declared "
            "authorship"
        )
    if (
        action in {"inferred_growth", "corroborate_candidate"}
        and character_authorship != "inferred"
    ):
        raise ValueError(
            "identity proposal inferred action requires inferred authorship"
        )
    if (
        action == "no_change"
        and reason_code not in {
            "proposal_no_change",
            "privacy_blocked",
            "contradiction_blocked",
        }
    ):
        raise ValueError(
            "identity proposal no_change reason_code is inconsistent"
        )
    if (
        action == "explicit_self_redefinition"
        and reason_code != "candidate_ready"
    ):
        raise ValueError(
            "identity proposal explicit reason_code is inconsistent"
        )
    if (
        action in {"inferred_growth", "corroborate_candidate"}
        and reason_code not in {
            "candidate_emerging",
            "candidate_ready",
        }
    ):
        raise ValueError(
            "identity proposal inferred reason_code is inconsistent"
        )
    validated: dict[str, object] = {
        "schema_version": schema_version,
        "action": action,
        "candidate_id": candidate_id,
        "proposed_changes": patches,
        "character_authorship": character_authorship,
        "identity_relevance": _require_enum(
            payload["identity_relevance"],
            context="identity proposal identity_relevance",
            allowed_values=models.IDENTITY_RELEVANCE_VALUES,
        ),
        "global_applicability": _require_enum(
            payload["global_applicability"],
            context="identity proposal global_applicability",
            allowed_values=models.GLOBAL_APPLICABILITY_VALUES,
        ),
        "confidence": _require_enum(
            payload["confidence"],
            context="identity proposal confidence",
            allowed_values=models.CONFIDENCE_VALUES,
        ),
        "private_detail_risk": _require_enum(
            payload["private_detail_risk"],
            context="identity proposal private_detail_risk",
            allowed_values=models.PRIVATE_DETAIL_RISK_VALUES,
        ),
        "character_owned_abstraction": _require_bounded_text(
            payload["character_owned_abstraction"],
            context="identity proposal character_owned_abstraction",
            max_chars=1200,
        ),
        "evidence_ref_ids": sorted(cited_evidence),
        "contradiction_candidate_ids": sorted(contradiction_ids),
        "reason_code": reason_code,
    }
    return_value = cast(models.IdentityProposalDecisionV1, validated)
    return return_value


def validate_identity_review_decision(
    payload: Mapping[str, object],
    *,
    proposal: Mapping[str, object],
    evidence_ref_ids: set[str],
    candidate_ids: set[str],
) -> models.IdentityReviewDecisionV1:
    """Validate one independent review-stage decision."""

    validated_proposal = validate_identity_proposal_decision(
        proposal,
        evidence_ref_ids=evidence_ref_ids,
        candidate_ids=candidate_ids,
    )
    expected_keys = frozenset({
        "schema_version",
        "verdict",
        "selected_candidate_id",
        "rejected_candidate_ids",
        "accepted_change_kind",
        "accepted_changes",
        "character_authorship",
        "identity_relevance",
        "coherence",
        "global_applicability",
        "review_confidence",
        "private_detail_risk",
        "character_owned_summary",
        "privacy_safe_evidence_summaries",
        "reason_code",
    })
    _require_exact_keys(
        payload,
        expected=expected_keys,
        context="identity review decision",
    )
    schema_version = _require_bounded_text(
        payload["schema_version"],
        context="identity review decision schema_version",
        max_chars=80,
    )
    if schema_version != models.IDENTITY_REVIEW_DECISION_SCHEMA_VERSION:
        raise ValueError(
            "identity review decision schema_version must be "
            f"{models.IDENTITY_REVIEW_DECISION_SCHEMA_VERSION!r}"
        )
    verdict = _require_enum(
        payload["verdict"],
        context="identity review decision verdict",
        allowed_values=models.REVIEW_VERDICTS,
    )
    selected_candidate_id = _require_optional_identifier(
        payload["selected_candidate_id"],
        context="identity review selected_candidate_id",
    )
    if (
        selected_candidate_id is not None
        and selected_candidate_id not in candidate_ids
    ):
        raise ValueError("identity review selected unknown candidate")
    rejected_candidate_ids = _require_text_list(
        payload["rejected_candidate_ids"],
        context="identity review rejected_candidate_ids",
        max_items=models.IDENTITY_CANDIDATE_PROMPT_LIMIT,
        max_chars=240,
    )
    unknown_rejections = sorted(
        set(rejected_candidate_ids).difference(candidate_ids)
    )
    if unknown_rejections:
        raise ValueError(
            "identity review rejected unknown candidates: "
            f"{unknown_rejections}"
        )
    if selected_candidate_id in rejected_candidate_ids:
        raise ValueError(
            "identity review cannot select and reject one candidate"
        )

    accepted_change_kind = payload["accepted_change_kind"]
    if accepted_change_kind is not None:
        accepted_change_kind = _require_enum(
            accepted_change_kind,
            context="identity review accepted_change_kind",
            allowed_values=models.ACCEPTED_CHANGE_KINDS,
        )
    raw_changes = payload["accepted_changes"]
    if not isinstance(raw_changes, list):
        raise ValueError("identity review accepted_changes must be a list")
    if len(raw_changes) > models.IDENTITY_PATCH_LIMIT:
        raise ValueError("identity review exceeds the patch limit")
    accepted_changes = [
        validate_identity_patch(
            _require_mapping(
                raw_patch,
                context=f"identity review accepted_changes[{index}]",
            )
        )
        for index, raw_patch in enumerate(raw_changes)
    ]

    summaries = _require_text_list(
        payload["privacy_safe_evidence_summaries"],
        context="identity review privacy_safe_evidence_summaries",
        max_items=models.IDENTITY_EVIDENCE_CARD_LIMIT,
        max_chars=models.IDENTITY_EVIDENCE_CARD_TEXT_LIMIT,
    )
    character_owned_summary = _require_bounded_text(
        payload["character_owned_summary"],
        context="identity review character_owned_summary",
        max_chars=1200,
    )
    _reject_handle_leakage(
        [character_owned_summary, *summaries],
        handles=evidence_ref_ids.union(candidate_ids),
    )

    proposal_action = validated_proposal["action"]
    if verdict == "accept":
        if proposal_action == "no_change":
            raise ValueError("identity review cannot accept no_change")
        if accepted_change_kind is None or not accepted_changes:
            raise ValueError(
                "identity review accept requires kind and changes"
            )
        if accepted_changes != validated_proposal["proposed_changes"]:
            raise ValueError(
                "identity review accepted changes must exactly match "
                "the proposal"
            )
        if not summaries:
            raise ValueError(
                "identity review accept requires privacy-safe summaries"
            )
        if proposal_action == "explicit_self_redefinition":
            if accepted_change_kind != "explicit_self_redefinition":
                raise ValueError(
                    "identity review accepted kind must match proposal"
                )
        elif proposal_action == "inferred_growth":
            if accepted_change_kind != "inferred_growth":
                raise ValueError(
                    "identity review accepted kind must match proposal"
                )
        elif selected_candidate_id != validated_proposal["candidate_id"]:
            raise ValueError(
                "identity review must select the corroborated candidate"
            )
        if (
            proposal_action != "corroborate_candidate"
            and selected_candidate_id is not None
        ):
            raise ValueError(
                "identity review cannot select a candidate for a new proposal"
            )
        contradictions = set(
            validated_proposal["contradiction_candidate_ids"]
        )
        if not contradictions.issubset(rejected_candidate_ids):
            raise ValueError(
                "identity review must reject every contradiction candidate"
            )
    else:
        if (
            accepted_change_kind is not None
            or accepted_changes
            or selected_candidate_id is not None
        ):
            raise ValueError(
                "identity review rejection/no_change cannot carry acceptance"
            )
        if verdict == "no_change" and proposal_action != "no_change":
            raise ValueError(
                "identity review no_change requires proposal no_change"
            )
        if verdict == "reject" and proposal_action == "no_change":
            raise ValueError(
                "identity review cannot reject a proposal no_change"
            )

    reason_code = _require_enum(
        payload["reason_code"],
        context="identity review reason_code",
        allowed_values=models.IDENTITY_GROWTH_REASON_CODES,
    )
    character_authorship = _require_enum(
        payload["character_authorship"],
        context="identity review character_authorship",
        allowed_values=models.CHARACTER_AUTHORSHIP_VALUES,
    )
    if verdict == "no_change" and reason_code != "proposal_no_change":
        raise ValueError(
            "identity review no_change reason_code is inconsistent"
        )
    if (
        verdict == "reject"
        and reason_code not in {
            "review_rejected",
            "privacy_blocked",
            "contradiction_blocked",
        }
    ):
        raise ValueError(
            "identity review reject reason_code is inconsistent"
        )
    if verdict == "accept":
        accepted_reason_codes = {
            "candidate_emerging",
            "candidate_ready",
        }
        if reason_code not in accepted_reason_codes:
            raise ValueError(
                "identity review accept reason_code is inconsistent"
            )
        if (
            accepted_change_kind == "explicit_self_redefinition"
            and reason_code != "candidate_ready"
        ):
            raise ValueError(
                "identity review explicit reason_code is inconsistent"
            )
        expected_authorship = (
            "self_declared"
            if accepted_change_kind == "explicit_self_redefinition"
            else "inferred"
        )
        if character_authorship != expected_authorship:
            raise ValueError(
                "identity review accepted authorship is inconsistent"
            )

    validated: dict[str, object] = {
        "schema_version": schema_version,
        "verdict": verdict,
        "selected_candidate_id": selected_candidate_id,
        "rejected_candidate_ids": sorted(rejected_candidate_ids),
        "accepted_change_kind": accepted_change_kind,
        "accepted_changes": accepted_changes,
        "character_authorship": character_authorship,
        "identity_relevance": _require_enum(
            payload["identity_relevance"],
            context="identity review identity_relevance",
            allowed_values=models.IDENTITY_RELEVANCE_VALUES,
        ),
        "coherence": _require_enum(
            payload["coherence"],
            context="identity review coherence",
            allowed_values=models.REVIEW_COHERENCE_VALUES,
        ),
        "global_applicability": _require_enum(
            payload["global_applicability"],
            context="identity review global_applicability",
            allowed_values=models.GLOBAL_APPLICABILITY_VALUES,
        ),
        "review_confidence": _require_enum(
            payload["review_confidence"],
            context="identity review review_confidence",
            allowed_values=models.CONFIDENCE_VALUES,
        ),
        "private_detail_risk": _require_enum(
            payload["private_detail_risk"],
            context="identity review private_detail_risk",
            allowed_values=models.PRIVATE_DETAIL_RISK_VALUES,
        ),
        "character_owned_summary": character_owned_summary,
        "privacy_safe_evidence_summaries": summaries,
        "reason_code": reason_code,
    }
    return_value = cast(models.IdentityReviewDecisionV1, validated)
    return return_value


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
            "identity review free text contains opaque input handles"
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
