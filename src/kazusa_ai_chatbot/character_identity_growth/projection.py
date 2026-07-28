"""Prompt-safe projections for identity proposal and review stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import cast

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.identity import (
    evidence_counts,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_effective_identity,
    validate_evidence_ref,
    validate_identity_evidence_card,
    validate_identity_patch,
    validate_identity_proposal_decision,
)


_PROPOSAL_INPUT_KEYS = frozenset({
    "schema_version",
    "current_identity",
    "evidence_cards",
    "current_candidates",
    "allowed_paths",
})
_CORE_IDENTITY_KEYS = (
    "name",
    "description",
    "gender",
    "age",
    "birthday",
    "backstory",
)
_REVISION_KINDS = frozenset({
    "seed",
    "explicit_turning_point",
    "corroborated_growth",
    "operator_reset",
})
_REVISION_CONFIDENCE_VALUES = frozenset({"seed", "high", "operator"})
_REVISION_SCOPE_KINDS = frozenset({
    "private",
    "group",
    "reflection",
    "self_cognition",
    "operator",
})
_CANDIDATE_STATUSES = frozenset({
    "emerging",
    "ready",
    "promoted",
    "rejected",
    "superseded",
})
_RUN_KINDS = frozenset({"episode", "daily_reflection", "operator_reset"})
_RUN_DISPOSITIONS = frozenset({
    "no_change",
    "candidate_updated",
    "revision_promoted",
    "rejected",
    "failed",
    "deferred",
})


def project_identity_for_growth_prompt(
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Project one full identity with numeric values replaced by bands."""

    validated = validate_effective_identity(identity)
    projected = deepcopy(dict(validated))
    for path in models.NUMERIC_IDENTITY_PATHS:
        value = _value_at_path(projected, path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"identity numeric path is invalid: {path}")
        _set_value_at_path(
            projected,
            path,
            _semantic_band_for_number(float(value)),
        )
    return projected


def project_identity_for_cognition(
    revision: Mapping[str, object],
    *,
    include_epistemic_core: bool = False,
) -> models.CharacterIdentityCognitionContextV1:
    """Project the latest revision into exact V2 appraisal partitions."""

    identity = _identity_from_revision(revision)
    core = {
        key: deepcopy(identity[key])
        for key in _CORE_IDENTITY_KEYS
    }
    personality = deepcopy(dict(identity["personality_brief"]))
    boundaries = deepcopy(dict(identity["boundary_profile"]))
    self_image = deepcopy(dict(identity["self_image"]))
    result: models.CharacterIdentityCognitionContextV1 = {
        "moral_identity": {
            "core": deepcopy(core),
            "personality": deepcopy(personality),
            "boundaries": deepcopy(boundaries),
            "self_image": deepcopy(self_image),
        },
        "existential_drive": {
            "core": deepcopy(core),
            "personality": deepcopy(personality),
            "self_image": deepcopy(self_image),
        },
        "relationship_social": {
            "personality": deepcopy(personality),
            "boundaries": deepcopy(boundaries),
        },
        "event_agency": {
            "personality": deepcopy(personality),
            "boundaries": deepcopy(boundaries),
        },
        "goal_threat_outcome": {
            "personality": deepcopy(personality),
            "boundaries": deepcopy(boundaries),
        },
        "goal_cognition": {
            "core": deepcopy(core),
            "personality": deepcopy(personality),
            "boundaries": deepcopy(boundaries),
            "self_image": deepcopy(self_image),
        },
        "epistemic_comparison_memory": (
            {"core": deepcopy(core)}
            if include_epistemic_core
            else {}
        ),
    }
    return result


def project_identity_for_surface(
    revision: Mapping[str, object],
) -> models.CharacterIdentitySurfaceContextV1:
    """Project the latest revision into text, visual, and naming contexts."""

    identity = _identity_from_revision(revision)
    personality = identity["personality_brief"]
    if not isinstance(personality, Mapping):
        raise ValueError("effective identity personality must be an object")
    result: models.CharacterIdentitySurfaceContextV1 = {
        "text": {
            "name": identity["name"],
            "personality": {
                key: personality[key]
                for key in ("tempo", "defense", "quirks")
            },
            "linguistic_texture_profile": deepcopy(
                dict(identity["linguistic_texture_profile"])
            ),
        },
        "visual": {
            key: deepcopy(identity[key])
            for key in (
                "name",
                "description",
                "gender",
                "age",
                "visual_characterization",
            )
        },
        "naming": {"name": identity["name"]},
    }
    return result


def project_identity_for_console(
    revision: Mapping[str, object],
) -> dict[str, object]:
    """Project one immutable revision without identity values or handles."""

    revision_number = _require_nonnegative_integer(
        revision.get("revision_number"),
        context="identity revision revision_number",
    )
    base_revision_number = _require_optional_nonnegative_integer(
        revision.get("base_revision_number"),
        context="identity revision base_revision_number",
    )
    revision_kind = _require_enum(
        revision.get("revision_kind"),
        context="identity revision revision_kind",
        allowed=_REVISION_KINDS,
    )
    changed_paths = _require_text_list(
        revision.get("changed_paths"),
        context="identity revision changed_paths",
        max_items=len(models.ALLOWED_IDENTITY_PATHS),
        max_chars=160,
    )
    if not set(changed_paths).issubset(models.ALLOWED_IDENTITY_PATHS):
        raise ValueError("identity revision changed_paths are invalid")

    change_diff = _project_console_change_diff(
        revision.get("change_diff"),
        changed_paths=changed_paths,
    )
    evidence_refs = _require_list(
        revision.get("evidence_refs"),
        context="identity revision evidence_refs",
    )
    evidence_root_count, evidence_local_date_count = (
        _console_evidence_counts(evidence_refs)
    )
    source_scope_kinds = _require_text_list(
        revision.get("source_scope_kinds"),
        context="identity revision source_scope_kinds",
        max_items=len(_REVISION_SCOPE_KINDS),
        max_chars=40,
    )
    if not set(source_scope_kinds).issubset(_REVISION_SCOPE_KINDS):
        raise ValueError("identity revision source_scope_kinds are invalid")

    projected = {
        "kind": "identity_revision",
        "revision_number": revision_number,
        "base_revision_number": base_revision_number,
        "revision_kind": revision_kind,
        "changed_paths": changed_paths,
        "change_diff": change_diff,
        "evidence_summary": _require_text(
            revision.get("evidence_summary"),
            context="identity revision evidence_summary",
            max_chars=2400,
        ),
        "source_scope_kinds": source_scope_kinds,
        "evidence_root_count": evidence_root_count,
        "evidence_local_date_count": evidence_local_date_count,
        "proposal_confidence": _require_enum(
            revision.get("proposal_confidence"),
            context="identity revision proposal_confidence",
            allowed=_REVISION_CONFIDENCE_VALUES,
        ),
        "review_confidence": _require_enum(
            revision.get("review_confidence"),
            context="identity revision review_confidence",
            allowed=_REVISION_CONFIDENCE_VALUES,
        ),
        "created_at": _require_text(
            revision.get("created_at"),
            context="identity revision created_at",
            max_chars=80,
        ),
    }
    return projected


def project_candidate_for_console(
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Project one candidate without semantic text or repository handles."""

    status = _require_enum(
        candidate.get("status"),
        context="identity candidate status",
        allowed=_CANDIDATE_STATUSES,
    )
    change_kind = _require_enum(
        candidate.get("change_kind"),
        context="identity candidate change_kind",
        allowed=models.ACCEPTED_CHANGE_KINDS,
    )
    raw_changes = _require_list(
        candidate.get("proposed_changes"),
        context="identity candidate proposed_changes",
    )
    proposed_paths: list[str] = []
    for index, raw_change in enumerate(raw_changes):
        change = _require_mapping(
            raw_change,
            context=f"identity candidate proposed_changes[{index}]",
        )
        path = _require_text(
            change.get("path"),
            context=f"identity candidate proposed_changes[{index}].path",
            max_chars=160,
        )
        if path not in models.ALLOWED_IDENTITY_PATHS:
            raise ValueError("identity candidate proposed path is invalid")
        proposed_paths.append(path)
    if len(proposed_paths) != len(set(proposed_paths)):
        raise ValueError("identity candidate proposed paths must be unique")

    distinct_local_dates = _require_text_list(
        candidate.get("distinct_local_dates"),
        context="identity candidate distinct_local_dates",
        max_items=64,
        max_chars=40,
    )
    source_scope_kinds = _require_text_list(
        candidate.get("source_scope_kinds"),
        context="identity candidate source_scope_kinds",
        max_items=len(models.EVIDENCE_SCOPE_KINDS),
        max_chars=40,
    )
    if not set(source_scope_kinds).issubset(models.EVIDENCE_SCOPE_KINDS):
        raise ValueError("identity candidate source scopes are invalid")

    projected = {
        "kind": "identity_candidate",
        "status": status,
        "base_revision_number": _require_nonnegative_integer(
            candidate.get("base_revision_number"),
            context="identity candidate base_revision_number",
        ),
        "change_kind": change_kind,
        "proposed_paths": sorted(proposed_paths),
        "root_count": _require_nonnegative_integer(
            candidate.get("distinct_episode_count"),
            context="identity candidate distinct_episode_count",
        ),
        "local_date_count": len(distinct_local_dates),
        "source_scope_kinds": sorted(source_scope_kinds),
        "fresh_post_revision_root_count": _require_nonnegative_integer(
            candidate.get("fresh_post_revision_root_count"),
            context="identity candidate fresh_post_revision_root_count",
        ),
        "reversal_of_paths": _require_console_identity_paths(
            candidate.get("reversal_of_paths"),
            context="identity candidate reversal_of_paths",
        ),
        "character_authorship": _require_enum(
            candidate.get("character_authorship"),
            context="identity candidate character_authorship",
            allowed=models.CHARACTER_AUTHORSHIP_VALUES,
        ),
        "proposal_confidence": _require_enum(
            candidate.get("proposal_confidence"),
            context="identity candidate proposal_confidence",
            allowed=models.CONFIDENCE_VALUES,
        ),
        "review_confidence": _require_enum(
            candidate.get("review_confidence"),
            context="identity candidate review_confidence",
            allowed=models.CONFIDENCE_VALUES,
        ),
        "privacy_review": _require_enum(
            candidate.get("privacy_review"),
            context="identity candidate privacy_review",
            allowed=models.PRIVATE_DETAIL_RISK_VALUES,
        ),
        "promoted_revision_number": _require_optional_nonnegative_integer(
            candidate.get("promoted_revision_number"),
            context="identity candidate promoted_revision_number",
        ),
        "rejection_reason": _require_optional_reason_code(
            candidate.get("rejection_reason"),
            context="identity candidate rejection_reason",
        ),
        "created_at": _require_text(
            candidate.get("created_at"),
            context="identity candidate created_at",
            max_chars=80,
        ),
        "updated_at": _require_text(
            candidate.get("updated_at"),
            context="identity candidate updated_at",
            max_chars=80,
        ),
    }
    return projected


def project_growth_run_for_console(
    run: Mapping[str, object],
) -> dict[str, object]:
    """Project one sanitized run without correlation or root handles."""

    reason_codes = {
        key: _require_enum(
            run.get(key),
            context=f"identity growth run {key}",
            allowed=models.IDENTITY_GROWTH_REASON_CODES,
        )
        for key in (
            "proposal_reason_code",
            "review_reason_code",
            "policy_reason_code",
            "persistence_reason_code",
        )
    }
    attempts = _require_mapping(
        run.get("attempt_count_by_stage"),
        context="identity growth run attempt_count_by_stage",
    )
    attempt_count_by_stage = {
        stage: _require_nonnegative_integer(
            attempts.get(stage),
            context=f"identity growth run {stage} attempts",
        )
        for stage in ("proposal", "review")
    }
    first_consumption = _project_console_consumption(
        run.get("first_consumption")
    )
    validation_error_codes = _require_text_list(
        run.get("validation_error_codes"),
        context="identity growth run validation_error_codes",
        max_items=32,
        max_chars=160,
    )
    projected = {
        "kind": "identity_growth_run",
        "run_kind": _require_enum(
            run.get("run_kind"),
            context="identity growth run run_kind",
            allowed=_RUN_KINDS,
        ),
        "base_revision_number": _require_nonnegative_integer(
            run.get("base_revision_number"),
            context="identity growth run base_revision_number",
        ),
        "root_count": len(
            _require_text_list(
                run.get("root_episode_ids"),
                context="identity growth run root_episode_ids",
                max_items=64,
                max_chars=240,
            )
        ),
        "source_evidence_count": _require_nonnegative_integer(
            run.get("source_evidence_count"),
            context="identity growth run source_evidence_count",
        ),
        "attempt_count_by_stage": attempt_count_by_stage,
        "lifecycle_state": _require_enum(
            run.get("lifecycle_state"),
            context="identity growth run lifecycle_state",
            allowed=models.RUN_LIFECYCLE_STATES,
        ),
        "disposition": _require_enum(
            run.get("disposition"),
            context="identity growth run disposition",
            allowed=_RUN_DISPOSITIONS,
        ),
        **reason_codes,
        "latest_reason_code": _latest_console_run_reason(reason_codes),
        "promoted_revision_number": _require_optional_nonnegative_integer(
            run.get("promoted_revision_number"),
            context="identity growth run promoted_revision_number",
        ),
        "validation_error_codes": sorted(validation_error_codes),
        "first_consumption": first_consumption,
        "post_commit_attempt_count": _require_nonnegative_integer(
            run.get("post_commit_attempt_count"),
            context="identity growth run post_commit_attempt_count",
        ),
        "started_at": _require_text(
            run.get("started_at"),
            context="identity growth run started_at",
            max_chars=80,
        ),
        "completed_at": _require_optional_text(
            run.get("completed_at"),
            context="identity growth run completed_at",
            max_chars=80,
        ),
    }
    return projected


def project_growth_health_for_console(
    health: Mapping[str, object],
) -> dict[str, object]:
    """Validate and label the exact public health projection."""

    expected_keys = frozenset(
        models.CharacterIdentityGrowthHealthV1.__annotations__
    )
    _require_exact_keys(
        health,
        expected=expected_keys,
        context="identity growth health",
    )
    projected: dict[str, object] = {
        "kind": "identity_growth_health",
        "state": _require_enum(
            health.get("state"),
            context="identity growth health state",
            allowed=models.IDENTITY_GROWTH_HEALTH_STATES,
        ),
        "latest_reason_code": _require_enum(
            health.get("latest_reason_code"),
            context="identity growth health latest_reason_code",
            allowed=models.IDENTITY_GROWTH_REASON_CODES,
        ),
        "latest_consumed_revision_number": (
            _require_optional_nonnegative_integer(
                health.get("latest_consumed_revision_number"),
                context=(
                    "identity growth health "
                    "latest_consumed_revision_number"
                ),
            )
        ),
    }
    for field in (
        "routed_count",
        "no_change_count",
        "emerging_candidate_count",
        "ready_candidate_count",
        "rejected_count",
        "failed_count",
        "promoted_count",
        "consumed_count",
        "latest_revision_number",
        "root_count",
        "local_date_count",
    ):
        projected[field] = _require_nonnegative_integer(
            health.get(field),
            context=f"identity growth health {field}",
        )
    return projected


def identity_projection_digest(
    *,
    revision_number: int,
    cognition_context: Mapping[str, object],
    surface_context: Mapping[str, object],
) -> str:
    """Hash the exact latest-identity contexts exposed to runtime consumers."""

    if (
        not isinstance(revision_number, int)
        or isinstance(revision_number, bool)
        or revision_number < 0
    ):
        raise ValueError("identity projection revision_number is invalid")
    payload = {
        "revision_number": revision_number,
        "cognition_context": deepcopy(dict(cognition_context)),
        "surface_context": deepcopy(dict(surface_context)),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def projected_identity_consumer_kinds(
    cognition_context: Mapping[str, object],
) -> list[str]:
    """Return the closed consumers receiving a nonempty identity projection."""

    expected_cognition = frozenset(
        models.CharacterIdentityCognitionContextV1.__annotations__
    )
    actual_cognition = frozenset(cognition_context)
    if actual_cognition != expected_cognition:
        raise ValueError(
            "identity cognition context must contain the exact consumer set"
        )
    consumers = {
        consumer
        for consumer, projection in cognition_context.items()
        if isinstance(projection, Mapping) and projection
    }
    consumers.update({"text", "visual", "naming"})
    if not consumers.issubset(models.IDENTITY_CONSUMER_KINDS):
        raise ValueError("identity projection contains an unknown consumer")
    return sorted(consumers)


def build_identity_proposal_input(
    *,
    current_identity: Mapping[str, object],
    evidence_refs: Sequence[Mapping[str, object]],
    evidence_cards: Sequence[Mapping[str, object]],
    current_candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the closed prompt-safe proposal input."""

    if not 1 <= len(evidence_cards) <= models.IDENTITY_EVIDENCE_CARD_LIMIT:
        raise ValueError(
            "identity proposal requires one to "
            f"{models.IDENTITY_EVIDENCE_CARD_LIMIT} evidence cards"
        )
    if len(current_candidates) > models.IDENTITY_CANDIDATE_PROMPT_LIMIT:
        raise ValueError(
            "identity proposal exceeds the current-candidate prompt limit"
        )

    validated_refs = [
        validate_evidence_ref(ref)
        for ref in evidence_refs
    ]
    refs_by_id = {
        ref["evidence_ref_id"]: ref
        for ref in validated_refs
    }
    if len(refs_by_id) != len(validated_refs):
        raise ValueError("identity proposal evidence refs must be unique")

    validated_cards = []
    for index, raw_card in enumerate(evidence_cards):
        card = _require_mapping(
            raw_card,
            context=f"identity evidence_cards[{index}]",
        )
        evidence_ref_id = _require_text(
            card.get("evidence_ref_id"),
            context=f"identity evidence_cards[{index}].evidence_ref_id",
            max_chars=240,
        )
        evidence_ref = refs_by_id.get(evidence_ref_id)
        if evidence_ref is None:
            raise ValueError(
                "identity evidence card has no repository reference"
            )
        validated_cards.append(
            validate_identity_evidence_card(
                card,
                evidence_ref=evidence_ref,
            )
        )
    card_ids = {
        card["evidence_ref_id"]
        for card in validated_cards
    }
    if card_ids != set(refs_by_id):
        raise ValueError(
            "identity evidence cards and repository refs must match exactly"
        )

    projected_candidates = [
        project_candidate_for_growth_prompt(candidate)
        for candidate in current_candidates
    ]
    candidate_ids = [
        candidate["candidate_id"]
        for candidate in projected_candidates
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("identity prompt candidates must be unique")

    result = {
        "schema_version": models.IDENTITY_PROPOSAL_INPUT_SCHEMA_VERSION,
        "current_identity": project_identity_for_growth_prompt(
            current_identity
        ),
        "evidence_cards": validated_cards,
        "current_candidates": projected_candidates,
        "allowed_paths": sorted(models.ALLOWED_IDENTITY_PATHS),
    }
    return result


def build_identity_review_input(
    *,
    proposal_input: Mapping[str, object],
    proposal: Mapping[str, object],
) -> dict[str, object]:
    """Build an independent review input from the same semantic context."""

    _require_exact_keys(
        proposal_input,
        expected=_PROPOSAL_INPUT_KEYS,
        context="identity proposal input",
    )
    schema_version = proposal_input["schema_version"]
    if schema_version != models.IDENTITY_PROPOSAL_INPUT_SCHEMA_VERSION:
        raise ValueError("identity proposal input schema_version is invalid")
    cards = _require_list(
        proposal_input["evidence_cards"],
        context="identity proposal input evidence_cards",
    )
    candidates = _require_list(
        proposal_input["current_candidates"],
        context="identity proposal input current_candidates",
    )
    evidence_ref_ids = {
        _require_text(
            _require_mapping(
                card,
                context="identity proposal input evidence card",
            ).get("evidence_ref_id"),
            context="identity proposal input evidence_ref_id",
            max_chars=240,
        )
        for card in cards
    }
    candidate_ids = {
        _require_text(
            _require_mapping(
                candidate,
                context="identity proposal input candidate",
            ).get("candidate_id"),
            context="identity proposal input candidate_id",
            max_chars=240,
        )
        for candidate in candidates
    }
    validated_proposal = validate_identity_proposal_decision(
        proposal,
        evidence_ref_ids=evidence_ref_ids,
        candidate_ids=candidate_ids,
    )
    result = {
        "schema_version": models.IDENTITY_REVIEW_INPUT_SCHEMA_VERSION,
        "current_identity": deepcopy(proposal_input["current_identity"]),
        "evidence_cards": deepcopy(cards),
        "current_candidates": deepcopy(candidates),
        "allowed_paths": deepcopy(proposal_input["allowed_paths"]),
        "proposal_decision": validated_proposal,
    }
    return result


def project_candidate_for_growth_prompt(
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Project one current candidate without raw repository references."""

    candidate_id = _require_text(
        candidate.get("candidate_id"),
        context="identity candidate candidate_id",
        max_chars=240,
    )
    status = _require_text(
        candidate.get("status"),
        context=f"identity candidate {candidate_id} status",
        max_chars=40,
    )
    if status not in {"emerging", "ready"}:
        raise ValueError(
            "identity prompt candidate status must be emerging or ready"
        )
    change_kind = _require_text(
        candidate.get("change_kind"),
        context=f"identity candidate {candidate_id} change_kind",
        max_chars=80,
    )
    if change_kind not in models.ACCEPTED_CHANGE_KINDS:
        raise ValueError("identity prompt candidate change_kind is invalid")

    raw_patches = _require_list(
        candidate.get("proposed_changes"),
        context=f"identity candidate {candidate_id} proposed_changes",
    )
    if not 1 <= len(raw_patches) <= models.IDENTITY_PATCH_LIMIT:
        raise ValueError("identity prompt candidate patch count is invalid")
    patches = [
        validate_identity_patch(
            _require_mapping(
                patch,
                context=f"identity candidate {candidate_id} patch",
            )
        )
        for patch in raw_patches
    ]
    patch_paths = {patch["path"] for patch in patches}
    if len(patch_paths) != len(patches):
        raise ValueError("identity prompt candidate has duplicate paths")

    raw_refs = _require_list(
        candidate.get("evidence_refs"),
        context=f"identity candidate {candidate_id} evidence_refs",
    )
    validated_refs = [
        validate_evidence_ref(
            _require_mapping(
                ref,
                context=f"identity candidate {candidate_id} evidence_ref",
            )
        )
        for ref in raw_refs
    ]
    counts = evidence_counts(validated_refs)
    reversal_of_paths = _require_text_list(
        candidate.get("reversal_of_paths"),
        context=f"identity candidate {candidate_id} reversal_of_paths",
        max_items=models.IDENTITY_PATCH_LIMIT,
        max_chars=160,
    )
    if not set(reversal_of_paths).issubset(patch_paths):
        raise ValueError(
            "identity candidate reversal paths must be proposed paths"
        )

    result = {
        "candidate_id": candidate_id,
        "status": status,
        "change_kind": change_kind,
        "proposed_changes": patches,
        "semantic_summary": _require_text(
            candidate.get("semantic_summary"),
            context=f"identity candidate {candidate_id} semantic_summary",
            max_chars=1200,
        ),
        "distinct_episode_count": counts["distinct_episode_count"],
        "distinct_local_dates": counts["distinct_local_dates"],
        "source_scope_kinds": sorted({
            ref["scope_kind"]
            for ref in validated_refs
        }),
        "reversal_of_paths": sorted(reversal_of_paths),
        "character_authorship": _require_enum(
            candidate.get("character_authorship"),
            context=f"identity candidate {candidate_id} authorship",
            allowed=models.CHARACTER_AUTHORSHIP_VALUES,
        ),
        "proposal_confidence": _require_enum(
            candidate.get("proposal_confidence"),
            context=f"identity candidate {candidate_id} proposal confidence",
            allowed=models.CONFIDENCE_VALUES,
        ),
        "review_confidence": _require_enum(
            candidate.get("review_confidence"),
            context=f"identity candidate {candidate_id} review confidence",
            allowed=models.CONFIDENCE_VALUES,
        ),
    }
    return result


def _project_console_change_diff(
    value: object,
    *,
    changed_paths: Sequence[str],
) -> list[dict[str, str]]:
    """Project exact changed paths while withholding before/after values."""

    raw_rows = _require_list(
        value,
        context="identity revision change_diff",
    )
    projected: list[dict[str, str]] = []
    for index, raw_row in enumerate(raw_rows):
        row = _require_mapping(
            raw_row,
            context=f"identity revision change_diff[{index}]",
        )
        path = _require_text(
            row.get("path"),
            context=f"identity revision change_diff[{index}].path",
            max_chars=160,
        )
        if path not in models.ALLOWED_IDENTITY_PATHS:
            raise ValueError("identity revision diff path is invalid")
        value_kind = _require_enum(
            row.get("value_kind"),
            context=f"identity revision change_diff[{index}].value_kind",
            allowed=frozenset({
                "text",
                "integer",
                "semantic_band",
                "closed_enum",
                "text_list",
            }),
        )
        projected.append({
            "path": path,
            "value_kind": value_kind,
            "change": "value_changed",
        })
    if [row["path"] for row in projected] != list(changed_paths):
        raise ValueError(
            "identity revision change_diff must match changed_paths"
        )
    return projected


def _console_evidence_counts(
    raw_refs: Sequence[object],
) -> tuple[int, int]:
    """Count repository roots and local dates without exposing either."""

    roots: set[str] = set()
    local_dates: set[str] = set()
    for index, raw_ref in enumerate(raw_refs):
        ref = _require_mapping(
            raw_ref,
            context=f"identity revision evidence_refs[{index}]",
        )
        roots.add(
            _require_text(
                ref.get("root_episode_id"),
                context=(
                    f"identity revision evidence_refs[{index}]"
                    ".root_episode_id"
                ),
                max_chars=240,
            )
        )
        local_dates.add(
            _require_text(
                ref.get("character_local_date"),
                context=(
                    f"identity revision evidence_refs[{index}]"
                    ".character_local_date"
                ),
                max_chars=40,
            )
        )
    return len(roots), len(local_dates)


def _require_console_identity_paths(
    value: object,
    *,
    context: str,
) -> list[str]:
    """Require a bounded list of canonical identity paths."""

    paths = _require_text_list(
        value,
        context=context,
        max_items=models.IDENTITY_PATCH_LIMIT,
        max_chars=160,
    )
    if not set(paths).issubset(models.ALLOWED_IDENTITY_PATHS):
        raise ValueError(f"{context} contains an unsupported path")
    return sorted(paths)


def _project_console_consumption(
    value: object,
) -> dict[str, object] | None:
    """Project a receipt without episode, correlation, or digest handles."""

    if value is None:
        return None
    receipt = _require_mapping(
        value,
        context="identity growth run first_consumption",
    )
    consumer_kinds = _require_text_list(
        receipt.get("consumer_kinds"),
        context="identity growth run first_consumption consumer_kinds",
        max_items=len(models.IDENTITY_CONSUMER_KINDS),
        max_chars=80,
    )
    if not set(consumer_kinds).issubset(models.IDENTITY_CONSUMER_KINDS):
        raise ValueError("identity consumption consumer_kinds are invalid")
    status = _require_enum(
        receipt.get("status"),
        context="identity growth run first_consumption status",
        allowed=frozenset({"consumed", "mismatch"}),
    )
    return {
        "claimed_at": _require_text(
            receipt.get("claimed_at"),
            context="identity growth run first_consumption claimed_at",
            max_chars=80,
        ),
        "loaded_revision_number": _require_nonnegative_integer(
            receipt.get("loaded_revision_number"),
            context=(
                "identity growth run first_consumption "
                "loaded_revision_number"
            ),
        ),
        "consumer_kinds": sorted(consumer_kinds),
        "status": status,
    }


def _latest_console_run_reason(
    reason_codes: Mapping[str, str],
) -> str:
    """Return the persisted terminal reason used by operator health."""

    return reason_codes["persistence_reason_code"]


def _identity_from_revision(
    revision: Mapping[str, object],
) -> models.CharacterEffectiveIdentityV1:
    """Validate and detach the effective identity from one revision."""

    raw_identity = revision.get("effective_identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("identity revision effective_identity is required")
    return validate_effective_identity(raw_identity)


def _semantic_band_for_number(value: float) -> str:
    """Return the nearest declared semantic band for a unit value."""

    ordered_bands = tuple(models.SEMANTIC_BAND_VALUES)
    band = min(
        ordered_bands,
        key=lambda name: (
            abs(value - models.SEMANTIC_BAND_VALUES[name]),
            ordered_bands.index(name),
        ),
    )
    return band


def _value_at_path(payload: Mapping[str, object], path: str) -> object:
    """Resolve one path from a validated identity."""

    current: object = payload
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise ValueError(
                f"identity prompt projection missing path: {path}"
            )
        current = current[segment]
    return current


def _set_value_at_path(
    payload: dict[str, object],
    path: str,
    value: object,
) -> None:
    """Replace one path in a copied identity."""

    current = payload
    segments = path.split(".")
    for segment in segments[:-1]:
        nested = current.get(segment)
        if not isinstance(nested, dict):
            raise ValueError(
                f"identity prompt projection missing path: {path}"
            )
        current = cast(dict[str, object], nested)
    current[segments[-1]] = value


def _require_mapping(
    value: object,
    *,
    context: str,
) -> Mapping[str, object]:
    """Require one string-keyed mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return cast(Mapping[str, object], value)


def _require_list(value: object, *, context: str) -> list[object]:
    """Require one list."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return list(value)


def _require_text(
    value: object,
    *,
    context: str,
    max_chars: int,
) -> str:
    """Require nonempty bounded text."""

    if not isinstance(value, str):
        raise ValueError(f"{context} must be text")
    text = value.strip()
    if not text or len(text) > max_chars:
        raise ValueError(f"{context} must be nonempty and bounded")
    return text


def _require_optional_text(
    value: object,
    *,
    context: str,
    max_chars: int,
) -> str | None:
    """Require optional bounded text."""

    if value is None:
        return None
    return _require_text(
        value,
        context=context,
        max_chars=max_chars,
    )


def _require_nonnegative_integer(
    value: object,
    *,
    context: str,
) -> int:
    """Require a non-boolean integer at or above zero."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _require_optional_nonnegative_integer(
    value: object,
    *,
    context: str,
) -> int | None:
    """Require an optional nonnegative integer."""

    if value is None:
        return None
    return _require_nonnegative_integer(value, context=context)


def _require_optional_reason_code(
    value: object,
    *,
    context: str,
) -> str | None:
    """Require an optional closed identity reason code."""

    if value is None:
        return None
    return _require_enum(
        value,
        context=context,
        allowed=models.IDENTITY_GROWTH_REASON_CODES,
    )


def _require_text_list(
    value: object,
    *,
    context: str,
    max_items: int,
    max_chars: int,
) -> list[str]:
    """Require a bounded unique text list."""

    raw_items = _require_list(value, context=context)
    if len(raw_items) > max_items:
        raise ValueError(f"{context} exceeds its item limit")
    items = [
        _require_text(
            item,
            context=f"{context}[{index}]",
            max_chars=max_chars,
        )
        for index, item in enumerate(raw_items)
    ]
    if len(items) != len(set(items)):
        raise ValueError(f"{context} must contain unique items")
    return items


def _require_enum(
    value: object,
    *,
    context: str,
    allowed: frozenset[str],
) -> str:
    """Require one closed enum value."""

    text = _require_text(value, context=context, max_chars=80)
    if text not in allowed:
        raise ValueError(f"{context} has an unsupported value")
    return text


def _require_exact_keys(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    """Require one exact object key set."""

    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unknown = sorted(actual.difference(expected))
        raise ValueError(
            f"{context} key mismatch; missing={missing}, unknown={unknown}"
        )
