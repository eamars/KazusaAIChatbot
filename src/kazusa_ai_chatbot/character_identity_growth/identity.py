"""Pure identity, lineage, transition, and health operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_effective_identity,
    validate_evidence_ref,
    validate_identity_patch,
)


def apply_identity_patches(
    current_identity: Mapping[str, object],
    patches: Sequence[Mapping[str, object]],
) -> tuple[
    models.CharacterEffectiveIdentityV1,
    list[models.IdentityChangeDiffV1],
]:
    """Apply strict replacements to a copied full identity snapshot."""

    validated_identity = validate_effective_identity(current_identity)
    validated_patches = [
        validate_identity_patch(patch)
        for patch in patches
    ]
    paths = [patch["path"] for patch in validated_patches]
    if len(paths) != len(set(paths)):
        raise ValueError("identity patches contain a duplicate path")

    updated = deepcopy(validated_identity)
    diffs: list[models.IdentityChangeDiffV1] = []
    for patch in sorted(
        validated_patches,
        key=lambda row: row["path"],
    ):
        path = patch["path"]
        before = deepcopy(_value_at_path(updated, path))
        after = _replacement_value(patch)
        if before == after:
            raise ValueError(f"identity patch is a no-op: {path}")
        _set_value_at_path(updated, path, deepcopy(after))
        diff: models.IdentityChangeDiffV1 = {
            "path": path,
            "value_kind": patch["value_kind"],
            "before": before,
            "after": deepcopy(after),
        }
        diffs.append(diff)

    validated_updated = validate_effective_identity(updated)
    return_value = (validated_updated, diffs)
    return return_value


def diff_effective_identities(
    before_identity: Mapping[str, object],
    after_identity: Mapping[str, object],
) -> list[models.IdentityChangeDiffV1]:
    """Return the exact sorted leaf diff between two full identities."""

    before = validate_effective_identity(before_identity)
    after = validate_effective_identity(after_identity)
    diffs: list[models.IdentityChangeDiffV1] = []
    for path in sorted(models.ALLOWED_IDENTITY_PATHS):
        before_value = deepcopy(_value_at_path(before, path))
        after_value = deepcopy(_value_at_path(after, path))
        if before_value == after_value:
            continue
        diffs.append({
            "path": path,
            "value_kind": _value_kind_for_path(path),
            "before": before_value,
            "after": after_value,
        })
    return diffs


def dedupe_evidence_refs(
    evidence_refs: Sequence[Mapping[str, object]],
) -> list[models.IdentityEvidenceRefV1]:
    """Collapse direct and reflection-derived cards by root episode."""

    by_root: dict[str, models.IdentityEvidenceRefV1] = {}
    for raw_ref in evidence_refs:
        current = validate_evidence_ref(raw_ref)
        root_id = current["root_episode_id"]
        existing = by_root.get(root_id)
        if existing is None:
            by_root[root_id] = deepcopy(current)
            continue
        _require_matching_root_metadata(existing, current)
        merged_derivatives = sorted(set(
            existing["derived_reflection_run_ids"]
            + current["derived_reflection_run_ids"]
        ))
        if (
            existing["source_kind"] == "daily_reflection"
            and current["source_kind"] == "settled_episode"
        ):
            merged = deepcopy(current)
        else:
            merged = deepcopy(existing)
        merged["derived_reflection_run_ids"] = merged_derivatives
        by_root[root_id] = merged

    deduped = sorted(
        by_root.values(),
        key=lambda row: (row["captured_at"], row["root_episode_id"]),
    )
    return deduped


def evidence_counts(
    evidence_refs: Sequence[Mapping[str, object]],
) -> models.IdentityEvidenceCountsV1:
    """Return cadence counts from deduplicated repository roots."""

    deduped = dedupe_evidence_refs(evidence_refs)
    local_dates = sorted({
        row["character_local_date"]
        for row in deduped
    })
    return_value: models.IdentityEvidenceCountsV1 = {
        "distinct_episode_count": len(deduped),
        "distinct_local_dates": local_dates,
    }
    return return_value


def candidate_transition_allowed(
    current_status: str,
    target_status: str,
) -> bool:
    """Return whether one candidate lifecycle edge is declared."""

    if current_status not in models.CANDIDATE_TRANSITIONS:
        raise ValueError(f"unknown current candidate status: {current_status}")
    if target_status not in models.CANDIDATE_TRANSITIONS:
        raise ValueError(f"unknown target candidate status: {target_status}")
    allowed = target_status in models.CANDIDATE_TRANSITIONS[current_status]
    return allowed


def derive_growth_health_state(
    *,
    latest_revision_number: int,
    receipt_status: object,
    latest_run_lifecycle_state: str | None,
    latest_reason_code: str,
    ready_candidate_count: int,
    emerging_candidate_count: int,
) -> models.IdentityGrowthHealthState:
    """Apply the declared operator-health precedence."""

    _require_nonnegative_integer(
        latest_revision_number,
        context="latest_revision_number",
    )
    _require_nonnegative_integer(
        ready_candidate_count,
        context="ready_candidate_count",
    )
    _require_nonnegative_integer(
        emerging_candidate_count,
        context="emerging_candidate_count",
    )
    if receipt_status not in {None, "consumed", "mismatch"}:
        raise ValueError("receipt_status must be consumed, mismatch, or None")
    if (
        latest_run_lifecycle_state is not None
        and latest_run_lifecycle_state not in models.RUN_LIFECYCLE_STATES
    ):
        raise ValueError(
            "latest_run_lifecycle_state is not supported: "
            f"{latest_run_lifecycle_state}"
        )
    if latest_reason_code not in models.IDENTITY_GROWTH_REASON_CODES:
        raise ValueError(
            f"latest_reason_code is not supported: {latest_reason_code}"
        )

    if receipt_status == "mismatch":
        return "consumption_error"
    pipeline_reasons = {
        "proposal_contract_failed",
        "review_contract_failed",
        "promotion_write_failed",
    }
    if (
        latest_run_lifecycle_state == "failed"
        or latest_reason_code in pipeline_reasons
    ):
        return "pipeline_error"
    if latest_revision_number > 0 and receipt_status is None:
        return "awaiting_consumption"
    if ready_candidate_count > 0:
        return "promotion_ready"
    semantic_reasons = {
        "review_rejected",
        "privacy_blocked",
        "contradiction_blocked",
    }
    if latest_reason_code in semantic_reasons:
        return "semantic_rejection"
    waiting_reasons = {
        "candidate_emerging",
        "cadence_wait",
        "duplicate_root",
        "stale_base",
    }
    if emerging_candidate_count > 0 or latest_reason_code in waiting_reasons:
        return "waiting_for_evidence"
    if latest_revision_number > 0 and receipt_status == "consumed":
        return "healthy_active"
    return "healthy_idle"


def _replacement_value(patch: models.IdentityPatchV1) -> object:
    """Return the effective replacement represented by one tagged patch."""

    value_kind = patch["value_kind"]
    if value_kind == "text":
        return patch["replacement_text"]
    if value_kind == "integer":
        return patch["replacement_integer"]
    if value_kind == "semantic_band":
        return models.SEMANTIC_BAND_VALUES[patch["replacement_band"]]
    if value_kind == "closed_enum":
        return patch["replacement_enum"]
    if value_kind == "text_list":
        return list(patch["replacement_items"])
    raise ValueError(f"unsupported identity patch value_kind: {value_kind}")


def _value_kind_for_path(path: str) -> models.PatchValueKind:
    """Return the tagged value kind for one canonical identity path."""

    if path in models.TEXT_IDENTITY_PATHS:
        return "text"
    if path in models.INTEGER_IDENTITY_PATHS:
        return "integer"
    if path in models.NUMERIC_IDENTITY_PATHS:
        return "semantic_band"
    if path in models.ENUM_IDENTITY_PATHS:
        return "closed_enum"
    if path in models.TEXT_LIST_IDENTITY_PATHS:
        return "text_list"
    raise ValueError(f"unsupported identity path: {path}")


def _value_at_path(
    payload: Mapping[str, object],
    path: str,
) -> object:
    """Resolve one already-validated identity path."""

    current: object = payload
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise ValueError(f"effective identity missing path: {path}")
        current = current[segment]
    return current


def _set_value_at_path(
    payload: dict[str, object],
    path: str,
    value: object,
) -> None:
    """Replace one already-validated identity path."""

    segments = path.split(".")
    current: dict[str, object] = payload
    for segment in segments[:-1]:
        nested = current.get(segment)
        if not isinstance(nested, dict):
            raise ValueError(f"effective identity missing path: {path}")
        current = cast(dict[str, object], nested)
    current[segments[-1]] = value


def _require_matching_root_metadata(
    existing: models.IdentityEvidenceRefV1,
    current: models.IdentityEvidenceRefV1,
) -> None:
    """Fail when one repository root carries conflicting cadence metadata."""

    compared_fields = (
        "character_local_date",
        "scope_kind",
    )
    conflicts = [
        field_name
        for field_name in compared_fields
        if existing[field_name] != current[field_name]
    ]
    if conflicts:
        root_id = existing["root_episode_id"]
        raise ValueError(
            f"root {root_id} has conflicting metadata: {conflicts}"
        )


def _require_nonnegative_integer(value: object, *, context: str) -> None:
    """Require one non-boolean integer count."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{context} must be a non-negative integer")
