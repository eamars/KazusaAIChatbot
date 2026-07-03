"""Shared operator helpers for memory-lane cleanup scripts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from bson import json_util

from scripts._db_export import (
    configure_logging,
    configure_stdout,
    load_project_env,
)
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.interaction_style_images import (
    validate_interaction_style_overlay,
)
from kazusa_ai_chatbot.db.script_operations import (
    count_lane_cleanup_field_values,
    count_lane_cleanup_rows,
    delete_lane_cleanup_row,
    find_lane_cleanup_row,
    load_lane_cleanup_rows,
    update_lane_cleanup_row,
)
from kazusa_ai_chatbot.time_boundary import (
    parse_storage_utc_datetime,
    storage_utc_now,
    storage_utc_now_iso,
)


REPORT_VERSION = "memory_lane_cleanup.v1"
DEFAULT_AFFINITY = 500
EPISODE_TTL_HOURS = 48
MANUAL_REVIEW = "manual_review"
KEEP = "keep"


def _text(value: object) -> str:
    """Return a stripped string for untrusted document fields."""

    if isinstance(value, str):
        text_value = value.strip()
    else:
        text_value = ""
    return text_value


def _list(value: object) -> list[Any]:
    """Return a list for untrusted document fields."""

    if isinstance(value, list):
        list_value = value
    else:
        list_value = []
    return list_value


def _dict(value: object) -> dict[str, Any]:
    """Return a dictionary for untrusted document fields."""

    if isinstance(value, dict):
        dict_value = dict(value)
    else:
        dict_value = {}
    return dict_value


def _storage_time_valid(value: object) -> bool:
    """Return whether a value is a parseable storage UTC timestamp."""

    if not isinstance(value, str) or not value.strip():
        return_value = False
        return return_value
    try:
        parse_storage_utc_datetime(value)
    except ValueError:
        return_value = False
        return return_value
    return_value = True
    return return_value


def _scrub_for_report(value: Any) -> Any:
    """Remove large derived fields before hashing or writing reports."""

    if isinstance(value, dict):
        scrubbed_dict = {
            key: _scrub_for_report(item)
            for key, item in value.items()
            if key != "embedding"
        }
        return scrubbed_dict
    if isinstance(value, list):
        scrubbed_list = [_scrub_for_report(item) for item in value]
        return scrubbed_list
    return value


def _stable_row_hash(row: Mapping[str, Any]) -> str:
    """Build a stable hash for exact-row drift detection."""

    scrubbed = _scrub_for_report(dict(row))
    rendered = json_util.dumps(
        scrubbed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    row_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return row_hash


def _write_report(output_path: Path, report: Mapping[str, Any]) -> None:
    """Write a cleanup report as Extended JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_report = json_util.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(rendered_report, encoding="utf-8")


def _read_report(input_path: Path) -> dict[str, Any]:
    """Read a cleanup report from Extended JSON."""

    report_text = input_path.read_text(encoding="utf-8")
    loaded_report = json_util.loads(report_text)
    if not isinstance(loaded_report, dict):
        raise ValueError(f"cleanup report must be a JSON object: {input_path}")
    return_value = dict(loaded_report)
    return return_value


def _finding(
    *,
    row_key: str,
    issue_code: str,
    issue_description: str,
    recommended_action: str,
    severity: str,
    evidence_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a lane cleanup finding."""

    finding = {
        "row_key": row_key,
        "issue_code": issue_code,
        "issue_description": issue_description,
        "recommended_action": recommended_action,
        "severity": severity,
        "evidence_fields": dict(evidence_fields),
    }
    return finding


def _action(
    *,
    collection_name: str,
    row_key: str,
    selector: Mapping[str, Any],
    expected_current_hash: str,
    action_name: str,
    set_fields: Mapping[str, Any],
    reason_code: str,
    unset_fields: Sequence[str] = (),
    push_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one exact-row cleanup action."""

    action_doc = {
        "collection_name": collection_name,
        "row_key": row_key,
        "selector": dict(selector),
        "expected_current_hash": expected_current_hash,
        "action": action_name,
        "reason_code": reason_code,
        "set_fields": dict(set_fields),
        "unset_fields": list(unset_fields),
        "push_fields": dict(push_fields or {}),
    }
    return action_doc


def _base_report(*, lane: str, mode: str) -> dict[str, Any]:
    """Build common report metadata."""

    report = {
        "report_version": REPORT_VERSION,
        "lane": lane,
        "mode": mode,
        "generated_at": storage_utc_now_iso(),
        "counts_before": {},
        "counts_after": {},
        "findings": [],
        "planned_actions": [],
        "applied_actions": [],
        "blocked_actions": [],
        "warnings": [],
    }
    return report


async def _collection_counts(collection_name: str) -> dict[str, int]:
    """Return basic active/total counts for a collection."""

    total_count = await count_lane_cleanup_rows(
        collection_name=collection_name,
        filter_doc={},
    )
    active_count = await count_lane_cleanup_rows(
        collection_name=collection_name,
        filter_doc={"status": "active"},
    )
    counts = {"total": total_count, "active": active_count}
    return counts


async def _load_collection(
    *,
    collection_name: str,
    sort_doc: Sequence[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Load a cleanup collection without vector embeddings."""

    rows = await load_lane_cleanup_rows(
        collection_name=collection_name,
        filter_doc={},
        projection={"embedding": 0},
        sort_doc=sort_doc,
    )
    return rows


async def _verify_current_hash(action_doc: Mapping[str, Any]) -> tuple[bool, str]:
    """Check whether the dry-run row still matches the live row."""

    current_row = await find_lane_cleanup_row(
        collection_name=str(action_doc["collection_name"]),
        filter_doc=_dict(action_doc["selector"]),
        projection={"embedding": 0},
    )
    if current_row is None:
        result = (False, "row_missing")
        return result
    current_hash = _stable_row_hash(current_row)
    expected_hash = _text(action_doc.get("expected_current_hash"))
    if current_hash != expected_hash:
        result = (False, "row_hash_changed")
        return result
    result = (True, "")
    return result


async def _apply_report_actions(
    *,
    input_path: Path,
    output_path: Path,
    lane: str,
    allowed_action_classes: set[str] | None = None,
) -> None:
    """Apply exact-row actions from a reviewed dry-run report."""

    input_report = _read_report(input_path)
    if input_report.get("report_version") != REPORT_VERSION:
        raise ValueError("input report version is not supported")
    if input_report.get("lane") != lane:
        raise ValueError(f"input report lane must be {lane}")

    report = _base_report(lane=lane, mode="apply")
    report["input_report"] = str(input_path)
    planned_actions = _list(input_report.get("planned_actions"))
    for raw_action in planned_actions:
        if not isinstance(raw_action, dict):
            continue
        action_doc = dict(raw_action)
        action_name = _text(action_doc.get("action"))
        if allowed_action_classes is not None:
            if action_name not in allowed_action_classes:
                report["blocked_actions"].append({
                    "row_key": action_doc.get("row_key", ""),
                    "action": action_name,
                    "blocked_reason": "action_class_not_requested",
                })
                continue

        hash_ok, blocked_reason = await _verify_current_hash(action_doc)
        if not hash_ok:
            report["blocked_actions"].append({
                "row_key": action_doc.get("row_key", ""),
                "action": action_name,
                "blocked_reason": blocked_reason,
            })
            continue

        if action_name == "delete":
            mutation_result = await delete_lane_cleanup_row(
                collection_name=str(action_doc["collection_name"]),
                filter_doc=_dict(action_doc["selector"]),
            )
        else:
            mutation_result = await update_lane_cleanup_row(
                collection_name=str(action_doc["collection_name"]),
                filter_doc=_dict(action_doc["selector"]),
                set_fields=_dict(action_doc.get("set_fields")),
                unset_fields=_list(action_doc.get("unset_fields")),
                push_fields=_dict(action_doc.get("push_fields")),
            )

        applied_action = {
            "row_key": action_doc.get("row_key", ""),
            "action": action_name,
            "reason_code": action_doc.get("reason_code", ""),
            "mutation_result": mutation_result,
        }
        report["applied_actions"].append(applied_action)

    _write_report(output_path, report)
    print(f"wrote apply report to {output_path}")


def _source_refs_missing_or_invalid(row: Mapping[str, Any]) -> bool:
    """Return whether a user-memory source-ref list is absent or invalid."""

    source_refs = row.get("source_refs")
    if not isinstance(source_refs, list) or not source_refs:
        return_value = True
        return return_value

    for source_ref in source_refs:
        if not isinstance(source_ref, dict):
            return_value = True
            return return_value
        source = _text(source_ref.get("source"))
        timestamp = _text(source_ref.get("timestamp"))
        row_id = _text(source_ref.get("conversation_row_id"))
        message_id = _text(source_ref.get("message_id"))
        episode_id = _text(source_ref.get("episode_id"))
        if source not in {
            "conversation_history",
            "cognitive_episode",
            "internal_source_case",
        }:
            return_value = True
            return return_value
        if not _storage_time_valid(timestamp):
            return_value = True
            return return_value
        if not (row_id or message_id or episode_id):
            return_value = True
            return return_value

    return_value = False
    return return_value


def _legacy_source_acknowledged(row: Mapping[str, Any]) -> bool:
    """Return whether missing source refs were already marked as legacy."""

    acknowledged = (
        _text(row.get("authority")) == "legacy_unverified_continuity"
        and _text(row.get("truth_status")) == "legacy_unverified"
        and _text(row.get("origin")) == "source_missing_migration"
    )
    return acknowledged


async def _build_user_memory_units_report() -> dict[str, Any]:
    """Build the user-memory cleanup dry-run report."""

    rows = await _load_collection(
        collection_name="user_memory_units",
        sort_doc=[("global_user_id", 1), ("unit_id", 1)],
    )
    profiles = await _load_collection(
        collection_name="user_profiles",
        sort_doc=[("global_user_id", 1)],
    )
    profile_ids = {
        _text(profile.get("global_user_id"))
        for profile in profiles
        if _text(profile.get("global_user_id"))
    }
    report = _base_report(lane="user_memory_units", mode="dry_run")
    report["counts_before"] = await _collection_counts("user_memory_units")
    write_time = _text(report["generated_at"])

    for row in rows:
        unit_id = _text(row.get("unit_id"))
        row_key = unit_id or _text(row.get("memory_unit_id")) or str(row.get("_id", ""))
        row_hash = _stable_row_hash(row)
        global_user_id = _text(row.get("global_user_id"))
        unit_type = _text(row.get("unit_type"))
        status = _text(row.get("status")) or "active"
        selector = {"unit_id": unit_id} if unit_id else {"_id": row.get("_id")}
        evidence = {
            "global_user_id": global_user_id,
            "unit_type": unit_type,
            "status": status,
        }

        if not global_user_id or global_user_id not in profile_ids:
            if status != "active":
                report["findings"].append(_finding(
                    row_key=row_key,
                    issue_code="archived_target_profile_missing",
                    issue_description=(
                        "Memory unit target profile is missing, but the row "
                        "is already inactive."
                    ),
                    recommended_action=KEEP,
                    severity="info",
                    evidence_fields=evidence,
                ))
                continue
            report["findings"].append(_finding(
                row_key=row_key,
                issue_code="target_profile_missing",
                issue_description="Memory unit target profile is missing.",
                recommended_action="archive",
                severity="error",
                evidence_fields=evidence,
            ))
            report["planned_actions"].append(_action(
                collection_name="user_memory_units",
                row_key=row_key,
                selector=selector,
                expected_current_hash=row_hash,
                action_name="archive",
                reason_code="target_profile_missing",
                set_fields={
                    "status": "archived",
                    "archived_at": write_time,
                    "updated_at": write_time,
                },
                push_fields={
                    "merge_history": {
                        "operation": "lane_integrity_migration",
                        "action": "archive",
                        "reason_code": "target_profile_missing",
                        "previous_status": status,
                        "timestamp": write_time,
                    }
                },
            ))
            continue

        if _source_refs_missing_or_invalid(row):
            if _legacy_source_acknowledged(row):
                report["findings"].append(_finding(
                    row_key=row_key,
                    issue_code="legacy_source_refs_acknowledged",
                    issue_description=(
                        "Memory unit has no valid source_refs but is marked "
                        "as legacy-unverified continuity."
                    ),
                    recommended_action=KEEP,
                    severity="info",
                    evidence_fields=evidence,
                ))
            else:
                report["findings"].append(_finding(
                    row_key=row_key,
                    issue_code="missing_source_refs",
                    issue_description="Memory unit has no valid source_refs.",
                    recommended_action="mark_legacy_unverified",
                    severity="warning",
                    evidence_fields=evidence,
                ))
                report["planned_actions"].append(_action(
                    collection_name="user_memory_units",
                    row_key=row_key,
                    selector=selector,
                    expected_current_hash=row_hash,
                    action_name="mark_legacy_unverified",
                    reason_code="missing_source_refs",
                    set_fields={
                        "authority": "legacy_unverified_continuity",
                        "truth_status": "legacy_unverified",
                        "origin": "source_missing_migration",
                        "updated_at": write_time,
                    },
                    push_fields={
                        "merge_history": {
                            "operation": "lane_integrity_migration",
                            "action": "mark_legacy_unverified",
                            "reason_code": "missing_source_refs",
                            "previous_status": status,
                            "previous_unit_type": unit_type,
                            "timestamp": write_time,
                        }
                    },
                ))

        if unit_type == "active_commitment" and not _text(row.get("due_at")):
            report["findings"].append(_finding(
                row_key=row_key,
                issue_code="active_commitment_missing_due_manual_review",
                issue_description=(
                    "Active commitment has no due_at; semantic ongoing-rule "
                    "review is required."
                ),
                recommended_action=MANUAL_REVIEW,
                severity="warning",
                evidence_fields=evidence,
            ))

    return report


async def _profile_reference_count_maps() -> dict[str, dict[str, int]]:
    """Count protected profile references in one grouped pass per collection."""

    count_maps = {
        "conversation_history": await count_lane_cleanup_field_values(
            collection_name="conversation_history",
            field_name="global_user_id",
        ),
        "user_memory_units": await count_lane_cleanup_field_values(
            collection_name="user_memory_units",
            field_name="global_user_id",
        ),
        "memory": await count_lane_cleanup_field_values(
            collection_name="memory",
            field_name="source_global_user_id",
        ),
        "conversation_episode_state": await count_lane_cleanup_field_values(
            collection_name="conversation_episode_state",
            field_name="global_user_id",
        ),
        "scheduled_events": await count_lane_cleanup_field_values(
            collection_name="scheduled_events",
            field_name="source_user_id",
        ),
    }
    return count_maps


def _profile_reference_counts(
    global_user_id: str,
    count_maps: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    """Return protected reference counts for one profile id."""

    counts = {
        collection_name: int(collection_counts.get(global_user_id, 0))
        for collection_name, collection_counts in count_maps.items()
    }
    return counts


def _profile_is_empty(row: Mapping[str, Any]) -> bool:
    """Return whether a profile is structurally empty."""

    platform_accounts = _list(row.get("platform_accounts"))
    suspected_aliases = _list(row.get("suspected_aliases"))
    relationship = _text(row.get("last_relationship_insight"))
    affinity = row.get("affinity")
    empty_affinity = affinity in (None, DEFAULT_AFFINITY)
    is_empty = (
        not platform_accounts
        and not suspected_aliases
        and not relationship
        and empty_affinity
    )
    return is_empty


async def _build_user_profiles_report() -> dict[str, Any]:
    """Build the user-profile cleanup dry-run report."""

    rows = await _load_collection(
        collection_name="user_profiles",
        sort_doc=[("global_user_id", 1)],
    )
    report = _base_report(lane="user_profiles", mode="dry_run")
    report["counts_before"] = await _collection_counts("user_profiles")
    reference_count_maps = await _profile_reference_count_maps()

    for row in rows:
        global_user_id = _text(row.get("global_user_id"))
        row_key = global_user_id or str(row.get("_id", ""))
        row_hash = _stable_row_hash(row)
        reference_counts = _profile_reference_counts(
            global_user_id,
            reference_count_maps,
        )
        reference_total = sum(reference_counts.values())
        evidence = {
            "global_user_id": global_user_id,
            "reference_counts": reference_counts,
        }
        selector = {"global_user_id": global_user_id}

        if not global_user_id:
            report["findings"].append(_finding(
                row_key=row_key,
                issue_code="missing_global_user_id",
                issue_description="Profile has no global_user_id.",
                recommended_action=MANUAL_REVIEW,
                severity="error",
                evidence_fields=evidence,
            ))
            continue

        if _profile_is_empty(row) and reference_total == 0:
            report["findings"].append(_finding(
                row_key=row_key,
                issue_code="empty_no_platform_unreferenced",
                issue_description="Profile is empty and has no protected refs.",
                recommended_action="archive_empty",
                severity="warning",
                evidence_fields=evidence,
            ))
            report["planned_actions"].append(_action(
                collection_name="user_profiles",
                row_key=row_key,
                selector=selector,
                expected_current_hash=row_hash,
                action_name="delete",
                reason_code="empty_no_platform_unreferenced",
                set_fields={},
            ))
        elif _profile_is_empty(row):
            report["findings"].append(_finding(
                row_key=row_key,
                issue_code="no_platform_history_linked",
                issue_description="Profile has no platform account but has refs.",
                recommended_action=MANUAL_REVIEW,
                severity="warning",
                evidence_fields=evidence,
            ))

    return report


def _memory_privacy_valid(row: Mapping[str, Any]) -> bool:
    """Return whether a shared-memory row has usable privacy review."""

    privacy_review = _dict(row.get("privacy_review"))
    details_removed = privacy_review.get("user_details_removed") is True
    risk = _text(privacy_review.get("private_detail_risk"))
    boundary = _text(privacy_review.get("boundary_assessment"))
    valid = details_removed and risk in {"low", "medium"} and bool(boundary)
    return valid


async def _build_shared_memory_report() -> dict[str, Any]:
    """Build the shared-memory cleanup dry-run report."""

    rows = await _load_collection(
        collection_name="memory",
        sort_doc=[("updated_at", -1), ("timestamp", -1)],
    )
    report = _base_report(lane="shared_memory", mode="dry_run")
    report["counts_before"] = await _collection_counts("memory")
    write_time = _text(report["generated_at"])

    for row in rows:
        memory_unit_id = _text(row.get("memory_unit_id"))
        row_key = memory_unit_id or str(row.get("_id", ""))
        row_hash = _stable_row_hash(row)
        status = _text(row.get("status")) or "active"
        authority = _text(row.get("authority"))
        source_kind = _text(row.get("source_kind"))
        source_global_user_id = _text(row.get("source_global_user_id"))
        evidence_refs = _list(row.get("evidence_refs"))
        selector = {"memory_unit_id": memory_unit_id}
        evidence = {
            "memory_unit_id": memory_unit_id,
            "status": status,
            "authority": authority,
            "source_kind": source_kind,
            "source_global_user_id": source_global_user_id,
        }
        should_archive = False
        reason_code = ""
        if status == "active" and source_global_user_id:
            should_archive = True
            reason_code = "wrong_lane_user_scoped"
        elif status == "active" and authority != "seed":
            if not evidence_refs or not _memory_privacy_valid(row):
                should_archive = True
                reason_code = "missing_provenance"

        if should_archive:
            report["findings"].append(_finding(
                row_key=row_key,
                issue_code=reason_code,
                issue_description="Active shared memory lacks shared-lane proof.",
                recommended_action="archive",
                severity="error",
                evidence_fields=evidence,
            ))
            report["planned_actions"].append(_action(
                collection_name="memory",
                row_key=row_key,
                selector=selector,
                expected_current_hash=row_hash,
                action_name="archive",
                reason_code=reason_code,
                set_fields={
                    "status": "expired",
                    "updated_at": write_time,
                    "lane_cleanup": {
                        "operation": "shared_memory_lane_cleanup",
                        "reason_code": reason_code,
                        "applied_at": write_time,
                    },
                },
            ))

    return report


def _style_scope_invalid(row: Mapping[str, Any]) -> bool:
    """Return whether a style-image scope shape is invalid."""

    scope_type = _text(row.get("scope_type"))
    style_image_id = _text(row.get("style_image_id"))
    global_user_id = _text(row.get("global_user_id"))
    platform = _text(row.get("platform"))
    platform_channel_id = _text(row.get("platform_channel_id"))
    if scope_type == "user":
        expected_id = f"user:{global_user_id}" if global_user_id else ""
        invalid = not global_user_id or style_image_id != expected_id
        return invalid
    if scope_type == "group_channel":
        expected_id = (
            f"group_channel:{platform}:{platform_channel_id}"
            if platform and platform_channel_id
            else ""
        )
        invalid = (
            not platform
            or not platform_channel_id
            or bool(global_user_id)
            or style_image_id != expected_id
        )
        return invalid
    return_value = True
    return return_value


async def _build_interaction_style_report() -> dict[str, Any]:
    """Build the interaction-style cleanup dry-run report."""

    rows = await _load_collection(
        collection_name="interaction_style_images",
        sort_doc=[("style_image_id", 1)],
    )
    report = _base_report(lane="interaction_style_images", mode="dry_run")
    report["counts_before"] = await _collection_counts("interaction_style_images")
    write_time = _text(report["generated_at"])

    for row in rows:
        style_image_id = _text(row.get("style_image_id"))
        row_key = style_image_id or str(row.get("_id", ""))
        row_hash = _stable_row_hash(row)
        status = _text(row.get("status"))
        source_ids = _list(row.get("source_reflection_run_ids"))
        selector = {"style_image_id": style_image_id}
        evidence = {
            "style_image_id": style_image_id,
            "scope_type": _text(row.get("scope_type")),
            "status": status,
            "revision": row.get("revision"),
        }
        issue_code = ""
        if _style_scope_invalid(row):
            issue_code = "scope_shape_invalid"
        elif status == "active" and not source_ids:
            issue_code = "missing_source_evidence"
        else:
            try:
                validate_interaction_style_overlay(_dict(row.get("overlay")))
            except ValueError:
                issue_code = "overlay_structure_invalid"

        if issue_code:
            report["findings"].append(_finding(
                row_key=row_key,
                issue_code=issue_code,
                issue_description="Style image is unsafe for runtime use.",
                recommended_action="disable_runtime_use",
                severity="error",
                evidence_fields=evidence,
            ))
            report["planned_actions"].append(_action(
                collection_name="interaction_style_images",
                row_key=row_key,
                selector=selector,
                expected_current_hash=row_hash,
                action_name="disable_runtime_use",
                reason_code=issue_code,
                set_fields={
                    "status": "disabled",
                    "updated_at": write_time,
                    "lane_cleanup": {
                        "operation": "interaction_style_images_lane_cleanup",
                        "reason_code": issue_code,
                        "applied_at": write_time,
                    },
                },
            ))

    return report


def _episode_due_action(row: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Classify one conversation-episode state lifecycle action."""

    status = _text(row.get("status")) or "active"
    expires_at = _text(row.get("expires_at"))
    updated_at = _text(row.get("updated_at"))
    if status != "active":
        result = (KEEP, "non_active_history", {})
        return result
    if expires_at:
        try:
            expiry = parse_storage_utc_datetime(expires_at)
        except ValueError:
            result = (MANUAL_REVIEW, "invalid_expiry_active", {})
            return result
        if expiry <= storage_utc_now():
            result = ("close_expired", "expired_active", {})
            return result
        result = (KEEP, "ok_active_current", {})
        return result
    if not _storage_time_valid(updated_at):
        result = (MANUAL_REVIEW, "missing_expiry_active", {})
        return result
    updated = parse_storage_utc_datetime(updated_at)
    derived_expiry_iso = (
        updated + timedelta(hours=EPISODE_TTL_HOURS)
    ).isoformat()
    if parse_storage_utc_datetime(derived_expiry_iso) <= storage_utc_now():
        result = ("close_expired", "missing_expiry_active", {})
        return result
    result = (
        "set_derived_expiry",
        "missing_expiry_active",
        {"expires_at": derived_expiry_iso},
    )
    return result


async def _build_conversation_episode_report() -> dict[str, Any]:
    """Build the conversation-episode-state cleanup dry-run report."""

    rows = await _load_collection(
        collection_name="conversation_episode_state",
        sort_doc=[("platform", 1), ("platform_channel_id", 1), ("global_user_id", 1)],
    )
    report = _base_report(lane="conversation_episode_state", mode="dry_run")
    report["counts_before"] = await _collection_counts("conversation_episode_state")
    write_time = _text(report["generated_at"])

    for row in rows:
        episode_state_id = _text(row.get("episode_state_id"))
        row_key = episode_state_id or str(row.get("_id", ""))
        row_hash = _stable_row_hash(row)
        selector = {
            "platform": _text(row.get("platform")),
            "platform_channel_id": _text(row.get("platform_channel_id")),
            "global_user_id": _text(row.get("global_user_id")),
        }
        action_name, issue_code, extra_fields = _episode_due_action(row)
        evidence = {
            "episode_state_id": episode_state_id,
            "status": _text(row.get("status")),
            "expires_at": _text(row.get("expires_at")),
            "updated_at": _text(row.get("updated_at")),
        }
        if action_name == KEEP:
            continue
        report["findings"].append(_finding(
            row_key=row_key,
            issue_code=issue_code,
            issue_description="Conversation episode state lifecycle needs cleanup.",
            recommended_action=action_name,
            severity="warning",
            evidence_fields=evidence,
        ))
        if action_name == MANUAL_REVIEW:
            continue

        set_fields = {
            "updated_at": write_time,
            "lifecycle_repair": {
                "operation": "conversation_episode_state_lane_cleanup",
                "action": action_name,
                "reason_code": issue_code,
                "applied_at": write_time,
            },
        }
        if action_name == "close_expired":
            set_fields.update({
                "status": "closed",
                "closed_at": write_time,
                "closed_reason": "expired_active_lane_repair",
            })
        else:
            set_fields.update(extra_fields)
        report["planned_actions"].append(_action(
            collection_name="conversation_episode_state",
            row_key=row_key,
            selector=selector,
            expected_current_hash=row_hash,
            action_name=action_name,
            reason_code=issue_code,
            set_fields=set_fields,
        ))

    return report


async def _build_character_state_report(*, dry_run_repair: bool) -> dict[str, Any]:
    """Build the character-state cleanup dry-run report."""

    rows = await _load_collection(
        collection_name="character_state",
        sort_doc=[("_id", 1)],
    )
    report = _base_report(lane="character_state", mode="dry_run")
    report["counts_before"] = await _collection_counts("character_state")

    global_rows = [
        row for row in rows
        if _text(row.get("_id")) == "global" or row.get("_id") == "global"
    ]
    if not global_rows:
        report["findings"].append(_finding(
            row_key="global",
            issue_code="malformed_structure",
            issue_description="Global character_state row is missing.",
            recommended_action=MANUAL_REVIEW,
            severity="error",
            evidence_fields={},
        ))
        return report

    row = global_rows[0]
    for field_name in ("mood", "global_vibe", "reflection_summary"):
        value = row.get(field_name)
        if isinstance(value, str):
            issue_code = "valid_global_character_state"
            severity = "keep"
            action_name = KEEP
        else:
            issue_code = "malformed_structure"
            severity = "manual_review"
            action_name = MANUAL_REVIEW
        report["findings"].append(_finding(
            row_key=f"global:{field_name}",
            issue_code=issue_code,
            issue_description=f"Character-state field {field_name} audited.",
            recommended_action=action_name,
            severity=severity,
            evidence_fields={"field_name": field_name},
        ))

    if not dry_run_repair:
        report["planned_actions"] = []
    return report


async def run_user_memory_units_audit(args: argparse.Namespace) -> None:
    """Run the user-memory-unit audit command."""

    report = await _build_user_memory_units_report()
    _write_report(args.output, report)
    print(f"wrote user-memory audit report to {args.output}")


async def run_user_memory_units_repair(args: argparse.Namespace) -> None:
    """Run the user-memory-unit repair command."""

    if args.apply:
        await _apply_report_actions(
            input_path=args.input,
            output_path=args.output,
            lane="user_memory_units",
        )
        return
    report = await _build_user_memory_units_report()
    _write_report(args.output, report)
    print(f"wrote user-memory dry-run report to {args.output}")


async def run_user_profiles_audit(args: argparse.Namespace) -> None:
    """Run the user-profile audit command."""

    report = await _build_user_profiles_report()
    _write_report(args.output, report)
    print(f"wrote user-profile audit report to {args.output}")


async def run_user_profiles_repair(args: argparse.Namespace) -> None:
    """Run the user-profile repair command."""

    if args.apply:
        await _apply_report_actions(
            input_path=args.input,
            output_path=args.output,
            lane="user_profiles",
        )
        return
    report = await _build_user_profiles_report()
    _write_report(args.output, report)
    print(f"wrote user-profile dry-run report to {args.output}")


async def run_shared_memory_repair(args: argparse.Namespace) -> None:
    """Run the shared-memory repair command."""

    if args.apply:
        allowed_actions = set(args.action_class or [])
        await _apply_report_actions(
            input_path=args.reviewed_report,
            output_path=args.output,
            lane="shared_memory",
            allowed_action_classes=allowed_actions,
        )
        return
    report = await _build_shared_memory_report()
    _write_report(args.output, report)
    print(f"wrote shared-memory dry-run report to {args.output}")


async def run_interaction_style_repair(args: argparse.Namespace) -> None:
    """Run the interaction-style repair command."""

    if args.apply:
        await _apply_report_actions(
            input_path=args.manifest,
            output_path=args.output,
            lane="interaction_style_images",
        )
        return
    report = await _build_interaction_style_report()
    _write_report(args.output, report)
    print(f"wrote interaction-style dry-run report to {args.output}")


async def run_conversation_episode_repair(args: argparse.Namespace) -> None:
    """Run the conversation-episode-state repair command."""

    if args.apply:
        await _apply_report_actions(
            input_path=args.input,
            output_path=args.output,
            lane="conversation_episode_state",
        )
        return
    report = await _build_conversation_episode_report()
    _write_report(args.output, report)
    print(f"wrote conversation-episode dry-run report to {args.output}")


async def run_character_state_audit(args: argparse.Namespace) -> None:
    """Run the character-state audit command."""

    if args.apply_repair:
        await _apply_report_actions(
            input_path=args.repair_file,
            output_path=args.output,
            lane="character_state",
        )
        return
    report = await _build_character_state_report(
        dry_run_repair=args.dry_run_repair,
    )
    _write_report(args.output, report)
    print(f"wrote character-state report to {args.output}")


def _run_async(handler: Any, args: argparse.Namespace) -> None:
    """Run one async cleanup handler with project DB configuration."""

    configure_stdout()
    configure_logging(args.verbose)
    load_project_env()

    async def _main() -> None:
        try:
            await handler(args)
        finally:
            await close_db()

    asyncio.run(_main())


def user_memory_units_audit_main() -> None:
    """CLI entrypoint for user-memory audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _run_async(run_user_memory_units_audit, args)


def user_memory_units_repair_main() -> None:
    """CLI entrypoint for user-memory repair."""

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.apply and args.input is None:
        parser.error("--apply requires --input")
    _run_async(run_user_memory_units_repair, args)


def user_profiles_audit_main() -> None:
    """CLI entrypoint for user-profile audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _run_async(run_user_profiles_audit, args)


def user_profiles_repair_main() -> None:
    """CLI entrypoint for user-profile repair."""

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.apply and args.input is None:
        parser.error("--apply requires --input")
    _run_async(run_user_profiles_repair, args)


def shared_memory_repair_main() -> None:
    """CLI entrypoint for shared-memory repair."""

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--reviewed-report", type=Path)
    parser.add_argument("--action-class", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.apply and args.reviewed_report is None:
        parser.error("--apply requires --reviewed-report")
    _run_async(run_shared_memory_repair, args)


def interaction_style_repair_main() -> None:
    """CLI entrypoint for interaction-style repair."""

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _run_async(run_interaction_style_repair, args)


def conversation_episode_repair_main() -> None:
    """CLI entrypoint for conversation-episode-state repair."""

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.apply and args.input is None:
        parser.error("--apply requires --input")
    _run_async(run_conversation_episode_repair, args)


def character_state_audit_main() -> None:
    """CLI entrypoint for character-state audit and repair."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-repair", action="store_true")
    parser.add_argument("--apply-repair", action="store_true")
    parser.add_argument("--repair-file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_artifacts/character_state_lane_repair_apply.json"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.apply_repair and args.repair_file is None:
        parser.error("--apply-repair requires --repair-file")
    _run_async(run_character_state_audit, args)
