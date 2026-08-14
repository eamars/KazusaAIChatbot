"""Delete the exact legacy scheduled-future-speech records."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.accepted_task.models import (
    ACCEPTED_TASK_SCHEMA_VERSION,
)
from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db._client import get_db


ACCEPTED_TASK_ID = "task-2c1831a6217342d7a5a24743d8eae669"
SCHEDULE_IDS = (
    "calendar_schedule_4944be41cc53d33b443640d10e2e7226",
    "calendar_schedule_49f5cad88af6d137fab09c108d603717",
    "calendar_schedule_59770930065b92900ffc676af106b457",
    "calendar_schedule_b812cbbd86f99e01505e319213ae0e5c",
)
SCHEDULE_SOURCE_REF_ID = "future_speak_background_work"
APPLY_CONFIRMATION = "DELETE_SCHEDULED_FUTURE_SPEECH_LEGACY_RECORDS"
LEGACY_CUTOVER_REASON = "scheduled_future_speech_legacy_cutover_2026-08-15"
LEGACY_TASK_STATE = "enqueue_failed"
LEGACY_SCHEDULE_STATUS = calendar_models.SCHEDULE_STATUS_CANCELLED
ACTIVE_RUN_STATUSES = {"pending", "running"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse read-only and exact-delete controls."""

    parser = argparse.ArgumentParser(
        description=(
            "Delete the exact legacy scheduled-future-speech records."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the exact validated task and schedule rows.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help="Required exact phrase for --apply.",
    )
    return parser.parse_args(argv)


async def _load_target_documents() -> dict[str, list[dict[str, Any]]]:
    """Load only the approved target rows and their bounded relationships."""

    db = await get_db()
    task = await db.accepted_tasks.find_one({
        "accepted_task_id": ACCEPTED_TASK_ID,
    })
    schedules_cursor = db.calendar_schedules.find({
        "schedule_id": {"$in": list(SCHEDULE_IDS)},
    })
    schedules = [
        dict(row)
        for row in await schedules_cursor.to_list(length=None)
    ]
    runs_cursor = db.calendar_runs.find({
        "schedule_id": {"$in": list(SCHEDULE_IDS)},
    })
    runs = [
        dict(row)
        for row in await runs_cursor.to_list(length=None)
    ]
    task_executor_ref = (
        str(task.get("executor_ref") or "")
        if isinstance(task, Mapping)
        else ""
    )
    job_filters: list[dict[str, Any]] = [
        {"accepted_task_id": ACCEPTED_TASK_ID},
        {"worker_metadata.accepted_task_id": ACCEPTED_TASK_ID},
    ]
    if task_executor_ref:
        job_filters.append({"job_id": task_executor_ref})
    jobs_cursor = db.background_work_jobs.find({"$or": job_filters})
    jobs = [
        dict(row)
        for row in await jobs_cursor.to_list(length=None)
    ]
    return {
        "accepted_tasks": [dict(task)] if isinstance(task, Mapping) else [],
        "calendar_schedules": schedules,
        "calendar_runs": runs,
        "background_work_jobs": jobs,
    }


def validate_cutover_documents(
    documents: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    """Return drift or safety errors for the exact deletion target set."""

    errors: list[str] = []
    tasks = list(documents.get("accepted_tasks", []))
    if len(tasks) != 1:
        errors.append(
            f"expected one accepted task, found {len(tasks)}"
        )
    else:
        task = tasks[0]
        if task.get("accepted_task_id") != ACCEPTED_TASK_ID:
            errors.append("accepted task identity drifted")
        if task.get("schema_version") != ACCEPTED_TASK_SCHEMA_VERSION:
            errors.append("accepted task schema is not v2")
        if task.get("task_kind") != "future_speak":
            errors.append("accepted task is not future_speak")
        if task.get("state") != LEGACY_TASK_STATE:
            errors.append("accepted task is not the exact legacy row state")
        if task.get("failure_summary") != LEGACY_CUTOVER_REASON:
            errors.append("accepted task legacy reason drifted")
        if "scheduled_future_speech_authority" in task:
            errors.append("accepted task unexpectedly carries authority")

    schedules = list(documents.get("calendar_schedules", []))
    schedules_by_id = {
        str(row.get("schedule_id")): row
        for row in schedules
    }
    if (
        len(schedules) != len(SCHEDULE_IDS)
        or set(schedules_by_id) != set(SCHEDULE_IDS)
    ):
        errors.append("calendar schedule target set drifted")
    for schedule_id in SCHEDULE_IDS:
        schedule = schedules_by_id.get(schedule_id)
        if schedule is None:
            continue
        if schedule.get("status") != LEGACY_SCHEDULE_STATUS:
            errors.append(f"schedule is not the exact legacy row: {schedule_id}")
        if schedule.get("cancel_reason") != LEGACY_CUTOVER_REASON:
            errors.append(f"schedule legacy reason drifted: {schedule_id}")
        if schedule.get("trigger_kind") != "future_cognition":
            errors.append(f"schedule trigger drifted: {schedule_id}")
        if not _has_source_ref(schedule, SCHEDULE_SOURCE_REF_ID):
            errors.append(f"schedule source marker missing: {schedule_id}")
        payload = schedule.get("payload")
        if isinstance(payload, Mapping) and (
            "scheduled_future_speech_authority" in payload
        ):
            errors.append(f"schedule unexpectedly carries authority: {schedule_id}")

    jobs = list(documents.get("background_work_jobs", []))
    if jobs:
        errors.append(
            "linked background work exists: "
            + ",".join(str(row.get("job_id")) for row in jobs)
        )
    active_runs = [
        row for row in documents.get("calendar_runs", [])
        if row.get("status") in ACTIVE_RUN_STATUSES
    ]
    if active_runs:
        errors.append(
            "linked calendar run is active: "
            + ",".join(str(row.get("run_id")) for row in active_runs)
        )
    return errors


def build_cutover_report(
    documents: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build a bounded operator report without semantic row contents."""

    tasks = list(documents.get("accepted_tasks", []))
    schedules = list(documents.get("calendar_schedules", []))
    runs = list(documents.get("calendar_runs", []))
    jobs = list(documents.get("background_work_jobs", []))
    errors = validate_cutover_documents(documents)
    return {
        "cutover": "scheduled_future_speech_legacy_records",
        "mode": "dry_run",
        "action": "hard_delete_exact_targets",
        "accepted_task_ids": [
            str(row.get("accepted_task_id")) for row in tasks
        ],
        "accepted_task_states": [str(row.get("state")) for row in tasks],
        "schedule_ids": [str(row.get("schedule_id")) for row in schedules],
        "schedule_statuses": [str(row.get("status")) for row in schedules],
        "calendar_run_ids": [str(row.get("run_id")) for row in runs],
        "calendar_run_statuses": [str(row.get("status")) for row in runs],
        "background_job_ids": [str(row.get("job_id")) for row in jobs],
        "validation_errors": errors,
        "ready": not errors,
    }


def build_delete_filters(
    documents: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build compare-and-delete filters from the validated target documents."""

    tasks = list(documents.get("accepted_tasks", []))
    if len(tasks) != 1:
        raise ValueError("cannot build task delete filter without one task")
    task = tasks[0]
    task_filter: dict[str, Any] = {
        "accepted_task_id": ACCEPTED_TASK_ID,
        "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
        "task_kind": "future_speak",
        "state": LEGACY_TASK_STATE,
        "failure_summary": LEGACY_CUTOVER_REASON,
        "scheduled_future_speech_authority": {"$exists": False},
    }
    if "executor_ref" in task:
        task_filter["executor_ref"] = task["executor_ref"]

    schedules_by_id = {
        str(row.get("schedule_id")): row
        for row in documents.get("calendar_schedules", [])
    }
    schedule_filters: dict[str, dict[str, Any]] = {}
    for schedule_id in SCHEDULE_IDS:
        schedule = schedules_by_id.get(schedule_id)
        if schedule is None:
            raise ValueError(
                f"cannot build schedule delete filter: {schedule_id}"
            )
        schedule_filter: dict[str, Any] = {
            "schedule_id": schedule_id,
            "status": LEGACY_SCHEDULE_STATUS,
            "cancel_reason": LEGACY_CUTOVER_REASON,
            "trigger_kind": "future_cognition",
            "idempotency_key": schedule["idempotency_key"],
            "payload.source_refs": {
                "$elemMatch": {"ref_id": SCHEDULE_SOURCE_REF_ID},
            },
            "payload.scheduled_future_speech_authority": {"$exists": False},
        }
        schedule_filters[schedule_id] = schedule_filter

    return {
        "accepted_tasks": task_filter,
        "calendar_schedules": schedule_filters,
    }


def _has_source_ref(row: Mapping[str, Any], ref_id: str) -> bool:
    """Return whether a schedule row carries one exact structural source ref."""

    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        return False
    source_refs = payload.get("source_refs")
    if not isinstance(source_refs, Sequence) or isinstance(source_refs, str):
        return False
    return any(
        isinstance(source_ref, Mapping)
        and source_ref.get("ref_id") == ref_id
        for source_ref in source_refs
    )


async def _apply_cutover() -> dict[str, Any]:
    """Delete the exact validated task and schedule rows."""

    before = await _load_target_documents()
    errors = validate_cutover_documents(before)
    if errors:
        raise RuntimeError("cutover precondition failed: " + "; ".join(errors))
    delete_filters = build_delete_filters(before)
    db = await get_db()

    deleted_schedule_ids: list[str] = []
    for schedule_id in sorted(SCHEDULE_IDS):
        result = await db.calendar_schedules.delete_one(
            delete_filters["calendar_schedules"][schedule_id],
        )
        if result.deleted_count != 1:
            raise RuntimeError(
                "schedule compare-and-delete matched no row: " + schedule_id
            )
        deleted_schedule_ids.append(schedule_id)

    task_result = await db.accepted_tasks.delete_one(
        delete_filters["accepted_tasks"],
    )
    if task_result.deleted_count != 1:
        raise RuntimeError(
            "accepted task compare-and-delete matched no row: "
            + ACCEPTED_TASK_ID
        )

    after = await _load_target_documents()
    remaining_task_ids = [
        str(row.get("accepted_task_id"))
        for row in after["accepted_tasks"]
    ]
    remaining_schedule_ids = [
        str(row.get("schedule_id"))
        for row in after["calendar_schedules"]
    ]
    remaining_target_ids = remaining_task_ids + remaining_schedule_ids
    if remaining_target_ids:
        raise RuntimeError(
            "cutover verification found undeleted targets: "
            + ",".join(remaining_target_ids)
        )
    return {
        "cutover": "scheduled_future_speech_legacy_records",
        "mode": "apply",
        "action": "hard_delete_exact_targets",
        "deleted_accepted_task_id": ACCEPTED_TASK_ID,
        "deleted_schedule_ids": deleted_schedule_ids,
        "preserved_calendar_run_ids": [
            str(row.get("run_id"))
            for row in before["calendar_runs"]
        ],
        "remaining_target_ids": [],
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Run read-only validation or the confirmed exact deletion."""

    if args.apply:
        if args.confirm != APPLY_CONFIRMATION:
            raise ValueError(
                "--apply requires the exact delete confirmation phrase"
            )
        return await _apply_cutover()
    documents = await _load_target_documents()
    return build_cutover_report(documents)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded cutover mode and print its operator report."""

    args = parse_args(argv)
    try:
        report = asyncio.run(_run(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Scheduled future-speech cutover failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print(
            "Scheduled future-speech cutover database operation failed.",
            file=sys.stderr,
        )
        return 1
    finally:
        asyncio.run(close_db())
    print(json.dumps(report, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
