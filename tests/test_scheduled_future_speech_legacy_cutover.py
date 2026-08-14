"""Deterministic tests for the bounded legacy-record cutover."""

from __future__ import annotations

from copy import deepcopy

from scripts import cutover_scheduled_future_speech_legacy_records as cutover


def _documents() -> dict[str, list[dict[str, object]]]:
    """Build the exact-shaped legacy target set."""

    schedules = [
        {
            "schedule_id": schedule_id,
            "status": cutover.LEGACY_SCHEDULE_STATUS,
            "cancel_reason": cutover.LEGACY_CUTOVER_REASON,
            "trigger_kind": "future_cognition",
            "idempotency_key": f"idempotency:{index}",
            "payload": {
                "source_refs": [{
                    "ref_id": cutover.SCHEDULE_SOURCE_REF_ID,
                }],
            },
        }
        for index, schedule_id in enumerate(cutover.SCHEDULE_IDS)
    ]
    return {
        "accepted_tasks": [{
            "accepted_task_id": cutover.ACCEPTED_TASK_ID,
            "schema_version": "accepted_task.v2",
            "task_kind": "future_speak",
            "state": cutover.LEGACY_TASK_STATE,
            "failure_summary": cutover.LEGACY_CUTOVER_REASON,
        }],
        "calendar_schedules": schedules,
        "calendar_runs": [{
            "run_id": "run-terminal-1",
            "schedule_id": cutover.SCHEDULE_IDS[0],
            "status": "completed",
        }],
        "background_work_jobs": [],
    }


def test_exact_legacy_target_set_is_ready_for_cutover() -> None:
    """The approved target shape passes the cutover safety checks."""

    documents = _documents()

    assert cutover.validate_cutover_documents(documents) == []
    report = cutover.build_cutover_report(documents)

    assert report["ready"] is True
    assert report["accepted_task_ids"] == [cutover.ACCEPTED_TASK_ID]
    assert report["schedule_ids"] == list(cutover.SCHEDULE_IDS)
    assert report["calendar_run_ids"] == ["run-terminal-1"]


def test_linked_job_blocks_cutover() -> None:
    """A linked job must prevent retirement of the orphaned task target."""

    documents = _documents()
    documents["background_work_jobs"].append({"job_id": "job-live"})

    errors = cutover.validate_cutover_documents(documents)

    assert errors == ["linked background work exists: job-live"]


def test_active_linked_run_blocks_cutover() -> None:
    """A pending or running linked run must remain outside this cutover."""

    documents = _documents()
    documents["calendar_runs"][0]["status"] = "pending"

    errors = cutover.validate_cutover_documents(documents)

    assert errors == ["linked calendar run is active: run-terminal-1"]


def test_report_does_not_mutate_loaded_documents() -> None:
    """Operator reporting remains read-only over the loaded snapshot."""

    documents = _documents()
    before = deepcopy(documents)

    cutover.build_cutover_report(documents)

    assert documents == before


def test_delete_filters_bind_exact_legacy_identity_and_provenance() -> None:
    """Deletion filters carry the exact state, reason, and source marker."""

    filters = cutover.build_delete_filters(_documents())

    task_filter = filters["accepted_tasks"]
    assert task_filter["accepted_task_id"] == cutover.ACCEPTED_TASK_ID
    assert task_filter["state"] == cutover.LEGACY_TASK_STATE
    assert task_filter["failure_summary"] == cutover.LEGACY_CUTOVER_REASON
    assert task_filter["scheduled_future_speech_authority"] == {
        "$exists": False,
    }

    schedule_filter = filters["calendar_schedules"][cutover.SCHEDULE_IDS[0]]
    assert schedule_filter["schedule_id"] == cutover.SCHEDULE_IDS[0]
    assert schedule_filter["status"] == cutover.LEGACY_SCHEDULE_STATUS
    assert schedule_filter["cancel_reason"] == cutover.LEGACY_CUTOVER_REASON
    assert schedule_filter["payload.source_refs"] == {
        "$elemMatch": {"ref_id": cutover.SCHEDULE_SOURCE_REF_ID},
    }
