"""Read-only preflight tests for the scheduled future-speech contract."""

from __future__ import annotations

from typing import Any

import pytest

import scripts.preflight_scheduled_future_speech_contract as preflight


class _FakeCursor:
    """Async iterable over one bounded fake collection scan."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = list(rows)

    def __aiter__(self) -> "_FakeCursor":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._rows:
            raise StopAsyncIteration
        return self._rows.pop(0)


class _FakeCollection:
    """Fake collection recording read-only scans and counts."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        count: int = 0,
    ) -> None:
        self.rows = list(rows)
        self.count = count
        self.scan_queries: list[dict[str, Any]] = []
        self.count_queries: list[dict[str, Any]] = []

    def find(
        self,
        query: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> _FakeCursor:
        del projection
        self.scan_queries.append(query)
        return _FakeCursor([
            row for row in self.rows if _matches_query(row, query)
        ])

    async def count_documents(self, query: dict[str, Any]) -> int:
        self.count_queries.append(query)
        return self.count


class _FakeDb:
    """Database facade with only collections used by the preflight."""

    def __init__(self, collections: dict[str, _FakeCollection]) -> None:
        self._collections = collections

    def __getitem__(self, name: str) -> _FakeCollection:
        return self._collections[name]


def _matches_query(row: dict[str, Any], query: dict[str, Any]) -> bool:
    """Apply the small Mongo query subset used by these preflight tests."""

    for field_name, expected in query.items():
        actual: object = row
        for path_part in field_name.split("."):
            if isinstance(actual, dict):
                actual = actual.get(path_part)
            elif isinstance(actual, list):
                actual = [
                    item.get(path_part)
                    for item in actual
                    if isinstance(item, dict)
                ]
            else:
                actual = None
        if isinstance(expected, dict) and "$in" in expected:
            accepted = expected["$in"]
            if not isinstance(accepted, list) or actual not in accepted:
                return False
        elif isinstance(actual, list):
            if expected not in actual:
                return False
        elif actual != expected:
            return False
    return True


def _legacy_active_rows() -> dict[str, list[dict[str, Any]]]:
    """Build active legacy future-speak rows without the new authority."""

    return {
        "accepted_tasks": [
            {
                "accepted_task_id": "task-legacy-001",
                "task_kind": "future_speak",
                "state": "pending",
            }
        ],
        "background_work_jobs": [
            {
                "job_id": "job-legacy-001",
                "requested_worker": "future_speak",
                "status": "queued",
            }
        ],
        "calendar_schedules": [
            {
                "schedule_id": "calendar_schedule_legacy_001",
                "trigger_kind": "future_cognition",
                "status": "active",
                "payload": {
                    "source_refs": [{
                        "ref_id": "future_speak_background_work",
                    }],
                },
            }
        ],
        "calendar_runs": [
            {
                "run_id": "calendar_run_legacy_001",
                "trigger_kind": "future_cognition",
                "status": "pending",
                "payload": {
                    "source_refs": [{
                        "ref_id": "future_speak_background_work",
                    }],
                },
            }
        ],
    }


def _compatible_active_rows() -> dict[str, list[dict[str, Any]]]:
    """Build active rows carrying a valid new authority."""

    from kazusa_ai_chatbot.cognition_shared.contracts import (
        build_scheduled_future_speech_authority,
    )

    authority = build_scheduled_future_speech_authority(
        source_episode_id="episode-1",
        source_message_id="msg-1",
        source_action_attempt_id="attempt-1",
        accepted_at_utc="2026-05-09T21:00:00+00:00",
        timezone="Pacific/Auckland",
        trigger_local="2026-05-10 13:00",
        platform="qq",
        channel_type="group",
        audience_kind="group",
        semantic_objective="在约定时间开始补偿考核。",
        authorized_content_summary="在约定时间开始补偿考核。",
        authorized_detail_refs=[
            {
                "evidence_handle": "e1",
                "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
                "provenance_role": "current_event",
            }
        ],
    )
    rows = _legacy_active_rows()
    for collection_name, collection_rows in rows.items():
        for row in collection_rows:
            if collection_name in {"calendar_schedules", "calendar_runs"}:
                row["payload"] = {
                    "source_refs": [{
                        "ref_id": "future_speak_background_work",
                    }],
                    "scheduled_future_speech_authority": dict(authority)
                }
            else:
                row["scheduled_future_speech_authority"] = dict(authority)
    return rows


def _fake_db(rows: dict[str, list[dict[str, Any]]]) -> _FakeDb:
    """Build a fake database with one collection per record kind."""

    collections = {
        name: _FakeCollection(collection_rows, count=0)
        for name, collection_rows in rows.items()
    }
    return _FakeDb(collections)


@pytest.mark.asyncio
async def test_preflight_blocks_active_legacy_future_speak_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active legacy records block cutover with bounded sample ids."""

    fake_db = _fake_db(_legacy_active_rows())

    async def get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(preflight, "get_db", get_db)

    report = await preflight.run_preflight(sample_limit=2)

    assert report["mode"] == "read_only"
    assert report["incompatible_active_count"] == 4
    assert report["deployment_blocked"] is True
    assert report["scans"]["accepted_tasks"]["incompatible_sample_ids"] == [
        "task-legacy-001"
    ]
    assert report["scans"]["background_work_jobs"][
        "incompatible_sample_ids"
    ] == ["job-legacy-001"]
    assert preflight.exit_code_from_report(report) == 1


@pytest.mark.asyncio
async def test_preflight_passes_when_active_records_carry_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cutover is unblocked when every active record carries the authority."""

    fake_db = _fake_db(_compatible_active_rows())

    async def get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(preflight, "get_db", get_db)

    report = await preflight.run_preflight(sample_limit=2)

    assert report["deployment_blocked"] is False
    assert report["incompatible_active_count"] == 0
    assert preflight.exit_code_from_report(report) == 0


@pytest.mark.asyncio
async def test_preflight_ignores_generic_future_cognition_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic scheduler rows are outside the future-speech cutover scan."""

    rows = _legacy_active_rows()
    rows["calendar_schedules"].append({
        "schedule_id": "calendar_schedule_generic_001",
        "trigger_kind": "future_cognition",
        "status": "active",
        "payload": {
            "source_refs": [{"ref_id": "cognitive_episode"}],
        },
    })
    rows["calendar_runs"].append({
        "run_id": "calendar_run_generic_001",
        "trigger_kind": "future_cognition",
        "status": "pending",
        "payload": {
            "source_refs": [{"ref_id": "cognitive_episode"}],
        },
    })
    fake_db = _fake_db(rows)

    async def get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(preflight, "get_db", get_db)

    report = await preflight.run_preflight(sample_limit=5)

    assert report["incompatible_active_count"] == 4
    assert report["scans"]["calendar_schedules"][
        "incompatible_sample_ids"
    ] == ["calendar_schedule_legacy_001"]
    assert report["scans"]["calendar_runs"][
        "incompatible_sample_ids"
    ] == ["calendar_run_legacy_001"]


@pytest.mark.asyncio
async def test_preflight_performs_no_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The preflight never mutates a collection."""

    fake_db = _fake_db(_legacy_active_rows())

    async def get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(preflight, "get_db", get_db)

    await preflight.run_preflight(sample_limit=1)

    for collection in fake_db._collections.values():
        assert not hasattr(collection, "delete_many")
        assert not hasattr(collection, "update_many")
        assert not hasattr(collection, "insert_one")


@pytest.mark.asyncio
async def test_preflight_reads_calendar_authority_from_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calendar schedule and run authority is read from the production payload."""

    from kazusa_ai_chatbot.cognition_shared.contracts import (
        build_scheduled_future_speech_authority,
    )

    authority = build_scheduled_future_speech_authority(
        source_episode_id="episode-1",
        source_message_id="msg-1",
        source_action_attempt_id="attempt-1",
        accepted_at_utc="2026-05-09T21:00:00+00:00",
        timezone="Pacific/Auckland",
        trigger_local="2026-05-10 13:00",
        platform="qq",
        channel_type="group",
        audience_kind="group",
        semantic_objective="在约定时间开始补偿考核。",
        authorized_content_summary="在约定时间开始补偿考核。",
        authorized_detail_refs=[
            {
                "evidence_handle": "e1",
                "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
                "provenance_role": "current_event",
            }
        ],
    )
    rows = _legacy_active_rows()
    for collection_name in ("calendar_schedules", "calendar_runs"):
        for row in rows[collection_name]:
            row["payload"] = {
                "source_refs": [{
                    "ref_id": "future_speak_background_work",
                }],
                "scheduled_future_speech_authority": dict(authority)
            }
    for collection_name in ("accepted_tasks", "background_work_jobs"):
        for row in rows[collection_name]:
            row["scheduled_future_speech_authority"] = dict(authority)
    # A calendar run carrying the authority only at the top level is not on
    # the production payload path and must remain incompatible.
    rows["calendar_runs"][0]["scheduled_future_speech_authority"] = dict(
        authority
    )
    del rows["calendar_runs"][0]["payload"][
        "scheduled_future_speech_authority"
    ]

    fake_db = _fake_db(rows)

    async def get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(preflight, "get_db", get_db)

    report = await preflight.run_preflight(sample_limit=2)

    assert report["deployment_blocked"] is True
    assert report["scans"]["calendar_schedules"][
        "incompatible_active_count"
    ] == 0
    assert report["scans"]["calendar_runs"][
        "incompatible_active_count"
    ] == 1
    assert report["scans"]["calendar_runs"]["incompatible_sample_ids"] == [
        "calendar_run_legacy_001"
    ]
    assert report["scans"]["accepted_tasks"][
        "incompatible_active_count"
    ] == 0
    assert report["scans"]["background_work_jobs"][
        "incompatible_active_count"
    ] == 0
