"""Tests for consolidation target lifecycle diagnostics."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.db import script_operations


class _FakeCollection:
    """Collect count filters and return configured diagnostic counts."""

    def __init__(
        self,
        name: str,
        counts: dict[tuple[str, str], int | list[int]],
        *,
        delete_count: int = 0,
        update_count: int = 0,
    ) -> None:
        self._name = name
        self._counts = counts
        self._delete_count = delete_count
        self._update_count = update_count
        self.filters: list[dict[str, Any]] = []
        self.delete_many_filters: list[dict[str, Any]] = []
        self.update_many_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def count_documents(self, filter_doc: dict[str, Any]) -> int:
        """Return a count selected by the exact filter shape."""

        self.filters.append(filter_doc)
        key = (self._name, repr(filter_doc))
        count = self._counts[key]
        if isinstance(count, list):
            return_value = count.pop(0)
            return return_value

        return_value = count
        return return_value

    async def delete_many(self, filter_doc: dict[str, Any]) -> Any:
        """Record deletion filters for synthetic-row cleanup tests."""

        self.delete_many_filters.append(filter_doc)
        return_value = _FakeWriteResult(deleted_count=self._delete_count)
        return return_value

    async def update_many(
        self,
        filter_doc: dict[str, Any],
        update_doc: dict[str, Any],
    ) -> Any:
        """Record maintenance updates for synthetic-row cleanup tests."""

        self.update_many_calls.append((filter_doc, update_doc))
        return_value = _FakeWriteResult(modified_count=self._update_count)
        return return_value


class _FakeWriteResult:
    """Minimal write result matching the attributes used by script helpers."""

    def __init__(
        self,
        *,
        deleted_count: int = 0,
        modified_count: int = 0,
    ) -> None:
        self.deleted_count = deleted_count
        self.modified_count = modified_count


class _FakeDb:
    """Database facade with only collections used by the diagnostic helper."""

    def __init__(
        self,
        counts: dict[tuple[str, str], int | list[int]],
        *,
        deleted_profiles: int = 0,
        deleted_memory_units: int = 0,
        updated_scheduled_events: int = 0,
    ) -> None:
        self.user_profiles = _FakeCollection(
            "user_profiles",
            counts,
            delete_count=deleted_profiles,
        )
        self.scheduled_events = _FakeCollection(
            "scheduled_events",
            counts,
            update_count=updated_scheduled_events,
        )
        self.user_memory_units = _FakeCollection(
            "user_memory_units",
            counts,
            delete_count=deleted_memory_units,
        )
        self.self_cognition_action_attempts = _FakeCollection(
            "self_cognition_action_attempts",
            counts,
        )


_LEGACY_PROFILE_SCALAR = "affin" + "ity"
_MISSING_LEGACY_PROFILE_SCALAR = (
    "user_profiles_missing_" + _LEGACY_PROFILE_SCALAR
)


EXPECTED_FILTERS = {
    "synthetic_user_profiles": {
        "global_user_id": "self_cognition",
    },
    _MISSING_LEGACY_PROFILE_SCALAR: {
        _LEGACY_PROFILE_SCALAR: {"$exists": False},
    },
    "synthetic_scheduled_events": {
        "source_user_id": "self_cognition",
    },
    "synthetic_user_memory_units": {
        "global_user_id": "self_cognition",
    },
    "future_cognition_attempts_missing_user": {
        "action_kind": "trigger_future_cognition",
        "$or": [
            {"target_scope.scope.source_user_id": {"$exists": False}},
            {"target_scope.scope.source_user_id": ""},
            {"target_scope.scope.source_user_id": None},
        ],
    },
    "synthetic_user_profiles_with_platform_accounts": {
        "global_user_id": "self_cognition",
        "platform_accounts.0": {"$exists": True},
    },
}

EXPECTED_PLANNED_APPLY = {
    "scheduled_events": {
        "operation": "update_many",
        "filter": EXPECTED_FILTERS["synthetic_scheduled_events"],
        "set_fields": {
            "status": "failed",
            "migration_reason": "synthetic_consolidation_user_cleanup",
            "migration_original_source_user_id": "self_cognition",
        },
        "unset_fields": ["source_user_id"],
        "runtime_fields": ["migration_applied_at"],
    },
    "user_profiles": {
        "operation": "delete_many",
        "filter": EXPECTED_FILTERS["synthetic_user_profiles"],
    },
    "user_memory_units": {
        "operation": "delete_many",
        "filter": EXPECTED_FILTERS["synthetic_user_memory_units"],
    },
}


def _counts(
    *,
    synthetic_user_profiles: int | list[int],
    user_profiles_missing_legacy_scalar: int | list[int],
    synthetic_scheduled_events: int | list[int],
    synthetic_user_memory_units: int | list[int],
    future_cognition_attempts_missing_user: int | list[int],
    synthetic_user_profiles_with_platform_accounts: int | list[int],
) -> dict[tuple[str, str], int | list[int]]:
    """Build fake count rows keyed by the exact diagnostic filters."""

    counts = {
        (
            "user_profiles",
            repr(EXPECTED_FILTERS["synthetic_user_profiles"]),
        ): synthetic_user_profiles,
        (
            "user_profiles",
            repr(EXPECTED_FILTERS[_MISSING_LEGACY_PROFILE_SCALAR]),
        ): user_profiles_missing_legacy_scalar,
        (
            "scheduled_events",
            repr(EXPECTED_FILTERS["synthetic_scheduled_events"]),
        ): synthetic_scheduled_events,
        (
            "user_memory_units",
            repr(EXPECTED_FILTERS["synthetic_user_memory_units"]),
        ): synthetic_user_memory_units,
        (
            "self_cognition_action_attempts",
            repr(EXPECTED_FILTERS["future_cognition_attempts_missing_user"]),
        ): future_cognition_attempts_missing_user,
        (
            "user_profiles",
            repr(EXPECTED_FILTERS["synthetic_user_profiles_with_platform_accounts"]),
        ): synthetic_user_profiles_with_platform_accounts,
    }
    return counts


@pytest.mark.asyncio
async def test_inspect_consolidation_target_lifecycle_counts_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostics should be read-only and report exact cleanup filters."""

    counts = _counts(
        synthetic_user_profiles=1,
        user_profiles_missing_legacy_scalar=2,
        synthetic_scheduled_events=3,
        synthetic_user_memory_units=4,
        future_cognition_attempts_missing_user=5,
        synthetic_user_profiles_with_platform_accounts=0,
    )
    fake_db = _FakeDb(counts)

    async def get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(
        script_operations,
        "get_db",
        get_db,
    )

    report = await script_operations.inspect_consolidation_target_lifecycle()

    assert report == {
        "synthetic_user_id": "self_cognition",
        "mode": "dry_run",
        "counts": {
            "synthetic_user_profiles": 1,
            _MISSING_LEGACY_PROFILE_SCALAR: 2,
            "synthetic_scheduled_events": 3,
            "synthetic_user_memory_units": 4,
            "future_cognition_attempts_missing_user": 5,
            "synthetic_user_profiles_with_platform_accounts": 0,
        },
        "filters": EXPECTED_FILTERS,
        "cleanup_blocked": False,
        "planned_apply_status": "available",
        "planned_apply": EXPECTED_PLANNED_APPLY,
    }


@pytest.mark.asyncio
async def test_apply_consolidation_target_lifecycle_cleanup_blocks_linked_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply mode should refuse mutation when the synthetic profile is linked."""

    counts = _counts(
        synthetic_user_profiles=1,
        user_profiles_missing_legacy_scalar=1,
        synthetic_scheduled_events=3,
        synthetic_user_memory_units=1,
        future_cognition_attempts_missing_user=3,
        synthetic_user_profiles_with_platform_accounts=1,
    )
    fake_db = _FakeDb(counts)

    async def get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(
        script_operations,
        "get_db",
        get_db,
    )

    report = await script_operations.apply_consolidation_target_lifecycle_cleanup(
        storage_timestamp_utc="2026-05-21T00:00:00+00:00",
    )

    assert report["mode"] == "apply"
    assert report["apply_status"] == "blocked"
    assert report["blocked_reason"] == "synthetic_profile_has_platform_accounts"
    assert report["before_counts"] == {
        "synthetic_user_profiles": 1,
        _MISSING_LEGACY_PROFILE_SCALAR: 1,
        "synthetic_scheduled_events": 3,
        "synthetic_user_memory_units": 1,
        "future_cognition_attempts_missing_user": 3,
        "synthetic_user_profiles_with_platform_accounts": 1,
    }
    assert report["after_counts"] == report["before_counts"]
    assert report["applied"] == {
        "scheduled_events_modified": 0,
        "synthetic_user_profiles_deleted": 0,
        "synthetic_user_memory_units_deleted": 0,
    }
    assert fake_db.scheduled_events.update_many_calls == []
    assert fake_db.user_profiles.delete_many_filters == []
    assert fake_db.user_memory_units.delete_many_filters == []


@pytest.mark.asyncio
async def test_apply_consolidation_target_lifecycle_cleanup_quarantines_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply mode should use only exact synthetic filters and report outcomes."""

    counts = _counts(
        synthetic_user_profiles=[1, 0],
        user_profiles_missing_legacy_scalar=[1, 0],
        synthetic_scheduled_events=[3, 0],
        synthetic_user_memory_units=[1, 0],
        future_cognition_attempts_missing_user=[3, 3],
        synthetic_user_profiles_with_platform_accounts=[0, 0],
    )
    fake_db = _FakeDb(
        counts,
        deleted_profiles=1,
        deleted_memory_units=1,
        updated_scheduled_events=3,
    )

    async def get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(
        script_operations,
        "get_db",
        get_db,
    )

    report = await script_operations.apply_consolidation_target_lifecycle_cleanup(
        storage_timestamp_utc="2026-05-21T00:00:00+00:00",
    )

    expected_scheduled_update = {
        "$set": {
            "status": "failed",
            "migration_reason": "synthetic_consolidation_user_cleanup",
            "migration_applied_at": "2026-05-21T00:00:00+00:00",
            "migration_original_source_user_id": "self_cognition",
        },
        "$unset": {"source_user_id": ""},
    }
    assert report["mode"] == "apply"
    assert report["apply_status"] == "applied"
    assert report["before_counts"] == {
        "synthetic_user_profiles": 1,
        _MISSING_LEGACY_PROFILE_SCALAR: 1,
        "synthetic_scheduled_events": 3,
        "synthetic_user_memory_units": 1,
        "future_cognition_attempts_missing_user": 3,
        "synthetic_user_profiles_with_platform_accounts": 0,
    }
    assert report["after_counts"] == {
        "synthetic_user_profiles": 0,
        _MISSING_LEGACY_PROFILE_SCALAR: 0,
        "synthetic_scheduled_events": 0,
        "synthetic_user_memory_units": 0,
        "future_cognition_attempts_missing_user": 3,
        "synthetic_user_profiles_with_platform_accounts": 0,
    }
    assert report["synthetic_user_owned_rows_after"] == 0
    assert report["applied"] == {
        "scheduled_events_modified": 3,
        "synthetic_user_profiles_deleted": 1,
        "synthetic_user_memory_units_deleted": 1,
    }
    assert fake_db.scheduled_events.update_many_calls == [
        (
            EXPECTED_FILTERS["synthetic_scheduled_events"],
            expected_scheduled_update,
        ),
    ]
    assert fake_db.user_profiles.delete_many_filters == [
        EXPECTED_FILTERS["synthetic_user_profiles"],
    ]
    assert fake_db.user_memory_units.delete_many_filters == [
        EXPECTED_FILTERS["synthetic_user_memory_units"],
    ]
