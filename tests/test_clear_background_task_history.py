"""Tests for the exact two-collection task-history maintenance boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _module() -> ModuleType:
    """Load the maintenance script without executing its CLI entrypoint."""

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "clear_background_task_history.py"
    )
    spec = importlib.util.spec_from_file_location(
        "clear_background_task_history",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCollection:
    """Provide count and deletion behavior for one allowlisted collection."""

    def __init__(self, count: int, *, preserve_after_delete: bool = False) -> None:
        self.count = count
        self.preserve_after_delete = preserve_after_delete
        self.delete_filters: list[dict[str, object]] = []

    async def count_documents(self, query: dict[str, object]) -> int:
        assert query == {}
        return self.count

    async def delete_many(
        self,
        query: dict[str, object],
    ) -> SimpleNamespace:
        self.delete_filters.append(query)
        deleted_count = self.count
        if not self.preserve_after_delete:
            self.count = 0
        return SimpleNamespace(deleted_count=deleted_count)


@pytest.mark.asyncio
async def test_dry_run_counts_only_the_exact_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default execution counts both approved collections without deletion."""

    module = _module()
    assert module.TARGET_COLLECTIONS == (
        "background_work_jobs",
        "accepted_tasks",
    )
    collections = {
        "background_work_jobs": _FakeCollection(3),
        "accepted_tasks": _FakeCollection(2),
    }

    async def get_db() -> dict[str, _FakeCollection]:
        return collections

    monkeypatch.setattr(module, "get_db", get_db)
    report = await module.clear_background_task_history(execute=False)

    assert report == {
        "mode": "dry_run",
        "before": {"background_work_jobs": 3, "accepted_tasks": 2},
        "deleted": {"background_work_jobs": 0, "accepted_tasks": 0},
        "remaining": {"background_work_jobs": 3, "accepted_tasks": 2},
    }
    assert all(
        not collection.delete_filters
        for collection in collections.values()
    )


@pytest.mark.asyncio
async def test_execute_requires_the_exact_confirmation_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The destructive branch rejects an absent or approximate confirmation."""

    module = _module()

    async def unexpected_clear(*, execute: bool) -> dict[str, object]:
        raise AssertionError(f"clear called with execute={execute}")

    monkeypatch.setattr(module, "clear_background_task_history", unexpected_clear)
    args = module.parse_args(["--execute", "--confirm", "DELETE_BACKGROUND"])

    with pytest.raises(ValueError, match="exact"):
        await module._run(args)


@pytest.mark.asyncio
async def test_execute_deletes_both_allowlisted_collections_and_verifies_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution reports before/deleted/remaining counts and reaches zero."""

    module = _module()
    collections = {
        "background_work_jobs": _FakeCollection(4),
        "accepted_tasks": _FakeCollection(5),
    }

    async def get_db() -> dict[str, _FakeCollection]:
        return collections

    monkeypatch.setattr(module, "get_db", get_db)
    report = await module.clear_background_task_history(execute=True)

    assert report["before"] == {
        "background_work_jobs": 4,
        "accepted_tasks": 5,
    }
    assert report["deleted"] == {
        "background_work_jobs": 4,
        "accepted_tasks": 5,
    }
    assert report["remaining"] == {
        "background_work_jobs": 0,
        "accepted_tasks": 0,
    }
    assert all(
        collection.delete_filters == [{}]
        for collection in collections.values()
    )


@pytest.mark.asyncio
async def test_execute_fails_when_post_delete_verification_is_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collection that remains populated makes the command fail closed."""

    module = _module()
    collections = {
        "background_work_jobs": _FakeCollection(
            1,
            preserve_after_delete=True,
        ),
        "accepted_tasks": _FakeCollection(0),
    }

    async def get_db() -> dict[str, _FakeCollection]:
        return collections

    monkeypatch.setattr(module, "get_db", get_db)

    with pytest.raises(RuntimeError, match="verification"):
        await module.clear_background_task_history(execute=True)
