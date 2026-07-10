"""Deterministic contracts for coding-run workspace locking."""

import asyncio
import multiprocessing
from pathlib import Path

import pytest

from kazusa_ai_chatbot.coding_agent.coding_run.locking import (
    acquire_workspace_locks,
    build_lock_keys,
)


def _hold_workspace_locks(
    workspace_root_text: str,
    keys: list[str],
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    """Hold one process-owned lock set until the parent releases the child."""

    async def hold() -> None:
        async with acquire_workspace_locks(
            workspace_root=Path(workspace_root_text),
            keys=keys,
            timeout_seconds=2.0,
        ) as acquired:
            if not acquired:
                return
            ready.set()
            await asyncio.to_thread(release.wait, 10.0)

    asyncio.run(hold())


def _raise_after_acquiring_workspace_locks(
    workspace_root_text: str,
    keys: list[str],
    ready: multiprocessing.synchronize.Event,
) -> None:
    """Exercise exception cleanup in a child process that owns kernel locks."""

    async def raise_after_acquiring() -> None:
        try:
            async with acquire_workspace_locks(
                workspace_root=Path(workspace_root_text),
                keys=keys,
                timeout_seconds=2.0,
            ) as acquired:
                if not acquired:
                    return
                ready.set()
                raise RuntimeError("release locks through context cleanup")
        except RuntimeError:
            return

    asyncio.run(raise_after_acquiring())


def test_lock_keys_are_sorted_and_source_free_runs_use_only_run_key(
    tmp_path: Path,
) -> None:
    """Lock identity is stable before a mutation enters the workspace."""

    source_free = build_lock_keys(
        run_id="a" * 32,
        source_identity=None,
    )
    source_backed = build_lock_keys(
        run_id="b" * 32,
        source_identity={
            "provider": "github",
            "owner": "fixture",
            "repo": "demo",
            "requested_ref": "main",
            "workspace_root": str(tmp_path),
        },
    )

    assert source_free == [f"run:{'a' * 32}"]
    assert source_backed == sorted(source_backed)
    assert source_backed[0].startswith("run:") or source_backed[0].startswith(
        "source:"
    )
    assert len(source_backed) == 2


@pytest.mark.asyncio
async def test_same_source_locks_contend_across_processes(tmp_path: Path) -> None:
    """Different runs sharing a source identity serialize at the source lock."""

    identity = {
        "provider": "github",
        "owner": "fixture",
        "repo": "demo",
        "requested_ref": "main",
    }
    first_keys = build_lock_keys(run_id="a" * 32, source_identity=identity)
    second_keys = build_lock_keys(run_id="b" * 32, source_identity=identity)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_workspace_locks,
        args=(str(tmp_path), first_keys, ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=5.0)
        async with acquire_workspace_locks(
            workspace_root=tmp_path,
            keys=second_keys,
            timeout_seconds=0.1,
        ) as acquired:
            assert acquired is False
    finally:
        release.set()
        process.join(timeout=5.0)
    assert process.exitcode == 0


@pytest.mark.asyncio
async def test_workspace_locks_release_after_exception(tmp_path: Path) -> None:
    """Exception exit releases every process-owned kernel lock."""

    keys = build_lock_keys(run_id="c" * 32, source_identity=None)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_raise_after_acquiring_workspace_locks,
        args=(str(tmp_path), keys, ready),
    )
    process.start()
    assert ready.wait(timeout=5.0)
    process.join(timeout=5.0)
    assert process.exitcode == 0
    async with acquire_workspace_locks(
        workspace_root=tmp_path,
        keys=keys,
        timeout_seconds=0.1,
    ) as acquired:
        assert acquired is True


@pytest.mark.asyncio
async def test_start_busy_does_not_create_a_run_before_source_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A contended source lock prevents start from writing a new run ledger."""

    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    workspace_root = tmp_path / "workspace"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_identity = {"local_root_hint": str(source_root)}
    source_keys = build_lock_keys(
        run_id="d" * 32,
        source_identity=source_identity,
    )
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_workspace_locks,
        args=(str(workspace_root), source_keys, ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=5.0)
        monkeypatch.setattr(supervisor, "CODING_RUN_LOCK_TIMEOUT_SECONDS", 0.1)
        response = await supervisor.start_coding_run({
            "question": "Explain the source.",
            "objective_type": "read_only",
            "workspace_root": str(workspace_root),
            "local_root_hint": str(source_root),
        })
    finally:
        release.set()
        process.join(timeout=5.0)

    run_root = workspace_root / "coding_runs"
    assert response["operation_outcome"] == "busy"
    assert response["run_id"]
    assert run_root.exists()
    assert list(run_root.iterdir()) == []
    assert process.exitcode == 0


@pytest.mark.asyncio
async def test_continuation_busy_preserves_ledger_and_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Contended continuation returns retryable busy without mutating its run."""

    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    async def fake_answer(_: dict[str, object]) -> dict[str, object]:
        return {
            "status": "succeeded",
            "answer_text": "Completed read-only answer.",
            "repository": None,
            "source_scope": None,
            "evidence": [],
            "limitations": [],
            "trace_summary": ["reading:succeeded"],
        }

    monkeypatch.setattr(supervisor, "answer_code_question", fake_answer)
    workspace_root = tmp_path / "workspace"
    started = await supervisor.start_coding_run({
        "question": "Explain the fixture.",
        "objective_type": "read_only",
        "workspace_root": str(workspace_root),
    })
    before = await supervisor.get_coding_run({
        "workspace_root": str(workspace_root),
        "run_id": started["run_id"],
    })
    keys = build_lock_keys(run_id=started["run_id"], source_identity=None)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_workspace_locks,
        args=(str(workspace_root), keys, ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=5.0)
        monkeypatch.setattr(supervisor, "CODING_RUN_LOCK_TIMEOUT_SECONDS", 0.1)
        response = await supervisor.continue_coding_run({
            "workspace_root": str(workspace_root),
            "run_id": started["run_id"],
            "action": "summarize",
        })
    finally:
        release.set()
        process.join(timeout=5.0)

    after = await supervisor.get_coding_run({
        "workspace_root": str(workspace_root),
        "run_id": started["run_id"],
    })
    assert response["operation_outcome"] == "busy"
    assert response["status"] == "completed"
    assert response["allowed_next_actions"] == ["summarize", "status"]
    assert after["events"] == before["events"]
    assert process.exitcode == 0
