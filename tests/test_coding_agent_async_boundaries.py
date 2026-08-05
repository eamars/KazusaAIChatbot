"""Deterministic event-loop responsiveness regressions for coding-agent paths.

The async coding-agent runtime offloads blocking synchronous stages to worker
threads so the brain event loop keeps scheduling other work. These tests hold
one representative synchronous stage in a controlled test double and prove
that an independent event-loop heartbeat keeps running while the stage is
blocked, then that the offloaded result arrives unchanged.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.asyncio

BLOCK_WAIT_SECONDS = 2.0
HEARTBEAT_WINDOW_SECONDS = 0.5
HEARTBEAT_INTERVAL_SECONDS = 0.005
TASK_TIMEOUT_SECONDS = 5.0

SOURCE_IDENTITY = {
    "provider": "github",
    "owner": "fixture",
    "repo": "demo",
    "current_commit": "abc123",
    "dirty_state": "clean",
}
REPOSITORY = {
    **SOURCE_IDENTITY,
    "source_url": "https://github.com/fixture/demo",
    "requested_ref": None,
    "resolved_ref": "main",
    "default_branch": "main",
    "local_root": None,
    "storage_kind": "existing_local_checkout",
    "managed_checkout": False,
}
SOURCE_SCOPE = {
    "kind": "repository",
    "repo_relative_path": None,
    "source_url": "local://github/fixture/demo",
    "requested_ref": "main",
    "interpretation": "test checkout",
}
PATCH_ARTIFACT = {
    "artifact_id": "patch-app",
    "base": "repository",
    "diff_text": "diff --git a/app.py b/app.py\n",
    "files": ["app.py"],
    "summary": "Change app value.",
}
EXECUTION_SPEC = {
    "tool": "pytest",
    "paths": [],
    "pytest_selectors": ["tests/test_app.py"],
    "timeout_seconds": 10,
}
APPROVAL = {
    "approved": True,
    "approved_by": "test-operator",
    "approved_at": "2026-08-05T00:00:00Z",
    "approval_reason": "Deterministic heartbeat regression.",
}


def _assert_heartbeat_while_stage_held(timing: dict[str, object]) -> None:
    """Require real loop beats while the blocking stage was still held."""

    beats = timing["beats"]
    first_beat_at = timing["first_beat_at"]
    released_at = timing["released_at"]
    assert isinstance(beats, int) and beats > 0
    assert isinstance(first_beat_at, float)
    assert isinstance(released_at, float)
    assert first_beat_at < released_at


async def _heartbeat_while_blocked(
    entered: threading.Event,
    release: threading.Event,
    timing: dict[str, object],
) -> None:
    """Run an event-loop heartbeat while a test double holds a stage."""

    await asyncio.to_thread(entered.wait, BLOCK_WAIT_SECONDS)
    assert entered.is_set()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + HEARTBEAT_WINDOW_SECONDS
    beats = 0
    while not release.is_set() and loop.time() < deadline:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        beats += 1
        if "first_beat_at" not in timing:
            timing["first_beat_at"] = time.monotonic()
    release.set()
    timing["beats"] = beats


def _blocking_stage(
    *,
    entered: threading.Event,
    release: threading.Event,
    timing: dict[str, object],
    return_value: object,
) -> object:
    """Synchronous double that blocks until the heartbeat releases it."""

    entered.set()
    release.wait(BLOCK_WAIT_SECONDS)
    timing["released_at"] = time.monotonic()
    return return_value


async def test_answer_code_question_keeps_loop_while_reading_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Direct reading offloads the synchronous reading stage."""

    from kazusa_ai_chatbot.coding_agent import supervisor

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}

    async def fake_fetching(request: dict[str, object]) -> dict[str, object]:
        result = {
            "status": "succeeded",
            "message": "resolved",
            "repository": REPOSITORY,
            "source_scope": SOURCE_SCOPE,
            "limitations": [],
            "trace_summary": ["fetching:succeeded"],
        }
        return result

    def blocking_reading(request: dict[str, object]) -> dict[str, object]:
        reading_result = {
            "status": "succeeded",
            "answer_text": "Reading returned from the worker thread.",
            "evidence": [],
            "limitations": [],
            "trace_summary": ["reading:succeeded"],
        }
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=reading_result,
        )
        return result  # type: ignore[return-value]

    monkeypatch.setattr(supervisor.code_fetching, "run", fake_fetching)
    monkeypatch.setattr(supervisor.code_reading, "run", blocking_reading)

    task = asyncio.create_task(supervisor.answer_code_question({
        "question": "Explain the fixture.",
        "workspace_root": str(tmp_path / "workspace"),
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    response = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert response["status"] == "succeeded"
    assert response["answer_text"] == "Reading returned from the worker thread."


async def test_generated_readback_keeps_loop_while_reading_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generated-artifact readback offloads the synchronous reading stage."""

    from kazusa_ai_chatbot.coding_agent import supervisor

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}
    writing_calls: list[int] = []

    def fake_writing(request: dict[str, object]) -> dict[str, object]:
        writing_calls.append(1)
        if len(writing_calls) == 1:
            result = {
                "status": "need_reading",
                "reading_source": {
                    "repository": REPOSITORY,
                    "source_scope": SOURCE_SCOPE,
                },
                "pending_artifacts": [],
                "limitations": [],
                "trace_summary": ["writing:need_reading"],
            }
            return result
        result = {
            "status": "succeeded",
            "mode": "create_new_project",
            "answer_text": "Proposal ready.",
            "patch_artifacts": [],
            "created_files": [],
            "changed_files": [],
            "validation": {
                "status": "succeeded",
                "parsed": True,
                "sandbox_applied": True,
                "errors": [],
                "warnings": [],
                "files": [],
            },
            "external_evidence": [],
            "session": None,
            "limitations": [],
            "trace_summary": ["writing:succeeded"],
        }
        return result

    def blocking_reading(request: dict[str, object]) -> dict[str, object]:
        reading_result = {
            "status": "succeeded",
            "answer_text": "Readback returned.",
            "evidence": [{
                "path": "src/module.py",
                "line_start": 1,
                "line_end": 1,
                "symbol_or_topic": "module",
                "excerpt": "def public_api():",
                "reason": "Generated artifact readback.",
            }],
            "limitations": [],
            "trace_summary": ["reading:readback"],
        }
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=reading_result,
        )
        return result  # type: ignore[return-value]

    monkeypatch.setattr(supervisor.code_writing, "run", fake_writing)
    monkeypatch.setattr(supervisor.code_reading, "run", blocking_reading)

    task = asyncio.create_task(supervisor.propose_code_change({
        "question": "Create a public module.",
        "workspace_root": str(tmp_path / "workspace"),
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    response = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert len(writing_calls) == 2
    assert response["status"] == "succeeded"


async def test_existing_repo_initial_reading_keeps_loop_while_reading_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing-repository proposal preparation offloads initial reading."""

    from kazusa_ai_chatbot.coding_agent import supervisor

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    repository = {**REPOSITORY, "local_root": str(source_root)}

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}

    async def fake_fetching(request: dict[str, object]) -> dict[str, object]:
        result = {
            "status": "succeeded",
            "message": "resolved",
            "repository": repository,
            "source_scope": SOURCE_SCOPE,
            "limitations": [],
            "trace_summary": ["fetching:succeeded"],
        }
        return result

    def blocking_reading(request: dict[str, object]) -> dict[str, object]:
        reading_result = {
            "status": "succeeded",
            "answer_text": "Initial reading returned.",
            "evidence": [{
                "path": "app.py",
                "line_start": 1,
                "line_end": 1,
                "symbol_or_topic": "VALUE",
                "excerpt": "VALUE = 1",
                "reason": "Initial source reading.",
            }],
            "limitations": [],
            "trace_summary": ["reading:initial"],
        }
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=reading_result,
        )
        return result  # type: ignore[return-value]

    def fake_modifying(request: dict[str, object]) -> dict[str, object]:
        result = {
            "status": "succeeded",
            "answer_text": "Prepared modification.",
            "modification_artifacts": [{
                "status": "succeeded",
                "artifact_id": "replace-value",
                "operation_kind": "replace",
                "target_path": "app.py",
                "replacement_or_insert_content": "VALUE = 2\n",
                "operation_summary": "Replace value.",
                "evidence_ids": ["evidence-1"],
                "exact_anchor": "VALUE = 1\n",
            }],
            "created_files": [],
            "changed_files": [],
            "limitations": [],
            "trace_summary": ["modifying:succeeded"],
            "trace": None,
        }
        return result

    monkeypatch.setattr(supervisor.code_fetching, "run", fake_fetching)
    monkeypatch.setattr(supervisor.code_reading, "run", blocking_reading)
    monkeypatch.setattr(supervisor.code_modifying, "run", fake_modifying)

    task = asyncio.create_task(supervisor.propose_code_change({
        "question": "Change VALUE to 2.",
        "workspace_root": str(workspace_root),
        "local_root_hint": str(source_root),
        "max_artifact_chars": 8000,
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    response = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert response["status"] == "succeeded"
    assert response["patch_artifacts"]


async def test_code_fetching_local_resolution_keeps_loop_while_resolving_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fetching offloads synchronous local checkout resolution."""

    from kazusa_ai_chatbot.coding_agent.code_fetching import agent as fetching_agent

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}

    def blocking_resolve(local_root_hint: str) -> dict[str, object]:
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=REPOSITORY,
        )
        return result  # type: ignore[return-value]

    monkeypatch.setattr(
        fetching_agent.local_checkout,
        "resolve_existing_checkout",
        blocking_resolve,
    )

    task = asyncio.create_task(fetching_agent.run({
        "question": "Explain the local checkout.",
        "local_root_hint": str(tmp_path / "checkout"),
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    result = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert result["status"] == "succeeded"


async def test_code_fetching_managed_clone_keeps_loop_while_materializing_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fetching offloads synchronous managed clone materialization."""

    from kazusa_ai_chatbot.coding_agent.code_fetching import agent as fetching_agent

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}

    def blocking_clone(source: object, workspace_root: str) -> dict[str, object]:
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=REPOSITORY,
        )
        return result  # type: ignore[return-value]

    monkeypatch.setattr(
        fetching_agent.managed_clone,
        "ensure_managed_checkout",
        blocking_clone,
    )

    task = asyncio.create_task(fetching_agent.run({
        "question": "Explain the repository.",
        "repo_url": "https://github.com/fixture/demo.git",
        "workspace_root": str(tmp_path / "workspace"),
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    result = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert result["status"] == "succeeded"


async def test_code_fetching_managed_download_keeps_loop_while_materializing_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fetching offloads synchronous managed raw-file download materialization."""

    from kazusa_ai_chatbot.coding_agent.code_fetching import agent as fetching_agent

    source_root = tmp_path / "downloads"
    source_root.mkdir()
    (source_root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    repository = {**REPOSITORY, "local_root": str(source_root)}

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}

    def blocking_download(source: object, workspace_root: str) -> dict[str, object]:
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=repository,
        )
        return result  # type: ignore[return-value]

    monkeypatch.setattr(
        fetching_agent.managed_download,
        "ensure_managed_raw_file_download",
        blocking_download,
    )

    task = asyncio.create_task(fetching_agent.run({
        "question": "Explain the raw file.",
        "repo_url": (
            "https://raw.githubusercontent.com/fixture/demo/main/app.py"
        ),
        "workspace_root": str(tmp_path / "workspace"),
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    result = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert result["status"] == "succeeded"


async def test_code_fetching_managed_inline_keeps_loop_while_materializing_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fetching offloads synchronous managed inline bundle materialization."""

    from kazusa_ai_chatbot.coding_agent.code_fetching import agent as fetching_agent

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}

    def blocking_inline(source: object, workspace_root: str) -> tuple[dict, dict]:
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=(REPOSITORY, SOURCE_SCOPE),
        )
        return result  # type: ignore[return-value]

    monkeypatch.setattr(
        fetching_agent.managed_inline,
        "materialize_inline_source_bundle",
        blocking_inline,
    )

    task = asyncio.create_task(fetching_agent.run({
        "question": "Explain the pasted module.",
        "inline_sources": [{
            "content": "def public_api():\n    return 1\n",
            "filename_hint": "module.py",
        }],
        "workspace_root": str(tmp_path / "workspace"),
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    result = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert result["status"] == "succeeded"


async def test_verify_source_free_keeps_loop_while_candidate_materialization_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Source-free verification offloads managed candidate materialization."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import (
        supervisor as verifying_supervisor,
    )

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}

    def blocking_materialize(request: dict[str, object]) -> dict[str, object]:
        apply_response = {
            "status": "succeeded",
            "apply_package_id": "apply-1",
            "apply_workspace_ref": {
                "kind": "managed_apply_workspace",
                "package_id": "apply-1",
            },
            "changed_files": [],
            "limitations": [],
            "trace_summary": ["apply:succeeded"],
        }
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=apply_response,
        )
        return result  # type: ignore[return-value]

    def fake_execute(request: dict[str, object]) -> dict[str, object]:
        result = {
            "status": "succeeded",
            "tool": request["execution"]["tool"],
            "exit_code": 0,
            "timed_out": False,
            "duration_ms": 1,
            "stdout_excerpt": "",
            "stderr_excerpt": "",
            "output_truncated": False,
            "executed_paths": [],
            "limitations": [],
            "trace_summary": ["execution:succeeded"],
        }
        return result

    monkeypatch.setattr(
        verifying_supervisor,
        "materialize_managed_candidate",
        blocking_materialize,
    )
    monkeypatch.setattr(verifying_supervisor, "execute_code_check", fake_execute)

    task = asyncio.create_task(verifying_supervisor.verify_and_repair_code_change({
        "workspace_root": str(tmp_path / "workspace"),
        "approval": APPROVAL,
        "execution_specs": [EXECUTION_SPEC],
        "initial_patch_artifacts": [PATCH_ARTIFACT],
        "expected_source_identity": SOURCE_IDENTITY,
        "max_artifact_chars": 8000,
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    response = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert response["status"] == "succeeded"


async def test_verify_source_free_keeps_loop_while_execution_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Source-free verification offloads bounded execution checks."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import (
        supervisor as verifying_supervisor,
    )

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}

    def fake_materialize(request: dict[str, object]) -> dict[str, object]:
        result = {
            "status": "succeeded",
            "apply_package_id": "apply-1",
            "apply_workspace_ref": {
                "kind": "managed_apply_workspace",
                "package_id": "apply-1",
            },
            "changed_files": [],
            "limitations": [],
            "trace_summary": ["apply:succeeded"],
        }
        return result

    def blocking_execute(request: dict[str, object]) -> dict[str, object]:
        execution_result = {
            "status": "succeeded",
            "tool": request["execution"]["tool"],
            "exit_code": 0,
            "timed_out": False,
            "duration_ms": 1,
            "stdout_excerpt": "",
            "stderr_excerpt": "",
            "output_truncated": False,
            "executed_paths": [],
            "limitations": [],
            "trace_summary": ["execution:succeeded"],
        }
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=execution_result,
        )
        return result  # type: ignore[return-value]

    monkeypatch.setattr(
        verifying_supervisor,
        "materialize_managed_candidate",
        fake_materialize,
    )
    monkeypatch.setattr(verifying_supervisor, "execute_code_check", blocking_execute)

    task = asyncio.create_task(verifying_supervisor.verify_and_repair_code_change({
        "workspace_root": str(tmp_path / "workspace"),
        "approval": APPROVAL,
        "execution_specs": [EXECUTION_SPEC],
        "initial_patch_artifacts": [PATCH_ARTIFACT],
        "expected_source_identity": SOURCE_IDENTITY,
        "max_artifact_chars": 8000,
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    response = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert response["status"] == "succeeded"


async def test_coding_run_preflight_keeps_loop_while_candidate_materialization_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Durable proposal preflight offloads managed candidate materialization."""

    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor as run_supervisor

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    repository = {**REPOSITORY, "local_root": str(source_root)}

    entered = threading.Event()
    release = threading.Event()
    timing: dict[str, object] = {}

    async def fake_propose(request: dict[str, object]) -> dict[str, object]:
        result = {
            "status": "succeeded",
            "mode": "edit_existing_repository",
            "answer_text": "Prepared proposal.",
            "repository": repository,
            "source_scope": SOURCE_SCOPE,
            "evidence": [],
            "patch_artifacts": [PATCH_ARTIFACT],
            "created_files": [],
            "changed_files": [{
                "path": "app.py",
                "change_type": "modify",
                "summary": "Change value.",
            }],
            "validation": {
                "status": "succeeded",
                "parsed": True,
                "sandbox_applied": True,
                "errors": [],
                "warnings": [],
                "files": ["app.py"],
            },
            "external_evidence": [],
            "session": None,
            "limitations": [],
            "trace_summary": ["proposal:succeeded"],
        }
        return result

    def blocking_materialize(request: dict[str, object]) -> dict[str, object]:
        apply_response = {
            "status": "succeeded",
            "apply_package_id": "preflight-apply",
            "apply_workspace_ref": {
                "kind": "managed_apply_workspace",
                "package_id": "preflight-apply",
            },
            "changed_files": [],
            "limitations": [],
            "trace_summary": ["apply:preflight"],
        }
        result = _blocking_stage(
            entered=entered,
            release=release,
            timing=timing,
            return_value=apply_response,
        )
        return result  # type: ignore[return-value]

    def fake_execute(request: dict[str, object]) -> dict[str, object]:
        result = {
            "status": "succeeded",
            "tool": request["execution"]["tool"],
            "exit_code": 0,
            "timed_out": False,
            "duration_ms": 1,
            "stdout_excerpt": "",
            "stderr_excerpt": "",
            "output_truncated": False,
            "executed_paths": [],
            "limitations": [],
            "trace_summary": ["execution:preflight"],
        }
        return result

    monkeypatch.setattr(run_supervisor, "propose_code_change", fake_propose)
    monkeypatch.setattr(run_supervisor, "CODING_AGENT_PREFLIGHT_EXECUTION", True)
    monkeypatch.setattr(
        run_supervisor,
        "materialize_managed_candidate",
        blocking_materialize,
    )
    monkeypatch.setattr(run_supervisor, "execute_code_check", fake_execute)

    task = asyncio.create_task(start_coding_run({
        "question": "Change VALUE to 2.",
        "objective_type": "propose_patch",
        "workspace_root": str(tmp_path / "workspace"),
        "local_root_hint": str(source_root),
        "source_scope_hint": "repository",
        "max_artifact_chars": 8000,
    }))
    await _heartbeat_while_blocked(entered, release, timing)
    response = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECONDS)

    _assert_heartbeat_while_stage_held(timing)
    assert response["status"] == "awaiting_approval"
    assert response["preflight"]["status"] == "passed"
