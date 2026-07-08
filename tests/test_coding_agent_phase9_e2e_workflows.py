"""Public deterministic E2E workflows for durable coding runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.asyncio

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


async def test_read_only_workflow_start_get_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A read-only workflow can be started and reloaded by run id."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    source_root = _source_tree(tmp_path)

    async def fake_answer(request: dict[str, object]) -> dict[str, object]:
        return _reading_response(answer_text="VALUE is defined in app.py.")

    monkeypatch.setattr(supervisor, "answer_code_question", fake_answer)

    started = await start_coding_run({
        "question": "Where is VALUE defined?",
        "objective_type": "read_only",
        "workspace_root": str(tmp_path / "workspace"),
        "local_root_hint": str(source_root),
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
    })

    assert started["status"] == "completed"
    assert reloaded["answer_text"] == "VALUE is defined in app.py."
    assert reloaded["evidence"][0]["path"] == "app.py"


async def test_proposal_workflow_waits_for_approval_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A proposal workflow pauses with patch artifacts before verification."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    source_root = _source_tree(tmp_path)
    before_hash = _hash_tree(source_root)
    verify_calls: list[dict[str, object]] = []

    async def fake_verify(request: dict[str, object]) -> dict[str, object]:
        verify_calls.append(request)
        return _verify_response(status="succeeded")

    monkeypatch.setattr(supervisor, "propose_code_change", _fake_propose)
    monkeypatch.setattr(supervisor, "verify_and_repair_code_change", fake_verify)

    started = await start_coding_run(_proposal_request(tmp_path, source_root))
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
    })

    assert started["status"] == "awaiting_approval"
    assert reloaded["patch_artifacts"] == [PATCH_ARTIFACT]
    assert verify_calls == []
    assert _hash_tree(source_root) == before_hash


async def test_approval_workflow_verifies_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An approved workflow reaches terminal verification through run APIs."""

    from kazusa_ai_chatbot.coding_agent import continue_coding_run
    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    source_root = _source_tree(tmp_path)
    before_hash = _hash_tree(source_root)
    verify_calls: list[dict[str, object]] = []

    async def fake_verify(request: dict[str, object]) -> dict[str, object]:
        verify_calls.append(request)
        return _verify_response(status="succeeded", attempt_count=2)

    monkeypatch.setattr(supervisor, "propose_code_change", _fake_propose)
    monkeypatch.setattr(supervisor, "verify_and_repair_code_change", fake_verify)

    started = await start_coding_run(_proposal_request(tmp_path, source_root))
    completed = await continue_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
        "action": "approve_and_verify",
        "approval": _approval(),
        "execution_specs": [EXECUTION_SPEC],
        "repair_attempt_limit": 1,
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
    })

    assert completed["status"] == "completed"
    assert verify_calls[0]["initial_patch_artifacts"] == [PATCH_ARTIFACT]
    assert len(reloaded["attempts"]) == 2
    assert len(reloaded["repair_attempts"]) == 1
    assert _hash_tree(source_root) == before_hash


async def test_cancel_workflow_persists_without_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation is visible through the public get API."""

    from kazusa_ai_chatbot.coding_agent import continue_coding_run
    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    source_root = _source_tree(tmp_path)
    before_hash = _hash_tree(source_root)

    monkeypatch.setattr(supervisor, "propose_code_change", _fake_propose)

    started = await start_coding_run(_proposal_request(tmp_path, source_root))
    cancelled = await continue_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
        "action": "cancel",
        "reason": "Manual cancellation.",
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
    })

    assert cancelled["status"] == "cancelled"
    assert reloaded["status"] == "cancelled"
    assert reloaded["execution_attempts"] == []
    assert _hash_tree(source_root) == before_hash


async def test_seeded_verify_repair_workflow_records_attempt_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Seeded verification records apply, execution, and repair attempts."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    source_root = _source_tree(tmp_path)
    before_hash = _hash_tree(source_root)

    async def fake_verify(request: dict[str, object]) -> dict[str, object]:
        return _verify_response(status="succeeded", attempt_count=2)

    monkeypatch.setattr(supervisor, "verify_and_repair_code_change", fake_verify)

    started = await start_coding_run({
        **_proposal_request(tmp_path, source_root),
        "objective_type": "verify_repair",
        "approval": _approval(),
        "execution_specs": [EXECUTION_SPEC],
        "repair_attempt_limit": 1,
        "initial_patch_artifacts": [PATCH_ARTIFACT],
        "expected_source_identity": SOURCE_IDENTITY,
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
    })
    serialized = json.dumps(reloaded, ensure_ascii=False)

    assert started["status"] == "completed"
    assert [attempt["attempt_index"] for attempt in reloaded["attempts"]] == [1, 2]
    assert reloaded["apply_attempts"]
    assert reloaded["execution_attempts"]
    assert str(source_root.resolve()) not in serialized
    assert str((tmp_path / "workspace").resolve()) not in serialized
    assert _hash_tree(source_root) == before_hash


async def _fake_propose(request: dict[str, object]) -> dict[str, object]:
    return _proposal_response()


def _source_tree(tmp_path: Path) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    (source_root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    tests_root = source_root / "tests"
    tests_root.mkdir(exist_ok=True)
    (tests_root / "test_app.py").write_text(
        "from app import VALUE\n\n"
        "def test_value():\n"
        "    assert VALUE == 2\n",
        encoding="utf-8",
    )
    return source_root


def _proposal_request(tmp_path: Path, source_root: Path) -> dict[str, object]:
    request = {
        "question": "Change VALUE to 2.",
        "objective_type": "propose_patch",
        "workspace_root": str(tmp_path / "workspace"),
        "local_root_hint": str(source_root),
        "source_scope_hint": "repository",
        "preferred_language": "English",
        "max_answer_chars": 2000,
        "max_artifact_chars": 8000,
        "session_id": "run-e2e",
    }
    return request


def _reading_response(*, answer_text: str) -> dict[str, object]:
    response = {
        "status": "succeeded",
        "answer_text": answer_text,
        "repository": REPOSITORY,
        "source_scope": SOURCE_SCOPE,
        "evidence": [{
            "path": "app.py",
            "line_start": 1,
            "line_end": 1,
            "symbol_or_topic": "VALUE",
            "excerpt": "VALUE = 1",
            "reason": "Current value.",
        }],
        "limitations": [],
        "trace_summary": ["reading:succeeded"],
    }
    return response


def _proposal_response() -> dict[str, object]:
    response = {
        "status": "succeeded",
        "mode": "edit_existing_repository",
        "answer_text": "Prepared proposal.",
        "repository": REPOSITORY,
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
    return response


def _verify_response(*, status: str, attempt_count: int = 1) -> dict[str, object]:
    attempts = [
        {
            "attempt_index": index,
            "proposal_status": "succeeded",
            "apply_status": "succeeded",
            "execution_statuses": ["failed" if index == 1 and attempt_count > 1 else status],
            "patch_artifact_count": 1,
            "changed_files": [{
                "path": "app.py",
                "change_type": "modify",
                "summary": "Verified change.",
            }],
            "apply_package_id": f"apply-{index}",
            "limitations": [],
            "trace_summary": [f"attempt:{index}"],
        }
        for index in range(1, attempt_count + 1)
    ]
    response = {
        "status": status,
        "answer_text": "Verification succeeded.",
        "repository": REPOSITORY,
        "source_scope": SOURCE_SCOPE,
        "attempts": attempts,
        "final_patch_artifacts": [PATCH_ARTIFACT],
        "final_changed_files": [{
            "path": "app.py",
            "change_type": "modify",
            "summary": "Verified change.",
        }],
        "final_apply": {
            "status": "succeeded",
            "apply_package_id": f"apply-{attempt_count}",
            "source_identity": SOURCE_IDENTITY,
            "apply_workspace_ref": {
                "kind": "managed_apply_workspace",
                "apply_package_id": f"apply-{attempt_count}",
                "source_identity": SOURCE_IDENTITY,
                "applied_files": ["app.py"],
            },
            "applied_files": ["app.py"],
            "changed_files": [{
                "path": "app.py",
                "change_type": "modify",
                "summary": "Verified change.",
            }],
            "validation": {"status": "succeeded", "errors": [], "warnings": []},
            "limitations": [],
            "trace_summary": ["patch_apply:succeeded"],
        },
        "final_execution": [{
            "status": status,
            "tool": "pytest",
            "exit_code": 0 if status == "succeeded" else 1,
            "timed_out": False,
            "duration_ms": 10,
            "stdout_excerpt": "pytest output",
            "stderr_excerpt": "",
            "output_truncated": False,
            "executed_paths": ["tests/test_app.py"],
            "limitations": [],
            "trace_summary": [f"execution:{status}"],
        }],
        "limitations": [],
        "trace_summary": ["verify:succeeded"],
    }
    return response


def _approval() -> dict[str, object]:
    approval = {
        "approved": True,
        "approved_by": "run-e2e",
        "approved_at": "2026-07-09T00:00:00Z",
        "approval_reason": "Deterministic public workflow.",
    }
    return approval


def _hash_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative_path] = digest
    return hashes
