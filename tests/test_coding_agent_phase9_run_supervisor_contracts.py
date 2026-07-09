"""Deterministic contracts for durable coding-run supervision."""

from __future__ import annotations

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


async def test_start_rejects_missing_workspace_without_ledger() -> None:
    """Run creation requires explicit workspace storage."""

    from kazusa_ai_chatbot.coding_agent import start_coding_run

    response = await start_coding_run({
        "question": "Explain the app.",
        "objective_type": "read_only",
    })

    assert response["status"] == "rejected"
    assert response["run_id"] == ""
    assert any("workspace" in item.casefold() for item in response["limitations"])


async def test_start_rejects_missing_and_illegal_objective(
    tmp_path: Path,
) -> None:
    """Run routing is a closed deterministic caller choice."""

    from kazusa_ai_chatbot.coding_agent import start_coding_run

    missing_response = await start_coding_run({
        "question": "Explain the app.",
        "workspace_root": str(tmp_path / "workspace"),
    })
    illegal_response = await start_coding_run({
        "question": "Explain the app.",
        "objective_type": "deploy",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert missing_response["status"] == "rejected"
    assert illegal_response["status"] == "rejected"
    assert not (tmp_path / "workspace" / "coding_runs").exists()


async def test_get_missing_run_is_rejected(tmp_path: Path) -> None:
    """Inspecting a missing run returns a public rejection."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run

    response = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": "missing",
    })

    assert response["status"] == "rejected"
    assert response["run_id"] == "missing"
    assert any("missing" in item.casefold() for item in response["limitations"])


async def test_status_continuation_projects_run_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Direct status continuation returns the public projection only."""

    from kazusa_ai_chatbot.coding_agent import continue_coding_run
    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    monkeypatch.setattr(supervisor, "propose_code_change", _fake_propose)

    started = await start_coding_run(_proposal_request(tmp_path))
    status_response = await continue_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
        "action": "status",
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
    })

    assert status_response["status"] == "awaiting_approval"
    assert status_response["run_id"] == started["run_id"]
    assert status_response["events"] == reloaded["events"]
    assert status_response["allowed_next_actions"] == [
        "revise_proposal",
        "summarize",
        "status",
        "approve_and_verify",
        "cancel",
    ]


async def test_read_only_start_persists_completed_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read-only runs persist a completed public projection."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    workspace_root = tmp_path / "workspace"
    source_root = tmp_path / "source"
    source_root.mkdir()
    calls: list[dict[str, object]] = []

    async def fake_answer(request: dict[str, object]) -> dict[str, object]:
        calls.append(request)
        return _reading_response(
            answer_text=(
                f"Looked at {source_root.resolve()} and skipped .env "
                "plus .git internals."
            ),
        )

    monkeypatch.setattr(supervisor, "answer_code_question", fake_answer)

    response = await start_coding_run({
        "question": "Explain the value.",
        "objective_type": "read_only",
        "workspace_root": str(workspace_root),
        "local_root_hint": str(source_root),
    })
    reloaded = await get_coding_run({
        "workspace_root": str(workspace_root),
        "run_id": response["run_id"],
    })
    serialized = json.dumps(reloaded, ensure_ascii=False)
    event_types = [event["event_type"] for event in reloaded["events"]]

    assert response["status"] == "completed"
    assert reloaded["status"] == "completed"
    assert response["answer_text"]
    assert calls[0]["question"] == "Explain the value."
    assert (workspace_root / "coding_runs" / response["run_id"] / "run.json").exists()
    assert event_types == [
        "run_created",
        "source_resolved",
        "evidence_collected",
        "completed",
    ]
    assert str(workspace_root.resolve()) not in serialized
    assert str(source_root.resolve()) not in serialized
    assert ".env" not in serialized
    assert ".git" not in serialized


async def test_proposal_start_waits_for_approval_without_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Proposal runs stop before side-effecting verification."""

    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    verify_calls: list[dict[str, object]] = []

    async def fake_propose(request: dict[str, object]) -> dict[str, object]:
        return _proposal_response()

    async def fake_verify(request: dict[str, object]) -> dict[str, object]:
        verify_calls.append(request)
        return _verify_response(status="succeeded")

    monkeypatch.setattr(supervisor, "propose_code_change", fake_propose)
    monkeypatch.setattr(supervisor, "verify_and_repair_code_change", fake_verify)

    response = await start_coding_run(_proposal_request(tmp_path))
    event_types = [event["event_type"] for event in response["events"]]

    assert response["status"] == "awaiting_approval"
    assert response["patch_artifacts"] == [PATCH_ARTIFACT]
    assert verify_calls == []
    assert "proposal_ready" in event_types
    assert "awaiting_approval" in event_types


async def test_cancel_non_terminal_run_persists_cancelled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation is a deterministic terminal continuation."""

    from kazusa_ai_chatbot.coding_agent import continue_coding_run
    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    monkeypatch.setattr(supervisor, "propose_code_change", _fake_propose)

    started = await start_coding_run(_proposal_request(tmp_path))
    cancelled = await continue_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
        "action": "cancel",
        "reason": "Caller chose not to apply.",
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
    })

    assert started["status"] == "awaiting_approval"
    assert cancelled["status"] == "cancelled"
    assert reloaded["status"] == "cancelled"
    assert cancelled["patch_artifacts"] == [PATCH_ARTIFACT]
    assert cancelled["execution_attempts"] == []


async def test_approve_requires_structured_approval_without_verifier_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Approval continuation rejects missing approval before side effects."""

    from kazusa_ai_chatbot.coding_agent import continue_coding_run
    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    verify_calls: list[dict[str, object]] = []

    async def fake_verify(request: dict[str, object]) -> dict[str, object]:
        verify_calls.append(request)
        return _verify_response(status="succeeded")

    monkeypatch.setattr(supervisor, "propose_code_change", _fake_propose)
    monkeypatch.setattr(supervisor, "verify_and_repair_code_change", fake_verify)

    started = await start_coding_run(_proposal_request(tmp_path))
    rejected = await continue_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
        "action": "approve_and_verify",
        "execution_specs": [EXECUTION_SPEC],
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
    })

    assert rejected["status"] == "rejected"
    assert reloaded["status"] == "awaiting_approval"
    assert verify_calls == []
    assert not (tmp_path / "workspace" / "patch_apply").exists()


async def test_approve_and_verify_uses_stored_patch_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Approval continuation composes the bounded verifier."""

    from kazusa_ai_chatbot.coding_agent import continue_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    verify_calls: list[dict[str, object]] = []

    async def fake_verify(request: dict[str, object]) -> dict[str, object]:
        verify_calls.append(request)
        return _verify_response(status="succeeded")

    monkeypatch.setattr(supervisor, "propose_code_change", _fake_propose)
    monkeypatch.setattr(supervisor, "verify_and_repair_code_change", fake_verify)

    started = await start_coding_run(_proposal_request(tmp_path))
    completed = await continue_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
        "action": "approve_and_verify",
        "approval": _approval(),
        "execution_specs": [EXECUTION_SPEC],
        "repair_attempt_limit": 1,
    })

    assert completed["status"] == "completed"
    assert completed["attempts"][0]["attempt_index"] == 1
    assert verify_calls[0]["initial_patch_artifacts"] == [PATCH_ARTIFACT]
    assert verify_calls[0]["expected_source_identity"] == SOURCE_IDENTITY
    assert verify_calls[0]["execution_specs"] == [EXECUTION_SPEC]


async def test_verify_repair_start_records_terminal_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Seeded verify-and-repair runs persist verifier attempts."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    async def fake_verify(request: dict[str, object]) -> dict[str, object]:
        return _verify_response(status="succeeded", attempt_count=2)

    monkeypatch.setattr(supervisor, "verify_and_repair_code_change", fake_verify)

    response = await start_coding_run({
        **_proposal_request(tmp_path),
        "objective_type": "verify_repair",
        "approval": _approval(),
        "execution_specs": [EXECUTION_SPEC],
        "initial_patch_artifacts": [PATCH_ARTIFACT],
        "expected_source_identity": SOURCE_IDENTITY,
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": response["run_id"],
    })

    assert response["status"] == "completed"
    assert len(response["attempts"]) == 2
    assert len(reloaded["repair_attempts"]) == 1
    assert reloaded["changed_files"] == [{
        "path": "app.py",
        "change_type": "modify",
        "summary": "Verified change.",
    }]


async def test_proposal_run_projects_created_files_and_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Durable proposal ledgers should preserve creation and alignment evidence."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    created_files = [{
        "path": "counter_cli/formatters.py",
        "role": "Create formatter helper.",
    }]
    alignment = {
        "status": "pass",
        "request_satisfied": True,
        "matched_criteria": ["criterion-header"],
        "missing_criteria": [],
        "blockers": [],
    }

    async def fake_propose(request: dict[str, object]) -> dict[str, object]:
        response = _proposal_response()
        response["created_files"] = created_files
        response["alignment"] = alignment
        response["trace_summary"] = [
            *response["trace_summary"],
            "writing_alignment:status=pass",
        ]
        return response

    monkeypatch.setattr(supervisor, "propose_code_change", fake_propose)

    response = await start_coding_run({
        **_proposal_request(tmp_path),
        "question": "Create formatter helper and wire the CLI.",
        "objective_type": "propose_patch",
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": response["run_id"],
    })

    assert response["created_files"] == created_files
    assert response["alignment"] == alignment
    assert reloaded["created_files"] == created_files
    assert reloaded["alignment"] == alignment
    assert "writing_alignment:status=pass" in reloaded["trace_summary"]


async def test_revision_request_reconstructs_source_free_prior_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Source-free revision requests carry the stored generated package state."""

    from kazusa_ai_chatbot.coding_agent import continue_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    calls: list[dict[str, object]] = []
    patch_artifact = {
        "artifact_id": "patch-package",
        "base": "generated_artifacts",
        "diff_text": (
            "diff --git a/src/runtime.py b/src/runtime.py\n"
            "new file mode 100644\n"
            "index 0000000..1111111\n"
            "--- /dev/null\n"
            "+++ b/src/runtime.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def run() -> int:\n"
            "+    return 1\n"
        ),
        "files": ["src/runtime.py"],
        "summary": "Create runtime.",
    }
    created_files = [{
        "path": "src/runtime.py",
        "role": "Provide runtime logic.",
    }]

    async def fake_propose(request: dict[str, object]) -> dict[str, object]:
        calls.append(request)
        response = _proposal_response()
        response["patch_artifacts"] = [patch_artifact]
        response["created_files"] = created_files
        response["changed_files"] = [{
            "path": "src/runtime.py",
            "change_type": "create",
            "summary": "Create runtime.",
        }]
        return response

    monkeypatch.setattr(supervisor, "propose_code_change", fake_propose)

    started = await start_coding_run({
        **_proposal_request(tmp_path),
        "question": "Create a source-free runtime package.",
        "local_root_hint": None,
        "source_scope_hint": None,
    })
    revised = await continue_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": started["run_id"],
        "action": "revise_proposal",
        "revision_instruction": "Add a dry-run mode without losing runtime.py.",
    })

    assert revised["status"] == "awaiting_approval"
    assert len(calls) == 2
    prior_artifacts = calls[1]["prior_generated_artifacts"]
    assert prior_artifacts == [{
        "artifact_id": "prior_src_runtime_py",
        "file_label": "runtime.py",
        "file_kind": "source",
        "content_format": "python",
        "path": "src/runtime.py",
        "content": "def run() -> int:\n    return 1\n",
        "purpose": "Provide runtime logic.",
    }]


async def test_terminal_run_rejects_late_cancel_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Terminal runs keep their status after invalid continuations."""

    from kazusa_ai_chatbot.coding_agent import continue_coding_run
    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import supervisor

    async def fake_answer(request: dict[str, object]) -> dict[str, object]:
        return _reading_response(answer_text="Completed answer.")

    monkeypatch.setattr(supervisor, "answer_code_question", fake_answer)

    completed = await start_coding_run({
        "question": "Explain the app.",
        "objective_type": "read_only",
        "workspace_root": str(tmp_path / "workspace"),
    })
    rejected = await continue_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": completed["run_id"],
        "action": "cancel",
    })
    reloaded = await get_coding_run({
        "workspace_root": str(tmp_path / "workspace"),
        "run_id": completed["run_id"],
    })

    assert rejected["status"] == "rejected"
    assert reloaded["status"] == "completed"
    assert reloaded["events"][-1]["event_type"] == "rejected"


async def _fake_propose(request: dict[str, object]) -> dict[str, object]:
    return _proposal_response()


def _proposal_request(tmp_path: Path) -> dict[str, object]:
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    (source_root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    request = {
        "question": "Change VALUE to 2.",
        "objective_type": "propose_patch",
        "workspace_root": str(tmp_path / "workspace"),
        "local_root_hint": str(source_root),
        "source_scope_hint": "repository",
        "preferred_language": "English",
        "max_answer_chars": 2000,
        "max_artifact_chars": 8000,
        "session_id": "run-contract",
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
        "approved_by": "run-contract",
        "approved_at": "2026-07-09T00:00:00Z",
        "approval_reason": "Deterministic run contract.",
    }
    return approval
