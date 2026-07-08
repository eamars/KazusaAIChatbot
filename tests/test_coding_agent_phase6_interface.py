"""Phase 6 public interface and managed-workspace integration contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_patching.models import PatchArtifact
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    compile_patch_operations,
)


MAX_DIFF_CHARS = 4000
MAX_FILES = 4


def test_code_execution_public_api_is_exported() -> None:
    """The top-level coding-agent package exposes the trusted executor."""

    from kazusa_ai_chatbot.coding_agent import (
        CodeExecutionRequest,
        CodeExecutionResponse,
        execute_code_check,
    )

    assert isinstance(execute_code_check, Callable)
    assert "apply_workspace_ref" in CodeExecutionRequest.__annotations__
    assert "stdout_excerpt" in CodeExecutionResponse.__annotations__


def test_execute_code_check_runs_in_phase5_apply_workspace(
    tmp_path: Path,
) -> None:
    """Execution uses the Phase 5 managed apply copy, not source root."""

    from kazusa_ai_chatbot.coding_agent import (
        apply_approved_patch,
        execute_code_check,
    )

    source_root = _source_root(tmp_path)
    before_text = (source_root / "app.py").read_text(encoding="utf-8")
    source_identity = _source_identity()
    workspace_root = tmp_path / "workspace"
    apply_response = apply_approved_patch({
        "workspace_root": str(workspace_root),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": _patch_artifacts(source_root),
        "approval": _approval(),
        "max_files": MAX_FILES,
        "max_diff_chars": MAX_DIFF_CHARS,
    })

    execution_response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_response["apply_package_id"],
        "apply_workspace_ref": apply_response["apply_workspace_ref"],
        "execution": {
            "tool": "python_compileall",
            "paths": ["app.py"],
            "pytest_selectors": [],
            "timeout_seconds": 10,
        },
    })

    assert apply_response["status"] == "succeeded"
    assert execution_response["status"] == "succeeded"
    assert execution_response["executed_paths"] == ["app.py"]
    assert (source_root / "app.py").read_text(encoding="utf-8") == before_text

    applied_path = (
        workspace_root
        / "patch_apply"
        / apply_response["apply_package_id"]
        / "source"
        / "app.py"
    )
    assert applied_path.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_execute_code_check_public_response_is_sanitized(
    tmp_path: Path,
) -> None:
    """Public execution metadata omits absolute roots and environment values."""

    from kazusa_ai_chatbot.coding_agent import (
        apply_approved_patch,
        execute_code_check,
    )

    source_root = _source_root(tmp_path)
    source_identity = _source_identity()
    workspace_root = tmp_path / "workspace"
    apply_response = apply_approved_patch({
        "workspace_root": str(workspace_root),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": _patch_artifacts(source_root),
        "approval": _approval(),
        "max_files": MAX_FILES,
        "max_diff_chars": MAX_DIFF_CHARS,
    })
    execution_response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_response["apply_package_id"],
        "apply_workspace_ref": apply_response["apply_workspace_ref"],
        "execution": {
            "tool": "python_compileall",
            "paths": ["app.py"],
            "pytest_selectors": [],
            "timeout_seconds": 10,
        },
    })
    serialized = json.dumps(execution_response, ensure_ascii=False)

    assert execution_response["status"] == "succeeded"
    assert str(source_root.resolve()) not in serialized
    assert str(workspace_root.resolve()) not in serialized
    assert "TOKEN=" not in serialized
    assert "diff --git" not in serialized


def _source_root(tmp_path: Path) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return source_root


def _patch_artifacts(source_root: Path) -> list[PatchArtifact]:
    patch_artifacts, _, _, errors = compile_patch_operations(
        repo_root=source_root,
        patch_operations=[{
            "operation_id": "replace-value",
            "kind": "replace",
            "path": "app.py",
            "anchor": "VALUE = 1\n",
            "content": "VALUE = 2\n",
            "summary": "Replace one value.",
        }],
        max_files=MAX_FILES,
        max_diff_chars=MAX_DIFF_CHARS,
    )
    assert errors == []
    return patch_artifacts


def _source_identity() -> dict[str, object]:
    source_identity = {
        "provider": "github",
        "owner": "fixture",
        "repo": "demo",
        "current_commit": "abc123",
        "dirty_state": "clean",
    }
    return source_identity


def _approval() -> dict[str, object]:
    approval = {
        "approved": True,
        "approved_by": "contract-test",
        "approved_at": "2026-07-08T00:00:00Z",
        "approval_reason": "Focused deterministic test.",
    }
    return approval
