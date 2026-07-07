"""Deterministic contracts for approved patch application."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_patching.models import PatchArtifact
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    compile_patch_operations,
)


MAX_DIFF_CHARS = 4000
MAX_FILES = 4


def test_patch_apply_rejects_missing_approval(tmp_path: Path) -> None:
    """Patch application requires a trusted structured approval object."""

    from kazusa_ai_chatbot.coding_agent import apply_approved_patch

    source_root = _source_root(tmp_path)
    patch_artifacts = _patch_artifacts(source_root)
    source_identity = _source_identity()

    response = apply_approved_patch({
        "workspace_root": str(tmp_path / "workspace"),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": patch_artifacts,
        "max_files": MAX_FILES,
        "max_diff_chars": MAX_DIFF_CHARS,
    })

    assert response["status"] == "rejected"
    assert response["apply_package_id"] == ""
    assert any("approval" in item.casefold() for item in response["limitations"])
    assert not (tmp_path / "workspace" / "patch_apply").exists()


def test_patch_apply_rejects_source_identity_mismatch(
    tmp_path: Path,
) -> None:
    """Stale source identity must fail closed before copying source."""

    from kazusa_ai_chatbot.coding_agent import apply_approved_patch

    source_root = _source_root(tmp_path)
    patch_artifacts = _patch_artifacts(source_root)
    source_identity = _source_identity()
    expected_identity = dict(source_identity)
    expected_identity["current_commit"] = "different"

    response = apply_approved_patch({
        "workspace_root": str(tmp_path / "workspace"),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": expected_identity,
        "patch_artifacts": patch_artifacts,
        "approval": _approval(),
        "max_files": MAX_FILES,
        "max_diff_chars": MAX_DIFF_CHARS,
    })

    assert response["status"] == "rejected"
    assert response["apply_package_id"] == ""
    assert any("identity" in item.casefold() for item in response["limitations"])
    assert not (tmp_path / "workspace" / "patch_apply").exists()


def test_patch_apply_succeeds_in_managed_copy_and_preserves_source(
    tmp_path: Path,
) -> None:
    """Approved patches apply into managed storage without source mutation."""

    from kazusa_ai_chatbot.coding_agent import apply_approved_patch

    source_root = _source_root(tmp_path)
    before_hashes = _hash_tree(source_root)
    patch_artifacts = _patch_artifacts(source_root)
    source_identity = _source_identity()

    response = apply_approved_patch({
        "workspace_root": str(tmp_path / "workspace"),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": patch_artifacts,
        "approval": _approval(),
        "max_files": MAX_FILES,
        "max_diff_chars": MAX_DIFF_CHARS,
    })
    after_hashes = _hash_tree(source_root)

    assert response["status"] == "succeeded"
    assert response["validation"]["status"] == "succeeded"
    assert response["applied_files"] == ["app.py"]
    assert response["changed_files"] == [{
        "path": "app.py",
        "change_type": "modify",
        "summary": "Compiled from structured patch operations.",
    }]
    assert response["apply_workspace_ref"]["kind"] == "managed_apply_workspace"
    assert response["apply_workspace_ref"]["apply_package_id"] == (
        response["apply_package_id"]
    )
    assert response["apply_workspace_ref"]["applied_files"] == ["app.py"]
    assert before_hashes == after_hashes

    applied_path = (
        tmp_path
        / "workspace"
        / "patch_apply"
        / response["apply_package_id"]
        / "source"
        / "app.py"
    )
    assert applied_path.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (source_root / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_patch_apply_rejects_unsafe_patch_path(tmp_path: Path) -> None:
    """Unsafe diff paths are rejected before managed apply storage is written."""

    from kazusa_ai_chatbot.coding_agent import apply_approved_patch

    source_root = _source_root(tmp_path)
    source_identity = _source_identity()
    response = apply_approved_patch({
        "workspace_root": str(tmp_path / "workspace"),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": [_unsafe_patch_artifact()],
        "approval": _approval(),
        "max_files": MAX_FILES,
        "max_diff_chars": MAX_DIFF_CHARS,
    })

    assert response["status"] == "rejected"
    assert response["apply_package_id"] == ""
    assert any("unsafe" in item.casefold() for item in response["limitations"])
    assert not (tmp_path / "workspace" / "patch_apply").exists()


def test_patch_apply_reports_patch_conflict(tmp_path: Path) -> None:
    """Patch conflicts fail without mutating the original source tree."""

    from kazusa_ai_chatbot.coding_agent import apply_approved_patch

    source_root = _source_root(tmp_path)
    before_hashes = _hash_tree(source_root)
    source_identity = _source_identity()
    response = apply_approved_patch({
        "workspace_root": str(tmp_path / "workspace"),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": [_conflicting_patch_artifact()],
        "approval": _approval(),
        "max_files": MAX_FILES,
        "max_diff_chars": MAX_DIFF_CHARS,
    })
    after_hashes = _hash_tree(source_root)

    assert response["status"] == "failed"
    assert response["apply_package_id"] == ""
    assert response["validation"]["status"] == "failed"
    assert any("apply" in item.casefold() for item in response["limitations"])
    assert before_hashes == after_hashes
    assert not (tmp_path / "workspace" / "patch_apply").exists()


def test_patch_apply_rejects_review_validation_failure_before_copy(
    tmp_path: Path,
) -> None:
    """Known-invalid patch artifacts cannot reach managed apply storage."""

    from kazusa_ai_chatbot.coding_agent import apply_approved_patch

    source_root = _source_root(tmp_path)
    before_hashes = _hash_tree(source_root)
    source_identity = _source_identity()
    response = apply_approved_patch({
        "workspace_root": str(tmp_path / "workspace"),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": [_invalid_python_patch_artifact()],
        "approval": _approval(),
        "max_files": MAX_FILES,
        "max_diff_chars": MAX_DIFF_CHARS,
    })
    after_hashes = _hash_tree(source_root)

    assert response["status"] == "failed"
    assert response["apply_package_id"] == ""
    assert response["validation"]["status"] == "failed"
    assert any(
        "syntactically valid" in item.casefold()
        for item in response["limitations"]
    )
    assert before_hashes == after_hashes
    assert not (tmp_path / "workspace" / "patch_apply").exists()


def test_patch_apply_public_response_is_sanitized(tmp_path: Path) -> None:
    """Public apply responses omit absolute roots and raw command output."""

    from kazusa_ai_chatbot.coding_agent import apply_approved_patch

    source_root = _source_root(tmp_path)
    patch_artifacts = _patch_artifacts(source_root)
    source_identity = _source_identity()
    workspace_root = tmp_path / "workspace"

    response = apply_approved_patch({
        "workspace_root": str(workspace_root),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": patch_artifacts,
        "approval": _approval(),
        "max_files": MAX_FILES,
        "max_diff_chars": MAX_DIFF_CHARS,
    })
    serialized = json.dumps(response, ensure_ascii=False)

    assert response["status"] == "succeeded"
    assert str(source_root.resolve()) not in serialized
    assert str(workspace_root.resolve()) not in serialized
    assert "stdout" not in serialized
    assert "stderr" not in serialized
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


def _unsafe_patch_artifact() -> PatchArtifact:
    artifact: PatchArtifact = {
        "artifact_id": "unsafe",
        "base": "repository",
        "diff_text": (
            "diff --git a/../outside.py b/../outside.py\n"
            "--- a/../outside.py\n"
            "+++ b/../outside.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        "files": ["../outside.py"],
        "summary": "Unsafe path.",
    }
    return artifact


def _conflicting_patch_artifact() -> PatchArtifact:
    artifact: PatchArtifact = {
        "artifact_id": "conflict",
        "base": "repository",
        "diff_text": (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 999\n"
            "+VALUE = 2\n"
        ),
        "files": ["app.py"],
        "summary": "Conflicting patch.",
    }
    return artifact


def _invalid_python_patch_artifact() -> PatchArtifact:
    artifact: PatchArtifact = {
        "artifact_id": "invalid-python",
        "base": "repository",
        "diff_text": (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+if broken\n"
        ),
        "files": ["app.py"],
        "summary": "Invalid Python patch.",
    }
    return artifact


def _hash_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative_path] = digest
    return hashes
