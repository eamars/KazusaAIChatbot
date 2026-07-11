"""Approved patch application into a managed coding-agent workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kazusa_ai_chatbot.coding_agent.code_patching.models import (
    CanonicalPatchOperationRecord,
    ChangedFileSummary,
    CodingPatchApplyRequest,
    CodingPatchApplyResponse,
    PatchApplyStatus,
    PatchArtifact,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    validate_canonical_operation_binding,
)
from kazusa_ai_chatbot.coding_agent.safety import (
    copy_managed_source_tree,
    managed_source_tree_digest,
)
from kazusa_ai_chatbot.config import CODING_AGENT_PREFLIGHT_EXECUTION
from kazusa_ai_chatbot.coding_agent.code_patching.patch_validation import (
    _git_apply_error,
    _parse_patch_artifacts,
    _run_git_apply,
    materialize_patch_artifacts_for_review,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

APPLY_ROOT_NAME = "patch_apply"
APPLIED_SOURCE_DIR_NAME = "source"
DEFAULT_MAX_DIFF_CHARS = 64_000
DEFAULT_MAX_FILES = 64


def apply_approved_patch(
    request: CodingPatchApplyRequest,
) -> CodingPatchApplyResponse:
    """Apply approved patch artifacts into a managed copy of source."""

    candidate_request = dict(request)
    candidate_request["authorization_purpose"] = "approved_verification"
    response = materialize_managed_candidate(candidate_request)
    return response


def materialize_managed_candidate(
    request: CodingPatchApplyRequest,
) -> CodingPatchApplyResponse:
    """Materialize a proposal in a managed candidate under closed authority."""

    source_identity = _mapping_or_empty(request.get("source_identity"))
    expected_identity = _mapping_or_empty(
        request.get("expected_source_identity"),
    )
    canonical_records = cast(
        list[CanonicalPatchOperationRecord],
        request.get("canonical_operation_records", []),
    )
    proposal_digest = request.get("proposal_digest", "")
    if not isinstance(proposal_digest, str):
        proposal_digest = ""
    canonical_binding_error = _canonical_binding_error(request)
    if canonical_binding_error:
        response = _response(
            status="rejected",
            source_identity=source_identity,
            errors=[canonical_binding_error],
            trace_summary=["patch_apply:rejected:canonical_binding"],
        )
        return response
    authorization_error = _authorization_error(request)
    if authorization_error:
        response = _response(
            status="rejected",
            source_identity=source_identity,
            errors=[authorization_error],
            trace_summary=["patch_apply:rejected:authorization"],
        )
        return response

    identity_error = _source_identity_error(
        source_identity=source_identity,
        expected_identity=expected_identity,
    )
    if identity_error:
        response = _response(
            status="rejected",
            source_identity=source_identity,
            errors=[identity_error],
            trace_summary=["patch_apply:rejected:source_identity"],
        )
        return response

    patch_artifacts = cast(
        list[PatchArtifact],
        request.get("patch_artifacts", []),
    )
    max_files = _positive_int(request.get("max_files"), DEFAULT_MAX_FILES)
    max_diff_chars = _positive_int(
        request.get("max_diff_chars"),
        DEFAULT_MAX_DIFF_CHARS,
    )
    parse_result = _parse_patch_artifacts(
        patch_artifacts=patch_artifacts,
        max_files=max_files,
        max_diff_chars=max_diff_chars,
    )
    files = cast(list[str], parse_result["files"])
    warnings = cast(list[str], parse_result["warnings"])
    errors = cast(list[str], parse_result["errors"])
    if errors:
        status = _parse_failure_status(errors)
        response = _response(
            status=status,
            source_identity=source_identity,
            errors=errors,
            warnings=warnings,
            trace_summary=["patch_apply:rejected:patch_artifacts"],
        )
        return response

    candidate_baseline = request.get("candidate_baseline", "resolved_source")
    roots_result = _candidate_roots(
        candidate_baseline=candidate_baseline,
        source_root_text=request.get("source_root"),
        workspace_root_text=request.get("workspace_root"),
    )
    if isinstance(roots_result, str):
        response = _response(
            status="rejected",
            source_identity=source_identity,
            errors=[roots_result],
            warnings=warnings,
            trace_summary=["patch_apply:rejected:path"],
        )
        return response
    source_root, workspace_root, source_free = roots_result

    review_validation = materialize_patch_artifacts_for_review(
        repo_root=None if source_free else source_root,
        workspace_root=workspace_root,
        patch_artifacts=patch_artifacts,
        max_files=max_files,
        max_diff_chars=max_diff_chars,
    )
    if review_validation["status"] != "succeeded":
        response = _response(
            status=review_validation["status"],
            source_identity=source_identity,
            errors=review_validation["errors"],
            warnings=review_validation["warnings"],
            trace_summary=["patch_apply:failed:review_validation"],
        )
        return response

    requested_package_id = request.get("apply_package_id")
    if requested_package_id is None:
        apply_package_id = uuid.uuid4().hex
        replace_existing_package = False
    elif _valid_package_id(requested_package_id):
        apply_package_id = str(requested_package_id)
        replace_existing_package = True
    else:
        response = _response(
            status="rejected",
            source_identity=source_identity,
            errors=["Managed apply package identity is invalid."],
            warnings=warnings,
            trace_summary=["patch_apply:rejected:package_identity"],
        )
        return response
    if source_free:
        apply_source_root_result = _create_empty_apply_workspace(
            workspace_root=workspace_root,
            apply_package_id=apply_package_id,
            replace_existing_package=replace_existing_package,
        )
    else:
        apply_source_root_result = _copy_source_to_apply_workspace(
            source_root=source_root,
            workspace_root=workspace_root,
            apply_package_id=apply_package_id,
            replace_existing_package=replace_existing_package,
        )
    if isinstance(apply_source_root_result, str):
        response = _response(
            status="failed",
            source_identity=source_identity,
            apply_package_id=apply_package_id,
            errors=[apply_source_root_result],
            warnings=warnings,
            trace_summary=["patch_apply:failed:copy"],
        )
        return response
    apply_source_root = apply_source_root_result

    diff_text = cast(str, parse_result["diff_text"])
    check_result = _run_git_apply(
        sandbox_root=apply_source_root,
        diff_text=diff_text,
    )
    check_error = _apply_error(check_result)
    if check_error:
        response = _response(
            status="failed",
            source_identity=source_identity,
            apply_package_id=apply_package_id,
            errors=[check_error],
            warnings=warnings,
            trace_summary=["patch_apply:failed:git_apply_check"],
        )
        return response

    apply_result = _run_git_apply(
        sandbox_root=apply_source_root,
        diff_text=diff_text,
        check_only=False,
    )
    apply_error = _apply_error(apply_result)
    if apply_error:
        response = _response(
            status="failed",
            source_identity=source_identity,
            apply_package_id=apply_package_id,
            errors=[apply_error],
            warnings=warnings,
            trace_summary=["patch_apply:failed:git_apply"],
        )
        return response

    candidate_tree_digest = request.get("candidate_tree_digest", "")
    if isinstance(candidate_tree_digest, str) and candidate_tree_digest:
        applied_tree_digest = managed_source_tree_digest(apply_source_root)
        if applied_tree_digest != candidate_tree_digest:
            response = _response(
                status="failed",
                source_identity=source_identity,
                apply_package_id=apply_package_id,
                errors=["Managed apply candidate identity does not match review."],
                warnings=warnings,
                trace_summary=["patch_apply:failed:candidate_identity"],
                canonical_operation_records=canonical_records,
                proposal_digest=proposal_digest,
                candidate_tree_digest=applied_tree_digest,
            )
            return response

    changed_files = _changed_file_summaries(
        source_root=source_root,
        files=files,
        patch_artifacts=patch_artifacts,
    )
    response = _response(
        status="succeeded",
        source_identity=source_identity,
        apply_package_id=apply_package_id,
        applied_files=files,
        changed_files=changed_files,
        warnings=warnings,
        trace_summary=[
            "patch_apply:authorization_accepted",
            "patch_apply:source_identity_matched",
            "patch_apply:managed_copy_created",
            "patch_apply:git_apply_succeeded",
        ],
        canonical_operation_records=canonical_records,
        proposal_digest=proposal_digest,
        candidate_tree_digest=(
            str(candidate_tree_digest)
            if isinstance(candidate_tree_digest, str)
            else ""
        ),
    )
    return response


def _authorization_error(request: Mapping[str, object]) -> str:
    """Validate the closed managed-candidate authorization purpose."""

    purpose = request.get("authorization_purpose")
    if purpose == "approved_verification":
        approval_error = _approval_error(request.get("approval"))
        if approval_error:
            return approval_error
        binding_error = _approval_binding_error(request)
        return binding_error
    if purpose == "preapproval_preflight":
        if not CODING_AGENT_PREFLIGHT_EXECUTION:
            return "Managed preflight execution is disabled."
        return ""
    return "Managed candidate authorization purpose is invalid."


def _canonical_binding_error(request: Mapping[str, object]) -> str:
    """Validate optional evaluation-time canonical patch provenance."""

    records_value = request.get("canonical_operation_records")
    digest_value = request.get("proposal_digest")
    revision_value = request.get("candidate_revision")
    if records_value is None and digest_value is None and revision_value is None:
        return ""
    if not isinstance(records_value, list):
        return "Canonical operation records are required."
    if not isinstance(digest_value, str) or not digest_value:
        return "Canonical operation proposal digest is required."
    if not isinstance(revision_value, int) or revision_value < 0:
        return "Canonical operation candidate revision is required."
    records = cast(list[CanonicalPatchOperationRecord], records_value)
    binding_error = validate_canonical_operation_binding(
        records=records,
        proposal_digest=digest_value,
        candidate_revision=revision_value,
    )
    return binding_error


def _candidate_roots(
    *,
    candidate_baseline: object,
    source_root_text: object,
    workspace_root_text: object,
) -> tuple[Path, Path, bool] | str:
    """Resolve baseline-specific candidate roots without widening containment."""

    if candidate_baseline == "resolved_source":
        roots_result = _resolve_roots(
            source_root_text=source_root_text,
            workspace_root_text=workspace_root_text,
        )
        if isinstance(roots_result, str):
            return roots_result
        source_root, workspace_root = roots_result
        return source_root, workspace_root, False
    if candidate_baseline == "empty_source_free":
        workspace_root = _workspace_root(workspace_root_text)
        if isinstance(workspace_root, str):
            return workspace_root
        return workspace_root, workspace_root, True
    return "Managed candidate baseline is invalid."


def _workspace_root(workspace_root_text: object) -> Path | str:
    """Resolve a workspace root for a source-free managed candidate."""

    if not isinstance(workspace_root_text, str) or not workspace_root_text.strip():
        return "Patch application requires a workspace root."
    workspace_root = Path(workspace_root_text).expanduser().resolve(strict=False)
    return workspace_root


def _approval_error(approval_value: object) -> str:
    if not isinstance(approval_value, Mapping):
        return "Patch application requires structured approval."
    if approval_value.get("approved") is not True:
        return "Patch application requires approved=True."
    for key in ("approved_by", "approved_at", "approval_reason"):
        value = approval_value.get(key)
        if not isinstance(value, str) or not value.strip():
            return "Patch application approval is incomplete."
    return ""


def _approval_binding_error(request: Mapping[str, object]) -> str:
    """Validate an optional internal approval-to-proposal binding."""

    binding = request.get("approval_binding")
    if binding is None:
        return ""
    if not isinstance(binding, Mapping):
        return "Patch application approval binding is invalid."
    if binding.get("schema_version") != "coding_action_loop_approval_binding.v1":
        return "Patch application approval binding is invalid."
    if binding.get("proposal_digest") != request.get("proposal_digest"):
        return "Patch application approval does not match the proposal."
    if binding.get("candidate_revision") != request.get("candidate_revision"):
        return "Patch application approval does not match the candidate revision."
    if binding.get("candidate_tree_digest") != request.get(
        "candidate_tree_digest"
    ):
        return "Patch application approval does not match the candidate tree."
    for field_name in (
        "approval_evidence_digest",
        "source_message_id",
    ):
        value = binding.get(field_name)
        if not isinstance(value, str) or not value:
            return "Patch application approval binding is incomplete."
    approval = request.get("approval")
    if not isinstance(approval, Mapping):
        return "Patch application approval binding is incomplete."
    evidence = approval.get("approval_evidence")
    if not isinstance(evidence, Mapping):
        return "Patch application approval evidence is incomplete."
    if binding.get("source_message_id") != evidence.get("source_message_id"):
        return "Patch application approval evidence identity does not match."
    serialized_evidence = json.dumps(
        dict(evidence),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    evidence_digest = hashlib.sha256(
        serialized_evidence.encode("utf-8"),
    ).hexdigest()
    if binding.get("approval_evidence_digest") != evidence_digest:
        return "Patch application approval evidence does not match."
    return ""


def _source_identity_error(
    *,
    source_identity: Mapping[str, object],
    expected_identity: Mapping[str, object],
) -> str:
    if not source_identity or not expected_identity:
        return "Patch application requires source identity."
    if source_identity.get("dirty_state") != "clean":
        return "Patch application requires a clean source identity."
    if expected_identity.get("dirty_state") != "clean":
        return "Patch application expected identity must be clean."
    identity_keys = ("provider", "owner", "repo", "current_commit", "dirty_state")
    for key in identity_keys:
        if source_identity.get(key) != expected_identity.get(key):
            return "Patch application source identity mismatch."
    return ""


def _resolve_roots(
    *,
    source_root_text: object,
    workspace_root_text: object,
) -> tuple[Path, Path] | str:
    if not isinstance(source_root_text, str) or not source_root_text.strip():
        return "Patch application requires a source root."
    if not isinstance(workspace_root_text, str) or not workspace_root_text.strip():
        return "Patch application requires a workspace root."
    try:
        source_root = Path(source_root_text).expanduser().resolve(strict=True)
    except OSError:
        return "Patch application source root cannot be resolved."
    if not source_root.is_dir():
        return "Patch application source root must be a directory."
    workspace_root = (
        Path(workspace_root_text)
        .expanduser()
        .resolve(strict=False)
    )
    if workspace_root == source_root or workspace_root.is_relative_to(source_root):
        return "Patch application workspace must be separate from source."
    return source_root, workspace_root


def _copy_source_to_apply_workspace(
    *,
    source_root: Path,
    workspace_root: Path,
    apply_package_id: str,
    replace_existing_package: bool,
) -> Path | str:
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        unresolved_apply_root = workspace_root / APPLY_ROOT_NAME
        if unresolved_apply_root.is_symlink():
            return "Patch application workspace is unsafe."
        apply_root = ensure_path_inside(
            unresolved_apply_root,
            workspace_root,
        )
        apply_root.mkdir(parents=True, exist_ok=True)
        unresolved_package_root = apply_root / apply_package_id
        if unresolved_package_root.is_symlink():
            return "Patch application package path is unsafe."
        package_root = ensure_path_inside(
            unresolved_package_root,
            workspace_root,
        )
        apply_source_root = ensure_path_inside(
            package_root / APPLIED_SOURCE_DIR_NAME,
            workspace_root,
        )
        if replace_existing_package and package_root.exists():
            shutil.rmtree(package_root)
        package_root.mkdir(parents=True, exist_ok=False)
        copy_managed_source_tree(
            source_root,
            apply_source_root,
            extra_excluded_names=(APPLY_ROOT_NAME,),
        )
    except (OSError, PathSafetyError):
        return "Patch application could not create the managed workspace."
    return apply_source_root


def _create_empty_apply_workspace(
    *,
    workspace_root: Path,
    apply_package_id: str,
    replace_existing_package: bool,
) -> Path | str:
    """Create an empty Git-backed managed candidate for generated artifacts."""

    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        unresolved_apply_root = workspace_root / APPLY_ROOT_NAME
        if unresolved_apply_root.is_symlink():
            return "Patch application workspace is unsafe."
        apply_root = ensure_path_inside(
            unresolved_apply_root,
            workspace_root,
        )
        apply_root.mkdir(parents=True, exist_ok=True)
        unresolved_package_root = apply_root / apply_package_id
        if unresolved_package_root.is_symlink():
            return "Patch application package path is unsafe."
        package_root = ensure_path_inside(
            unresolved_package_root,
            workspace_root,
        )
        apply_source_root = ensure_path_inside(
            package_root / APPLIED_SOURCE_DIR_NAME,
            workspace_root,
        )
        if replace_existing_package and package_root.exists():
            shutil.rmtree(package_root)
        package_root.mkdir(parents=True, exist_ok=False)
        apply_source_root.mkdir()
    except (OSError, PathSafetyError):
        return "Patch application could not create the managed workspace."

    try:
        initialized = subprocess.run(
            ["git", "init", "-q"],
            cwd=apply_source_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except FileNotFoundError:
        return "git executable is unavailable for patch application."
    except subprocess.TimeoutExpired:
        return "Patch application managed workspace initialization timed out."
    if initialized.returncode != 0:
        return "Patch application could not initialize the managed workspace."
    return apply_source_root


def _valid_package_id(value: object) -> bool:
    """Return whether an apply package id is safe for a managed path."""

    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        return False
    valid = all(
        character.isalnum() or character in {"-", "_"}
        for character in value
    )
    return valid


def _apply_error(result: object) -> str:
    if isinstance(result, str):
        return "Patch application could not run git apply."
    if result.returncode == 0:
        return ""
    error = _git_apply_error(result.stderr)
    return error


def _changed_file_summaries(
    *,
    source_root: Path,
    files: list[str],
    patch_artifacts: list[PatchArtifact],
) -> list[ChangedFileSummary]:
    summaries: list[ChangedFileSummary] = []
    summary_by_file = _artifact_summary_by_file(patch_artifacts)
    for safe_path in files:
        source_path = ensure_path_inside(source_root / safe_path, source_root)
        change_type = "modify"
        if not source_path.exists():
            change_type = "create"
        summary: ChangedFileSummary = {
            "path": safe_path,
            "change_type": change_type,
            "summary": summary_by_file.get(safe_path, "Applied patch artifact."),
        }
        summaries.append(summary)
    return summaries


def _artifact_summary_by_file(
    patch_artifacts: list[PatchArtifact],
) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for artifact in patch_artifacts:
        summary_value = artifact.get("summary", "")
        if not isinstance(summary_value, str) or not summary_value.strip():
            summary_value = "Applied patch artifact."
        artifact_files = artifact.get("files", [])
        if not isinstance(artifact_files, list):
            continue
        for path_value in artifact_files:
            if isinstance(path_value, str):
                summaries[path_value] = summary_value
    return summaries


def _parse_failure_status(errors: list[str]) -> PatchApplyStatus:
    rejected_fragments = (
        "unsafe",
        "exceeds",
        "too many",
        "No patch artifacts",
        "No safe file paths",
    )
    for error in errors:
        for fragment in rejected_fragments:
            if fragment.casefold() in error.casefold():
                return "rejected"
    return "failed"


def _positive_int(value: object, fallback: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def _mapping_or_empty(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        mapped = dict(value)
        return mapped
    return {}


def _workspace_ref(
    *,
    apply_package_id: str,
    source_identity: Mapping[str, object],
    applied_files: list[str],
) -> dict[str, object]:
    kind = ""
    if apply_package_id:
        kind = "managed_apply_workspace"
    workspace_ref = {
        "kind": kind,
        "apply_package_id": apply_package_id,
        "source_identity": dict(source_identity),
        "applied_files": list(applied_files),
    }
    return workspace_ref


def _response(
    *,
    status: PatchApplyStatus,
    source_identity: Mapping[str, object],
    apply_package_id: str = "",
    applied_files: list[str] | None = None,
    changed_files: list[ChangedFileSummary] | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    trace_summary: list[str] | None = None,
    canonical_operation_records: list[CanonicalPatchOperationRecord] | None = None,
    proposal_digest: str = "",
    candidate_tree_digest: str = "",
) -> CodingPatchApplyResponse:
    safe_applied_files = applied_files or []
    safe_errors = errors or []
    safe_warnings = warnings or []
    response: CodingPatchApplyResponse = {
        "status": status,
        "apply_package_id": apply_package_id,
        "source_identity": dict(source_identity),
        "apply_workspace_ref": cast(
            dict[str, object],
            _workspace_ref(
                apply_package_id=apply_package_id,
                source_identity=source_identity,
                applied_files=safe_applied_files,
            ),
        ),
        "applied_files": safe_applied_files,
        "changed_files": changed_files or [],
        "validation": {
            "status": status,
            "errors": safe_errors,
            "warnings": safe_warnings,
        },
        "limitations": safe_errors,
        "trace_summary": trace_summary or [],
        "canonical_operation_records": canonical_operation_records or [],
        "proposal_digest": proposal_digest,
        "candidate_tree_digest": candidate_tree_digest,
    }
    return response
