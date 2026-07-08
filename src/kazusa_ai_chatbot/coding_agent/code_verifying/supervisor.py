"""Direct orchestration for bounded verify-and-repair work."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import kazusa_ai_chatbot.coding_agent.code_fetching as code_fetching
from kazusa_ai_chatbot.coding_agent.code_executing import (
    execute_code_check,
)
from kazusa_ai_chatbot.coding_agent.code_executing.models import (
    CodeExecutionResponse,
    CodeExecutionSpec,
)
from kazusa_ai_chatbot.coding_agent.code_patching.apply import (
    apply_approved_patch,
)
from kazusa_ai_chatbot.coding_agent.code_patching.models import (
    ChangedFileSummary,
    CodingPatchApplyResponse,
    PatchArtifact,
)
from kazusa_ai_chatbot.coding_agent.code_verifying.models import (
    CodingVerifyRepairRequest,
    CodingVerifyRepairResponse,
    ExecutionRepairFeedback,
    VerifyRepairAttempt,
    VerifyRepairStatus,
)
from kazusa_ai_chatbot.coding_agent.models import (
    CodingPatchProposalResponse,
)
from kazusa_ai_chatbot.coding_agent.supervisor import (
    propose_code_change,
)

ALLOWED_EXECUTION_TOOLS = {"python_compileall", "pytest"}
DEFAULT_REPAIR_ATTEMPTS = 1
MAX_REPAIR_ATTEMPTS = 2
DEFAULT_REPAIR_FEEDBACK_CHARS = 4000
MIN_REPAIR_EXCERPT_CHARS = 200
SUMMARY_LIMIT = 8
COMMAND_OUTPUT_PATTERNS = (
    re.compile(r"python(?:\.exe)?\s+-m\s+pytest[^\n\r]*", re.IGNORECASE),
    re.compile(r"python(?:\.exe)?\s+-m\s+compileall[^\n\r]*", re.IGNORECASE),
)
FAILED_TEST_RE = re.compile(r"FAILED\s+([^\s]+)")
FAILURE_SUMMARY_MARKERS = (
    "FAILED ",
    "AssertionError",
    "Error:",
    "Exception:",
    "Traceback",
    "timed out",
)


async def verify_and_repair_code_change(
    request: CodingVerifyRepairRequest,
) -> CodingVerifyRepairResponse:
    """Apply a proposal, run checks, and attempt bounded repair."""

    validation_error = _request_validation_error(request)
    if validation_error:
        response = _terminal_response(
            status="rejected",
            limitations=[validation_error],
            trace_summary=["verify_repair:rejected:request"],
        )
        return response

    fetching_result = await code_fetching.run(request)
    if fetching_result["status"] != "succeeded":
        response = _terminal_response(
            status=fetching_result["status"],
            limitations=fetching_result["limitations"],
            trace_summary=fetching_result["trace_summary"],
        )
        return response

    repository = fetching_result["repository"]
    source_scope = fetching_result["source_scope"]
    if repository is None or source_scope is None:
        response = _terminal_response(
            status="failed",
            limitations=["Source fetching succeeded without source contract."],
            trace_summary=fetching_result["trace_summary"],
        )
        return response

    source_identity = _source_identity_from_repository(repository)
    expected_identity = _expected_source_identity(request, source_identity)
    proposal_response = await _initial_proposal_response(
        request=request,
        repository=repository,
        source_scope=source_scope,
    )
    if proposal_response["status"] != "succeeded":
        response = _terminal_response(
            status=proposal_response["status"],
            repository=_repository_summary(repository),
            source_scope=source_scope,
            final_patch_artifacts=proposal_response["patch_artifacts"],
            final_changed_files=proposal_response["changed_files"],
            limitations=proposal_response["limitations"],
            trace_summary=[
                *fetching_result["trace_summary"],
                *proposal_response["trace_summary"],
            ],
        )
        return response

    protected_paths = _protected_verification_paths(request["execution_specs"])
    required_owner_paths = _required_source_owner_paths(
        proposal_response["patch_artifacts"],
        protected_paths=protected_paths,
    )
    repair_attempt_limit = _repair_attempt_limit(request)
    patch_artifacts = proposal_response["patch_artifacts"]
    proposal_status = proposal_response["status"]
    attempts: list[VerifyRepairAttempt] = []
    final_apply: CodingPatchApplyResponse | None = None
    final_execution: list[CodeExecutionResponse] = []
    limitations = [*fetching_result["limitations"], *proposal_response["limitations"]]
    trace_summary = [
        *fetching_result["trace_summary"],
        *proposal_response["trace_summary"],
    ]

    for attempt_index in range(1, repair_attempt_limit + 2):
        apply_response = apply_approved_patch({
            "workspace_root": request["workspace_root"],
            "source_root": repository["local_root"],
            "source_identity": source_identity,
            "expected_source_identity": expected_identity,
            "patch_artifacts": patch_artifacts,
            "approval": request["approval"],
            "max_diff_chars": request.get("max_artifact_chars"),
        })
        final_apply = _sanitize_public_value(
            apply_response,
            source_root=Path(repository["local_root"]),
            workspace_root=Path(request["workspace_root"]),
        )
        if apply_response["status"] != "succeeded":
            attempt = _attempt_summary(
                attempt_index=attempt_index,
                proposal_status=proposal_status,
                apply_response=final_apply,
                execution_results=[],
                patch_artifacts=patch_artifacts,
            )
            attempts.append(attempt)
            limitations.extend(apply_response["limitations"])
            trace_summary.extend(apply_response["trace_summary"])
            response = _terminal_response(
                status=apply_response["status"],
                repository=_repository_summary(repository),
                source_scope=source_scope,
                attempts=attempts,
                final_patch_artifacts=patch_artifacts,
                final_changed_files=apply_response["changed_files"],
                final_apply=final_apply,
                limitations=limitations,
                trace_summary=trace_summary,
            )
            return response

        execution_results = _run_execution_specs(
            request=request,
            apply_response=apply_response,
        )
        final_execution = _sanitize_public_value(
            execution_results,
            source_root=Path(repository["local_root"]),
            workspace_root=Path(request["workspace_root"]),
        )
        attempt = _attempt_summary(
            attempt_index=attempt_index,
            proposal_status=proposal_status,
            apply_response=final_apply,
            execution_results=final_execution,
            patch_artifacts=patch_artifacts,
        )
        attempts.append(attempt)
        trace_summary.extend(apply_response["trace_summary"])
        trace_summary.extend(_execution_trace(execution_results))

        if _all_execution_succeeded(execution_results):
            response = _terminal_response(
                status="succeeded",
                repository=_repository_summary(repository),
                source_scope=source_scope,
                attempts=attempts,
                final_patch_artifacts=patch_artifacts,
                final_changed_files=apply_response["changed_files"],
                final_apply=final_apply,
                final_execution=final_execution,
                limitations=limitations,
                trace_summary=trace_summary,
                answer_text="Verification succeeded after applying the patch.",
            )
            return response

        terminal_status = _terminal_status_from_execution(execution_results)
        if terminal_status == "rejected" or attempt_index > repair_attempt_limit:
            response = _terminal_response(
                status=terminal_status,
                repository=_repository_summary(repository),
                source_scope=source_scope,
                attempts=attempts,
                final_patch_artifacts=patch_artifacts,
                final_changed_files=apply_response["changed_files"],
                final_apply=final_apply,
                final_execution=final_execution,
                limitations=limitations,
                trace_summary=trace_summary,
                answer_text="Verification did not succeed.",
            )
            return response

        repair_feedback = build_execution_repair_feedback(
            attempt_index=attempt_index,
            execution_results=execution_results,
            workspace_root=Path(request["workspace_root"]),
            source_root=Path(repository["local_root"]),
            max_chars=_max_repair_feedback_chars(request),
        )
        repair_request = build_repair_proposal_request(
            base_request=request,
            repository=repository,
            source_scope=source_scope,
            repair_feedback=repair_feedback,
            previous_patch_artifacts=patch_artifacts,
            required_source_owner_paths=required_owner_paths,
            protected_verification_paths=protected_paths,
        )
        proposal_response = await propose_code_change(repair_request)
        proposal_errors = validate_repair_proposal(
            proposal=proposal_response,
            required_source_owner_paths=required_owner_paths,
            protected_verification_paths=protected_paths,
        )
        if proposal_errors:
            limitations.extend(proposal_errors)
            response = _terminal_response(
                status="failed",
                repository=_repository_summary(repository),
                source_scope=source_scope,
                attempts=attempts,
                final_patch_artifacts=proposal_response["patch_artifacts"],
                final_changed_files=proposal_response["changed_files"],
                final_apply=final_apply,
                final_execution=final_execution,
                limitations=limitations,
                trace_summary=[
                    *trace_summary,
                    "verify_repair:repair_proposal_rejected",
                ],
                answer_text="Repair proposal failed verification constraints.",
            )
            return response
        patch_artifacts = proposal_response["patch_artifacts"]
        required_owner_paths = _required_source_owner_paths(
            patch_artifacts,
            protected_paths=protected_paths,
        )
        proposal_status = proposal_response["status"]
        trace_summary.extend(proposal_response["trace_summary"])

    response = _terminal_response(
        status="failed",
        repository=_repository_summary(repository),
        source_scope=source_scope,
        attempts=attempts,
        final_patch_artifacts=patch_artifacts,
        final_apply=final_apply,
        final_execution=final_execution,
        limitations=limitations,
        trace_summary=trace_summary,
    )
    return response


def build_execution_repair_feedback(
    *,
    attempt_index: int,
    execution_results: list[CodeExecutionResponse],
    workspace_root: Path,
    source_root: Path,
    max_chars: int,
) -> ExecutionRepairFeedback:
    """Convert failed execution responses into bounded repair evidence."""

    failed_results = [
        result
        for result in execution_results
        if result["status"] in {"failed", "timed_out"}
    ]
    failed_tools = _unique_strings(result["tool"] for result in failed_results)
    failed_paths = _failed_paths(failed_results)
    exit_codes = [
        {
            "tool": result["tool"],
            "exit_code": result["exit_code"],
            "status": result["status"],
        }
        for result in failed_results
    ]
    stdout_text = "\n".join(result["stdout_excerpt"] for result in failed_results)
    stderr_text = "\n".join(result["stderr_excerpt"] for result in failed_results)
    redacted_stdout = _redact_execution_text(
        stdout_text,
        workspace_root=workspace_root,
        source_root=source_root,
    )
    redacted_stderr = _redact_execution_text(
        stderr_text,
        workspace_root=workspace_root,
        source_root=source_root,
    )
    excerpt_cap = _feedback_excerpt_cap(max_chars)
    stdout_excerpt = redacted_stdout[:excerpt_cap]
    stderr_excerpt = redacted_stderr[:excerpt_cap]
    failure_summaries = _failure_summaries(
        stdout_excerpt=stdout_excerpt,
        stderr_excerpt=stderr_excerpt,
    )
    overall_status = _terminal_status_from_execution(failed_results)
    output_truncated = any(result["output_truncated"] for result in failed_results)
    if len(redacted_stdout) > len(stdout_excerpt):
        output_truncated = True
    if len(redacted_stderr) > len(stderr_excerpt):
        output_truncated = True

    feedback: ExecutionRepairFeedback = {
        "feedback_source": "execution_verification",
        "attempt_index": attempt_index,
        "overall_status": overall_status,
        "failed_tools": failed_tools,
        "failed_paths": failed_paths,
        "exit_codes": exit_codes,
        "failure_summaries": failure_summaries,
        "stdout_excerpt": stdout_excerpt,
        "stderr_excerpt": stderr_excerpt,
        "output_truncated": output_truncated,
        "instruction": (
            "Return a corrected complete patch proposal that fixes the failed "
            "verification behavior without editing protected verification tests."
        ),
    }
    bounded_feedback = _bounded_feedback(feedback, max_chars=max_chars)
    return bounded_feedback


def build_repair_proposal_request(
    *,
    base_request: Mapping[str, object],
    repository: Mapping[str, object],
    source_scope: Mapping[str, object],
    repair_feedback: ExecutionRepairFeedback,
    previous_patch_artifacts: list[PatchArtifact],
    required_source_owner_paths: list[str],
    protected_verification_paths: list[str],
) -> dict[str, object]:
    """Build the source-backed repair proposal request."""

    request = _proposal_request_from_base(base_request)
    enriched_feedback: dict[str, object] = {
        **repair_feedback,
        "required_source_owner_paths": required_source_owner_paths,
        "protected_verification_paths": protected_verification_paths,
        "previous_patch_artifacts": previous_patch_artifacts,
        "source_scope": dict(source_scope),
    }
    request["question"] = str(base_request.get("question") or "")
    request["local_root_hint"] = str(repository["local_root"])
    request["repair_feedback"] = enriched_feedback
    return request


def validate_repair_proposal(
    *,
    proposal: CodingPatchProposalResponse,
    required_source_owner_paths: list[str],
    protected_verification_paths: list[str],
) -> list[str]:
    """Validate repaired proposal scope before a managed apply attempt."""

    if proposal["status"] != "succeeded":
        errors = list(proposal["limitations"])
        if not errors:
            errors = ["Repair proposal did not succeed."]
        return errors

    validation = proposal["validation"]
    validation_errors = validation.get("errors")
    if isinstance(validation_errors, list) and validation_errors:
        errors = [
            str(error)
            for error in validation_errors
        ]
        return errors

    changed_paths = _changed_paths_from_proposal(proposal)
    errors: list[str] = []
    for path in required_source_owner_paths:
        if path in changed_paths:
            continue
        errors.append(f"Repair proposal omitted required source owner path: {path}")
    for path in protected_verification_paths:
        if path not in changed_paths:
            continue
        errors.append(f"Repair proposal modified protected verification path: {path}")
    return errors


def _request_validation_error(request: Mapping[str, object]) -> str:
    approval_error = _approval_error(request.get("approval"))
    if approval_error:
        return approval_error
    if not _has_source_fields(request):
        return "Verify repair requires an explicit source-backed request."
    workspace_root = request.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root.strip():
        return "Verify repair requires a workspace root."
    execution_specs = request.get("execution_specs")
    if not isinstance(execution_specs, list) or not execution_specs:
        return "Verify repair requires at least one execution spec."
    for spec in execution_specs:
        if not isinstance(spec, Mapping):
            return "Verify repair execution specs must be structured objects."
        spec_error = _execution_spec_error(spec)
        if spec_error:
            return spec_error
    initial_artifacts = request.get("initial_patch_artifacts")
    if initial_artifacts is not None and not isinstance(initial_artifacts, list):
        return "Verify repair initial patch artifacts must be a list."
    if isinstance(initial_artifacts, list) and initial_artifacts:
        expected_identity = request.get("expected_source_identity")
        if not isinstance(expected_identity, Mapping):
            return "Verify repair requires expected source identity."
    return ""


def _approval_error(approval_value: object) -> str:
    if not isinstance(approval_value, Mapping):
        return "Verify repair requires structured approval."
    if approval_value.get("approved") is not True:
        return "Verify repair requires approved=True."
    for key in ("approved_by", "approved_at", "approval_reason"):
        value = approval_value.get(key)
        if not isinstance(value, str) or not value.strip():
            return "Verify repair approval is incomplete."
    return ""


def _has_source_fields(request: Mapping[str, object]) -> bool:
    source_fields = (
        "source_url",
        "repo_url",
        "repo_hint",
        "local_root_hint",
        "local_path_hint",
    )
    has_source = any(bool(request.get(field)) for field in source_fields)
    return has_source


def _execution_spec_error(spec: Mapping[str, object]) -> str:
    tool = spec.get("tool")
    if tool not in ALLOWED_EXECUTION_TOOLS:
        return "Verify repair execution tool is unsupported."
    if tool == "python_compileall":
        paths = spec.get("paths")
        if not isinstance(paths, list) or not paths:
            return "Verify repair compile execution requires paths."
    if tool == "pytest":
        selectors = spec.get("pytest_selectors")
        if not isinstance(selectors, list) or not selectors:
            return "Verify repair pytest execution requires selectors."
    return ""


async def _initial_proposal_response(
    *,
    request: CodingVerifyRepairRequest,
    repository: Mapping[str, object],
    source_scope: Mapping[str, object],
) -> CodingPatchProposalResponse:
    initial_patch_artifacts = request.get("initial_patch_artifacts")
    if isinstance(initial_patch_artifacts, list) and initial_patch_artifacts:
        changed_paths = _paths_from_patch_artifacts(initial_patch_artifacts)
        proposal = _proposal_response_from_artifacts(
            patch_artifacts=initial_patch_artifacts,
            repository=_repository_summary(repository),
            source_scope=source_scope,
            changed_paths=changed_paths,
        )
        return proposal

    proposal_request = _proposal_request_from_base(request)
    proposal = await propose_code_change(proposal_request)
    return proposal


def _proposal_response_from_artifacts(
    *,
    patch_artifacts: list[PatchArtifact],
    repository: dict[str, object],
    source_scope: Mapping[str, object],
    changed_paths: list[str],
) -> CodingPatchProposalResponse:
    changed_files = [
        {
            "path": path,
            "change_type": "modify",
            "summary": "Seeded patch artifact.",
        }
        for path in changed_paths
    ]
    proposal: CodingPatchProposalResponse = {
        "status": "succeeded",
        "mode": "edit_existing_repository",
        "answer_text": "Using supplied patch artifacts.",
        "repository": repository,
        "source_scope": dict(source_scope),  # type: ignore[typeddict-item]
        "evidence": [],
        "patch_artifacts": patch_artifacts,
        "created_files": [],
        "changed_files": changed_files,
        "validation": {
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": changed_paths,
        },
        "external_evidence": [],
        "session": None,
        "limitations": [],
        "trace_summary": ["verify_repair:initial_artifacts_supplied"],
    }
    return proposal


def _proposal_request_from_base(
    base_request: Mapping[str, object],
) -> dict[str, object]:
    request: dict[str, object] = {}
    fields = (
        "question",
        "source_url",
        "repo_url",
        "repo_hint",
        "local_root_hint",
        "local_path_hint",
        "requested_ref",
        "source_scope_hint",
        "workspace_root",
        "preferred_language",
        "max_answer_chars",
        "max_artifact_chars",
        "session_id",
    )
    for field in fields:
        value = base_request.get(field)
        if value is not None:
            request[field] = value
    return request


def _run_execution_specs(
    *,
    request: CodingVerifyRepairRequest,
    apply_response: CodingPatchApplyResponse,
) -> list[CodeExecutionResponse]:
    execution_results: list[CodeExecutionResponse] = []
    for execution_spec in request["execution_specs"]:
        execution_result = execute_code_check({
            "workspace_root": request["workspace_root"],
            "apply_package_id": apply_response["apply_package_id"],
            "apply_workspace_ref": apply_response["apply_workspace_ref"],
            "execution": execution_spec,
        })
        execution_results.append(execution_result)
    return execution_results


def _attempt_summary(
    *,
    attempt_index: int,
    proposal_status: str,
    apply_response: CodingPatchApplyResponse,
    execution_results: list[CodeExecutionResponse],
    patch_artifacts: list[PatchArtifact],
) -> VerifyRepairAttempt:
    limitations = [
        *apply_response["limitations"],
        *[
            limitation
            for result in execution_results
            for limitation in result["limitations"]
        ],
    ]
    trace_summary = [
        *apply_response["trace_summary"],
        *_execution_trace(execution_results),
    ]
    attempt: VerifyRepairAttempt = {
        "attempt_index": attempt_index,
        "proposal_status": proposal_status,
        "apply_status": apply_response["status"],
        "execution_statuses": [
            result["status"]
            for result in execution_results
        ],
        "patch_artifact_count": len(patch_artifacts),
        "changed_files": apply_response["changed_files"],
        "apply_package_id": apply_response["apply_package_id"] or None,
        "limitations": limitations,
        "trace_summary": trace_summary,
    }
    return attempt


def _all_execution_succeeded(
    execution_results: list[CodeExecutionResponse],
) -> bool:
    result = all(item["status"] == "succeeded" for item in execution_results)
    return result


def _terminal_status_from_execution(
    execution_results: list[CodeExecutionResponse],
) -> VerifyRepairStatus:
    if any(result["status"] == "rejected" for result in execution_results):
        return "rejected"
    if any(result["status"] == "timed_out" for result in execution_results):
        return "timed_out"
    return "failed"


def _execution_trace(
    execution_results: list[CodeExecutionResponse],
) -> list[str]:
    trace: list[str] = []
    for result in execution_results:
        trace.extend(result["trace_summary"])
    return trace


def _repair_attempt_limit(request: Mapping[str, object]) -> int:
    value = request.get("repair_attempt_limit")
    if not isinstance(value, int):
        return DEFAULT_REPAIR_ATTEMPTS
    if value < 0:
        return 0
    if value > MAX_REPAIR_ATTEMPTS:
        return MAX_REPAIR_ATTEMPTS
    return value


def _max_repair_feedback_chars(request: Mapping[str, object]) -> int:
    value = request.get("max_repair_feedback_chars")
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_REPAIR_FEEDBACK_CHARS


def _expected_source_identity(
    request: Mapping[str, object],
    source_identity: dict[str, object],
) -> dict[str, object]:
    expected_identity = request.get("expected_source_identity")
    if isinstance(expected_identity, Mapping):
        mapped_identity = dict(expected_identity)
        return mapped_identity
    return dict(source_identity)


def _source_identity_from_repository(
    repository: Mapping[str, object],
) -> dict[str, object]:
    source_identity = {
        "provider": repository["provider"],
        "owner": repository["owner"],
        "repo": repository["repo"],
        "current_commit": repository["current_commit"],
        "dirty_state": repository["dirty_state"],
    }
    return source_identity


def _repository_summary(repository: Mapping[str, object]) -> dict[str, object]:
    summary = {
        "provider": repository["provider"],
        "owner": repository["owner"],
        "repo": repository["repo"],
        "source_url": repository["source_url"],
        "requested_ref": repository["requested_ref"],
        "resolved_ref": repository["resolved_ref"],
        "current_commit": repository["current_commit"],
        "default_branch": repository["default_branch"],
        "storage_kind": repository["storage_kind"],
        "managed_checkout": repository["managed_checkout"],
        "dirty_state": repository["dirty_state"],
    }
    return summary


def _protected_verification_paths(
    execution_specs: list[CodeExecutionSpec],
) -> list[str]:
    paths: list[str] = []
    for spec in execution_specs:
        tool = spec["tool"]
        if tool == "pytest":
            paths.extend(_pytest_selector_paths(spec.get("pytest_selectors")))
    unique_paths = _unique_strings(paths)
    return unique_paths


def _pytest_selector_paths(value: object) -> list[str]:
    paths: list[str] = []
    for selector in _string_list(value):
        path = selector.split("::", 1)[0]
        if path:
            paths.append(path)
    return paths


def _required_source_owner_paths(
    patch_artifacts: list[PatchArtifact],
    *,
    protected_paths: list[str],
) -> list[str]:
    protected = set(protected_paths)
    paths = [
        path
        for path in _paths_from_patch_artifacts(patch_artifacts)
        if path not in protected
    ]
    unique_paths = _unique_strings(paths)
    return unique_paths


def _paths_from_patch_artifacts(patch_artifacts: list[PatchArtifact]) -> list[str]:
    paths: list[str] = []
    for artifact in patch_artifacts:
        files = artifact.get("files")
        if not isinstance(files, list):
            continue
        for path in _string_list(files):
            paths.append(path)
    unique_paths = _unique_strings(paths)
    return unique_paths


def _changed_paths_from_proposal(
    proposal: CodingPatchProposalResponse,
) -> list[str]:
    paths: list[str] = []
    changed_files = proposal.get("changed_files")
    if isinstance(changed_files, list):
        for item in changed_files:
            if not isinstance(item, Mapping):
                continue
            path = item.get("path")
            if isinstance(path, str):
                paths.append(path)
    paths.extend(_paths_from_patch_artifacts(proposal["patch_artifacts"]))
    unique_paths = _unique_strings(paths)
    return unique_paths


def _failed_paths(
    failed_results: list[CodeExecutionResponse],
) -> list[str]:
    paths: list[str] = []
    for result in failed_results:
        paths.extend(result["executed_paths"])
        output = f"{result['stdout_excerpt']}\n{result['stderr_excerpt']}"
        for match in FAILED_TEST_RE.finditer(output):
            paths.append(match.group(1))
    unique_paths = _unique_strings(paths)
    return unique_paths


def _failure_summaries(
    *,
    stdout_excerpt: str,
    stderr_excerpt: str,
) -> list[str]:
    summaries: list[str] = []
    for line in f"{stdout_excerpt}\n{stderr_excerpt}".splitlines():
        text = " ".join(line.strip().split())
        if not text:
            continue
        if not _is_failure_summary_line(text):
            continue
        summaries.append(text[:600])
        if len(summaries) >= SUMMARY_LIMIT:
            break
    return summaries


def _is_failure_summary_line(text: str) -> bool:
    for marker in FAILURE_SUMMARY_MARKERS:
        if marker in text:
            return True
    return False


def _redact_execution_text(
    text: str,
    *,
    workspace_root: Path,
    source_root: Path,
) -> str:
    redacted = text
    for pattern in COMMAND_OUTPUT_PATTERNS:
        redacted = pattern.sub("[verification command omitted]", redacted)
    roots = [
        workspace_root.resolve(strict=False),
        source_root.resolve(strict=False),
    ]
    for root in roots:
        root_text = str(root)
        redacted = redacted.replace(root_text, "[local-root]")
        redacted = redacted.replace(root_text.replace("\\", "/"), "[local-root]")
    redacted = re.sub(
        r"(?i)(token|password|credential|secret)[A-Za-z0-9_]*=\S+",
        "[secret-like-value]",
        redacted,
    )
    return redacted


def _feedback_excerpt_cap(max_chars: int) -> int:
    cap = max(max_chars // 3, MIN_REPAIR_EXCERPT_CHARS)
    return cap


def _bounded_feedback(
    feedback: ExecutionRepairFeedback,
    *,
    max_chars: int,
) -> ExecutionRepairFeedback:
    serialized_size = len(str(feedback))
    if serialized_size <= max_chars:
        return feedback
    feedback["stdout_excerpt"] = feedback["stdout_excerpt"][:MIN_REPAIR_EXCERPT_CHARS]
    feedback["stderr_excerpt"] = feedback["stderr_excerpt"][:MIN_REPAIR_EXCERPT_CHARS]
    feedback["failure_summaries"] = feedback["failure_summaries"][:4]
    feedback["output_truncated"] = True
    return feedback


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        strings.append(text)
    return strings


def _unique_strings(values: object) -> list[str]:
    unique: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value in unique:
            continue
        unique.append(value)
    return unique


def _sanitize_public_value(
    value: object,
    *,
    source_root: Path,
    workspace_root: Path,
) -> object:
    if isinstance(value, str):
        redacted = _redact_execution_text(
            value,
            workspace_root=workspace_root,
            source_root=source_root,
        )
        return redacted
    if isinstance(value, list):
        sanitized_items = [
            _sanitize_public_value(
                item,
                source_root=source_root,
                workspace_root=workspace_root,
            )
            for item in value
        ]
        return sanitized_items
    if isinstance(value, dict):
        sanitized_dict = {
            key: _sanitize_public_value(
                item,
                source_root=source_root,
                workspace_root=workspace_root,
            )
            for key, item in value.items()
        }
        return sanitized_dict
    return value


def _terminal_response(
    *,
    status: str,
    repository: dict[str, object] | None = None,
    source_scope: Mapping[str, object] | None = None,
    attempts: list[VerifyRepairAttempt] | None = None,
    final_patch_artifacts: list[PatchArtifact] | None = None,
    final_changed_files: list[ChangedFileSummary] | None = None,
    final_apply: CodingPatchApplyResponse | None = None,
    final_execution: list[CodeExecutionResponse] | None = None,
    limitations: list[str] | None = None,
    trace_summary: list[str] | None = None,
    answer_text: str = "",
) -> CodingVerifyRepairResponse:
    response: CodingVerifyRepairResponse = {
        "status": _verify_status(status),
        "answer_text": answer_text,
        "repository": repository,  # type: ignore[typeddict-item]
        "source_scope": dict(source_scope) if source_scope is not None else None,
        "attempts": attempts or [],
        "final_patch_artifacts": final_patch_artifacts or [],
        "final_changed_files": final_changed_files or [],
        "final_apply": final_apply,
        "final_execution": final_execution or [],
        "limitations": limitations or [],
        "trace_summary": trace_summary or [],
    }
    return response


def _verify_status(status: str) -> VerifyRepairStatus:
    if status == "succeeded":
        return "succeeded"
    if status == "rejected":
        return "rejected"
    if status == "timed_out":
        return "timed_out"
    return "failed"
