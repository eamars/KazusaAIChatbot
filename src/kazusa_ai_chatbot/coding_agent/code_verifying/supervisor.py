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
    materialize_managed_candidate,
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
from kazusa_ai_chatbot.config import (
    CODING_AGENT_REPAIR_BUNDLE_CHAR_LIMIT,
    CODING_AGENT_REPAIR_MAX_CALLS,
)

ALLOWED_EXECUTION_TOOLS = {"python_compileall", "pytest"}
DEFAULT_REPAIR_ATTEMPTS = CODING_AGENT_REPAIR_MAX_CALLS
MAX_REPAIR_ATTEMPTS = 6
DEFAULT_REPAIR_FEEDBACK_CHARS = CODING_AGENT_REPAIR_BUNDLE_CHAR_LIMIT
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
MISSING_MODULE_RE = re.compile(
    r"ModuleNotFoundError:\s+No module named ['\"](?P<module>[A-Za-z_][\w.]*)['\"]",
)
MISSING_INTERPRETER_MARKERS = (
    "No such file or directory",
    "Python executable was not found",
    "python: command not found",
)


def build_execution_failure_bundle(
    *,
    spec_id: str,
    execution: Mapping[str, object],
    candidate_root: Path,
) -> dict[str, object]:
    """Preserve bounded structural evidence for one failed execution spec."""

    stdout_excerpt = _request_text(execution.get("stdout_excerpt"))
    stderr_excerpt = _request_text(execution.get("stderr_excerpt"))
    exception_text = _root_exception_text(stdout_excerpt, stderr_excerpt)
    failure_kind = _failure_kind(execution, exception_text)
    trace_frames = _failure_trace_frames(
        text="\n".join((stdout_excerpt, stderr_excerpt)),
        candidate_root=candidate_root,
    )
    bundle = {
        "failure_id": _failure_signature(
            spec_id=spec_id,
            failure_kind=failure_kind,
            exception_text=exception_text,
        ),
        "spec_id": spec_id,
        "tool": _request_text(execution.get("tool")),
        "selector": _first_executed_path(execution),
        "failure_kind": failure_kind,
        "exception_type": _exception_type(exception_text),
        "exception_message": exception_text,
        "trace_frames": trace_frames,
        "failure_signature": _failure_signature(
            spec_id=spec_id,
            failure_kind=failure_kind,
            exception_text=exception_text,
        ),
        "related_evidence_ids": [],
        "stdout_excerpt": stdout_excerpt,
        "stderr_excerpt": stderr_excerpt,
    }
    return bundle


def classify_execution_failure(
    bundle: Mapping[str, object],
    *,
    candidate_root: Path,
) -> str:
    """Classify environment failures before spending a repair attempt."""

    exception_message = _request_text(bundle.get("exception_message"))
    if any(marker in exception_message for marker in MISSING_INTERPRETER_MARKERS):
        return "environment_dependency_missing"
    missing_module = _missing_module(exception_message)
    if not missing_module:
        return _request_text(bundle.get("failure_kind")) or "exception"
    if _candidate_mentions_module(candidate_root, missing_module):
        return "exception"
    return "environment_dependency_missing"


def _root_exception_text(stdout_excerpt: str, stderr_excerpt: str) -> str:
    """Extract the terminal exception cause from complete bounded output."""

    combined_text = "\n".join((stderr_excerpt, stdout_excerpt))
    missing_module_match = MISSING_MODULE_RE.search(combined_text)
    if missing_module_match is not None:
        return missing_module_match.group(0)
    for line in [*stderr_excerpt.splitlines(), *stdout_excerpt.splitlines()]:
        if "Error" in line or "Exception" in line or "not found" in line:
            return line[:500]
    return "Execution failed without a parseable exception message."


def _request_text(value: object) -> str:
    """Convert an optional request value into a bounded plain string."""

    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text


def _failure_kind(execution: Mapping[str, object], exception_text: str) -> str:
    if execution.get("status") == "timed_out":
        return "timeout"
    tool = _request_text(execution.get("tool"))
    if tool == "python_compileall":
        return "compile_error"
    if "AssertionError" in exception_text:
        return "assertion"
    return "exception"


def _failure_trace_frames(*, text: str, candidate_root: Path) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    frame_re = re.compile(r'File "(?P<path>[^"]+)", line (?P<line>\d+)')
    for match in frame_re.finditer(text):
        path_value = Path(match.group("path"))
        try:
            relative_path = path_value.resolve().relative_to(candidate_root.resolve())
        except ValueError:
            continue
        frames.append({
            "path": relative_path.as_posix(),
            "line": int(match.group("line")),
            "function": "",
            "code_line": "",
        })
        if len(frames) >= 8:
            break
    return frames


def _failure_signature(*, spec_id: str, failure_kind: str, exception_text: str) -> str:
    normalized_text = re.sub(r"\d+", "#", exception_text.casefold())
    return f"{spec_id}:{failure_kind}:{normalized_text[:240]}"


def _exception_type(exception_text: str) -> str:
    if ":" not in exception_text:
        return ""
    exception_type = exception_text.split(":", 1)[0].strip()
    return exception_type


def _first_executed_path(execution: Mapping[str, object]) -> str:
    paths = execution.get("executed_paths")
    if not isinstance(paths, list):
        return ""
    for path in paths:
        if isinstance(path, str):
            return path
    return ""


def _missing_module(exception_message: str) -> str:
    match = MISSING_MODULE_RE.search(exception_message)
    if match is None:
        return ""
    module_name = match.group("module").split(".", 1)[0]
    return module_name


def _candidate_mentions_module(candidate_root: Path, module_name: str) -> bool:
    """Return whether the missing root is present in the candidate manifest."""

    module_path = candidate_root / module_name
    if module_path.exists() or (candidate_root / f"{module_name}.py").exists():
        return True
    return False


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

    if not _has_source_fields(request):
        response = _verify_source_free_candidate(request)
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
    proposal_response, protected_omissions = (
        _proposal_without_protected_verification_paths(
            proposal=proposal_response,
            protected_paths=protected_paths,
        )
    )
    if protected_omissions and not proposal_response["patch_artifacts"]:
        response = _terminal_response(
            status="rejected",
            repository=_repository_summary(repository),
            source_scope=source_scope,
            final_patch_artifacts=[],
            final_changed_files=[],
            limitations=[
                *fetching_result["limitations"],
                *proposal_response["limitations"],
                "No non-protected patch artifacts remain for approved verification.",
            ],
            trace_summary=[
                *fetching_result["trace_summary"],
                *proposal_response["trace_summary"],
                "verify_repair:rejected:protected_initial_artifacts",
            ],
        )
        return response
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

        environment_blocker = _environment_blocker(
            execution_results=execution_results,
            candidate_root=_apply_candidate_root(
                workspace_root=Path(request["workspace_root"]),
                apply_response=apply_response,
            ),
        )
        if environment_blocker is not None:
            response = _terminal_response(
                status="blocked",
                repository=_repository_summary(repository),
                source_scope=source_scope,
                attempts=attempts,
                final_patch_artifacts=patch_artifacts,
                final_changed_files=apply_response["changed_files"],
                final_apply=final_apply,
                final_execution=final_execution,
                blockers=[environment_blocker],
                limitations=[environment_blocker["message"]],
                trace_summary=trace_summary,
                answer_text="Verification is blocked by an environment dependency.",
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


def _verify_source_free_candidate(
    request: CodingVerifyRepairRequest,
) -> CodingVerifyRepairResponse:
    """Verify generated proposal artifacts through the canonical candidate path."""

    patch_artifacts = request.get("initial_patch_artifacts")
    expected_identity = request.get("expected_source_identity")
    if not isinstance(patch_artifacts, list) or not isinstance(expected_identity, Mapping):
        response = _terminal_response(
            status="rejected",
            limitations=["Source-free verification requires bound patch artifacts."],
            trace_summary=["verify_repair:rejected:source_free_binding"],
        )
        return response
    apply_response = materialize_managed_candidate({
        "workspace_root": request["workspace_root"],
        "source_identity": dict(expected_identity),
        "expected_source_identity": dict(expected_identity),
        "patch_artifacts": patch_artifacts,
        "candidate_baseline": "empty_source_free",
        "authorization_purpose": "approved_verification",
        "approval": request["approval"],
        "max_diff_chars": request.get("max_artifact_chars"),
    })
    if apply_response["status"] != "succeeded":
        response = _terminal_response(
            status=apply_response["status"],
            final_patch_artifacts=patch_artifacts,
            final_changed_files=apply_response["changed_files"],
            final_apply=apply_response,
            limitations=apply_response["limitations"],
            trace_summary=apply_response["trace_summary"],
        )
        return response
    execution_results = _run_execution_specs(
        request=request,
        apply_response=apply_response,
    )
    attempt = _attempt_summary(
        attempt_index=1,
        proposal_status="succeeded",
        apply_response=apply_response,
        execution_results=execution_results,
        patch_artifacts=patch_artifacts,
    )
    status = _terminal_status_from_execution(execution_results)
    if _all_execution_succeeded(execution_results):
        status = "succeeded"
    response = _terminal_response(
        status=status,
        attempts=[attempt],
        final_patch_artifacts=patch_artifacts,
        final_changed_files=apply_response["changed_files"],
        final_apply=apply_response,
        final_execution=execution_results,
        limitations=attempt["limitations"],
        trace_summary=[*apply_response["trace_summary"], *_execution_trace(execution_results)],
        answer_text="Source-free proposal verification completed.",
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
    if not _has_source_fields(request):
        if not isinstance(initial_artifacts, list) or not initial_artifacts:
            return "Verify repair requires source fields or proposal artifacts."
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


def _proposal_without_protected_verification_paths(
    *,
    proposal: CodingPatchProposalResponse,
    protected_paths: list[str],
) -> tuple[CodingPatchProposalResponse, list[str]]:
    protected = set(protected_paths)
    if not protected:
        return proposal, []

    kept_artifacts: list[PatchArtifact] = []
    omitted_paths: list[str] = []
    for artifact in proposal["patch_artifacts"]:
        artifact_paths = _paths_from_patch_artifacts([artifact])
        protected_matches = [
            path
            for path in artifact_paths
            if path in protected
        ]
        if protected_matches:
            omitted_paths.extend(protected_matches)
            continue
        kept_artifacts.append(artifact)

    unique_omissions = _unique_strings(omitted_paths)
    if not unique_omissions:
        return proposal, []

    omitted = set(unique_omissions)
    changed_files = _changed_files_without_paths(
        changed_files=proposal["changed_files"],
        omitted_paths=omitted,
    )
    validation = _validation_without_paths(
        validation=proposal["validation"],
        omitted_paths=omitted,
    )
    updated: CodingPatchProposalResponse = {
        **proposal,
        "patch_artifacts": kept_artifacts,
        "changed_files": changed_files,
        "validation": validation,
        "limitations": [
            *proposal["limitations"],
            *_protected_omission_limitations(unique_omissions),
        ],
        "trace_summary": [
            *proposal["trace_summary"],
            _protected_omission_trace(unique_omissions),
        ],
    }
    return updated, unique_omissions


def _changed_files_without_paths(
    *,
    changed_files: list[ChangedFileSummary],
    omitted_paths: set[str],
) -> list[ChangedFileSummary]:
    kept: list[ChangedFileSummary] = []
    for changed_file in changed_files:
        path = changed_file.get("path")
        if path in omitted_paths:
            continue
        kept.append(changed_file)
    return kept


def _validation_without_paths(
    *,
    validation: Mapping[str, object],
    omitted_paths: set[str],
) -> dict[str, object]:
    updated = dict(validation)
    files = updated.get("files")
    if isinstance(files, list):
        updated["files"] = [
            path
            for path in files
            if not isinstance(path, str) or path not in omitted_paths
        ]
    return updated


def _protected_omission_limitations(paths: list[str]) -> list[str]:
    return [
        f"Omitted protected verification path from approved apply: {path}"
        for path in paths
    ]


def _protected_omission_trace(paths: list[str]) -> str:
    return f"verify_repair:protected_initial_artifacts_omitted count={len(paths)}"


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


def _environment_blocker(
    *,
    execution_results: list[CodeExecutionResponse],
    candidate_root: Path,
) -> dict[str, object] | None:
    """Return a typed blocker when verification needs an external dependency."""

    for index, execution_result in enumerate(execution_results, start=1):
        if execution_result["status"] not in {"failed", "timed_out"}:
            continue
        bundle = build_execution_failure_bundle(
            spec_id=f"execution-{index}",
            execution=execution_result,
            candidate_root=candidate_root,
        )
        classification = classify_execution_failure(
            bundle,
            candidate_root=candidate_root,
        )
        if classification != "environment_dependency_missing":
            continue
        missing_module = _missing_module(
            _request_text(bundle.get("exception_message")),
        )
        blocker = {
            "code": "environment_dependency_missing",
            "blocker_kind": "environment",
            "message": "Verification requires an unavailable external dependency.",
            "question": "Install the dependency in the execution environment, then retry verification.",
            "options": ["Install dependency", "Retry verification", "Cancel run"],
            "resume_target": "retry_verification",
            "details": {
                "missing_module": missing_module,
                "tool": execution_result["tool"],
                "selector": _first_executed_path(execution_result),
            },
        }
        return blocker
    return None


def _apply_candidate_root(
    *,
    workspace_root: Path,
    apply_response: CodingPatchApplyResponse,
) -> Path:
    """Return the final managed candidate used by the execution boundary."""

    candidate_root = (
        workspace_root
        / "patch_apply"
        / apply_response["apply_package_id"]
        / "source"
    )
    return candidate_root


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
    blockers: list[dict[str, object]] | None = None,
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
        "blockers": blockers or [],
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
    if status == "blocked":
        return "blocked"
    return "failed"
