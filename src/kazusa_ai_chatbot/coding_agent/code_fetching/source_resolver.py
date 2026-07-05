"""Deterministic resolution for source-intake output."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any

from kazusa_ai_chatbot.coding_agent.code_fetching import github
from kazusa_ai_chatbot.coding_agent.code_fetching import source_intake
from kazusa_ai_chatbot.coding_agent.code_fetching.github import GitHubSource
from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeFetchingRequest,
    ResultStatus,
)
from kazusa_ai_chatbot.coding_agent.code_fetching.source_intake import (
    SourceIntakeResult,
    SourceMention,
)

_SOURCE_PRIORITY = {
    "repository": 0,
    "directory": 1,
    "file": 2,
}
_EXPLICIT_SOURCE_FIELDS = ("source_url", "repo_url", "repo_hint")
_SOURCE_FAMILY_ISSUE_CODES = {
    "github_issue",
    "github_pull",
    "github_discussion",
    "package_reference",
    "local_path",
}
_RETRYABLE_INTAKE_ISSUES = {
    "source_not_visible_in_request",
    "malformed_source",
}
_NO_SOURCE_LIMITATION = (
    "Provide a public GitHub repository, tree, blob, raw file, or owner/repo "
    "source."
)


@dataclass(frozen=True)
class SourceResolution:
    """Resolver output used before checkout or download preparation."""

    status: ResultStatus
    message: str
    source: GitHubSource | None
    limitations: tuple[str, ...]
    trace_summary: tuple[str, ...]
    issue_code: str | None
    retry_feedback: tuple[str, ...]


@dataclass(frozen=True)
class _ResolvedCandidate:
    source: GitHubSource
    raw_text: str
    role: str


@dataclass(frozen=True)
class _SourceProblem:
    issue_code: str
    message: str
    raw_text: str
    role: str


async def select_source_for_request(
    request: CodeFetchingRequest,
    trace_summary: list[str],
) -> SourceResolution:
    """Resolve request source fields, invoking source intake when needed."""

    if _has_explicit_remote_source(request):
        resolution = resolve_source_request(request, None)
        trace_summary.extend(resolution.trace_summary)
        return resolution

    question = request.get("question", "")
    if not question:
        resolution = _problem_resolution(
            status="needs_user_input",
            issue_code="no_source_found",
            message="No code source was provided.",
            limitations=[_NO_SOURCE_LIMITATION],
            trace_summary=["source_resolver:no_source_found"],
        )
        trace_summary.extend(resolution.trace_summary)
        return resolution

    intake_result = await source_intake.run_source_intake(question)
    resolution = resolve_source_request(request, intake_result)
    if not _should_retry_source_intake(resolution):
        trace_summary.extend(resolution.trace_summary)
        return resolution

    retry_intake = await source_intake.run_source_intake(
        question,
        retry_feedback=list(resolution.retry_feedback),
    )
    retry_resolution = resolve_source_request(request, retry_intake)
    combined_trace = [
        *resolution.trace_summary,
        "source_intake:retried_once",
        *retry_resolution.trace_summary,
    ]
    final_resolution = SourceResolution(
        status=retry_resolution.status,
        message=retry_resolution.message,
        source=retry_resolution.source,
        limitations=retry_resolution.limitations,
        trace_summary=tuple(combined_trace),
        issue_code=retry_resolution.issue_code,
        retry_feedback=retry_resolution.retry_feedback,
    )
    trace_summary.extend(final_resolution.trace_summary)
    return final_resolution


def resolve_source_request(
    request: CodeFetchingRequest,
    intake_result: SourceIntakeResult | None,
) -> SourceResolution:
    """Resolve trusted request fields or source-intake output."""

    if _has_explicit_remote_source(request):
        resolution = _resolve_explicit_sources(request)
        return resolution
    if intake_result is None:
        resolution = _problem_resolution(
            status="needs_user_input",
            issue_code="no_source_found",
            message="No code source was provided.",
            limitations=[_NO_SOURCE_LIMITATION],
            trace_summary=["source_resolver:no_source_found"],
        )
        return resolution

    resolution = _resolve_intake_sources(request, intake_result)
    return resolution


def source_resolution_to_dict(resolution: SourceResolution) -> dict[str, Any]:
    """Serialize a source resolution for trace artifacts."""

    source = None
    if resolution.source is not None:
        source = {
            "owner": resolution.source.owner,
            "repo": resolution.source.repo,
            "source_url": resolution.source.source_url,
            "source_kind": resolution.source.source_kind,
            "requested_ref": resolution.source.requested_ref,
            "repo_relative_path": resolution.source.repo_relative_path,
        }
    return {
        "status": resolution.status,
        "message": resolution.message,
        "source": source,
        "limitations": list(resolution.limitations),
        "trace_summary": list(resolution.trace_summary),
        "issue_code": resolution.issue_code,
        "retry_feedback": list(resolution.retry_feedback),
    }


def _resolve_explicit_sources(
    request: CodeFetchingRequest,
) -> SourceResolution:
    candidates: list[_ResolvedCandidate] = []
    problems: list[_SourceProblem] = []

    for field_name in ("source_url", "repo_url"):
        source_text = request.get(field_name)
        if source_text:
            _collect_source_text(
                source_text,
                role="primary_code_source",
                family_hint="unknown",
                candidates=candidates,
                problems=problems,
            )

    repo_hint = request.get("repo_hint")
    if repo_hint:
        source = github.parse_repo_hint(repo_hint)
        if source is None:
            problems.append(
                _SourceProblem(
                    issue_code="malformed_source",
                    message="repo_hint must use owner/repo format.",
                    raw_text=repo_hint,
                    role="primary_code_source",
                )
            )
        else:
            candidates.append(
                _ResolvedCandidate(
                    source=source,
                    raw_text=repo_hint,
                    role="primary_code_source",
                )
            )

    resolution = _resolution_from_candidates(
        request=request,
        task_source_mode="single_primary",
        candidates=candidates,
        problems=problems,
        block_primary_problems=False,
    )
    return resolution


def _resolve_intake_sources(
    request: CodeFetchingRequest,
    intake_result: SourceIntakeResult,
) -> SourceResolution:
    question = request.get("question", "")
    visible_spans = source_intake.build_visible_source_spans(question)
    candidates: list[_ResolvedCandidate] = []
    problems: list[_SourceProblem] = []

    for mention in intake_result.source_mentions:
        if not _source_text_is_visible(
            question,
            visible_spans,
            mention.raw_text,
        ):
            problems.append(
                _SourceProblem(
                    issue_code="source_not_visible_in_request",
                    message=(
                        "Source intake returned text that was not visible "
                        "in the request."
                    ),
                    raw_text=mention.raw_text,
                    role=mention.role,
                )
            )
            continue
        _collect_source_text(
            mention.raw_text,
            role=mention.role,
            family_hint=mention.family_hint,
            candidates=candidates,
            problems=problems,
        )

    resolution = _resolution_from_candidates(
        request=request,
        task_source_mode=intake_result.task_source_mode,
        candidates=candidates,
        problems=problems,
        block_primary_problems=True,
    )
    return resolution


def _collect_source_text(
    source_text: str,
    *,
    role: str,
    family_hint: str,
    candidates: list[_ResolvedCandidate],
    problems: list[_SourceProblem],
) -> None:
    if role in {"supporting_context", "reference_only"}:
        _collect_non_primary_source_text(
            source_text,
            role=role,
            family_hint=family_hint,
            candidates=candidates,
            problems=problems,
        )
        return

    source = _parse_supported_source(source_text)
    if source is not None:
        candidates.append(
            _ResolvedCandidate(
                source=source,
                raw_text=source_text,
                role=role,
            )
        )
        return

    problem = _source_problem(source_text, role=role, family_hint=family_hint)
    problems.append(problem)


def _collect_non_primary_source_text(
    source_text: str,
    *,
    role: str,
    family_hint: str,
    candidates: list[_ResolvedCandidate],
    problems: list[_SourceProblem],
) -> None:
    if (
        role == "reference_only"
        and family_hint in {"github_issue", "github_pull", "github_discussion"}
    ):
        source = _repository_source_from_github_discussion_url(source_text)
        if source is not None:
            candidates.append(
                _ResolvedCandidate(
                    source=source,
                    raw_text=source_text,
                    role="primary_code_source",
                )
            )
            problems.append(
                _SourceProblem(
                    issue_code="reference_only_source_ignored",
                    message="GitHub thread URL used only to identify its repository.",
                    raw_text=source_text,
                    role=role,
                )
            )
            return

    source = _parse_supported_source(source_text)
    if source is not None and role == "scope_modifier":
        candidates.append(
            _ResolvedCandidate(
                source=source,
                raw_text=source_text,
                role=role,
            )
        )
        return

    problem = _source_problem(source_text, role=role, family_hint=family_hint)
    problems.append(problem)


def _resolution_from_candidates(
    *,
    request: CodeFetchingRequest,
    task_source_mode: str,
    candidates: list[_ResolvedCandidate],
    problems: list[_SourceProblem],
    block_primary_problems: bool,
) -> SourceResolution:
    required_problem = _required_supporting_problem(problems)
    if required_problem is not None:
        resolution = _problem_resolution(
            status="needs_user_input",
            issue_code="required_supporting_source_unsupported",
            message=required_problem.message,
            limitations=[required_problem.message],
            trace_summary=[
                "source_resolver:required_supporting_source_unsupported",
            ],
        )
        return resolution

    if block_primary_problems:
        primary_problem = _primary_source_problem(problems)
        if primary_problem is not None:
            resolution = _resolution_without_candidates([primary_problem])
            return resolution

    normalized_candidates = _apply_requested_ref(
        candidates,
        request.get("requested_ref"),
    )
    if not normalized_candidates:
        resolution = _resolution_without_candidates(problems)
        return resolution

    selection_problem = _source_selection_problem(
        normalized_candidates,
        task_source_mode,
    )
    if selection_problem is not None:
        return selection_problem

    selected_source = _select_most_specific_source(normalized_candidates)
    if selected_source is None:
        resolution = _problem_resolution(
            status="needs_user_input",
            issue_code="ambiguous_primary_source",
            message="Multiple source scopes were provided; choose one.",
            limitations=["Multiple source scopes were provided; choose one."],
            trace_summary=["source_resolver:ambiguous_primary_source"],
        )
        return resolution

    scope_hint = request.get("source_scope_hint")
    if scope_hint and selected_source.source_kind != scope_hint:
        resolution = _problem_resolution(
            status="needs_user_input",
            issue_code="ambiguous_primary_source",
            message="source_scope_hint conflicts with the resolved source shape.",
            limitations=[
                "source_scope_hint conflicts with the resolved source shape.",
            ],
            trace_summary=["source_resolver:ambiguous_primary_source"],
        )
        return resolution

    limitations = _optional_problem_limitations(problems)
    resolution = SourceResolution(
        status="succeeded",
        message="Source candidate selected.",
        source=selected_source,
        limitations=tuple(limitations),
        trace_summary=("source_resolver:succeeded",),
        issue_code=None,
        retry_feedback=(),
    )
    return resolution


def _source_selection_problem(
    candidates: Sequence[_ResolvedCandidate],
    task_source_mode: str,
) -> SourceResolution | None:
    repositories = {
        (candidate.source.owner, candidate.source.repo)
        for candidate in candidates
    }
    if len(repositories) <= 1:
        return None

    if task_source_mode == "compare_sources":
        resolution = _problem_resolution(
            status="rejected",
            issue_code="unsupported_multi_source",
            message="Multiple-source code reading is not supported.",
            limitations=["Multiple-source code reading is not supported."],
            trace_summary=["source_resolver:unsupported_multi_source"],
        )
        return resolution

    resolution = _problem_resolution(
        status="needs_user_input",
        issue_code="ambiguous_primary_source",
        message="Multiple repository sources were provided; choose one.",
        limitations=["Multiple repository sources were provided; choose one."],
        trace_summary=["source_resolver:ambiguous_primary_source"],
    )
    return resolution


def _select_most_specific_source(
    candidates: Sequence[_ResolvedCandidate],
) -> GitHubSource | None:
    most_specific_priority = max(
        _SOURCE_PRIORITY[candidate.source.source_kind]
        for candidate in candidates
    )
    most_specific = [
        candidate.source
        for candidate in candidates
        if _SOURCE_PRIORITY[candidate.source.source_kind]
        == most_specific_priority
    ]
    distinct_scopes = {
        (
            candidate.source_kind,
            candidate.requested_ref,
            candidate.repo_relative_path,
        )
        for candidate in most_specific
    }
    if len(distinct_scopes) > 1:
        return None
    return most_specific[0]


def _resolution_without_candidates(
    problems: Sequence[_SourceProblem],
) -> SourceResolution:
    if not problems:
        resolution = _problem_resolution(
            status="needs_user_input",
            issue_code="no_source_found",
            message="No supported code source was provided.",
            limitations=[_NO_SOURCE_LIMITATION],
            trace_summary=["source_resolver:no_source_found"],
        )
        return resolution

    if any(
        problem.issue_code == "source_not_visible_in_request"
        for problem in problems
    ):
        retry_feedback = [
            f"Use only source text visible in the request. Rejected: {problem.raw_text}"
            for problem in problems
            if problem.issue_code == "source_not_visible_in_request"
        ]
        resolution = SourceResolution(
            status="needs_user_input",
            message="Source intake returned text that was not visible in the request.",
            source=None,
            limitations=tuple(problem.message for problem in problems),
            trace_summary=("source_resolver:source_not_visible_in_request",),
            issue_code="source_not_visible_in_request",
            retry_feedback=tuple(retry_feedback),
        )
        return resolution

    first_problem = problems[0]
    status: ResultStatus = "rejected"
    if first_problem.issue_code == "malformed_source":
        status = "needs_user_input"
    resolution = _problem_resolution(
        status=status,
        issue_code=first_problem.issue_code,
        message=first_problem.message,
        limitations=[problem.message for problem in problems],
        trace_summary=[f"source_resolver:{first_problem.issue_code}"],
        retry_feedback=_retry_feedback_for_problem(first_problem),
    )
    return resolution


def _source_problem(
    source_text: str,
    *,
    role: str,
    family_hint: str,
) -> _SourceProblem:
    issue_code = _issue_code_for_source_text(source_text, family_hint)
    message = _message_for_source_problem(
        source_text,
        issue_code=issue_code,
        family_hint=family_hint,
        role=role,
    )
    problem = _SourceProblem(
        issue_code=issue_code,
        message=message,
        raw_text=github.redact_source_text(source_text),
        role=role,
    )
    return problem


def _issue_code_for_source_text(source_text: str, family_hint: str) -> str:
    if family_hint in _SOURCE_FAMILY_ISSUE_CODES:
        return "unsupported_source_family"

    reason = github.unsupported_source_reason(source_text)
    if reason:
        if reason.startswith("Only public GitHub"):
            return "unsupported_provider"
        return "unsupported_source_family"

    parsed_url = urlparse(source_text)
    host = parsed_url.netloc.lower()
    if parsed_url.scheme.lower() in {"npm", "pypi", "cargo", "crate", "go", "gomod"}:
        return "unsupported_source_family"
    if parsed_url.scheme in {"http", "https"} and host not in (
        "github.com",
        "www.github.com",
        "raw.githubusercontent.com",
    ):
        return "unsupported_provider"
    if parsed_url.scheme in {"http", "https"} and host:
        return "malformed_source"
    if _looks_like_local_path(source_text):
        return "unsupported_source_family"
    if family_hint == "documentation_url":
        return "unsupported_provider"
    if family_hint == "web_page":
        return "unsupported_provider"
    return "malformed_source"


def _message_for_source_problem(
    source_text: str,
    *,
    issue_code: str,
    family_hint: str,
    role: str,
) -> str:
    redacted_text = github.redact_source_text(source_text)
    if role == "supporting_context":
        return f"Required supporting source is not supported: {redacted_text}"
    if issue_code == "unsupported_provider":
        return (
            "Only public GitHub repository sources are supported. "
            f"Source: {redacted_text}"
        )
    if issue_code == "unsupported_source_family":
        return f"Source family is not supported for code fetching: {redacted_text}"
    if issue_code == "malformed_source":
        return f"Source text is malformed or unsupported: {redacted_text}"
    return f"Unsupported source ({family_hint}): {redacted_text}"


def _parse_supported_source(source_text: str) -> GitHubSource | None:
    source = github.parse_github_source(source_text)
    if source is not None:
        return source
    if "://" not in source_text:
        source = github.parse_repo_hint(source_text)
        return source
    return None


def _repository_source_from_github_discussion_url(
    source_text: str,
) -> GitHubSource | None:
    parsed_url = urlparse(source_text.strip())
    if parsed_url.scheme not in {"http", "https"}:
        return None
    if parsed_url.netloc.lower() not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed_url.path.split("/") if part]
    if len(parts) < 4:
        return None
    if parts[2].lower() not in {"issues", "pull", "pulls", "discussions"}:
        return None
    source = github.parse_repo_hint(f"{parts[0]}/{parts[1]}")
    return source


def _source_text_is_visible(
    question: str,
    visible_spans: Sequence[str],
    source_text: str,
) -> bool:
    if source_text in question:
        return True
    if source_text in visible_spans:
        return True
    for span in visible_spans:
        if source_text in span:
            return True
    return False


def _apply_requested_ref(
    candidates: Sequence[_ResolvedCandidate],
    requested_ref: str | None,
) -> list[_ResolvedCandidate]:
    if not requested_ref:
        return list(candidates)

    updated_candidates: list[_ResolvedCandidate] = []
    for candidate in candidates:
        updated_source = github.with_requested_ref(candidate.source, requested_ref)
        updated_candidates.append(
            _ResolvedCandidate(
                source=updated_source,
                raw_text=candidate.raw_text,
                role=candidate.role,
            )
        )
    return updated_candidates


def _required_supporting_problem(
    problems: Sequence[_SourceProblem],
) -> _SourceProblem | None:
    for problem in problems:
        if problem.role == "supporting_context":
            return problem
    return None


def _optional_problem_limitations(
    problems: Sequence[_SourceProblem],
) -> list[str]:
    limitations: list[str] = []
    for problem in problems:
        if problem.role == "supporting_context":
            continue
        if problem.issue_code == "source_not_visible_in_request":
            continue
        if problem.issue_code == "reference_only_source_ignored":
            limitations.append(problem.message)
        else:
            limitations.append(problem.message)
    return limitations


def _primary_source_problem(
    problems: Sequence[_SourceProblem],
) -> _SourceProblem | None:
    for problem in problems:
        if problem.role == "primary_code_source":
            return problem
    return None


def _retry_feedback_for_problem(problem: _SourceProblem) -> list[str]:
    if problem.issue_code != "malformed_source":
        return []
    feedback = [
        "Return a source that matches supported provider grammar. "
        f"Rejected: {problem.raw_text}"
    ]
    return feedback


def _problem_resolution(
    *,
    status: ResultStatus,
    issue_code: str,
    message: str,
    limitations: list[str],
    trace_summary: list[str],
    retry_feedback: list[str] | None = None,
) -> SourceResolution:
    feedback = [] if retry_feedback is None else retry_feedback
    resolution = SourceResolution(
        status=status,
        message=message,
        source=None,
        limitations=tuple(limitations),
        trace_summary=tuple(trace_summary),
        issue_code=issue_code,
        retry_feedback=tuple(feedback),
    )
    return resolution


def _should_retry_source_intake(resolution: SourceResolution) -> bool:
    if not resolution.retry_feedback:
        return False
    return resolution.issue_code in _RETRYABLE_INTAKE_ISSUES


def _has_explicit_remote_source(request: CodeFetchingRequest) -> bool:
    return any(request.get(field_name) for field_name in _EXPLICIT_SOURCE_FIELDS)


def _looks_like_local_path(source_text: str) -> bool:
    if len(source_text) >= 3 and source_text[1:3] == ":\\":
        return True
    return source_text.startswith(("./", "../", "/", "~"))
