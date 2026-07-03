"""Deterministic source-scope selection."""

from dataclasses import dataclass
from collections.abc import Sequence

from kazusa_ai_chatbot.coding_agent.code_fetching.github import GitHubSource

_SOURCE_PRIORITY = {
    "repository": 0,
    "directory": 1,
    "file": 2,
}


@dataclass(frozen=True)
class SourceSelection:
    """Result of deterministic source-candidate selection."""

    status: str
    message: str
    source: GitHubSource | None


def choose_source(candidates: Sequence[GitHubSource]) -> SourceSelection:
    """Choose exactly one supported source candidate.

    Args:
        candidates: Supported GitHub source candidates from request fields.

    Returns:
        Selection result with a chosen source or clarification status.
    """

    if not candidates:
        selection = SourceSelection(
            status="needs_user_input",
            message="No supported code source was provided.",
            source=None,
        )
        return selection

    repositories = {(candidate.owner, candidate.repo) for candidate in candidates}
    if len(repositories) > 1:
        selection = SourceSelection(
            status="needs_user_input",
            message="Multiple repository sources were provided; choose one.",
            source=None,
        )
        return selection

    most_specific_priority = max(
        _SOURCE_PRIORITY[candidate.source_kind] for candidate in candidates
    )
    most_specific = [
        candidate
        for candidate in candidates
        if _SOURCE_PRIORITY[candidate.source_kind] == most_specific_priority
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
        selection = SourceSelection(
            status="needs_user_input",
            message="Multiple source scopes were provided; choose one.",
            source=None,
        )
        return selection

    selected_source = most_specific[0]
    selection = SourceSelection(
        status="succeeded",
        message="Resolved one source candidate.",
        source=selected_source,
    )
    return selection
