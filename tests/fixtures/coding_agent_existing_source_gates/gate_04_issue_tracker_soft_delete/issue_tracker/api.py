"""Route-like service helpers for the issue tracker."""

from __future__ import annotations

from issue_tracker.models import Issue
from issue_tracker.store import InMemoryIssueStore


def create_issue(
    store: InMemoryIssueStore,
    title: str,
    *,
    assignee: str | None = None,
) -> Issue:
    """Create an issue through the service layer."""

    issue = store.create_issue(title=title, assignee=assignee)
    return issue


def get_issue(store: InMemoryIssueStore, issue_id: str) -> Issue | None:
    """Return an issue through the service layer."""

    issue = store.get_issue(issue_id)
    return issue


def list_issues(store: InMemoryIssueStore) -> list[Issue]:
    """Return all issues through the service layer."""

    issues = store.list_issues()
    return issues


def delete_issue(store: InMemoryIssueStore, issue_id: str) -> bool:
    """Delete an issue through the service layer."""

    deleted = store.delete_issue(issue_id)
    return deleted
