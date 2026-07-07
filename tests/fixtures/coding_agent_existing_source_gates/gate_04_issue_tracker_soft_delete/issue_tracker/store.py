"""In-memory storage for issues."""

from __future__ import annotations

from issue_tracker.models import Issue


class InMemoryIssueStore:
    """Store issues by id in insertion order."""

    def __init__(self) -> None:
        self._issues: dict[str, Issue] = {}
        self._next_id = 1

    def create_issue(
        self,
        *,
        title: str,
        assignee: str | None = None,
    ) -> Issue:
        """Create and persist a new issue."""

        issue_id = str(self._next_id)
        self._next_id += 1
        issue = Issue(id=issue_id, title=title, assignee=assignee)
        self._issues[issue_id] = issue
        return issue

    def get_issue(self, issue_id: str) -> Issue | None:
        """Return an issue by id if it exists."""

        issue = self._issues.get(issue_id)
        return issue

    def list_issues(self) -> list[Issue]:
        """Return stored issues in insertion order."""

        issues = list(self._issues.values())
        return issues

    def delete_issue(self, issue_id: str) -> bool:
        """Remove an issue permanently."""

        if issue_id not in self._issues:
            return False
        del self._issues[issue_id]
        return True
