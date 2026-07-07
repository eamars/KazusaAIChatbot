from issue_tracker.store import InMemoryIssueStore


def test_delete_issue_removes_record() -> None:
    store = InMemoryIssueStore()
    issue = store.create_issue(title="Fix import")

    deleted = store.delete_issue(issue.id)

    assert deleted is True
    assert store.get_issue(issue.id) is None
    assert store.list_issues() == []


def test_delete_missing_issue_returns_false() -> None:
    store = InMemoryIssueStore()

    deleted = store.delete_issue("missing")

    assert deleted is False
